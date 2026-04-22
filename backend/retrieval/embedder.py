"""
Internal API Embedder — On-Premise Adapter
==========================================
- **Dense:**  Gọi REST API nội bộ tại 10.9.3.75:30010 (1024-d, BAAI/bge-m3).
- **Sparse:** Sử dụng ``fastembed`` BM25 (Qdrant/bm25) chạy local CPU.

Kết hợp 2 luồng này tạo thành Hybrid Search (Dense + Sparse) cho Qdrant.
"""

from __future__ import annotations

import logging
from typing import List, Union

import requests
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding

from backend.retrieval.base import BaseEmbedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EMBEDDING_ENDPOINT: str = "http://10.9.3.75:30010/api/v1/embedding"
_VECTOR_DIM: int = 1024
_REQUEST_TIMEOUT: int = 60  # seconds


# ---------------------------------------------------------------------------
# Singleton guard
# ---------------------------------------------------------------------------
_instance: "InternalAPIEmbedder | None" = None


class InternalAPIEmbedder(BaseEmbedder):
    """
    Adapter Embedding gọi REST API nội bộ (Dense) + fastembed BM25 (Sparse).

    * Singleton: chỉ tạo 1 instance duy nhất trong toàn bộ process.
    * Dense fallback: trả về zero-vectors nếu API lỗi để pipeline không crash.
    * Sparse fallback: trả về SparseVector rỗng nếu BM25 model lỗi.
    """

    def __new__(cls, *args, **kwargs) -> "InternalAPIEmbedder":
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialized = False
        return _instance

    def __init__(
        self,
        endpoint: str = _EMBEDDING_ENDPOINT,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        if self._initialized:
            return
        self.endpoint: str = endpoint
        self.timeout: int = timeout
        self._session: requests.Session = requests.Session()

        # --- Sparse BM25 model (fastembed, chạy local CPU) ---
        try:
            self.sparse_model: SparseTextEmbedding | None = SparseTextEmbedding(model_name="Qdrant/bm25")
            logger.info("✅ [InternalAPIEmbedder] BM25 sparse model (Qdrant/bm25) đã sẵn sàng.")
        except Exception as e:
            print(f"⚠️ [InternalAPIEmbedder] Lỗi khởi tạo BM25 FastEmbed: {e}")
            logger.warning("⚠️ BM25 FastEmbed không khả dụng — sparse vectors sẽ trả rỗng. Lỗi: %s", e)
            self.sparse_model = None

        self._initialized = True
        logger.info("✅ [InternalAPIEmbedder] Singleton khởi tạo — endpoint: %s", self.endpoint)

    # ------------------------------------------------------------------
    # Core Dense encoder (REST API — KHÔNG CHỈNH SỬA)
    # ------------------------------------------------------------------
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Gửi danh sách văn bản tới API và nhận về Dense vectors (1024-d).

        Returns:
            List 2-D float.  Nếu API sập → trả về zero-vectors.
        """
        if not texts:
            return []

        # Guard: đảm bảo mọi phần tử là chuỗi không rỗng
        texts = [str(t) if t else "N/A" for t in texts]
        texts = [t if t.strip() else "N/A" for t in texts]
        
        # Fix 6D: Sanitize encoding — loại bỏ null bytes và ký tự không hợp lệ
        texts = [t.replace('\x00', '').encode('utf-8', errors='ignore').decode('utf-8') for t in texts]

        payload = {"texts": texts, "normalize": True}

        try:
            resp = self._session.post(
                self.endpoint, json=payload, timeout=self.timeout,
            )
            resp.raise_for_status()

            data = resp.json()
            embeddings: List[List[float]] = data.get("embeddings", [])

            if len(embeddings) != len(texts):
                logger.warning(
                    "⚠️ [InternalAPIEmbedder] Số lượng embeddings trả về (%d) "
                    "≠ số texts gửi đi (%d). Padding zero-vectors.",
                    len(embeddings), len(texts),
                )
                while len(embeddings) < len(texts):
                    embeddings.append([0.0] * _VECTOR_DIM)

            return embeddings

        except requests.exceptions.Timeout:
            print(f"❌ [InternalAPIEmbedder] Timeout ({self.timeout}s) calling {self.endpoint}")
            logger.error(
                "❌ [InternalAPIEmbedder] Timeout (%ds) khi gọi %s — trả zero-vectors.",
                self.timeout, self.endpoint,
            )
        except requests.exceptions.HTTPError as exc:
            print(f"❌ [InternalAPIEmbedder] HTTP {exc.response.status_code if exc.response is not None else '???'} calling {self.endpoint}")
            logger.error(
                "❌ [InternalAPIEmbedder] HTTP %s khi gọi %s — trả zero-vectors.",
                exc.response.status_code if exc.response is not None else "???",
                self.endpoint,
            )
        except requests.exceptions.ConnectionError:
            print(f"❌ [InternalAPIEmbedder] Connection Error calling {self.endpoint}")
            logger.error(
                "❌ [InternalAPIEmbedder] Không thể kết nối tới %s — trả zero-vectors.",
                self.endpoint,
            )
        except Exception as e:
            print(f"❌ [InternalAPIEmbedder] Unknown Error: {e}")
            logger.exception("❌ [InternalAPIEmbedder] Lỗi không xác định — trả zero-vectors.")

        # Fallback: zero-vectors
        return [[0.0] * _VECTOR_DIM for _ in texts]

    # ------------------------------------------------------------------
    # BaseEmbedder interface — Dense (KHÔNG CHỈNH SỬA)
    # ------------------------------------------------------------------
    def encode_dense(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return self.encode(texts)

    def encode_query_dense(self, text: str) -> List[float]:
        return self.encode([text])[0]

    # ------------------------------------------------------------------
    # Sparse — fastembed BM25 (Qdrant/bm25, chạy local CPU)
    # ------------------------------------------------------------------
    def encode_sparse_documents(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[SparseVector]:
        """Encode documents thành BM25 sparse vectors (dùng cho ingestion/indexing)."""
        if isinstance(texts, str):
            texts = [texts]

        if not self.sparse_model:
            return [SparseVector(indices=[], values=[]) for _ in texts]

        try:
            embeddings = list(self.sparse_model.embed(texts, batch_size=batch_size))
            result = []
            for emb in embeddings:
                sparse_vec = SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist(),
                )
                result.append(sparse_vec)
            return result
        except Exception as e:
            print(f"❌ [InternalAPIEmbedder] Lỗi khi encode BM25 documents: {e}")
            logger.exception("❌ BM25 encode_sparse_documents thất bại — trả rỗng.")
            return [SparseVector(indices=[], values=[]) for _ in texts]

    def encode_query_sparse(self, text: str) -> SparseVector:
        """Encode query thành BM25 sparse vector (dùng cho search/retrieval)."""
        if not text or not self.sparse_model:
            return SparseVector(indices=[], values=[])

        try:
            query_embeddings = list(self.sparse_model.query_embed(text))
            if query_embeddings:
                emb = query_embeddings[0]
                return SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist(),
                )
            return SparseVector(indices=[], values=[])
        except Exception as e:
            print(f"❌ [InternalAPIEmbedder] Lỗi khi encode BM25 query: {e}")
            logger.exception("❌ BM25 encode_query_sparse thất bại — trả rỗng.")
            return SparseVector(indices=[], values=[])

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<InternalAPIEmbedder endpoint={self.endpoint!r} sparse={'BM25' if self.sparse_model else 'None'}>"

embedder = InternalAPIEmbedder()

