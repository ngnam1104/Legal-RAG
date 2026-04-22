"""
Internal API Reranker — On-Premise Adapter
===========================================
Gọi REST API nội bộ tại 10.9.3.75:30546 để re-rank documents.
KHÔNG dùng ``sentence-transformers`` hay ``CrossEncoder``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RERANKING_ENDPOINT: str = "http://10.9.3.75:30546/api/v1/reranking"
_REQUEST_TIMEOUT: int = 60  # seconds

# ---------------------------------------------------------------------------
# Singleton guard
# ---------------------------------------------------------------------------
_instance: "InternalAPIReranker | None" = None


class InternalAPIReranker:
    """
    Adapter Reranker gọi REST API nội bộ.

    * Singleton: chỉ tạo 1 instance duy nhất.
    * Fallback: nếu API sập → giữ nguyên thứ tự gốc với score = 0.0.
    """

    def __new__(cls, *args, **kwargs) -> "InternalAPIReranker":
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialized = False
        return _instance

    def __init__(
        self,
        endpoint: str = _RERANKING_ENDPOINT,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        if self._initialized:
            return
        self.endpoint: str = endpoint
        self.timeout: int = timeout
        self._session: requests.Session = requests.Session()
        self._initialized = True
        logger.info("✅ [InternalAPIReranker] Singleton khởi tạo — endpoint: %s", self.endpoint)

    # ------------------------------------------------------------------
    # Core rerank
    # ------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Gửi query + docs lên API reranking, nhận về scores, sắp xếp giảm dần.

        Returns:
            List[{"index": int, "document": str, "score": float}]
            cắt theo ``top_k``.
        """
        if not docs:
            return []

        # Guard
        query = str(query).strip() or "N/A"
        docs = [str(d) if d else "N/A" for d in docs]
        docs = [d if d.strip() else "N/A" for d in docs]

        payload = {"query": query, "docs": docs}

        try:
            resp = self._session.post(
                self.endpoint, json=payload, timeout=self.timeout,
            )
            resp.raise_for_status()

            data = resp.json()
            scores: List[float] = data.get("scores", [])

            # Đảm bảo số scores khớp số docs
            if len(scores) < len(docs):
                logger.warning(
                    "⚠️ [InternalAPIReranker] Scores (%d) < docs (%d). Padding 0.0.",
                    len(scores), len(docs),
                )
                scores.extend([0.0] * (len(docs) - len(scores)))

            # Map score → doc, sắp xếp giảm dần
            results = [
                {"index": i, "document": doc, "score": float(scores[i])}
                for i, doc in enumerate(docs)
            ]
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except requests.exceptions.Timeout:
            logger.error(
                "❌ [InternalAPIReranker] Timeout (%ds) khi gọi %s.",
                self.timeout, self.endpoint,
            )
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "❌ [InternalAPIReranker] HTTP %s khi gọi %s.",
                exc.response.status_code if exc.response is not None else "???",
                self.endpoint,
            )
        except requests.exceptions.ConnectionError:
            logger.error(
                "❌ [InternalAPIReranker] Không thể kết nối tới %s.",
                self.endpoint,
            )
        except Exception:
            logger.exception("❌ [InternalAPIReranker] Lỗi không xác định.")

        # Fallback: giữ nguyên thứ tự, score = 0.0
        return [
            {"index": i, "document": doc, "score": 0.0}
            for i, doc in enumerate(docs[:top_k])
        ]

    # ------------------------------------------------------------------
    # Convenience — tương thích interface cũ (DocumentReranker)
    # ------------------------------------------------------------------
    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Nhận list candidates dạng pipeline (chứa payload), rerank rồi trả
        lại list candidate đã enriched ``rerank_score``.
        """
        if not candidates:
            return []

        def _build_text(item: Dict[str, Any]) -> str:
            p = item.get("payload", item)
            title = p.get("title", "")
            citation = p.get("reference_citation", "")
            content = p.get("chunk_text", "") or p.get("text", "")
            return f"{title}\n{citation}\n{content}".strip()

        docs = [_build_text(c) for c in candidates]
        ranked = self.rerank(query, docs, top_k=len(docs))

        # Map lại score vào candidates theo original index
        score_map = {r["index"]: r["score"] for r in ranked}
        enriched = []
        for i, item in enumerate(candidates):
            entry = dict(item)
            entry["score"] = score_map.get(i, 0.0)
            entry["rerank_score"] = score_map.get(i, 0.0)
            enriched.append(entry)

        enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
        return enriched[:top_k]

    def __repr__(self) -> str:
        return f"<InternalAPIReranker endpoint={self.endpoint!r}>"

reranker = InternalAPIReranker()
