import os
import torch
from typing import List, Dict, Union, Tuple
from backend.config import settings

# --- THIẾT LẬP CACHE HUGGINGFACE TRƯỚC KHI IMPORT MODEL ---
if settings.HF_HOME:
    os.environ["HF_HOME"] = settings.HF_HOME
if settings.SENTENCE_TRANSFORMERS_HOME:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = settings.SENTENCE_TRANSFORMERS_HOME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import models sau khi đã set biến môi trường
from FlagEmbedding import BGEM3FlagModel
from qdrant_client.models import SparseVector
from backend.retrieval.base import BaseEmbedder

# --- GLOBAL MODEL CACHE ---
_MODEL_INSTANCES = {}

class LocalBGEHybridEncoder(BaseEmbedder):
    """
    Sử dụng BGE-M3 bản gốc (PyTorch) để tạo vectors.
    Đảm bảo đồng bộ 100% với logic trong Notebook.
    """
    def __init__(self, model_name: str = None, device: str = None):
        if not model_name:
            model_name = settings.LEGAL_DENSE_MODEL
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model_name = model_name
        self.device = device
        
        # Singleton logic: Chỉ load 1 lần duy nhất trong toàn bộ process
        model_key = f"{model_name}_{device}"
        if model_key not in _MODEL_INSTANCES:
            print(f"⏳ Đang nạp mô hình BGE-M3 (PyTorch): {model_name} trên thiết bị: {device}...")
            _MODEL_INSTANCES[model_key] = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"), device=device)
            print(f"✅ Đã tải xong Embedding Model (Native).")
        else:
            print(f"♻️ Sử dụng Embedding Model đã có sẵn trong Cache.")
            
        self.model = _MODEL_INSTANCES[model_key]

    @staticmethod
    def _to_sparse_vector(weights: Dict[str, float]) -> SparseVector:
        if not weights:
            return SparseVector(indices=[], values=[])
        pairs = [(int(k), float(v)) for k, v in weights.items() if float(v) != 0.0]
        pairs.sort(key=lambda x: x[0])
        return SparseVector(
            indices=[idx for idx, _ in pairs],
            values=[val for _, val in pairs],
        )

    def encode_hybrid(self, texts: Union[str, List[str]], batch_size: int = 16) -> Tuple[List[List[float]], List[SparseVector]]:
        """
        Thực hiện cả dense và sparse encoding trong DUY NHẤT 1 LẦN pass qua model.
        Sử dụng Lexicon Weights (Học máy) nguyên bản của BGE-M3.
        """
        # Guard: handle None input
        if texts is None:
            texts = ["N/A"]
        if isinstance(texts, str):
            texts = [texts]

        # Guard: ensure all items are non-empty strings (prevents TextEncodeInput tokenizer errors)
        texts = [str(t) if t is not None else "" for t in texts]
        texts = [t if t.strip() else "N/A" for t in texts]

        # SPEED OPTIMIZATION: SINGLE FORWARD PASS
        outputs = self.model.encode(
            texts,
            batch_size=(128 if torch.cuda.is_available() else batch_size),
            max_length=2048,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # 1. Dense (Pooling mặc định của BGE-M3)
        dense_vecs = outputs["dense_vecs"]
        dense_list = dense_vecs.tolist() if hasattr(dense_vecs, "tolist") else [list(vec) for vec in dense_vecs]

        # 2. Sparse (Lexical weights nguyên bản)
        lexical_weights = outputs["lexical_weights"]
        sparse_list = [self._to_sparse_vector(weights) for weights in lexical_weights]

        return dense_list, sparse_list

    def encode_dense(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[List[float]]:
        return self.encode_hybrid(texts, batch_size=batch_size)[0]

    def encode_sparse_documents(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[SparseVector]:
        return self.encode_hybrid(texts, batch_size=batch_size)[1]

    def encode_query_sparse(self, text: str) -> SparseVector:
        return self.encode_hybrid([text], batch_size=1)[1][0]

    def encode_query_dense(self, text: str) -> List[float]:
        return self.encode_hybrid([text], batch_size=1)[0][0]

# --- KHỞI TẠO SINGLETON TỐI ƯU ---
_embedder_cache = None

def get_embedder():
    global _embedder_cache
    if _embedder_cache is not None:
        return _embedder_cache
        
    server_url = os.getenv("EMBEDDING_SERVER_URL")
    if server_url and not os.getenv("IS_EMBEDDING_SERVER"):
        from backend.retrieval.remote_embedder import RemoteBGEHybridEncoder
        print(f"🔗 [Embedder] Chế độ Remote: Kết nối tới {server_url}")
        _embedder_cache = RemoteBGEHybridEncoder()
    else:
        print(f"🏠 [Embedder] Chế độ Local: Đang chuẩn bị nạp mô hình vào RAM...")
        _embedder_cache = LocalBGEHybridEncoder()
    return _embedder_cache

# Proxy class đơn giản
class EmbedderProxy:
    def __getattr__(self, name):
        return getattr(get_embedder(), name)
    def __repr__(self):
        return "<EmbedderProxy (BGE-M3 Native)>"

embedder = EmbedderProxy()
