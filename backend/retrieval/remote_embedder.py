import os
import requests
from typing import List, Union
from qdrant_client.models import SparseVector
from backend.retrieval.base import BaseEmbedder

class RemoteBGEHybridEncoder(BaseEmbedder):
    """
    Gửi mã hóa tới Embedding Server (Container khác).
    Giúp API và Worker khởi động ngay lập tức mà không cần nạp mô hình cục bộ.
    """
    def __init__(self, endpoint: str = None):
        if not endpoint:
            # Giá trị mặc định trong Docker mạng nội bộ
            endpoint = os.getenv("EMBEDDING_SERVER_URL", "http://embedding:8001").rstrip('/')
        self.endpoint = endpoint

    def encode_dense(self, texts: Union[str, List[str]], batch_size: int = 16) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            resp = requests.post(
                f"{self.endpoint}/embed_dense", 
                json={"texts": texts, "batch_size": batch_size},
                timeout=120
            )
            resp.raise_for_status()
            return resp.json()["vectors"]
        except Exception as e:
            raise Exception(f"❌ Lỗi gọi Embedding Server (Dense): {e}")

    def encode_sparse_documents(self, texts: Union[str, List[str]], batch_size: int = 16) -> List[SparseVector]:
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            resp = requests.post(
                f"{self.endpoint}/embed_sparse",
                json={"texts": texts, "batch_size": batch_size},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()["vectors"]
            # Chốt lại định dạng SparseVector của Qdrant
            return [SparseVector(indices=v["indices"], values=v["values"]) for v in data]
        except Exception as e:
            raise Exception(f"❌ Lỗi gọi Embedding Server (Sparse): {e}")

    def encode_query_dense(self, text: str) -> List[float]:
        return self.encode_dense([text], batch_size=1)[0]

    def encode_query_sparse(self, text: str) -> SparseVector:
        return self.encode_sparse_documents([text], batch_size=1)[0]
