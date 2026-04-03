from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional
from qdrant_client.models import SparseVector, FieldCondition

class BaseEmbedder(ABC):
    """Lớp trừu tượng cho Mô hình Embedding (Tạo Vector)."""
    @abstractmethod
    def encode_dense(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[List[float]]:
        pass
        
    @abstractmethod
    def encode_sparse_documents(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[SparseVector]:
        pass

    @abstractmethod
    def encode_query_sparse(self, text: str) -> SparseVector:
        pass

    @abstractmethod
    def encode_query_dense(self, text: str) -> List[float]:
        pass

class BaseRetriever(ABC):
    """Lớp trừu tượng cho thao tác tìm kiếm Vector DB."""
    @abstractmethod
    def search(self, query: str, limit: int = 10, filter_conditions: Optional[List[FieldCondition]] = None) -> List[Dict]:
        pass

class BaseReranker(ABC):
    """Lớp trừu tượng cho thuật toán Reranking."""
    @abstractmethod
    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        pass
