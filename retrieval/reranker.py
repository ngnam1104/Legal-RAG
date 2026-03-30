from typing import List, Dict
from retrieval.base import BaseReranker

class DocumentReranker(BaseReranker):
    def __init__(self):
        pass

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Sắp xếp lại các Documents dựa trên score trả về từ Hybrid Search.
        Nếu muốn dùng mô hình Reranker của BAAI (CrossEncoder), có thể tích hợp tại đây.
        Hiện tại dùng mặc định score của Qdrant RRF.
        """
        if not documents:
            return []
            
        # Sắp xếp lại theo điểm số (cao xuống thấp)
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
        
        # Lấy top K làm context
        return sorted_docs[:top_k]

reranker = DocumentReranker()
