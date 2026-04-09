from abc import ABC, abstractmethod
from typing import Dict, Any
from backend.agent.state import AgentState

class BaseRAGStrategy(ABC):
    """
    Abstract Universal Reasoning Framework for RAG pipelines.
    Mọi luồng (LegalQA, SectorSearch, ConflictAnalyzer) đều phải implement 5 node cốt lõi này.
    """
    
    @abstractmethod
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """
        BƯỚC 1: Phân tích & Tái cấu trúc truy vấn.
        Xử lý HyDE, trích xuất Entity, map metadata filters.
        Output: Cập nhật `rewritten_queries`, `metadata_filters`, `pending_tasks`.
        """
        pass
        
    @abstractmethod
    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """
        BƯỚC 2: Truy xuất dữ liệu (Vector DB / Hybrid Search).
        Output: Cập nhật danh sách `raw_hits`. 
        """
        pass
        
    @abstractmethod
    def resolve_references(self, state: AgentState) -> Dict[str, Any]:
        """
        BƯỚC 2.5: Truy xuất đệ quy (Recursive Reference Resolution).
        Tìm các Điều/Khoản được dẫn chiếu trong `raw_hits` và lấy thêm nội dung.
        Output: Cập nhật `recursive_hits`.
        """
        pass
        
    @abstractmethod
    def grade(self, state: AgentState) -> Dict[str, Any]:
        """
        BƯỚC 3: CRAG Filter (Kiểm định Ngữ cảnh).
        Output: Trả về `filtered_context` (chỉ giữ chunk tốt) và `is_sufficient` (Boolean).
        """
        pass
        
    @abstractmethod
    def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        BƯỚC 4: Sinh Output theo Chain-of-Thought dựa trên Task / Core Domain.
        Output: Cập nhật `draft_response`.
        """
        pass
        
    @abstractmethod
    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """
        BƯỚC 5: Đánh giá Output (Self-Correction).
        Output: Cập nhật `final_response`, `feedback`, và `pass_flag`.
        """
        pass
