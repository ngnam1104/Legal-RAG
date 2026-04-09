from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLMClient(ABC):
    """Lớp nền tảng trừu tượng cho tất cả các LLM Client."""
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, model: str = None) -> str:
        """
        Gửi hội thoại tới LLM và trả về phản hồi dưới dạng string.
        
        :param messages: Danh sách dictionary chứa 'role' và 'content'.
        :param temperature: Sự sáng tạo của model.
        :param model: Tên model muốn dùng (tuỳ chọn).
        """
        pass
