from typing import List, Dict
import requests
from backend.config import settings
from backend.llm.base import BaseLLMClient

class OllamaClient(BaseLLMClient):
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip('/')
        
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.3, model: str = None) -> str:
        payload = {
            "model": model or settings.OLLAMA_CHAT_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        headers = {}
        if settings.OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"
            
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", 
                json=payload,
                headers=headers,
                timeout=30 # Thêm timeout để tránh treo
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Không thể kết nối tới Ollama tại {self.base_url}. \n"
                "Mẹo: Nếu bạn chạy Ollama trên Windows, hãy đảm bảo đã thiết lập biến môi trường OLLAMA_HOST=0.0.0.0 và khởi động lại Ollama."
            )
        except Exception as e:
            raise Exception(f"Lỗi khi gọi Ollama API: {str(e)}")
