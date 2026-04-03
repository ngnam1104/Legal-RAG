import re
from typing import List, Dict, Optional
import requests
from backend.config import settings
from backend.llm.base import BaseLLMClient

class OllamaClient(BaseLLMClient):
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip('/')
        self.api_key = settings.OLLAMA_API_KEY
        
    def _clean_reasoning(self, text: str) -> str:
        """Loại bỏ các block <think>...</think> hoặc 'Thinking process' khỏi phản hồi cuối cùng."""
        # Remove <think>...</think>
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove common "Thinking Process" headers if they remain
        text = re.sub(r'(?i)Thinking Process:.*?\n\n', '', text, flags=re.DOTALL)
        return text.strip()

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.3, model: str = None) -> str:
        model_name = model or settings.OLLAMA_CHAT_MODEL
        
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        # Một số Cloud Ollama (như qwen3.5:cloud qua proxy) yêu cầu Header cụ thể
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            # Ưu tiên OLLAMA-API-KEY cho cloud/proxy, fallback về Bearer
            headers["OLLAMA-API-KEY"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        try:
            # Tăng timeout lên 60s cho các model Cloud / Reasoning (nó cần thời gian để 'suy nghĩ')
            response = requests.post(
                f"{self.base_url}/api/chat", 
                json=payload,
                headers=headers,
                timeout=90 
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            
            # Xử lý làm sạch nếu model trả về nội dung nháp (thinking process)
            return self._clean_reasoning(content)
        
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 502:
                raise Exception(f"Ollama Cloud ({model_name}) trả về 502 Bad Gateway. Server Cloud tạm thời quá tải, vui lòng thử lại sau.")
            raise Exception(f"Lỗi khi gọi Ollama API ({model_name}): HTTP {status} - {str(e)}")
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama API ({model_name}) bị quá tải hoặc phản hồi quá lâu. Vui lòng thử lại.")
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Không thể kết nối tới Ollama tại {self.base_url}. \n"
                "Kiểm tra xem Ollama đã được bật chưa (Lệnh: 'ollama serve')."
            )
        except Exception as e:
            raise Exception(f"Lỗi khi gọi Ollama API ({model_name}): {str(e)}")
