"""
Internal LLM Client — On-Premise Adapter
=========================================
Gọi REST API nội bộ Llama-3 8B tại 10.9.3.75:30028.
Payload có format riêng (KHÔNG phải chuẩn OpenAI).

**ĐẶC BIỆT**: ``json_structure`` bị lỗi trên server → KHÔNG dùng.
              Thay vào đó, khi cần JSON output, nối thêm prompt chỉ thị
              vào ``system_prompt``.
"""


from __future__ import annotations
import time
import datetime

import logging
from typing import Dict, List, Optional

import requests

from backend.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LLM_ENDPOINT: str = "http://10.9.3.75:30028/api/llama3/8b"
_REQUEST_TIMEOUT: int = 120  # seconds — LLM inference có thể lâu

_JSON_ENFORCEMENT_PROMPT: str = (
    "\nQUAN TRỌNG: Bạn BẮT BUỘC phải trả lời bằng định dạng JSON hợp lệ. "
    "Không kèm theo bất kỳ văn bản giải thích nào ngoài khối JSON."
)

# ---------------------------------------------------------------------------
# Singleton guard
# ---------------------------------------------------------------------------
_instance: "InternalLLMClient | None" = None


class InternalLLMClient(BaseLLMClient):
    """
    LLM Client gọi API Llama-3 8B nội bộ.

    * Singleton: chỉ khởi tạo 1 lần.
    * Mapping: chuyển đổi ``messages`` (OpenAI-style) sang payload nội bộ.
    * Anti-hallucination: tự nối prompt JSON nếu ``response_format`` yêu cầu.
    """

    def __new__(cls, *args, **kwargs) -> "InternalLLMClient":
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialized = False
        return _instance

    def __init__(
        self,
        endpoint: str = _LLM_ENDPOINT,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        if self._initialized:
            return
        self.endpoint: str = endpoint
        self.timeout: int = timeout
        self._session: requests.Session = requests.Session()
        self._initialized = True
        logger.info("✅ [InternalLLMClient] Singleton khởi tạo — endpoint: %s", self.endpoint)

    # ------------------------------------------------------------------
    # Public API (BaseLLMClient interface)
    # ------------------------------------------------------------------
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[Dict] = None,
        model: str = None,  # giữ tương thích interface BaseLLMClient
        max_input_length: int = 20000,  # Tăng lên 20000 theo yêu cầu
    ) -> str:
        """
        Gửi hội thoại tới LLM nội bộ và trả về chuỗi kết quả.

        Args:
            messages:        Danh sách ``{"role": "system"|"user", "content": "..."}``.
            temperature:     Nhiệt độ sinh (0.0-1.0).
            max_tokens:      Độ dài tối đa output.
            response_format: Nếu ``{"type": "json_object"}`` → nối thêm prompt JSON.
            model:           Không sử dụng (giữ tương thích interface).
            max_input_length: Giới hạn độ dài input (default 4000, tăng cho prompt dài).

        Returns:
            Chuỗi phản hồi từ LLM.  Trả ``""`` nếu API lỗi.
        """
        # ---- 1. Tách messages thành system_prompt & conversation ----
        system_parts: List[str] = []
        conversation: List[str] = []

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                conversation.append(f"User: {content}")
            elif role == "assistant":
                conversation.append(f"AI: {content}")
            else:
                conversation.append(f"{role}: {content}")

        system_prompt_text = "\n".join(system_parts)

        # ---- 2. Chống ảo giác JSON ----
        if response_format and response_format.get("type") == "json_object":
            system_prompt_text += _JSON_ENFORCEMENT_PROMPT

        # FIX: Gom System Prompt vào câu hỏi của User vì API không hỗ trợ key 'system_prompt'
        final_question = ""
        if system_prompt_text.strip():
            final_question += f"[SYSTEM PROMPT]\n{system_prompt_text}\n\n[CONVERSATION]\n"
        
        if conversation:
            final_question += "\n".join(conversation)
            # Thêm token mồi để LLM hiểu đến lượt mình
            if not final_question.endswith("AI:"):
                final_question += "\nAI: "
        else:
            final_question += "User: Hello\nAI: "

        # ---- 3. Build payload theo format API nội bộ ----
        payload = {
            "questions": [final_question],
            "contexts": [""],             # 🐛 FIX CỐT LÕI: Phải truyền chuỗi rỗng "" để độ dài mảng khớp với questions
            "lang": "vi",
            "use_en_model": False,
            "batch_size": 1,
            "max_decoding_length": max_tokens,
            "max_input_length": max_input_length,
            "repetition_penalty": 0,
            "temperature": temperature,
            "do_sample": True,
            "no_repeat_ngram_size": 0,
            "add_generation_prompt": True,
            "tokenize": False,
            "histories": []
        }

        import time
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.post(
                    self.endpoint, json=payload, timeout=self.timeout,
                )
                resp.raise_for_status()

                data = resp.json()
                results: List[str] = data.get("result", [])
                answer = results[0] if results else ""
                return answer.strip()

            except requests.exceptions.Timeout:
                logger.error("❌ [InternalLLMClient] Timeout (Attempt %d/%d).", attempt, max_retries)
                print(f"🔥 DEBUG: Timeout khi gọi API LLM (Lần thử {attempt}/{max_retries}).")
            except requests.exceptions.HTTPError as exc:
                err_detail = exc.response.text if exc.response is not None else "???"
                logger.error("❌ [InternalLLMClient] HTTP Error (Attempt %d/%d): %s", attempt, max_retries, err_detail)
                print(f"🔥 DEBUG HTTP ERROR: {exc.response.status_code} - {err_detail} (Lần thử {attempt}/{max_retries})") 
            except requests.exceptions.ConnectionError:
                logger.error("❌ [InternalLLMClient] Lỗi kết nối (Attempt %d/%d).", attempt, max_retries)
                print(f"🔥 DEBUG: ConnectionError (Sai IP hoặc port) (Lần thử {attempt}/{max_retries}).")
            except Exception as e:
                logger.exception("❌ [InternalLLMClient] Lỗi không xác định (Attempt %d/%d).", attempt, max_retries)
                print(f"🔥 DEBUG FATAL ERROR: {str(e)} (Lần thử {attempt}/{max_retries})")

            if attempt < max_retries:
                time.sleep(8)
                
        # Fallback sau khi hết max_retries
        return ""

    def batch_chat_completion(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[Dict] = None,
        max_input_length: int = 20000,
    ) -> List[str]:
        """
        Xử lý mẻ (Batch) nhiều câu hỏi trong cùng một request tới server.
        Args:
            messages_list: Danh sách các list tin nhắn.
        Returns:
            Danh sách các câu trả lời tương ứng.
        """
        if not messages_list:
            return []

        questions = []
        for messages in messages_list:
            system_parts: List[str] = []
            conversation: List[str] = []
            for msg in messages:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "system":
                    system_parts.append(content)
                elif role == "user":
                    conversation.append(f"User: {content}")
                elif role == "assistant":
                    conversation.append(f"AI: {content}")
                else:
                    conversation.append(f"{role}: {content}")

            system_prompt_text = "\n".join(system_parts)
            if response_format and response_format.get("type") == "json_object":
                system_prompt_text += _JSON_ENFORCEMENT_PROMPT

            final_q = ""
            if system_prompt_text.strip():
                final_q += f"[SYSTEM PROMPT]\n{system_prompt_text}\n\n[CONVERSATION]\n"
            
            if conversation:
                final_q += "\n".join(conversation)
                if not final_q.endswith("AI:"):
                    final_q += "\nAI: "
            else:
                final_q += "User: Hello\nAI: "
            
            questions.append(final_q)

        payload = {
            "questions": questions,
            "contexts": [""] * len(questions),
            "lang": "vi",
            "use_en_model": False,
            "batch_size": 8,
            "max_decoding_length": max_tokens,
            "max_input_length": max_input_length,
            "repetition_penalty": 0,
            "temperature": temperature,
            "do_sample": True,
            "no_repeat_ngram_size": 0,
            "add_generation_prompt": True,
            "tokenize": False,
            "histories": []
        }

        max_retries = 2 # Giảm retry cho batch để tránh treo quá lâu
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.post(self.endpoint, json=payload, timeout=600) # Tăng timeout lên 600s cho mẻ Batch mẻ lớn
                resp.raise_for_status()
                data = resp.json()
                results = data.get("result", [])
                return [r.strip() for r in results]
            except Exception as e:
                logger.error("❌ [InternalLLMClient] Batch failed (Attempt %d/%d): %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(5)
        
        return [""] * len(questions)

    def __repr__(self) -> str:
        return f"<InternalLLMClient endpoint={self.endpoint!r}>"
