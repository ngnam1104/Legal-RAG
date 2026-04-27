import logging
from typing import Dict, List, Optional
try:
    from icllmlib import LLM
except ImportError:
    LLM = None

from backend.llm.base import BaseLLMClient
from backend.config import settings

logger = logging.getLogger(__name__)

# Thay đổi các thông số này theo cấu hình ICLLM thực tế của công ty
_ICLLM_CONFIG = {
    "AppCode": "LEGAL_RAG",
    "FunctionCode": "standard_chat",
    "ModelLLM": "llama3.1-8b", 
    "UrlPrompt": "https://staging.pontusinc.com/api/chatbot/v1/prompt/list",
    "LLMName": "legal_rag_chat",
    "UrlLLMApi": "http://10.9.3.75:30031/api/llama3/8b", # Hoặc endpoint mới như 10.9.3.241:5564/api/Qas/v2
    "BaseDirLog": "logs/llm_logs",
    "BaseDirPostProcess": "logs/llm_logs/post_process",
    "BaseDirPrompt": "logs/llm_logs/prompt",
    "IsLog": True,
    "IsShowConsole": False, # Có thể để True nếu muốn in chi tiết ICLLM ra terminal
    "IsGetPromptOnline": False, # Đặt False để chạy prompt dưới ổ cứng, hãy đảm bảo thư mục BaseDirPrompt có template phù hợp
}

_JSON_ENFORCEMENT_PROMPT = (
    "\nQUAN TRỌNG: Bạn BẮT BUỘC phải trả lời bằng định dạng JSON hợp lệ. "
    "Không kèm theo bất kỳ văn bản giải thích nào ngoài khối JSON."
)

_instance = None

class ICLLMClient(BaseLLMClient):
    """
    Adapter tích hợp thư viện ICLLM để có tính năng tự động ghi log vào ổ cứng.
    """
    def __new__(cls, *args, **kwargs) -> "ICLLMClient":
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialized = False
        return _instance

    def __init__(self):
        if self._initialized:
            return
            
        if LLM is None:
            logger.error("❌ [ICLLMClient] Thư viện 'icllmlib' không được tìm thấy. Vui lòng cài đặt hoặc kiểm tra PYTHONPATH.")
            raise ImportError("Không thể khởi tạo ICLLMClient do thiếu thư viện nội bộ 'icllmlib'.")
            
        self.llm_engine = LLM(
            app_code=_ICLLM_CONFIG["AppCode"],
            function_code=_ICLLM_CONFIG["FunctionCode"],
            model_llm=_ICLLM_CONFIG["ModelLLM"],
            url_prompt=_ICLLM_CONFIG["UrlPrompt"],
            llm_name=_ICLLM_CONFIG["LLMName"],
            url_llm_api=_ICLLM_CONFIG["UrlLLMApi"],
            base_dir_log=_ICLLM_CONFIG["BaseDirLog"],
            base_dir_post_process=_ICLLM_CONFIG["BaseDirPostProcess"],
            base_dir_prompt=_ICLLM_CONFIG["BaseDirPrompt"],
            is_log=_ICLLM_CONFIG["IsLog"],
            is_show_console=_ICLLM_CONFIG["IsShowConsole"],
            is_get_prompt_online=_ICLLM_CONFIG["IsGetPromptOnline"],
        )
        
        # Pull prompt template
        self.llm_engine.get_prompt()
        self._initialized = True
        logger.info("✅ [ICLLMClient] Khởi tạo thành công qua ICLLMLib.")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[Dict] = None,
        model: str = None,
        max_input_length: int = 20000,
    ) -> str:
        
        # 1. Xây dựng prompt string tương tự logic trong InternalLLMClient để duy trì độ tương thích
        system_parts = []
        conversation = []

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

        final_question = ""
        if system_prompt_text.strip():
            final_question += f"[SYSTEM PROMPT]\n{system_prompt_text}\n\n[CONVERSATION]\n"
        
        if conversation:
            final_question += "\n".join(conversation)
            if not final_question.endswith("AI:"):
                final_question += "\nAI: "
        else:
            final_question += "User: Hello\nAI: "

        try:
            response = self.llm_engine.generate(
                params={
                    "thinking_mode": "off",
                    "max_new_tokens": str(max_tokens),
                    "max_tokens": str(max_tokens)
                },
                prompt=final_question,
                temperature=temperature,
                is_translate_context=False,
                is_translate_prompt=False,
                is_translate_result=False
            )
            
            if response and len(response) > 0 and response[0].get("is_valid"):
                return response[0].get("answer_norm", "").strip()
            else:
                logger.error(f"❌ [ICLLM] Sinh thất bại hoặc bị chặn, response: {response}")
                return ""
                
        except Exception as e:
            logger.exception(f"❌ [ICLLM] Gặp lỗi hệ thống: {str(e)}")
            return ""

    def batch_chat_completion(
        self, 
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: Optional[Dict] = None,
        max_input_length: int = 20000,
    ) -> List[str]:
        # Giả lập Batch vì ICLLM có thể chỉ nhận đơn request trên lớp wrapper hiện tại
        results = []
        for msgs in messages_list:
            res = self.chat_completion(
                messages=msgs, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                response_format=response_format,
                max_input_length=max_input_length
            )
            results.append(res)
        return results

    def __repr__(self) -> str:
        return f"<ICLLMClient using {_ICLLM_CONFIG['UrlLLMApi']}>"
