import os
import logging
from typing import Dict, List, Optional
try:
    from icllmlib import LLM
except ImportError:
    LLM = None

from backend.models.interfaces import BaseLLMClient

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
        
        # 1. Xây dựng prompt string từ messages (OpenAI-style → flat prompt)
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
                    "max_tokens": str(max_tokens),
                    "do_sample": "false",
                    "repetition_penalty": "1.0"
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
        """
        Xử lý mẻ (Batch) nhiều câu hỏi SONG SONG qua ICLLM bằng ThreadPoolExecutor.
        Số worker song song = env LLM_PARALLEL_WORKERS (mặc định 4).
        Inter-batch sleep giữa các micro-batch để tránh quá tải server.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        if not messages_list:
            return []

        parallel_workers  = int(os.environ.get("LLM_PARALLEL_WORKERS",  8))
        micro_batch_size  = int(os.environ.get("LLM_MICRO_BATCH_SIZE",  parallel_workers))
        inter_batch_sleep = float(os.environ.get("LLM_INTER_BATCH_SLEEP", 2.0))

        def _call_one(idx_msgs):
            idx, msgs = idx_msgs
            max_retries = 3
            for attempt in range(max_retries):
                res = self.chat_completion(
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    max_input_length=max_input_length,
                )
                if res:
                    return idx, res
                
                logger.warning(f"⚠️ [Batch] Lỗi/Timeout gọi LLM (idx={idx}, attempt={attempt+1}/{max_retries}). Đang thử lại...")
                time.sleep(2 ** attempt)
            
            # Thất bại hoàn toàn sau 3 lần
            logger.error(f"❌ [Batch] LLM thất bại hoàn toàn sau {max_retries} lần thử (idx={idx}).")
            try:
                import os
                os.makedirs(".debug", exist_ok=True)
                with open(".debug/llm_failures.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n--- [API Call Failed Completely] idx={idx} ---\n")
                    if msgs and len(msgs) > 0 and "content" in msgs[0]:
                        f.write(f"Prompt preview: {str(msgs[0]['content'])[:500]}...\n")
                    f.write("-" * 50 + "\n")
            except:
                pass

            return idx, ""

        all_results = [""] * len(messages_list)
        for batch_start in range(0, len(messages_list), micro_batch_size):
            batch_slice = list(enumerate(
                messages_list[batch_start : batch_start + micro_batch_size],
                start=batch_start,
            ))
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(_call_one, item): item[0] for item in batch_slice}
                for future in as_completed(futures):
                    try:
                        idx, res = future.result()
                        all_results[idx] = res
                    except Exception as e:
                        orig_idx = futures[future]
                        logger.error(f"❌ [Batch] Worker idx={orig_idx} gặp lỗi: {e}")
                        all_results[orig_idx] = ""

            if batch_start + micro_batch_size < len(messages_list):
                time.sleep(inter_batch_sleep)

        return all_results


    def __repr__(self) -> str:
        return f"<ICLLMClient using {_ICLLM_CONFIG['UrlLLMApi']}>"
