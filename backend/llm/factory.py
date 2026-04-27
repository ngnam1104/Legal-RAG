from typing import List, Dict, Optional
from backend.config import settings

# Lazy loading to avoid importing packages if not used
_clients = {}

def get_client():
    if "internal" not in _clients:
        from backend.llm.icllm_client import ICLLMClient
        _clients["internal"] = ICLLMClient()
    return _clients["internal"]

def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.1, provider: str = None, model: str = None, llm_preset: str = None) -> str:
    """Unified entry point for LLM chat completion using the On-Premise Internal API. Always returns str."""
    client = get_client()
    result = client.chat_completion(messages, temperature=temperature)
    return result if isinstance(result, str) else (str(result) if result is not None else "")

def batch_completion(
    messages_list: List[List[Dict[str, str]]], 
    temperature: float = 0.1, 
    max_tokens: int = 1024, 
    max_input_length: int = 4000,
    response_format: Optional[Dict] = None # Thêm dòng này
) -> List[str]:
    """Unified entry point for Batch LLM completion."""
    client = get_client()
    return client.batch_chat_completion(
        messages_list, 
        temperature=temperature, 
        max_tokens=max_tokens, 
        max_input_length=max_input_length,
        response_format=response_format # Truyền tiếp xuống client
    )