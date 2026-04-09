from typing import List, Dict
from backend.config import settings

# Lazy loading to avoid importing packages if not used
_clients = {}

def get_client(provider: str):
    provider = provider.lower()
    if provider not in _clients:
        if provider == "gemini":
            from backend.llm.gemini_client import GeminiClient
            _clients[provider] = GeminiClient()
        elif provider == "ollama":
            from backend.llm.ollama_client import OllamaClient
            _clients[provider] = OllamaClient()
        else:
            # Default to Groq (or OpenAI compatible)
            from backend.llm.groq_client import GroqClient
            _clients["groq"] = GroqClient()
            provider = "groq"
    return _clients[provider]

def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.1, provider: str = None, model: str = None, llm_preset: str = None) -> str:
    """Unified entry point for LLM chat completion. Always returns str, never None."""
    # 1. Preset Mapping
    if llm_preset:
        if llm_preset == "groq_8b":
            provider = "groq"
            # Force everything to 8B even if a node attempts to use CORE_MODEL
            model = settings.LLM_ROUTING_MODEL
        elif llm_preset == "groq_70b":
            provider = "groq"
            # Keep existing logic: Use whatever model is passed (ROUTING or CORE)
            model = model or settings.LLM_ROUTING_MODEL
        elif llm_preset == "gemini":
            provider = "gemini"
            model = settings.GEMINI_CHAT_MODEL
        elif llm_preset == "ollama":
            provider = "ollama"
            model = settings.OLLAMA_CHAT_MODEL
            
    # 2. Final Fallbacks
    provider = provider or settings.LLM_PROVIDER or "groq"
    model = model or getattr(settings, "LLM_ROUTING_MODEL", settings.LLM_CHAT_MODEL)
    
    client = get_client(provider)
    result = client.chat_completion(messages, temperature=temperature, model=model)
    # Guard: LLM APIs can return None (e.g. Groq message.content=null, Gemini safety filter)
    return result if isinstance(result, str) else (str(result) if result is not None else "")
