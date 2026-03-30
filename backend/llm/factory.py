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

def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.3, provider: str = None, model: str = None) -> str:
    """Unified entry point for LLM chat completion."""
    provider = provider or settings.LLM_PROVIDER or "groq"
    client = get_client(provider)
    return client.chat_completion(messages, temperature=temperature, model=model)
