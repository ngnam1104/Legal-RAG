from typing import List, Dict
from openai import OpenAI
from backend.config import settings
from backend.llm.base import BaseLLMClient

class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY or "dummy_key",
            base_url=settings.LLM_BASE_URL,
        )

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.3, model: str = None) -> str:
        response = self.client.chat.completions.create(
            model=model or settings.LLM_CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
