import time
import random
from typing import List, Dict
from openai import OpenAI, RateLimitError
from backend.config import settings
from backend.llm.base import BaseLLMClient

class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.LLM_API_KEY or "dummy_key",
            base_url=settings.LLM_BASE_URL,
        )

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1, model: str = None) -> str:
        max_retries = settings.LLM_MAX_RETRIES
        retry_delay = settings.LLM_RETRY_DELAY  # High delay for Free Tier reliability

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model or settings.LLM_CHAT_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"\n🚨 [Groq 429] RATE LIMIT HIT! (TPM/RPM quota exceeded).")
                    print(f"   ∟ Model: {model or settings.LLM_CHAT_MODEL}")
                    print(f"   ∟ Waiting {wait_time:.1f}s for quota refill... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ [Groq] Rate Limit reached maximum retries. Job failed.")
                    raise e
            except Exception as e:
                print(f"❌ [Groq] Error: {e}")
                raise e
