from typing import List, Dict
from backend.config import settings
from backend.llm.base import BaseLLMClient

class GeminiClient(BaseLLMClient):
    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.3, model: str = None) -> str:
        from google.genai import types as genai_types
        system_instruction = None
        contents = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_instruction = m["content"]
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    genai_types.Content(
                        role=gemini_role,
                        parts=[genai_types.Part(text=m["content"])],
                    )
                )

        config = genai_types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        response = self.client.models.generate_content(
            model=model or settings.GEMINI_CHAT_MODEL,
            contents=contents,
            config=config,
        )
        return response.text
