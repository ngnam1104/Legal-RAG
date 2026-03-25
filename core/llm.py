"""
Adapter LLM thống nhất: Groq (OpenAI-compatible) hoặc Gemini 3 Flash.
"""
from typing import List, Dict
from openai import OpenAI
from google.genai import types as genai_types
from core.config import settings

_openai_client = OpenAI(
    api_key=settings.LLM_API_KEY or "dummy_key",
    base_url=settings.LLM_BASE_URL,
)

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.3, provider: str = None, model: str = None) -> str:
    """
    Trả về string answer từ provider cấu hình qua LLM_PROVIDER.
    messages: [{role, content}]
    """
    provider = (provider or settings.LLM_PROVIDER or "groq").lower()

    if provider == "gemini":
        client = _get_gemini_client()

        # Tách system instruction (nếu có) ra khỏi contents
        system_instruction = None
        contents = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_instruction = m["content"]
            else:
                # Gemini SDK dùng "model" thay vì "assistant"
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

        response = client.models.generate_content(
            model=model or settings.GEMINI_CHAT_MODEL,
            contents=contents,
            config=config,
        )
        return response.text

    # Default: Groq / OpenAI compatible
    response = _openai_client.chat.completions.create(
        model=model or settings.LLM_CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
