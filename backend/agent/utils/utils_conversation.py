import json
import logging
from typing import List, Dict, Any, Optional
from backend.llm.factory import chat_completion
from backend.config import settings

logger = logging.getLogger("conversation_utils")


ENTITY_EXTRACTION_PROMPT = """Bạn là một hệ thống trích xuất thông tin pháp lý.
Hãy đọc [Câu hỏi] và [Câu trả lời] bên dưới để trích xuất:
1. Tên văn bản pháp luật chính đang được nhắc tới (current_document).
2. Danh sách các thực thể pháp lý khác (entities).

Định dạng trả về duy nhất là JSON:
{{
  "current_document": "Tên đầy đủ của văn bản, bao gồm cả số hiệu nếu có",
  "entities": ["Thực thể 1", "Thực thể 2"]
}}

[Câu hỏi]: {query}
[Câu trả lời]: {answer}

JSON:"""



def extract_entities(query: str, answer: str) -> Dict[str, Any]:
    """Trích xuất thực thể và văn bản hiện tại sau mỗi lượt phản hồi."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(query=query, answer=answer)
    
    try:
        resp = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            model=settings.LLM_ROUTING_MODEL
        )
        # Tìm block JSON
        start = resp.find("{")
        end = resp.rfind("}") + 1
        if start != -1 and end != -1:
            data = json.loads(resp[start:end])
            return {
                "current_document": data.get("current_document"),
                "entities": data.get("entities", [])
            }
    except Exception as e:
        logger.error(f"    [Extraction Error] {e}")
    
    return {"current_document": None, "entities": []}
