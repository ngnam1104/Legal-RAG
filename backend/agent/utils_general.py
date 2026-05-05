"""
utils_general.py — General-purpose utilities for the Agent pipeline.
====================================================================
Hợp nhất từ: sub_timer.py + utils_general_chat.py + utils_conversation.py
"""

import json
import time
import logging
from contextlib import contextmanager
from typing import List, Dict, Any

from backend.models.llm_factory import chat_completion
import os

logger = logging.getLogger("utils_general")


# =====================================================================
# SubTimer — Đo thời gian chi tiết từng bước trong pipeline RAG
# =====================================================================

class SubTimer:
    """Đo thời gian chi tiết của các sub-steps bên trong một node LangGraph.
    
    Sử dụng:
        timer = SubTimer("Retrieve")
        with timer.step("Qdrant_Search"):
            ... # code tìm kiếm
        with timer.step("Rerank"):
            ... # code rerank
        
        timer.results()  
        # => {"Retrieve.Qdrant_Search": 0.45, "Retrieve.Rerank": 0.19}
    """

    def __init__(self, parent_name: str):
        """
        Args:
            parent_name: Tên node cha (VD: "Retrieve", "Generate", "Route").
                         Sẽ được dùng làm prefix cho tên sub-step.
        """
        self._parent = parent_name
        self._timings: Dict[str, float] = {}

    @contextmanager
    def step(self, name: str):
        """Context manager đo 1 sub-step.
        
        Args:
            name: Tên bước nhỏ (VD: "Qdrant_Search", "LLM_Call").
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - t0
            key = f"{self._parent}.{name}"
            # Cộng dồn nếu cùng tên (VD: nhiều lần gọi Neo4j)
            self._timings[key] = self._timings.get(key, 0.0) + duration

    def results(self) -> Dict[str, float]:
        """Trả về dict {sub_step_name: duration_seconds}."""
        return dict(self._timings)


# =====================================================================
# General Chat — Luồng chat thông thường (bypass RAG)
# =====================================================================

from backend.prompt import GENERAL_SYSTEM_PROMPT, ENTITY_EXTRACTION_PROMPT

def execute_general_chat(query: str, history: List[Dict[str, str]] = None, file_chunks: List[Dict[str, Any]] = None, llm_preset: str = None) -> str:
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    
    if file_chunks:
        file_text = ""
        for idx, f_chunk in enumerate(file_chunks, start=1):
            text = f_chunk.get("text_to_embed", f_chunk.get("unit_text", ""))
            file_text += f"[Tài liệu đính kèm {idx}]\n{text}\n"
        messages[0]["content"] += f"\n\nBẠN HÃY ƯU TIÊN SỬ DỤNG TÀI LIỆU SAU ĐÂY ĐỂ TRẢ LỜI CÂU HỎI (NẾU CÓ THỂ):\n<tai_lieu>\n{file_text.strip()}\n</tai_lieu>"
    
    if history:
        # Lấy tối đa 10 tin nhắn gần nhất để tránh quá tải context
        recent_history = history[-10:]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": f"QUERY =\n{query}"})
    raw = chat_completion(messages, temperature=0.7, model=os.environ.get("LLM_CORE_MODEL", "llama3"), llm_preset=llm_preset)
    return raw if raw else ""


class GeneralChatFlow:
    """Luồng chat thông thường — trả lời mọi câu hỏi không liên quan pháp lý."""
    
    def __init__(self):
        self.system_prompt = GENERAL_SYSTEM_PROMPT

    def execute(self, query: str) -> str:
        print(f"    → [General Chat] Gọi LLM trực tiếp (không RAG)...")
        t0 = time.perf_counter()
        raw_answer = execute_general_chat(query)
        from backend.utils.text_utils import extract_thinking_and_answer
        _, answer = extract_thinking_and_answer(raw_answer)
        print(f"    → [General Chat] ✅ LLM trả lời ({time.perf_counter()-t0:.1f}s)")
        return answer

general_chat_flow = GeneralChatFlow()


# =====================================================================
# Conversation Utilities — Entity extraction sau mỗi lượt chat
# =====================================================================

# ENTITY_EXTRACTION_PROMPT imported from backend.prompt


def extract_entities(query: str, answer: str) -> Dict[str, Any]:
    """Trích xuất thực thể và văn bản hiện tại sau mỗi lượt phản hồi."""
    prompt = ENTITY_EXTRACTION_PROMPT.format(query=query, answer=answer)
    
    try:
        resp = chat_completion(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            model=os.environ.get("LLM_ROUTING_MODEL", "llama3")
        )
        
        from backend.utils.text_utils import extract_json_from_text
        json_str = extract_json_from_text(resp)
        
        if json_str:
            data = json.loads(json_str)
            return {
                "current_document": data.get("current_document"),
                "entities": data.get("entities", [])
            }
    except Exception as e:
        logger.error(f"    [Extraction Error] {e}")
    
    return {"current_document": None, "entities": []}
