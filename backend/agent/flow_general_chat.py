import time
from typing import List, Dict
from backend.llm.factory import chat_completion

# --- PROMPT ---
GENERAL_SYSTEM_PROMPT = """
Bạn là trợ lý AI thông minh, thân thiện và hữu ích.
Trả lời câu hỏi của người dùng một cách rõ ràng, súc tích.
Bạn có thể trả lời mọi chủ đề: công nghệ, cuộc sống, khoa học, toán học, lập trình, v.v.
Trả lời bằng tiếng Việt nếu người dùng hỏi bằng tiếng Việt.
"""

# --- STANDALONE FUNCTIONS FOR LANGGRAPH ---

def execute_general_chat(query: str, history: List[Dict[str, str]] = None) -> str:
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    
    if history:
        # Lấy tối đa 10 tin nhắn gần nhất để tránh quá tải context
        recent_history = history[-10:]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": query})
    
    return chat_completion(messages, temperature=0.5)

# --- COMPATIBILITY CLASS ---

class GeneralChatFlow:
    """Luồng chat thông thường — trả lời mọi câu hỏi không liên quan pháp lý."""
    
    def __init__(self):
        self.system_prompt = GENERAL_SYSTEM_PROMPT

    def execute(self, query: str) -> str:
        print(f"    → [General Chat] Gọi LLM trực tiếp (không RAG)...")
        t0 = time.perf_counter()
        answer = execute_general_chat(query)
        print(f"    → [General Chat] ✅ LLM trả lời ({time.perf_counter()-t0:.1f}s)")
        return answer

general_chat_flow = GeneralChatFlow()
