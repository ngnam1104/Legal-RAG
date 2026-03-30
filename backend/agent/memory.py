import json
from typing import List, Dict
import redis
from backend.config import settings

class ChatSessionManager:
    """Quản lý Memory (Lưu 7 turn chat gần nhất) qua Redis hoặc Memory cục bộ."""

    def __init__(self, max_turns: int = 7):
        self.max_turns = max_turns
        self.use_redis = False
        self.local_sessions = {}
        
        try:
            self.redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
        except Exception:
            print("Cảnh báo: Không kết nối được Redis, sẽ dùng bộ nhớ RAM cục bộ cho session.")

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        if self.use_redis:
            val = self.redis_client.get(f"session:{session_id}")
            return json.loads(val) if val else []
        return self.local_sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})

        # Giữ lại `max_turns` lượt hội thoại gần nhất (nhân 2 vì 1 lượt có 1 hỏi 1 đáp)
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        if self.use_redis:
            self.redis_client.set(f"session:{session_id}", json.dumps(history), ex=86400) # 1 ngày
        else:
            self.local_sessions[session_id] = history
