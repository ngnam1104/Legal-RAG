import json
import sqlite3
import os
import uuid
import time
from typing import List, Dict, Optional
from datetime import datetime
import redis
from backend.config import settings
from backend.llm.factory import chat_completion

TITLE_PROMPT = """
Bạn là một trợ lý AI pháp luật chuyên nghiệp. Hãy tóm tắt câu hỏi của người dùng thành một tiêu đề ngắn gọn, chuyên nghiệp (tối đa 8 từ).
Ưu tiên đưa tên loại luật và mã hiệu luật (ví dụ: Nghị định 117/2020) vào tiêu đề nếu có trong câu hỏi. 
Nếu không có mã hiệu, hãy tóm tắt chủ đề pháp luật liên quan nhất.
Chỉ trả về chuỗi tiêu đề, không kèm theo bất kỳ lời dẫn hay dấu ngoặc kép nào.

Câu hỏi: {query}
Tiêu đề:
"""

GENERAL_TITLE_PROMPT = """
Bạn là một trợ lý AI thân thiện. Hãy tóm tắt câu hỏi hoặc lời chào của người dùng thành một tiêu đề ngắn gọn, tự nhiên (tối đa 8 từ).
Không cần dính dáng tới các thuật ngữ pháp lý nếu người dùng không hỏi về pháp luật.
Chỉ trả về chuỗi tiêu đề, không kèm theo bất kỳ lời dẫn hay dấu ngoặc kép nào.

Câu hỏi: {query}
Tiêu đề:
"""

# Đường dẫn SQLite DB - persist trên disk
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
SQLITE_DB_PATH = os.path.join(DATA_DIR, "chat_history.db")


class ChatSessionManager:
    """
    Quản lý Memory 2 tầng:
    - Redis: Short-term rolling window (7 turns gần nhất) cho LLM context, TTL 24h
    - SQLite: Long-term persistent storage (toàn bộ lịch sử, vĩnh viễn)
    """

    def __init__(self, max_turns: int = 7):
        self.max_turns = max_turns
        self.use_redis = False
        self.local_sessions = {}
        self.temp_chunks = {} # {session_id: [chunks]}

        # Redis (short-term)
        try:
            self.redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
        except Exception:
            print("⚠️ Redis không khả dụng, dùng RAM cho short-term memory.")

        # SQLite (long-term)
        self._init_sqlite()

    def _init_sqlite(self):
        """Khởi tạo bảng SQLite nếu chưa tồn tại."""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Phiên chat mới',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                references_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # =====================================================================
    # SESSION MANAGEMENT (SQLite)
    # =====================================================================

    def create_session(self, title: str = None, session_id: str = None) -> str:
        """Tạo phiên chat mới, trả về session_id."""
        sid = session_id or str(uuid.uuid4())
        now = datetime.now().isoformat()
        title = title or "Phiên chat mới"

        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, title, now, now)
            )
            conn.commit()
        finally:
            conn.close()
        return sid

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """Lấy danh sách phiên chat có ít nhất 1 trao đổi (User + Assistant)."""
        conn = self._get_conn()
        try:
            # Chỉ lấy các session có ít nhất 2 messages (User + Bot)
            query = """
                SELECT s.id, s.title, s.created_at, s.updated_at 
                FROM sessions s
                WHERE (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) >= 2
                ORDER BY s.updated_at DESC
                LIMIT ?
            """
            rows = conn.execute(query, (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Lấy thông tin 1 phiên."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_session_title(self, session_id: str, title: str):
        """Cập nhật tiêu đề phiên chat (auto-generate từ câu hỏi đầu tiên)."""
        conn = self._get_conn()
        try:
            conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                         (title, datetime.now().isoformat(), session_id))
            conn.commit()
        finally:
            conn.close()

    def delete_session(self, session_id: str):
        """Xóa phiên chat và toàn bộ messages."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

        # Xóa cache Redis
        if self.use_redis:
            self.redis_client.delete(f"session:{session_id}")

    # =====================================================================
    # MESSAGE MANAGEMENT
    # =====================================================================

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Short-term history (từ Redis) cho LLM context.
        Chỉ trả về max_turns * 2 messages gần nhất.
        """
        if self.use_redis:
            val = self.redis_client.get(f"session:{session_id}")
            if val:
                return json.loads(val)
        elif session_id in self.local_sessions:
            return self.local_sessions[session_id]

        # Fallback: Lấy từ SQLite (limited)
        return self._get_recent_messages(session_id, limit=self.max_turns * 2)

    def get_full_history(self, session_id: str) -> List[Dict]:
        """
        Long-term history (từ SQLite) — toàn bộ messages.
        Dùng để hiển thị trên UI khi mở lại phiên chat cũ.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT role, content, references_json, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            ).fetchall()
            result = []
            for r in rows:
                msg = {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
                try:
                    msg["references"] = json.loads(r["references_json"])
                except (json.JSONDecodeError, TypeError):
                    msg["references"] = []
                result.append(msg)
            return result
        finally:
            conn.close()

    def _get_recent_messages(self, session_id: str, limit: int = 14) -> List[Dict[str, str]]:
        """Lấy N messages gần nhất từ SQLite (fallback khi Redis down)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            ).fetchall()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        finally:
            conn.close()

    def add_message(self, session_id: str, role: str, content: str, references: List[Dict] = None, mode: str = "GENERAL_CHAT"):
        """
        Ghi message vào CẢ HAI tầng:
        1. Redis (short-term, rolling window)
        2. SQLite (long-term, permanent)
        """
        references = references or []
        now = datetime.now().isoformat()

        # Đảm bảo session tồn tại
        self.create_session(session_id=session_id)

        # --- TẦNG 1: Redis (short-term) ---
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        if self.use_redis:
            self.redis_client.set(f"session:{session_id}", json.dumps(history), ex=86400)
        else:
            self.local_sessions[session_id] = history

        # --- TẦNG 2: SQLite (long-term) ---
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, references_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(references, ensure_ascii=False), now)
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

            # Auto-generate title từ câu hỏi đầu tiên của user bằng LLM
            if role == "user":
                # Lấy message count VÀ title hiện tại để bảo vệ việc chỉ chạy LLM 1 lần duy nhất
                session_info = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
                current_title = session_info["title"] if session_info else "Phiên chat mới"

                msg_count = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'",
                    (session_id,)
                ).fetchone()[0]

                # Chỉ gọi LLM nếu là tin nhắn đầu tiên VÀ tiêu đề vẫn đang là mặc định
                if msg_count == 1 and current_title == "Phiên chat mới":
                    try:
                        # Chọn prompt dựa trên mode
                        current_prompt = TITLE_PROMPT if mode != "GENERAL_CHAT" else GENERAL_TITLE_PROMPT
                        
                        # Dùng LLM để tạo title thay vì cắt chuỗi
                        title_resp = chat_completion([{"role": "user", "content": current_prompt.format(query=content)}], temperature=0.1)
                        title = title_resp.strip().replace('"', '').replace("'", "")
                        if not title or len(title) > 100: # Fallback if LLM fails or is too long
                            title = content[:40] + ("..." if len(content) > 40 else "")
                    except Exception as e:
                        print(f"      ⚠️ Gợi ý tiêu đề thất bại: {e}. Dùng fallback.")
                        title = content[:40] + ("..." if len(content) > 40 else "")
                    
                    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))

            conn.commit()
        finally:
            conn.close()

    def get_message_count(self, session_id: str) -> int:
        """Đếm số messages trong một phiên."""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def delete_last_turn(self, session_id: str):
        """Xóa 2 tin nhắn cuối cùng (User + Assistant) trong phiên chat."""
        conn = self._get_conn()
        try:
            # Lấy IDs của 2 tin nhắn cuối cùng
            rows = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 2",
                (session_id,)
            ).fetchall()
            ids = [r['id'] for r in rows]
            if ids:
                placeholders = ','.join(['?'] * len(ids))
                conn.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", ids)
                conn.commit()
                print(f"    → [Memory] Đã xóa {len(ids)} tin nhắn cuối của session {session_id}")
        finally:
            conn.close()

        # Đồng bộ lại short-term cache sau khi xóa
        updated_history = self._get_recent_messages(session_id, limit=self.max_turns * 2)
        if self.use_redis:
            self.redis_client.set(f"session:{session_id}", json.dumps(updated_history), ex=86400)
        else:
            self.local_sessions[session_id] = updated_history

    # =====================================================================
    # TEMPORARY CHUNKS (Session-local RAM)
    # =====================================================================

    def set_temp_chunks(self, session_id: str, chunks: List[Dict]):
        """Lưu trữ chunks tạm thời vào RAM cho phiên chat hiện tại."""
        self.temp_chunks[session_id] = chunks
        print(f"    → [Memory] Đã lưu {len(chunks)} chunks tạm thời cho session {session_id}")

    def get_temp_chunks(self, session_id: str) -> List[Dict]:
        """Lấy chunks tạm thời của phiên chat."""
        return self.temp_chunks.get(session_id, [])

    def clear_temp_chunks(self, session_id: str):
        """Xóa chunks tạm thời."""
        if session_id in self.temp_chunks:
            del self.temp_chunks[session_id]
            print(f"    → [Memory] Đã xóa chunks tạm thời của session {session_id}")
