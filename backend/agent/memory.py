import json
import os
import uuid
import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import redis
import psycopg2
from psycopg2 import pool as pg_pool
from backend.models.llm_factory import chat_completion
from backend.prompt import TITLE_PROMPT, GENERAL_TITLE_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory")

_pg_pool = None
try:
    pg_url = os.environ.get("POSTGRES_URL", "postgresql://legal_rag:legal_rag_secret@localhost:5433/legal_chat_history")
    _pg_pool = pg_pool.SimpleConnectionPool(1, 5, pg_url)
    _test_conn = _pg_pool.getconn()
    _pg_pool.putconn(_test_conn)
    logger.info("✅ PostgreSQL connected for long-term memory.")
except Exception as e:
    logger.error(f"❌ Thuần Docker yêu cầu PostgreSQL. Lỗi kết nối: {e}")
    raise e

class ChatSessionManager:
    """
    Quản lý Memory 2 tầng:
    - Redis: Short-term rolling window (7 turns gần nhất) cho LLM context, TTL 24h
    - PostgreSQL: Long-term persistent storage (toàn bộ lịch sử, vĩnh viễn)
    """

    def __init__(self, max_turns: int = 7):
        self.max_turns = max_turns
        self.use_redis = False
        self.local_sessions = {}
        self.temp_chunks = {} # {(session_id, doc_id): {"chunks": [chunks], "expires_at": timestamp}}

        # Redis (short-term)
        try:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ Redis connected for memory.")
        except Exception:
            logger.warning("⚠️ Redis không khả dụng, dùng RAM cục bộ cho short-term memory.")

        # Long-term DB: PostgreSQL
        self._init_postgres()

    def _init_postgres(self):
        """Khởi tạo schema PostgreSQL."""
        conn = _pg_pool.getconn()
        try:
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
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    references_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_state (
                    session_id TEXT PRIMARY KEY,
                    current_document TEXT,
                    entities_json TEXT,
                    last_intent TEXT,
                    last_rewritten_query TEXT,
                    updated_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
            conn.commit()
            cursor.close()
        finally:
            _pg_pool.putconn(conn)

    # =====================================================================
    # SESSION MANAGEMENT
    # =====================================================================

    def create_session(self, title: str = None, session_id: str = None) -> str:
        """Tạo phiên chat mới, trả về session_id."""
        sid = session_id or str(uuid.uuid4())
        now = datetime.now().isoformat()
        title = title or "Phiên chat mới"

        conn = _pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (sid, title, now, now)
            )
            conn.commit()
            cursor.close()
        finally:
            _pg_pool.putconn(conn)
        return sid

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """Lấy danh sách phiên chat có ít nhất 1 trao đổi."""
        conn = _pg_pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
                SELECT s.id, s.title, s.created_at, s.updated_at 
                FROM sessions s
                WHERE (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) >= 2
                ORDER BY s.updated_at DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            return [dict(r) for r in rows]
        finally:
            _pg_pool.putconn(conn)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Lấy thông tin 1 phiên."""
        conn = _pg_pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
            row = cursor.fetchone()
            cursor.close()
            return dict(row) if row else None
        finally:
            _pg_pool.putconn(conn)

    def update_session_title(self, session_id: str, title: str):
        conn = _pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE sessions SET title = %s, updated_at = %s WHERE id = %s",
                           (title, datetime.now().isoformat(), session_id))
            conn.commit()
            cursor.close()
        finally:
            _pg_pool.putconn(conn)

    def delete_session(self, session_id: str):
        conn = _pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
            conn.commit()
            cursor.close()
        finally:
            _pg_pool.putconn(conn)

        if self.use_redis:
            self.redis_client.delete(f"session:{session_id}")
        if session_id in self.local_sessions:
            del self.local_sessions[session_id]
        to_delete = [k for k in self.temp_chunks.keys() if k[0] == session_id]
        for k in to_delete:
            del self.temp_chunks[k]
        if self.use_redis:
            keys = self.redis_client.keys(f"temp_chunks:{session_id}:*")
            if keys:
                self.redis_client.delete(*keys)

    # =====================================================================
    # MESSAGE MANAGEMENT
    # =====================================================================

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        if self.use_redis:
            val = self.redis_client.get(f"session:{session_id}")
            if val:
                return json.loads(val)
        elif session_id in self.local_sessions:
            return self.local_sessions[session_id]
        return self._get_recent_messages(session_id, limit=self.max_turns * 2)

    def get_full_history(self, session_id: str) -> List[Dict]:
        conn = _pg_pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT role, content, references_json, created_at FROM messages WHERE session_id = %s ORDER BY id ASC",
                (session_id,)
            )
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for r in rows:
                msg = {"role": r["role"], "content": r["content"], "created_at": r["created_at"]}
                try:
                    msg["references"] = json.loads(r["references_json"])
                except:
                    msg["references"] = []
                result.append(msg)
            return result
        finally:
            _pg_pool.putconn(conn)

    def _get_recent_messages(self, session_id: str, limit: int = 14) -> List[Dict[str, str]]:
        conn = _pg_pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT role, content FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            cursor.close()
            return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
        finally:
            _pg_pool.putconn(conn)

    def add_message(self, session_id: str, role: str, content: str, references: List[Dict] = None, mode: str = "GENERAL_CHAT"):
        references = references or []
        now = datetime.now().isoformat()
        self.create_session(session_id=session_id)

        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        if self.use_redis:
            self.redis_client.set(f"session:{session_id}", json.dumps(history), ex=86400)
        else:
            self.local_sessions[session_id] = history

        conn = _pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, references_json, created_at) VALUES (%s, %s, %s, %s, %s)",
                (session_id, role, content, json.dumps(references, ensure_ascii=False), now)
            )
            cursor.execute("UPDATE sessions SET updated_at = %s WHERE id = %s", (now, session_id))

            if role == "user":
                cursor.execute("SELECT title FROM sessions WHERE id = %s", (session_id,))
                session_info = cursor.fetchone()
                current_title = session_info[0] if session_info else "Phiên chat mới"

                cursor.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = %s AND role = 'user'",
                    (session_id,)
                )
                msg_count = cursor.fetchone()[0]

                if msg_count == 1 and current_title == "Phiên chat mới":
                    try:
                        current_prompt = TITLE_PROMPT if mode != "GENERAL_CHAT" else GENERAL_TITLE_PROMPT
                        title_resp = chat_completion([{"role": "user", "content": current_prompt.format(query=content)}], temperature=0.1)
                        title = title_resp.strip().replace('"', '').replace("'", "")
                        if not title or len(title) > 100:
                            title = content[:40] + ("..." if len(content) > 40 else "")
                    except Exception as e:
                        logger.error(f"      ⚠️ Gợi ý tiêu đề thất bại: {e}. Dùng fallback.")
                        title = content[:40] + ("..." if len(content) > 40 else "")
                    
                    cursor.execute("UPDATE sessions SET title = %s WHERE id = %s", (title, session_id))

            conn.commit()
            cursor.close()
        finally:
            _pg_pool.putconn(conn)

    def get_message_count(self, session_id: str) -> int:
        conn = _pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = %s", (session_id,))
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row else 0
        finally:
            _pg_pool.putconn(conn)

    def delete_last_turn(self, session_id: str):
        conn = _pg_pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT id FROM messages WHERE session_id = %s ORDER BY id DESC LIMIT 2",
                (session_id,)
            )
            rows = cursor.fetchall()
            ids = [r['id'] for r in rows]
            if ids:
                placeholders = ','.join(['%s'] * len(ids))
                cursor.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", ids)
                conn.commit()
                logger.debug(f"    → [Memory] Đã xóa {len(ids)} tin nhắn cuối của session {session_id}")
            cursor.close()
        finally:
            _pg_pool.putconn(conn)

        updated_history = self._get_recent_messages(session_id, limit=self.max_turns * 2)
        if self.use_redis:
            self.redis_client.set(f"session:{session_id}", json.dumps(updated_history), ex=86400)
        else:
            self.local_sessions[session_id] = updated_history

    # =====================================================================
    # CONVERSATION STATE
    # =====================================================================

    def get_state(self, session_id: str) -> Dict[str, Any]:
        conn = _pg_pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM conversation_state WHERE session_id = %s", (session_id,))
            row = cursor.fetchone()
            cursor.close()
            if row:
                state = dict(row)
                try:
                    state["entities"] = json.loads(state["entities_json"])
                except:
                    state["entities"] = []
                return state
            else:
                return {
                    "session_id": session_id,
                    "current_document": None,
                    "entities": [],
                    "last_intent": None,
                    "last_rewritten_query": None
                }
        finally:
            _pg_pool.putconn(conn)

    def update_state(self, session_id: str, state_dict: Dict[str, Any]):
        self.create_session(session_id=session_id)
        now = datetime.now().isoformat()
        entities_json = json.dumps(state_dict.get("entities", []), ensure_ascii=False)
        
        conn = _pg_pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversation_state (
                    session_id, current_document, entities_json, last_intent, last_rewritten_query, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(session_id) DO UPDATE SET
                    current_document = excluded.current_document,
                    entities_json = excluded.entities_json,
                    last_intent = excluded.last_intent,
                    last_rewritten_query = excluded.last_rewritten_query,
                    updated_at = excluded.updated_at
            """, (
                session_id, 
                state_dict.get("current_document"),
                entities_json,
                state_dict.get("last_intent"),
                state_dict.get("last_rewritten_query"),
                now
            ))
            conn.commit()
            cursor.close()
        finally:
            _pg_pool.putconn(conn)

    # =====================================================================
    # TEMPORARY CHUNKS (Session-local RAM with Scoping and TTL - Step 4)
    # =====================================================================

    def set_temp_chunks(self, session_id: str, chunks: List[Dict], document_id: str = "default"):
        """Lưu trữ chunks tạm thời. Scoped by session + document, TTL 5m."""
        # Clean old chunks for this session if document changed and it's not "default"
        if document_id != "default":
            self.clear_temp_chunks(session_id, exclude_doc_id=document_id)

        if self.use_redis:
            key = f"temp_chunks:{session_id}:{document_id}"
            self.redis_client.set(key, json.dumps(chunks, ensure_ascii=False), ex=300) # 5 minutes
        else:
            self.temp_chunks[(session_id, document_id)] = {
                "chunks": chunks,
                "expires_at": time.time() + 300
            }
        logger.info(f"    → [Memory] Đã lưu {len(chunks)} chunks cho session {session_id}, doc {document_id}")

    def get_temp_chunks(self, session_id: str, document_id: str = "default") -> List[Dict]:
        """Lấy chunks tạm thời. Kiểm tra TTL."""
        if self.use_redis:
            key = f"temp_chunks:{session_id}:{document_id}"
            val = self.redis_client.get(key)
            return json.loads(val) if val else []
        
        entry = self.temp_chunks.get((session_id, document_id))
        if entry:
            if time.time() < entry["expires_at"]:
                return entry["chunks"]
            else:
                del self.temp_chunks[(session_id, document_id)] # Expired
        return []

    def clear_temp_chunks(self, session_id: str, document_id: str = None, exclude_doc_id: str = None):
        """Xóa chunks tạm thời. Có thể xóa cụ thể 1 doc hoặc toàn bộ session."""
        if self.use_redis:
            if document_id:
                self.redis_client.delete(f"temp_chunks:{session_id}:{document_id}")
            else:
                keys = self.redis_client.keys(f"temp_chunks:{session_id}:*")
                if exclude_doc_id:
                    keys = [k for k in keys if not k.endswith(f":{exclude_doc_id}")]
                if keys:
                    self.redis_client.delete(*keys)
        else:
            if document_id:
                self.temp_chunks.pop((session_id, document_id), None)
            else:
                to_del = [k for k in self.temp_chunks.keys() if k[0] == session_id]
                if exclude_doc_id:
                    to_del = [k for k in to_del if k[1] != exclude_doc_id]
                for k in to_del:
                    del self.temp_chunks[k]
        
        logger.debug(f"    → [Memory] Đã dọn dẹp chunks của session {session_id}")
