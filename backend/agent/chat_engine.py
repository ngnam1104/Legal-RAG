from typing import List, Dict, Optional
from backend.agent.graph import app as legal_graph_app
from backend.agent.memory import ChatSessionManager
from backend.agent.query_router import RouteIntent

class RAGEngine:
    def __init__(self):
        self.memory = ChatSessionManager()
        self.graph = legal_graph_app
        # Mapping Node names to user-friendly messages
        self.node_messages = {
            "preprocess": "📂 Đang xử lý tài liệu...",
            "condense": "🧠 Đang phân tích ngữ cảnh câu hỏi...",
            "understand": "🔍 Đang lập kế hoạch truy vấn...",
            "retrieve": "📚 Đang tìm kiếm cơ sở dữ liệu pháp luật...",
            "resolve_references": "🔗 Đang giải quyết các dẫn chiếu...",
            "grade": "⚖️ Đang đánh giá độ phù hợp của dữ liệu...",
            "generate": "✍️ Đang tổng hợp câu trả lời...",
            "reflect": "🛡️ Đang kiểm tra tính chính xác của phản hồi...",
            "general": "💬 Đang trả lời câu hỏi thông thường..."
        }

    async def chat(self, session_id: str, query: str, mode: str = "LEGAL_QA", file_path: str = None, llm_preset: str = "groq_8b", top_k: int = 3, use_reflection: bool = True, use_rerank: bool = True):
        """Streaming flow bằng LangGraph Orchestration."""
        from backend.config import settings
        import time
        import asyncio
        
        t0 = time.perf_counter()
        print(f"\n" + "="*80)
        print(f"🚀 [LANGGRAPH ENGINE] CHAT REQUEST RECEIVED (ASYNC)")
        print(f"   - Session: {session_id} | Mode: {mode}")
        print(f"   - Input: \"{query}\"")
        print("="*80)

        # 1. Prepare Initial State
        history = self.memory.get_history(session_id)
        initial_state = {
            "query": query,
            "session_id": session_id,
            "mode": mode,
            "file_path": file_path,
            "top_k": top_k,
            "use_reflection": use_reflection,
            "use_rerank": use_rerank,
            "llm_preset": llm_preset,
            "history": history,
            "retry_count": 0,
            "references": [],
            "metrics": {}
        }

        # 2. Invoke Graph (Streaming Events)
        final_state = {}
        last_node = ""
        
        try:
            # LangGraph astream_events hỗ trợ bóc tách từng chặng xử lý
            async for event in self.graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event["name"]
                
                # Check for disconnect regularly if integrated with FastAPI, 
                # but here we just yield events for the caller to handle.
                
                if kind == "on_node_start":
                    msg = self.node_messages.get(name)
                    if msg:
                        yield {"type": "step", "content": msg}
                
                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event["data"]["output"]

            if not final_state:
                 yield {"type": "error", "content": "Hệ thống không nhận được trạng thái cuối cùng."}
                 return

            answer = final_state.get("answer", "Xin lỗi, đã có lỗi xảy ra.")
            references = final_state.get("references", [])
            condensed_query = final_state.get("condensed_query", query)
            detected_mode = final_state.get("detected_mode", mode)

            # 3. Store Memory
            print(f"\n🧠 [Memory] Committing to SQLite...")
            self.memory.add_message(session_id, "user", query, mode=mode)
            self.memory.add_message(session_id, "assistant", answer, references=references, mode=mode)

            session_info = self.memory.get_session(session_id)
            current_title = session_info.get("title", "Phiên chat mới") if session_info else "Phiên chat mới"

            yield {
                "type": "final",
                "content": {
                    "answer": answer,
                    "mode": mode,
                    "detected_mode": detected_mode,
                    "session_id": session_id,
                    "title": current_title,
                    "standalone_query": condensed_query,
                    "original_query": query,
                    "references": references,
                }
            }

        except asyncio.CancelledError:
            print(f"  [!] LangGraph Execution CANCELLED BY USER.")
            yield {"type": "cancelled", "content": "Đã dừng phản hồi."}
        except Exception as e:
            print(f"  [!] LangGraph Streaming Error: {e}")
            yield {"type": "error", "content": f"Lỗi hệ thống: {str(e)}"}
    
rag_engine = RAGEngine()

