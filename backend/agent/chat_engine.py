from typing import List, Dict, Optional
from backend.agent.graph import app as legal_graph_app
from backend.agent.memory import ChatSessionManager
from backend.agent.query_router import RouteIntent

class RAGEngine:
    def __init__(self):
        self.memory = ChatSessionManager()
        self.graph = legal_graph_app

    async def chat(self, session_id: str, query: str, mode: str = "LEGAL_QA", file_path: str = None, top_k: int = 3, use_reflection: bool = True, use_rerank: bool = None) -> dict:
        """Thực thi luồng bằng LangGraph Orchestration (Async)."""
        from backend.config import settings
        import time
        import asyncio

        if use_rerank is None:
            use_rerank = settings.ENABLE_RERANK
        
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
            "history": history,
            "retry_count": 0,
            "references": [],
            "metrics": {}
        }

        # 2. Invoke Graph
        answer = "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý."
        references = []
        condensed_query = query
        detected_mode = mode
        is_cancelled = False
        
        try:
            # LangGraph ainvoke hỗ trợ ngắt kết nối (Cancellation)
            final_state = await self.graph.ainvoke(initial_state)
            answer = final_state.get("answer", answer)
            references = final_state.get("references", [])
            condensed_query = final_state.get("condensed_query", query)
            detected_mode = final_state.get("detected_mode", mode)
        except asyncio.CancelledError:
            print(f"  [!] LangGraph Execution CANCELLED BY USER.")
            is_cancelled = True
            raise # Re-raise để FastAPI handler biết là disconnected
        except Exception as e:
            print(f"  [!] LangGraph Execution Error: {e}")
            answer = f"Lỗi hệ thống LangGraph: {str(e)}"

        # 3. Store Memory (Chỉ lưu nếu KHÔNG bị hủy)
        if not is_cancelled:
            print(f"\n🧠 [Step 3/3] Committing to Memory...")
            self.memory.add_message(session_id, "user", query, mode=mode)
            self.memory.add_message(session_id, "assistant", answer, references=references, mode=mode)

            total = time.perf_counter() - t0
            print(f"📊 [SUMMARY] LangGraph request finalized. Total latency: {total:.1f}s")
            print("="*80 + "\n")

            session_info = self.memory.get_session(session_id)
            current_title = session_info.get("title", "Phiên chat mới") if session_info else "Phiên chat mới"

            return {
                "answer": answer,
                "mode": mode,
                "detected_mode": detected_mode,
                "session_id": session_id,
                "title": current_title,
                "standalone_query": condensed_query,
                "original_query": query,
                "references": references,
            }
        else:
            return {"status": "cancelled"}
    
rag_engine = RAGEngine()

