from typing import List, Dict, Optional, Any
import logging
import time
import asyncio
from backend.agent.graph import app as legal_graph_app
from backend.agent.memory import ChatSessionManager
from backend.agent.query_router import RouteIntent
from backend.agent.utils.utils_conversation import extract_entities

logger = logging.getLogger("chat_engine")

class RAGEngine:
    def __init__(self):
        self.memory = ChatSessionManager()
        self.graph = legal_graph_app
        # Mapping Node names to user-friendly messages (Synced with graph.py)
        self.node_messages = {
            "preprocess": "📂 Đang xử lý tài liệu...",
            "router_preprocess": "📂 Đang xử lý tài liệu...",
            "condense": "🧠 Đang phân tích ngữ cảnh câu hỏi...",
            "detect_only": "🎯 Đang xác định chủ đề...",
            "understand": "🔍 Đang lập kế hoạch truy vấn...",
            "retrieve": "📚 Đang tìm kiếm cơ sở dữ liệu pháp luật...",
            "grade": "⚖️ Đang đánh giá độ phù hợp của dữ liệu...",
            "bump_retry": "🔄 Đang thử lại với truy vấn bổ sung...",
            "generate": "✍️ Đang tổng hợp câu trả lời...",
            "reflect": "🛡️ Đang kiểm tra tính chính xác của phản hồi...",
            "general": "💬 Đang trả lời câu hỏi thông thường...",
            "reset_for_batch": "📦 Đang xử lý tập dữ liệu tiếp theo..."
        }

    async def chat(self, session_id: str, query: str, mode: str = "LEGAL_QA", file_path: str = None, llm_preset: str = "groq_8b", top_k: int = 5, use_reflection: bool = None, use_grading: bool = None, use_rerank: bool = None):
        """Streaming flow bằng LangGraph Orchestration."""
        from backend.config import settings
        import time
        import asyncio
        
        # Merge input params with global ablation config
        # Nếu truyền None (từ API) -> ưu tiên dùng config "mã nguồn" như user yêu cầu
        final_use_reflection = use_reflection if use_reflection is not None else settings.ENABLE_REFLECTION
        final_use_rerank = use_rerank if use_rerank is not None else settings.ENABLE_RERANK
        final_use_grading = use_grading if use_grading is not None else settings.ENABLE_GRADING

        t0 = time.perf_counter()
        logger.info("\n" + "="*80)
        logger.info("🚀 [LANGGRAPH ENGINE] CHAT REQUEST RECEIVED")
        logger.info(f"   - Session: {session_id} | Mode: {mode}")
        logger.info(f"   - Input Query: \"{query}\"")
        logger.info("="*80)

        # 0. Load Conversation State (New)
        conv_state = self.memory.get_state(session_id)
        logger.info(f"   - Current State: doc='{conv_state.get('current_document')}', entities={conv_state.get('entities')}")

        # 1. Prepare Initial State
        history = self.memory.get_history(session_id)
        initial_state = {
            "query": query,
            "session_id": session_id,
            "mode": mode,
            "file_path": file_path,
            "top_k": top_k,
            "use_grading": final_use_grading,
            "use_reflection": final_use_reflection,
            "use_rerank": final_use_rerank,
            "llm_preset": llm_preset,
            "history": history,
            "conversation_state": conv_state, # Inject state
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
                
                elif kind == "on_chain_end":
                    output_data = event.get("data", {}).get("output")
                    # Chụp lấy state cuối cùng (chứa answer) thay vì dựa vào tên node cứng 'LangGraph'
                    if isinstance(output_data, dict) and "answer" in output_data:
                        final_state = output_data

            if not final_state:
                 yield {"type": "error", "content": "Hệ thống không nhận được trạng thái cuối cùng."}
                 return

            answer = final_state.get("answer", "Xin lỗi, đã có lỗi xảy ra.")
            references = final_state.get("references", [])
            # standalone_query: câu hỏi viết lại thuần, không có HyDE — dùng cho entity extraction
            standalone_query = final_state.get("standalone_query") or final_state.get("condensed_query", query)
            condensed_query = final_state.get("condensed_query", query)  # HyDE query, chỉ dùng cho retrieval
            detected_mode = final_state.get("detected_mode", mode)
            
            related_docs = []
            if "graph_context" in final_state and isinstance(final_state["graph_context"], dict):
                related_docs = final_state["graph_context"].get("lateral_docs", [])

            # 3. Post-Turn Processing (State Update & Entity Extraction)
            logger.info(f"🧠 [Engine] Post-turn processing...")
            
            # Extract new state entities from standalone_query (clean, no HyDE)
            new_info = extract_entities(standalone_query, answer)
            
            # Update state dict
            updated_conv_state = {
                "session_id": session_id,
                "current_document": new_info.get("current_document") or conv_state.get("current_document"),
                "entities": list(set(conv_state.get("entities", []) + new_info.get("entities", []))),
                "last_intent": detected_mode,
                "last_rewritten_query": standalone_query  # Lưu câu hỏi thuần, không có HyDE
            }
            self.memory.update_state(session_id, updated_conv_state)
            
            # 4. Store Memory
            logger.info(f"💾 [Memory] Committing messages to SQLite...")
            self.memory.add_message(session_id, "user", query, mode=mode)
            self.memory.add_message(session_id, "assistant", answer, references=references, mode=mode)

            session_info = self.memory.get_session(session_id)
            current_title = session_info.get("title", "Phiên chat mới") if session_info else "Phiên chat mới"
            
            logger.info(f"✅ Turn finished in {time.perf_counter() - t0:.2f}s")

            yield {
                "type": "final",
                "content": {
                    "answer": answer,
                    "mode": mode,
                    "detected_mode": detected_mode,
                    "session_id": session_id,
                    "title": current_title,
                    "standalone_query": standalone_query,  # Câu hỏi viết lại thuần (FE hiển thị)
                    "original_query": query,
                    "references": references,
                    "related_docs": related_docs,
                    "metrics": final_state.get("metrics", {})
                }
            }

        except asyncio.CancelledError:
            logger.warning(f"  [!] LangGraph Execution CANCELLED BY USER.")
            yield {"type": "cancelled", "content": "Đã dừng phản hồi."}
        except Exception as e:
            logger.error(f"  [!] LangGraph Streaming Error: {e}", exc_info=True)
            yield {"type": "error", "content": f"Lỗi hệ thống: {str(e)}"}
    
rag_engine = RAGEngine()

