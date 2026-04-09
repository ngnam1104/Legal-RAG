from typing import Dict, Any
import time

from backend.agent.state import AgentState
from backend.agent.strategies.base import BaseRAGStrategy
from backend.agent.utils_legal_qa import (
    rewrite_legal_query,
    build_legal_context,
    grade_documents,
    chat_completion,
    strip_thinking_tags,
    ANSWER_PROMPT,
    reflect_on_answer,
    CORRECTION_PROMPT
)
from backend.config import settings
from backend.retrieval.hybrid_search import retriever

class LegalQAStrategy(BaseRAGStrategy):
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """Phân tích câu hỏi và sinh HyDE, trích xuất filter."""
        query = state.get("condensed_query", state["query"])
        rewrite_data = rewrite_legal_query(query, llm_preset=state.get("llm_preset"))
        
        hypothetical = rewrite_data.get("hypothetical_answer") or query
        return {
            "rewritten_queries": [hypothetical],
            "metadata_filters": rewrite_data.get("filters", {}),
            "pending_tasks": []
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Truy xuất Qdrant bằng Hybrid Search (với Appendix filtering)."""
        rewritten_queries = state.get("rewritten_queries") or [state.get("condensed_query") or state["query"]]
        kw = rewritten_queries[0] or state.get("condensed_query") or state["query"]
        filters = state.get("metadata_filters", {})
        legal_type = filters.get("legal_type")
        doc_number = filters.get("doc_number")
        is_appendix = filters.get("is_appendix")
        article_ref = filters.get("article_ref")
        
        # LLM trả về JSON boolean đôi khi bị ép về string
        if isinstance(is_appendix, str):
            is_appendix = "true" in is_appendix.lower()
            
        use_rerank = state.get("use_rerank", False)
        
        hits = retriever.search(
            query=kw,
            expand_context=True,
            max_neighbors=5,
            use_rerank=use_rerank,
            legal_type=legal_type,
            doc_number=doc_number,
            is_appendix=is_appendix,
            article_ref=article_ref,
            limit=15
        )
        
        if (legal_type or doc_number or is_appendix is not None or article_ref) and not hits:
            # Fallback nếu filter quá khắt khe
            hits = retriever.search(
                query=kw, 
                expand_context=True, 
                max_neighbors=5, 
                use_rerank=use_rerank,
                limit=15
            )
            
        return {"raw_hits": hits}

    def resolve_references(self, state: AgentState) -> Dict[str, Any]:
        """BƯỚC 2.5: Truy xuất đệ quy Điều/Khoản được dẫn chiếu."""
        from backend.agent.utils_legal_qa import resolve_recursive_references
        hits = state.get("raw_hits", [])
        
        # Recursive Reference Resolution
        final_hits_with_refs = resolve_recursive_references(hits)
        
        # State: Lưu riêng các hit phụ để grade
        recursive_only = [h for h in final_hits_with_refs if h not in hits]
        return {"recursive_hits": recursive_only}

    def grade(self, state: AgentState) -> Dict[str, Any]:
        """Kiểm định Hits so với câu hỏi gốc (Tiết kiệm token: Truncation Grading)."""
        hits = state.get("raw_hits", [])
        all_hits = hits + state.get("recursive_hits", [])
        
        file_chunks = state.get("file_chunks", [])
        query = state.get("condensed_query", state["query"])
        
        if file_chunks:
            from backend.agent.utils_conflict_analyzer import get_pruner
            import numpy as np
            model = get_pruner()
            if model:
                # Semantic sorting of file chunks against the query
                q_emb = model.encode(query, show_progress_bar=False)
                c_texts = [f.get("text_to_embed", f.get("unit_text", "")) for f in file_chunks]
                c_emb = model.encode(c_texts, show_progress_bar=False)
                
                sims = np.dot(c_emb, q_emb) / (np.linalg.norm(c_emb, axis=1) * np.linalg.norm(q_emb) + 1e-10)
                scored = list(zip(file_chunks, sims))
                scored.sort(key=lambda x: x[1], reverse=True)
                file_chunks = [item[0] for item in scored]
        
        if not all_hits and not file_chunks:
            return {
                "is_sufficient": False,
                "filtered_context": ""
            }
            
        # 1. Build FULL context for Generate step
        context_text = build_legal_context(all_hits, file_chunks=file_chunks)
        
        # 2. Build TRUNCATED context for Grade step (SAVE TOKENS: ~150 chars/chunk)
        truncated_parts = []
        if file_chunks:
            for idx, f_chunk in enumerate(file_chunks, start=1):
                f_text = f_chunk.get("text_to_embed", f_chunk.get("unit_text", ""))[:200]
                truncated_parts.append(f"[Tài liệu đính kèm {idx}]: {f_text}...")
                
        for h in all_hits:
            title = h.get("title", "")
            ref = h.get("article_ref", "")
            # Chỉ lấy 150 ký tự đầu tiên để LLM phán đoán độ liên quan
            text = h.get("text", "")[:150]
            truncated_parts.append(f"[{title}] {ref}: {text}...")
        
        truncated_context = "\n".join(truncated_parts)
        query = state.get("condensed_query", state["query"])
        
        # 3. Grade using truncated text
        is_relevant = grade_documents(query=query, context=truncated_context, llm_preset=state.get("llm_preset"))
        
        is_best_effort = False
        if not is_relevant and all_hits:
            # Fallback nếu GPT-grader bảo không khớp, nhưng ta vẫn có hits
            print("       ⚠️ Legal QA Gradder said 'No'. Switching to Best-Effort mode.")
            is_relevant = True
            is_best_effort = True
            
        return {
            "filtered_context": context_text, 
            "is_sufficient": is_relevant,
            "is_best_effort": is_best_effort
        }

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Sinh câu trả lời từ context."""
        context_text = state.get("filtered_context", "")
        query = state.get("condensed_query", state["query"])
        
        if not context_text:
            return {"draft_response": "Xin lỗi, tôi không tìm thấy quy định pháp luật nào liên quan đến câu hỏi của bạn."}
            
        # Thêm reference logic và đảm bảo được sort theo score (Rerank score)
        refs = []
        combined_hits = state.get("raw_hits", []) + state.get("recursive_hits", [])
        combined_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for h in combined_hits:
            refs.append({
                "title": h.get("title", ""),
                "article": h.get("article_ref", h.get("document_number", "")),
                "score": h.get("score", 0),
                "chunk_id": h.get("chunk_id", ""),
                "text_preview": h.get("text", "")[:200],
                "document_number": h.get("document_number", ""),
                "url": h.get("url", "")
            })
            
        history_msgs = state.get("history", [])[-6:] # Giữ lại 3 lượt QA gần nhất
        history_str = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_msgs]) if history_msgs else "(Không có lịch sử)"
            
        prompt = ANSWER_PROMPT.format(
            history=history_str,
            context=context_text, 
            query=query, 
            supplemental_context=state.get("supplemental_context", "")
        )
        answer = strip_thinking_tags(chat_completion(
            [{"role": "user", "content": prompt}], 
            temperature=0.1, 
            model=settings.LLM_CORE_MODEL, 
            llm_preset=state.get("llm_preset")
        ))
        
        if state.get("is_best_effort"):
            # Chèn cảnh báo Best-effort vào đầu câu trả lời
            disclaimer = "> [!NOTE]\n> **Thông báo:** Hệ thống không tìm thấy quy định trực tiếp, tuy nhiên dựa trên nội dung liên quan nhất tìm thấy, tôi xin cung cấp thông tin tham khảo như sau:\n\n"
            answer = disclaimer + answer
            
        return {
            "draft_response": answer,
            "references": refs
        }

    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """Kiểm tra ảo giác và trích dẫn."""
        if not state.get("use_reflection", True):
            return {
                "final_response": state.get("draft_response", ""),
                "pass_flag": True,
                "feedback": ""
            }
            
        query = state.get("condensed_query", state["query"])
        context_text = state.get("filtered_context", "")
        draft = state.get("draft_response", "")
        
        reflection = reflect_on_answer(query, context_text, draft, llm_preset=state.get("llm_preset"))
        is_pass = reflection.get("pass", True)
        feedback = reflection.get("feedback", "")
        
        if not is_pass:
            correction = CORRECTION_PROMPT.format(
                feedback=feedback, 
                context=context_text, 
                query=query,
                supplemental_context=state.get("supplemental_context", "")
            )
            answer = strip_thinking_tags(chat_completion(
                [{"role": "user", "content": correction}], 
                temperature=0.1, 
                model=settings.LLM_CORE_MODEL, 
                llm_preset=state.get("llm_preset")
            ))
            answer += "\n\n---\n*🔄 Câu trả lời đã được tự kiểm tra và cải thiện bởi Reflection Agent.*"
        else:
            answer = draft + "\n\n---\n*✅ Câu trả lời đã qua kiểm duyệt chất lượng (Reflection Agent).*"
            
        return {
            "final_response": answer,
            "pass_flag": is_pass,
            "feedback": feedback
        }
