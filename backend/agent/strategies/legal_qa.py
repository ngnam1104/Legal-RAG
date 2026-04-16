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
        
        filters = state.get("router_filters", {}) or {}
        filters.update(rewrite_data.get("filters", {}))
        
        return {
            "rewritten_queries": [hypothetical],
            "metadata_filters": filters,
            "pending_tasks": []
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Truy xuất Qdrant + Neo4j Graph Expansion (Bottom-Up + Lateral)."""
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
        
        # --- NEO4J GRAPH EXPANSION ---
        graph_context = {"lateral_docs": [], "document_toc": "", "sibling_texts": []}
        chunk_ids = [h.get("chunk_id", "") for h in hits if h.get("chunk_id")]
        
        if chunk_ids:
            try:
                from backend.retrieval.graph_db import bottom_up_expand, lateral_expand
                
                # YÊU CẦU 2: Bottom-Up — lấy TOC + sibling chunks
                bu_result = bottom_up_expand(chunk_ids)
                graph_context["document_toc"] = bu_result.get("document_toc", "")
                graph_context["sibling_texts"] = bu_result.get("sibling_texts", [])
                
                # YÊU CẦU 1: Lateral Expansion — tài liệu cùng ngành
                graph_context["lateral_docs"] = lateral_expand(chunk_ids)
                
                if graph_context["document_toc"]:
                    print(f"       📋 [Neo4j] Got document TOC ({len(graph_context['document_toc'])} chars)")
                if graph_context["lateral_docs"]:
                    print(f"       🔗 [Neo4j] Found {len(graph_context['lateral_docs'])} related docs via Lateral Expansion")
            except Exception as e:
                print(f"       ⚠️ [Neo4j] Graph expansion failed (non-fatal): {e}")
            
        return {"raw_hits": hits, "graph_context": graph_context}

    def grade(self, state: AgentState) -> Dict[str, Any]:
        """Kiểm định Hits so với câu hỏi gốc (Tiết kiệm token: Truncation Grading)."""
        hits = state.get("raw_hits", [])
        
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
        
        if not hits and not file_chunks:
            return {
                "is_sufficient": False,
                "filtered_context": ""
            }
            
        # 1. Build FULL context for Generate step (with graph_context)
        graph_ctx = state.get("graph_context", {})
        context_text = build_legal_context(hits, file_chunks=file_chunks, graph_context=graph_ctx)
        
        # 2. Build TRUNCATED context for Grade step (SAVE TOKENS: ~150 chars/chunk)
        truncated_parts = []
        if file_chunks:
            for idx, f_chunk in enumerate(file_chunks, start=1):
                f_text = f_chunk.get("text_to_embed", f_chunk.get("unit_text", ""))[:200]
                truncated_parts.append(f"[Tài liệu đính kèm {idx}]: {f_text}...")
                
        for h in hits:
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
        if not is_relevant and hits:
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
        """Sinh câu trả lời từ context + lateral docs."""
        context_text = state.get("filtered_context", "")
        query = state.get("condensed_query", state["query"])
        graph_ctx = state.get("graph_context", {})
        
        if not context_text:
            return {"draft_response": "Xin lỗi, tôi không tìm thấy quy định pháp luật nào liên quan đến câu hỏi của bạn."}
            
        # Thêm reference logic và đảm bảo được sort theo score (Rerank score)
        refs = []
        combined_hits = state.get("raw_hits", [])
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
        
        # Build supplemental context from lateral docs (YÊU CẦU 1)
        supplemental = state.get("supplemental_context", "")
        lateral_docs = graph_ctx.get("lateral_docs", [])
        if lateral_docs:
            lateral_lines = ["\n**📚 Tài liệu tham khảo thêm (cùng lĩnh vực):**"]
            for ld in lateral_docs:
                lateral_lines.append(f"- {ld.get('title', 'N/A')} ({ld.get('document_number', '')}) — Ngành: {ld.get('shared_sector', '')}")
            supplemental += "\n".join(lateral_lines)
            
        prompt = ANSWER_PROMPT.format(
            history=history_str,
            context=context_text, 
            query=query, 
            supplemental_context=supplemental
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
            
        # Append lateral docs to answer
        if lateral_docs:
            answer += "\n\n---\n**📚 Tài liệu tham khảo thêm (cùng lĩnh vực):**\n"
            for ld in lateral_docs:
                answer += f"- **{ld.get('title', '')}** ({ld.get('document_number', '')})\n"
            
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
