from typing import Dict, Any
import time

from backend.agent.state import AgentState
from backend.agent.strategies.base import BaseRAGStrategy
from backend.agent.utils_sector_search import (
    transform_sector_query,
    deduplicate_by_document,
    _heuristic_date_filter,
    grade_relevance_batch,
    map_reduce_aggregate,
    generate_executive_summary,
    check_coverage_bias,
    supplemental_search_by_basis
)
from backend.retrieval.hybrid_search import retriever


class SectorSearchStrategy(BaseRAGStrategy):
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """Sector Query Planner: Trích xuất keywords, legal_sectors, date range, filters."""
        query = state.get("condensed_query", state["query"])
        tf = transform_sector_query(query, llm_preset=state.get("llm_preset"))
        
        # --- NẮM BẮT TÀI LIỆU (Nếu có) ---
        file_chunks = state.get("file_chunks", [])
        file_analysis = {}
        if file_chunks:
            # Optimize: Rerank file chunks based on query before analysis
            from backend.agent.utils_conflict_analyzer import get_pruner
            import numpy as np
            model = get_pruner()
            if model and query:
                q_emb = model.encode(query, show_progress_bar=False)
                c_texts = [f.get("text_to_embed", f.get("unit_text", "")) for f in file_chunks]
                c_emb = model.encode(c_texts, show_progress_bar=False)
                sims = np.dot(c_emb, q_emb) / (np.linalg.norm(c_emb, axis=1) * np.linalg.norm(q_emb) + 1e-10)
                scored = list(zip(file_chunks, sims))
                scored.sort(key=lambda x: x[1], reverse=True)
                file_chunks = [item[0] for item in scored]

            from backend.agent.utils_sector_search import analyze_document_focus
            file_analysis = analyze_document_focus(file_chunks, llm_preset=state.get("llm_preset"))
            print(f"       🧠 Document Analysis Focus: {file_analysis.get('focus_summary')}")
            
            # Gộp keywords từ file vào query search
            if file_analysis.get("suggested_keywords"):
                tf["keywords"] = f"{tf.get('keywords', query)} {file_analysis['suggested_keywords']}"
        
        # Merge filters + extra extraction
        filters = state.get("router_filters", {}) or {}
        filters.update(tf.get("filters", {}))
        filters["legal_sectors"] = list(set(tf.get("legal_sectors", []) + file_analysis.get("legal_sectors", [])))
        filters["effective_date_range"] = tf.get("effective_date_range", {})
        
        return {
            "rewritten_queries": [tf.get("keywords") or query],
            "metadata_filters": filters,
            "file_analysis": file_analysis.get("focus_summary", ""),
            "pending_tasks": []
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Broad Fetch: Qdrant Top 15 + Neo4j Sector MapReduce."""
        rewritten_queries = state.get("rewritten_queries") or [state.get("condensed_query") or state["query"]]
        kw = rewritten_queries[0] or state.get("condensed_query") or state["query"]
        filters = state.get("metadata_filters", {})
        
        use_rerank = state.get("use_rerank", False)
        
        hits = retriever.search(
            query=kw,
            expand_context=False,  # Sector search chỉ cần metadata
            use_rerank=use_rerank,
            legal_type=filters.get("legal_type"),
            doc_number=filters.get("doc_number"),
            limit=15
        )
        
        # --- YÊU CẦU 3: NEO4J SECTOR MAPREDUCE ---
        graph_context = {"sector_mapreduce": [], "lateral_docs": []}
        sectors = filters.get("legal_sectors", [])
        
        if sectors:
            try:
                from backend.retrieval.graph_db import sector_mapreduce
                for sector_name in sectors[:3]:  # Giới hạn 3 ngành
                    mr_result = sector_mapreduce(sector_name)
                    if mr_result:
                        graph_context["sector_mapreduce"].extend(mr_result)
                        print(f"       📊 [Neo4j] Sector MapReduce for '{sector_name}': {len(mr_result)} groups")
            except Exception as e:
                print(f"       ⚠️ [Neo4j] Sector MapReduce failed (non-fatal): {e}")
        
        return {"raw_hits": hits, "graph_context": graph_context}



    def grade(self, state: AgentState) -> Dict[str, Any]:
        """
        Strict Filter & Deduplication + MapReduce Aggregation.
        
        Pipeline:
        1. Deduplicate by document_number
        2. Heuristic date filter (pure Python)
        3. LLM relevance batch filter (1 call)
        4. MapReduce aggregate → Markdown table (pure Python)
        """
        query = state.get("condensed_query", state["query"])
        hits = state.get("raw_hits", [])
        filters = state.get("metadata_filters", {})
        
        if not hits:
            return {
                "raw_hits": [],
                "filtered_context": "",
                "is_sufficient": False
            }
        
        # Step 1: Dedup by document_number (giữ chunk score cao nhất)
        unique_docs = deduplicate_by_document(hits)
        print(f"       📋 Dedup: {len(hits)} chunks → {len(unique_docs)} unique docs")
        
        # Step 2: Heuristic date filter (không tốn token)
        date_range = filters.get("effective_date_range", {})
        date_filtered = _heuristic_date_filter(unique_docs, date_range)
        if len(date_filtered) < len(unique_docs):
            print(f"       📅 Date filter: {len(unique_docs)} → {len(date_filtered)} docs")
        
        # Step 3: LLM relevance batch (1 call nhẹ)
        relevant_docs = grade_relevance_batch(query, date_filtered, llm_preset=state.get("llm_preset"))
        print(f"       ✅ Relevance filter: {len(date_filtered)} → {len(relevant_docs)} docs")
        
        is_best_effort = False
        if not relevant_docs and date_filtered:
            # --- BEST-EFFORT FALLBACK ---
            # Nếu lọc sạch không còn gì, lấy Top 3 văn bản có điểm cao nhất làm tham khảo
            print("       ⚠️ No relevant docs found by LLM. Using Best-Effort (Top 3 closest matches).")
            relevant_docs = date_filtered[:3]
            is_best_effort = True
            
        # Step 4: MapReduce Aggregation (pure Python)
        table_markdown = map_reduce_aggregate(relevant_docs)
        
        return {
            "raw_hits": relevant_docs, 
            "filtered_context": table_markdown,
            "is_sufficient": len(relevant_docs) > 0,
            "is_best_effort": is_best_effort
        }

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Executive Summary + Markdown Table Report.
        Chỉ dùng 1 LLM call (~100 tokens output) để sinh summary.
        """
        query = state.get("condensed_query", state["query"])
        table_markdown = state.get("filtered_context", "")
        
        file_chunks = state.get("file_chunks", [])
        
        if not table_markdown and not file_chunks:
            return {"draft_response": "Không tìm thấy văn bản pháp luật nào phù hợp với truy vấn."}
        
        history_msgs = state.get("history", [])[-6:]
        history_str = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_msgs]) if history_msgs else "(Không có lịch sử)"
        
        # Generate Executive Summary (1 LLM call)
        summary = generate_executive_summary(
            query, 
            table_markdown, 
            file_chunks, 
            file_analysis=state.get("file_analysis", ""),
            history_str=history_str,
            llm_preset=state.get("llm_preset")
        )
        
        if state.get("is_best_effort"):
            # Chèn cảnh báo Best-effort vào đầu báo cáo
            best_effort_msg = "> [!NOTE]\n> **Thông báo:** Hệ thống không tìm thấy quy định trực tiếp, tuy nhiên dựa trên nội dung liên quan nhất tìm thấy, tôi xin cung cấp thông tin tham khảo như sau:\n\n"
            summary = best_effort_msg + summary
            
        # Assemble final report: Summary + Table
        report = ""
        if file_chunks:
            file_insight = state.get("file_analysis", "Tài liệu tải lên chứa các nội dung liên quan đến lĩnh vực này.")
            report += f"### 💡 Phân tích từ Tài liệu tải lên\n> {file_insight}\n\n---\n\n"
            
        report += f"### 📊 Tổng quan\n\n{summary}\n\n---\n\n### 📚 Danh sách văn bản pháp luật\n{table_markdown}"
        
        # Build references (đảm bảo được sort theo score)
        refs = []
        sorted_hits = sorted(state.get("raw_hits", []), key=lambda x: x.get("score", 0), reverse=True)
        for h in sorted_hits:
            refs.append({
                "title": h.get("title", ""),
                "article": h.get("article_ref", h.get("document_number", "")),
                "score": h.get("score", 0),
                "chunk_id": h.get("chunk_id", ""),
                "document_number": h.get("document_number", ""),
                "url": h.get("url", "")
            })
            
        return {
            "draft_response": report,
            "references": refs
        }

    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """
        Coverage Check: Phát hiện thiên lệch và bổ sung kết quả.
        
        Nếu danh sách chỉ toàn 1 loại (VD: chỉ Luật), trigger supplemental search
        qua legal_basis_refs để tìm Nghị định/Thông tư hướng dẫn.
        Chỉ retry tối đa 1 lần.
        """
        hits = state.get("raw_hits", [])
        retry_count = state.get("retry_count", 0)
        
        # Skip coverage check nếu đã retry hoặc ít kết quả
        if retry_count >= 3 or len(hits) < 3:
            return {
                "final_response": state.get("draft_response", ""),
                "pass_flag": True,
                "feedback": ""
            }
        
        # Coverage Check
        coverage = check_coverage_bias(hits)
        
        if coverage["biased"] and coverage["basis_doc_numbers"]:
            print(f"       🔍 Coverage bias detected: {coverage['dominant_type']} dominates ({coverage['type_distribution']})")
            print(f"       🔍 Missing types: {coverage['missing_types']}")
            print(f"       🔄 Triggering supplemental search for {len(coverage['basis_doc_numbers'])} basis docs...")
            
            # Supplemental search
            existing_doc_nums = {h.get("document_number") for h in hits}
            supplemental = supplemental_search_by_basis(
                coverage["basis_doc_numbers"], existing_doc_nums
            )
            
            if supplemental:
                print(f"       ✅ Found {len(supplemental)} supplemental docs")
                
                # Rebuild report with supplemental results
                all_hits = hits + supplemental
                query = state.get("condensed_query", state["query"])
                new_table = map_reduce_aggregate(all_hits)
                new_summary = generate_executive_summary(query, new_table, llm_preset=state.get("llm_preset"))
                
                report = f"### 📊 Tổng quan\n\n{new_summary}\n\n---\n\n### 📚 Danh sách văn bản pháp luật\n{new_table}"
                report += "\n\n> *🔄 Danh sách đã được bổ sung tự động các văn bản hướng dẫn thi hành (Coverage Check).*"
                
                # Build updated refs (đảm bảo được sort theo score)
                refs = []
                all_hits_sorted = sorted(all_hits, key=lambda x: x.get("score", 0), reverse=True)
                for h in all_hits_sorted:
                    refs.append({
                        "title": h.get("title", ""),
                        "article": h.get("article_ref", h.get("document_number", "")),
                        "score": h.get("score", 0),
                        "chunk_id": h.get("chunk_id", ""),
                        "document_number": h.get("document_number", ""),
                        "url": h.get("url", "")
                    })
                
                return {
                    "final_response": report,
                    "references": refs,
                    "raw_hits": all_hits,
                    "pass_flag": True,
                    "feedback": f"Coverage Check: Bổ sung {len(supplemental)} văn bản hướng dẫn."
                }
            else:
                print(f"       ℹ️ No supplemental docs found, keeping original results")
        
        # No bias or no supplemental found — pass through
        return {
            "final_response": state.get("draft_response", ""),
            "pass_flag": True,
            "feedback": ""
        }
