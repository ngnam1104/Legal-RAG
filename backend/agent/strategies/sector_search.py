from typing import Dict, Any
import time

from backend.agent.state import AgentState
from backend.agent.strategies.base import BaseRAGStrategy
from backend.agent.utils.utils_sector_search import (
    deduplicate_by_document,
    group_by_document,
    _heuristic_date_filter,
    map_reduce_aggregate,
    generate_executive_summary
)
from backend.agent.utils.utils_legal_qa import filter_cited_references
from backend.config import settings
from backend.retrieval.hybrid_search import retriever


class SectorSearchStrategy(BaseRAGStrategy):
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """Sector Query Planner: Sử dụng tham số từ SuperRouter."""
        query = state.get("condensed_query") or state["query"]
        filters = state.get("router_filters", {}) or {}
        
        file_chunks = state.get("file_chunks", [])
        file_analysis = {}
        if file_chunks:
            from backend.agent.utils.utils_conflict_analyzer import get_pruner
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

            from backend.agent.utils.utils_sector_search import analyze_document_focus
            file_analysis = analyze_document_focus(file_chunks, llm_preset=state.get("llm_preset"))
            
            if file_analysis.get("suggested_keywords"):
                query = f"{query} {file_analysis['suggested_keywords']}"
        
        # Ensure sector and legal_sectors are lists before concatenation to avoid TypeError
        sector_val = filters.get("sector", [])
        sector_list = [sector_val] if isinstance(sector_val, str) else list(sector_val) if sector_val else []
        
        fa_val = file_analysis.get("legal_sectors", [])
        fa_list = [fa_val] if isinstance(fa_val, str) else list(fa_val) if fa_val else []
        
        filters["legal_sectors"] = list(set(sector_list + fa_list))
        # Note: SuperRouter doesn't extract effective_date_range explicitly like transform_sector_query did
        # but sector_search can still proceed without it.
        
        
        is_appendix = filters.get("is_appendix")
        if isinstance(is_appendix, str):
            filters["is_appendix"] = True if is_appendix.lower() == "true" else None
        elif is_appendix is False:
            filters["is_appendix"] = None
        
        return {
            "rewritten_queries": [query],
            "metadata_filters": filters,
            "file_analysis": file_analysis.get("focus_summary", ""),
            "pending_tasks": []
        }

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Truy xuất tài liệu từ Qdrant theo 3 class Sector Search."""
        from backend.retrieval.hybrid_search import retriever
        from backend.agent.utils.sub_timer import SubTimer
        timer = SubTimer("Retrieve")
        
        rewritten_queries = state.get("rewritten_queries") or [state.get("condensed_query") or state["query"]]
        query = state.get("condensed_query") or state["query"]
        kw = rewritten_queries[0]
        filters = state.get("metadata_filters", {})
        sectors = filters.get("legal_sectors", [])
        use_rerank = state.get("use_rerank", True)
        llm_preset = state.get("llm_preset")
        
        all_hierarchical_hits = []
        
        # ====== BLOCK 1: UPLOAD PRIORITY ======
        session_id = state.get("session_id")
        if session_id and state.get("file_chunks"):
            with timer.step("Session_Search"):
                session_hits, source = retriever.search_by_session(
                    session_id=session_id,
                    query=kw,
                    top_k=settings.MAX_RETRIEVAL_HITS,
                    use_rerank=use_rerank
                )
            if session_hits:
                all_hierarchical_hits = session_hits
                print(f"       🔍 [Retrieve-Sector] Using {len(all_hierarchical_hits)} hits from upload session.")
                
        # ====== BLOCK 2: 3 SPECIALIZED RETRIEVAL FUNCTIONS ======
        if not all_hierarchical_hits:
            from backend.agent.utils.utils_sector_search import (
                retrieve_by_sector,
                retrieve_by_topic_hybrid,
                retrieve_by_article_clause
            )
            
            mode = state.get("sector_search_class")
            if not mode:
                # Heuristic fallback
                if sectors and len(kw.split()) < 4:
                    mode = "sector"
                elif "điều" in kw.lower() or "khoản" in kw.lower():
                    mode = "article"
                else:
                    mode = "topic"
                    
            print(f"       🔍 [Retrieve-Sector] Mode: {mode.upper()}")
            
            with timer.step("Qdrant_Search"):
                if mode == "sector" and sectors:
                    all_hierarchical_hits = retrieve_by_sector(sectors[0])
                elif mode == "article":
                    all_hierarchical_hits = retrieve_by_article_clause(kw, top_k=settings.MAX_RETRIEVAL_HITS)
                else: # topic
                    all_hierarchical_hits = retrieve_by_topic_hybrid(kw, top_k=settings.MAX_RETRIEVAL_HITS)
                    if sectors:
                        sector_hits = retrieve_by_sector(sectors[0])
                        hits_ids = {h["id"] for h in all_hierarchical_hits}
                        for sh in sector_hits[:5]:
                            if sh["id"] not in hits_ids:
                                sh["score"] = float(sh.get("score", 0.0)) * 0.8
                                all_hierarchical_hits.append(sh)
                                hits_ids.add(sh["id"])
                            
        broad_hits = []
        
        try:
            from backend.retrieval.graph_db import search_docs_by_keyword, sector_based_on_reverse
            graph_keyword_hits = search_docs_by_keyword(query, limit=8)
            for h in graph_keyword_hits:
                payload = getattr(h, "payload", {}) or {}
                fake_hit = {
                    "id": getattr(h, "id", f"neo4j_kw_{payload.get('document_number')}"),
                    "score": getattr(h, "score", 0.95),
                    "document_number": payload.get("document_number", ""),
                    "title": payload.get("title", ""),
                    "text": payload.get("chunk_text", ""),
                    "legal_type": payload.get("legal_type", "Document"),
                    "chunk_id": getattr(h, "id", f"neo4j_kw_{payload.get('document_number')}")
                }
                all_hierarchical_hits.append(fake_hit)
        except Exception:
            pass
            
        final_hits = all_hierarchical_hits + broad_hits
        
        doc_number = filters.get("doc_number")
        article_ref = filters.get("article_ref")
        
        if doc_number:
            try:
                from backend.retrieval.graph_db import fetch_document_administrative_metadata, run_cypher
                admin_meta = fetch_document_administrative_metadata(doc_number)
                if admin_meta:
                    graph_context = state.get("graph_context", {})
                    graph_context["admin_metadata"] = admin_meta
                    state["graph_context"] = graph_context
                
                query_lower = query.lower()
                if "căn cứ" in query_lower or "dựa trên" in query_lower:
                    based_cypher = """
                    MATCH (d:Document)-[r:BASED_ON]->(b:Document) 
                    WHERE d.document_number CONTAINS $doc_num OR toLower(d.title) CONTAINS toLower($doc_num)
                    RETURN b.id AS id, b.title AS title, b.document_number AS document_number
                    LIMIT 20
                    """
                    res_based = run_cypher(based_cypher, {"doc_num": doc_number})
                    for r in res_based:
                        if r.get("document_number"):
                            fake_hit = {
                                "id": r.get("id", f"neo4j_based_{r.get('document_number')}"),
                                "score": 0.99,
                                "document_number": r.get('document_number'),
                                "title": r.get('title'),
                                "text": f"Văn bản này được ban hành dựa trên căn cứ là phần lớn của {doc_number}",
                                "legal_type": "Document",
                                "chunk_id": getattr(r, "id", f"neo4j_based_{r.get('document_number')}")
                            }
                            final_hits.insert(0, fake_hit)
            except Exception as e:
                pass
                
        if doc_number and article_ref:
            try:
                from backend.retrieval.graph_db import fetch_specific_article
                graph_articles = fetch_specific_article(doc_number, article_ref)
                for ga in graph_articles:
                    fake_hit = {
                        "id": f"graph_{ga.get('document_number')}_{ga.get('article_ref')}",
                        "score": 1.0,
                        "document_number": ga.get("document_number"),
                        "article_ref": ga.get("article_ref"),
                        "title": ga.get("title"),
                        "text": ga.get("article_text"),
                        "is_appendix": "phụ lục" in ga.get("article_ref", "").lower(),
                        "url": "",
                        "chunk_id": f"graph_{ga.get('document_number')}_{ga.get('article_ref')}"
                    }
                    if fake_hit["text"]:
                        final_hits.insert(0, fake_hit)
            except Exception as e:
                pass
                
        seen_chunks = set()
        dedup_hits = []
        for h in final_hits:
            cid = h.get("chunk_id")
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                dedup_hits.append(h)
        
        with timer.step("Neo4j_Graph"):
            graph_context = state.get("graph_context", {})
            graph_context.update({"sector_mapreduce": [], "sector_derivatives": []})
            
            chunk_ids = [h.get("chunk_id", "") for h in dedup_hits if h.get("chunk_id")]
    
            if sectors:
                try:
                    from backend.retrieval.graph_db import sector_mapreduce
                    for sector_name in sectors[:3]:
                        mr_result = sector_mapreduce(sector_name)
                        if mr_result:
                            graph_context["sector_mapreduce"].extend(mr_result)
                        
                        # Reverse BASED_ON query
                        derived = sector_based_on_reverse(sector_name)
                        if derived:
                            graph_context["sector_derivatives"].extend(derived)
                            # Inject fake hits for some derivatives to increase awareness
                            for drv in derived[:5]:
                                fake_hit = {
                                    "id": f"neo4j_deriv_{drv.get('derived_doc_number')}",
                                    "score": 0.85, # Lower score than direct matches
                                    "document_number": drv.get("derived_doc_number"),
                                    "title": drv.get("derived_title"),
                                    "text": f"Văn bản phái sinh liên quan: {drv.get('derived_doc_number')} ({drv.get('derived_title')}), căn cứ vào {drv.get('base_doc_number')}",
                                    "legal_type": "Document",
                                    "chunk_id": f"neo4j_deriv_{drv.get('derived_doc_number')}"
                                }
                                dedup_hits.append(fake_hit)
                except Exception as e:
                    pass
            
        return {"raw_hits": dedup_hits[:25], "graph_context": graph_context, "metrics": timer.results()}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Executive Summary + Bảng thống kê + Tóm tắt nội dung từng văn bản."""
        from backend.agent.utils.sub_timer import SubTimer
        timer = SubTimer("Generate")
        
        query = state.get("standalone_query") or state.get("condensed_query") or state["query"]
        hits = state.get("raw_hits", [])
        filters = state.get("metadata_filters", {})
        
        if not hits:
            return {"final_response": "Không tìm thấy văn bản pháp luật nào phù hợp với truy vấn."}
            
        unique_docs = group_by_document(hits)
        date_range = filters.get("effective_date_range", {})
        relevant_docs = _heuristic_date_filter(unique_docs, date_range)
        
        with timer.step("BuildContext"):
            # Bảng thống kê MapReduce (giữ lại cho câu hỏi đếm)
            table_markdown = map_reduce_aggregate(relevant_docs)
            
            graph_context = state.get("graph_context", {})
            if graph_context and graph_context.get("admin_metadata"):
                table_markdown = f"{graph_context['admin_metadata']}\n\n---\n\n{table_markdown}"
            
            file_chunks = state.get("file_chunks", [])
            
            if not table_markdown and not file_chunks:
                return {"final_response": "Không tìm thấy văn bản pháp luật nào phù hợp với truy vấn.", "metrics": timer.results()}
            
            # Tạo tóm tắt nội dung liên quan cho từng văn bản (top 5)
            doc_summaries = []
            for doc in relevant_docs[:5]:
                doc_num = doc.get("document_number", "N/A")
                title = doc.get("title", "N/A")
                text_snippet = doc.get("text", "")
                if text_snippet:
                    # Cắt ngắn và format snippet
                    snippet = text_snippet[:800].strip()
                    if len(text_snippet) > 800:
                        snippet += "..."
                    doc_summaries.append(
                        f"#### 📄 {doc_num} — {title}\n> {snippet}\n"
                    )
            
            content_section = ""
            if doc_summaries:
                content_section = "### 📝 Nội dung liên quan\n\n" + "\n".join(doc_summaries)
            
            history_msgs = state.get("history", [])[-6:]
            history_str = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_msgs]) if history_msgs else "(Không có lịch sử)"
            
            # Thêm content_section vào table_markdown để LLM tóm tắt có thể đọc được nội dung chi tiết
            enhanced_context_for_summary = f"{table_markdown}\n\n{content_section}"
        
        with timer.step("LLM_Call"):
            summary_thinking, summary = generate_executive_summary(
            query, 
            enhanced_context_for_summary, 
            file_chunks, 
            file_analysis=state.get("file_analysis", ""),
            history_str=history_str,
            llm_preset=state.get("llm_preset")
        )
        
        report = ""
        if file_chunks:
            file_insight = state.get("file_analysis", "Tài liệu tải lên chứa các nội dung liên quan đến lĩnh vực này.")
            report += f"### 💡 Phân tích từ Tài liệu tải lên\n> {file_insight}\n\n---\n\n"
            
        report += f"### 📝 Tóm tắt Tổng quan (Executive Summary)\n\n{summary}\n\n---\n\n"
        
        if content_section:
            report += f"{content_section}\n\n---\n\n"
            
        # Thêm thông tin phái sinh nếu có
        derivatives = graph_context.get("sector_derivatives", [])
        if derivatives:
            report += "### 🔗 Văn bản phái sinh liện quan\n"
            report += "Các văn bản sau được ban hành dựa trên các văn bản chính yếu trong lĩnh vực:\n"
            added = set()
            for d in derivatives:
                dnum = d.get('derived_doc_number')
                if dnum and dnum not in added:
                    report += f"- **{dnum}** ({d.get('derived_title')}) - Căn cứ: {d.get('base_doc_number')}\n"
                    added.add(dnum)
                    if len(added) >= 5: break
            report += "\n---\n\n"
        
        report += f"### 📊 Bảng Thống kê Văn bản\n{table_markdown}"
        
        with timer.step("FilterRefs"):
            refs = []
            sorted_hits = sorted(relevant_docs, key=lambda x: x.get("score", 0), reverse=True)
            for h in sorted_hits:
                # Emit TẤT CẢ các điều khoản được tìm thấy thay vì chỉ 1
                all_arts = h.get("all_articles") or [h.get("article_ref", h.get("document_number", ""))]
                for art in all_arts:
                    refs.append({
                        "title": h.get("title", ""),
                        "article": art,
                        "score": h.get("score", 0),
                        "chunk_id": f"{h.get('chunk_id', '')}_{art}",
                        "text_preview": h.get("text", ""),
                        "document_number": h.get("document_number", ""),
                        "url": h.get("url", "")
                    })
                
            cited_refs = filter_cited_references(report, refs)
            
        return {
            "final_response": report,
            "thinking_content": summary_thinking,
            "references": cited_refs,
            "metrics": timer.results()
        }
