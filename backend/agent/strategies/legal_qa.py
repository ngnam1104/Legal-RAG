from typing import Dict, Any
import time

from backend.agent.state import AgentState
from backend.agent.strategies.base import BaseRAGStrategy
from backend.agent.utils.utils_legal_qa import (
    build_legal_context,
    chat_completion,
    strip_thinking_tags,
    ANSWER_PROMPT,
    filter_cited_references
)
from backend.config import settings
from backend.retrieval.hybrid_search import retriever

class LegalQAStrategy(BaseRAGStrategy):
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """Lấy tham số từ SuperRouter + phân tích file upload (nếu có)."""
        hypothetical = state.get("condensed_query") or state["query"]
        filters = state.get("router_filters", {}) or {}
        file_analysis = ""
        
        # Nếu có file upload, trích xuất keywords bổ sung cho retrieval
        file_chunks = state.get("file_chunks", [])
        if file_chunks:
            import re
            sample_text = ""
            for c in file_chunks[:3]:
                sample_text += c.get("text_to_embed", c.get("unit_text", "")) + "\n"
            
            # Trích doc_numbers từ file để bổ sung filter
            doc_nums = re.findall(r'\d+/\d{4}/[A-Za-zĐđ\-]+', sample_text)
            if doc_nums and not filters.get("doc_number"):
                filters["doc_number"] = doc_nums[0]
                print(f"       📎 [Understand] File upload → Detected doc_number: {doc_nums[0]}")
            
            # Trích keywords pháp lý từ file để bổ sung query
            keywords = re.findall(r'(?:Điều|Khoản|Mục|Chương|Phụ lục)\s+\d+[a-z]?', sample_text)
            if keywords:
                kw_str = ", ".join(list(set(keywords))[:3])
                hypothetical = f"{hypothetical} ({kw_str})"
                print(f"       📎 [Understand] File upload → Enriched query with: {kw_str}")
            
            file_analysis = sample_text[:200]
        
        print(f"       🧠 [Understand] Prepared Query: '{hypothetical[:100]}...'")
        if filters:
            print(f"       🧠 [Understand] Extracted Filters: {filters}")
            
        return {
            "rewritten_queries": [hypothetical],
            "metadata_filters": filters,
            "file_analysis": file_analysis,
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
        
        # LLM trả về JSON boolean đôi khi bị ép về string. 
        if isinstance(is_appendix, str):
            is_appendix = True if is_appendix.lower() == "true" else None
        elif is_appendix is False:
            is_appendix = None
            
        use_rerank = state.get("use_rerank", True)
        
        hits = retriever.search(
            query=kw,
            expand_context=True,
            max_neighbors=5,
            use_rerank=use_rerank,
            legal_type=legal_type,
            doc_number=doc_number,
            is_appendix=is_appendix,
            article_ref=article_ref,
            limit=settings.MAX_RETRIEVAL_HITS
        )
        if (legal_type or doc_number or is_appendix is not None or article_ref) and not hits:
            print(f"       ⚠️ [Retrieve] No hits with filters. Dropping filters and retrying broad search...")
            hits = retriever.search(
                query=kw, 
                expand_context=True, 
                max_neighbors=5, 
                use_rerank=use_rerank,
                limit=settings.MAX_RETRIEVAL_HITS
            )
        
        print(f"       🔍 [Retrieve] Qdrant search returned {len(hits)} hits")
        
        # --- NEO4J GRAPH EXPANSION ---
        graph_context = {"lateral_docs": [], "document_toc": "", "sibling_texts": [], "signer_info": "", "based_on_info": "", "year_info": ""}
        chunk_ids = [h.get("chunk_id", "") for h in hits if h.get("chunk_id")]
        
        if doc_number:
            main_query = state.get("condensed_query", state["query"])
            query_lower = main_query.lower()
            try:
                from backend.retrieval.graph_db import run_cypher
                try:
                    from backend.retrieval.graph_db import fetch_document_administrative_metadata
                    admin_meta = fetch_document_administrative_metadata(doc_number)
                    if admin_meta:
                        graph_context['admin_metadata'] = admin_meta
                except ImportError:
                    pass

                if any(kw in query_lower for kw in ["căn cứ", "dựa trên", "cơ sở"]):
                    based_on_cypher = """
                    MATCH (d:Document {document_number: $doc_num})-[r:BASED_ON]->(b) 
                    RETURN b.name AS basis_name, b.document_number AS basis_num, r.target_text AS basis_text
                    """
                    res_based = run_cypher(based_on_cypher, {"doc_num": doc_number})
                    if res_based:
                        bases = []
                        for r in res_based:
                            bn = r.get("basis_name") or r.get("basis_num") or "Văn bản"
                            bases.append(bn)
                        graph_context["based_on_info"] = f"Văn bản {doc_number} được ban hành dựa trên các căn cứ: {'; '.join(bases)}."
            except Exception as e:
                print(f"       ⚠️ [Neo4j] Direct metadata query failed: {e}")

        if doc_number:
            try:
                from backend.retrieval.graph_db import fetch_specific_article, fetch_full_document_articles
                graph_articles = []
                if article_ref:
                    print(f"       🎯 [Neo4j] Direct Entity Query for exact match: {doc_number} - {article_ref}")
                    graph_articles = fetch_specific_article(doc_number, article_ref)
                
                if not graph_articles and not chunk_ids:
                    graph_articles = fetch_full_document_articles(doc_number)
                
                if graph_articles:
                    for ga in graph_articles:
                        art_text = ga.get("article_text", "")
                        if art_text:
                            graph_context["sibling_texts"].append(f"[{ga.get('article_ref', '')}] {art_text}")
                        for cl in (ga.get("clauses") or []):
                            if cl.get("text"):
                                graph_context["sibling_texts"].append(f"[{ga.get('article_ref', '')} - {cl.get('name', '')}] {cl['text']}")
                    if graph_context["sibling_texts"]:
                        print(f"       🎯 [Neo4j] Direct entity query recovered {len(graph_context['sibling_texts'])} article/clause texts")
            except Exception as e:
                pass
        
        if chunk_ids:
            try:
                from backend.retrieval.graph_db import bottom_up_expand, lateral_expand
                bu_result = bottom_up_expand(chunk_ids)
                graph_context["document_toc"] = bu_result.get("document_toc", "")
                graph_context["sibling_texts"].extend(bu_result.get("sibling_texts", []))
                graph_context["lateral_docs"] = lateral_expand(chunk_ids)
                
            except Exception as e:
                pass
            
        return {"raw_hits": hits, "graph_context": graph_context}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Sinh câu trả lời từ context + lateral docs."""
        hits = state.get("raw_hits", [])
        file_chunks = state.get("file_chunks", [])
        graph_ctx = state.get("graph_context", {})
        
        # Build legal context (previously in grade)
        context_text = build_legal_context(hits, file_chunks=file_chunks, graph_context=graph_ctx)
        
        # Dùng standalone_query (câu hỏi thuần, không có HyDE) để LLM tổng hợp.
        # condensed_query (kèm HyDE) chỉ dành cho retrieval embedding.
        query = state.get("standalone_query") or state.get("condensed_query") or state["query"]
        
        if not context_text:
            return {"final_response": "Xin lỗi, tôi không tìm thấy quy định pháp luật nào liên quan đến câu hỏi của bạn."}
            
        # Thêm reference logic và đảm bảo được sort theo score
        refs = []
        combined_hits = state.get("raw_hits", [])
        combined_hits.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for h in combined_hits:
            refs.append({
                "title": h.get("title", ""),
                "article": h.get("article_ref", h.get("document_number", "")),
                "score": h.get("score", 0),
                "chunk_id": h.get("chunk_id", ""),
                "text_preview": h.get("text", ""),
                "document_number": h.get("document_number", ""),
                "url": h.get("url", "")
            })
            
        history_msgs = state.get("history", [])[-6:]
        history_str = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_msgs]) if history_msgs else "(Không có lịch sử)"
        
        supplemental = state.get("supplemental_context", "")
            
        prompt = ANSWER_PROMPT.format(
            history=history_str,
            context=context_text, 
            query=query, 
            supplemental_context=supplemental
        )
        print(f"       ✍️ [Generate] Generating answer with {len(refs)} candidate references...")
        answer = strip_thinking_tags(chat_completion(
            [{"role": "user", "content": prompt}], 
            temperature=0.1, 
            model=settings.LLM_CORE_MODEL, 
            llm_preset=state.get("llm_preset")
        ))
        
        # Lọc chỉ giữ references thực sự được trích dẫn trong câu trả lời
        cited_refs = filter_cited_references(answer, refs)
        print(f"       📌 [Generate] Cited {len(cited_refs)}/{len(refs)} references")
            
        return {
            "final_response": answer,
            "references": cited_refs
        }
