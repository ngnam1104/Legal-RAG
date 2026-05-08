"""
LegalChatStrategy — Unified Legal RAG Strategy (GraphRAG Architecture)
======================================================================
Hợp nhất LEGAL_QA + SECTOR_SEARCH + CONFLICT_ANALYZER thành 1 mode duy nhất.

Pipeline:
  1. Understand: SuperRouter đã xử lý → chuẩn bị query + filters
  2. Retrieve:   QdrantNeo4jRetriever (Vector Search → Neo4j Subgraph 2-hop)
  3. Generate:   GraphRAG Prompt (Nodes + Edges + Query → LLM)
  4. Reflect:    Reviewer agent kiểm tra ảo giác (optional)
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from backend.agent.state import AgentState
from backend.agent.utils_legal import (
    fetch_related_graph,
    format_graph_context,
    build_legal_context,
    filter_cited_references,
)
from backend.prompt import GRAPHRAG_PROMPT, ANSWER_PROMPT, REFLECT_PROMPT
from backend.models.llm_factory import chat_completion
import os
from backend.models.embedder import embedder

logger = logging.getLogger("legal_chat")


# ---------------------------------------------------------------------------
# Abstract Base Class (inlined from old strategies/base.py)
# ---------------------------------------------------------------------------
class BaseRAGStrategy(ABC):
    @abstractmethod
    def understand(self, state: AgentState) -> Dict[str, Any]:
        pass

    @abstractmethod
    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate(self, state: AgentState) -> Dict[str, Any]:
        pass

    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """Default reflection: pass-through (subclasses may override)."""
        return {"pass_flag": True, "feedback": "Reflection not implemented for this strategy."}


# ---------------------------------------------------------------------------
# GraphRAG Generation Prompt
# ---------------------------------------------------------------------------
# Prompts are imported from backend.prompt


class LegalChatStrategy(BaseRAGStrategy):
    """Unified GraphRAG strategy for all legal queries."""

    # ------------------------------------------------------------------
    # 1. UNDERSTAND
    # ------------------------------------------------------------------
    def understand(self, state: AgentState) -> Dict[str, Any]:
        """Chuẩn bị query và filters từ SuperRouter output."""
        hypothetical = state.get("condensed_query") or state["query"]
        filters = state.get("router_filters", {}) or {}
        file_analysis = ""

        # Nếu có file upload, enrichment
        file_chunks = state.get("file_chunks", [])
        if file_chunks:
            sample_text = ""
            for c in file_chunks[:3]:
                sample_text += c.get("text_to_embed", c.get("unit_text", "")) + "\n"

            doc_nums = re.findall(r'\d+/\d{4}/[A-Za-zĐđ\-]+', sample_text)
            if doc_nums and not filters.get("doc_number"):
                filters["doc_number"] = doc_nums[0]
                print(f"       📎 [Understand] File upload → Detected doc_number: {doc_nums[0]}")

            keywords = re.findall(r'(?:Điều|Khoản|Mục|Chương|Phụ lục)\s+\d+[a-z]?', sample_text)
            if keywords:
                kw_str = ", ".join(list(set(keywords))[:3])
                hypothetical = f"{hypothetical} ({kw_str})"
                print(f"       📎 [Understand] File upload → Enriched query with: {kw_str}")

            file_analysis = sample_text[:200]

        print(f"       🧠 [Understand] Prepared Query: '{hypothetical}'")
        if filters:
            print(f"       🧠 [Understand] Extracted Filters: {filters}")

        return {
            "rewritten_queries": [hypothetical],
            "metadata_filters": filters,
            "file_analysis": file_analysis,
            "pending_tasks": []
        }

    # ------------------------------------------------------------------
    # 2. RETRIEVE  (HybridRetriever + QdrantNeo4jRetriever + Subgraph)
    # ------------------------------------------------------------------
    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """
        3-phase retrieval:
          Phase 1: HybridRetriever (Dense+Sparse RRF + Rerank + Expand)
                   → Chất lượng cao nhất: filter, BM25 sparse, reranker, context expand.
          Phase 2: QdrantNeo4jRetriever — enrich với Neo4j node data + thu thập entity_ids.
          Phase 3: fetch_related_graph — 2-hop subgraph expansion cho GraphRAG context.
        """
        from backend.agent.utils_general import SubTimer
        timer = SubTimer("Retrieve")

        rewritten_queries = state.get("rewritten_queries") or [state.get("condensed_query") or state["query"]]
        query = rewritten_queries[0] or state.get("condensed_query") or state["query"]
        filters = state.get("metadata_filters", {}) or {}

        from concurrent.futures import ThreadPoolExecutor
        from backend.retrieval.graph_search import entity_retriever
        from backend.retrieval.hybrid_search import retriever as hybrid_retriever

        # ── Phase In-Memory Search (Cho file tải lên) ──
        file_chunks = state.get("file_chunks", [])
        if file_chunks and len(file_chunks) > 0:
            with timer.step("InMemorySearch"):
                try:
                    from backend.models.embedder import embedder
                    import numpy as np
                    
                    query_vector = np.array(embedder.encode_query_dense(query))
                    query_norm = np.linalg.norm(query_vector)
                    
                    scored_chunks = []
                    for chunk in file_chunks:
                        chunk_vec = chunk.get("vector")
                        if chunk_vec:
                            vec = np.array(chunk_vec)
                            norm = np.linalg.norm(vec)
                            score = np.dot(query_vector, vec) / (query_norm * norm) if query_norm > 0 and norm > 0 else 0.0
                            scored_chunks.append((score, chunk))
                        else:
                            scored_chunks.append((0.0, chunk))
                            
                    scored_chunks.sort(key=lambda x: x[0], reverse=True)
                    # Giữ tối đa 10 chunks có liên quan nhất
                    top_file_chunks = [c for score, c in scored_chunks[:10]]
                    state["file_chunks"] = top_file_chunks
                    print(f"       📎 [Retrieve] In-Memory Search: Lọc top {len(top_file_chunks)}/{len(file_chunks)} chunks từ file tải lên.")
                except Exception as e:
                    print(f"       ⚠️ [Retrieve] In-Memory Search failed: {e}")

        # ── Phase 0+1: SONG SONG — Entity Graph Search & Hybrid Search ──
        # Hai phase này độc lập hoàn toàn nên chạy đồng thời để tiết kiệm 1-3s mỗi query.
        # Boost từ Phase 0 được áp dụng SAU KHI cả hai hoàn thành.
        def _run_graph_search():
            return entity_retriever.search(query)

        def _run_hybrid_search():
            return hybrid_retriever.search(
                query=query,
                expand_context=True,
                max_neighbors=2, # Giảm từ 8 xuống 2 để tránh tràn ngữ cảnh
                use_rerank=state.get("use_rerank", True),
                legal_type=filters.get("legal_type"),
                doc_number=filters.get("doc_number"),
                article_ref=filters.get("article_ref"),
                sector=filters.get("sector"),
                limit=int(os.environ.get("MAX_RETRIEVAL_HITS", 20)),
            )

        with timer.step("Phase0_and_Phase1_Parallel"):
            with ThreadPoolExecutor(max_workers=2) as exe:
                future_graph  = exe.submit(_run_graph_search)
                future_hybrid = exe.submit(_run_hybrid_search)
                graph_res = future_graph.result()
                hits      = future_hybrid.result()

        graph_boost_chunk_ids = graph_res.get("chunk_ids", [])
        entity_pre_context    = graph_res.get("graph_context", "")

        # Áp dụng Graph Boost sau khi cả hai phase xong
        if graph_boost_chunk_ids:
            boost_set = set(graph_boost_chunk_ids)
            for hit in hits:
                cid = str(hit.get("chunk_id") or hit.get("id", ""))
                if cid in boost_set:
                    hit["rerank_score"] = hit.get("rerank_score", hit.get("score", 0)) + 0.3
                    hit["score"] = hit["rerank_score"]
            hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)

        print(
            f"       🔍 [Retrieve] Phase0_Graph: {len(graph_boost_chunk_ids)} boost_ids, "
            f"{len(graph_res.get('doc_numbers',[]))} doc_hints "
            f"| Phase1_Hybrid: {len(hits)} hits"
        )

        # ── Phase 1.5: Fetch Graph-hinted documents bị Vector Search bỏ sót ──
        # entity_retriever.search() đã biết chính xác các văn bản liên quan (doc_numbers).
        # Nếu Qdrant chưa kéo được chunks từ đó, ta scroll Qdrant theo doc_number.
        # OPTIMIZATION: Batch query toàn bộ missing_docs cùng lúc (MatchAny), không loop tuần tự
        graph_doc_numbers = graph_res.get("doc_numbers", [])
        if graph_doc_numbers:
            existing_doc_nums = {h.get("document_number", "") for h in hits}
            missing_docs = [d for d in graph_doc_numbers if d not in existing_doc_nums][:3]
            if missing_docs:
                with timer.step("Graph_Doc_Fetch"):
                    try:
                        from qdrant_client import models
                        from backend.agent.utils_legal_optimization import optimize_batch_fetch
                        
                        # Strategy selection: "batch" (baseline) | "neo4j" (Cách 2) | "rerank" (Cách 1) | "hybrid" (kết hợp)
                        strategy = os.environ.get("GRAPH_DOC_FETCH_STRATEGY", "neo4j")
                        
                        # Phase A: Batch query từ Qdrant (luôn chạy để có baseline)
                        missing_filter = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="document_number",
                                    match=models.MatchAny(any=missing_docs)
                                ),
                                models.FieldCondition(
                                    key="is_active",
                                    match=models.MatchValue(value=True)
                                )
                            ]
                        )
                        
                        # Single scroll call: lấy chunks cho TẤT CẢ missing_docs cùng lúc
                        batch_hits, _ = hybrid_retriever.client.scroll(
                            collection_name=hybrid_retriever.collection_name,
                            scroll_filter=missing_filter,
                            with_payload=["chunk_text", "document_number", "chunk_id", "id", "article_ref", "title"],
                            with_vectors=False,
                            limit=len(missing_docs) * 5  # ~5 chunks per doc
                        )
                        
                        # Convert to dict format
                        batch_hits_dict = []
                        for bp in batch_hits:
                            if not bp.payload:
                                continue
                            payload = bp.payload
                            hit_dict = {
                                "id": bp.id,
                                "chunk_id": payload.get("chunk_id", ""),
                                "document_number": payload.get("document_number", ""),
                                "title": payload.get("title", ""),
                                "text": payload.get("chunk_text", ""),
                                "score": 0.6,
                                "payload": payload,
                            }
                            batch_hits_dict.append(hit_dict)
                        
                        # Phase B: Apply optimization strategy
                        if strategy == "batch":
                            # Baseline: Use raw batch hits + boost
                            final_hits = batch_hits_dict
                            for fh in final_hits:
                                fh["score"] = fh.get("score", 0) + 0.25
                            print(f"       📌 [Graph Doc Fetch] Strategy=BATCH: {len(final_hits)} chunks")
                        
                        elif strategy in ["neo4j", "rerank", "hybrid"]:
                            # Optimized strategies
                            final_hits = optimize_batch_fetch(
                                batch_hits=batch_hits_dict,
                                missing_docs=missing_docs,
                                query=query,
                                reranker_client=hybrid_retriever.reranker if hasattr(hybrid_retriever, 'reranker') else None,
                                strategy=strategy,
                            )
                            for fh in final_hits:
                                fh["score"] = fh.get("score", 0) + 0.15  # Conservative boost
                        else:
                            # Unknown strategy, fallback to batch
                            final_hits = batch_hits_dict
                            for fh in final_hits:
                                fh["score"] = fh.get("score", 0) + 0.25
                        
                        # Append to hits
                        hits.extend(final_hits)
                        
                        if final_hits:
                            print(f"       📌 [Graph Doc Fetch] Strategy={strategy.upper()}: Added {len(final_hits)} chunks")
                    
                    except Exception as e:
                        import logging; logging.getLogger(__name__).warning(f"Graph doc fetch error: {e}")

        # ── Phase 1.8: UNIFIED RERANKING (Gộp File Up + Global DB) ──
        with timer.step("Unified_Reranking"):
            # 1. Chuẩn hóa file_chunks thành format của hits
            normalized_file_hits = []
            for idx, fc in enumerate(state.get("file_chunks", [])):
                normalized_file_hits.append({
                    "id": fc.get("chunk_id") or f"upload_{idx}",
                    "chunk_id": fc.get("chunk_id") or f"upload_{idx}",
                    "text": fc.get("chunk_text", fc.get("text", fc.get("unit_text", ""))),
                    "title": fc.get("title", "Tài liệu tải lên"),
                    "document_number": fc.get("document_number", "File Upload"),
                    "article_ref": fc.get("article_ref", ""),
                    "score": 0.95, # Ưu tiên cực cao
                    "payload": fc,
                    "_source": "upload"
                })
            
            # 2. Gộp danh sách
            all_candidates = normalized_file_hits + hits
            
            # 3. Chạy Reranker trên toàn bộ danh sách gộp
            if all_candidates and state.get("use_rerank", True):
                from backend.models.reranker import reranker
                # Rerank trả về danh sách đã sắp xếp theo điểm số Cross-Encoder
                reranked_all = reranker.rerank(query, all_candidates)
                # Lấy Top 10 tinh túy nhất
                final_top_hits = reranked_all[:10]
            else:
                final_top_hits = all_candidates[:10]
            
            # 4. Tách lại để giữ logic build_legal_context (hoặc cập nhật state)
            state["raw_hits"] = [h for h in final_top_hits if h.get("_source") != "upload"]
            state["file_chunks"] = [h["payload"] for h in final_top_hits if h.get("_source") == "upload"]
            
            print(f"       🎯 [Unified Rerank] Picked top 10 from {len(all_candidates)} candidates (Upload: {len(state['file_chunks'])}, DB: {len(state['raw_hits'])})")

        # Thu thập entity_ids từ Final Top Hits (để expand graph nếu cần)
        entity_ids = [
            str(h.get("chunk_id") or h.get("id", ""))
            for h in final_top_hits if h.get("chunk_id") or h.get("id")
        ]

        # ── Phase 2: QdrantNeo4jRetriever — enrich ──
        # Giữ nguyên nhưng chỉ enrich cho những gì còn lại sau Rerank
        is_pure_upload = len(final_top_hits) > 0 and all(h.get("_source") == "upload" for h in final_top_hits)
        
        neo4j_hits, neo4j_entity_ids = [], []
        if not is_pure_upload:
            with timer.step("QdrantNeo4j_Enrich"):
                neo4j_hits, neo4j_entity_ids = self._qdrant_neo4j_search(query, state)
        else:
            print("       📎 [Retrieve] Pure Upload context after Rerank. Skipping Neo4j.")

        # Update final hits after enrichment
        # Lưu ý: Ở đây ta chỉ muốn enrich metadata chứ không muốn làm xáo trộn thứ tự Rerank
        # Nên ta sẽ map metadata từ neo4j_hits vào final_top_hits
        if neo4j_hits:
            neo4j_map = {str(nh.get("chunk_id")): nh for nh in neo4j_hits}
            for fh in final_top_hits:
                cid = str(fh.get("chunk_id"))
                if cid in neo4j_map:
                    # Cập nhật metadata cho cả wrapper và payload bên trong
                    fh.update(neo4j_map[cid])
                    if "payload" in fh and isinstance(fh["payload"], dict):
                        fh["payload"].update(neo4j_map[cid])
                    fh["_source"] = "neo4j_enriched"

        # Merge entity_ids (unique, giữ thứ tự) để dùng cho Phase 3
        seen_ids = dict.fromkeys(entity_ids)
        for eid in neo4j_entity_ids:
            seen_ids.setdefault(eid)
        all_entity_ids = list(seen_ids.keys())

        # ── Phase 3: Lighter Subgraph Expansion (Siblings Only) ──
        # Subgraph chính (nodes/edges) đã lấy từ Phase 0 để tiết kiệm thời gian.
        # Phase 3 chỉ "vét" thêm các đoạn văn liên quan ở các văn bản khác (siblings).
        graph_ctx = {
            "nodes": graph_res.get("nodes", []),
            "edges": graph_res.get("edges", []),
            "entity_context": entity_pre_context,
            "node_rel_lines": [], 
            "lateral_docs": [],
            "document_toc": "", 
            "sibling_texts": []
        }
        with timer.step("Neo4j_Siblings_Only"):
            if all_entity_ids and not is_pure_upload:
                from backend.agent.utils_legal import fetch_sibling_context
                siblings = fetch_sibling_context(all_entity_ids)
                graph_ctx["sibling_texts"] = siblings
                print(f"       🕸️ [Retrieve] Phase 3 Lighter: Found {len(siblings)} sibling texts via graph.")

        return {"raw_hits": hits, "graph_context": graph_ctx, "metrics": timer.results()}

    def _build_qdrant_filter(self, state: AgentState):
        """Tạo Qdrant filter từ metadata_filters trong state."""
        from qdrant_client import models
        filters = state.get("metadata_filters", {}) or {}
        must_conditions = []

        # Luôn lọc văn bản còn hiệu lực
        must_conditions.append(
            models.FieldCondition(key="is_active", match=models.MatchValue(value=True))
        )

        legal_type = filters.get("legal_type")
        if legal_type:
            must_conditions.append(
                models.FieldCondition(key="legal_type", match=models.MatchValue(value=legal_type))
            )

        doc_number = filters.get("doc_number")
        if doc_number:
            must_conditions.append(
                models.Filter(should=[
                    models.FieldCondition(key="document_number", match=models.MatchText(text=doc_number)),
                    models.FieldCondition(key="title",           match=models.MatchText(text=doc_number)),
                ])
            )

        article_ref = filters.get("article_ref")
        if article_ref:
            must_conditions.append(
                models.FieldCondition(key="article_ref", match=models.MatchText(text=article_ref))
            )

        return models.Filter(must=must_conditions)

    def _qdrant_neo4j_search(self, query: str, state: AgentState):
        """
        QdrantNeo4jRetriever — enrich hits với Neo4j node data.
        Trả về (hits, entity_ids). Nếu Neo4j unavailable → trả ([], []).
        KHÔNG làm fallback — Phase 1 (HybridRetriever) đã đảm bảo có kết quả.
        """
        try:
            from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
            from backend.database.qdrant_client import client as qdrant_client
            from backend.database.neo4j_client import get_neo4j_driver

            neo4j_driver = get_neo4j_driver()
            if not neo4j_driver:
                logger.debug("Neo4j driver not available — skipping QdrantNeo4jRetriever enrichment.")
                return [], []

            # Filter out upload chunks to avoid unnecessary Neo4j queries
            current_hits = state.get("raw_hits", [])
            existing_ids = {h.get("chunk_id") for h in current_hits if h.get("chunk_id")}
            
            top_k = state.get("top_k") or int(os.environ.get("MAX_RETRIEVAL_HITS", 20))

            # Cypher: lấy node + parent metadata
            # Lưu ý: $id là tham số mặc định được neo4j-graphrag truyền vào từ Qdrant result
            retrieval_query = """
            MATCH (node) 
            WHERE node.qdrant_id = $id OR node.id = $id
            OPTIONAL MATCH (node)-[:BELONGS_TO|PART_OF*1..2]->(parent:Document)
            RETURN node {
                .*,
                parent_title:           parent.title,
                parent_doc_number:      parent.document_number,
                parent_url:             parent.url,
                doc_effective_date:     parent.effective_date,
                doc_issuing_authority:  parent.issuing_authority
            } AS metadata
            """

            retriever_obj = QdrantNeo4jRetriever(
                driver=neo4j_driver,
                client=qdrant_client,
                collection_name=os.environ.get("QDRANT_COLLECTION", "legal_rag_docs_nam"),
                id_property_neo4j="qdrant_id",
                id_property_external="id",
                using="dense",
                retrieval_query=retrieval_query,
            )

            dense_vector = embedder.encode_query_dense(query)
            qdrant_filter = self._build_qdrant_filter(state)

            # Thử truyền filter (hỗ trợ tùy phiên bản neo4j-graphrag)
            try:
                results = retriever_obj.search(
                    query_vector=dense_vector,
                    top_k=top_k,
                    filter=qdrant_filter,
                )
            except Exception as inner_e:
                # Bắt mọi lỗi liên quan đến argument (TypeError, ValueError, Pydantic ValidationError)
                results = retriever_obj.search(
                    query_vector=dense_vector,
                    top_k=top_k,
                )

            hits = []
            entity_ids = []
            for item in results.items:
                content  = item.content  or ""
                metadata = item.metadata or {}
                node_id  = metadata.get("id") or metadata.get("qdrant_id") or ""

                hits.append({
                    "id":                node_id,
                    "score":             1.0,
                    "chunk_id":          node_id,
                    "document_number":   metadata.get("parent_doc_number") or metadata.get("document_number", ""),
                    "article_ref":       metadata.get("name", ""),
                    "title":             metadata.get("parent_title") or metadata.get("title", ""),
                    "text":              metadata.get("text", content),
                    "url":               metadata.get("parent_url") or metadata.get("url", ""),
                    "effective_date":    metadata.get("doc_effective_date", ""),
                    "issuing_authority": metadata.get("doc_issuing_authority", ""),
                    "is_appendix":       metadata.get("is_table", False),
                    "_source":           "neo4j_graphrag",
                })
                if node_id:
                    entity_ids.append(str(node_id))

            return hits, entity_ids

        except Exception as e:
            import traceback
            logger.warning(f"QdrantNeo4jRetriever enrichment skipped: {e}")
            # logger.debug(traceback.format_exc())
            return [], []


    # ------------------------------------------------------------------
    # 3. GENERATE  (GraphRAG Prompt)
    # ------------------------------------------------------------------
    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Sinh câu trả lời sử dụng GraphRAG: Nodes + Edges + Vector Context → LLM."""
        from backend.agent.utils_general import SubTimer
        timer = SubTimer("Generate")

        hits = state.get("raw_hits", [])
        file_chunks = state.get("file_chunks", [])
        graph_ctx = state.get("graph_context", {})
        query = state.get("standalone_query") or state.get("condensed_query") or state["query"]

        with timer.step("BuildContext"):
            # Format graph nodes and edges
            nodes_list = graph_ctx.get("nodes", [])
            edges_list = graph_ctx.get("edges", [])
            entity_context = graph_ctx.get("entity_context", "")
            node_rel_lines = graph_ctx.get("node_rel_lines", [])

            nodes_str = "\n".join(f"  • {n}" for n in nodes_list) if nodes_list else "(Không có dữ liệu đồ thị)"
            edges_str = "\n".join(f"  • {e}" for e in edges_list) if edges_list else "(Không có mối liên hệ)"
            entity_str = entity_context if entity_context else "(Không có thực thể)"
            node_rel_str = "\n".join(f"  • {nr}" for nr in node_rel_lines) if node_rel_lines else "(Không có quan hệ thực thể)"

            # Build vector context (traditional) as supplemental
            vector_context = build_legal_context(hits, file_chunks=file_chunks, graph_context=graph_ctx)

            if not vector_context and not nodes_list:
                return {
                    "final_response": "Xin lỗi, tôi không tìm thấy quy định pháp luật nào liên quan đến câu hỏi của bạn.",
                    "metrics": timer.results()
                }

            # Build references
            refs = []
            for h in sorted(hits, key=lambda x: x.get("score", 0), reverse=True):
                refs.append({
                    "title": h.get("title", ""),
                    "article": h.get("article_ref", h.get("document_number", "")),
                    "score": h.get("score", 0),
                    "chunk_id": h.get("chunk_id", ""),
                    "text_preview": h.get("text", ""),
                    "document_number": h.get("document_number", ""),
                    "url": h.get("url", "")
                })

        # Build history
        history_msgs = state.get("history", [])[-6:]
        history_str = "\n".join(
            [f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history_msgs]
        ) if history_msgs else "(Không có lịch sử)"

        # Choose prompt based on graph availability
        if nodes_list or entity_context:
            prompt = GRAPHRAG_PROMPT.format(
                history=history_str,
                nodes_str=nodes_str,
                edges_str=edges_str,
                entity_str=entity_str,
                node_rel_str=node_rel_str,
                vector_context=vector_context,
                query=query,
            )
        else:
            # Fallback to traditional prompt if no graph data
            supplemental = state.get("supplemental_context", "")
            prompt = ANSWER_PROMPT.format(
                history=history_str,
                context=vector_context,
                query=query,
                supplemental_context=supplemental,
            )

        print(f"       ✍️ [Generate] GraphRAG prompt with {len(nodes_list)} nodes, {len(edges_list)} edges, {len(node_rel_lines)} node_rels, {len(refs)} refs")

        with timer.step("LLM_Call"):
            answer = chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                model=os.environ.get("LLM_CORE_MODEL", "llama3"),
                llm_preset=state.get("llm_preset")
            )

        with timer.step("FilterRefs"):
            cited_refs = filter_cited_references(answer, refs)
            print(f"       📌 [Generate] Cited {len(cited_refs)}/{len(refs)} references")

        return {
            "final_response": answer,
            "references": cited_refs,
            "metrics": timer.results()
        }
    # ------------------------------------------------------------------
    # 4. REFLECT  (Reviewer Agent — Optional)
    # ------------------------------------------------------------------
    def reflect(self, state: AgentState) -> Dict[str, Any]:
        """
        Reviewer agent: kiểm tra hallucination và completeness.
        Trả về pass_flag, feedback, và corrected_answer nếu cần.
        """
        draft = state.get("final_response", "")
        query = state.get("standalone_query") or state["query"]
        graph_ctx = state.get("graph_context", {})
        hits = state.get("raw_hits", [])

        # Build minimal context string for review
        nodes_list = graph_ctx.get("nodes", [])
        context_for_review = "\n".join(nodes_list[:20]) if nodes_list else ""
        if not context_for_review:
            context_for_review = "\n".join(h.get("text", "")[:500] for h in hits[:5])

        if not draft or not context_for_review:
            return {"pass_flag": True, "feedback": "Không có đủ dữ liệu để review."}

        review_prompt = REFLECT_PROMPT.format(
            query=query,
            draft=draft,
            context=context_for_review[:8000]
        )

        try:
            from backend.utils.text_utils import extract_json_from_text, strip_thinking_tags
            import json

            response = chat_completion(
                [{"role": "user", "content": review_prompt}],
                temperature=0.0,
                llm_preset=state.get("llm_preset"),
            )

            clean = strip_thinking_tags(response or "")
            json_str = extract_json_from_text(clean)
            if json_str:
                data = json.loads(json_str)
                passed = data.get("pass", True)
                issues = data.get("issues", [])
                corrected = data.get("corrected_answer", "")

                feedback = "; ".join(issues) if issues else "OK"
                print(f"       🔍 [Reflect] pass={passed}, issues={len(issues)}")

                result = {"pass_flag": passed, "feedback": feedback}
                if not passed and corrected:
                    result["final_response"] = corrected
                return result

        except Exception as e:
            logger.error(f"Reflect failed: {e}")

        return {"pass_flag": True, "feedback": "Review skipped due to error."}
