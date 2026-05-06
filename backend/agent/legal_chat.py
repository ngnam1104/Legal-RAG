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

        print(f"       🧠 [Understand] Prepared Query: '{hypothetical[:100]}...'")
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

        # ── Phase 0: Entity Graph Retrieval (Pre-Retrieve) ──
        graph_boost_chunk_ids = []
        entity_pre_context = ""
        with timer.step("Entity_Graph_Search"):
            from backend.retrieval.graph_search import entity_retriever
            graph_res = entity_retriever.search(query)
            graph_boost_chunk_ids = graph_res.get("chunk_ids", [])
            entity_pre_context = graph_res.get("graph_context", "")
            
        # ── Phase 1: HybridRetriever (Dense + Sparse + Rerank + Expand) ──
        with timer.step("Hybrid_Search"):
            from backend.retrieval.hybrid_search import retriever as hybrid_retriever
            hits = hybrid_retriever.search(
                query=query,
                expand_context=True,
                max_neighbors=8,
                use_rerank=state.get("use_rerank", True),
                legal_type=filters.get("legal_type"),
                doc_number=filters.get("doc_number"),
                article_ref=filters.get("article_ref"),
                limit=int(os.environ.get("MAX_RETRIEVAL_HITS", 20)),
                graph_boost_chunk_ids=graph_boost_chunk_ids
            )

        # Thu thập entity_ids từ Hybrid hits
        entity_ids = [
            str(h.get("chunk_id") or h.get("id", ""))
            for h in hits if h.get("chunk_id") or h.get("id")
        ]
        print(f"       🔍 [Retrieve] HybridRetriever → {len(hits)} hits, {len(entity_ids)} entity_ids")

        # ── Phase 2: QdrantNeo4jRetriever — enrich + bổ sung entity_ids từ Neo4j ──
        with timer.step("QdrantNeo4j_Enrich"):
            neo4j_hits, neo4j_entity_ids = self._qdrant_neo4j_search(query, state)

        # Merge entity_ids (unique, giữ thứ tự)
        seen_ids = dict.fromkeys(entity_ids)
        for eid in neo4j_entity_ids:
            seen_ids.setdefault(eid)
        all_entity_ids = list(seen_ids.keys())

        # Merge neo4j_hits: chỉ thêm hits chưa có trong hybrid_hits
        hybrid_chunk_ids = {str(h.get("chunk_id") or h.get("id", "")) for h in hits}
        for nh in neo4j_hits:
            nid = str(nh.get("chunk_id") or nh.get("id", ""))
            if nid and nid not in hybrid_chunk_ids:
                hits.append(nh)
                hybrid_chunk_ids.add(nid)

        if neo4j_hits:
            print(f"       🔗 [Retrieve] QdrantNeo4j added {len(neo4j_hits)} extra hits → total {len(hits)}")

        # ── Phase 3: 2-hop Subgraph Expansion ──
        graph_ctx = {"nodes": [], "edges": [], "entity_context": "", "node_rel_lines": [], "lateral_docs": [], "document_toc": "", "sibling_texts": []}
        with timer.step("Neo4j_Subgraph"):
            if all_entity_ids:
                subgraph = fetch_related_graph(all_entity_ids)
                if subgraph:
                    formatted = format_graph_context(subgraph)
                    graph_ctx["nodes"] = formatted["nodes"]
                    graph_ctx["edges"] = formatted["edges"]
                    # Merge context từ Phase 0 và Phase 3
                    merged_entity_ctx = entity_pre_context
                    phase3_ctx = formatted.get("entity_context", "")
                    if phase3_ctx:
                        merged_entity_ctx = merged_entity_ctx + "\n" + phase3_ctx if merged_entity_ctx else phase3_ctx
                    
                    graph_ctx["entity_context"] = merged_entity_ctx
                    graph_ctx["node_rel_lines"] = formatted.get("node_rel_lines", [])
                    graph_ctx["sibling_texts"] = formatted.get("sibling_texts", [])
                    print(f"       🕸️ [Retrieve] Subgraph: {len(graph_ctx['nodes'])} nodes, {len(graph_ctx['edges'])} edges, {len(graph_ctx['node_rel_lines'])} node_rels, entities={bool(graph_ctx['entity_context'])}, siblings={len(graph_ctx['sibling_texts'])}")

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

            top_k = state.get("top_k") or int(os.environ.get("MAX_RETRIEVAL_HITS", 20))

            # Cypher: lấy node + parent metadata + ontology relations (cải tiến so với cũ)
            retrieval_query = """
            MATCH (node)
            WHERE node.qdrant_id = $id OR node.id = $id
            OPTIONAL MATCH (node)-[:BELONGS_TO|PART_OF*1..2]->(parent)
            OPTIONAL MATCH (node)-[:BELONGS_TO]->(doc:Document)
            RETURN node {
                .*,
                parent_title:           coalesce(parent.title, doc.title),
                parent_doc_number:      coalesce(parent.document_number, doc.document_number),
                parent_url:             coalesce(parent.url, doc.url),
                doc_effective_date:     doc.effective_date,
                doc_issuing_authority:  doc.issuing_authority
            } AS metadata
            """

            retriever_obj = QdrantNeo4jRetriever(
                driver=neo4j_driver,
                client=qdrant_client,
                collection_name=os.environ.get("QDRANT_COLLECTION", "legal_hybrid_rag_docs"),
                id_property_neo4j="qdrant_id",
                id_property_external="id",
                retrieval_query=retrieval_query,
            )

            dense_vector = embedder.encode_query_dense(query)
            qdrant_filter = self._build_qdrant_filter(state)

            # Thư truyền filter (hỗ trợ tùy phiên bản neo4j-graphrag)
            try:
                results = retriever_obj.search(
                    query_vector=dense_vector,
                    top_k=top_k,
                    filter=qdrant_filter,
                )
            except TypeError:
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
            logger.warning(f"QdrantNeo4jRetriever enrichment skipped: {e}")
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
