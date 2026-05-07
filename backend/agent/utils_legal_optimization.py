"""
utils_legal_optimization.py — Advanced Graph_Doc_Fetch Optimization Strategies
================================================================================
Cung cấp 2 chiến lược tối ưu hóa việc fetch missing documents:
  1. Cách 1: Batch Scroll + Rerank (Hybrid Search)
  2. Cách 2: Neo4j Sibling Expansion (Graph-based)
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("legal_optimization")


def fetch_siblings_from_graph(missing_docs: List[str]) -> Dict[str, Any]:
    """
    **Cách 2: Mở rộng ngữ cảnh bằng Neo4j Graph Expansion (TỐI ƯU NHẤT)**
    
    Thay vì kéy dữ liệu từ Qdrant bằng .scroll() (tốn I/O mạng), 
    dùng Neo4j để tìm các chunk "anh em" (sibling) trong cùng Document/Article.
    
    Returns:
        {
            "sibling_chunks": List[Dict] — Chunks từ cùng document (được sắp xếp)
            "sibling_texts": List[str] — Text nội dung đã hợp nhất
            "metadata": Dict — Thông tin document từ Neo4j
        }
    """
    from backend.database.neo4j_client import get_neo4j_driver, run_cypher
    
    if not missing_docs:
        return {"sibling_chunks": [], "sibling_texts": [], "metadata": {}}
    
    driver = get_neo4j_driver()
    if not driver:
        return {"sibling_chunks": [], "sibling_texts": [], "metadata": {}}
    
    # Query Neo4j: Lấy ALL chunks từ missing_docs (document hierarchy)
    # Lấy không chỉ chunk, mà cả article/clause context để có đầy đủ ngữ cảnh
    query = """
    UNWIND $doc_numbers AS doc_num
    MATCH (d:Document {document_number: doc_num})
    OPTIONAL MATCH (art:LegalArticle)-[:BELONGS_TO]->(d)
    OPTIONAL MATCH (c:Chunk)-[:BELONGS_TO|PART_OF*1..2]->(art)
    WHERE c.qdrant_id IS NOT NULL
    RETURN DISTINCT
        d.document_number AS doc_number,
        d.title AS doc_title,
        d.document_toc AS doc_toc,
        art.name AS article_ref,
        c.qdrant_id AS chunk_id,
        c.text AS chunk_text,
        c.chunk_index AS chunk_index
    ORDER BY d.document_number, art.name, c.chunk_index
    LIMIT 100
    """
    
    sibling_chunks = []
    sibling_texts = []
    metadata = {}
    
    try:
        with driver.session() as session:
            for r in session.run(query, doc_numbers=missing_docs).data():
                if not r.get("chunk_id"):
                    continue
                
                # Collect metadata once
                if not metadata:
                    metadata = {
                        "document_number": r.get("doc_number"),
                        "document_title": r.get("doc_title"),
                        "document_toc": r.get("doc_toc"),
                    }
                
                # Build chunk dict
                chunk_dict = {
                    "chunk_id": r.get("chunk_id"),
                    "document_number": r.get("doc_number"),
                    "article_ref": r.get("article_ref"),
                    "text": r.get("chunk_text", ""),
                    "chunk_index": r.get("chunk_index", 0),
                    "source": "neo4j_sibling",
                }
                sibling_chunks.append(chunk_dict)
                
                # Format text: [Article Ref] - Text snippet
                if r.get("chunk_text"):
                    article_label = f"[{r.get('article_ref', 'N/A')}]" if r.get("article_ref") else ""
                    sibling_texts.append(f"{article_label}\n{r.get('chunk_text', '')}\n")
        
        logger.info(
            f"  [Graph Siblings] Fetched {len(sibling_chunks)} sibling chunks from {len(missing_docs)} docs via Neo4j"
        )
        
    except Exception as e:
        logger.warning(f"  [Graph Siblings] Neo4j query failed: {e}")
    
    return {
        "sibling_chunks": sibling_chunks,
        "sibling_texts": sibling_texts,
        "metadata": metadata,
    }


def rerank_and_select_top_k(
    batch_hits: List[Dict[str, Any]],
    reranker_client,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    **Cách 1: Batch Scroll + Rerank để lọc "vàng"**
    
    Sau khi batch scroll trả ~15-20 chunks, dùng Reranker để chấm điểm lại
    và chỉ giữ top K chunks "chất lượng cao".
    
    Args:
        batch_hits: Danh sách chunks từ batch scroll
        reranker_client: Reranker model instance
        query: Original query
        top_k: Số chunks cuối cùng giữ lại
    
    Returns:
        List[Dict] — Top K reranked chunks
    """
    if not batch_hits:
        return []
    
    if len(batch_hits) <= top_k:
        return batch_hits
    
    try:
        # Prepare texts for reranking
        texts = [
            h.get("text") or h.get("payload", {}).get("chunk_text", "") 
            for h in batch_hits
        ]
        
        # Rerank using the reranker model
        reranked_scores = reranker_client.rank(query, texts)
        
        # Sort by rerank scores (descending)
        scored_hits = [
            {**hit, "rerank_score": score}
            for hit, score in zip(batch_hits, reranked_scores)
        ]
        scored_hits.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # Return top K
        result = scored_hits[:top_k]
        logger.info(
            f"  [Rerank Filter] Filtered {len(batch_hits)} → {len(result)} top chunks "
            f"(top score: {result[0].get('rerank_score', 0):.2f})"
        )
        
        return result
        
    except Exception as e:
        logger.warning(f"  [Rerank Filter] Reranking failed: {e}. Returning original hits.")
        return batch_hits[:top_k]


def optimize_batch_fetch(
    batch_hits: List[Dict[str, Any]],
    missing_docs: List[str],
    query: str,
    reranker_client,
    strategy: str = "neo4j",  # "neo4j" | "rerank"
) -> List[Dict[str, Any]]:
    """
    **Dispatcher: Chọn chiến lược tối ưu phù hợp**
    
    Args:
        batch_hits: Chunks từ batch scroll (Qdrant)
        missing_docs: Document numbers cần fetch
        query: Original query
        reranker_client: Reranker model
        strategy: "neo4j" (dùng graph) | "rerank" (dùng reranker) | "hybrid" (cả hai)
    
    Returns:
        List[Dict] — Optimized/filtered chunks
    """
    optimized_results = []
    
    if strategy == "neo4j":
        # **Cách 2**: Fetch sibling chunks từ Neo4j graph
        graph_result = fetch_siblings_from_graph(missing_docs)
        sibling_chunks = graph_result.get("sibling_chunks", [])
        
        # Normalize sibling chunks to same format as batch_hits
        for sc in sibling_chunks[:10]:  # Limit to top 10 siblings
            optimized_results.append({
                "id": sc.get("chunk_id"),
                "chunk_id": sc.get("chunk_id"),
                "document_number": sc.get("document_number"),
                "text": sc.get("text"),
                "score": 0.65,  # Graph-sourced baseline
                "payload": {"chunk_text": sc.get("text")},
                "source": "neo4j_sibling",
            })
    
    elif strategy == "rerank":
        # **Cách 1**: Rerank batch hits và lọc top K
        optimized_results = rerank_and_select_top_k(
            batch_hits=batch_hits,
            reranker_client=reranker_client,
            query=query,
            top_k=5,
        )
    
    elif strategy == "hybrid":
        # **Kết hợp cả hai**: Graph siblings + Reranked batch hits
        # Lấy từ Neo4j
        graph_result = fetch_siblings_from_graph(missing_docs)
        sibling_chunks = graph_result.get("sibling_chunks", [])
        
        # Normalize siblings
        for sc in sibling_chunks[:5]:
            optimized_results.append({
                "id": sc.get("chunk_id"),
                "chunk_id": sc.get("chunk_id"),
                "document_number": sc.get("document_number"),
                "text": sc.get("text"),
                "score": 0.70,  # Graph-sourced boost
                "payload": {"chunk_text": sc.get("text")},
                "source": "neo4j_sibling",
            })
        
        # Rerank batch hits
        reranked_batch = rerank_and_select_top_k(
            batch_hits=batch_hits,
            reranker_client=reranker_client,
            query=query,
            top_k=5,
        )
        optimized_results.extend(reranked_batch)
        
        # Sort by score (descending) and take top 8
        optimized_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        optimized_results = optimized_results[:8]
    
    logger.info(
        f"  [Graph_Doc_Fetch Optimize] Strategy={strategy} → {len(optimized_results)} final chunks"
    )
    
    return optimized_results
