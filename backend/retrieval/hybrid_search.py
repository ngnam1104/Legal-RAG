from typing import List, Dict, Optional, Any
from qdrant_client import models
from backend.retrieval.vector_db import client
from backend.retrieval.embedder import embedder
from backend.retrieval.reranker import reranker
from backend.config import settings
import re
import time

def detect_sector_hints(query: str, hot_sectors: List[str]) -> List[str]:
    lowered = query.lower()
    hits = [sector for sector in hot_sectors if sector.lower() in lowered]
    return hits[:3]

def parse_chunk_order(chunk_id: str):
    if not chunk_id:
        return (10**9, 10**9, "")
    clause_match = re.search(r"::c(\d+)", chunk_id)
    point_match = re.search(r"::p(\d+)", chunk_id)
    clause_idx = int(clause_match.group(1)) if clause_match else 10**9
    point_idx = int(point_match.group(1)) if point_match else 0
    return (clause_idx, point_idx, chunk_id)

class HybridRetriever:
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.client = client
        self.hybrid_encoder = embedder
        self.reranker = reranker
        self.hot_sectors = [] # You might want to pre-load hot sectors here or fetch from db.

    def build_filter(
        self,
        query: str,
        is_appendix: Optional[bool] = None,
        legal_type: Optional[str] = None,
        doc_number: Optional[str] = None,
        include_inactive: bool = False,
    ):
        must_conditions = []
        should_conditions = []
        
        if not include_inactive:
            must_conditions.append(models.FieldCondition(key="is_active", match=models.MatchValue(value=True)))
            
        if is_appendix is not None:
            must_conditions.append(models.FieldCondition(key="is_appendix", match=models.MatchValue(value=is_appendix)))
        if legal_type:
            must_conditions.append(models.FieldCondition(key="legal_type", match=models.MatchValue(value=legal_type)))
        if doc_number:
            must_conditions.append(models.FieldCondition(key="document_number", match=models.MatchValue(value=doc_number)))
        
        # Sector hints logic (disabled if hot_sectors is empty)
        for sector in detect_sector_hints(query, self.hot_sectors):
            should_conditions.append(models.FieldCondition(key="legal_sectors", match=models.MatchValue(value=sector)))
            should_conditions.append(models.FieldCondition(key="title", match=models.MatchText(text=sector)))

        if not must_conditions and not should_conditions:
            return None
        return models.Filter(must=must_conditions or None, should=should_conditions or None)

    def broad_retrieve(
        self,
        query: str,
        top_k: int = 15,
        is_appendix: Optional[bool] = None,
        legal_type: Optional[str] = None,
        doc_number: Optional[str] = None,
        include_inactive: bool = False,
    ):
        query_filter = self.build_filter(query, is_appendix, legal_type, doc_number, include_inactive)
        dense_query = self.hybrid_encoder.encode_query_dense(query)
        sparse_query = self.hybrid_encoder.encode_query_sparse(query)

        prefetch_limit = max(top_k * 4, 30)
        raw_hits = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_query, using="dense", limit=prefetch_limit, filter=query_filter),
                models.Prefetch(query=sparse_query, using="sparse", limit=prefetch_limit, filter=query_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=prefetch_limit,
            with_payload=True,
        ).points

        dedup = []
        seen_chunk_ids = set()
        for hit in raw_hits:
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id") or str(hit.id)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            dedup.append({
                "id": hit.id,
                "payload": payload,
                "rrf_score": float(hit.score),
                "broad_rank": len(dedup) + 1,
            })
            if len(dedup) >= top_k:
                break
        return dedup

    def expand_context(self, refined_hits: List[Dict[str, Any]], max_neighbors: int = 10):
        """
        Small-to-Big Retrieval: Khi một Khoản được tìm thấy, tự động gộp toàn bộ các Khoản 
        trong cùng một Điều để tạo ra ngữ cảnh đầy đủ (Full Article).
        """
        expanded_results = []
        for item in refined_hits:
            payload = item.get("payload", {})
            article_id = payload.get("article_id")
            document_id = payload.get("document_id")
            
            # Logic trích xuất nội dung thực (unit_text) từ chunk_text format
            def extract_unit(p):
                ct = p.get("chunk_text", "")
                markers = ["[NOI DUNG DIEU/KHOAN]", "[NOI DUNG]", "[PHAN "]
                for marker in markers:
                    if marker in ct:
                        return ct.split(marker)[-1].strip()
                return ct.strip()

            if not article_id:
                expanded = dict(item)
                expanded_payload = dict(payload)
                expanded_payload["expanded_context_text"] = extract_unit(payload)
                expanded_payload["neighbor_chunks"] = []
                expanded["payload"] = expanded_payload
                expanded_results.append(expanded)
                continue

            # Truy vấn lấy tất cả các chunks thuộc cùng một Điều
            must = [models.FieldCondition(key="article_id", match=models.MatchValue(value=article_id))]
            if document_id:
                must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))

            neighbor_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must),
                with_payload=True,
                with_vectors=False,
                limit=100, # Một Điều thường không quá 100 khoản/điểm
            )

            neighbor_payloads = [p.payload for p in neighbor_points if p.payload]
            # Sắp xếp các khoản theo thứ tự đúng trong văn bản
            neighbor_payloads.sort(key=lambda x: parse_chunk_order(str(x.get("chunk_id", ""))))

            # Giới hạn số lượng neighbors nếu Điều quá dài (tránh vượt token LLM)
            if len(neighbor_payloads) > max_neighbors:
                neighbor_payloads = neighbor_payloads[:max_neighbors]

            # Gộp nội dung của tất cả các Khoản thành một Điều hoàn chỉnh (Small-to-Big)
            expanded_context_parts = []
            for np in neighbor_payloads:
                ref = np.get("reference_tag") or np.get("reference_citation") or "N/A"
                # Quan trọng: Lấy nội dung thực từ chunk_text
                unit = extract_unit(np)
                expanded_context_parts.append(f"[{ref}]\n{unit}")

            expanded_context_text = "\n\n".join(expanded_context_parts).strip()
            
            expanded = dict(item)
            expanded_payload = dict(payload)
            # Lưu vết các neighbors để FE có thể hiển thị nếu cần
            expanded_payload["neighbor_chunks"] = [{"chunk_id": np.get("chunk_id"), "reference_tag": np.get("reference_tag")} for np in neighbor_payloads]
            expanded_payload["expanded_context_text"] = expanded_context_text
            expanded["payload"] = expanded_payload
            expanded_results.append(expanded)

        return expanded_results

    def search(
        self,
        query: str,
        limit: Optional[int] = None, # Will be ignored for now to enforce the 40/10 vs 15 rule
        is_appendix: Optional[bool] = None,
        legal_type: Optional[str] = None,
        doc_number: Optional[str] = None,
        expand_context: bool = True,
        max_neighbors: int = 8,
        use_rerank: bool = True,
        include_inactive: bool = False,
    ) -> List[Dict]:
        """Thực hiện Hybrid Search: Broad Retrieval -> (Optional) Reranking -> Context Expansion."""
        if use_rerank:
            # Quy tắc mới: Rerank lấy 40 từ RRF, sau đó giữ lại 10 bản ghi chất lượng nhất
            broad_top_k = 40
            rerank_top_l = 10
        else:
            # Quy tắc mới: Không Rerank, lấy thẳng 15 bản ghi tốt nhất từ RRF
            broad_top_k = 15
            rerank_top_l = 15

        # Nếu người dùng truyền limit cụ thể, override rerank_top_l (để tương thích ngược nếy cần)
        if limit is not None:
            rerank_top_l = limit

        t0 = time.perf_counter()
        broad_hits = self.broad_retrieve(query, top_k=broad_top_k, is_appendix=is_appendix, legal_type=legal_type, doc_number=doc_number, include_inactive=include_inactive)
        
        t1 = time.perf_counter()
        if use_rerank:
            reranked_hits = self.reranker.rerank(query=query, candidates=broad_hits, top_k=rerank_top_l)
        else:
            # Tắt Rerank: lấy thẳng kết quả RRF, chỉ sort theo rrf_score
            reranked_hits = sorted(broad_hits, key=lambda x: x.get('rrf_score', 0), reverse=True)[:rerank_top_l]
            for item in reranked_hits:
                item['score'] = item.get('rrf_score', 0)
                item['rerank_score'] = item.get('rrf_score', 0)
        
        t2 = time.perf_counter()
        if expand_context:
            final_hits = self.expand_context(refined_hits=reranked_hits, max_neighbors=max_neighbors)
        else:
            final_hits = reranked_hits
        t3 = time.perf_counter()

        # Format output similar to old retriver for compatibility
        formatted_results = []
        for idx, item in enumerate(final_hits, start=1):
            p = item["payload"]
            formatted_results.append({
                "id": item["id"],
                "score": item.get("rerank_score", item.get("rrf_score", 0)),
                "document_number": p.get("document_number", ""),
                "article_ref": p.get("article_ref", ""),
                "title": p.get("title", ""),
                "text": p.get("expanded_context_text", p.get("chunk_text", "")),
                "breadcrumb_path": p.get("breadcrumb_path", ""),
                "reference_citation": p.get("reference_citation", ""),
                "is_appendix": p.get("is_appendix", False),
                "is_active": p.get("is_active", True),
                "legal_type": p.get("legal_type", ""),
                "issuance_date": p.get("issuance_date", ""),
                "url": p.get("url", ""),
                # Tinh gọn base_laws: ưu tiên payload gộp từ refs nếu có, fallback về base_laws cũ
                "base_laws": p.get("base_laws") or [r.get("basis_line") for r in p.get("legal_basis_refs", []) if r.get("basis_line")],
                "chunk_id": p.get("chunk_id", ""),
                "pipeline_metrics": {
                    "broad_count": len(broad_hits),
                    "rerank_count": len(reranked_hits),
                    "final_count": len(final_hits),
                    "total_seconds": t3 - t0
                }
            })

        return formatted_results

        return formatted_results

retriever = HybridRetriever()
