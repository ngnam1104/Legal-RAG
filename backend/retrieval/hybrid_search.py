from typing import List, Dict, Optional, Any
from qdrant_client import models
from backend.database.qdrant_client import client
from backend.models.embedder import embedder
from backend.models.reranker import reranker
import os
from backend.database.neo4j_client import get_neo4j_driver
import re
import time
from types import SimpleNamespace

try:
    from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
except ImportError:
    QdrantNeo4jRetriever = None

class ScoredPointMock(dict):
    """Giả lập ScoredPoint để hỗ trợ cả hit.get() và hit.score"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

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
        self.collection_name = collection_name or os.environ.get("QDRANT_COLLECTION", "legal_hybrid_rag_docs")
        self.client = client
        self.hybrid_encoder = embedder
        self.reranker = reranker
        self.hot_sectors = []

    # =========================================================================
    # BLOCK 1: SESSION-PRIORITY RETRIEVAL (Ưu tiên File Upload)
    # Beat 1: Truy vấn Qdrant filter session_id → nếu score > threshold → dùng
    # Beat 2: Fallback mở rộng toàn bộ kho Vector DB
    # =========================================================================
    def search_by_session(
        self,
        session_id: str,
        query: str,
        threshold_upload: float = 0.45,
        top_k: int = 10,
        use_rerank: bool = True,
        **kwargs
    ) -> list:
        """
        Định tuyến Tra cứu ưu tiên File Upload (2-Beat Routing).

        Beat 1: Truy vấn Qdrant với bộ lọc session_id.
                 Nếu điểm số score vượt ngưỡng threshold_upload → return ngay.
        Beat 2: Nếu không chunk nào qua ngưỡng → Fallback mở rộng toàn kho.

        Returns:
            tuple: (hits: List[Dict], source: str)
                   source = "upload" nếu Beat 1 thành công, "global" nếu Fallback.
        """
        import logging
        logger = logging.getLogger("hybrid_search")

        if not session_id or not query:
            logger.warning("  [Session Search] Missing session_id or query. Skipping.")
            return [], "global"

        try:
            # --- BEAT 1: Truy vấn chỉ trong session upload ---
            session_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            )

            dense_query = self.hybrid_encoder.encode_query_dense(query)

            session_hits = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_query,
                using="dense",
                query_filter=session_filter,
                limit=top_k,
                with_payload=True,
            ).points

            if session_hits:
                # Kiểm tra điểm số cao nhất
                best_score = max(float(h.score) for h in session_hits)
                logger.info(
                    f"  [Session Search] Beat 1: Found {len(session_hits)} chunks "
                    f"in session '{session_id}', best_score={best_score:.4f}, "
                    f"threshold={threshold_upload}"
                )

                if best_score >= threshold_upload:
                    # Trích xuất kết quả đạt ngưỡng
                    qualified = []
                    for hit in session_hits:
                        if float(hit.score) >= threshold_upload * 0.7:  # Lấy cả chunk gần ngưỡng
                            payload = hit.payload or {}
                            qualified.append({
                                "id": hit.id,
                                "score": float(hit.score),
                                "payload": payload,
                                "document_number": payload.get("document_number", ""),
                                "article_ref": payload.get("article_ref", ""),
                                "title": payload.get("title", ""),
                                "text": payload.get("expanded_context_text")
                                        or payload.get("chunk_text")
                                        or payload.get("text", ""),
                                "chunk_id": payload.get("chunk_id", ""),
                                "is_appendix": payload.get("is_appendix", False),
                                "url": payload.get("url", ""),
                                "source": "upload",
                            })
                    logger.info(
                        f"  [Session Search] ✅ Beat 1 SUCCESS: {len(qualified)} chunks qualified."
                    )
                    return qualified, "upload"

                logger.info(
                    f"  [Session Search] Beat 1 below threshold ({best_score:.4f} < {threshold_upload}). "
                    f"Falling back to global search..."
                )

            else:
                logger.info(
                    f"  [Session Search] Beat 1: No chunks found for session '{session_id}'. "
                    f"Falling back to global search..."
                )

        except Exception as e:
            logger.error(f"  [Session Search] Beat 1 error: {e}. Falling back to global search...")

        # --- BEAT 2: Fallback toàn kho Vector DB ---
        try:
            global_hits = self.search(
                query=query,
                limit=top_k,
                expand_context=True,
                use_rerank=use_rerank,
                **kwargs
            )
            logger.info(
                f"  [Session Search] Beat 2 (Global Fallback): {len(global_hits)} hits."
            )
            return global_hits, "global"

        except Exception as e:
            logger.error(f"  [Session Search] Beat 2 error: {e}. Returning empty.")
            return [], "global"

    def _build_base_conditions(
        self,
        query: str,
        legal_type: Optional[str] = None,
        doc_number: Optional[str] = None,
        include_inactive: bool = False,
        article_ref: Optional[str] = None,
    ) -> List:
        """Tạo danh sách điều kiện lọc cơ bản (KHÔNG bao gồm is_appendix)."""
        must_conditions = []
        if not include_inactive:
            must_conditions.append(models.FieldCondition(key="is_active", match=models.MatchValue(value=True)))
        if legal_type:
            must_conditions.append(models.FieldCondition(key="legal_type", match=models.MatchValue(value=legal_type)))
        if doc_number:
            condition = models.Filter(
                should=[
                    models.FieldCondition(key="document_number", match=models.MatchText(text=doc_number)),
                    models.FieldCondition(key="title", match=models.MatchText(text=doc_number))
                ]
            )
            must_conditions.append(condition)
        # Removed article_ref hard filter, it should be a should condition
        return must_conditions

    def build_filter(
        self,
        query: str,
        is_appendix: Optional[bool] = None,
        legal_type: Optional[str] = None,
        doc_number: Optional[str] = None,
        include_inactive: bool = False,
        article_ref: Optional[str] = None,
    ):
        must_conditions = self._build_base_conditions(query, legal_type, doc_number, include_inactive, article_ref)
        should_conditions = []

        if is_appendix is not None:
            must_conditions.append(models.FieldCondition(key="is_appendix", match=models.MatchValue(value=is_appendix)))

        if article_ref:
            should_conditions.append(models.FieldCondition(key="article_ref", match=models.MatchText(text=article_ref)))

        for sector in detect_sector_hints(query, self.hot_sectors):
            should_conditions.append(models.FieldCondition(key="legal_sectors", match=models.MatchValue(value=sector)))
            should_conditions.append(models.FieldCondition(key="title", match=models.MatchText(text=sector)))

        if not must_conditions and not should_conditions:
            return None
        return models.Filter(must=must_conditions or None, should=should_conditions or None)

    # =========================================================================
    # CHIẾN THUẬT 1: TIERED PREFETCH (Main Content 15 + Appendix 5)
    # 4 luồng Prefetch song song, gộp bằng RRF trong 1 lần gọi API duy nhất.
    # =========================================================================
    def broad_retrieve(
        self,
        query: str,
        top_k: int = 15,
        is_appendix: Optional[bool] = None,
        legal_type: Optional[str] = None,
        doc_number: Optional[str] = None,
        include_inactive: bool = False,
        main_limit: int = 10,
        appendix_limit: int = 5,
        article_ref: Optional[str] = None,
    ):
        dense_query = self.hybrid_encoder.encode_query_dense(query)
        sparse_query = self.hybrid_encoder.encode_query_sparse(query)

        # Kiểm tra xem collection có hỗ trợ sparse vector không
        collection_info = self.client.get_collection(self.collection_name)
        has_sparse = "sparse" in (collection_info.config.params.vectors or {})
        if not has_sparse:
            # Nếu dùng multivector (với vector chính mang tên 'dense' nhưng config.params.vectors là dict)
            # hoặc fallback nếu cấu trúc config khác
             try:
                 vectors_config = collection_info.config.params.vectors
                 if hasattr(vectors_config, "keys"):
                     has_sparse = "sparse" in vectors_config.keys()
             except:
                 pass

        if not has_sparse:
            # DIAGNOSTIC: In thử vector để kiểm tra lỗi đầu vào
            v_sample = dense_query[:5] if dense_query else []
            print(f"       DEBUG: Dense Query Sample: {v_sample}")
            
            # OPTIMIZATION: Nếu chỉ có dense, gọi trực tiếp query_points không qua RRF để tránh lỗi/overhead
            if is_appendix is not None or article_ref is not None:
                query_filter = self.build_filter(query, is_appendix=is_appendix, legal_type=legal_type,
                                                 doc_number=doc_number, include_inactive=include_inactive, article_ref=article_ref)
                raw_hits = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query,
                    using="dense",
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True
                ).points
            else:
                # Tiered fallback: Vẫn ưu tiên Nội dung chính qua limit, nhưng gộp kết quả
                base_conditions = self._build_base_conditions(query, legal_type, doc_number, include_inactive, article_ref)
                main_filter = models.Filter(must=base_conditions + [models.FieldCondition(key="is_appendix", match=models.MatchValue(value=False))])
                
                # Tìm mẻ chính trước
                main_hits = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query, using="dense", query_filter=main_filter, limit=main_limit, with_payload=True
                ).points
                
                # FALLBACK: Nếu tiered filter trả 0 hits (dữ liệu cũ thiếu is_appendix),
                # thử lại KHÔNG filter is_appendix
                if not main_hits:
                    fallback_filter = models.Filter(must=base_conditions) if base_conditions else None
                    main_hits = self.client.query_points(
                        collection_name=self.collection_name,
                        query=dense_query, using="dense",
                        query_filter=fallback_filter,
                        limit=top_k, with_payload=True
                    ).points

                # Nếu thiếu, tìm mẻ phụ (appendix)
                appendix_hits = []
                if len(main_hits) < top_k:
                    appendix_filter = models.Filter(must=base_conditions + [models.FieldCondition(key="is_appendix", match=models.MatchValue(value=True))])
                    appendix_hits = self.client.query_points(
                        collection_name=self.collection_name,
                        query=dense_query, using="dense", query_filter=appendix_filter, limit=appendix_limit, with_payload=True
                    ).points
                raw_hits = main_hits + appendix_hits

            # CRITICAL FALLBACK: Nếu vẫn là 0 hits và có doc_number, thử tìm chính xác theo metadata (ID-based retrieval)
            if not raw_hits and doc_number:
                print(f"       ⚠️ [Retrieve] Vector mismatch suspected. Attempting metadata scroll for: {doc_number}")
                # Dùng MatchText để linh hoạt với hậu tố /TT-BYT...
                must_cond = [models.FieldCondition(key="document_number", match=models.MatchText(text=doc_number))]
                if not include_inactive:
                    must_cond.append(models.FieldCondition(key="is_active", match=models.MatchValue(value=True)))
                # KHÔNG filter thêm article_ref hay is_appendix ở đây để lấy TOÀN BỘ document
                # Kể cả phụ lục cuối văn bản (vd: 168 trạm y tế = hàng chục chunks cuối)
                
                # CRITICAL FALLBACK: thường doc có rất nhiều phụ lục (ví dụ 168 trạm y tế = 60+ chunks)
                # tăng limit lên 200 để luôn bắt được toàn bộ
                scroll_limit = 200
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(must=must_cond),
                    with_payload=True,
                    limit=scroll_limit
                )
                if points:
                    # Sort: ưu tiên is_appendix=False trước (nội dung chính), sau đó appendix theo chunk order
                    points_sorted = sorted(
                        points,
                        key=lambda p: (
                            1 if (p.payload or {}).get('is_appendix') else 0,
                            parse_chunk_order(str((p.payload or {}).get('chunk_id', '')))
                        )
                    )
                    print(f"       ✅ [Retrieve] Metadata scroll found {len(points_sorted)} matching chunks!")
                    # Map Record objects to a compatible format (Mock ScoredPoint)
                    raw_hits = [ScoredPointMock(id=p.id, payload=p.payload, score=1.0) for p in points_sorted]
        else:
            # ---- TIERED PREFETCH RRF: Chạy khi có đầy đủ bộ đôi Dense/Sparse ----
            if is_appendix is not None or article_ref is not None:
                query_filter = self.build_filter(query, is_appendix=is_appendix, legal_type=legal_type,
                                                 doc_number=doc_number, include_inactive=include_inactive, article_ref=article_ref)
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
            else:
                base_conditions = self._build_base_conditions(query, legal_type, doc_number, include_inactive, article_ref)
                main_filter = models.Filter(must=base_conditions + [models.FieldCondition(key="is_appendix", match=models.MatchValue(value=False))])
                appendix_filter = models.Filter(must=base_conditions + [models.FieldCondition(key="is_appendix", match=models.MatchValue(value=True))])
                
                main_prefetch_limit = max(main_limit * 4, 30)
                appendix_prefetch_limit = max(appendix_limit * 4, 15)

                raw_hits = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(query=dense_query, using="dense", limit=main_prefetch_limit, filter=main_filter),
                        models.Prefetch(query=sparse_query, using="sparse", limit=main_prefetch_limit, filter=main_filter),
                        models.Prefetch(query=dense_query, using="dense", limit=appendix_prefetch_limit, filter=appendix_filter),
                        models.Prefetch(query=sparse_query, using="sparse", limit=appendix_prefetch_limit, filter=appendix_filter),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=main_limit + appendix_limit,
                    with_payload=True,
                ).points

        # Dedup
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

    # =========================================================================
    # CHIẾN THUẬT 4: WINDOW RETRIEVAL cho Phụ lục (Small-to-Big)
    # Phụ lục bị cắt cứ mỗi 2000 ký tự, không theo Điều/Khoản.
    # Dùng document_id + reference_citation + chunk_index để lấy chunk liền kề.
    # =========================================================================
    def _window_retrieve_appendix(self, payload: Dict[str, Any], window_size: int = 1) -> List[Dict]:
        """Lấy chunk liền trước/sau cho phụ lục bị cắt ngang."""
        document_id = payload.get("document_id")
        chunk_index = payload.get("chunk_index")
        ref_citation = payload.get("reference_citation", "")

        if not document_id or chunk_index is None:
            return []

        # Tìm chunk i-1 và i+1 với CÙNG document_id
        target_indices = list(range(max(1, chunk_index - window_size), chunk_index + window_size + 1))
        target_indices = [i for i in target_indices if i != chunk_index]  # Bỏ chính nó

        if not target_indices:
            return []

        must = [
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            models.FieldCondition(key="is_appendix", match=models.MatchValue(value=True)),
        ]

        neighbor_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must),
            with_payload=True,
            with_vectors=False,
            limit=200,
        )

        # Lọc chỉ lấy chunk có chunk_index liền kề VÀ cùng reference_citation
        neighbors = []
        for p in neighbor_points:
            if not p.payload:
                continue
            p_idx = p.payload.get("chunk_index")
            p_ref = p.payload.get("reference_citation", "")
            # CHỐT CHẶN: Chỉ ghép nếu CÙNG document_id + CÙNG reference_citation
            if p_idx in target_indices and p_ref == ref_citation:
                neighbors.append(p.payload)

        # Sắp xếp theo chunk_index
        neighbors.sort(key=lambda x: x.get("chunk_index", 0))
        return neighbors

    def retrieve_specific_reference(self, document_number: str, ref_id: str) -> List[Dict]:
        """
        Truy xuất chính xác một Điều khoản hoặc Phụ lục dựa trên số hiệu văn bản và mã định danh.
        Dùng cho cơ chế Recursive Retrieval.
        """
        if not document_number or not ref_id:
            return []

        # 1. Xây dựng filter (ưu tiên article_ref hoặc reference_citation)
        must = [
            models.FieldCondition(key="document_number", match=models.MatchValue(value=document_number))
        ]
        
        # Thử khớp với article_ref HOẶC reference_citation (cho phụ lục)
        should = [
            models.FieldCondition(key="article_ref", match=models.MatchText(text=ref_id)),
            models.FieldCondition(key="reference_citation", match=models.MatchValue(value=ref_id)),
        ]
        
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must, should=should),
            with_payload=True,
            limit=20, # Thường 1 Điều/Phụ lục không quá 20 chunks
        )
        
        if not points:
            return []
            
        results = []
        for p in points:
            payload = p.payload
            results.append({
                "id": p.id,
                "score": 1.0, # Exact match
                "document_number": payload.get("document_number", document_number),
                "article_ref": payload.get("article_ref", ref_id),
                "title": payload.get("title", ""),
                "text": payload.get("expanded_context_text") or payload.get("chunk_text") or payload.get("text") or payload.get("content") or "",
                "reference_citation": payload.get("reference_citation", ""),
                "is_appendix": payload.get("is_appendix", False),
                "url": payload.get("url", ""),
                "chunk_id": payload.get("chunk_id", "")
            })
            
        return results

    def deactivate_document_by_number(self, document_number: str) -> int:
        """
        Tìm tất cả point có `document_number` và cập nhật payload `is_active` = False.
        Trả về số lượng chunks đã được vô hiệu hoá.
        """
        if not document_number:
            return 0
            
        must = [
            models.FieldCondition(key="document_number", match=models.MatchValue(value=document_number)),
            models.FieldCondition(key="is_active", match=models.MatchValue(value=True))
        ]
        
        # Scroll để lấy tất cả ID của document
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must),
            with_payload=False,
            limit=10000,
        )
        
        if not points:
            return 0
            
        point_ids = [p.id for p in points]
        
        # Cập nhật Payload
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"is_active": False},
            points=point_ids
        )
        
        return len(point_ids)

    def expand_context(self, refined_hits: List[Dict[str, Any]], max_neighbors: int = 10):
        """
        Small-to-Big Retrieval:
        - NỘI DUNG CHÍNH: Gộp toàn bộ các Khoản trong cùng Điều (article_id).
        - PHỤ LỤC: Window Retrieval lấy chunk liền kề (i-1, i+1) cùng reference_citation.
        """
        expanded_results = []
        for item in refined_hits:
            payload = item.get("payload", {})
            article_id = payload.get("article_ref")
            document_id = payload.get("document_id")
            is_appendix = payload.get("is_appendix", False)

            def extract_unit(p):
                ct = p.get("chunk_text", "")
                markers = ["[NỘI DUNG", "[NOI DUNG DIEU/KHOAN]", "[NOI DUNG]", "[PHAN "]
                for marker in markers:
                    if marker in ct:
                        return ct.split(marker)[-1].strip().lstrip("]").strip()
                return ct.strip()

            # --- CHIẾN THUẬT 4: Window Retrieval cho Phụ lục ---
            if is_appendix:
                expanded = dict(item)
                expanded_payload = dict(payload)
                neighbors = self._window_retrieve_appendix(payload, window_size=1)

                if neighbors:
                    # Gộp nội dung: chunk trước + chunk hiện tại + chunk sau
                    window_parts = []
                    for np in neighbors:
                        np_idx = np.get("chunk_index", "?")
                        window_parts.append(f"[Phần {np_idx}]\n{extract_unit(np)}")

                    # Chèn chunk hiện tại vào đúng vị trí
                    current_idx = payload.get("chunk_index", 0)
                    current_part = f"[Phần {current_idx} - KẾT QUẢ TÌM KIẾM]\n{extract_unit(payload)}"

                    all_parts = []
                    inserted = False
                    for np in neighbors:
                        np_idx = np.get("chunk_index", 0)
                        if not inserted and np_idx > current_idx:
                            all_parts.append(current_part)
                            inserted = True
                        all_parts.append(f"[Phần {np_idx}]\n{extract_unit(np)}")
                    if not inserted:
                        all_parts.append(current_part)

                    expanded_payload["expanded_context_text"] = "\n\n".join(all_parts)
                    expanded_payload["neighbor_chunks"] = [{"chunk_id": np.get("chunk_id"), "chunk_index": np.get("chunk_index")} for np in neighbors]
                else:
                    expanded_payload["expanded_context_text"] = extract_unit(payload)
                    expanded_payload["neighbor_chunks"] = []

                expanded["payload"] = expanded_payload
                expanded_results.append(expanded)
                continue

            # --- NỘI DUNG CHÍNH: Small-to-Big qua article_id ---
            if not article_id:
                expanded = dict(item)
                expanded_payload = dict(payload)
                expanded_payload["expanded_context_text"] = extract_unit(payload)
                expanded_payload["neighbor_chunks"] = []
                expanded["payload"] = expanded_payload
                expanded_results.append(expanded)
                continue

            must = [models.FieldCondition(key="article_ref", match=models.MatchValue(value=article_id))]
            if document_id:
                must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))

            neighbor_points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must),
                with_payload=True,
                with_vectors=False,
                limit=100,
            )

            neighbor_payloads = [p.payload for p in neighbor_points if p.payload]
            neighbor_payloads.sort(key=lambda x: parse_chunk_order(str(x.get("chunk_id", ""))))

            if len(neighbor_payloads) > max_neighbors:
                neighbor_payloads = neighbor_payloads[:max_neighbors]

            expanded_context_parts = []
            for np in neighbor_payloads:
                ref = np.get("reference_tag") or np.get("reference_citation") or "N/A"
                unit = extract_unit(np)
                expanded_context_parts.append(f"[{ref}]\n{unit}")

            expanded_context_text = "\n\n".join(expanded_context_parts).strip()

            expanded = dict(item)
            expanded_payload = dict(payload)
            expanded_payload["neighbor_chunks"] = [{"chunk_id": np.get("chunk_id"), "reference_tag": np.get("reference_tag")} for np in neighbor_payloads]
            expanded_payload["expanded_context_text"] = expanded_context_text
            expanded["payload"] = expanded_payload
            expanded_results.append(expanded)

        return expanded_results

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
        is_appendix: Optional[bool] = None,
        legal_type: Optional[str] = None,
        doc_number: Optional[str] = None,
        expand_context: bool = True,
        max_neighbors: int = 8,
        use_rerank: bool = True,
        include_inactive: bool = False,
        article_ref: Optional[str] = None,
    ) -> List[Dict]:
        """Thực hiện Hybrid Search: Tiered Prefetch -> (Optional) Reranking -> Context Expansion."""
        if use_rerank:
            broad_top_k = 15
            rerank_top_l = 8
        else:
            broad_top_k = 25
            rerank_top_l = 8

        if limit is not None:
            rerank_top_l = limit

        t0 = time.perf_counter()
        # Khi có doc_number cụ thể, tăng broad_top_k để bắt đủ tất cả chunks (kể cả Phụ lục cuối)
        actual_broad_top_k = 60 if doc_number else broad_top_k
        broad_hits = self.broad_retrieve(
            query, top_k=actual_broad_top_k,
            main_limit=20,
            appendix_limit=5,
            is_appendix=is_appendix, legal_type=legal_type,
            doc_number=doc_number, include_inactive=include_inactive,
            article_ref=article_ref
        )

        t1 = time.perf_counter()
        if use_rerank:
            reranked_hits = self.reranker.rerank_candidates(query=query, candidates=broad_hits, top_k=rerank_top_l)
            print(f"       ✅ [Reranker] Đã re-score thành công {len(reranked_hits)} candidates trong {time.perf_counter() - t1:.2f}s")
        else:
            reranked_hits = sorted(broad_hits, key=lambda x: x.get('rrf_score', 0), reverse=True)[:rerank_top_l]
            for item in reranked_hits:
                item['score'] = item.get('rrf_score', 0)
                item['rerank_score'] = item.get('rrf_score', 0)

        # --- DOCUMENT TYPOLOGY BOOST (Rank Optimization) ---
        # Ưu tiên các văn bản cấp cao (Nghị định, Luật) hơn các văn bản hành chính (Quyết định, Thông tư)
        # giúp hệ thống trích dẫn đúng nguồn gốc thay vì tài liệu phái sinh.
        for item in reranked_hits:
            p = item.get("payload", {})
            l_type = str(p.get("legal_type", "")).upper()
            boost = 0.0
            
            # Boost cho Nghị định (NĐ-CP) và Luật (L) - Rất nhỏ để chỉ làm tie-breaker
            if "NGHỊ ĐỊNH" in l_type or l_type.endswith("NĐ-CP") or l_type == "LUẬT":
                boost += 0.005
            elif "NGHỊ QUYẾT" in l_type:
                boost += 0.003
            
            # Giảm ưu tiên nhẹ cho Phụ lục để ưu tiên nội dung chính nếu có cùng relevance
            if p.get("is_appendix"):
                boost -= 0.002
                
            item['rerank_score'] = item.get('rerank_score', 0.0) + boost
            item['score'] = item['rerank_score']

        # Re-sort sau khi áp dụng boost
        reranked_hits = sorted(reranked_hits, key=lambda x: x.get('rerank_score', 0.0), reverse=True)

        t2 = time.perf_counter()
        
        # --- CRITICAL RECOVERY: Nếu 0 hits, thử tìm theo số hiệu trích xuất bằng Regex từ Query ---
        if not reranked_hits:
            # Fix 1.1: Regex bắt đầy đủ hậu tố cơ quan (-BYT, -NĐ-CP, -BCT...)
            # (?i) = case-insensitive để bắt cả chữ thường từ LLM
            doc_ids = re.findall(r"(?i)(\d+/\d{4}/(?:TT|NĐ|NĐ-CP|QH|L|QĐ|NQ)(?:-[A-ZĐa-zđÀ-ỹ]+)*)", query)
            if not doc_ids:
                 # Thử format ngắn hơn (chỉ số hiệu/năm)
                 doc_ids = re.findall(r"(\d+/\d{4})", query)
            
            # Chuẩn hóa chữ hoa cho hậu tố
            doc_ids = [d.upper() if '/' in d else d for d in doc_ids]
            unique_doc_ids = sorted(list(set(doc_ids)))
            temp_hits = []
            seen_ids = set()

            if unique_doc_ids:
                print(f"       ⚠️ [Retrieve] 0 vector hits. Emergency Regex found: {unique_doc_ids}. Scrolling metadata...")
                for doc_id in unique_doc_ids:
                    # Fix 1.2 & 6B: Tăng limit=20 để bao phủ nhiều Điều, thêm is_active filter
                    scroll_must = [models.FieldCondition(key="document_number", match=models.MatchText(text=doc_id))]
                    if not include_inactive:
                        scroll_must.append(models.FieldCondition(key="is_active", match=models.MatchValue(value=True)))
                    
                    points, _ = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=models.Filter(must=scroll_must),
                        limit=50,  # Tăng từ 20 lên 50 để bao phủ tất cả phiếu có thể
                        with_payload=True
                    )
                    if points:
                        # Fix 1.2: Sắp xếp theo article_ref để ưu tiên đầy đủ các Điều
                        points_sorted = sorted(points, key=lambda p: (p.payload or {}).get("article_ref", "ZZZ"))
                        print(f"       ✅ [Retrieve] Emergency scroll recovered {len(points_sorted)} chunks for {doc_id}!")
                        for p in points_sorted:
                            if p.id not in seen_ids:
                                temp_hits.append(ScoredPointMock(id=p.id, payload=p.payload, score=1.0))
                                seen_ids.add(p.id)
            
            if temp_hits:
                reranked_hits = temp_hits

        # --- KEYWORD RESCUE: Sau rerank, kiểm tra lại broad_hits để bắt chunk bị bỏ sót ---
        # Trường hợp điển hình: bảng phụ lục chứa tên riêng (xã Nghĩa Thành) không match embedding
        # nhưng lại có keyword exact trong chunk_text
        if len(broad_hits) > len(reranked_hits) and broad_hits:
            # Trích xuất các từ "nặng" từ query (tên riêng, địa danh > 3 ký tự)
            import unicodedata
            query_words = [w.strip() for w in re.split(r'[\s,;.]+', query) if len(w.strip()) > 3]
            reranked_chunk_ids = {
                str(item.get("payload", {}).get("chunk_id") or item.get("id"))
                for item in reranked_hits
            }
            rescued = []
            for item in broad_hits:
                cid = str(item.get("payload", {}).get("chunk_id") or item.get("id"))
                if cid in reranked_chunk_ids:
                    continue
                ct = item.get("payload", {}).get("chunk_text", "")
                # Tìm chunk có ít nhất 2 keyword quan trọng từ query
                matches = sum(1 for w in query_words if w in ct)
                if matches >= 2:
                    item_clone = dict(item)
                    item_clone["score"] = item_clone.get("rrf_score", 0)
                    item_clone["rerank_score"] = item_clone.get("rrf_score", 0) + (matches * 0.1) # Boost dựa trên số match
                    item_clone["match_count"] = matches
                    rescued.append(item_clone)
            if rescued:
                # Sắp xếp rescued theo số match giảm dần
                rescued = sorted(rescued, key=lambda x: x.get("match_count", 0), reverse=True)
                print(f"       🔑 [Keyword Rescue] Injecting {len(rescued)} missed chunks back into context")
                reranked_hits = reranked_hits + rescued[:10]  # Tăng lên 10 để tránh mất chunk ở cuối mảng

        
        if expand_context:
            final_hits = self.expand_context(refined_hits=reranked_hits, max_neighbors=max_neighbors)
        else:
            final_hits = reranked_hits
        t3 = time.perf_counter()

        formatted_results = []
        for idx, item in enumerate(final_hits, start=1):
            p = item["payload"]
            formatted_results.append({
                "id": item["id"],
                "score": item.get("rerank_score", item.get("rrf_score", 0)),
                "document_number": p.get("document_number", ""),
                "article_ref": p.get("article_ref", ""),
                "title": p.get("title", ""),
                "text": p.get("expanded_context_text") or p.get("chunk_text") or p.get("text") or p.get("content") or "",
                "breadcrumb_path": p.get("breadcrumb_path", ""),
                "reference_citation": p.get("reference_citation", ""),
                "is_appendix": p.get("is_appendix", False),
                "is_active": p.get("is_active", True),
                "legal_type": p.get("legal_type", ""),
                "issuance_date": p.get("issuance_date", ""),
                "url": p.get("url", ""),
                "base_laws": p.get("base_laws") or [r.get("basis_line") for r in p.get("legal_basis_refs", []) if r.get("basis_line")],
                "chunk_id": p.get("chunk_id", ""),
                "issuing_authority": p.get("issuing_authority", ""),
                "effective_date": p.get("effective_date", ""),
                "legal_sectors": p.get("legal_sectors", []),
                "legal_basis_refs": p.get("legal_basis_refs", []),
                "pipeline_metrics": {
                    "broad_count": len(broad_hits),
                    "rerank_count": len(reranked_hits),
                    "final_count": len(final_hits),
                    "total_seconds": t3 - t0
                }
            })

        return formatted_results

    def graph_search(
        self,
        query: str,
        top_k: int = 5,
        explore_depth: int = 1
    ) -> List[Dict]:
        """
        Entity-Aware GraphRAG Retrieval:
        1. Hybrid Search → danh sách chunk (qdrant_id).
        2. Neo4j Query 1 — doc-level relations (AMENDS, BASED_ON...).
        3. Neo4j Query 2 — free-form entities (HAS_ENTITY → Organization, Fee...).
        4. Neo4j Query 3 — dynamic node relations (RESPONSIBLE_FOR, SIGNED_BY...).
        5. Inject cả 3 loại dữ liệu vào text context cho LLM.
        """
        driver = get_neo4j_driver()
        if not driver:
            print("⚠️ Không khởi tạo được Neo4j Driver. Fallback sang Vector search.")
            return self.search(query, limit=top_k)

        base_hits = self.search(query, limit=top_k)
        if not base_hits:
            return []

        chunk_ids = [hit.get("chunk_id") or str(hit.get("id", "")) for hit in base_hits]

        # ── Query 1: Doc-level relations ──
        doc_rel_query = """
        UNWIND $chunk_ids AS cid
        MATCH (c) WHERE c.qdrant_id = cid OR c.id = cid
        OPTIONAL MATCH (c)-[:BELONGS_TO|PART_OF*1..3]->(doc:Document)
        OPTIONAL MATCH (doc)-[dr]->(other_doc:Document)
        WHERE type(dr) IN ['AMENDS','REPLACES','REPEALS','BASED_ON','GUIDES','APPLIES','ISSUED_WITH','ASSIGNS','CORRECTS']
        RETURN cid AS chunk_id,
               collect(DISTINCT {
                 rel_type: type(dr),
                 source: doc.document_number,
                 target: other_doc.document_number,
                 chunk_text: dr.chunk_text
               }) AS doc_relations
        """

        # ── Query 2: Free-form entities ──
        entity_query = """
        UNWIND $chunk_ids AS cid
        MATCH (c) WHERE c.qdrant_id = cid OR c.id = cid
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
        WHERE e.name IS NOT NULL
        RETURN cid AS chunk_id,
               collect(DISTINCT {label: labels(e)[0], name: e.name}) AS entities
        """

        # ── Query 3: Dynamic node relations (RESPONSIBLE_FOR, SIGNED_BY...) ──
        node_rel_query = """
        UNWIND $chunk_ids AS cid
        MATCH (c) WHERE c.qdrant_id = cid OR c.id = cid
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(src_ent)-[nr]->(tgt)
        WHERE nr IS NOT NULL AND type(nr) NOT IN ['HAS_ENTITY']
        RETURN cid AS chunk_id,
               collect(DISTINCT {
                 relationship: type(nr),
                 source_type: labels(src_ent)[0],
                 source_node: src_ent.name,
                 target_type: labels(tgt)[0],
                 target_node: tgt.name,
                 chunk_text: nr.chunk_text
               }) AS node_relations
        """

        try:
            with driver.session() as session:
                doc_rel_map  = {r["chunk_id"]: r for r in session.run(doc_rel_query, chunk_ids=chunk_ids).data()}
                entity_map   = {r["chunk_id"]: r for r in session.run(entity_query, chunk_ids=chunk_ids).data()}
                node_rel_map = {r["chunk_id"]: r for r in session.run(node_rel_query, chunk_ids=chunk_ids).data()}

            for hit in base_hits:
                cid = hit.get("chunk_id") or str(hit.get("id", ""))
                doc_rels   = [r for r in (doc_rel_map.get(cid) or {}).get("doc_relations", []) if r.get("rel_type")]
                entities   = [e for e in (entity_map.get(cid) or {}).get("entities", []) if e.get("name")]
                node_rels  = [n for n in (node_rel_map.get(cid) or {}).get("node_relations", []) if n.get("relationship")]

                injected = ""
                if entities:
                    # Group by label
                    by_label = {}
                    for e in entities:
                        by_label.setdefault(e["label"], []).append(e["name"])
                    ent_str = "; ".join(f"{lbl}: {', '.join(names)}" for lbl, names in by_label.items())
                    injected += f"\n[THỰC THỂ]: {ent_str}"

                if doc_rels:
                    rel_str = "; ".join(
                        f"{r['source']} --{r['rel_type']}--> {r['target']}"
                        for r in doc_rels if r.get("source") and r.get("target")
                    )
                    injected += f"\n[QUAN HỆ VĂN BẢN]: {rel_str}"

                if node_rels:
                    nr_str = "; ".join(
                        f"{n['source_node']} --{n['relationship']}--> {n['target_node']}"
                        for n in node_rels if n.get("source_node") and n.get("target_node")
                    )
                    injected += f"\n[QUAN HỆ THỰC THỂ]: {nr_str}"

                if injected:
                    hit["text"] = hit.get("text", "") + "\n" + injected

                hit["graph_entities"]      = entities
                hit["graph_doc_relations"] = doc_rels
                hit["graph_node_relations"]= node_rels

            return base_hits

        except Exception as e:
            print(f"Lỗi Graph Search: {e}")
            return base_hits

retriever = HybridRetriever()

