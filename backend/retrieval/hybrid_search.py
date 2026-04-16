from typing import List, Dict, Optional, Any
from qdrant_client import models
from backend.retrieval.vector_db import client
from backend.retrieval.embedder import embedder
from backend.retrieval.reranker import reranker
from backend.config import settings
from backend.retrieval.graph_db import get_neo4j_driver
import re
import time

try:
    from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
except ImportError:
    QdrantNeo4jRetriever = None

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
        self.hot_sectors = []

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
            must_conditions.append(models.FieldCondition(key="document_number", match=models.MatchValue(value=doc_number)))
        if article_ref:
            # Use MatchText for article_ref to allow "Điều 2" to match "Điều 2. Đối tượng áp dụng"
            must_conditions.append(models.FieldCondition(key="article_ref", match=models.MatchText(text=article_ref)))
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

        # Nếu caller CHỈ ĐỊNH cụ thể is_appendix -> fallback về single-tier (backward compat)
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
            # ---- TIERED PREFETCH: 4 luồng song song ----
            base_conditions = self._build_base_conditions(query, legal_type, doc_number, include_inactive, article_ref)

            # Luồng 1+2: NỘI DUNG CHÍNH (is_appendix=false), Limit cao hơn
            main_filter = models.Filter(must=base_conditions + [
                models.FieldCondition(key="is_appendix", match=models.MatchValue(value=False))
            ]) if base_conditions else models.Filter(must=[
                models.FieldCondition(key="is_appendix", match=models.MatchValue(value=False))
            ])

            # Luồng 3+4: PHỤ LỤC (is_appendix=true), Limit thấp hơn
            appendix_filter = models.Filter(must=base_conditions + [
                models.FieldCondition(key="is_appendix", match=models.MatchValue(value=True))
            ]) if base_conditions else models.Filter(must=[
                models.FieldCondition(key="is_appendix", match=models.MatchValue(value=True))
            ])

            main_prefetch_limit = max(main_limit * 4, 30)
            appendix_prefetch_limit = max(appendix_limit * 4, 15)

            raw_hits = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # Luồng 1: Dense cho Nội dung chính (Ưu tiên)
                    models.Prefetch(query=dense_query, using="dense", limit=main_prefetch_limit, filter=main_filter),
                    # Luồng 2: Sparse cho Nội dung chính
                    models.Prefetch(query=sparse_query, using="sparse", limit=main_prefetch_limit, filter=main_filter),
                    # Luồng 3: Dense cho Phụ lục
                    models.Prefetch(query=dense_query, using="dense", limit=appendix_prefetch_limit, filter=appendix_filter),
                    # Luồng 4: Sparse cho Phụ lục (nhạy với số hiệu "Mẫu 01", "PHỤ LỤC I")
                    models.Prefetch(query=sparse_query, using="sparse", limit=appendix_prefetch_limit, filter=appendix_filter),
                ],
                # CHIẾN THUẬT 2: RRF tự nhiên - không hardcode weight
                # Chunk đứng Top 1 ở cả Dense + Sparse sẽ được đẩy lên cao nhất tự động
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=main_limit + appendix_limit,  # Tổng: 20 candidates
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
                "text": payload.get("expanded_context_text") or payload.get("chunk_text", ""),
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
            article_id = payload.get("article_id")
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

            must = [models.FieldCondition(key="article_id", match=models.MatchValue(value=article_id))]
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
        use_rerank: bool = False,
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
        broad_hits = self.broad_retrieve(
            query, top_k=broad_top_k,
            main_limit=20,
            appendix_limit=5,
            is_appendix=is_appendix, legal_type=legal_type,
            doc_number=doc_number, include_inactive=include_inactive,
            article_ref=article_ref
        )

        t1 = time.perf_counter()
        if use_rerank:
            reranked_hits = self.reranker.rerank(query=query, candidates=broad_hits, top_k=rerank_top_l)
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
                "text": p.get("expanded_context_text", p.get("chunk_text", "")),
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
        Sử dụng neo4j_graphrag.retrievers.QdrantNeo4jRetriever để tự động hoá
        quy trình Vector Search -> Graph Traversal (tìm Relationship: AMENDS, REPLACES...)
        """
        if not QdrantNeo4jRetriever:
            print("⚠️ Cần cài đặt `neo4j-graphrag` để dùng QdrantNeo4jRetriever. Fallback sang Vector search.")
            return self.search(query, limit=top_k)

        driver = get_neo4j_driver()
        if not driver:
            print("⚠️ Không khởi tạo được Neo4j Driver. Fallback sang Vector search.")
            return self.search(query, limit=top_k)

        try:
            # Note: We must specify vector_name='dense' since we use Hybrid Qdrant Collection
            from neo4j_graphrag.embeddings import OpenAIEmbeddings # Fallback if we need an embedder interface
            
            # Since neo4j-graphrag natively expects an embedder for its own Qdrant querying 
            # if we don't pass vector directly, we'll configure it. 
            # However QdrantNeo4jRetriever handles vector search out of the box if we provide correct parameters.
            
            retriever = QdrantNeo4jRetriever(
                driver=driver,
                client=self.client,
                collection_name=self.collection_name,
                id_property_external="id",  # ID in Qdrant
                id_property_neo4j="chunk_id" # ID loaded via Chunk Node
            )
            
            # Run GraphRAG Search (This yields nodes augmented with exact old_text/new_text relations)
            records = retriever.search(query_text=query, top_k=top_k)
            
            driver.close()
            
            # Format output compatible with our system
            results = []
            for item in records.items:
                results.append({
                    "id": getattr(item, "id", ""),
                    "score": 1.0,
                    "title": "Graph Retrieval Hit",
                    "text": getattr(item, "content", ""),
                    "chunk_id": getattr(item, "id", "")
                })
            return results
            
        except Exception as e:
            print(f"Lỗi Graph Search: {e}")
            if driver: driver.close()
            return self.search(query, limit=top_k)

retriever = HybridRetriever()
