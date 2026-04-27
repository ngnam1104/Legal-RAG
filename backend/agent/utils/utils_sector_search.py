import json
import re
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from backend.llm.factory import chat_completion
from backend.retrieval.hybrid_search import retriever
from backend.config import settings
from backend.agent.utils.utils_legal_qa import strip_thinking_tags

# =============================================================================
# PROMPTS
# =============================================================================



DOCUMENT_FOCUS_PROMPT = """
FILE_CONTEXT = {file_context}

Bạn là Chuyên gia Phân tích Văn bản Pháp quy.
Nhiệm vụ: Đọc tóm tắt FILE_CONTEXT người dùng tải lên và trích xuất các THÔNG TIN CHỦ ĐẠO để tìm kiếm các văn bản pháp luật liên quan trong CSDL.

BẮT BUỘC TRẢ VỀ JSON trong markdown code block:
```json
{{
    "focus_summary": "Đoạn tóm tắt ngắn gọn nội dung cốt lõi của file (1-2 câu)...",
    "suggested_keywords": "Các từ khóa pháp lý quan trọng để tìm kiếm luật liên quan...",
    "legal_sectors": ["Lĩnh vực 1", "Lĩnh vực 2"]
}}
```"""

RELEVANCE_BATCH_PROMPT = """
QUERY = {query}
DOC_LIST = {doc_list}

Bạn là Người Kiểm Duyệt Pháp lý (Strict Filter).
Mục tiêu: Đọc danh sách văn bản trong DOC_LIST và LOẠI BỎ các văn bản CHẮC CHẮN KHÔNG LIÊN QUAN đến QUERY.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

BƯỚC 1: Suy luận trong thẻ <thinking>
<thinking>
- Chủ đề cốt lõi của truy vấn là gì?
- Duyệt qua từng văn bản:
  + [doc_0] → Liên quan / Không liên quan vì: ...
  + [doc_1] → Liên quan / Không liên quan vì: ...
  + ...
</thinking>

BƯỚC 2: Trả về MỘT MẢNG JSON chứa các ID được GIỮ LẠI trong markdown code block:
```json
["doc_0", "doc_1"]
```
Nếu không có văn bản nào liên quan, trả về [].
"""

EXECUTIVE_SUMMARY_PROMPT = """
HISTORY = {history}
QUERY = {query}
FILE_ANALYSIS = {file_analysis}
TABLE = {table}

Bạn là Chuyên gia Tổng hợp Pháp lý. 
Nhiệm vụ: Viết đoạn "Tổng quan" (Executive Summary) cho báo cáo dựa trên chủ đề QUERY.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI VIẾT VĂN BẢN CUỐI CÙNG.

<thinking>
- Cấu trúc đoạn Tổng quan:
  + Giới thiệu: Nội dung tài liệu người dùng tải lên trong FILE_ANALYSIS (nếu có) và mối liên hệ với hệ thống pháp luật.
  + Thống kê: Tổng số văn bản tìm thấy từ TABLE, phân bố theo loại.
  + Thông tin còn thiếu: Có thiếu loại văn bản nào quan trọng (Nghị định hướng dẫn, Thông tư...) không?
- Nếu KHÔNG TÌM THẤY QUY ĐỊNH TRỰC TIẾP → xác nhận rõ ràng, vẫn tóm tắt nội dung "Tham khảo gần nhất".
- Nếu người dùng HỎI VỀ CĂN CỨ CỦA LƯỢT TRẢ LỜI TRƯỚC: Bỏ qua bảng tổng hợp mới, trực tiếp trích xuất Metadata (Số hiệu, Tên văn bản) từ LỊCH SỬ HỘI THOẠI để thông báo lại cho họ.
</thinking>

Sau khi suy luận xong, chỉ viết đoạn văn Tổng quan (tối đa 150 từ). KHÔNG liệt kê lại bảng."""



# =============================================================================
# 1. UNDERSTAND — Sector Query Planner
# =============================================================================

# --- BLOCK 2: 3 SPECIALIZED RETRIEVAL FUNCTIONS ---

def retrieve_by_sector(sector_name: str) -> List[Dict]:
    """Hàm 1 (Tìm theo Ngành): Chỉ dùng Metadata Filtering, không dùng Vector similarity."""
    from backend.retrieval.hybrid_search import retriever
    from qdrant_client import models
    import logging
    logger = logging.getLogger("sector_search")
    
    logger.info(f"  [Sector Mode 1] Retrieving purely by Sector Filter: {sector_name}")
    try:
        # Sử dụng Qdrant scroll với filter để không dùng vector search
        must_conditions = [
            models.FieldCondition(key="legal_sectors", match=models.MatchValue(value=sector_name)),
            models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
            models.FieldCondition(key="is_appendix", match=models.MatchValue(value=False))
        ]
        scroll_filter = models.Filter(must=must_conditions)
        
        points, _ = retriever.client.scroll(
            collection_name=retriever.collection_name,
            scroll_filter=scroll_filter,
            limit=50,
            with_payload=True
        )
        
        hits = []
        for p in points:
            payload = p.payload or {}
            hits.append({
                "id": p.id,
                "score": 1.0,  # Exact match score
                "payload": payload,
                "document_number": payload.get("document_number", ""),
                "article_ref": payload.get("article_ref", ""),
                "title": payload.get("title", ""),
                "text": payload.get("expanded_context_text") or payload.get("chunk_text") or payload.get("text", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "is_appendix": payload.get("is_appendix", False),
                "url": payload.get("url", "")
            })
        
        # Deduplication để trả về danh sách văn bản đại diện
        from backend.agent.utils.utils_sector_search import deduplicate_by_document
        return deduplicate_by_document(hits)
    except Exception as e:
        logger.error(f"  [Sector Mode 1] Error: {e}")
        return []

def retrieve_by_topic_hybrid(query: str, top_k: int = 15) -> List[Dict]:
    """
    Hàm 2 (Tìm theo Chủ đề - Hybrid): Kết hợp BM25 (Sparse) và Dense Vector.
    Đặc biệt chỉ target vào Tiêu đề (và summary) văn bản (thường là chunk đầu tiên - is_appendix=False).
    """
    from backend.retrieval.hybrid_search import retriever
    from qdrant_client import models
    import logging
    logger = logging.getLogger("sector_search")
    
    logger.info(f"  [Sector Mode 2] Hybrid Topic Search targetting Document Titles: {query}")
    try:
        dense_query = retriever.hybrid_encoder.encode_query_dense(query)
        sparse_query = retriever.hybrid_encoder.encode_query_sparse(query)
        
        # Chỉ tìm trong các chunk nội dung chính (is_appendix=False) để bắt Title/Summary chuẩn xác
        topic_filter = models.Filter(
            must=[
                models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
                models.FieldCondition(key="is_appendix", match=models.MatchValue(value=False))
            ]
        )
        
        prefetch_limit = top_k * 2
        raw_hits = retriever.client.query_points(
            collection_name=retriever.collection_name,
            prefetch=[
                models.Prefetch(query=dense_query, using="dense", limit=prefetch_limit, filter=topic_filter),
                models.Prefetch(query=sparse_query, using="sparse", limit=prefetch_limit, filter=topic_filter),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points
        
        # Format results
        hits = []
        for p in raw_hits:
            payload = p.payload or {}
            # Tăng điểm nếu query khớp chính xác một phần Tiêu đề văn bản (BM25 mạnh ở đây)
            boost = 0.0
            if payload.get("title") and any(word.lower() in payload.get("title", "").lower() for word in query.split() if len(word) > 3):
                boost = 0.1
                
            hits.append({
                "id": p.id,
                "score": float(p.score) + boost,
                "payload": payload,
                "document_number": payload.get("document_number", ""),
                "article_ref": payload.get("article_ref", ""),
                "title": payload.get("title", ""),
                "text": payload.get("expanded_context_text") or payload.get("chunk_text") or payload.get("text", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "is_appendix": payload.get("is_appendix", False),
                "url": payload.get("url", "")
            })
            
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits
    except Exception as e:
        logger.error(f"  [Sector Mode 2] Error: {e}")
        return []

def retrieve_by_article_clause(query: str, top_k: int = 20) -> List[Dict]:
    """
    Hàm 3 (Tìm theo Điều khoản): Quét Text Embedding (Dense Vector) tìm nội dung chi tiết.
    Dùng toàn bộ kho (không dùng sparse để tránh bias từ khóa lên thẻ tiêu đề).
    """
    from backend.retrieval.hybrid_search import retriever
    from qdrant_client import models
    import logging
    logger = logging.getLogger("sector_search")
    
    logger.info(f"  [Sector Mode 3] Deep Vector Search for precise clauses: {query}")
    try:
        dense_query = retriever.hybrid_encoder.encode_query_dense(query)
        
        # Tìm rộng điều khoản (bao gồm cả phụ lục biểu mẫu nếu có)
        broad_filter = models.Filter(
            must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
        )
        
        raw_hits = retriever.client.query_points(
            collection_name=retriever.collection_name,
            query=dense_query,
            using="dense",
            query_filter=broad_filter,
            limit=top_k,
            with_payload=True,
        ).points
        
        hits = []
        for p in raw_hits:
            payload = p.payload or {}
            hits.append({
                "id": p.id,
                "score": float(p.score),
                "payload": payload,
                "document_number": payload.get("document_number", ""),
                "article_ref": payload.get("article_ref", ""),
                "title": payload.get("title", ""),
                "text": payload.get("expanded_context_text") or payload.get("chunk_text") or payload.get("text", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "is_appendix": payload.get("is_appendix", False),
                "url": payload.get("url", "")
            })
            
        # Áp dụng reranker cho kết quả mode 3 vì cần độ chính xác ngữ nghĩa cao nhất ở mức clause
        if hasattr(retriever, "reranker") and retriever.reranker:
            enriched = retriever.reranker.rerank_candidates(query, hits, top_k=top_k)
            for item in enriched:
                item["score"] = item.get("rerank_score", item.get("score"))
            return enriched
        
        return hits
    except Exception as e:
        logger.error(f"  [Sector Mode 3] Error: {e}")
        return []
# 2. GRADE — Deduplication + Relevance Filter
# =============================================================================

def deduplicate_by_document(hits: List[Dict]) -> List[Dict]:
    """
    Group theo document_number, giữ chunk có score cao nhất cho mỗi văn bản.
    Trả về danh sách unique docs (1 entry/văn bản).
    """
    doc_map: Dict[str, Dict] = {}
    for h in hits:
        doc_num = h.get("document_number", "")
        if not doc_num:
            continue
        existing = doc_map.get(doc_num)
        if existing is None or h.get("score", 0) > existing.get("score", 0):
            doc_map[doc_num] = h
    return list(doc_map.values())


def group_by_document(hits: List[Dict]) -> List[Dict]:
    """
    Thay thế deduplicate_by_document giúp giữ lại TẤT CẢ các điều khoản (article_ref)
    đã được tìm thấy cho cùng một văn bản thay vì chỉ giữ 1 chunk.
    """
    doc_map: Dict[str, Dict] = {}
    for h in hits:
        doc_num = h.get("document_number", "")
        if not doc_num:
            continue
            
        if doc_num not in doc_map:
            # Tạo entry mới, copy hit và khởi tạo list điều khoản
            doc_map[doc_num] = dict(h)
            doc_map[doc_num]["all_articles"] = []
            
        # Thu thập article_ref nếu có
        art = h.get("article_ref", "")
        if art and art not in doc_map[doc_num]["all_articles"]:
            doc_map[doc_num]["all_articles"].append(art)
            
        # Cập nhật nội dung chunk có score cao nhất để làm đại diện (summary/preview)
        if h.get("score", 0) > doc_map[doc_num].get("score", 0):
            # Lưu lại all_articles đã thu thập được
            current_articles = doc_map[doc_num]["all_articles"]
            doc_map[doc_num].update(h)
            doc_map[doc_num]["all_articles"] = current_articles
            
    return list(doc_map.values())


def _heuristic_date_filter(hits: List[Dict], date_range: Dict) -> List[Dict]:
    """Lọc theo effective_date_range nếu có. Pure Python, không gọi LLM."""
    year_from = date_range.get("from", "")
    year_to = date_range.get("to", "")
    
    if not year_from and not year_to:
        return hits  # Không có filter thời gian
    
    try:
        y_from = int(year_from) if year_from else 1900
        y_to = int(year_to) if year_to else 2100
    except ValueError:
        return hits
    
    filtered = []
    for h in hits:
        eff_date = h.get("effective_date", "") or h.get("issuance_date", "")
        if not eff_date:
            filtered.append(h)  # Giữ lại nếu không có ngày (thiếu thông tin)
            continue
        # Extract year from date string
        m = re.search(r'\b(19|20)\d{2}\b', str(eff_date))
        if m:
            year = int(m.group())
            if y_from <= year <= y_to:
                filtered.append(h)
        else:
            filtered.append(h)  # Giữ lại nếu không parse được
    return filtered


def grade_relevance_batch(query: str, hits: List[Dict], llm_preset: str = None) -> List[Dict]:
    """
    Lọc tính liên quan bằng 1 LLM call duy nhất.
    Input: danh sách hits đã dedup.
    Output: danh sách hits liên quan.
    """
    if not hits:
        return []
    
    # Build compact doc list (chỉ metadata, ~50 tokens/entry)
    doc_lines = []
    for idx, h in enumerate(hits):
        doc_id = f"doc_{idx}"
        doc_num = h.get("document_number", "N/A")
        legal_type = h.get("legal_type", "N/A")
        title = h.get("title", "N/A")[:80]
        doc_lines.append(f"[{doc_id}] {doc_num} | {legal_type} | {title}")
    
    doc_list_text = "\n".join(doc_lines)
    
    messages = [{"role": "user", "content": RELEVANCE_BATCH_PROMPT.format(
        query=query, doc_list=doc_list_text
    )}]
    
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset)
        resp = resp or ""
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        allowed_ids = json.loads(resp)
        if not isinstance(allowed_ids, list):
            raise ValueError("Expected list")
        
        filtered = []
        for idx, h in enumerate(hits):
            if f"doc_{idx}" in allowed_ids:
                filtered.append(h)
        return filtered if filtered else hits  # Fallback: giữ hết nếu LLM lọc sạch
    except Exception as e:
        print(f"       ⚠️ Relevance Batch filter failed: {e}. Keeping all.")
        return hits


# =============================================================================
# 3. MAP-REDUCE AGGREGATOR (Pure Python — KHÔNG gọi LLM)
# =============================================================================

def map_reduce_aggregate(hits: List[Dict]) -> str:
    """
    Map: Nhóm văn bản theo issuing_authority.
    Reduce: Build bảng Markdown cấu trúc cho mỗi nhóm.
    Output: Chuỗi Markdown hoàn chỉnh.
    """
    if not hits:
        return ""
    
    # === MAP PHASE: Group by issuing_authority ===
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for h in hits:
        authority = h.get("issuing_authority", "") or "Không rõ cơ quan"
        groups[authority].append(h)
    
    # === REDUCE PHASE: Build Markdown tables ===
    md_parts = []
    global_idx = 0
    
    for authority, docs in sorted(groups.items()):
        # Sort docs within group by effective_date descending
        def sort_key(d):
            eff = d.get("effective_date", "") or d.get("issuance_date", "")
            m = re.search(r'\b(19|20)\d{2}\b', str(eff))
            return int(m.group()) if m else 0
        
        docs_sorted = sorted(docs, key=sort_key, reverse=True)
        
        md_parts.append(f"\n#### 🏛️ {authority}\n")
        md_parts.append("| STT | Số hiệu | Tên văn bản | Loại | Ngày hiệu lực | Lĩnh vực |")
        md_parts.append("| :---: | :--- | :--- | :---: | :---: | :--- |")
        
        for doc in docs_sorted:
            global_idx += 1
            doc_num = doc.get("document_number", "N/A")
            title = doc.get("title", "N/A")
            is_appendix = doc.get("is_appendix", False)
            
            # Nếu là phụ lục, thêm nhãn vào tiêu đề để người dùng dễ nhận biết
            display_title = f"[PHỤ LỤC] {title}" if is_appendix else title
            
            # Truncate title for table readability
            if len(display_title) > 80:
                display_title = display_title[:77] + "..."
            legal_type = doc.get("legal_type", "N/A")
            eff_date = doc.get("effective_date", "") or doc.get("issuance_date", "N/A")
            sectors = doc.get("legal_sectors", [])
            sectors_str = ", ".join(sectors[:3]) if sectors else "—"
            
            md_parts.append(
                f"| {global_idx} | {doc_num} | {display_title} | {legal_type} | {eff_date} | {sectors_str} |"
            )
    
    # Add statistics footer
    type_counts = Counter(h.get("legal_type", "N/A") for h in hits)
    stats_parts = [f"{t}: {c}" for t, c in type_counts.most_common()]
    md_parts.append(f"\n> **Tổng cộng:** {len(hits)} văn bản | Phân bố: {', '.join(stats_parts)}")
    
    return "\n".join(md_parts)


# =============================================================================
# 4. GENERATE — Executive Summary (1 LLM call, ~100 tokens output)
# =============================================================================

def generate_executive_summary(query: str, table_markdown: str, file_chunks: List[Dict[str, Any]] = None, file_analysis: str = "", history_str: str = "", llm_preset: str = None) -> tuple[str, str]:
    """Sinh đoạn Executive Summary lồng ghép phân tích file."""
    if not file_analysis and file_chunks:
        file_analysis = "Người dùng có tải lên tài liệu liên quan đến chủ đề này."
    
    prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        history=history_str,
        query=query, 
        table=table_markdown or "Hiện tại không tìm thấy văn bản nào khớp hoàn toàn.",
        file_analysis=file_analysis or "Không có tài liệu đính kèm."
    )
    
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_CORE_MODEL, llm_preset=llm_preset)
        from backend.utils.text_utils import extract_thinking_and_answer
        return extract_thinking_and_answer(resp)
    except Exception as e:
        print(f"       ⚠️ Executive Summary generation failed: {e}")
        return "", f"Tổng quan về các văn bản pháp luật liên quan đến \"{query}\"."

def analyze_document_focus(file_chunks: List[Dict[str, Any]], llm_preset: str = None) -> dict:
    """LLM 'nắm bắt' nội dung file để định hướng tìm kiếm."""
    if not file_chunks:
        return {}
        
    # Lấy 3 chunk đầu tiên để phân tích trọng tâm
    context = ""
    for c in file_chunks[:3]:
        context += c.get("text_to_embed", c.get("unit_text", "")) + "\n"
        
    messages = [{"role": "user", "content": DOCUMENT_FOCUS_PROMPT.format(file_context=context)}]
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset)
        resp = resp or ""
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        return json.loads(resp)
    except Exception as e:
        print(f"       ⚠️ Document Analysis failed: {e}")
        return {}



