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


TOC_DRILL_DOWN_PROMPT = """
QUERY = {query}
TITLE = {title}
TOC = {toc}

Bạn là Chuyên gia Điều hướng Văn bản Pháp luật.
Nhiệm vụ của bạn là đọc Mục lục (TOC) của văn bản TITLE và xác định các ĐIỀU (Article) hoặc MỤC (Section) có khả năng cao nhất chứa câu trả lời cho QUERY.

YÊU CẦU:
1. Trả về tối đa 3 tọa độ (ví dụ: "Điều 5", "Điều 10", "Mục 2").
2. Chỉ trả về mảng JSON trong markdown code block.

```json
["Điều X", "Điều Y"]
```
"""


# =============================================================================
# 1. UNDERSTAND — Sector Query Planner
# =============================================================================




# =============================================================================
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

def generate_executive_summary(query: str, table_markdown: str, file_chunks: List[Dict[str, Any]] = None, file_analysis: str = "", history_str: str = "", llm_preset: str = None) -> str:
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
        return strip_thinking_tags(resp)
    except Exception as e:
        print(f"       ⚠️ Executive Summary generation failed: {e}")
        return f"Tổng quan về các văn bản pháp luật liên quan đến \"{query}\"."

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


# =============================================================================
# 5. REFLECT — Coverage Check & Supplemental Search
# =============================================================================

def check_coverage_bias(hits: List[Dict]) -> Dict[str, Any]:
    """
    Phân tích phân bố legal_type trong kết quả.
    Phát hiện thiên lệch: ví dụ chỉ toàn Luật mà thiếu Nghị định hướng dẫn.
    Returns: {"biased": bool, "dominant_type": str, "missing_types": [...], "basis_doc_numbers": [...]}
    """
    if len(hits) < 3:
        return {"biased": False, "dominant_type": "", "missing_types": [], "basis_doc_numbers": []}
    
    type_counts = Counter(h.get("legal_type", "N/A") for h in hits)
    total = sum(type_counts.values())
    
    # Check if one type dominates > 80%
    dominant_type, dominant_count = type_counts.most_common(1)[0]
    ratio = dominant_count / total if total > 0 else 0
    
    # Hierarchy: Luật thường đi kèm Nghị định, Nghị định đi kèm Thông tư
    EXPECTED_COMPANIONS = {
        "Luật": ["Nghị định", "Thông tư"],
        "Nghị định": ["Thông tư", "Luật"],
        "Thông tư": ["Nghị định"],
        "Nghị quyết": ["Nghị định", "Luật"],
    }
    
    missing_types = []
    if ratio > 0.8 and dominant_type in EXPECTED_COMPANIONS:
        companions = EXPECTED_COMPANIONS[dominant_type]
        present_types = set(type_counts.keys())
        missing_types = [t for t in companions if t not in present_types]
    
    # Extract doc_numbers from legal_basis_refs for supplemental search
    basis_doc_numbers = []
    if missing_types:
        seen = set()
        for h in hits:
            for ref in h.get("legal_basis_refs", []):
                dn = ref.get("doc_number", "")
                if dn and dn not in seen:
                    basis_doc_numbers.append(dn)
                    seen.add(dn)
    
    return {
        "biased": bool(missing_types),
        "dominant_type": dominant_type,
        "missing_types": missing_types,
        "basis_doc_numbers": basis_doc_numbers[:5],  # Limit to 5 supplemental searches
        "type_distribution": dict(type_counts)
    }


def supplemental_search_by_basis(basis_doc_numbers: List[str], existing_doc_nums: set) -> List[Dict]:
    """
    Tìm kiếm bổ sung dựa trên legal_basis_refs của các văn bản đã tìm được.
    Dùng Qdrant filter chính xác theo document_number.
    """
    supplemental_hits = []
    
    for doc_num in basis_doc_numbers:
        if doc_num in existing_doc_nums:
            continue  # Đã có rồi, bỏ qua
        try:
            hits = retriever.search(
                query=doc_num,
                expand_context=False,
                use_rerank=False,
                doc_number=doc_num,
                limit=1  # Chỉ cần 1 chunk đại diện
            )
            if hits:
                supplemental_hits.append(hits[0])
                existing_doc_nums.add(doc_num)
        except Exception as e:
            print(f"       ⚠️ Supplemental search for {doc_num} failed: {e}")
    
    return supplemental_hits


def drill_down_toc_with_llm(query: str, title: str, toc: str, llm_preset: str = None) -> List[str]:
    """Sử dụng LLM để tìm các Điều/Mục trong TOC liên quan đến Query."""
    if not toc or not query:
        return []
        
    messages = [{"role": "user", "content": TOC_DRILL_DOWN_PROMPT.format(
        query=query, title=title, toc=toc # Đã tăng giới hạn ngữ cảnh cho LLM
    )}]
    
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset)
        resp = resp or ""
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        coords = json.loads(resp)
        return coords if isinstance(coords, list) else []
    except Exception as e:
        print(f"       ⚠️ TOC Drill-down failed for '{title}': {e}")
        return []
