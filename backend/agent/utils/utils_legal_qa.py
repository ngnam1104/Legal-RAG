from typing import List, Dict, Any, Optional
import json
import re
import time
from backend.llm.factory import chat_completion
from backend.retrieval.hybrid_search import retriever
from backend.config import settings

def strip_thinking_tags(text: str) -> str:
    """Loại bỏ thẻ <thinking>...</thinking> khỏi output LLM trước khi hiển thị hoặc parse JSON."""
    return re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()

# --- PROMPTS ---
ANSWER_PROMPT = """
HISTORY = {history}
CONTEXT = {context}
QUERY = {query}
SUPPLEMENTAL_CONTEXT = {supplemental_context}

BẠN LÀ MỘT HỆ THỐNG TRÍ TUỆ NHÂN TẠO PHÁP LÝ HOẠT ĐỘNG TRONG MÔI TRƯỜNG ĐÓNG (CLOSED-DOMAIN).
ĐÂY LÀ QUY TRÌNH BẮT BUỘC KHÔNG THỂ THƯƠNG LƯỢNG: BẠN CHỈ ĐƯỢC PHÉP TRẢ LỜI CHO `QUERY` DỰA **HOÀN TOÀN** VÀO VĂN BẢN `CONTEXT` (VÀ `SUPPLEMENTAL_CONTEXT`). KHÔNG BAO GIỜ SỬ DỤNG KIẾN THỨC CÓ SẴN CỦA BẠN.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI VIẾT CÂU TRẢ LỜI CUỐI CÙNG.

═══════════════════════════════════════════════════
BƯỚC 1: SUY LUẬN BẮT BUỘC (viết trong thẻ <thinking>)
═══════════════════════════════════════════════════
<thinking>
A. QUÉT TIÊU ĐỀ VÀ METADATA: Duyệt qua TỪNG tài liệu trong CONTEXT. 
   - ĐẶC BIỆT LƯU Ý thẻ <thong_tin_van_ban_chinh_xac> (nếu có). Đây là nguồn sự thật tuyệt đối để trả lời các câu hỏi về: Nhấn mạnh Nơi nhận, Lãnh đạo ký ban hành, Căn cứ pháp lý, Hệ thống văn bản (bãi bỏ, thay thế văn bản nào).
   - Tìm các Điều/Khoản/Phụ lục có TIÊU ĐỀ hoặc NỘI DUNG khớp trực tiếp với từ khóa trong QUERY (ví dụ: "thẩm quyền", "điều kiện", "người ký", "nơi nhận").
   → Liệt kê: "[Nguồn X] - [Điều/Phụ lục Y] - KHỚP vì: ..."

B. TRÍCH XUẤT THÔ: Với mỗi tài liệu khớp (hoặc metadata khớp), trích dẫn NGUYÊN VĂN thông tin liên quan. KHÔNG suy diễn, KHÔNG thêm thông tin. Chỉ copy-paste chính xác.

C. GIẢI QUYẾT XUNG ĐỘT (nếu có):
   - Nếu 2 tài liệu cùng đề cập một vấn đề → Ưu tiên văn bản quy định TRỰC TIẾP hành vi/mức phạt đó.
   - Thứ tự ưu tiên giá trị pháp lý: Luật → Nghị định → Thông tư → Quyết định.
   - Nếu có cả NỘI DUNG CHÍNH và PHỤ LỤC → NỘI DUNG CHÍNH là căn cứ chính, PHỤ LỤC là chi tiết bổ sung.

D. XÁC ĐỊNH CÁCH TRÍCH DẪN:
   - Nếu nguồn là NỘI DUNG CHÍNH (Điều/Khoản) → Trích dẫn: "Căn cứ [Loại VB] [Số hiệu] [Tên VB] - [Điều X, Khoản Y]"
   - Nếu nguồn là PHỤ LỤC → Trích dẫn: "Căn cứ [Loại VB] [Số hiệu] [Tên VB] - [Phụ lục số/tên]"
   - Nếu nguồn là NỘI DUNG CHUNG (không có Điều cụ thể) → Trích dẫn: "Căn cứ phần nội dung chung của [Loại VB] [Số hiệu] [Tên VB]"
</thinking>

═══════════════════════════════════════════════════
BƯỚC 2: VIẾT CÂU TRẢ LỜI CUỐI CÙNG (sau thẻ </thinking>)
═══════════════════════════════════════════════════
YÊU CẦU:
1. CÂU MỞ ĐẦU BẮT BUỘC theo đúng định dạng trích dẫn đã xác định ở Bước 1D. Nếu NGƯỜI DÙNG CHỈ HỎI XIN TRÍCH DẪN/CĂN CỨ CỦA CÂU TỪ LƯỢT TRƯỚC: Hãy trực tiếp nhìn vào HISTORY và trích xuất Metadata. Chú ý: Nếu người dùng hỏi "văn bản này căn cứ vào các luật nào" (Căn cứ ban hành), lập tức tìm thẻ `<can_cu_phap_ly>` trong CONTEXT để liệt kê các base laws.
2. ƯU TIÊN TUYỆT ĐỐI thông tin từ "TÀI LIỆU TẢI LÊN" (nằm trong thẻ `<tai_lieu>`). Đây là nguồn sự thật cao nhất. Nếu có mâu thuẫn giữa Phụ lục/Nội quy trong file và Luật chung trong DB (`<can_cu>`), hãy nhấn mạnh quy định trong file của người dùng là căn cứ trực tiếp nhất.
3. Nếu CONTEXT có đủ thông tin, hãy trả lời thẳng thắn (Được/Không được, Đúng/Sai, Mức phạt là bao nhiêu).
4. Nếu CONTEXT KHÔNG TRỰC TIẾP CHỨA câu trả lời cho QUERY, bắt đầu bằng: "Dựa trên các quy định liên quan nhất tìm thấy, ..."
5. QUY TẮC TRÍCH DẪN: Ưu tiên trích dẫn văn bản có giá trị pháp lý cao hơn (Luật → Nghị định → Thông tư → Quyết định). Chỉ trích dẫn tài liệu phụ nếu tài liệu chính không chứa chi tiết đó.
6. NGẶT NGHÈO: Tuyệt đối không nhắc đến bất kỳ tên Luật, Điều khoản, hay số liệu nào không xuất hiện chữ-nguyên-chữ trong CONTEXT (hoặc trong HISTORY nếu hỏi lại căn cứ). Nội dung hoàn toàn phải bám sát.
7. QUY TẮC CẤM ẢO GIÁC TUYỆT ĐỐI (CRITICAL):
   - KHÔNG ĐƯỢC PHÉP tự sáng tác bất kỳ số hiệu văn bản (vd: 123/2024/TT-BCT), tên Quyết định, hay tên Luật nào KHÔNG có trong context.
   - Nếu bạn không chắc chắn hoặc context không có số hiệu cụ thể, hãy ghi "theo quy định hiện hành" thay vì bịa ra một số hiệu ngẫu nhiên.
   - Bất kỳ trích dẫn nào bắt đầu bằng "Quyết định số...", "Thông tư số..." PHẢI tìm thấy chính xác 100% trong thẻ `<nguon>` hoặc `<vi_tri>` của context.

TIÊU CHUẨN TRÍCH DẪN BỔ SUNG:
Nếu bên dưới có "PHẦN THÔNG TIN BỔ SUNG TỪ THAM CHIẾU", hãy ưu tiên sử dụng SUPPLEMENTAL_CONTEXT để giải thích các nội dung mà câu trả lời chính nhắc tới.
"""





def build_legal_context(hits: List[Dict[str, Any]], file_chunks: List[Dict[str, Any]] = None, max_chars: int = None, graph_context: Dict[str, Any] = None) -> str:
    """
    Xây dựng Ngữ cảnh Pháp lý (Legal Context) sử dụng 5 chiến thuật:
    1. Document TOC Injection: Đưa Mục lục văn bản (từ Neo4j) vào đầu context chống Lost-in-the-Middle.
    2. XML Context Separation: Tách biệt <tai_lieu_tam> (upload) và <tai_lieu_db> (hệ thống).
    3. Lost-in-the-Middle Reordering: Đưa hits quan trọng nhất ra 2 đầu.
    4. Hard Character Limit: Đảm bảo không vượt quá cửa sổ ngữ cảnh LLM.
    5. Sibling Expansion: Ghép sibling texts từ Bottom-Up traversal (Neo4j).
    """
    import re
    if max_chars is None:
        max_chars = settings.MAX_CONTEXT_CHARS

    context_parts = []
    current_chars = 0
    doc_counter = 0
    
    # --- Chiến thuật Đặc Biệt: Metadata Information (Signer/BasedOn/Year) ---
    if graph_context:
        meta_info = []
        if graph_context.get("year_info"):
            meta_info.append(graph_context["year_info"])
        if graph_context.get("signer_info"):
            meta_info.append(graph_context["signer_info"])
        if graph_context.get("based_on_info"):
            meta_info.append(graph_context["based_on_info"])
        if graph_context.get("admin_metadata"):
            meta_info.append(graph_context["admin_metadata"])
            
        if meta_info:
            meta_block = "<thong_tin_van_ban_chinh_xac>\n" + "\n".join(meta_info) + "\n</thong_tin_van_ban_chinh_xac>\n"
            context_parts.append(meta_block)
            current_chars += len(meta_block)
    
    # --- Chiến thuật 0: Document TOC (từ Neo4j graph_context) ---
    if graph_context and graph_context.get("document_toc"):
        toc_text = graph_context["document_toc"]  # Removed [:2000] hard slice
        toc_block = f"<muc_luc_van_ban>\n{toc_text}\n</muc_luc_van_ban>\n"
        context_parts.append(toc_block)
        current_chars += len(toc_block)

    # --- [TÀI LIỆU TẢI LÊN]: Bọc trong thẻ <tai_lieu_tam> ---
    if file_chunks:
        context_parts.append("<tai_lieu_tam>")
        for idx, f_chunk in enumerate(file_chunks, start=1):
            text = f_chunk.get("text_to_embed", f_chunk.get("unit_text", ""))
            # Nén trắng
            text = re.sub(r'\s*\n\s*', '\n', text)
            text = re.sub(r' {2,}', ' ', text).strip()
            chunk_info = f"[File Chunk {idx}]\n{text}\n"
            context_parts.append(chunk_info)
        context_parts.append("</tai_lieu_tam>\n")

    # --- [CƠ SỞ PHÁP LÝ TỪ HỆ THỐNG DB]: Bọc trong thẻ <tai_lieu_db> ---
    # Chiến thuật 1: Lost-in-the-Middle Reordering
    if hits and len(hits) > 2:
        sorted_hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
        reordered = [None] * len(sorted_hits)
        
        left = 0
        right = len(sorted_hits) - 1
        for i, hit in enumerate(sorted_hits):
            if i % 2 == 0:
                reordered[left] = hit
                left += 1
            else:
                reordered[right] = hit
                right -= 1
        hits = reordered

    context_parts.append("<tai_lieu_db>")
    
    # Chiến thuật 2: XML Context Injection
    for hit in hits:
        doc_counter += 1
        ref = hit.get("article_ref") or hit.get("reference_tag") or "N/A"
        doc_number = hit.get("document_number") or "N/A"
        title = hit.get("title") or "N/A"
        ref_citation = hit.get("reference_citation") or ref
        text = hit.get("text", "")
        # Nén trắng
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        is_appendix = hit.get("is_appendix", False)
        base_laws = hit.get("base_laws", [])

        if is_appendix:
            loai = "PHỤ LỤC / BẢNG BIỂU ĐÍNH KÈM - Đây là dữ liệu chi tiết, định mức hoặc biểu mẫu."
        else:
            loai = "NỘI DUNG CHÍNH - Quy định trực tiếp trong văn bản pháp luật."

        base_law_xml = f"\n    <can_cu_phap_ly>{', '.join(base_laws)}</can_cu_phap_ly>" if base_laws else ""
        
        chunk_xml = (
            f'<can_cu id="{doc_counter}">\n'
            f'  <metadata>\n'
            f'    <nguon>{doc_number} ({title})</nguon>\n'
            f'    <vi_tri>{ref_citation}</vi_tri>\n'
            f'    <loai_noi_dung>{loai}</loai_noi_dung>{base_law_xml}\n'
            f'  </metadata>\n'
            f'  <noi_dung>\n'
            f'    {text}\n'
            f'  </noi_dung>\n'
            f'</can_cu>\n'
        )

        context_parts.append(chunk_xml)
    
    # --- Sibling texts từ Neo4j Bottom-Up Expansion ---
    if graph_context and graph_context.get("sibling_texts"):
        for sib_text in graph_context["sibling_texts"][:10]: # Limit to 10 siblings
            sib_text_clean = re.sub(r'\s*\n\s*', '\n', sib_text)
            sib_block = f'<can_cu_bo_sung>\n  {sib_text_clean}\n</can_cu_bo_sung>\n'
            context_parts.append(sib_block)
    
    context_parts.append("</tai_lieu_db>")

    return "\n".join(context_parts)


def extract_legal_references(text: str) -> List[str]:
    """
    Sử dụng Regex để phát hiện các Điều khoản hoặc Phụ lục được nhắc tới.
    """
    import re
    # Patterns phổ biến: Điều X, Phụ lục Y (số hoặc La Mã)
    # Chúng ta không cần quá phức tạp, chỉ cần bắt đúng từ khóa và định danh
    patterns = [
        r"Điều\s+\d+",
        r"Phụ\s+lục\s+[\d\w\-]+", # Thêm dấu gạch ngang cho Phụ lục 02-A
        r"Phụ\s+lục\s+[IVXLCDM]+",
        r"Mẫu\s+số\s+\d+", # Thêm mẫu số nếu hay dẫn chiếu
        r"Khoản\s+\d+\s+Điều\s+\d+" # Kết hợp để tìm đúng Article cha
    ]
    
    found = []
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            # Chuẩn hóa: "điều 3" -> "Điều 3"
            norm = m.strip().replace("  ", " ").capitalize()
            # Nếu là Khoản X Điều Y -> Lấy Điều Y để truy xuất
            if "Khoản" in norm and "Điều" in norm:
                norm = "Điều" + norm.split("Điều")[1]
            if norm not in found:
                found.append(norm)
    return found

def resolve_recursive_references(primary_hits: List[Dict[str, Any]], max_supplemental_chars: int = 15000) -> List[Dict[str, Any]]:
    """
    Duyệt qua các hits hiện tại, tìm references và truy xuất thêm.
    Fix 6A: Giới hạn tổng ký tự bổ sung để tránh Context Overflow.
    """
    all_hits = list(primary_hits)
    seen_refs = set()
    
    # Đánh dấu các bản ghi hiện tại đã nạp để không nạp trùng
    for h in primary_hits:
        doc_num = h.get("document_number", "")
        art_ref = h.get("article_ref", "")
        if doc_num and art_ref:
            seen_refs.add(f"{doc_num}::{art_ref}")

    new_hits = []
    total_supplemental_chars = 0
    
    for h in primary_hits:
        text = h.get("text", "")
        doc_num = h.get("document_number", "")
        if not doc_num: continue
        
        refs = extract_legal_references(text)
        for ref_id in refs:
            # Chuẩn hóa để check trùng (ví dụ "Điều 2" vs "Điều 02")
            clean_ref = ref_id.strip().lower()
            ref_key = f"{doc_num}::{clean_ref}"
            
            # Kiểm tra xem đã có trong set chưa (bao gồm cả các bản ghi ban đầu)
            if ref_key not in seen_refs:
                # Fix 6A: Kiểm tra giới hạn ký tự trước khi fetch
                if total_supplemental_chars >= max_supplemental_chars:
                    print(f"      ⚠️ [Recursive Ref] Reached supplemental char limit ({max_supplemental_chars}). Stopping.")
                    break
                    
                print(f"      🔗 Found internal reference: {ref_id} in {doc_num}. Fetching...")
                supp_hits = retriever.retrieve_specific_reference(doc_num, ref_id)
                if supp_hits:
                    for sh in supp_hits:
                        sh["is_supplemental"] = True
                        sh["score"] = 1.0  # Cấp score tột đỉnh để không bị loại ở khâu Lost-in-the-middle
                        total_supplemental_chars += len(sh.get("text", ""))
                        new_hits.append(sh)
                        seen_refs.add(ref_key)
                        
    # Giới hạn số lượng tham chiếu bổ sung để không làm loãng context (Max 10 để bao phủ tốt hơn)
    return all_hits + new_hits[:10]


def filter_cited_references(answer_text: str, refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lọc danh sách references chỉ giữ lại các chunk thực sự được trích dẫn trong câu trả lời.
    Giúp UI chỉ hiển thị nguồn tham chiếu có ý nghĩa thay vì toàn bộ top-k.
    
    Logic khớp:
      1. document_number xuất hiện trong answer (VD: "51/2025/TT-BYT")
      2. article_ref xuất hiện trong answer (VD: "Điều 5")
      3. Partial match — chỉ phần số hiệu ngắn (VD: "51/2025")
    Fallback: Nếu không phát hiện citation nào → giữ top 3 theo score.
    """
    if not answer_text or not refs:
        return refs

    answer_lower = answer_text.lower()
    cited = []

    for ref in refs:
        doc_num = ref.get("document_number", "")
        article = ref.get("article", "") or ""
        title = ref.get("title", "")
        text_preview = ref.get("text_preview", "")

        is_cited = False

        # Check 1: Document match (VD: "51/2025/TT-BYT")
        has_doc_match = False
        if doc_num and len(doc_num) > 3:
            if doc_num.lower() in answer_lower:
                has_doc_match = True
            else:
                parts = doc_num.split("/")
                if len(parts) >= 2:
                    short_num = f"{parts[0]}/{parts[1]}"
                    if short_num.lower() in answer_lower:
                        has_doc_match = True

        # Check 2: Article reference match (VD: "Điều 5")
        has_article_match = False
        if article:
            article_clean = article.strip().lower()
            if article_clean and len(article_clean) > 2 and article_clean in answer_lower:
                has_article_match = True

        # Quyết định Is Cited (Strict mode)
        if has_article_match:
            is_cited = True
        elif has_doc_match and not article:
            # Chunk chung của văn bản (không có điều khoản)
            is_cited = True
            
        # Check 3: Text snippet match
        if not is_cited and title and len(title) > 15:
            title_words = title.split()
            if len(title_words) > 3:
                title_fragment = " ".join(title_words[:5]).lower()
                if title_fragment in answer_lower:
                    is_cited = True

        if is_cited:
            # --- [SANITIZATION - LÀM SẠCH DỮ LIỆU UI] ---
            # 1. Rút gọn Article: Nếu có dạng "ABC > Điều 5", bỏ ABC nếu quá dài
            if " > " in article:
                parts = article.split(" > ")
                # Nếu phần tiền tố quá dài (>40 ký tự), thường là lặp lại tên văn bản
                if len(parts[0]) > 40:
                    ref["article"] = parts[-1]
            
            # 2. Làm sạch Text Preview: Nếu bắt đầu bằng "Điều X", "Khoản Y" mà trùng với article -> bỏ
            clean_text = text_preview.strip()
            art_label = ref["article"].split(" > ")[-1].strip()
            if clean_text.lower().startswith(art_label.lower()):
                # Cắt bỏ phần lặp lại đầu câu (VD: "Điều 5. Nội dung..." -> "Nội dung...")
                clean_text = re.sub(r'^' + re.escape(art_label) + r'[\.\s\:\-]+', '', clean_text, flags=re.IGNORECASE).strip()
            ref["text_preview"] = clean_text
            
            cited.append(ref)

    # Fallback: Nếu LLM tóm tắt hoàn toàn / không cite metadata → giữ duy nhất 1 nguồn đáng tin nhất
    if not cited and refs:
        fallback_ref = sorted(refs, key=lambda x: x.get("score", 0), reverse=True)[0]
        # Vẫn áp dụng sanitization cho fallback
        if " > " in fallback_ref.get("article", ""):
            parts = fallback_ref["article"].split(" > ")
            if len(parts[0]) > 40: fallback_ref["article"] = parts[-1]
        return [fallback_ref]

    return cited

