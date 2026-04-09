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
REWRITE_PROMPT = """
Bạn là hệ thống Tối ưu Khởi vấn Pháp lý (Legal Query Rewriter) kiêm trích xuất dữ liệu.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

Nhiệm vụ: 
1. Phân tích câu hỏi gốc của người dùng. Viết một "câu trả lời giả định" (HyDE) HOẶC "cách diễn đạt lại từ 3 góc độ khác nhau" gộp thành một đoạn văn duy nhất. Tập trung vào các từ khóa pháp luật cốt lõi, tên riêng, số hiệu (nhằm mục đích vector search).
2. TỐI ƯU CẤU TRÚC PHÁP LÝ: Nếu câu hỏi hỏi về "đối tượng áp dụng", "phạm vi điều chỉnh", "giải thích thuật ngữ", hoặc "hiệu lực", hãy đảm bảo trong hypothetical_answer có nhắc đến các cụm từ "Điều 1", "Điều 2", "Điều 3", "Điều 4" hoặc "Điều 5" để hệ thống ưu tiên tìm các điều khoản mở đầu quan trọng này.
3. Trích xuất các điều kiện lọc (nếu có) như loại văn bản pháp lý (Luật, Nghị định, Thông tư...) hoặc số hiệu văn bản (123/2024).

Câu hỏi gốc: {query}

BƯỚC 1: Suy luận trong thẻ <thinking>
<thinking>
- Ý định cốt lõi của câu hỏi là gì? (Hỏi về quy định cụ thể, mức phạt, quy trình, hay khái niệm?)
- Từ khóa pháp lý trọng tâm nào cần có trong hypothetical_answer?
- Có nhắc đến loại văn bản hoặc số hiệu cụ thể không?
- Câu hỏi có liên quan đến Phụ lục/bảng biểu/mẫu không?
- Nếu hỏi về "đối tượng áp dụng" → nên nhắm đến "Điều 2"
</thinking>

BƯỚC 2: Sau khi suy luận xong, TRẢ VỀ JSON trong markdown code block:
```json
{{
    "hypothetical_answer": "(Bắt buộc) Đoạn văn tóm tắt nội dung/góc độ truy vấn...",
    "filters": {{
        "legal_type": "(Tùy chọn) Luật/Nghị định/Nghị quyết/Thông tư/Quyết định/...",
        "doc_number": "(Tùy chọn) Số hiệu văn bản hoặc năm, vd: 123/2024 hoặc 2024",
        "is_appendix": "(Tùy chọn) true nếu liên quan danh mục/mẫu/bảng, false nếu không.",
        "article_ref": "(Tùy chọn) Rất Quan Trọng: CHỈ BÁO CÁO DUY NHẤT 1 ĐIỀU KHOẢN (VD: 'Điều 2'). Tuyệt đối không ghi nhiều điều khoản hay dấu phẩy. Đặc biệt với câu hỏi 'đối tượng áp dụng' -> 'Điều 2'."
    }}
}}
```
"""

GRADER_PROMPT = """
Bạn là Người Đánh Giá Ngữ Cảnh (Context Grader).
Nhiệm vụ: Đọc NGỮ CẢNH được cung cấp và đánh giá xem nó có chứa DỮ LIỆU LIÊN QUAN để trả lời CÂU HỎI của người dùng hay không.
Chú ý: Bạn KHÔNG CẦN trả lời câu hỏi, chỉ cần đánh giá YES hoặc NO.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

NGỮ CẢNH:
{context}

CÂU HỎI:
{query}

BƯỚC 1: Suy luận
<thinking>
- Câu hỏi hỏi về chủ đề gì?
- Ngữ cảnh có nhắc đến chủ đề đó không? Có điều khoản/quy định nào liên quan trực tiếp?
- Dù không trả lời trực tiếp, ngữ cảnh có chứa thông tin có thể dùng để tham khảo không?
</thinking>

BƯỚC 2: Trả về JSON trong markdown code block:
```json
{{
    "is_relevant": "yes" hoặc "no"
}}
```
"""

TRANSFORM_PROMPT = """
Bạn là Chuyên gia Tối ưu Truy vấn.
Lần tìm kiếm trước chưa tìm thấy kết quả phù hợp. Hãy viết lại câu hỏi dưới đây dưới một góc độ khác (ví dụ: dùng từ đồng nghĩa, hoặc bóc tách ý chính khái quát hơn) để hệ thống có thể tìm thấy.
Không giải thích, chỉ trả về CÂU HỎI MỚI.

CÂU HỎI GỐC: {query}
"""

ANSWER_PROMPT = """
BẠN LÀ MỘT HỆ THỐNG TRÍ TUỆ NHÂN TẠO PHÁP LÝ HOẠT ĐỘNG TRONG MÔI TRƯỜNG ĐÓNG (CLOSED-DOMAIN).
ĐÂY LÀ QUY TRÌNH BẮT BUỘC KHÔNG THỂ THƯƠNG LƯỢNG: BẠN CHỈ ĐƯỢC PHÉP TRẢ LỜI DỰA **HOÀN TOÀN** VÀO VĂN BẢN NGỮ CẢNH DƯỚI ĐÂY. KHÔNG BAO GIỜ SỬ DỤNG KIẾN THỨC CÓ SẴN CỦA BẠN.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI VIẾT CÂU TRẢ LỜI CUỐI CÙNG.

LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{history}

NGỮ CẢNH (DO HỆ THỐNG RAG CUNG CẤP):
{context}

Câu hỏi hiện tại của người dùng: {query}

═══════════════════════════════════════════════════
BƯỚC 1: SUY LUẬN BẮT BUỘC (viết trong thẻ <thinking>)
═══════════════════════════════════════════════════
<thinking>
A. QUÉT TIÊU ĐỀ: Duyệt qua TỪNG tài liệu trong ngữ cảnh. Tìm các Điều/Khoản/Phụ lục có TIÊU ĐỀ hoặc NỘI DUNG khớp trực tiếp với từ khóa trong câu hỏi (ví dụ: "thẩm quyền", "điều kiện", "mức phạt", "đối tượng áp dụng").
   → Liệt kê: "[Nguồn X] - [Điều/Phụ lục Y] - KHỚP vì: ..."

B. TRÍCH XUẤT THÔ: Với mỗi tài liệu khớp, trích dẫn NGUYÊN VĂN đoạn văn bản liên quan. KHÔNG suy diễn, KHÔNG thêm thông tin. Chỉ copy-paste chính xác.

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
1. CÂU MỞ ĐẦU BẮT BUỘC theo đúng định dạng trích dẫn đã xác định ở Bước 1D. Nếu NGƯỜI DÙNG CHỈ HỎI XIN TRÍCH DẪN/CĂN CỨ CỦA CÂU TỪ LƯỢT TRƯỚC: Hãy trực tiếp nhìn vào LỊCH SỬ HỘI THOẠI và trích xuất Metadata. Chú ý: Nếu người dùng hỏi "văn bản này căn cứ vào các luật nào" (Căn cứ ban hành), lập tức tìm thẻ `<can_cu_phap_ly>` trong NGỮ CẢNH để liệt kê các base laws.
2. ƯU TIÊN TUYỆT ĐỐI thông tin từ "TÀI LIỆU TẢI LÊN" (nằm trong thẻ `<tai_lieu>`). Đây là nguồn sự thật cao nhất. Nếu có mâu thuẫn giữa Phụ lục/Nội quy trong file và Luật chung trong DB (`<can_cu>`), hãy nhấn mạnh quy định trong file của người dùng là căn cứ trực tiếp nhất.
3. Nếu Ngữ cảnh có đủ thông tin, hãy trả lời thẳng thắn (Được/Không được, Đúng/Sai, Mức phạt là bao nhiêu).
4. Nếu NGỮ CẢNH KHÔNG TRỰC TIẾP CHỨA câu trả lời, bắt đầu bằng: "Dựa trên các quy định liên quan nhất tìm thấy, ..."
5. QUY TẮC TRÍCH DẪN: Ưu tiên trích dẫn văn bản có giá trị pháp lý cao hơn (Luật → Nghị định → Thông tư → Quyết định). Chỉ trích dẫn tài liệu phụ nếu tài liệu chính không chứa chi tiết đó.
6. NGẶT NGHÈO: Tuyệt đối không nhắc đến bất kỳ tên Luật, Điều khoản, hay số liệu nào không xuất hiện chữ-nguyên-chữ trong Phần NGỮ CẢNH (hoặc trong Lịch sử nếu hỏi lại căn cứ). Nội dung hoàn toàn phải bám sát.

TIÊU CHUẨN TRÍCH DẪN BỔ SUNG:
Nếu bên dưới có "PHẦN THÔNG TIN BỔ SUNG TỪ THAM CHIẾU", hãy ưu tiên sử dụng nó để giải thích các nội dung mà câu trả lời chính nhắc tới.
{supplemental_context}
"""

REFLECTION_PROMPT = """
Bạn là Reflection Agent (Kiểm duyệt viên an toàn chống ảo giác).

BẠN PHẢI THỰC HIỆN ĐỐI CHIẾU TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

NGỮ CẢNH (Sự thật gốc - Nguồn duy nhất được phép trích dẫn):
{context}

CÂU HỎI NGƯỜI DÙNG:
{query}

CÂU TRẢ LỜI BỊ KIỂM DUYỆT CỦA AI LUẬT SƯ:
{answer}

═══════════════════════════════════════════════════
BƯỚC 1: ĐỐI CHIẾU (viết trong thẻ <thinking>)
═══════════════════════════════════════════════════
<thinking>
A. LIỆT KÊ TRÍCH DẪN: Tìm TẤT CẢ các trích dẫn pháp lý trong câu trả lời (Luật X, Điều Y, Nghị định Z, Phụ lục W, số liệu cụ thể).
   → Danh sách: [Trích dẫn 1], [Trích dẫn 2], ...

B. ĐỐI CHIẾU TỪNG TRÍCH DẪN:
   Với mỗi trích dẫn ở trên, tìm kiếm NGUYÊN VĂN trong phần NGỮ CẢNH:
   - [Trích dẫn 1] → TÌM THẤY tại [vị trí trong ngữ cảnh] / KHÔNG TÌM THẤY → ẢO GIÁC!
   - [Trích dẫn 2] → TÌM THẤY / KHÔNG TÌM THẤY → ẢO GIÁC!

C. KIỂM TRA SỐ LIỆU: Các con số (mức phạt, thời hạn, tỷ lệ) có khớp CHÍNH XÁC với ngữ cảnh không?

D. ĐÁNH GIÁ TRÍCH DẪN: Câu trả lời có mở đầu bằng "Căn cứ..." với trích dẫn cụ thể không?
   - Nếu nguồn là Phụ lục → phải ghi "Phụ lục [Tên/Số]" thay vì "Điều"
   - Nếu nguồn là nội dung chung → phải ghi "phần nội dung chung của [Tên VB]"
</thinking>

═══════════════════════════════════════════════════
BƯỚC 2: PHÁN QUYẾT (trả về JSON trong markdown code block)
═══════════════════════════════════════════════════
Quy tắc:
- Nếu phát hiện BẤT KỲ trích dẫn nào không tồn tại trong ngữ cảnh → pass: false
- Nếu câu trả lời là câu từ chối vì thiếu ngữ cảnh ("Xin lỗi, dữ liệu hiện tại...") → pass: true
- Nếu thiếu trích dẫn cụ thể (không có "Căn cứ...") → pass: false

```json
{{
    "pass": true/false,
    "citation_ok": true/false,
    "hallucination_detected": true/false,
    "relevance_ok": true/false,
    "feedback": "Phân tích rõ lý do FAIL (Vị trí nào là ảo giác, câu nào nói leo). Nhắc AI Luật Sư từ chối trả lời nếu thiếu dữ kiện."
}}
```
"""

CORRECTION_PROMPT = """
HỆ THỐNG KIỂM DUYỆT TỰ ĐỘNG ĐÃ TỪ CHỐI CÂU TRẢ LỜI VỪA RỒI CỦA BẠN VÌ LÝ DO NGHIÊM TRỌNG:
{feedback}

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI VIẾT CÂU TRẢ LỜI MỚI.

NGỮ CẢNH (Hãy đọc thật kỹ, nếu không có chữ nào về câu trả lời ở đây, TUYỆT ĐỐI không sáng tác):
{context}

Câu hỏi gốc: {query}

<thinking>
- Lỗi cụ thể mà kiểm duyệt chỉ ra là gì?
- Quét lại ngữ cảnh: có đoạn nào THỰC SỰ liên quan đến câu hỏi không?
- Nếu có → trích xuất nguyên văn để trích dẫn chính xác.
- Nếu không có → phải từ chối trả lời.
- Xác định cách trích dẫn đúng: Điều/Khoản cho nội dung chính, Phụ lục cho phụ lục, phần nội dung chung cho nội dung chung.
</thinking>

Sau khi suy luận xong, viết câu trả lời mới.
**CẢNH BÁO: BẠN PHẢI TUÂN THỦ TÍNH "CLOSED-DOMAIN"**.
Nếu phần NGỮ CẢNH KHÔNG CHỨA ĐỦ nội dung trả lời, bạn chỉ được phép trả lời:
"Xin lỗi, dữ liệu hiện tại của hệ thống không chứa thông tin hoặc quy định trực tiếp về vấn đề này."

TIÊU CHUẨN TRÍCH DẪN BỔ SUNG:
{supplemental_context}
"""

def rewrite_legal_query(query: str, llm_preset: str = None) -> dict:
    messages = [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}]
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset)
        resp = resp or ""
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        return json.loads(resp)
    except Exception as e:
        print(f"       ⚠️ Query Rewrite JSON parse failed: {e}. Fallback to raw.")
        return {"hypothetical_answer": query, "filters": {}}

def grade_documents(query: str, context: str, llm_preset: str = None) -> bool:
    if not context.strip():
        return False
    messages = [{"role": "user", "content": GRADER_PROMPT.format(context=context, query=query)}]
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset)
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        data = json.loads(resp)
        return data.get("is_relevant", "no").lower() == "yes"
    except Exception as e:
        print(f"       ⚠️ Document Grade JSON parse failed: {e}. Fallback to passing.")
        return True # Fallback pass

def transform_query(query: str, llm_preset: str = None) -> str:
    messages = [{"role": "user", "content": TRANSFORM_PROMPT.format(query=query)}]
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset)
        return strip_thinking_tags(resp)
    except Exception as e:
        print(f"       ⚠️ Query Transform failed: {e}. Fallback to original.")
        return query

def build_legal_context(hits: List[Dict[str, Any]], file_chunks: List[Dict[str, Any]] = None, max_chars: int = None) -> str:
    """
    Xây dựng Ngữ cảnh Pháp lý (Legal Context) sử dụng 3 chiến thuật nâng cao:
    1. Lost-in-the-Middle Reordering: Đưa hits quan trọng nhất ra 2 đầu.
    2. XML Context Injection: Phân tách rõ Metadata và Nội dung.
    3. Hard Character Limit: Đảm bảo không vượt quá cửa sổ ngữ cảnh LLM.
    4. Cấu trúc Nén (Compression): Loại bỏ khoảng trắng thừa để tiết kiệm token.
    """
    import re
    if max_chars is None:
        max_chars = settings.MAX_CONTEXT_CHARS

    context_parts = []
    current_chars = 0
    doc_counter = 0

    # --- Ưu tiên File Upload (File Chunks) ---
    if file_chunks:
        context_parts.append("<!-- BẮT ĐẦU: DỮ LIỆU TỪ FILE USER TẢI LÊN -->")
        for idx, f_chunk in enumerate(file_chunks, start=1):
            text = f_chunk.get("text_to_embed", f_chunk.get("unit_text", ""))
            # Nén trắng
            text = re.sub(r'\s*\n\s*', '\n', text)
            text = re.sub(r' {2,}', ' ', text).strip()
            chunk_info = f"[File Chunk {idx}]\n{text}\n"
            if current_chars + len(chunk_info) < max_chars:
                context_parts.append(chunk_info)
                current_chars += len(chunk_info)
        context_parts.append("<!-- KẾT THÚC: DỮ LIỆU TỪ FILE USER TẢI LÊN -->\n")

    # --- Chiến thuật 1: Lost-in-the-Middle Reordering ---
    if hits and len(hits) > 2:
        # Sắp xếp hits theo score giảm dần (đảm bảo input chuẩn)
        sorted_hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
        reordered = [None] * len(sorted_hits)
        
        left = 0
        right = len(sorted_hits) - 1
        for i, hit in enumerate(sorted_hits):
            # i=0 (H1) -> reordered[0], i=1 (H2) -> reordered[last], i=2 (H3) -> reordered[1]...
            if i % 2 == 0:
                reordered[left] = hit
                left += 1
            else:
                reordered[right] = hit
                right -= 1
        hits = reordered

    # --- Chiến thuật 2: XML Context Injection ---
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

        # Xác định loại nội dung để LLM xử lý đúng logic (ví dụ: bảng biểu có độ tin cậy thấp hơn điều khoản chính)
        if is_appendix:
            loai = "PHỤ LỤC / BẢNG BIỂU ĐÍNH KÈM - Đây là dữ liệu chi tiết, định mức hoặc biểu mẫu."
        else:
            loai = "NỘI DUNG CHÍNH - Quy định trực tiếp trong văn bản pháp luật."

        # Build XML
        base_law_xml = f"\n    <can_cu_phap_ly>{', '.join(base_laws)}</can_cu_phap_ly>" if base_laws else ""
        
        chunk_xml = (
            f'<tai_lieu id="{doc_counter}">\n'
            f'  <metadata>\n'
            f'    <nguon>{doc_number} ({title})</nguon>\n'
            f'    <vi_tri>{ref_citation}</vi_tri>\n'
            f'    <loai_noi_dung>{loai}</loai_noi_dung>{base_law_xml}\n'
            f'  </metadata>\n'
            f'  <noi_dung>\n'
            f'    {text}\n'
            f'  </noi_dung>\n'
            f'</tai_lieu>\n'
        )

        # --- Chiến thuật 3: Hard Character Limit ---
        if current_chars + len(chunk_xml) > max_chars:
            print(f"       ⚠️ [Context Builder] Reached character limit ({max_chars}). Skipping remaining {len(hits)-doc_counter} chunks.")
            break

        context_parts.append(chunk_xml)
        current_chars += len(chunk_xml)

    return "\n".join(context_parts)

def reflect_on_answer(query: str, context: str, answer: str, llm_preset: str = None) -> dict:
    # Truyền toàn bộ context. Không truncate vì LLM có thể đọc nhót và đánh dấu nhầm Ảo giác!
    truncated_context = context
    
    messages = [{"role": "user", "content": REFLECTION_PROMPT.format(
        context=truncated_context, query=query, answer=answer
    )}]
    try:
        resp = chat_completion(messages, temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset)
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        return json.loads(resp)
    except Exception as e:
        print(f"Reflection parse error: {e}. Auto-pass.")
        return {"pass": True, "feedback": ""}

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

def resolve_recursive_references(primary_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Duyệt qua các hits hiện tại, tìm references và truy xuất thêm.
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
                print(f"      🔗 Found internal reference: {ref_id} in {doc_num}. Fetching...")
                supp_hits = retriever.retrieve_specific_reference(doc_num, ref_id)
                if supp_hits:
                    for sh in supp_hits:
                        sh["is_supplemental"] = True
                        sh["score"] = 1.0  # Cấp score tột đỉnh để không bị loại ở khâu Lost-in-the-middle
                        new_hits.append(sh)
                        seen_refs.add(ref_key)
                        
    # Giới hạn số lượng tham chiếu bổ sung để không làm loãng context (Max 10 để bao phủ tốt hơn)
    return all_hits + new_hits[:10]


