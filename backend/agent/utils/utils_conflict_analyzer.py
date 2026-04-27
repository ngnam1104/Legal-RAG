import json
import time
from typing import List, Dict, Any, Optional

from backend.llm.factory import chat_completion
from backend.utils.document_parser import parser
from backend.retrieval.hybrid_search import retriever
from backend.config import settings
from backend.agent.utils.utils_legal_qa import strip_thinking_tags

# --- LAZY LOAD LOCAL EMBEDDING CACHE ---
# Chúng ta đã chuyển sang sử dụng Internal API Reranker nội bộ thay thế model python cục bộ
from backend.retrieval.reranker import reranker as api_reranker

# --- PROMPTS ---

FLOW_ROUTER_PROMPT = """
QUERY = {query}

Bạn là Người Điều Phối Xung Đột Pháp Lý.
Người dùng đang trong CHẾ ĐỘ PHÂN TÍCH XUNG ĐỘT và có đính kèm một File Tài Liệu.

Hãy xác định chuẩn xác luồng phân tích mà người dùng muốn thực hiện thông qua QUERY:
1. "VS_FILE": Người dùng muốn kiểm tra xem câu hỏi/tình huống của họ có vi phạm/xung đột với CHÍNH FILE TÀI LIỆU họ vừa tải lên hay không.
2. "VS_DB": Người dùng muốn hệ thống trích xuất thông tin/điều khoản từ FILE TÀI LIỆU, sau đó đem đi kiểm tra xung đột với CƠ SỞ DỮ LIỆU LUẬT (Qdrant DB).

Trả về DUY NHẤT 1 chuỗi JSON:
{{
   "intent": "VS_FILE" hoặc "VS_DB"
}}
"""

IE_PROMPT = """
Bạn là hệ thống Trích xuất Thông tin Pháp lý (Information Extraction JSON).

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

TEXT = {text}

Nhiệm vụ của bạn là đọc đoạn văn bản trong TEXT (Nội quy/Quy chế/Hợp đồng của công ty) và trích xuất ra 2 dạng thông tin:
1. Metadata: Ngày ban hành, Cơ quan/Tổ chức ban hành.
2. Mệnh đề (Claims): Phân rã đoạn văn thành các nguyên tử gồm [Chủ Thể] + [Hành Vi] + [Điều Kiện] + [Hệ Quả].

BƯỚC 1: Suy luận trong thẻ <thinking>
<thinking>
- Duyệt qua từng câu/đoạn trong văn bản.
- Với mỗi câu, tự hỏi: "Câu này hoàn toàn mang tính ĐỊNH NGHĨA/GIỚI THIỆU hay nó chứa một NGHĨA VỤ/LỆNH CẤM/KHOẢN THU/QUYỀN LỢI (Logic Deontic)?"
- Nếu là ĐỊNH NGHĨA → LOẠI BỎ, không trích xuất.
- Nếu là RÀNG BUỘC → Phân rã thành: Chủ thể + Hành vi + Điều kiện + Hệ quả.
</thinking>

BƯỚC 2: Trả về JSON trong markdown code block:
```json
{{
  "metadata": {{
    "ngay_ban_hanh": "string",
    "co_quan_ban_hanh": "string"
  }},
  "claims": [
    {{
      "chu_the": "string",
      "hanh_vi": "string",
      "dieu_kien": "string",
      "he_qua": "string",
      "raw_text": "câu gốc chứa mệnh đề"
    }}
  ]
}}
```
"""

HYDE_PROMPT = """
Bạn là Luật sư am hiểu Văn phong Pháp lệnh Việt Nam.
Chuyển đổi mệnh đề đời thường sau đây thành một đoạn văn bản giả định mang văn phong của Pháp luật Việt Nam.
Mục đích: Dùng văn bản giả định này để tra cứu (Vector Search) trong CSDL Luật.

CHU_THE = {chu_the}
HANH_VI = {hanh_vi}
DIEU_KIEN = {dieu_kien}
HE_QUA = {he_qua}

Mệnh đề được xác định từ các biến trên.
Chỉ trả về đoạn văn giả định, không giải thích.
"""

REVIEWER_PROMPT = """
LEGAL_CONTEXT = {legal_context}
JUDGE_DECISION = {judge_decision}

Bạn là REVIEWER AGENT (Hội đồng Giám đốc / Board of Review).
Đây là quyết định sơ thẩm (JUDGE_DECISION) của Judge Agent, nhằm rà soát chéo.

BẠN PHẢI THỰC HIỆN ĐỐI CHIẾU TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

Căn Cứ Pháp Luật Gốc (nguồn sống) chính là LEGAL_CONTEXT.

═══════════════════════════════════════════════════
BƯỚC 1: ĐỐI CHIẾU (viết trong thẻ <thinking>)
═══════════════════════════════════════════════════
<thinking>
A. KIỂM TRA TRÍCH DẪN: Judge có trích dẫn Điều/Khoản/Văn bản nào? Đối chiếu từng trích dẫn với phần "Căn Cứ Pháp Luật Gốc" ở trên:
   - [Trích dẫn X] → TÌM THẤY / KHÔNG TÌM THẤY (Hallucinated citation!)
   
B. KIỂM TRA LOGIC: Judge có bắt bẻ cứng nhắc từ ngữ không? (Ví dụ: "luật không cấm nhưng bắt lỗi vì luật không ghi rõ" → sai logic, nên là Neutral chứ không phải Contradiction)

C. KIỂM TRA PROPOSED_DB_UPDATE: Nếu Judge đề xuất vô hiệu hoá văn bản cũ, có hợp lý không? Văn bản mới có thực sự thay thế văn bản cũ không?

D. Kết luận: Judge đúng hay sai? Nếu sai, nhãn nào mới đúng?
</thinking>

═══════════════════════════════════════════════════
BƯỚC 2: PHÁN QUYẾT (trả về JSON trong markdown code block)
═══════════════════════════════════════════════════
```json
{{
    "is_approved": true/false,
    "corrected_label": "Nếu false, điền nhãn mới (Contradiction|Entailment|Neutral)",
    "final_reasoning": "Lập luận chốt hạ cuối cùng.",
    "proposed_db_update": {{
        "is_needed": true/false,
        "old_document_number": "...",
        "new_document_number": "..."
    }}
}}
```
"""

STRICT_JUDGE_PROMPT = """
HISTORY = {history}
FILE_CONTEXT = {file_context}
LEGAL_CONTEXT = {legal_context}

Bạn là STRICT JUDGE AGENT (Thẩm phán Khó tính).
Nhiệm vụ: Đánh giá mệnh đề phân rã (Internal Claim) của công ty trong FILE_CONTEXT so với Căn cứ pháp luật trong LEGAL_CONTEXT. Mọi sự khác biệt hoặc nới lỏng quy định từ công ty đều bị coi là Contradiction.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

NHÃN PHÁN QUYẾT: 
- Contradiction (Xung đột trái luật): Bất kỳ vi phạm hoặc sai lệch nào, dù là nhỏ nhất.
- Entailment (Hoàn toàn hợp pháp): Đáp ứng CHÍNH XÁC và TOÀN BỘ tiêu chuẩn luật pháp đưa ra. 
- Neutral (Pháp luật không quy định / Được phép theo thỏa thuận tự do): Thực sự chưa có luật nào cover.

Dữ liệu Mệnh đề Kiểm tra (Từ File/Tình huống của người dùng):
+ Phân rã mệnh đề: Chu_the: {chu_the} | Hanh_vi: {hanh_vi} | He_qua: {he_qua}
+ Trích xuất từ FILE_CONTEXT: nguồn {org} ngày {date}

═══════════════════════════════════════════════════
BƯỚC 1: SUY LUẬN BẮT BUỘC (viết trong thẻ <thinking>)
═══════════════════════════════════════════════════
<thinking>
A. Soi lỗi khắt khe: Luật yêu cầu A, công ty yêu cầu A và B -> có phải công ty đang tự ý thêm điều kiện làm khó người lao động? Nếu có -> Contradiction!
B. XÁC ĐỊNH CĂN CỨ TRÍCH DẪN: Bắt buộc chỉ đích danh Điều/Khoản vi phạm.
C. Kết luận: Đưa ra phán quyết tàn nhẫn.
</thinking>

═══════════════════════════════════════════════════
BƯỚC 2: PHÁN QUYẾT (trả về JSON trong markdown code block)
═══════════════════════════════════════════════════
```json
{{
    "label": "Contradiction | Entailment | Neutral",
    "reasoning": "Tóm tắt ngắn gọn lập luận khắt khe...",
    "reference_law": "Căn cứ [Loại VB] [Số hiệu] - [Điều/Khoản]",
    "proposed_db_update": {{
        "is_needed": false
    }}
}}
```
"""

def score_contradiction_by_embedding(claim_text: str, hit_text: str) -> float:
    """
    So sánh Semantic Contradiction qua API Reranker (Inverse Score).
    Điểm Rerank càng thấp (cosine distance/similarity thấp), mức độ Contradiction (sai khác ngữ nghĩa) càng cao.
    Returns: divergence score (0.0 đến 1.0).
    """
    try:
        res = api_reranker.rerank(claim_text, [hit_text], top_k=1)
        if res:
            sim_score = res[0].get("score", 0.0)
            return 1.0 - sim_score  # Invert: similarity thấp => divergence cao
    except Exception as e:
        print(f"Lỗi score_contradiction_by_embedding: {e}")
    return 0.5

# --- UTILS FOR CONFLICT ---

def extract_json_conflict(text: str) -> dict:
    import re
    import json
    try:
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception as e:
        print(f"[Fallback JSON Error]: {e}")
        return {}


def format_legal_context(hits: List[Dict[str, Any]]) -> str:
    import re
    parts = []
    for i, h in enumerate(hits, 1):
        ref = h.get('article_ref') or 'N/A'
        text = h.get('text', '')
        
        # Nén trắng
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        
        # Giữ context mỏng (truncate nếu quá 3000 từ)
        truncated_text = text # Đã gỡ bỏ giới hạn ngữ cảnh 3000 ký tự
        
        # Chuyển đổi thành XML format
        xml_block = (
            f'<tai_lieu id="{i}">\n'
            f'  <dieu_khoan>{ref}</dieu_khoan>\n'
            f'  <noi_dung>{truncated_text}</noi_dung>\n'
            f'</tai_lieu>'
        )
        parts.append(xml_block)
    return "\n\n".join(parts)


# --- STRATEGY DELEGATED FUNCTIONS ---

def route_conflict_intent(query: str, llm_preset: str = None) -> str:
    """Xác định luồng hỏi vs file hay hỏi vs db khi có file."""
    resp = chat_completion(
        [{"role": "user", "content": FLOW_ROUTER_PROMPT.format(query=query)}],
        temperature=0.1, model=settings.LLM_ROUTING_MODEL, llm_preset=llm_preset
    )
    data = extract_json_conflict(resp)
    return data.get("intent", "VS_DB")

def extract_claims_from_text(chunk_text: str, llm_preset: str = None) -> dict:
    """Gọi LLM (Understand) phân tích 1 khối văn bản."""
    ie_resp = chat_completion(
        [{"role": "user", "content": IE_PROMPT.format(text=chunk_text)}], 
        temperature=0.1, 
        model=settings.LLM_ROUTING_MODEL,
        llm_preset=llm_preset
    )
    return extract_json_conflict(ie_resp)

def hyde_retrieve(claim: dict, use_rerank: bool = True, llm_preset: str = None) -> List[Dict[str, Any]]:
    """Tạo câu query giả định và query Qdrant (Retrieve)."""
    hyde_query = chat_completion(
        [{"role": "user", "content": HYDE_PROMPT.format(
            chu_the=claim.get("chu_the", ""), hanh_vi=claim.get("hanh_vi", ""),
            dieu_kien=claim.get("dieu_kien", ""), he_qua=claim.get("he_qua", "")
        )}], 
        temperature=0.1, 
        model=settings.LLM_ROUTING_MODEL,
        llm_preset=llm_preset
    )
    
    # Guard: nếu LLM trả về None/empty, fallback về raw_text của claim
    if not hyde_query or not hyde_query.strip():
        hyde_query = claim.get("raw_text") or claim.get("hanh_vi") or "quy định pháp luật"
    hyde_query = strip_thinking_tags(hyde_query).strip() or hyde_query
    
    # 2. Retrieve with Time-Travel (include_inactive) để kiểm tra Lex Posterior
    # Trả về Broad result 10 chunks
    hits = retriever.search(
        query=hyde_query, 
        expand_context=True, 
        use_rerank=use_rerank, 
        include_inactive=True,
        limit=settings.MAX_RETRIEVAL_HITS 
    )
    
    return hits

def cross_encoder_prune_with_scores(claim_text: str, hits: List[Dict[str, Any]], top_k: int = 3) -> List[tuple]:
    """(Grade) Dùng API Reranker nội bộ để cắt tỉa hits."""
    if not hits:
        return []
        
    if not claim_text or not claim_text.strip():
        return [(h, h.get('score', 0)) for h in hits[:top_k]]
    
    try:
        # Sử dụng API Reranker hiện có để rerank
        enriched_hits = api_reranker.rerank_candidates(query=claim_text, candidates=hits, top_k=top_k)
        
        # Trả về list tuple (hit, score) theo định dạng cũ
        scored_hits = [(h, float(h.get('rerank_score', 0))) for h in enriched_hits]
        return scored_hits
    except Exception as e:
        print(f"    ⚠️ [Grade] Lỗi gọi API Reranker: {e}. Fallback to No Prune.")
        return [(h, h.get('score', 0)) for h in hits[:top_k]]

def cross_encoder_prune(claim_text: str, hits: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """(Grade) Wrapper cũ giữ nguyên tương thích."""
    scored = cross_encoder_prune_with_scores(claim_text, hits, top_k)
    return [h for h, s in scored]

def judge_claim(claim: dict, hits: List[Dict[str, Any]], metadata: dict, history_str: str = "", llm_preset: str = None) -> dict:
    """(Generate) Chạy Judge Agent."""
    if not hits:
        return {
            "label": "Neutral",
            "reasoning": "Không tìm thấy căn cứ pháp lý liên quan trên CSDL.",
            "reference_law": "N/A"
        }
        
    lex_posterior_warning = ""
    active_docs = [h for h in hits if h.get("is_active", True)]
    inactive_docs = [h for h in hits if not h.get("is_active", True)]
    
    if inactive_docs and active_docs:
        oldest_inactive = inactive_docs[0]
        latest_active = active_docs[0]
        ref = oldest_inactive.get("article_ref") or oldest_inactive.get("document_number")
        new_ref = latest_active.get("article_ref") or latest_active.get("document_number")
        lex_posterior_warning = f"\n[CẢNH BÁO LEX POSTERIOR]: Phát hiện quy định tại '{ref}' đã HẾT HIỆU LỰC (is_active=False), có thể bị thay thế bởi '{new_ref}'.\n\n"
    
    deontic_warning = metadata.get("deontic_context", "")
    legal_context = lex_posterior_warning + deontic_warning + format_legal_context(hits)
    base_laws = hits[0].get("base_laws", []) if hits else []
    
    judge_resp = chat_completion(
        [{"role": "user", "content": STRICT_JUDGE_PROMPT.format(
            history=history_str,
            file_context=claim.get("raw_text", "N/A"),
            chu_the=claim.get("chu_the", ""), hanh_vi=claim.get("hanh_vi", ""), he_qua=claim.get("he_qua", ""),
            org=metadata.get("co_quan_ban_hanh", "N/A"), date=metadata.get("ngay_ban_hanh", "N/A"),
            legal_context=legal_context, base_laws=base_laws
        )}], 
        temperature=0.1, 
        model=settings.LLM_CORE_MODEL,
        llm_preset=llm_preset
    )
    from backend.utils.text_utils import extract_thinking_and_answer
    thinking, answer_json = extract_thinking_and_answer(judge_resp)
    
    res = extract_json_conflict(answer_json)
    res["thinking_content"] = thinking
    return res

def review_claim(claim: dict, judge_dec: dict, hits: List[Dict[str, Any]], llm_preset: str = None) -> dict:
    """(Reflect) Chạy Reviewer Agent để chốt phán quyết chống Hallucination."""
    if not hits: # Skip if no hits
        return judge_dec
        
    legal_context = format_legal_context(hits)
    
    review_resp = chat_completion(
        [{"role": "user", "content": REVIEWER_PROMPT.format(
            legal_context=legal_context, 
            judge_decision=json.dumps(judge_dec, ensure_ascii=False)
        )}], 
        temperature=0.1, 
        model=settings.LLM_ROUTING_MODEL,
        llm_preset=llm_preset
    )
    from backend.utils.text_utils import extract_thinking_and_answer
    thinking, answer_json = extract_thinking_and_answer(review_resp)
    
    review_dec = extract_json_conflict(answer_json)
    
    # Merge findings
    if review_dec.get("is_approved", True):
        judge_dec["thinking_content"] = judge_dec.get("thinking_content", "").strip() + "\n\n" + thinking.strip()
        return judge_dec
    else:
        return {
            "label": review_dec.get("corrected_label", judge_dec.get("label", "Neutral")),
            "reasoning": review_dec.get("final_reasoning", judge_dec.get("reasoning", "")),
            "reference_law": judge_dec.get("reference_law", "N/A"),
            "proposed_db_update": review_dec.get("proposed_db_update", judge_dec.get("proposed_db_update")),
            "thinking_content": judge_dec.get("thinking_content", "").strip() + "\n\n" + thinking.strip()
        }
