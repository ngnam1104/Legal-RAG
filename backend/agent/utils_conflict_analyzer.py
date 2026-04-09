import json
import time
from typing import List, Dict, Any, Optional

from backend.llm.factory import chat_completion
from backend.utils.document_parser import parser
from backend.retrieval.hybrid_search import retriever
from backend.config import settings
from backend.agent.utils_legal_qa import strip_thinking_tags

# --- LAZY LOAD LOCAL EMBEDDING CACHE ---
# Chúng ta sử dụng all-MiniLM-L6-v2 vì nhỏ (~90MB), tính Cosine RẤT nhanh trên CPU.
_semantic_pruner = None

def get_pruner():
    from backend.config import settings
    global _semantic_pruner
    if _semantic_pruner is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_pruner = SentenceTransformer('all-MiniLM-L6-v2')
            print("    🧠 [Local Model] Loaded sentence-transformers (all-MiniLM-L6-v2) for Pruning.")
        except Exception as e:
            print(f"    ⚠️ [Local Model] Error loading sentence-transformers: {e}. Fallback to No Prune.")
            _semantic_pruner = False
    return _semantic_pruner

# --- PROMPTS ---

FLOW_ROUTER_PROMPT = """
Bạn là Người Điều Phối Xung Đột Pháp Lý.
Người dùng đang trong CHẾ ĐỘ PHÂN TÍCH XUNG ĐỘT và có đính kèm một File Tài Liệu.
Câu hỏi của người dùng: "{query}"

Hãy xác định chuẩn xác luồng phân tích mà người dùng muốn thực hiện:
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

Nhiệm vụ của bạn là đọc đoạn văn bản Nội quy/Quy chế/Hợp đồng của công ty và trích xuất ra 2 dạng thông tin:
1. Metadata: Ngày ban hành, Cơ quan/Tổ chức ban hành.
2. Mệnh đề (Claims): Phân rã đoạn văn thành các nguyên tử gồm [Chủ Thể] + [Hành Vi] + [Điều Kiện] + [Hệ Quả].

Đoạn văn bản:
{text}

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

Mệnh đề:
- Chủ thể: {chu_the}
- Hành vi: {hanh_vi}
- Điều kiện: {dieu_kien}
- Hệ quả: {he_qua}

Chỉ trả về đoạn văn giả định, không giải thích.
"""

JUDGE_PROMPT = """
Bạn là JUDGE AGENT (Trọng tài Sơ thẩm).
Hãy đánh giá mệnh đề nội bộ (Internal Claim) của công ty so với Căn cứ pháp luật (Legal Context) dưới đây.

BẠN PHẢI THỰC HIỆN SUY LUẬN TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{history}

NHÃN PHÁN QUYẾT: 
- Contradiction (Xung đột trái luật) 
- Entailment (Hoàn toàn hợp pháp)
- Neutral (Pháp luật không quy định / Được phép theo thỏa thuận tự do)

LUẬT CŨ VS LUẬT MỚI: 
Nếu bạn phát hiện văn bản tải lên MỚI HƠN văn bản trong CSDL và có sự mâu thuẫn → Đề xuất vô hiệu hoá văn bản cũ trên DB.

Dữ liệu Mệnh đề Kiểm tra (Từ File/Tình huống của người dùng):
+ Văn bản gốc (Nguồn sự thật): {file_context}
+ Phân rã mệnh đề: Chu_the: {chu_the} | Hanh_vi: {hanh_vi} | He_qua: {he_qua}
+ Trích xuất từ: {org} ngày {date}

Căn Cứ Pháp Luật So Sánh (DB hoặc File):
{legal_context}

═══════════════════════════════════════════════════
BƯỚC 1: SUY LUẬN BẮT BUỘC (viết trong thẻ <thinking>)
═══════════════════════════════════════════════════
<thinking>
A. Lex Superior (Thứ bậc): Dựa vào cơ quan ban hành, đối chiếu luật (nếu có base_laws: {base_laws}). Công ty đưa ra quy định so với văn bản pháp luật như thế nào? Luật cao hơn trói luật thấp hơn.

B. Lex Posterior (Thời gian): Nếu có thông tin ngày tháng, quy định mới thay thế cũ. Mệnh đề nội bộ có vi phạm luật đang hiện hành hay áp dụng sai luật CŨ (is_active=False) không?

C. Deontic Logic (Nghĩa vụ so với Phép tuỳ nghi): Quy định nội bộ BẮT BUỘC có vi phạm Lệnh CẤM của Luật không? Hay Luật KHÔNG CẤM/GIAO QUYỀN nên tính là CẤP PHÉP?

D. XÁC ĐỊNH CĂN CỨ TRÍCH DẪN:
   - Nếu nguồn là Điều/Khoản → Trích dẫn: "Căn cứ [Loại VB] [Số hiệu] [Tên VB] - Điều X, Khoản Y"
   - Nếu nguồn là Phụ lục → Trích dẫn: "Căn cứ [Loại VB] [Số hiệu] [Tên VB] - Phụ lục số/tên"
   - Nếu nguồn là Nội dung chung → Trích dẫn: "Căn cứ phần nội dung chung của [Loại VB] [Số hiệu] [Tên VB]"

E. XỬ LÝ TRUY VẤN LỊCH SỬ: Nếu query của người dùng (trong mệnh đề gốc) không phải là kiểm tra nội bộ mà là HỎI XIN METADATA ĐIỀU LUẬT vừa được nhắc tới: Hãy lấy Metadata đó từ Lịch Sử Hội Thoại và ghi vào phần kết luận (Entailment).

F. Kết luận: Dựa trên A, B, C → Nhãn nào phù hợp nhất?
   - Nếu có xung đột trực tiếp → Contradiction
   - Nếu hoàn toàn phù hợp → Entailment  
   - Nếu luật không quy định → Neutral
</thinking>

═══════════════════════════════════════════════════
BƯỚC 2: PHÁN QUYẾT (trả về JSON trong markdown code block)
═══════════════════════════════════════════════════
```json
{{
    "label": "Contradiction | Entailment | Neutral",
    "reasoning": "Tóm tắt ngắn gọn kết luận từ bước suy luận...",
    "reference_law": "Chuỗi trích dẫn 'Căn cứ...' đã xác định ở Bước 1D",
    "proposed_db_update": {{
        "is_needed": true/false,
        "old_document_number": "Số hiệu văn bản cũ cần vô hiệu hoá (nếu có)",
        "new_document_number": "Số hiệu văn bản mới tải lên thay thế (nếu có)",
        "reason": "Lý do thay thế"
    }}
}}
```
"""

REVIEWER_PROMPT = """
Bạn là REVIEWER AGENT (Hội đồng Giám đốc / Board of Review).
Đây là quyết định sơ thẩm của Judge Agent, nhằm rà soát chéo.

BẠN PHẢI THỰC HIỆN ĐỐI CHIẾU TRONG THẺ <thinking> TRƯỚC KHI TRẢ VỀ KẾT QUẢ.

Căn Cứ Pháp Luật Gốc (nguồn sống):
{legal_context}

Judge Agent Phán Quyết:
{judge_decision}

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

# --- UTILS FOR CONFLICT ---

def extract_json_conflict(text: str) -> dict:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].strip()
        return json.loads(text)
    except:
        return {}

def format_legal_context(hits: List[Dict[str, Any]]) -> str:
    import re
    parts = []
    for i, h in enumerate(hits, 1):
        ref = h.get('article_ref', 'N/A')
        text = h.get('text', '')
        
        # Nén trắng
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        
        # Giữ context mỏng (truncate nếu quá 3000 từ)
        truncated_text = text[:3000] + ("..." if len(text) > 3000 else "")
        
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
        limit=15 
    )
    
    return hits

def cross_encoder_prune_with_scores(claim_text: str, hits: List[Dict[str, Any]], top_k: int = 3) -> List[tuple]:
    """Lọc local model và trả về (hit, score)."""
    if not hits:
        return []
        
    model = get_pruner()
    if not model or len(hits) == 0:
        return [(h, h.get('score', 0)) for h in hits[:top_k]]
    
    import numpy as np
    
    # Cosine Similarity local compute
    docs = [h.get("text") or "" for h in hits]
    # Guard: filter out empty docs to avoid tokenizer errors
    docs = [d if d.strip() else "N/A" for d in docs]
    if not claim_text or not claim_text.strip():
        return [(h, h.get('score', 0)) for h in hits[:top_k]]
    
    # Encode
    claim_emb = model.encode(claim_text)
    docs_emb = model.encode(docs)
    
    # Cosine sim
    norm_claim = np.linalg.norm(claim_emb)
    norm_docs = np.linalg.norm(docs_emb, axis=1)
    
    scores = np.dot(docs_emb, claim_emb) / (norm_docs * norm_claim + 1e-10)
    
    # Zip old hits with newly computed scores, sort and return top K
    scored_hits = [ (hits[i], float(scores[i])) for i in range(len(hits)) ]
    scored_hits.sort(key=lambda x: x[1], reverse=True)
    
    return scored_hits[:top_k]

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
    
    legal_context = lex_posterior_warning + format_legal_context(hits)
    base_laws = hits[0].get("base_laws", []) if hits else []
    
    judge_resp = chat_completion(
        [{"role": "user", "content": JUDGE_PROMPT.format(
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
    
    return extract_json_conflict(judge_resp)

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
    
    review_dec = extract_json_conflict(review_resp)
    
    # Merge findings
    if review_dec.get("is_approved", True):
        return judge_dec
    else:
        return {
            "label": review_dec.get("corrected_label", judge_dec.get("label", "Neutral")),
            "reasoning": review_dec.get("final_reasoning", judge_dec.get("reasoning", "")),
            "reference_law": judge_dec.get("reference_law", "N/A"),
            "proposed_db_update": review_dec.get("proposed_db_update", judge_dec.get("proposed_db_update"))
        }
