import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from backend.llm.factory import chat_completion
from backend.utils.document_parser import parser
from backend.retrieval.hybrid_search import retriever

# --- PROMPTS ---
IE_PROMPT = """
Bạn là hệ thống Trích xuất Thông tin Pháp lý (Information Extraction JSON).
Nhiệm vụ của bạn là đọc đoạn văn bản Nội quy/Quy chế/Hợp đồng của công ty và trích xuất ra 2 dạng thông tin:
1. Metadata: Ngày ban hành, Cơ quan/Tổ chức ban hành.
2. Mệnh đề (Claims): Phân rã đoạn văn thành các nguyên tử gồm [Chủ Thể] + [Hành Vi] + [Điều Kiện] + [Hệ Quả].

Trả về DUY NHẤT một chuỗi JSON chuẩn THEO ĐÚNG CẤU TRÚC SAU (không gộp trong markdown), không giải thích:
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

Đoạn văn bản:
{text}
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

SUY LUẬN CHAIN-OF-THOUGHT BẮT BUỘC:
1. Lex Superior (Thứ bậc): Công ty đưa ra quy định so với {base_laws} (nếu có). Luật cao hơn trói luật thấp hơn.
2. Lex Posterior (Thời gian): (Nếu có thông tin ngày tháng, quy định mới thay thế cũ).
3. Deontic Logic (Nghĩa vụ so với Lệnh cấm): Quy định nội bộ BẮT BUỘC có vi phạm Lệnh CẤM của Luật không? Hay Luật KHÔNG CẤM nên tính là CẤP PHÉP?

NHÃN PHÁN QUYẾT: 
- Contradiction (Xung đột trái luật) 
- Entailment (Hoàn toàn hợp pháp)
- Neutral (Pháp luật không quy định / Được phép theo thỏa thuận tự do)

Dữ liệu Mệnh đề Nội quy:
+ Chu_the: {chu_the} | Hanh_vi: {hanh_vi} | He_qua: {he_qua}
+ Trích xuất từ file ban hành bởi: {org} ngày {date}

Căn Cứ Pháp Luật Tìm Được:
{legal_context}

Trả về JSON duy nhất (không có markdown code block):
{{
    "label": "Contradiction | Entailment | Neutral",
    "reasoning": "Chuỗi suy luận chi tiết của bạn...",
    "conflict_text": "Đoạn văn bị xung đột (nếu có, nếu không để rỗng)",
    "reference_law": "Tên văn bản và Điều Khoản áp dụng"
}}
"""

REVIEWER_PROMPT = """
Bạn là REVIEWER AGENT (Kiểm duyệt viên / Hội đồng Bào chữa).
Đây là quyết định sơ thẩm của Judge Agent cho một quy định của Công ty, dựa trên một tập Căn cứ pháp luật.
Nhiệm vụ của bạn: Kiểm tra xem Judge Agent có bị ẢO GIÁC (hallucination) không? Có trích dẫn sai luật không? Lập luận Deontic Logic có quá cứng nhắc không?

Căn Cứ Pháp Luật Gốc:
{legal_context}

Judge Agent Phán Quyết:
{judge_decision}

Nếu Judge bị sai hoặc ảo giác văn bản, sửa lại "is_approved": false và cung cấp bản sửa đổi.
Trả về JSON (không có markdown code block):
{{
    "is_approved": true/false,
    "corrected_label": "Nếu false, điền nhãn mới, nếu true để nguyên",
    "final_reasoning": "Lập luận chốt hạ cuối cùng."
}}
"""

# --- UTILS ---

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
    parts = []
    for i, h in enumerate(hits, 1):
        ref = h.get('article_ref', 'N/A')
        text = h.get('text', '')
        parts.append(f"[{i}] {ref}\n{text}")
    return "\n".join(parts)

# --- STANDALONE FUNCTIONS FOR LANGGRAPH ---

async def analyze_single_claim(claim: dict, metadata: dict) -> dict:
    # 1. HyDE
    hyde_query = chat_completion([{"role": "user", "content": HYDE_PROMPT.format(
        chu_the=claim.get("chu_the", ""), hanh_vi=claim.get("hanh_vi", ""),
        dieu_kien=claim.get("dieu_kien", ""), he_qua=claim.get("he_qua", "")
    )}], temperature=0.3)
    
    # 2. Retrieve with Time-Travel (include_inactive)
    # Tự động tuân theo quy tắc 40/10 (nếu rerank) hoặc 15 (nếu RRF)
    hits = retriever.search(query=hyde_query, expand_context=True, include_inactive=True)
    
    # 2.5 Conflict Mining (Lex Posterior Logic)
    lex_posterior_warning = ""
    active_docs = [h for h in hits if h.get("is_active", True)]
    inactive_docs = [h for h in hits if not h.get("is_active", True)]
    
    if inactive_docs and active_docs:
        oldest_inactive = inactive_docs[0]
        latest_active = active_docs[0]
        ref = oldest_inactive.get("article_ref") or oldest_inactive.get("document_number")
        new_ref = latest_active.get("article_ref") or latest_active.get("document_number")
        lex_posterior_warning = f"\n[CẢNH BÁO KHAI PHÁ DỮ LIỆU - LEX POSTERIOR]: Phát hiện quy định tại '{ref}' đã HẾT HIỆU LỰC (is_active=False), có thể bị thay thế bởi '{new_ref}'. Hãy kiểm tra xem Mệnh đề của người dùng có đang áp dụng sai luật cũ không.\n\n"
    
    legal_context = lex_posterior_warning + format_legal_context(hits)
    # Giới hạn cứng 12,000 ký tự để tránh lỗi 413 trên Groq Free/Dev Tier
    if len(legal_context) > 12000:
        legal_context = legal_context[:12000] + "\n... (Dữ liệu quá dài, đã được cắt bớt để tối ưu hóa bộ nhớ) ..."
        
    base_laws = hits[0].get("base_laws", []) if hits else []
    
    # 3. Judge
    judge_resp = chat_completion([{"role": "user", "content": JUDGE_PROMPT.format(
        chu_the=claim.get("chu_the", ""), hanh_vi=claim.get("hanh_vi", ""), he_qua=claim.get("he_qua", ""),
        org=metadata.get("co_quan_ban_hanh", "N/A"), date=metadata.get("ngay_ban_hanh", "N/A"),
        legal_context=legal_context, base_laws=base_laws
    )}], temperature=0.2)
    judge_dec = extract_json_conflict(judge_resp)
    
    # 4. Review
    review_resp = chat_completion([{"role": "user", "content": REVIEWER_PROMPT.format(
        legal_context=legal_context, judge_decision=json.dumps(judge_dec, ensure_ascii=False)
    )}], temperature=0.2)
    review_dec = extract_json_conflict(review_resp)
    
    final_label = review_dec.get("corrected_label") if not review_dec.get("is_approved", True) else judge_dec.get("label", "Neutral")
    final_reason = review_dec.get("final_reasoning", judge_dec.get("reasoning", ""))
    
    return {
        "claim": claim.get("raw_text", f"{claim.get('chu_the')} {claim.get('hanh_vi')}"),
        "label": final_label,
        "reference_law": judge_dec.get("reference_law", "N/A"),
        "conflict_reasoning": final_reason,
        "hits": hits # for references
    }

# --- COMPATIBILITY CLASS ---

class ConflictAnalyzerFlow:
    def __init__(self):
        self.retriever = retriever

    def process_file(self, file_path: str) -> dict:
        t_start = time.perf_counter()
        print(f"    🔎 [Phase 1/3] File Loading & Deep Parsing...")
        chunks = parser.parse_and_chunk(file_path)
        if not chunks:
            return {"answer": "Không trích xuất được văn bản từ file.", "references": []}

        sample_chunks = chunks[:3]
        print(f"       ✅ Done: Parsed {len(chunks)} chunks. Analyzing top {len(sample_chunks)}.")
        
        final_results = []
        all_references = []
        seen_chunk_ids = set()
        
        print(f"    ⚖️ [Phase 2/3] Multi-Agent Conflict Analysis (Parallelized)...")
        
        async def run_analysis():
            results = []
            for idx, c in enumerate(sample_chunks, 1):
                chunk_text = c.get("text_to_embed", c.get("unit_text", ""))
                ie_resp = chat_completion([{"role": "user", "content": IE_PROMPT.format(text=chunk_text)}], temperature=0.1)
                ie_data = extract_json_conflict(ie_resp)
                metadata = ie_data.get("metadata", {})
                claims = ie_data.get("claims", [])
                
                # Parallelize claims within chunk
                chunk_tasks = [analyze_single_claim(claim, metadata) for claim in claims]
                chunk_results = await asyncio.gather(*chunk_tasks)
                results.extend(chunk_results)
            return results

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        final_results = loop.run_until_complete(run_analysis())
        
        # Deduplicate references
        for r in final_results:
            for h in r.get("hits", []):
                cid = h.get("chunk_id", "")
                if cid and cid not in seen_chunk_ids:
                    all_references.append({
                        "title": h.get("title", ""),
                        "article": h.get("article_ref", h.get("document_number", "")),
                        "score": h.get("score", 0),
                        "chunk_id": cid,
                        "text_preview": h.get("text", "")[:200],
                        "legal_type": h.get("legal_type", "Chưa rõ"),
                        "document_number": h.get("document_number", ""),
                        "url": h.get("url", "")
                    })
                    seen_chunk_ids.add(cid)

        print(f"    📝 [Phase 3/3] Generating final analysis report...")
        md_table = "### ⚠️ Kết Quả Phân Tích Xung Đột Pháp Lý\n\n"
        md_table += "| Mệnh đề Nội quy (Claim) | Phán quyết | Căn cứ Pháp lý | Giải thích chi tiết |\n"
        md_table += "| :--- | :---: | :--- | :--- |\n"
        
        for r in final_results:
            icon = "❌" if "contradiction" in str(r['label']).lower() else ("✅" if "entailment" in str(r['label']).lower() else "⚪")
            md_table += f"| {r['claim']} | {icon} **{r['label']}** | {r['reference_law']} | {r['conflict_reasoning']} |\n"
        
        print(f"       ✅ Total latency: {time.perf_counter()-t_start:.1f}s")
        return {"answer": md_table, "references": all_references}

conflict_analyzer_flow = ConflictAnalyzerFlow()
