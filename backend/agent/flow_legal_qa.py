from typing import List, Dict, Any, Optional
import json
import time
from backend.llm.factory import chat_completion
from backend.retrieval.hybrid_search import retriever

# --- PROMPTS ---
REWRITE_PROMPT = """
Bạn là hệ thống Tối ưu Khởi vấn Pháp lý (Legal Query Rewriter) kiêm trích xuất dữ liệu.
Nhiệm vụ: 
1. Phân tích câu hỏi gốc của người dùng. Viết một "câu trả lời giả định" (HyDE) HOẶC "cách diễn đạt lại từ 3 góc độ khác nhau" gộp thành một đoạn văn duy nhất. Tập trung vào các từ khóa pháp luật cốt lõi, tên riêng, số hiệu (nhằm mục đích vector search).
2. Trích xuất các điều kiện lọc (nếu có) như loại văn bản pháp lý (Luật, Nghị định, Thông tư...) hoặc số hiệu văn bản (123/2024).

Câu hỏi gốc: {query}

BẮT BUỘC TRẢ VỀ JSON hợp lệ với cấu trúc KHÔNG ĐƯỢC THIẾU TÊN TRƯỜNG:
{{
    "hypothetical_answer": "(Bắt buộc) Đoạn văn tóm tắt nội dung/góc độ truy vấn...",
    "filters": {{
        "legal_type": "(Tùy chọn) Luật/Nghị định/Nghị quyết/Thông tư/Quyết định/...",
        "doc_number": "(Tùy chọn) Số hiệu văn bản hoặc năm, vd: 123/2024 hoặc 2024"
    }}
}}

Chỉ output ra JSON, tuyệt đối không giải thích thêm.
"""

GRADER_PROMPT = """
Bạn là Người Đánh Giá Ngữ Cảnh (Context Grader).
Nhiệm vụ: Đọc NGỮ CẢNH được cung cấp và đánh giá xem nó có chứa DỮ LIỆU LIÊN QUAN để trả lời CÂU HỎI của người dùng hay không.
Chú ý: Bạn KHÔNG CẦN trả lời câu hỏi, chỉ cần đánh giá YES (có liên quan/có thể dùng để trả lời) hoặc NO (không liên quan/thiếu thông tin).

NGỮ CẢNH:
{context}

CÂU HỎI:
{query}

TRẢ VỀ DUY NHẤT một chuỗi JSON (không markdown, không giải thích):
{{
    "is_relevant": "yes" hoặc "no"
}}
"""

TRANSFORM_PROMPT = """
Bạn là Chuyên gia Tối ưu Truy vấn.
Lần tìm kiếm trước chưa tìm thấy kết quả phù hợp. Hãy viết lại câu hỏi dưới đây dưới một góc độ khác (ví dụ: dùng từ đồng nghĩa, hoặc bóc tách ý chính khái quát hơn) để hệ thống có thể tìm thấy.
Không giải thích, chỉ trả về CÂU HỎI MỚI.

CÂU HỎI GỐC: {query}
"""

ANSWER_PROMPT = """
BẠN LÀ MỘT HỆ THỐNG TRÍ TUỆ NHÂN TẠO PHÁP LÝ HOẠT ĐỘNG TRONG MÔI TRƯỜNG ĐÓNG (CLOSED-DOMAIN).
ĐÂY LÀ QUY TRÌNH BẮT BUỘC KHÔNG THỂ THƯƠNG LƯẠNG: BẠN CHỈ ĐƯỢC PHÉP TRẢ LỜI DỰA **HOÀN TOÀN** VÀO VĂN BẢN NGỮ CẢNH DƯỚI ĐÂY. KHÔNG BAO GIỜ SỬ DỤNG KIẾN THỨC CÓ SẴN CỦA BẠN.

NGỮ CẢNH (DO HỆ THỐNG RAG CUNG CẤP):
{context}

YÊU CẦU CHO CÂU TRẢ LỜI:
1. KIỂM TRA DỮ LIỆU: Đọc kỹ NGỮ CẢNH. Nếu NGỮ CẢNH KHÔNG TRỰC TIẾP CHỨA thông tin để trả lời đầy đủ, HÃY TRẢ LỜI CHÍNH XÁC CÂU SAU VÀ DỪNG LẠI (Không giải thích thêm):
   "Xin lỗi, dữ liệu hiện tại của hệ thống không chứa thông tin hoặc quy định trực tiếp về vấn đề này."
2. Nếu Ngữ cảnh có đủ thông tin, hãy trả lời thẳng thắn (Được/Không được, Đúng/Sai, Mức phạt là bao nhiêu) và phải trích dẫn RÕ RÀNG (Căn cứ theo Khoản X Điều Y của văn bản Z).
3. NGẶT NGHÈO: Tuyệt đối không nhắc đến bất kỳ tên Luật, Điều khoản, hay số liệu nào không xuất hiện chữ-nguyên-chữ trong Phần NGỮ CẢNH. Một con số sai lệch là vi phạm nghiêm trọng!

Câu hỏi của người dùng: {query}
"""

REFLECTION_PROMPT = """
Bạn là Reflection Agent (Kiểm duyệt viên an toàn chống ảo giác).

Nhiệm vụ: Đánh giá THẬT KHẮT KHE xem AI Luật Sư có làm đúng quy tắc môi trường kín (Closed-Domain) không. Xem xét:

1. **Kiểm tra Ảo giác (Hallucination)**: AI Luật sư có tự ý sinh ra bất kỳ đoạn nào trích dẫn "Điều X Luật Y" hay "Thông tư Z" mà BẠN KHÔNG TÌM THẤY TRONG NGỮ CẢNH ĐƯỢC CHỨA KHÔNG? Một chữ ngoài luồng cũng là ảo giác (hallucination). -> Đánh `pass: false` ngay lập tức!
2. **Trực tiếp**: Nếu câu trả lời là câu từ chối vì thiếu ngữ cảnh ("Xin lỗi, dữ liệu hiện tại..."), hãy luôn đánh `pass: true`.
3. **Trích dẫn có căn cứ (Citation)**: Nếu AI tự sáng tác câu trả lời mà thiếu trích dẫn cụ thể (Căn cứ phần nào trong Ngữ cảnh), đánh `pass: false`.

NGỮ CẢNH (Sự thật gốc):
{context}

CÂU HỎI NGƯỜI DÙNG:
{query}

CÂU TRẢ LỜI BỊ KIỂM DUYỆT CỦA AI LAWSUIT:
{answer}

Trả về JSON duy nhất:
{{
    "pass": true/false,
    "citation_ok": true/false,
    "hallucination_detected": true/false,
    "relevance_ok": true/false,
    "feedback": "Phân tích rõ lý do FAIL (Vị trí nào là ảo giác, câu nào nói leo). Nhắc AI Luật Sư từ chối trả lời nếu thiếu dữ kiện."
}}
"""

CORRECTION_PROMPT = """
HỆ THỐNG KIỂM DUYỆT TỰ ĐỘNG ĐÃ TỪ CHỐI CÂU TRẢ LỜI VỪA RỒI CỦA BẠN VÌ LÝ DO NGHIÊM TRỌNG:
{feedback}

Hãy viết lại một câu trả lời để thay thế.
**CẢNH BÁO: BẠN PHẢI TUÂN THỦ TÍNH "CLOSED-DOMAIN"**.
Nếu bạn cảm thấy thực sự phần NGỮ CẢNH cung cấp KHÔNG CHỨA ĐỦ nội dung trả lời (nhưng bạn biết ở đâu đó trên Internet), bạn **KHÔNG ĐƯỢC PHÉP NÓI LÊN**. Bạn chỉ được phép trả lời chính xác câu này:
"Xin lỗi, dữ liệu hiện tại của hệ thống không chứa thông tin hoặc quy định trực tiếp về vấn đề này."

NGỮ CẢNH (Hãy đọc thật kỹ, nếu không có chữ nào về câu trả lời ở đây, TUYỆT ĐỐI không sáng tác):
{context}

Câu hỏi gốc: {query}
"""

def rewrite_legal_query(query: str) -> dict:
    messages = [{"role": "user", "content": REWRITE_PROMPT.format(query=query)}]
    try:
        resp = chat_completion(messages, temperature=0.1)
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        return json.loads(resp)
    except Exception as e:
        print(f"       ⚠️ Query Rewrite JSON parse failed: {e}. Fallback to raw.")
        return {"hypothetical_answer": query, "filters": {}}

def grade_documents(query: str, context: str) -> bool:
    if not context.strip():
        return False
    messages = [{"role": "user", "content": GRADER_PROMPT.format(context=context, query=query)}]
    try:
        resp = chat_completion(messages, temperature=0.0)
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        data = json.loads(resp)
        return data.get("is_relevant", "no").lower() == "yes"
    except Exception as e:
        print(f"       ⚠️ Document Grade JSON parse failed: {e}. Fallback to passing.")
        return True # Fallback pass

def transform_query(query: str) -> str:
    messages = [{"role": "user", "content": TRANSFORM_PROMPT.format(query=query)}]
    try:
        resp = chat_completion(messages, temperature=0.3)
        return resp.strip()
    except Exception as e:
        print(f"       ⚠️ Query Transform failed: {e}. Fallback to original.")
        return query

def build_legal_context(hits: List[Dict[str, Any]], file_chunks: List[Dict[str, Any]] = None, max_chars: int = 25000) -> str:
    context_parts = []
    current_chars = 0
    
    if file_chunks:
        context_parts.append("--- DỮ LIỆU TỪ FILE BẠN TẢI LÊN ---")
        for idx, f_chunk in enumerate(file_chunks, start=1):
            text = f_chunk.get("text_to_embed", f_chunk.get("unit_text", ""))
            chunk_info = f"[File Chunk {idx}]\n{text}\n"
            if current_chars + len(chunk_info) < max_chars:
                context_parts.append(chunk_info)
                current_chars += len(chunk_info)
        context_parts.append("--- DỮ LIỆU TỪ HỆ THỐNG PHÁP LUẬT ---")

    if hits:
        reordered_hits = []
        for idx, hit in enumerate(hits):
            if idx % 2 == 0:
                reordered_hits.append(hit)
            else:
                reordered_hits.insert(0, hit)
        hits = reordered_hits

    for idx, hit in enumerate(hits, start=1):
        ref = hit.get("article_ref") or hit.get("reference_tag") or "N/A"
        doc_id = hit.get("document_number") or hit.get("title") or "N/A"
        text = hit.get("text", "")
        base_laws = hit.get("base_laws", [])
        base_law_info = f" | Căn cứ pháp lý gốc: {', '.join(base_laws)}" if base_laws else ""
        
        chunk_info = f"[Ngữ cảnh {idx}] {doc_id} - {ref}{base_law_info}\n{text}\n"
        
        if current_chars + len(chunk_info) > max_chars:
            break
            
        context_parts.append(chunk_info)
        current_chars += len(chunk_info)
        
    return "\n".join(context_parts)

def reflect_on_answer(query: str, context: str, answer: str) -> dict:
    messages = [{"role": "user", "content": REFLECTION_PROMPT.format(
        context=context, query=query, answer=answer
    )}]
    try:
        resp = chat_completion(messages, temperature=0.0)
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        return json.loads(resp)
    except Exception as e:
        print(f"Reflection parse error: {e}. Auto-pass.")
        return {"pass": True, "feedback": ""}

class LegalQAFlow:
    def __init__(self):
        self.retriever = retriever

    def execute(self, query: str, top_k: int = 3, use_reflection: bool = True, use_rerank: bool = True, file_chunks: List[Dict[str, Any]] = None) -> dict:
        t0 = time.perf_counter()
        
        # Step 1: Query Rewrite
        print(f"    🔍 [Phase 1/4] Rewriting Query for optimization & metadata extraction...")
        rewrite_data = rewrite_legal_query(query)
        rewritten_query = rewrite_data.get("hypothetical_answer", query)
        filters = rewrite_data.get("filters", {})
        legal_type = filters.get("legal_type")
        doc_number = filters.get("doc_number")
        t1 = time.perf_counter()
        
        # Step 2: Hybrid Retrieval
        print(f"    🔎 [Phase 2/4] Hybrid Retrieval (Qdrant + BM25) [Rerank: {use_rerank}, Top-K: {top_k}]...")
        hits = self.retriever.search(
            query=rewritten_query, 
            expand_context=True, 
            max_neighbors=5, 
            use_rerank=use_rerank,
            legal_type=legal_type,
            doc_number=doc_number
        )
        t_retrieval_end = time.perf_counter()
        
        if (legal_type or doc_number) and not hits:
            hits = self.retriever.search(
                query=rewritten_query, expand_context=True, max_neighbors=5, use_rerank=use_rerank
            )
            t_retrieval_end = time.perf_counter()

        references = []
        for h in hits:
            references.append({
                "title": h.get("title", ""),
                "article": h.get("article_ref", h.get("document_number", "")),
                "score": h.get("score", 0),
                "chunk_id": h.get("chunk_id", ""),
                "text_preview": h.get("text", "")[:200],
                "document_number": h.get("document_number", ""),
                "url": h.get("url", "")
            })
        
        if not hits and not file_chunks:
            return {"answer": "Xin lỗi, tôi không tìm thấy quy định pháp luật nào liên quan đến câu hỏi của bạn.", "references": []}

        # Step 3: Build Context & Generate Answer
        print(f"    📝 [Phase 3/4] Building Context & Generating Answer...")
        context_text = build_legal_context(hits, file_chunks=file_chunks)
        prompt = ANSWER_PROMPT.format(context=context_text, query=query)
        answer = chat_completion([{"role": "user", "content": prompt}], temperature=0.3)
        t_llm = time.perf_counter() - t_retrieval_end

        # Step 4: Reflection
        t_reflection = 0
        if use_reflection:
            print(f"    🛡️ [Phase 4/4] Reflection Agent (Anti-hallucination check)...")
            t3 = time.perf_counter()
            reflection = reflect_on_answer(query, context_text, answer)
            
            if not reflection.get("pass", True):
                feedback = reflection.get("feedback", "Cần cải thiện trích dẫn và độ chính xác.")
                correction = CORRECTION_PROMPT.format(feedback=feedback, context=context_text, query=query)
                answer = chat_completion([{"role": "user", "content": correction}], temperature=0.2)
                answer += "\n\n---\n*🔄 Câu trả lời đã được tự kiểm tra và cải thiện bởi Reflection Agent.*"
            else:
                answer += "\n\n---\n*✅ Câu trả lời đã qua kiểm duyệt chất lượng (Reflection Agent).*"
            t_reflection = time.perf_counter() - t3

        return {
            "answer": answer,
            "references": references,
            "metrics": {
                "rewrite_time": t1 - t0,
                "retrieval_time": t_retrieval_end - t1,
                "llm_time": t_llm,
                "reflection_time": t_reflection,
                "total_time": time.perf_counter() - t0
            }
        }

legal_qa_flow = LegalQAFlow()
