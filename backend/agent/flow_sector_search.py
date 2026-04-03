import json
import time
from typing import List, Dict, Any, Optional
from backend.llm.factory import chat_completion
from backend.retrieval.hybrid_search import retriever

# --- PROMPTS ---

SECTOR_TRANSFORM_PROMPT = """
Bạn là Chuyên gia Khai thác Dữ liệu. Nhiệm vụ của bạn:
1. Trích xuất từ khóa trọng tâm (keywords) từ câu hỏi để phục vụ tìm kiếm Text Search (Không cần sinh câu trả lời giả định HyDE).
2. Trích xuất các điều kiện lọc metadata (nếu có).

Câu hỏi: {query}

BẮT BUỘC TRẢ VỀ JSON hợp lệ:
{{
    "keywords": "Từ khóa cốt lõi để tìm kiếm...",
    "filters": {{
        "legal_type": "(Tùy chọn) Luật/Nghị định/Nghị quyết/...",
        "doc_number": "(Tùy chọn) Số hiệu văn bản hoặc năm, vd: 123/2024 hoặc 2024"
    }}
}}
Chỉ output ra JSON, tuyệt đối không giải thích thêm.
"""

STRICT_FILTER_PROMPT = """
Bạn là Người Kiểm Duyệt (Strict Filter).
Mục tiêu: Đọc danh sách các văn bản trả về và LOẠI BỎ những văn bản CHẮC CHẮN KHÔNG LIÊN QUAN đến truy vấn của người dùng.
Truy vấn: {query}

Danh sách văn bản:
{docs}

TRẢ VỀ DUY NHẤT một mảng JSON chứa `doc_id` của các văn bản LIÊN QUAN (ĐƯỢC GIỮ LẠI). Ví dụ: ["doc_1", "doc_2"]. Nếu không có văn bản nào liên quan, hãy trả về [].
Chỉ output ra JSON mảng, tuyệt đối không giải thích thêm.
"""

SECTOR_REPORT_PROMPT = """
Bạn là Chuyên gia Tổng hợp Pháp lý cấp cao.
Dưới đây là danh sách các văn bản pháp luật tìm được từ cơ sở dữ liệu liên quan đến chủ đề: "{query}"

NHIỆM VỤ:
Trình bày danh sách văn bản theo định dạng Markdown chuẩn xác. Danh sách đã được sắp xếp theo thời gian ở vòng ngoài, KHÔNG TỰ Ý ĐẢO LỘN THỨ TỰ.

YÊU CẦU TRÌNH BÀY (MARKDOWN):
### 📚 Danh sách văn bản pháp luật ({query})

* **[Năm] - [Số hiệu]**: [Tiêu đề văn bản]
  * *Tóm tắt ngắn gọn:* [Giải thích 2-3 câu về điểm chính liên quan đến truy vấn]

---
(Tiếp tục liệt kê cho đến hết danh sách)

Danh sách văn bản (Đã lọc & Sắp xếp):
{docs_context}
"""

# --- STANDALONE FUNCTIONS FOR LANGGRAPH ---

def transform_sector_query(query: str) -> dict:
    messages = [{"role": "user", "content": SECTOR_TRANSFORM_PROMPT.format(query=query)}]
    try:
        resp = chat_completion(messages, temperature=0.1)
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        return json.loads(resp)
    except Exception as e:
        print(f"       ⚠️ Query Transform JSON parse failed: {e}. Fallback to raw.")
        return {"keywords": query, "filters": {}}

def strict_filter_docs(query: str, hits: List[Dict]) -> List[Dict]:
    if not hits:
        return []
        
    doc_entries = []
    for idx, h in enumerate(hits):
        doc_id = f"doc_{idx}"
        # Giảm text lấy từ 300 xuống 150 ký tự để nén gọn Prompt Size
        text = h.get("text", "")[:150]
        title = h.get("title", "")
        doc_entries.append(f"[{doc_id}] {title}\n{text}")
        
    docs_text = "\n\n".join(doc_entries)
    
    messages = [{"role": "user", "content": STRICT_FILTER_PROMPT.format(query=query, docs=docs_text)}]
    try:
        resp = chat_completion(messages, temperature=0.0)
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
        allowed_ids = json.loads(resp)
        if not isinstance(allowed_ids, list):
            raise ValueError
        
        # Keep only allowed
        filtered_hits = []
        for idx, h in enumerate(hits):
            if f"doc_{idx}" in allowed_ids:
                filtered_hits.append(h)
        return filtered_hits
    except Exception as e:
        print(f"       ⚠️ Strict Filter JSON parse failed: {e}. Keeping all 15 hits.")
        return hits # Fallback pass all

def sort_and_group_docs(hits: List[Dict], file_chunks: List[Dict] = None) -> str:
    # Gom nhóm theo document_number
    unique_docs = {}
    
    for h in hits:
        doc_num = h.get("document_number", "N/A")
        if doc_num not in unique_docs:
            # Try parsing doc_year
            doc_year = 9999
            if "issuance_date" in h and h["issuance_date"]:
                try:
                    # extract year from string if possible
                    import re
                    match = re.search(r'\b(19|20)\d{2}\b', str(h["issuance_date"]))
                    if match:
                        doc_year = int(match.group())
                except: pass
                
            if doc_year == 9999: # fallback
                import re
                match = re.search(r'/(19|20)\d{2}/', doc_num)
                if match:
                    doc_year = int(match.group().replace('/', ''))
            
            unique_docs[doc_num] = {
                "title": h.get("title", "N/A"),
                "year": doc_year,
                "type": h.get("legal_type", "N/A"),
                "content_preview": h.get("text", "")[:200]
            }
            
    if file_chunks:
        unique_docs["FILE_UPLOAD"] = {
            "title": "Tài liệu người dùng cung cấp",
            "year": 3000, # Put it at the very top (we sort reverse=True)
            "type": "Văn bản đính kèm",
            "content_preview": file_chunks[0].get("text_to_embed", "")[:200]
        }
        
    # Sort by descending year
    sorted_docs = sorted(unique_docs.items(), key=lambda x: x[1]['year'], reverse=True)
    
    context_parts = []
    for doc_num, info in sorted_docs:
        display_year = "User File" if info['year'] == 3000 else info['year']
        context_parts.append(f"Năm: {display_year} | Số hiệu: {doc_num} | Loại: {info['type']} | Tiêu đề: {info['title']}\nTrích đoạn: {info['content_preview']}\n")
        
    return "\n".join(context_parts)

def generate_sector_report(query: str, docs_context: str) -> str:
    prompt = SECTOR_REPORT_PROMPT.format(query=query, docs_context=docs_context)
    messages = [
        {"role": "user", "content": prompt}
    ]
    return chat_completion(messages, temperature=0.1)

# --- CLASS COMPAT ---
class SectorSearchFlow:
    def __init__(self):
        self.retriever = retriever
    def execute(self, query: str, file_chunks: List[Dict[str, Any]] = None) -> dict:
        t0 = time.perf_counter()
        
        tf = transform_sector_query(query)
        kw = tf.get("keywords", query)
        # Tự động tuân theo quy tắc 40/10 (nếu rerank) hoặc 15 (nếu RRF)
        hits = self.retriever.search(query=kw, expand_context=False, legal_type=fi.get("legal_type"), doc_number=fi.get("doc_number"))
        filtered_hits = strict_filter_docs(query, hits)
        
        docs_context = sort_and_group_docs(filtered_hits, file_chunks)
        answer = generate_sector_report(query, docs_context)
        
        references = [{
            "title": h.get("title", ""),
            "article": h.get("article_ref", h.get("document_number", "")),
            "score": h.get("score", 0),
            "chunk_id": h.get("chunk_id", ""),
            "document_number": h.get("document_number", ""),
            "url": h.get("url", "")
        } for h in filtered_hits]
        
        return {"answer": answer, "references": references, "metrics": {"total_time": time.perf_counter()-t0}}
sector_search_flow = SectorSearchFlow()
