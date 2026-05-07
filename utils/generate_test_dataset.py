import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.database.neo4j_client import get_neo4j_driver
from backend.models.llm_factory import chat_completion
from backend.utils.text_utils import extract_json_from_text

# Cấu hình
TARGET_DOCS = ["43/2018/TT-BCT", "207/2025/NĐ-CP", "12/2008/TTLT-BYT-BNV"]
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "qa_evaluation", "Chatbot_test_2mode_3docs.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

CATEGORIES = [
    "Lớp 1: Siêu dữ liệu & Căn cứ",
    "Lớp 2: Phạm vi & Thời gian",
    "Lớp 3: Nội dung thực chất (Tình huống)",
    "Lớp 4: Logic xử lý xung đột (Ngoại lệ)",
    "Lớp 6: Liên kết văn bản"
]

PROMPT_TEMPLATE = """Bạn là một chuyên gia Pháp lý và Kiểm thử hệ thống RAG (Retrieval-Augmented Generation).
Nhiệm vụ của bạn là đọc toàn bộ nội dung của văn bản pháp luật dưới đây và tạo ra ĐÚNG 3 câu hỏi ĐÁNH ĐỐ, THỰC TẾ và VÔ CÙNG KHÓ kèm theo câu trả lời (Ground Truth).
Yêu cầu bắt buộc đối với câu hỏi:
- KHÔNG hỏi lý thuyết suông (VD: "Luật này quy định gì?", "Điều kiện là gì?").
- PHẢI tạo ra các "Case Study" (tình huống thực tế) giống như một người dân hoặc doanh nghiệp đang gặp rắc rối pháp lý đi hỏi luật sư.
- Ví dụ về format tình huống: "Năm 2023, tôi gửi đơn xin làm chứng chỉ hành nghề dược. Tôi thực hành tại một quầy thuốc có người sở hữu là bằng trung cấp... Đến nay tôi bị từ chối với lý do XYZ. Tôi xin hỏi như vậy có đúng không? Căn cứ theo quy định nào?"
- Đòi hỏi khả năng kết nối thông tin từ nhiều điều khoản khác nhau (Multi-hop).
- Xử lý các tình huống ngoại lệ, điều khoản chuyển tiếp, hoặc xung đột luật.

YÊU CẦU VỀ ĐỘ CHÍNH XÁC (CHỐNG ẢO GIÁC - HALLUCINATION):
- Tuyệt đối KHÔNG BỊA ĐẶT thông tin, điều khoản, hay quy định pháp luật.
- Mọi câu trả lời và trích dẫn (Điều, Khoản) phải được lấy TRỰC TIẾP và CHÍNH XÁC từ phần nội dung văn bản được cung cấp dưới đây.
- Nếu ngữ cảnh không có đủ thông tin để tạo tình huống phức tạp cho chuyên đề được yêu cầu, hãy tạo tình huống ở mức độ vừa phải nhưng ĐẢM BẢO CHÍNH XÁC 100%.

LƯU Ý QUAN TRỌNG: 
TẤT CẢ 3 câu hỏi này phải thuộc chuyên đề: "{category}".
Hãy tập trung đào sâu vào các khía cạnh khó nhất của chuyên đề này. Khai thác những tình huống hóc búa nhất.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC LÀ MỘT DANH SÁCH JSON HỢP LỆ (ARRAY OF OBJECTS):
[
  {{
    "document_id": "{doc_id}",
    "question": "Câu hỏi khó tình huống...",
    "category": "{category}",
    "answer": "Câu trả lời đúng và đầy đủ.",
    "citation": "Trích dẫn chính xác Điều, Khoản"
  }}
]

Nội dung văn bản {doc_id}:
======================================
{context}
======================================

Chỉ trả về danh sách JSON hợp lệ chứa đúng 3 object. Bắt đầu bằng dấu [ và kết thúc bằng dấu ].
"""

def main():
    print(f"🔄 Đang kết nối Neo4j...")
    driver = get_neo4j_driver()
    if not driver:
        print("❌ Lỗi: Không có kết nối Neo4j.")
        sys.exit(1)
        
    # Đọc dữ liệu JSON hiện có để append
    all_qa_pairs = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_qa_pairs = json.loads(content)
                    print(f"📥 Đã tải {len(all_qa_pairs)} câu hỏi có sẵn từ file JSON.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc file JSON cũ: {e}. Sẽ tạo danh sách mới.")
    
    new_qa_count = 0
    
    for doc_id in TARGET_DOCS:
        print(f"\n--- Đang trích xuất nội dung văn bản {doc_id} ---")
        try:
            with driver.session() as session:
                query = """
                MATCH (d:Document {document_number: $doc})
                MATCH (c:Chunk) WHERE c.id STARTS WITH d.id + '::'
                RETURN c.text AS text ORDER BY c.id
                """
                res = session.run(query, doc=doc_id)
                chunks = [r['text'] for r in res]
                
            if not chunks:
                print(f"⚠️ Không tìm thấy chunk nào cho {doc_id}.")
                continue
                
            print(f"✅ Đã tải {len(chunks)} chunks. Ghép lại thành context...")
            context = "\n".join(chunks)
            if len(context) > 60000:
                context = context[:60000] + "\n...[TRUNCATED]"
                
            # Duyệt qua từng chuyên đề
            for category in CATEGORIES:
                print(f"\n   >>> Chuyên đề: {category}")
                # Mỗi chuyên đề chạy 2 đợt, mỗi đợt 5 câu
                for batch in range(1, 3):
                    print(f"   🧠 Đang gọi LLM (Đợt {batch}/2) cho {doc_id}...")
                    prompt = PROMPT_TEMPLATE.format(doc_id=doc_id, context=context, category=category)
                    
                    # Call LLM: Giảm nhiệt độ xuống 0.3 để bám sát thực tế, chống ảo giác
                    start_time = time.time()
                    response = chat_completion([{"role": "user", "content": prompt}], temperature=0.3) 
                    print(f"   ⏱️ LLM hoàn thành sau {time.time() - start_time:.1f}s")
                    
                    # Extract JSON
                    json_str = extract_json_from_text(response)
                    if not json_str:
                        print(f"   ❌ Lỗi: LLM không trả về JSON hợp lệ.")
                        continue
                        
                    qa_list = json.loads(json_str)
                    # Sửa lỗi LLM trả về object thay vì array
                    if isinstance(qa_list, dict):
                        if "question" in qa_list:
                            qa_list = [qa_list]
                        else:
                            # Nếu dict có chứa 1 key bao bên ngoài list, ví dụ {"questions": [...]}
                            possible_list = list(qa_list.values())[0] if qa_list else []
                            qa_list = possible_list if isinstance(possible_list, list) else [qa_list]
                    
                    if not isinstance(qa_list, list):
                        qa_list = []
                        
                    print(f"   ✅ Đã tạo thành công {len(qa_list)} câu hỏi.")
                    
                    all_qa_pairs.extend(qa_list)
                    new_qa_count += len(qa_list)
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {doc_id}: {e}")

    if new_qa_count > 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
        print(f"\n🎉 ĐÃ APPEND THÀNH CÔNG {new_qa_count} CÂU HỎI MỚI. TỔNG SỐ CÂU HỎI TRONG FILE: {len(all_qa_pairs)}")
    else:
        print("\n⚠️ Không có câu hỏi mới nào được tạo.")

if __name__ == "__main__":
    main()
