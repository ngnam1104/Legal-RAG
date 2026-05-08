import os
import sys
import json
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.database.neo4j_client import get_neo4j_driver
from backend.models.llm_factory import chat_completion
from backend.utils.text_utils import extract_json_from_text

# Cấu hình mục tiêu
TARGET_DOCS = [
    "105/2014/NĐ-CP", "105/2016/QH13", "42/2025/NĐ-CP", "18/2008/QH12", 
    "02/2017/TTVPCP","32/2018/TT-BYT", "11/2025/TT-BYT", "77/2015/QH13", 
    "24/2024/NĐ-CP", "01/2023/TT-VPCP"
]

# Chuyên đề cho chế độ bình thường
NORMAL_CATEGORIES = [
    "Lớp 1: Siêu dữ liệu & Căn cứ",
    "Lớp 2: Phạm vi & Thời gian",
    "Lớp 3: Nội dung thực chất (Tình huống)",
    "Lớp 4: Logic xử lý xung đột (Ngoại lệ)",
    "Lớp 5: Liên kết văn bản"
]

NORMAL_PROMPT_TEMPLATE = """Bạn là một chuyên gia Pháp lý và Kiểm thử hệ thống RAG (Retrieval-Augmented Generation).
Nhiệm vụ của bạn là đọc toàn bộ nội dung của văn bản pháp luật dưới đây và tạo ra ĐÚNG 5 câu hỏi THỰC TẾ, TRỌNG TÂM và CÓ TÍNH ỨNG DỤNG CAO kèm theo câu trả lời (Ground Truth).
Yêu cầu bắt buộc đối với câu hỏi:
- KHÔNG hỏi lý thuyết suông (VD: "Luật này quy định gì?", "Điều kiện là gì?").
- PHẢI tạo ra các "Case Study" (tình huống thực tế) giống như một người dân hoặc doanh nghiệp đang gặp rắc rối pháp lý đi hỏi luật sư.
- Ví dụ về format tình huống: "Năm 2025, tôi gửi đơn xin làm chứng chỉ hành nghề dược. Tôi thực hành tại một quầy thuốc có người sở hữu là bằng trung cấp... Đến nay tôi bị từ chối với lý do XYZ. Tôi xin hỏi như vậy có đúng không? Căn cứ theo quy định nào?"
- Tập trung vào các quy định chính và cách áp dụng thực tế.
- Xử lý các tình huống ngoại lệ, điều khoản chuyển tiếp, hoặc xung đột luật.

YÊU CẦU VỀ ĐỘ CHÍNH XÁC (CHỐNG ẢO GIÁC - HALLUCINATION):
- Tuyệt đối KHÔNG BỊA ĐẶT thông tin, điều khoản, hay quy định pháp luật.
- Mọi câu trả lời và trích dẫn (Điều, Khoản) phải được lấy TRỰC TIẾP và CHÍNH XÁC từ phần nội dung văn bản được cung cấp dưới đây.

LƯU Ý QUAN TRỌNG: 
TẤT CẢ 5 câu hỏi này phải thuộc chuyên đề: "{category}".
Hãy sử dụng thêm [THÔNG TIN ĐỒ THỊ] bên dưới để tạo các câu hỏi liên kết văn bản hoặc thực thể chính xác.

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
[NỘI DUNG VĂN BẢN]:
{context}

[THÔNG TIN ĐỒ THỊ (Mối quan hệ & Thực thể)]:
{graph_metadata}
======================================

Chỉ trả về danh sách JSON hợp lệ chứa đúng 5 object. Bắt đầu bằng dấu [ và kết thúc bằng dấu ].
"""

LONG_TERM_PROMPT_TEMPLATE = """Bạn là một chuyên gia Pháp lý thiết bộ test cho Chatbot RAG dài hạn.
Nhiệm vụ: Dựa trên văn bản pháp luật dưới đây, hãy tạo ra 01 "Phiên hội thoại" (Session) gồm 6 lượt hỏi đáp liên tục có tính kế thừa ngữ cảnh cực cao, bám sát các nội dung trọng tâm và các điều khoản quan trọng nhất của văn bản. Tránh các câu hỏi đánh đố vào các điều khoản cuối cùng (thường là về điều khoản thi hành/hiệu lực).

Yêu cầu cho từng lượt (Turn):
- Lượt 1: Câu hỏi tra cứu cơ bản về văn bản {doc_id}.
- Lượt 2: Câu hỏi kế thừa lượt 1 bằng đại từ thay thế (nó, văn bản này, người đó...), yêu cầu chi tiết hơn về metadata hoặc căn cứ.
- Lượt 3 (Tình huống thực tế): Tạo ra một "Case Study" thực tế chi tiết và có độ phức tạp cao, bám sát các nội dung trọng tâm và các điều khoản quan trọng nhất của văn bản. Tránh các câu hỏi đánh đố vào các điều khoản cuối cùng (thường là về điều khoản thi hành/hiệu lực).
- Lượt 4 (Đào sâu & Phân tích): Đào sâu vào các khía cạnh logic, điều kiện loại trừ hoặc các bước xử lý tiếp theo dựa trên tình huống ở lượt 3. Yêu cầu câu hỏi có độ dài và chiều sâu chuyên môn, buộc Chatbot phải suy luận từ các quy định chính.
- Lượt 5 (Thủ tục & Hồ sơ): Hỏi về quy trình thực hiện, các bước hoặc giấy tờ cần thiết liên quan đến tình huống trên.
- Lượt 6 (Nâng cao): Hỏi về tính kế thừa, quy định chuyển tiếp hoặc mối quan hệ với các văn bản khác (Dựa trên thông tin Đồ thị).

YÊU CẦU VỀ ĐỘ CHÍNH XÁC (CHỐNG ẢO GIÁC):
- Tuyệt đối KHÔNG BỊA ĐẶT. Câu trả lời và trích dẫn phải lấy TRỰC TIẾP từ văn bản.
- CẤM TẠO RA CÁC CÂU TRẢ LỜI MANG TÍNH CHẤT PHỦ ĐỊNH HOẶC KHÁI QUÁT SAI LỆCH (Ví dụ: "Văn bản này chỉ hướng dẫn chung chung không có quy định cụ thể") nếu trong văn bản (context) thực sự có điều khoản quy định cụ thể về vấn đề đó. Phải đọc kỹ context trước khi kết luận.
- Đảm bảo tính logic xuyên suốt cả 6 lượt hỏi đáp.

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC LÀ MỘT DANH SÁCH JSON CHỨA 1 OBJECT (Session):
[
  {{
    "session_name": "Tên mô tả phiên (ví dụ: Xử lý tình huống vi phạm dược...)",
    "document_id": "{doc_id}",
    "turns": [
      {{
        "turn": 1,
        "query": "Câu hỏi lượt 1...",
        "expected_answer": "Câu trả lời mẫu...",
        "citation": "Điều... Khoản..."
      }},
      ... (đủ 6 lượt)
    ]
  }}
]

Nội dung văn bản {doc_id}:
======================================
[NỘI DUNG VĂN BẢN]:
{context}

[THÔNG TIN ĐỒ THỊ (Mối quan hệ & Thực thể)]:
{graph_metadata}
======================================

Chỉ trả về danh sách JSON hợp lệ. Bắt đầu bằng dấu [ và kết thúc bằng dấu ].
"""

def generate_normal(driver, output_file):
    all_qa_pairs = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_qa_pairs = json.load(f)
        except: pass

    new_count = 0
    for doc_id in TARGET_DOCS:
        print(f"\n--- NORMAL MODE: Processing {doc_id} ---")
        doc_data = get_doc_data_from_db(driver, doc_id)
        if not doc_data: continue

        context = doc_data["context"]
        graph_metadata = doc_data["graph_metadata"]

        for category in NORMAL_CATEGORIES:
            print(f"   >>> Category: {category}")
            prompt = NORMAL_PROMPT_TEMPLATE.format(doc_id=doc_id, context=context, graph_metadata=graph_metadata, category=category)
            response = chat_completion([{"role": "user", "content": prompt}], temperature=0.3)
            json_str = extract_json_from_text(response)
            if json_str:
                try:
                    data = json.loads(json_str)
                    qa_list = data if isinstance(data, list) else [data]
                    all_qa_pairs.extend(qa_list)
                    new_count += len(qa_list)
                except:
                    print("   ❌ Lỗi parse JSON.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã tạo {new_count} câu hỏi normal vào {output_file}")

def generate_long_term(driver, output_file):
    all_sessions = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_sessions = json.load(f)
        except: pass

    new_count = 0
    for doc_id in TARGET_DOCS:
        print(f"\n--- LONG_TERM MODE: Processing {doc_id} (Target: 2 sessions) ---")
        doc_data = get_doc_data_from_db(driver, doc_id)
        if not doc_data: continue

        context = doc_data["context"]
        graph_metadata = doc_data["graph_metadata"]

        for i in range(2):
            print(f"   🧠 Đang gọi LLM để tạo Session {i+1}/2 cho {doc_id} (Vui lòng đợi)...")
            prompt = LONG_TERM_PROMPT_TEMPLATE.format(doc_id=doc_id, context=context, graph_metadata=graph_metadata)
            response = chat_completion([{"role": "user", "content": prompt}], temperature=0.6)
            json_str = extract_json_from_text(response)
            if json_str:
                try:
                    data = json.loads(json_str)
                    sessions = data if isinstance(data, list) else [data]
                    all_sessions.extend(sessions)
                    new_count += len(sessions)
                    print(f"   ✅ Đã xong 1 session ({len(sessions[0]['turns'])} lượt).")
                    
                    # Lưu ngay lập tức để bảo toàn dữ liệu
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(all_sessions, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"   ❌ Lỗi xử lý hoặc lưu JSON: {e}")
            time.sleep(1) # Tránh rate limit nếu có

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_sessions, f, ensure_ascii=False, indent=4)
    print(f"✅ Tổng cộng đã tạo {new_count} phiên multi-turn vào {output_file}")
    
    # Xuất file text để copy cho ChatGPT
    export_path = output_file.replace(".json", "_for_chatgpt.txt")
    export_to_chatgpt_txt(all_sessions, export_path)

def export_to_chatgpt_txt(sessions, export_path):
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("=== BỘ CÂU HỎI TEST MULTI-TURN CHO CHATGPT ===\n")
        f.write("Hướng dẫn: Copy từng lượt hỏi dưới đây vào ChatGPT trong cùng một phiên chat.\n\n")
        for i, sess in enumerate(sessions):
            f.write(f"SESSION {i+1}: {sess['session_name']} (Văn bản: {sess['document_id']})\n")
            f.write("="*50 + "\n")
            for turn in sess['turns']:
                f.write(f"Lượt {turn['turn']}: {turn['query']}\n")
            f.write("\n" + "-"*50 + "\n\n")
    print(f"👉 Đã xuất file câu hỏi cho ChatGPT tại: {export_path}")

def get_doc_data_from_db(driver, doc_id):
    try:
        with driver.session() as session:
            # 1. Lấy nội dung text
            chunk_query = """
            MATCH (d:Document {document_number: $doc})
            MATCH (c:Chunk) WHERE c.id STARTS WITH d.id + '::'
            RETURN c.text AS text ORDER BY c.id
            """
            chunk_res = session.run(chunk_query, doc=doc_id)
            chunks = [r['text'] for r in chunk_res]
            if not chunks: 
                print(f"⚠️ Không tìm thấy context cho {doc_id}")
                return None
            
            # 2. Lấy thông tin quan hệ văn bản (Graph Metadata)
            rel_query = """
            MATCH (d:Document {document_number: $doc_id})
            OPTIONAL MATCH (d)-[r]->(other:Document)
            WHERE type(r) IN ['AMENDS','REPLACES','BASED_ON','GUIDES','APPLIES','ISSUED_WITH']
            RETURN type(r) AS rel_type, other.document_number AS target_doc, other.title AS target_title
            """
            rel_res = session.run(rel_query, doc_id=doc_id)
            rels = []
            for r in rel_res:
                if r['target_doc']:
                    rels.append(f"- {r['rel_type']}: {r['target_doc']} ({r['target_title']})")
            
            # 3. Lấy thực thể tiêu biểu
            ent_query = """
            MATCH (d:Document {document_number: $doc_id})
            MATCH (d)<-[:PART_OF|BELONGS_TO*1..2]-(c:Chunk)
            MATCH (c)-[:HAS_ENTITY]->(e)
            WHERE NOT labels(e)[0] IN ['Chunk', 'Document', 'LegalArticle', 'Article', 'Clause']
            RETURN labels(e)[0] AS type, e.name AS name, count(c) AS weight
            ORDER BY weight DESC LIMIT 15
            """
            ent_res = session.run(ent_query, doc_id=doc_id)
            entities = []
            for e in ent_res:
                entities.append(f"- {e['type']}: {e['name']} (Liên kết {e['weight']} đoạn)")

        context = "\n".join(chunks)
        graph_info = "Quan hệ văn bản:\n" + ("\n".join(rels) if rels else "Không có")
        graph_info += "\n\nThực thể chính:\n" + ("\n".join(entities) if entities else "Không có")
        
        return {
            "context": context[:40000] if len(context) > 40000 else context,
            "graph_metadata": graph_info
        }
    except Exception as e:
        print(f"❌ Lỗi truy vấn DB cho {doc_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Tạo bộ test QA cho Legal-RAG")
    parser.add_argument("--mode", choices=["normal", "long_term"], default="normal", help="Chế độ: normal hoặc long_term")
    args = parser.parse_args()

    print(f"🔄 Đang kết nối Neo4j...")
    driver = get_neo4j_driver()
    if not driver:
        print("❌ Kết nối Neo4j thất bại.")
        return

    if args.mode == "normal":
        output = os.path.join(os.path.dirname(__file__), "..", "tests", "qa_evaluation", "Chatbot_test_normal.json")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        generate_normal(driver, output)
    else:
        output = os.path.join(os.path.dirname(__file__), "..", "tests", "long_term_evaluation", "long_term_test_data.json")
        os.makedirs(os.path.dirname(output), exist_ok=True)
        generate_long_term(driver, output)

if __name__ == "__main__":
    main()
