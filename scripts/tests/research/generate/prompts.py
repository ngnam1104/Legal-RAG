MODE1_SEARCH_PROMPT = """
Bạn là người dùng đang tìm kiếm văn bản pháp luật.
Dựa vào các trích đoạn từ các tài liệu pháp luật có liên quan cùng một chủ đề được cung cấp bên dưới, hãy tạo ra 1 câu truy vấn tìm kiếm duy nhất đại diện cho một hành vi tìm kiếm thực tế của người dùng nhằm tìm ra cụm tài liệu này.
(chọn một góc độ: tìm từ khoá chung, hoặc liên kết tên các luật, hoặc tìm theo mục đích tình huống tổng quát).

CẤU TRÚC JSON TRẢ VỀ:
Hãy trả về DUY NHẤT một JSON Object (không bọc trong markdown) có dạng:
{{
  "items": [
    {{
      "search_type": "Một nhãn phân loại (vd: general_topic, exact_name, natural_intent)",
      "query": "Nội dung câu truy vấn tìm kiếm",
      "expected_chunk_ids": ["id 1", "id 2"]
    }}
  ]
}}

METADATA CÁC VĂN BẢN (TOÀN BỘ CỤM):
{full_metadata}
"""

MODE2_QA_PROMPT = """
Bạn là một chuyên gia soạn đề thi pháp lý.
Dựa vào nội dung cụm các điều khoản luật được cung cấp bên dưới (chú trọng trường 'chunk_text'), hãy tạo ra 1 bộ Câu hỏi - Câu trả lời bám sát các văn bản này. Vì các tài liệu có chung chủ đề nhưng có thể từ các văn bản khác nhau, hãy tổng hợp để tạo câu hỏi đa chiều.

Để linh hoạt, bạn hãy chọn MỘT trong hai kịch bản sau để đặt câu hỏi:
- Kịch bản 1: Đưa cụ thể tên/số hiệu điều luật vào trong câu hỏi (vd: "Theo khoản 1 Điều 5 Luật X và Nghị định Y...").
- Kịch bản 2: KHÔNG đưa tên luật vào câu hỏi, mà chỉ cung cấp tình huống/hành vi để kiểm tra xem hệ thống có tự tìm được luật áp dụng hay không (vd: "Nếu doanh nghiệp vi phạm... thì bị xử phạt như thế nào?").

LƯU Ý: Câu trả lời ("ground_truth_answer") phải cực kỳ chính xác, tổng hợp từ các đoạn tài liệu được cấp, giải thích rõ logic và có trích dẫn điều khoản cụ thể.

CẤU TRÚC JSON TRẢ VỀ:
Hãy trả về DUY NHẤT một JSON Object (không bọc trong markdown) có dạng:
{{
  "items": [
    {{
      "qa_type": "Nhãn phân loại (vd: direct_explanation, real_world_scenario, procedural_synthesis)",
      "question": "Câu hỏi chi tiết (luân phiên kịch bản có hoặc không có số hiệu luật)",
      "includes_law_name": true/false (tuỳ kịch bản),
      "ground_truth_answer": "Câu trả lời đúng, tóm tắt ý từ các tài liệu đã cho",
      "expected_chunk_ids": ["id 1", "id 2"]
    }}
  ]
}}

METADATA VÀ NỘI DUNG CÁM ĐIỀU KHOẢN (CỤM LIÊN QUAN):
{full_metadata}
"""

MODE3_CONFLICT_PROMPT = """
Dựa vào cụm các quy định pháp luật liên quan được cung cấp dưới đây, hãy sinh ra 1 tình huống pháp lý (có thể là một mệnh đề hoặc hoàn cảnh đóng vai) đa dạng nhằm kiểm tra khả năng lập luận pháp lý và XUNG ĐỘT (Conflict Detection) trên nhiều tài liệu.

Bạn có thể chọn một trong các góc độ sau để rèn luyện (chỉ lấy 1):
- Góc độ 1: Một tình huống Vi phạm hoàn toàn hoặc Trái ngược với quy định trong các luật (label: "contradiction").
- Góc độ 2: Một tình huống Tuân thủ hoàn toàn luật nhưng có độ phức tạp cao, hoặc một hành vi được cả 2 luật cho phép/bổ sung (label: "entailment").
- Góc độ 3: Ngoại lệ hoặc Quy định chéo: tình huống yêu cầu phải đối chiếu cả tài liệu A và tài liệu B mới ra quyết định được là đúng hay sai (label: "complex_reasoning").

CẤU TRÚC JSON TRẢ VỀ:
Hãy trả về DUY NHẤT một JSON Object (không bọc trong markdown) có dạng:
{{
  "items": [
    {{
      "conflict_type": "Nhãn phân loại (vd: inter_law_contradiction, complex_entailment, cross_exception)",
      "claim": "Lời khẳng định, nội quy giả định hoặc hành vi quy trình",
      "label": "entailment, contradiction hoặc complex_reasoning",
      "expected_chunk_ids": ["id 1", "id 2"],
      "reasoning": "Lập luận phân tích logic dựa trên việc xâu chuỗi các tài liệu để chứng minh."
    }}
  ]
}}

VĂN BẢN VÀ CÁC ĐIỀU KHOẢN PHÁP LUẬT (CỤM TÀI LIỆU CÙNG CHỦ ĐỀ):
{full_metadata}
"""
