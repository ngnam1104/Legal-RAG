import json
from backend.llm.factory import chat_completion

class RouteIntent(str):
    SECTOR_SEARCH = "SECTOR_SEARCH"
    LEGAL_QA = "LEGAL_QA"
    CONFLICT_ANALYZER = "CONFLICT_ANALYZER"
    GENERAL_CHAT = "GENERAL_CHAT"
    AUTO = "AUTO"

class QueryRouter:
    def __init__(self):
        self.system_prompt = """
        Bạn là Bộ định tuyến truy vấn (Query Router). 
        Nhiệm vụ của bạn là phân loại ý định trong câu hỏi của người dùng vào 1 trong 4 Luồng (Intent) và trích xuất các tham số bộ lọc (nếu có).

        BỐN Ý ĐỊNH BAO GỒM:
        1. "SECTOR_SEARCH": Tìm kiếm, liệt kê danh sách NHIỀU văn bản hoặc tìm văn bản cốt lõi (Không đi sâu giải thích nội dung. Ví dụ: "Các luật về môi trường", "Cho tôi bản 2024").
        2. "LEGAL_QA": Tư vấn nghiệp vụ, giải quyết tình huống, giải thích sâu, hoặc yêu cầu trích xuất/tóm tắt/liệt kê nội dung chi tiết bên trong MỘT văn bản (Ví dụ: "Điều 5 nói gì", "Làm sao để công chứng", "Còn hiệu lực không").
        3. "CONFLICT_ANALYZER": Yêu cầu phân tích rủi ro, rà soát xung đột hợp đồng/thỏa thuận/nội quy so với luật bảo vệ. (Thường áp dụng nếu có file đính kèm, hoặc câu hỏi nhờ rà soát nội dung).
        4. "GENERAL_CHAT": Câu hỏi thông thường, không liên quan đến pháp lý hoặc nằm ngoài khả năng.

        THAM SỐ CẦN TRÍCH XUẤT (Chỉ lấy nếu người dùng nhắc đến, nếu không thì để null):
        - legal_type: Loại văn bản (Nghị định, Thông tư, Luật, Quyết định...)
        - year: Năm ban hành (VD: 2024, 2025...)
        - sector: Lĩnh vực (VD: Thuế, Xây dựng, Hôn nhân gia đình, Lao động...)

        === QUY TẮC BẮT BUỘC (VÀNG) ===
        1. Bạn phải xuất kết quả DUY NHẤT dưới dạng JSON. Không có bất kỳ text nào nằm ngoài khối JSON.
        2. Tư duy từng bước (Chain of Thought - CoT): Trong JSON, phải có trường "reasoning" để giải thích lý do trước.

        === VÍ DỤ MẪU (Few-Shot) ===

        Câu: "Tìm các nghị định về thuế thu nhập cá nhân năm 2023"
        {"reasoning": "Yêu cầu tìm kiếm danh sách nhiều văn bản theo tiêu chí", "intent": "SECTOR_SEARCH", "filters": {"legal_type": "Nghị định", "year": 2023, "sector": "Thuế"}}

        Câu: "Công ty có được giữ bằng đại học của bản gốc nhân viên không?"
        {"reasoning": "Tình huống pháp lý cụ thể cần viện dẫn luật lao động", "intent": "LEGAL_QA", "filters": {"legal_type": null, "year": null, "sector": "Lao động"}}

        Câu: "Nội quy công ty tôi soạn thế này có đúng luật chưa?"
        {"reasoning": "Yêu cầu rà soát, đánh giá xung đột của tài liệu với quy định", "intent": "CONFLICT_ANALYZER", "filters": {"legal_type": null, "year": null, "sector": null}}

        Câu: "Luật Đất đai 2013 còn hiệu lực không? Mức phạt là bao nhiêu?"
        {"reasoning": "Hỏi chi tiết về nội dung/tình trạng của một luật cụ thể", "intent": "LEGAL_QA", "filters": {"legal_type": "Luật", "year": 2013, "sector": "Đất đai"}}

        Câu: "Hôm nay trời có mưa không?"
        {"reasoning": "Không liên quan đến pháp luật", "intent": "GENERAL_CHAT", "filters": {"legal_type": null, "year": null, "sector": null}}

        === KẾT THÚC VÍ DỤ ===

        Chỉ trả về ĐÚNG MỘT khối JSON hợp lệ theo định dạng sau:
        {
          "reasoning": "giải thích ngắn gọn",
          "intent": "SECTOR_SEARCH | LEGAL_QA | CONFLICT_ANALYZER | GENERAL_CHAT",
          "filters": {
            "legal_type": "...",
            "year": 2026,
            "sector": "..."
          }
        }
        """

    def route_query(self, query: str, has_file_attachment: bool = False) -> tuple[str, dict]:
        if has_file_attachment:
            # Nếu có file đính kèm, mặc định thường là rà soát rủi ro/xung đột
            pass

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Câu hỏi: {query}"}
        ]
        
        import re
        try:
            response_text = chat_completion(messages, temperature=0.1)
            
            # Trích xuất linh hoạt vùng JSON (Kể cả bị bọc markdown hay text thừa)
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                response_text = match.group(0)
            else:
                raise ValueError("Không tìm thấy khối JSON trong phản hồi.")
                
            data = json.loads(response_text)
            intent = data.get("intent", "GENERAL_CHAT")
            filters = data.get("filters", {})
            
            if intent in [RouteIntent.SECTOR_SEARCH, RouteIntent.LEGAL_QA, RouteIntent.CONFLICT_ANALYZER, RouteIntent.GENERAL_CHAT]:
                return intent, filters
                
            return RouteIntent.GENERAL_CHAT, filters
        except Exception as e:
            print(f"Router error: {str(e)}. Fallback to GENERAL_CHAT.")
            return RouteIntent.GENERAL_CHAT, {}

router = QueryRouter()
