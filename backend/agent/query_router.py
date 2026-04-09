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
        Nhiệm vụ của bạn là phân loại câu hỏi của người dùng vào 1 trong 4 Luồng:
        1. SECTOR_SEARCH: Câu hỏi xin danh sách, liệt kê các văn bản pháp luật, tra cứu khung pháp lý tổng thể.
        2. LEGAL_QA: Câu hỏi nghiệp vụ, tình huống pháp lý cụ thể cần tư vấn, cần viện dẫn điều khoản.
        3. CONFLICT_ANALYZER: Yêu cầu rà soát, đối chiếu nội dung file/nội quy với pháp luật để tìm điểm trái luật.
        4. GENERAL_CHAT: Câu hỏi thông thường KHÔNG liên quan đến pháp lý.

        === VÍ DỤ MẪU (Few-Shot) ===

        Câu hỏi: "Liệt kê cho tôi các văn bản hiện hành về PCCC"
        {"intent": "SECTOR_SEARCH"}

        Câu hỏi: "Tổng hợp luật đất đai từ 2013 đến 2024"
        {"intent": "SECTOR_SEARCH"}

        Câu hỏi: "Có những nghị định nào liên quan đến bảo hiểm xã hội?"
        {"intent": "SECTOR_SEARCH"}

        Câu hỏi: "Công ty có được giữ bằng đại học của nhân viên không?"
        {"intent": "LEGAL_QA"}

        Câu hỏi: "Mức phạt đi muộn theo Bộ luật Lao động là bao nhiêu?"
        {"intent": "LEGAL_QA"}

        Câu hỏi: "Theo Luật Đất đai 2024, điều kiện thu hồi đất là gì?"
        {"intent": "LEGAL_QA"}

        Câu hỏi: "Kiểm tra file hợp đồng này có vi phạm luật không?"
        {"intent": "CONFLICT_ANALYZER"}

        Câu hỏi: "Nội quy công ty tôi có trái Bộ luật Lao động không?"
        {"intent": "CONFLICT_ANALYZER"}

        Câu hỏi: "Xin chào, bạn là ai?"
        {"intent": "GENERAL_CHAT"}

        Câu hỏi: "1+1 bằng mấy?"
        {"intent": "GENERAL_CHAT"}

        === KẾT THÚC VÍ DỤ ===

        Chỉ trả về JSON hợp lệ theo định dạng: {"intent": "SECTOR_SEARCH" | "LEGAL_QA" | "CONFLICT_ANALYZER" | "GENERAL_CHAT"}
        Tuyệt đối không giải thích thêm.
        """

    def route_query(self, query: str, has_file_attachment: bool = False) -> str:
        if has_file_attachment:
            # Nếu có file đính kèm, mặc định thường là rà soát rủi ro/xung đột
            # (Trừ khi user hỏi cụ thể khác, nhưng an toàn nhất là kích hoạt Conflict Analyzer / Document QA)
            pass

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Câu hỏi: {query}"}
        ]
        
        try:
            # Force JSON mode by prompt
            response_text = chat_completion(messages, temperature=0.1)
            
            # Extract JSON block if surrounded by markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
                
            data = json.loads(response_text)
            intent = data.get("intent", "LEGAL_QA")
            
            if intent in [RouteIntent.SECTOR_SEARCH, RouteIntent.LEGAL_QA, RouteIntent.CONFLICT_ANALYZER, RouteIntent.GENERAL_CHAT]:
                return intent
                
            return RouteIntent.LEGAL_QA # default fallback
        except Exception as e:
            print(f"Router error: {str(e)}. Fallback to LEGAL_QA.")
            return RouteIntent.LEGAL_QA

router = QueryRouter()
