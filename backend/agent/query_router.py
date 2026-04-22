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
        self.super_system_prompt = """
        Bạn là SIÊU ĐIỀU PHỐI (Super Router) của Trợ lý Pháp lý AI.
        Nhiệm vụ của bạn là thực hiện CÙNG LÚC 3 công việc: Viết lại câu hỏi mồ côi (kế thừa HISTORY và CONTEXT), Phân loại ý định, và Trích xuất tham số.
        
        Quy tắc BẮT BUỘC để Viết lại câu hỏi mồ côi (Standalone Query):
        1. Đọc HISTORY, CONTEXT và QUERY mới nhất.
        2. Nếu QUERY ngầm ám chỉ văn bản/chủ đề trong HISTORY ("nghị quyết này", "văn bản nêu trên"), BẮT BUỘC rà soát HISTORY để tìm SỐ HIỆU CHÍNH XÁC (VD: 53/2025/NQ-HĐND) và thay thế thẳng đại từ đó bằng số hiệu trong QUERY. Cấm để lại đại từ chỉ định.
        3. Sử dụng CONTEXT để giải quyết các đại từ nếu HISTORY quá ngắn hoặc không rõ ràng.
        4. Phục hồi hoàn toàn đại từ ("nó", "điều đó", "luật kia").
        5. (Tính năng HyDE) Bổ sung một đoạn "câu trả lời giả định" tối ưu từ khóa pháp lý (đặc biệt là tên văn bản + số hiệu) vào thẳng câu hỏi luôn để tạo thành "hypothetical_query" dựa trên QUERY.
        
        Quy tắc Phân loại (Routing):
        - SECTOR_SEARCH: Dùng khi yêu cầu tìm kiếm, tổng hợp, liệt kê, thống kê, hoặc tóm tắt các văn bản pháp luật liên quan đến một CHỦ ĐỀ, LĨNH VỰC, hoặc PHẠM TRÙ cụ thể. VD: "Văn bản nào quy định về bảo hiểm y tế?", "Liệt kê các nghị định về đất đai", "Có bao nhiêu thông tư về giáo dục?", "Tổng hợp quy định về an toàn lao động", "Tìm tài liệu liên quan đến phòng cháy chữa cháy".
        - CONFLICT_ANALYZER: Dùng khi yêu cầu **đối chiếu, so sánh** giữa một khẳng định/tình huống/văn bản với văn bản pháp luật, hoặc hỏi về **mâu thuẫn, chồng chéo, thay thế, bãi bỏ** giữa các quy định. BẮT BUỘC có tính chất "va chạm", "so sánh", hoặc "kiểm tra tính hợp pháp". VD: "Quy định A có mâu thuẫn với quy định B không?", "Nghị định X có bãi bỏ thông tư Y không?", "Nội quy công ty tôi có vi phạm luật lao động không?".
        - LEGAL_QA (Mặc định ưu tiên): Hỏi về nội dung chi tiết của MỘT văn bản cụ thể, thủ tục, hoặc trích xuất chuyên sâu. VD: "Điều 5 Nghị định 100 nói gì?", "Thủ tục xin giấy phép xây dựng?", "Cơ quan nào cấp phép?". Nếu phân vân giữa Sector_Search và Legal_QA, ưu tiên LEGAL_QA khi câu hỏi chỉ hướng về 1-2 văn bản cụ thể.
        - GENERAL_CHAT: Chào hỏi, rỗng tuếch, không liên quan luật.
        
        Quy tắc Trích xuất Bộ Lọc (Filters) (QUAN TRỌNG):
        - Chỉ trích xuất từ câu hỏi người dùng (đã qua viết lại).
        - doc_number: Bắt buộc SAO CHÉP NGUYÊN VĂN số hiệu (kể cả hậu tố như "51/2025/TT-BYT"). Tuyệt đối không cắt đuôi.
        - article_ref: CHỈ có khi user đích danh gọi tên "Điều X", "Phụ lục Y". Không tự đoán.
        - legal_type: "Luật", "Thông tư", "Nghị định"...
        - year: (2024, 2025)
        - sector: Lĩnh vực chuyên môn (Đất đai, Y tế, Giáo dục)

        TRẢ VỀ JSON DUY NHẤT:
        ```json
        {
            "reasoning": "Tại sao lại phân loại vào Intent này?",
            "intent": "LEGAL_QA | SECTOR_SEARCH | CONFLICT_ANALYZER | GENERAL_CHAT",
            "standalone_query": "[CÂU HỎI VIẾT LẠI HOÀN CHỈNH - Đã thay thế đầy đủ đại từ]",
            "hypothetical_query": "[Câu hỏi viết lại] + [CÂU TRẢ LỜI GIẢ ĐỊNH TỪ KHÓA]",
            "filters": {
                "legal_type": "...",
                "doc_number": "...",
                "article_ref": "...",
                "year": 2025,
                "sector": "..."
            }
        }
        ```
        """

    def super_route_query(self, query: str, history: list = None, conv_state: dict = None, has_file_attachment: bool = False, llm_preset: str = None) -> tuple[str, str, str, dict]:
        """Thực hiện gộp Prompt: Standalone query + Intent Routing + Metadata Extraction."""
        if history is None:
            history = []
        
        # Xử lý history thành Text để đẩy vào LLM
        history_str = "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in history[-4:]]) if history else "(Không có lịch sử)"
        
        # Xử lý conversation state thành bối cảnh
        context_str = "(Không có bối cảnh bổ sung)"
        if conv_state:
            curr_doc = conv_state.get("current_document")
            entities = conv_state.get("entities", [])
            if curr_doc or entities:
                context_str = f"Văn bản đang thảo luận: {curr_doc or 'Chưa xác định'}\nThực thể pháp lý liên quan: {', '.join(entities) if entities else 'None'}"

        user_prompt = f"CONTEXT = {context_str}\n\nHISTORY = {history_str}\n\nQUERY = {query}"
        
        messages = [
            {"role": "system", "content": self.super_system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        import re
        try:
            response_text = chat_completion(messages, temperature=0.1, llm_preset=llm_preset)
            response_text = response_text or ""
            response_text = response_text.replace("```json", "").replace("```", "").strip()
                
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                response_text = match.group(0)
            else:
                raise ValueError("Không tìm thấy khối JSON trong phản hồi.")
                
            data = json.loads(response_text)
            intent = str(data.get("intent", "GENERAL_CHAT")).upper()
            standalone_query = data.get("standalone_query", query)
            hypo_query = data.get("hypothetical_query", standalone_query)
            filters = data.get("filters", {})
            reasoning = data.get("reasoning", "N/A")
            
            # Chuẩn hóa giá trị None trong filter
            for k, v in filters.items():
                if v == "null" or v == "":
                    filters[k] = None
            
            # Only print Intent to keep terminal clean as requested
            print(f"       🎯 [SuperRouter] Intent: {intent}")

            if intent not in [RouteIntent.SECTOR_SEARCH, RouteIntent.LEGAL_QA, RouteIntent.CONFLICT_ANALYZER, RouteIntent.GENERAL_CHAT]:
                intent = RouteIntent.LEGAL_QA
                
            return intent, standalone_query, hypo_query, filters
        except Exception as e:
            print(f"SuperRouter error: {str(e)}. Fallback to LEGAL_QA and raw query.")
            return RouteIntent.LEGAL_QA, query, query, {}

router = QueryRouter()
