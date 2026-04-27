import json
import re
from backend.llm.factory import chat_completion
from backend.utils.text_utils import strip_thinking_tags, extract_json_from_text

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
        2. Nếu QUERY tiếp nối chủ đề của văn bản đang thảo luận trong HISTORY, BẮT BUỘC phải chèn định danh/số hiệu văn bản đó vào QUERY mới. (VD: Nếu lịch sử đang bàn về Thông tư 54/2025/TT-BYT, câu hỏi "Việc bệnh viện tự bào chế..." phải được viết lại thành "Việc bệnh viện tự bào chế... theo Thông tư 54/2025/TT-BYT..."). Phải giữ nguyên ngữ cảnh pháp lý và ý định tra cứu.
        3. Phục hồi hoàn toàn đại từ ("nó", "điều đó", "luật kia"). Cấm để lại đại từ chỉ định thay cho Tên văn bản.
        4. (Tính năng HyDE) Bổ sung một đoạn "câu trả lời giả định" tối ưu từ khóa pháp lý (đặc biệt là tên văn bản + số hiệu) vào thẳng câu hỏi luôn để tạo thành "hypothetical_query" dựa trên QUERY.
        
        Quy tắc Phân loại (Routing):
        - SECTOR_SEARCH: Dùng khi yêu cầu tìm kiếm, tổng hợp, liệt kê, thống kê, hoặc tóm tắt các văn bản pháp luật liên quan đến một CHỦ ĐỀ, LĨNH VỰC, hoặc PHẠM TRÙ cụ thể. VD: "Văn bản nào quy định về bảo hiểm y tế?", "Liệt kê các nghị định về đất đai", "Có bao nhiêu thông tư về giáo dục?".
        - CONFLICT_ANALYZER: CHỈ dùng khi người dùng chủ động yêu cầu KIỂM TRA MÂU THUẪN, so sánh chồng chéo, hoặc kiểm tra tính hợp pháp của một văn bản/tình huống so với luật ban hành. KHÔNG dùng cho mục đích hỏi về "tính thay thế/cộng gộp" của hiệu lực văn bản (trường hợp đó dùng LEGAL_QA). VD: "Quy định A có mâu thuẫn với quy định B không?", "Nội quy công ty tôi có vi phạm luật không?".
        - LEGAL_QA (Mặc định ưu tiên tối cao): Dùng cho MỌI câu hỏi về nội dung pháp luật, thủ tục, hiệu lực, mức phạt, đối tượng, định nghĩa pháp lý, TRÁCH NHIỆM, QUYỀN LỢI, ĐỘ TUỔI, MỨC ĐÓNG. NGAY CẢ KHI CÂU HỎI KHÔNG CHỨA TỪ KHÓA như "luật", "nằm trong" --> VẪN PHẢI VÀO LEGAL_QA. Tuyệt đối KHÔNG chọn GENERAL_CHAT cho các câu hỏi này. NẾU TUYỆT ĐỐI PHÂN VÂN, LUÔN CHỌN LEGAL_QA.
        - GENERAL_CHAT: Chào hỏi đơn thuần, tán gẫu xã giao, không liên quan bất kỳ khía cạnh hành vi hay quy định xã hội nào.
        
        Quy tắc Trích xuất Bộ Lọc (Filters) (QUAN TRỌNG):
        - Chỉ trích xuất từ câu hỏi người dùng (đã qua viết lại).
        - doc_number: Bắt buộc SAO CHÉP NGUYÊN VĂN số hiệu (VD: 53/2025/NQ-HĐND). NẾU KHÔNG CÓ SỐ HIỆU CHUẨN nhưng CÓ ĐÍCH DANH TÊN VĂN BẢN (VD: "Luật Dược 2024", "Luật Đất đai"), hãy ĐIỀN TÊN VĂN BẢN ĐÓ VÀO ĐÂY. NẾU KHÔNG CẢ 2, ĐỂ NULL.
        - article_ref: CHỈ có khi user đích danh gọi tên "Điều X", "Phụ lục Y". Không tự đoán.
        - legal_type: CHỈ trích xuất khi người dùng nhắc ĐÍCH DANH loại văn bản (vd: "Nghị định", "Luật", "Thông tư"). NẾU KHÔNG CÓ TỪ NÀY TRONG CÂU HỎI, BẮT BUỘC ĐỂ NULL. Tuyệt đối không tự suy diễn dựa vào ngữ cảnh.
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

    def super_route_query(self, query: str, history: list = None, conv_state: dict = None, has_file_attachment: bool = False, llm_preset: str = None) -> tuple[str, str, str, dict, dict]:
        """Thực hiện gộp Prompt: Standalone query + Intent Routing + Metadata Extraction."""
        from backend.agent.utils.sub_timer import SubTimer
        timer = SubTimer("Route")
        
        if history is None:
            history = []

        # HEURISTIC FAST PATH: Nhận diện câu chào hỏi/ngắn gọn
        q_clean = query.strip().lower().rstrip("?.!")
        greetings = ["chào", "chào bạn", "hello", "hi", "hey", "tạm biệt", "cảm ơn", "thanks", "ok", "vâng", "dạ"]
        if q_clean in greetings:
            print(f"       🎯 [SuperRouter] Fast Path Detect: GENERAL_CHAT")
            return RouteIntent.GENERAL_CHAT, query, query, {}, {"Route.FastPath": 0.001}
        
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
            with timer.step("LLM_Call"):
                response_text = chat_completion(messages, temperature=0.1, llm_preset=llm_preset)
                response_text = response_text or ""
            
            with timer.step("JSON_Parse"):
                # 1. Loại bỏ thinking tags
                clean_response = strip_thinking_tags(response_text)
                
                # 2. Trích xuất JSON bằng helper mới (balanced braces)
                json_str = extract_json_from_text(clean_response)
                
                if not json_str:
                    # Fallback cuối cùng: tìm khối giữa các dấu ```json
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                    else:
                        raise ValueError("Không tìm thấy khối JSON hợp lệ trong phản hồi.")
                
                data = json.loads(json_str)
                    
                intent = str(data.get("intent", "GENERAL_CHAT")).upper()
                standalone_query = data.get("standalone_query", query)
                hypo_query = data.get("hypothetical_query", standalone_query)
                filters = data.get("filters", {})
                reasoning = data.get("reasoning", "N/A")
                
                # Chuẩn hóa giá trị None trong filter
                for k, v in filters.items():
                    if v == "null" or v == "":
                        filters[k] = None
                
                # Chỉ in Intent để log sạch theo yêu cầu
                print(f"       🎯 [SuperRouter] Intent: {intent}")

                if intent not in [RouteIntent.SECTOR_SEARCH, RouteIntent.LEGAL_QA, RouteIntent.CONFLICT_ANALYZER, RouteIntent.GENERAL_CHAT]:
                    intent = RouteIntent.LEGAL_QA
                    
            return intent, standalone_query, hypo_query, filters, timer.results()
        except Exception as e:
            print(f"SuperRouter error: {str(e)}. Fallback to LEGAL_QA and raw query.")
            return RouteIntent.LEGAL_QA, query, query, {}, {}

router = QueryRouter()
