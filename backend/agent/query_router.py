import json
import re
from backend.models.llm_factory import chat_completion
from backend.utils.text_utils import strip_thinking_tags, extract_json_from_text

class RouteIntent(str):
    LEGAL_CHAT = "LEGAL_CHAT"
    GENERAL_CHAT = "GENERAL_CHAT"
    AUTO = "AUTO"

class QueryRouter:
    def __init__(self):
        from backend.prompt import ROUTER_PROMPT
        self.super_system_prompt = ROUTER_PROMPT

    def super_route_query(self, query: str, history: list = None, conv_state: dict = None, has_file_attachment: bool = False, llm_preset: str = None) -> tuple[str, str, str, dict, dict]:
        """Thực hiện gộp Prompt: Standalone query + Intent Routing + Metadata Extraction."""
        from backend.agent.utils_general import SubTimer
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

                if intent not in [RouteIntent.LEGAL_CHAT, RouteIntent.GENERAL_CHAT]:
                    intent = RouteIntent.LEGAL_CHAT
                    
            return intent, standalone_query, hypo_query, filters, timer.results()
        except Exception as e:
            print(f"SuperRouter error: {str(e)}. Fallback to LEGAL_CHAT and raw query.")
            return RouteIntent.LEGAL_CHAT, query, query, {}, {}

router = QueryRouter()
