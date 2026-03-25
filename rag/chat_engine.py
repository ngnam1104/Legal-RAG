from typing import List, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import settings  # noqa: E402
from core.db import ensure_qdrant_collection, COLLECTION_NAME  # noqa: E402
from core.llm import chat_completion  # noqa: E402
from rag.retriever import retriever  # noqa: E402


class ChatSessionManager:
    """Quản lý Memory (Lưu 7 turn chat gần nhất)."""

    def __init__(self, max_turns: int = 7):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_turns = max_turns

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"role": role, "content": content})

        max_messages = self.max_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]


class RAGEngine:
    def __init__(self):
        ensure_qdrant_collection(COLLECTION_NAME)
        self.memory = ChatSessionManager()
        self.system_prompt = """Bạn là một Trợ lý AI Chuyên gia Pháp luật Việt Nam, hoạt động dưới dạng bộ não trung tâm của một hệ thống RAG (Retrieval-Augmented Generation). 
Nhiệm vụ của bạn là đọc các trích dẫn pháp luật được cung cấp (Context) và trả lời người dùng một cách chính xác, khách quan và tuân thủ pháp luật tuyệt đối.

--- KHÔNG BAO GIỜ VI PHẠM CÁC NGUYÊN TẮC SAU ---
1. Tôn trọng Dữ liệu (Groundedness): CHỈ sử dụng thông tin có trong phần [CONTEXT]. Tuyệt đối KHÔNG tự bịa đặt, suy đoán hoặc sử dụng kiến thức bên ngoài nếu [CONTEXT] không đề cập.
2. Trả lời "Không biết": Nếu [CONTEXT] trống hoặc không chứa thông tin để trả lời [USER_QUERY], hãy nói: "Dựa trên cơ sở dữ liệu hiện tại, tôi không tìm thấy thông tin quy định về vấn đề này."
3. Trích dẫn rõ ràng: Luôn đề cập rõ tên văn bản, số hiệu, hoặc Điều/Khoản khi đưa ra thông tin.

--- HƯỚNG DẪN XỬ LÝ THEO CHẾ ĐỘ (MODE) ---
Hệ thống sẽ truyền vào một thẻ [MODE] để chỉ định loại yêu cầu. Hãy định dạng câu trả lời theo đúng yêu cầu của từng Mode:

🔴 NẾU [MODE] = "Q_AND_A" (Hỏi đáp trực tiếp)
- Mục tiêu: Trả lời trực diện, ngắn gọn và chính xác câu hỏi của người dùng.
- Định dạng: 
  + Mở đầu bằng câu trả lời trực tiếp (Có/Không, hoặc thông tin cốt lõi).
  + Diễn giải chi tiết bằng gạch đầu dòng dựa trên [CONTEXT].
  + Kết luận (nếu cần).
- Giọng điệu: Khách quan, dứt khoát như một luật sư tư vấn.

🔴 NẾU [MODE] = "DOMAIN_DISCOVERY" (Khám phá văn bản theo lĩnh vực)
- Mục tiêu: Tổng hợp, phân loại và trình bày danh sách các văn bản pháp luật liên quan.
- Định dạng:
  + Cung cấp một đoạn tóm tắt ngắn: "Tìm thấy [X] văn bản liên quan đến yêu cầu của bạn."
  + Lập danh sách các văn bản theo dạng Markdown bullet hoặc bảng (Table). 
  + Mỗi văn bản phải hiển thị rõ: Số hiệu, Tiêu đề, Cơ quan ban hành, và Ngày ban hành (nếu có trong Context).
  + KHÔNG giải thích dông dài nội dung chi tiết của từng văn bản trừ khi được yêu cầu.

🔴 NẾU [MODE] = "CONFLICT_DETECTION" (Phát hiện xung đột pháp lý)
- Mục tiêu: Phân tích sự khác biệt, mâu thuẫn hoặc sự thay thế giữa "Văn bản Mới" và "Văn bản Cũ" trong [CONTEXT].
- Định dạng:
  1. Trạng thái: (Chỉ rõ là "Có mâu thuẫn", "Thay thế một phần", "Bãi bỏ toàn bộ", hoặc "Chỉ bổ sung, không xung đột").
  2. Phân tích chi tiết:
     - Quy định tại Văn bản Cũ: [Trích xuất tóm tắt từ Context]
     - Quy định tại Văn bản Mới: [Trích xuất tóm tắt từ Context]
  3. Điểm khác biệt cốt lõi: Nêu bật điểm "vênh" nhau giữa 2 văn bản một cách dễ hiểu nhất.
  4. Kết luận áp dụng: Nhắc nhở người dùng ưu tiên áp dụng văn bản ban hành sau (theo nguyên tắc áp dụng pháp luật)."""

    def rewrite_query(self, history: List[Dict[str, str]], current_query: str, provider: str = None, model: str = None) -> str:
        """Sử dụng LLM để viết lại Standalone Query từ ngữ cảnh chat history."""
        if not history:
            return current_query

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        rewrite_prompt = (
            f"Lịch sử trò chuyện:\n{history_text}\n\n"
            f"Câu hỏi hiện tại: {current_query}\n\n"
            "Dựa trên lịch sử, hãy viết lại câu hỏi hiện tại thành một câu Standalone Query độc lập, "
            "rõ ràng ngữ nghĩa (giữ nguyên các thực thể/vbpl được đề cập). Chỉ trả về câu đã viết lại, không giải thích gì thêm."
        )

        try:
            rewritten = chat_completion(
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.0,
                provider=provider,
                model=model,
            )
            return rewritten.strip()
        except Exception as e:
            print(f"Error compiling rewrite logic, fallback to current query: {e}")
            return current_query

    def retrieve(self, standalone_query: str) -> tuple[str, list]:
        """Search Qdrant lấy top 5 chunks và format kèm metadata."""
        try:
            search_result = retriever.search(standalone_query, limit=5)

            context_parts = []
            references = []

            for hit in search_result:
                title = hit.get("title", "Không rõ văn bản")
                article = hit.get("article_ref", "Không rõ")
                text = hit.get("text", "")

                snippet = f"[{title} - {article}]: {text}"
                context_parts.append(snippet)
                references.append(
                    {"title": title, "article": article, "score": hit.get("score")}
                )

            return "\n\n---\n\n".join(context_parts), references
        except Exception as e:
            print(f"Lỗi truy xuất Qdrant: {e}")
            return "", []

    def chat(self, session_id: str, query: str, mode: str = "qa", provider: str = None, model: str = None) -> dict:
        """Quy trình Chat hỗ trợ 3 Mode: qa, related, conflict."""
        history = self.memory.get_history(session_id)
        standalone_query = self.rewrite_query(history, query, provider, model) if mode == "qa" else query
        context_text, references = self.retrieve(standalone_query)

        if mode == "qa":
            sys_mode = "Q_AND_A"
        elif mode == "related":
            sys_mode = "DOMAIN_DISCOVERY"
        elif mode == "conflict":
            sys_mode = "CONFLICT_DETECTION"
        else:
            return {
                "answer": "Chế độ không hợp lệ.",
                "standalone_query": query,
                "references": [],
            }

        messages = [{"role": "system", "content": self.system_prompt}]
        user_prompt = f"""--- ĐẦU VÀO TỪ HỆ THỐNG ---
[MODE]: {sys_mode}
[CONTEXT]:
{context_text}

[USER_QUERY]:
{standalone_query}

[CÂU TRẢ LỜI CỦA BẠN]:"""

        messages.append({"role": "user", "content": user_prompt})

        try:
            answer = chat_completion(
                messages=messages,
                temperature=0.3 if mode != "conflict" else 0.1,
                provider=provider,
                model=model,
            )

            if mode == "qa":
                self.memory.add_message(session_id, "user", query)
                self.memory.add_message(session_id, "assistant", answer)

            return {
                "answer": answer,
                "standalone_query": standalone_query,
                "references": references,
            }
        except Exception as e:
            return {
                "answer": f"Lỗi khởi tạo LLM: {str(e)}",
                "standalone_query": standalone_query,
                "references": references,
            }


rag_engine = RAGEngine()
