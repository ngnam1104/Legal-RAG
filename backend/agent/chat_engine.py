from typing import List, Dict
from backend.llm import chat_completion
from retrieval import retriever, reranker
from backend.agent.memory import ChatSessionManager

class RAGEngine:
    def __init__(self):
        self.memory = ChatSessionManager()
        self.system_prompt = """Bạn là Trợ lý AI Chuyên gia Pháp luật Việt Nam. 
1. Tôn trọng Dữ liệu (Groundedness): CHỈ sử dụng thông tin trong [CONTEXT]. Tuyệt đối KHÔNG tự bịa đặt.
2. Trả lời "Không biết": Nếu [CONTEXT] không chứa thông tin, hãy nói: "Dựa trên tài liệu hiện tại, tôi không tìm thấy thông tin quy định về vấn đề này."
3. Trích dẫn rõ ràng: Đề cập cụ thể Tên VB, số hiệu, hoặc Điều/Khoản."""

    def rewrite_query(self, history: List[Dict[str, str]], query: str, provider: str = None) -> str:
        """Sử dụng LLM để viết lại Query dựa trên ngữ cảnh ngắn hạn."""
        if not history:
            return query

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        rewrite_prompt = (
            f"Lịch sử:\n{history_text}\n\nCâu hỏi hiện tại: {query}\n\n"
            "Hãy viết lại câu hỏi thành một câu Standalone Query độc lập, rõ nghĩa. CHỈ TRẢ VỀ CÂU VIẾT LẠI."
        )

        try:
            return chat_completion([{"role": "user", "content": rewrite_prompt}], temperature=0.0, provider=provider).strip()
        except:
            return query

    def reflect_and_correct(self, query: str, context: str, draft_answer: str, provider: str = None) -> str:
        """Tự kiểm duyệt (Reflection) sinh ra câu trả lời cuối, kiểm tra xem có "ảo giác" không."""
        reflection_prompt = f"""[CONTEXT]:\n{context}\n\n[USER_QUERY]:\n{query}\n\n[DRAFT_ANSWER]:\n{draft_answer}\n
Hãy kiểm tra xem DRAFT_ANSWER có thông tin nào KHÔNG CÓ TRONG CONTEXT không. 
Nếu có, hãy sửa lại để loại bỏ chi tiết sai lệch. Nếu đã chính xác, CHỈ CẦN sinh ra nguyên văn DRAFT_ANSWER đó, không trả lời thêm câu nào khác."""
        try:
            return chat_completion([{"role": "user", "content": reflection_prompt}], temperature=0.1, provider=provider).strip()
        except:
            return draft_answer

    def chat(self, session_id: str, query: str, provider: str = None, top_k: int = 3, use_reflection: bool = True) -> dict:
        # Ngắn hạn: Ngữ cảnh từ history
        history = self.memory.get_history(session_id)
        
        # 1. ReAct / Rewrite
        standalone_query = self.rewrite_query(history, query, provider)
        
        # 2. Hybrid Search + Payload Filter (nếu có logic extract the doc_number, nhưng mặc định sẽ search bình thường)
        raw_docs = retriever.search(standalone_query, limit=10)
        
        # 3. Rerank -> Sort theo score -> Chọn Top K
        refined_docs = reranker.rerank(standalone_query, raw_docs, top_k=top_k)
        
        context_parts = []
        references = []
        for doc in refined_docs:
            snippet = f"[{doc['title']} - {doc['article_ref']}]: {doc['text']}"
            context_parts.append(snippet)
            references.append({"title": doc['title'], "article": doc['article_ref'], "score": doc['score']})
            
        context_text = "\n\n---\n\n".join(context_parts)
        
        # 4. Prompt LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"[CONTEXT]:\n{context_text}\n\n[USER_QUERY]:\n{standalone_query}\n\n[CÂU TRẢ LỜI CỦA BẠN]:"}
        ]
        
        try:
            draft_answer = chat_completion(messages, temperature=0.3, provider=provider)
            
            # 5. Reflection (Tự kiểm soát)
            if use_reflection:
                final_answer = self.reflect_and_correct(standalone_query, context_text, draft_answer, provider)
            else:
                final_answer = draft_answer

            # Lưu Memory
            self.memory.add_message(session_id, "user", query)
            self.memory.add_message(session_id, "assistant", final_answer)

            return {
                "answer": final_answer,
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
