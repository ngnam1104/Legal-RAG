import os
import sys
import streamlit as st

# Allow importing from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chat_engine import rag_engine
from rag.retriever import retriever
from rag.document_manager import document_manager

st.set_page_config(page_title="Vibecoding Chatbot Pháp Luật VN", page_icon="⚖️", layout="wide")

st.title("⚖️ Vibecoding Chatbot Trợ Lý Pháp Luật Việt Nam")
st.markdown("Hệ thống tra cứu, phân tích tương quan và phát hiện xung đột văn bản pháp luật sử dụng **Qdrant** và **Llama 3**.")

# -----------------
# 1. SIDEBAR: Tùy chỉnh & Upload
# -----------------
with st.sidebar:
    st.header("⚙️ Cài đặt Mô Hình (LLM)")
    llm_provider = st.selectbox("Chọn Provider:", options=["Groq", "Gemini"], index=0)
    if llm_provider == "Groq":
        llm_model = st.selectbox("Chọn Model:", options=["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"])
    else:
        llm_model = st.selectbox("Chọn Model:", options=["gemini-3-flash-preview", "gemini-1.5-pro-latest"])
    
    st.divider()
    st.header("⚙️ Cài đặt Tìm Kiếm")
    
    chat_mode = st.radio(
        "Chế độ Chat (Mode):",
        options=["Hỏi đáp (Q&A)", "Tìm VBPL Liên Quan", "Khai Phá Xung Đột"],
        index=0
    )
    
    # Map name back to string mode code
    mode_map = {
        "Hỏi đáp (Q&A)": "qa",
        "Tìm VBPL Liên Quan": "related",
        "Khai Phá Xung Đột": "conflict"
    }
    current_mode = mode_map[chat_mode]
    
    st.divider()
    filter_appendix = st.checkbox("Bỏ qua Phụ Lục", value=False)
    doc_number = st.text_input("Lọc số hiệu văn bản (Tuỳ chọn)", placeholder="VD: 1415/QĐ-UBND")
    limit_docs = st.slider("Số kết quả (Top-K)", min_value=1, max_value=10, value=3)
    
    st.divider()
    st.header("📄 Thêm Văn Bản Mới (Check Xung Đột)")
    new_doc_number = st.text_input("Số hiệu văn bản mới:")
    new_doc_text = st.text_area("Nội dung văn bản:")
    if st.button("Tải lên & Quét Xung Đột"):
        if not new_doc_number or not new_doc_text:
            st.warning("Vui lòng nhập đủ số hiệu và nội dung!")
        else:
            with st.spinner("Đang phân tích và đồng bộ..."):
                res = document_manager.add_document(
                    content=new_doc_text,
                    metadata={"document_number": new_doc_number, "title": f"Văn bản {new_doc_number}", "is_appendix": False}
                )
                if res.get("status") == "success":
                    st.success(f"Đã lưu thành {res['chunks_inserted']} đoạn vectors.")
                    if res["conflicts_found"]:
                        st.error(f"⚠️ Phát hiện có thể xung đột/thay thế các Căn cứ cũ: {', '.join(res['conflicts_found'])}")
                    else:
                        st.info("✅ Văn bản mới không phát hiện xung đột với cơ sở dữ liệu hiện tại.")
                else:
                    st.error(res.get("message", "Lỗi tải lên."))

# -----------------
# 2. MAIN CHAT UI
# -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Render lịch sử tin nhắn
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Nếu message của assistant có references, show nó trong expander
        if msg["role"] == "assistant" and msg.get("references"):
            with st.expander("📚 Căn cứ / Tham chiếu"):
                for idx, ref in enumerate(msg["references"]):
                    tag = "[PHỤ LỤC]" if ref.get("is_appendix") else "[NỘI DUNG]"
                    st.markdown(f"**{idx + 1}. {tag} {ref.get('title', '')}**")
                    st.markdown(f"*{ref.get('article_ref', '')}* (Văn bản số: {ref.get('document_number', '')})")
                    # Nếu có xung đột bị đánh dấu trong payload
                    if ref.get("conflicted_by"):
                        st.warning(f"⚠️ Lưu ý: Văn bản này đã bị báo xung đột/thay thế bởi văn bản: {', '.join(ref['conflicted_by'])}")
                    st.markdown(f"> {ref.get('text', '')[:300]}...")
                    st.divider()

# Xử lý input từ người dùng
if prompt := st.chat_input("Hỏi tôi về các điều khoản, luật pháp..."):
    # Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt, "references": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 1. Truy xuất dữ liệu tuỳ chọn (Nếu là QA/Related/Conflict thì Qdrant logic lấy top)
        with st.spinner("Đang truy xuất CSDL..."):
            retrieved_docs = retriever.search(
                query=prompt,
                limit=limit_docs,
                filter_appendix=filter_appendix,
                doc_number=doc_number if doc_number.strip() else None
            )
            
        # 2. Sinh câu trả lời qua chat_engine
        with st.spinner(f"Chế độ [{chat_mode}] - Đang tổng hợp phân tích..."):
            session_id = "default_user_session"
            
            # Thay vì gọi LLM tay, giờ ta tái dùng hàm chat() và truyền mode vào chat_engine.
            response_dict = rag_engine.chat(
                session_id=session_id,
                query=prompt,
                mode=current_mode,
                provider=llm_provider.lower(),
                model=llm_model
            )
            
            answer = response_dict.get("answer", "Lỗi sinh ngôn ngữ.")
            
        # Hiển thị kết quả text
        message_placeholder.markdown(answer)
        
        # Lấy thêm payload conflicted_by để hiển thị UX
        for doc in retrieved_docs:
            if "conflicted_by" not in doc:
                doc["conflicted_by"] = []
                
        # Hiển thị references trong expander
        if retrieved_docs:
            with st.expander("📚 Căn cứ / Tham chiếu"):
                for idx, ref in enumerate(retrieved_docs):
                    tag = "[PHỤ LỤC]" if ref.get("is_appendix") else "[NỘI DUNG]"
                    st.markdown(f"**{idx + 1}. {tag} {ref.get('title', '')}**")
                    st.markdown(f"*{ref.get('article_ref', '')}* (Văn bản số: {ref.get('document_number', '')})")
                    if ref.get("conflicted_by"):
                        st.warning(f"⚠️ Lưu ý: Văn bản này có rủi ro bị thay thế/xung đột bởi: {', '.join(ref['conflicted_by'])}")
                    st.markdown(f"> {ref.get('text', '')[:300]}...")
                    st.divider()
                    
    # Update Session State
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "references": retrieved_docs
    })
