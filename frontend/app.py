import os
import requests
import streamlit as st

# Resolve API URL: Docker sets API_URL env var, local dev falls back to localhost
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Vibecoding Chatbot Pháp Luật VN", page_icon="⚖️", layout="wide")

st.title("⚖️ Vibecoding Chatbot Trợ Lý Pháp Luật Việt Nam")
st.markdown("Hệ thống tra cứu, phân tích tương quan và phát hiện xung đột văn bản pháp luật sử dụng **Qdrant**, **BGE-M3 (Hybrid)** và **FastAPI**.")

# -----------------
# 1. SIDEBAR: Tùy chỉnh & Upload
# -----------------
with st.sidebar:
    st.header("⚙️ Cài đặt Mô Hình (LLM)")
    providers = ["Groq", "Gemini", "Ollama"]
    try:
        from backend.config import settings
        default_index = providers.index(settings.LLM_PROVIDER.capitalize())
    except:
        default_index = 0
        
    llm_provider = st.selectbox("Chọn Provider:", options=providers, index=default_index)
    
    st.divider()
    st.header("⚙️ Cài đặt Tìm Kiếm")
    
    use_reflection = st.checkbox("Có tự kiểm tra (Reflection)?", value=True)
    limit_docs = st.slider("Số kết quả (Top-K Context)", min_value=1, max_value=10, value=3)
    
    st.divider()
    st.header("📄 Thêm Văn Bản Mới")
    uploaded_file = st.file_uploader("Tải lên PDF/Docx để add vào CSDL", type=["pdf", "docx", "doc"])
    
    if st.button("Tải lên & Xử lý"):
        if uploaded_file is not None:
            with st.spinner("Đang tải lên..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success(res.json().get("message", "Thành công"))
                    else:
                        st.error(f"Lỗi {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Không thể kết nối đến backend API: {e}")
        else:
            st.warning("Vui lòng chọn file!")

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
            with st.expander("📚 Căn cứ / Tham chiếu Tốt Nhất (Top K)"):
                for idx, ref in enumerate(msg["references"]):
                    st.markdown(f"**{idx + 1}. {ref.get('title', '')}**")
                    st.markdown(f"*{ref.get('article', '')}* (Score: {ref.get('score', 0):.4f})")
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
        
        with st.spinner("Đang suy nghĩ (ReAct + Retrieval + Reflection)..."):
            session_id = "default_user_session"
            
            payload = {
                "session_id": session_id,
                "query": prompt,
                "provider": llm_provider.lower(),
                "top_k": limit_docs,
                "use_reflection": use_reflection
            }
            
            answer = "Lỗi kết nối Server"
            retrieved_docs = []
            
            try:
                response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
                    
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Lỗi dữ liệu trả về.")
                    retrieved_docs = data.get("references", [])
                else:
                    answer = f"Lỗi API: {response.text}"
            except Exception as e:
                answer = f"Exception: {e}"
            
        # Hiển thị kết quả text
        message_placeholder.markdown(answer)
        
        # Hiển thị references trong expander
        if retrieved_docs:
            with st.expander("📚 Căn cứ / Tham chiếu"):
                for idx, ref in enumerate(retrieved_docs):
                    st.markdown(f"**{idx + 1}. {ref.get('title', '')}**")
                    st.markdown(f"*{ref.get('article', '')}* (Score: {ref.get('score', 0):.4f})")
                    st.divider()
                    
    # Update Session State
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "references": retrieved_docs
    })
