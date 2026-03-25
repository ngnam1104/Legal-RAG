import os
import sys
import time
import urllib.request

# Thêm đường dẫn root để load modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdrant_client import QdrantClient
from core.nlp import get_embedder
from core.llm import chat_completion
from core.config import settings

# --- CẤU HÌNH ---
# Cho phép tự động phát hiện localhost vs qdrant tuỳ vào môi trường (Docker hay Local)
QDRANT_URL = "http://localhost:6333"
try:
    if urllib.request.urlopen("http://qdrant:6333", timeout=1).getcode() == 200:
        QDRANT_URL = "http://qdrant:6333"
except Exception:
    pass

# Ghi đè URL nếu người dùng đã set biến môi trường, hoặc force thành http://qdrant:6333 nếu user muốn test hardcode
QDRANT_URL = os.environ.get("QDRANT_URL", QDRANT_URL)

COLLECTION_NAME = settings.QDRANT_COLLECTION # Mặc định: legal_vn_200_docs
TEST_QUERY = "Quy trình cấp phép khai thác khoáng sản"


def test_1_check_qdrant():
    print("\n" + "=" * 60)
    print(f"🚀 TEST 1: KIỂM TRA QDRANT DB ({QDRANT_URL})")
    print("=" * 60)
    try:
        qclient = QdrantClient(url=QDRANT_URL)
        collections = qclient.get_collections()
        print("✅ Kết nối đến Qdrant thành công!")
        print("Danh sách các collections hiện có tại DB:")
        for col in collections.collections:
            count = qclient.count(col.name).count
            print(f"  📦 {col.name}: {count} vectors")

        if not any(c.name == COLLECTION_NAME for c in collections.collections):
            print(f"⚠️ Cảnh báo: DB không có collection '{COLLECTION_NAME}'. Tự động tạo nó mới có thể test tiếp (Sẽ bị rỗng).")
        return qclient
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")
        print("Vui lòng đảm bảo Docker Compose đang chạy Qdrant ở cổng 6333.")
        sys.exit(1)


def test_2_query_vectors(qclient):
    print("\n" + "=" * 60)
    print(f"🚀 TEST 2: TRUY VẤN VECTOR TRONG DB '{COLLECTION_NAME}'")
    print("=" * 60)
    print(f"🔍 Truy vấn: '{TEST_QUERY}'")

    print("⏳ Đang tải model Embedding (BGE-M3)...")
    try:
        embedder = get_embedder()
        query_vec = embedder.encode(TEST_QUERY, show_progress_bar=False)[0]
    except Exception as e:
        print(f"❌ Lỗi tải model embedding (Thử kiểm tra kết nối mạng/HuggingFace): {e}")
        sys.exit(1)

    try:
        response = qclient.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=3,
        )
        print("\n📄 Kết quả truy xuất từ Qdrant:")
        print("-" * 50)
        context_parts = []
        for hit in response.points:
            p = hit.payload
            title = p.get("title", "Không rõ title")
            article = p.get("article_ref", "")
            text = p.get("chunk_text", "")
            print(f"  [{hit.score:.4f}] {title} — {article}")
            context_parts.append(f"[{title} - {article}]: {text}")

        if not context_parts:
            print(f"⚠️ Không tìm thấy kết quả nào trong DB '{COLLECTION_NAME}'. DB có thể đang rỗng.")
            # Vẫn tạo mock context để không crash pipeline
            context_parts.append("[Mock Data]: Thử nghiệm hệ thống vì DB rỗng dữ liệu.")

        print("-" * 50)
        context_text = "\n\n---\n\n".join(context_parts)
        return context_text
    except Exception as e:
        print(f"❌ Lỗi truy vấn Qdrant: {e}")
        sys.exit(1)


def test_3_full_rag_pipeline(context_text):
    print("\n" + "=" * 60)
    print("🚀 TEST 3: FULL RAG PIPELINE (GỬI CONTEXT CHO GROQ & GEMINI)")
    print("=" * 60)

    from rag.chat_engine import RAGEngine

    # Tạo object RAGEngine mà không cần khởi tạo lại kết nối Qdrant trong __init__
    engine = RAGEngine.__new__(RAGEngine)
    engine.__init__.__func__(RAGEngine)  

    user_prompt = f"""--- ĐẦU VÀO TỪ HỆ THỐNG ---
[MODE]: Q_AND_A
[CONTEXT]:
{context_text}

[USER_QUERY]:
{TEST_QUERY}

[CÂU TRẢ LỜI CỦA BẠN]:"""

    messages = [
        {"role": "system", "content": engine.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # --- Gọi GROQ ---
    print("\n🟢 Đang gửi ngữ cảnh tới GROQ (llama-3.1-8b-instant)...")
    try:
        t0 = time.perf_counter()
        ans_groq = chat_completion(messages, temperature=0.3, provider="groq")
        t1 = time.perf_counter()
        print(ans_groq)
        print(f"\n⏱ {t1 - t0:.2f}s  ✅ Groq xử lý thành công")
    except Exception as e:
        print(f"❌ Groq FAILED: {e}")

    # --- Gọi GEMINI ---
    print("\n🔵 Đang gửi ngữ cảnh tới GEMINI (gemini-3-flash-preview)...")
    try:
        t0 = time.perf_counter()
        ans_gemini = chat_completion(messages, temperature=0.3, provider="gemini")
        t1 = time.perf_counter()
        print(ans_gemini)
        print(f"\n⏱ {t1 - t0:.2f}s  ✅ Gemini xử lý thành công")
    except Exception as e:
        print(f"❌ Gemini FAILED: {e}")


if __name__ == "__main__":
    os.environ["QDRANT_URL"] = QDRANT_URL

    # Chạy lần lượt các bước
    qclient = test_1_check_qdrant()
    
    context_text = test_2_query_vectors(qclient)
    
    test_3_full_rag_pipeline(context_text)

    print("\n🎉 HOÀN TẤT ALL TESTS TRONG PIPELINE!")
