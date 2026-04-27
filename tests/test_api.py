import time 
import os
import sys
import requests
import json

# Fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Đảm bảo encoding chuẩn cho terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# ==========================================
# THÔNG TIN CÁC ENDPOINT API NỘI BỘ
# ==========================================
from backend.llm.factory import get_client
from backend.retrieval.reranker import reranker as internal_reranker
from backend.retrieval.embedder import embedder as internal_embedder
from backend.config import settings

# ICLLM Client
llm_client = get_client()

# Raw Endpoints for legacy tests
LLM_URL = "http://10.9.3.75:30031/api/llama3/8b"
RERANK_URL = "http://10.9.3.75:30546/api/v1/reranking"
EMBEDDING_URL = "http://10.9.3.75:30010/api/v1/embedding"

def summarize_large_json(obj):
    """
    Hàm đệ quy để thu gọn các cục data quá lớn (như Vector 1024 chiều, chuỗi quá dài)
    giúp terminal in ra dễ nhìn hơn.
    """
    if isinstance(obj, dict):
        return {k: summarize_large_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        if len(obj) == 0:
            return obj
        
        # Nhận diện mảng số (Vector Embedding)
        if all(isinstance(x, (int, float)) for x in obj):
            if len(obj) > 10:
                return f"<Vector/List gồm {len(obj)} phần tử số>"
            return obj
        
        # Với các mảng object khác, nếu quá dài thì chỉ in 3 phần tử đầu
        if len(obj) > 3:
            summarized = [summarize_large_json(x) for x in obj[:3]]
            summarized.append(f"<... và {len(obj) - 3} phần tử khác đã bị ẩn ...>")
            return summarized
        
        return [summarize_large_json(x) for x in obj]
    elif isinstance(obj, str) and len(obj) > 500:
        # Cắt ngắn text quá dài
        return obj[:500] + "... <TRUNCATED_TEXT>"
    else:
        return obj

def print_result(api_name, response):
    """Hàm hỗ trợ in kết quả đẹp mắt để dễ phân tích"""
    print(f"\n{'='*60}")
    print(f"🚀 KẾT QUẢ TEST: {api_name}")
    print(f"{'='*60}")
    print(f"HTTP Status Code: {response.status_code}")
    
    if response.status_code == 200:
        try:
            # Lấy JSON và đưa qua hàm thu gọn trước khi in
            raw_json = response.json()
            summarized_json = summarize_large_json(raw_json)
            print(json.dumps(summarized_json, ensure_ascii=False, indent=2))
        except Exception:
            print("Response không phải định dạng JSON:")
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
    else:
        print(f"❌ Lỗi gọi API. Chi tiết:\n{response.text}")

# ------------------------------------------
# 1. TEST API EMBEDDING
# ------------------------------------------
def test_embedding():
    print("\n⏳ Đang gọi API Embedding qua InternalAPIEmbedder (backend)...")
    texts = [
        "Tài liệu 1: Hướng dẫn an toàn lao động.",
        "Tài liệu 2: Quy định về bảo mật thông tin."
    ]
    
    try:
        start_t = time.time()
        vectors = internal_embedder.encode_dense(texts)
        end_t = time.time()
        
        print(f"\n{'='*60}")
        print(f"🚀 KẾT QUẢ TEST: API EMBEDDING")
        print(f"{'='*60}")
        print(f"Thời gian: {end_t - start_t:.2f}s")
        print(f"✅ Đã nhận được {len(vectors)} vectors.")
        if vectors:
            print(f"Kích thước vector đầu tiên: {len(vectors[0])} chiều.")
            print(f"Giá trị mẫu (5 phần tử đầu): {vectors[0][:5]}")
            
    except Exception as e:
        print(f"❌ Lỗi kết nối Embedding API: {e}")

# ------------------------------------------
# 2. TEST API RERANKING
# ------------------------------------------
def test_reranking():
    print("\n⏳ Đang gọi API Reranking qua InternalAPIReranker (backend)...")
    query = "An toàn lao động là gì?"
    docs = [
        "Công ty yêu cầu mọi người phải đội mũ bảo hiểm khi vào công trường.",
        "Hướng dẫn nướng bánh mì bằng nồi chiên không dầu.",
        "Quy định số 10 về các biện pháp đảm bảo an toàn, vệ sinh lao động năm 2024."
    ]
    
    try:
        start_t = time.time()
        results = internal_reranker.rerank(query, docs, top_k=3)
        end_t = time.time()
        
        print(f"\n{'='*60}")
        print(f"🚀 KẾT QUẢ TEST: API RERANKING (ICLLM Wrapper)")
        print(f"{'='*60}")
        print(f"Thời gian: {end_t - start_t:.2f}s")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"❌ Lỗi kết nối Reranking API: {e}")

def test_llm_stress_test():
    print("⏳ Đang chuẩn bị dữ liệu Stress Test qua ICLLMClient...")
    
    # 1. Tạo một đoạn text gốc
    base_text = "Theo Điều 15 Nghị định 100/2019/NĐ-CP, người điều khiển phương tiện phải tuân thủ tốc độ. "
    
    # 2. Nhân bản đoạn text này lên 500 lần để tạo ra một Context siêu dài (khoảng 6000-8000 tokens)
    massive_context = base_text * 500 
    
    print(f"📏 Độ dài chuỗi Context giả lập: {len(massive_context):,} ký tự.")
    print("⏳ Đang gửi request qua ICLLMClient với max_input_length=8000...")
    
    messages = [
        {"role": "user", "content": f"Hãy tóm tắt ngắn gọn quy định trong đoạn văn bản sau đây:\n\n{massive_context}"}
    ]
    
    start_time = time.time()
    try:
        # ICLLMClient.chat_completion
        response_text = llm_client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            max_input_length=8000
        )
        end_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Thời gian phản hồi: {end_time - start_time:.2f} giây")
        
        if response_text:
            print(f"✅ THÀNH CÔNG! ICLLM đã trả về kết quả.")
            print(f"🤖 Output: {response_text[:300]}... <TRUNCATED>")
            print(f"📝 Xem chi tiết log tại: logs/llm_logs/")
        else:
            print(f"❌ THẤT BẠI. ICLLM trả về chuỗi rỗng. Hãy kiểm tra console/log.")
            
    except Exception as e:
        print(f"❌ Lỗi khi gọi ICLLM: {e}")

# ==========================================
# CHẠY TEST
# ==========================================
if __name__ == "__main__":
    print("BẮT ĐẦU KIỂM TRA KẾT NỐI API NỘI BỘ TỪ MÁY LOCAL...")

    test_embedding()
    test_reranking()
    test_llm_stress_test()
    print("\n✅ HOÀN TẤT TEST. BẠN HÃY COPY TOÀN BỘ KẾT QUẢ TRÊN TERMINAL ĐỂ TIẾN HÀNH PHÂN TÍCH.")