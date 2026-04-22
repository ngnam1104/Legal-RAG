import time 
import os
import sys
import requests
import json

# Đảm bảo encoding chuẩn cho terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
# ==========================================
# THÔNG TIN CÁC ENDPOINT API NỘI BỘ
# ==========================================
LLM_URL = "http://10.9.3.75:30028/api/llama3/8b"
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
    print("\n⏳ Đang gọi API Embedding...")
    payload = {
        "texts": [
            "Tài liệu 1: Hướng dẫn an toàn lao động.",
            "Tài liệu 2: Quy định về bảo mật thông tin."
        ],
        "normalize": False
    }
    
    try:
        response = requests.post(EMBEDDING_URL, json=payload, timeout=30)
        print_result("API EMBEDDING", response)
    except Exception as e:
        print(f"❌ Lỗi kết nối Embedding API: {e}")

# ------------------------------------------
# 2. TEST API RERANKING
# ------------------------------------------
def test_reranking():
    print("\n⏳ Đang gọi API Reranking...")
    payload = {
        "query": "An toàn lao động là gì?",
        "docs": [
            "Công ty yêu cầu mọi người phải đội mũ bảo hiểm khi vào công trường.",
            "Hướng dẫn nướng bánh mì bằng nồi chiên không dầu.",
            "Quy định số 10 về các biện pháp đảm bảo an toàn, vệ sinh lao động năm 2024."
        ]
    }
    
    try:
        response = requests.post(RERANK_URL, json=payload, timeout=30)
        print_result("API RERANKING", response)
    except Exception as e:
        print(f"❌ Lỗi kết nối Reranking API: {e}")

def test_llm_stress_test():
    print("⏳ Đang chuẩn bị dữ liệu Stress Test...")
    
    # 1. Tạo một đoạn text gốc
    base_text = "Theo Điều 15 Nghị định 100/2019/NĐ-CP, người điều khiển phương tiện phải tuân thủ tốc độ. "
    
    # 2. Nhân bản đoạn text này lên 500 lần để tạo ra một Context siêu dài (khoảng 6000-8000 tokens)
    massive_context = base_text * 500 
    
    print(f"📏 Độ dài chuỗi Context giả lập: {len(massive_context):,} ký tự.")
    print("⏳ Đang gửi request ép giới hạn max_input_length lên 8000...")
    
    payload = {
        "questions": ["Hãy tóm tắt ngắn gọn quy định trong đoạn văn bản trên."],
        "contexts": [massive_context],
        "lang": "vi",
        "use_en_model": False,
        "batch_size": 1,
        "max_decoding_length": 1024,
        "max_input_length": 8000,   # Cố tình set vượt mức 4000 mặc định
        "repetition_penalty": 0,
        "temperature": 0.1,
        "do_sample": True,
        "no_repeat_ngram_size": 0,
        "add_generation_prompt": True,
        "tokenize": False,
        "histories": []
    }
    
    start_time = time.time()
    try:
        response = requests.post(LLM_URL, json=payload, timeout=120)
        end_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"HTTP Status: {response.status_code}")
        print(f"Thời gian phản hồi: {end_time - start_time:.2f} giây")
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})
            print(f"✅ THÀNH CÔNG! API đã chấp nhận Context dài.")
            print(f"📊 Token Usage: {usage}")
            print(f"🤖 Output: {data.get('result', [''])[0][:200]}...") # In một đoạn nhỏ của kết quả
        elif response.status_code == 422:
            print("❌ THẤT BẠI (HTTP 422 Validation Error).")
            print("Phân tích: FastAPI Gateway đã chặn request vì Pydantic schema bắt buộc max_input_length <= 4000.")
            print("Chi tiết lỗi:", json.dumps(response.json(), ensure_ascii=False, indent=2))
        else:
            print(f"❌ THẤT BẠI (HTTP {response.status_code}).")
            print("Phân tích: Có thể Model bị tràn VRAM (OOM) hoặc Inference Engine từ chối độ dài này.")
            print("Chi tiết:", response.text[:500])
            
    except Exception as e:
        print(f"❌ Lỗi kết nối hoặc Timeout: {e}")

# ==========================================
# CHẠY TEST
# ==========================================
if __name__ == "__main__":
    print("BẮT ĐẦU KIỂM TRA KẾT NỐI API NỘI BỘ TỪ MÁY LOCAL...")

    test_embedding()
    test_reranking()
    test_llm_stress_test()
    print("\n✅ HOÀN TẤT TEST. BẠN HÃY COPY TOÀN BỘ KẾT QUẢ TRÊN TERMINAL ĐỂ TIẾN HÀNH PHÂN TÍCH.")