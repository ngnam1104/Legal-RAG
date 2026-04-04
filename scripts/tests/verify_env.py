import os
import requests
import redis
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Cấu hình đường dẫn: 3 cấp lên tới Root (scripts/tests/file.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

# Load env từ thư mục gốc
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

def print_result(name, success, message=""):
    symbol = "✅" if success else "❌"
    print(f"{symbol} {name}: {message}")

def verify():
    print("="*50)
    print("🔍 HỆ THỐNG KIỂM TRA MÔI TRƯỜNG (LIGHTWEIGHT)")
    print("="*50)

    # 1. Kiểm tra Redis (Cần cho Celery)
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(redis_url)
        r.ping()
        print_result("Redis", True, f"Kết nối ổn định tại {redis_url}")
    except Exception as e:
        print_result("Redis", False, f"Không thể kết nối. Lỗi: {e}")
        print("   👉 Mẹo: Thử chạy 'docker compose up -d redis' nếu bạn chưa có Redis local.")

    # 2. Kiểm tra Ollama Cloud/Reasoning
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
        model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:cloud")
        api_key = os.getenv("OLLAMA_API_KEY")
        
        headers = {"OLLAMA-API-KEY": api_key, "Authorization": f"Bearer {api_key}"} if api_key else {}
        
        resp = requests.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False},
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            print_result("Ollama", True, f"Model '{model}' phản hồi tốt.")
        else:
            print_result("Ollama", False, f"Lỗi HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print_result("Ollama", False, f"Lỗi kết nối: {e}")

    # 3. Kiểm tra Qdrant Cloud
    try:
        q_url = os.getenv("QDRANT_URL")
        q_key = os.getenv("QDRANT_API_KEY")
        client = QdrantClient(url=q_url, api_key=q_key)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print_result("Qdrant", True, f"Kết nối thành công. Tìm thấy {len(col_names)} collections.")
    except Exception as e:
        print_result("Qdrant", False, f"Lỗi kết nối: {e}")

    # 4. Kiểm tra Cache Path
    hf_home = os.getenv("HF_HOME", "D:/huggingface_cache")
    if os.path.exists(hf_home):
        print_result("Cache Folder", True, f"Thư mục '{hf_home}' tồn tại.")
    else:
        print_result("Cache Folder", False, f"Thư mục '{hf_home}' KHÔNG tồn tại. Kiểm tra ổ D: của bạn.")

    print("="*50)

if __name__ == "__main__":
    verify()
