"""
Phân tích chi tiết khả năng kết nối tới Ollama API.
Chạy: python scripts/tests/diagnostic_ollama.py
"""
import os
import requests
import json
import sys
from dotenv import load_dotenv

# Allow importing from root directory (if needed in future)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

def diagnostic_ollama():
    print("=" * 50)
    print("🔬 OLLAMA API DIAGNOSTIC (Detailed)")
    print("=" * 50)
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
    model_name = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:cloud")
    api_key = os.getenv("OLLAMA_API_KEY")

    print(f"📡 Target: {base_url}")
    print(f"🤖 Model: {model_name}")
    print(f"🔑 API Key: {'Present' if api_key else 'Missing'}")
    print("-" * 50)

    # 1. Connectivity and Server Type
    print("🔍 Step 1: Checking server identity...")
    try:
        # Try /api/tags to list models
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        print(f"   HTTP Status: {resp.status_code}")
        print(f"   Server Header: {resp.headers.get('Server', 'Unknown')}")
        
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"   Available models on this endpoint: {[m['name'] for m in models]}")
    except Exception as e:
        print(f"   ❌ Connectivity error: {e}")

    # 2. Testing /api/chat (Standard Ollama API)
    print("\n🔍 Step 2: Testing /api/chat (Standard Ollama API)...")
    chat_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    try:
        resp = requests.post(f"{base_url}/api/chat", json=chat_payload, headers=headers, timeout=15)
        print(f"   HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            content = resp.json().get("message", {}).get("content", "")
            print(f"   ✅ Chat Success! Response snippet: {content[:100]}...")
        else:
            print(f"   ❌ Chat Failed. Error: {resp.text}")
    except Exception as e:
        print(f"   ❌ Chat Error: {e}")

    # 3. Testing /v1/chat/completions (OpenAI Compatible API)
    print("\n🔍 Step 3: Testing /v1/chat/completions (OpenAI Compatible)...")
    v1_payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    }
    try:
        resp = requests.post(f"{base_url}/v1/chat/completions", json=v1_payload, headers=headers, timeout=15)
        print(f"   HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   ✅ V1 API Success! Response snippet: {content[:100]}...")
        else:
            print(f"   ❌ V1 API Failed. Error: {resp.text}")
    except Exception as e:
        print(f"   ❌ V1 API Error: {e}")

    # 4. Checking if we need OLLAMA-API-KEY header
    print("\n🔍 Step 4: Testing with OLLAMA-API-KEY header...")
    custom_headers = {"OLLAMA-API-KEY": api_key} if api_key else {}
    try:
        resp = requests.post(f"{base_url}/api/chat", json=chat_payload, headers=custom_headers, timeout=15)
        print(f"   HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   ✅ Success with OLLAMA-API-KEY header!")
    except Exception:
        pass

    print("-" * 50)
    print("💡 Analysis:")
    print("Lý do 'ollama run' được nhưng API không được có thể là:")
    print("1. 'ollama run' khi thấy :cloud sẽ kết nối trực tiếp tới cloud (ollama.com) nếu app hỗ trợ.")
    print("2. API yêu cầu Header hoặc Endpoint khác (Ollama standard vs OpenAI compatible).")
    print("3. Model name trong API cần prefix (ví dụ 'ollama/qwen3.5:cloud' hoặc bỏ ':cloud').")
    print("=" * 50)

if __name__ == "__main__":
    diagnostic_ollama()
