# 📖 Hướng dẫn Cài đặt & Chạy hệ thống Legal-RAG

> Tài liệu dành cho máy mới muốn clone repo về và chạy hoàn chỉnh hệ thống.

---

## 📋 Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt phần mềm cần thiết](#2-cài-đặt-phần-mềm-cần-thiết)
3. [Clone & Cấu hình dự án](#3-clone--cấu-hình-dự-án)
4. [Chuẩn bị Dữ liệu](#4-chuẩn-bị-dữ-liệu)
5. [Khởi động hệ thống](#5-khởi-động-hệ-thống)
6. [Kiểm tra hoạt động](#6-kiểm-tra-hoạt-động)
7. [Xử lý sự cố thường gặp](#7-xử-lý-sự-cố-thường-gặp)
8. [Kiến trúc tổng quan](#8-kiến-trúc-tổng-quan)

---

## 1. Yêu cầu hệ thống

### ✅ Phần cứng tối thiểu
| Thành phần | Tối thiểu | Khuyến nghị |
|:---|:---|:---|
| **RAM** | 8 GB | 16 GB trở lên |
| **Ổ cứng** | 15 GB trống | 30 GB SSD |
| **CPU** | 4 cores | 8 cores |
| **GPU** | Không bắt buộc | NVIDIA (hỗ trợ CUDA tốt hơn) |

> ⚠️ **Lưu ý RAM**: Embedding model BGE-M3 (PyTorch) chiếm ~5GB RAM. Nếu máy chỉ có 8GB, hãy dùng chế độ ONNX INT8 (hướng dẫn ở mục 4).

### ✅ Hệ điều hành
- Windows 10/11 (đã test)
- Linux / macOS (tương thích, cần chuyển script `.ps1` sang `.sh`)

---

## 2. Cài đặt phần mềm cần thiết

Bạn **BẮT BUỘC** phải cài đặt **tất cả** các phần mềm sau trước khi tiếp tục:

### 2.1. Python 3.11+
- Tải tại: https://www.python.org/downloads/
- Khi cài đặt, **tick ✅ "Add Python to PATH"**.
- Kiểm tra:
  ```powershell
  python --version
  # Kết quả mong muốn: Python 3.11.x hoặc 3.12.x
  ```

### 2.2. Node.js 18+ (cho Frontend)
- Tải tại: https://nodejs.org/ (chọn bản LTS)
- Kiểm tra:
  ```powershell
  node --version   # v18.x trở lên
  npm --version    # 9.x trở lên
  ```

### 2.3. Docker Desktop (cho Qdrant & Redis)
- Tải tại: https://www.docker.com/products/docker-desktop/
- Sau khi cài, **mở Docker Desktop** và đảm bảo nó đang chạy (icon cá voi xanh ở taskbar).
- Kiểm tra:
  ```powershell
  docker --version          # Docker version 24.x+
  docker-compose --version  # Docker Compose version v2.x+
  ```

### 2.4. Git
- Tải tại: https://git-scm.com/downloads
- Kiểm tra:
  ```powershell
  git --version
  ```

### 2.5. API Key cho LLM (chọn 1 trong 3)
Hệ thống cần **ít nhất 1 API Key** của nhà cung cấp LLM:

| Provider | Đăng ký | Ghi chú |
|:---|:---|:---|
| **Groq** (mặc định, miễn phí) | https://console.groq.com/ | Tốc độ nhanh nhất, Free tier hào phóng |
| **Google Gemini** | https://aistudio.google.com/apikey | Chất lượng tốt, Free tier |
| **Ollama** (chạy local) | https://ollama.com/ | Không cần API key, nhưng cần GPU mạnh |

---

## 3. Clone & Cấu hình dự án

### 3.1. Clone repository
```powershell
git clone <repository-url>
cd Legal-RAG
```

### 3.2. Tạo file cấu hình `.env`
```powershell
Copy-Item .env.example -Destination .env
```

Mở file `.env` bằng trình soạn thảo và điền thông tin:

```dotenv
# ─── LLM (Chọn 1 provider) ───
# Nếu dùng Groq (khuyến nghị cho tốc độ):
LLM_PROVIDER=groq
LLM_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_CHAT_MODEL=llama-3.1-8b-instant

# Nếu dùng Gemini thay thế:
# LLM_PROVIDER=gemini
# LLM_API_KEY=AIzaxxxxxxxxxxxxxxxxxxxxx
# LLM_CHAT_MODEL=gemini-2.0-flash

# ─── VECTOR DB (Qdrant Local Docker) ───
QDRANT_URL=http://localhost:6335
QDRANT_COLLECTION=legal_rag_docs_5000
QDRANT_READ_ONLY=false

# ─── CACHE ───
HF_HOME=D:/huggingface_cache
SENTENCE_TRANSFORMERS_HOME=D:/huggingface_cache/sentence_transformers

# ─── REDIS ───
REDIS_URL=redis://localhost:6379/0
```

> ⚠️ **Quan trọng**: Đường dẫn `HF_HOME` nên trỏ tới **ổ đĩa có dung lượng trống >10GB**. Embedding model BGE-M3 sẽ được tải về đây (~2.4GB).

### 3.3. Cài đặt Frontend dependencies
```powershell
cd frontend
npm install
cd ..
```

---

## 4. Chuẩn bị Dữ liệu

### 4.1. Dữ liệu Vector Database (Qdrant)

Đây là phần **quan trọng nhất**. Qdrant cần có dữ liệu đã được nạp (ingest) để hệ thống hoạt động.

**Cách 1: Khôi phục từ Snapshot (Nhanh nhất) ✅**

Nếu bạn được cung cấp file snapshot (`.snapshot`):
```powershell
# 1. Khởi động Qdrant trước
docker-compose up -d qdrant

# 2. Chờ Qdrant sẵn sàng (~30 giây)
Start-Sleep -Seconds 30

# 3. Khôi phục snapshot qua API
Invoke-WebRequest -Uri "http://localhost:6335/collections/legal_rag_docs_5000/snapshots/upload" `
  -Method Post `
  -InFile "path/to/your/snapshot.snapshot" `
  -ContentType "multipart/form-data"
```

**Cách 2: Nạp dữ liệu từ file PDF/DOCX gốc (Tạo mới)**

Nếu không có snapshot:
```powershell
# 1. Đặt các file PDF/DOCX vào thư mục legal_docs/
mkdir legal_docs   # Nếu chưa có

# 2. Chạy script nạp dữ liệu
python -m scripts.ingest_local
```

> ⚠️ Quá trình nạp dữ liệu có thể mất **vài giờ** tùy số lượng văn bản.

### 4.2. Embedding Model (BGE-M3) — Tự động tải

Lần đầu chạy, hệ thống sẽ **tự động tải** model BGE-M3 (~2.4GB) từ HuggingFace về thư mục `HF_HOME`. Bạn không cần làm gì thêm.

**[Tùy chọn] Dùng ONNX INT8 để tiết kiệm RAM:**

Nếu máy ít RAM (≤8GB), hãy tạo bản ONNX:
```powershell
python -m scripts.convert_bge_onnx
```
Model ONNX sẽ được lưu vào `HF_HOME/bge-m3-onnx/` và hệ thống sẽ tự động phát hiện và sử dụng.

---

## 5. Khởi động hệ thống

### Cách 1: Dùng script tự động (Khuyến nghị) ✅

```powershell
.\quick_start.ps1
```

Script sẽ tự động thực hiện:
1. Khởi động Docker containers (Redis + Qdrant)
2. Tạo Python virtual environment
3. Cài đặt thư viện Python
4. Kiểm tra file `.env`
5. Dọn dẹp cổng cũ
6. Khởi động các dịch vụ song song (mỗi dịch vụ 1 cửa sổ terminal)

### Cách 2: Khởi động thủ công từng bước

```powershell
# Bước 1: Khởi động Docker (Redis + Qdrant)
docker-compose up -d

# Bước 2: Tạo & kích hoạt môi trường Python
python -m venv venv
.\venv\Scripts\Activate.ps1

# Bước 3: Cài đặt thư viện
pip install -r requirements.txt

# Bước 4: Khởi động Embedding Server (Cửa sổ terminal 1)
python -m backend.retrieval.server
# → Chờ cho tới khi thấy: "✅ Đã tải xong Embedding Model" (1-3 phút)

# Bước 5: Khởi động API Backend (Cửa sổ terminal 2)
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Bước 6: Khởi động Frontend (Cửa sổ terminal 3)
cd frontend
npm run dev -- --port 3000
```

---

## 6. Kiểm tra hoạt động

Sau khi khởi động xong (chờ ~2-3 phút), truy cập các địa chỉ:

| Dịch vụ | URL | Mô tả |
|:---|:---|:---|
| **Frontend** | http://localhost:3000 | Giao diện chatbot |
| **API Backend** | http://localhost:8000/docs | Swagger API docs |
| **Embedding Server** | http://localhost:8001/health | Health check |
| **Qdrant Dashboard** | http://localhost:6335/dashboard | Quản lý vector DB |
| **Redis** | `localhost:6379` | Chat memory cache |

### Kiểm tra nhanh:
```powershell
# Test API Backend
Invoke-WebRequest -Uri "http://localhost:8000/health"

# Test Embedding Server
Invoke-WebRequest -Uri "http://localhost:8001/health"

# Test Qdrant
Invoke-WebRequest -Uri "http://localhost:6335/collections"
```

---

## 7. Xử lý sự cố thường gặp

### ❌ Lỗi "Docker not found"
→ Cài Docker Desktop và **khởi động lại máy**.

### ❌ Embedding Server bị "bad allocation" / Out of Memory
→ Máy không đủ RAM cho PyTorch BGE-M3. Giải pháp:
```powershell
# Tạo bản ONNX INT8 nhẹ hơn ~4x
python -m scripts.convert_bge_onnx
```

### ❌ Qdrant trả về "Collection not found"
→ Chưa nạp dữ liệu. Xem lại **mục 4.1** để khôi phục snapshot hoặc ingest dữ liệu thủ công.

### ❌ Frontend lỗi "Failed to fetch" / CORS error
→ Kiểm tra:
1. API Backend có đang chạy ở port 8000 không?
2. File `frontend/.env` hoặc biến `NEXT_PUBLIC_API_URL` có đúng `http://localhost:8000/api` không?

### ❌ LLM trả về lỗi "Invalid API Key"
→ Kiểm tra lại `LLM_API_KEY` trong file `.env`. Đảm bảo key hợp lệ và provider đúng.

### ❌ Redis không kết nối được
→ Kiểm tra Docker container Redis:
```powershell
docker ps   # Xem container legal-rag-redis có đang chạy không
docker logs legal-rag-redis   # Xem logs lỗi
```
> Lưu ý: Nếu Redis không chạy, hệ thống vẫn hoạt động nhưng chat memory sẽ dùng RAM (mất khi khởi động lại).

---

## 8. Kiến trúc tổng quan

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────┐
│   Frontend      │────▶│   FastAPI Backend     │────▶│  LLM API     │
│   (Next.js)     │     │   (Port 8000)         │     │  (Groq/      │
│   Port 3000     │◀────│                       │     │   Gemini/    │
└─────────────────┘     │   ┌───────────────┐   │     │   Ollama)    │
                        │   │  LangGraph    │   │     └──────────────┘
                        │   │  Orchestrator │   │
                        │   └───────┬───────┘   │
                        │           │           │
                        │   ┌───────▼───────┐   │     ┌──────────────┐
                        │   │  Hybrid       │───┼────▶│  Qdrant      │
                        │   │  Retriever    │   │     │  (Port 6335) │
                        │   └───────┬───────┘   │     └──────────────┘
                        │           │           │
                        │   ┌───────▼───────┐   │     ┌──────────────┐
                        │   │  Embedding    │───┼────▶│  Redis       │
                        │   │  Server 8001  │   │     │  (Port 6379) │
                        │   └───────────────┘   │     └──────────────┘
                        └──────────────────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │  SQLite DB   │
                            │  (Chat       │
                            │   History)   │
                            └──────────────┘
```

### Tóm tắt Các Dịch vụ cần chạy (4 dịch vụ):

| # | Dịch vụ | Port | Vai trò |
|:--|:--------|:-----|:--------|
| 1 | Docker: Redis | 6379 | Short-term chat memory |
| 2 | Docker: Qdrant | 6335 | Vector DB lưu trữ VBPL |
| 3 | Embedding Server | 8001 | Chuyển text → vector (BGE-M3) |
| 4 | FastAPI Backend | 8000 | API chính + LangGraph Agent |
| 5 | Next.js Frontend | 3000 | Giao diện chatbot |

---

*Tài liệu tạo ngày: 03/04/2026. Phiên bản: 3.0 (LangGraph Edition)*
