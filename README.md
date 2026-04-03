# ⚖️ Legal-RAG: Trợ lý Pháp luật Việt Nam Thông minh

Hệ thống **Advanced Agentic RAG** mã nguồn mở chuyên biệt cho văn bản pháp luật Việt Nam. Ứng dụng công nghệ Hybrid Search (Dense + Sparse), Kiến trúc Đa tác vụ (Multi-agent) và cơ chế Tự phản hồi (Reflection) để đảm bảo độ chính xác pháp lý tối đa.

---

## 🌟 Tính năng Nổi bật

- **🔍 Tra cứu Hybrid (Dense + Sparse)**: Kết hợp sức mạnh của vector embedding (`BGE-M3`) và tìm kiếm từ khóa truyền thống để xử lý các thuật ngữ chuyên môn phức tạp.
- **🤖 Query Router & Multi-turn**: Tự động phân loại ý định người dùng và duy trì ngữ cảnh hội thoại (Query Condensation) để xử lý câu hỏi tiếp nối.
- **🛡️ Reflection Agent (Tự kiểm duyệt)**: Cơ chế AI tự kiểm tra câu trả lời của chính mình trước khi phản hồi người dùng để loại bỏ hoàn toàn hiện tượng "ảo giác" (hallucination).
- **📋 4 Chế độ Hoạt động Chuyên biệt**:
    1. **Legal QA**: Giải đáp tình huống dựa trên căn cứ pháp luật cụ thể.
    2. **Sector Search**: Tổng hợp, liệt kê văn bản theo lĩnh vực/ngành.
    3. **Conflict Analyzer**: Tải lên file nội quy/hợp đồng để đối soát xem có nội dung nào trái luật không.
    4. **General Chat**: Trò chuyện tự do về mọi chủ đề không liên quan đến pháp luật.
- **💾 2-Layer Memory System**: Quản lý hội thoại thông minh với Redis (truy xuất nhanh) và SQLite (lưu trữ vĩnh viễn).

---

## 🏗️ Kiến trúc Hệ thống & Luồng Dữ liệu

Dưới đây là sơ đồ luồng dữ liệu tổng thể từ lúc người dùng đặt câu hỏi đến khi nhận được phản hồi đã qua kiểm duyệt:

```mermaid
graph TD
    User([Người dùng]) -->|Hỏi| UI[Next.js Frontend]
    UI -->|REST API| API[FastAPI Backend]
    
    subgraph "Orchestration Layer"
        API -->|1. Routing| Router{Query Router}
        Router -->|Hỏi đáp| QA[Legal QA Flow]
        Router -->|Liệt kê| Sector[Sector Search Flow]
        Router -->|Phân tích file| Conflict[Conflict Analyzer Flow]
        Router -->|Chat| General[General Chat Flow]
    end
    
    subgraph "Retrieval Engine"
        QA & Sector & Conflict -->|2. Hybrid Search| Qdrant[(Qdrant Vector DB)]
        Qdrant <-->|Embedding| Embed[BAAI/bge-m3 Model]
    end
    
    subgraph "Quality Control"
        QA -->|3. Reflection| ReflAgent[Reflection Agent]
        ReflAgent -.->|Nếu ảo giác| QA
    end
    
    subgraph "Persistence"
        API <-->|Short-term| Redis[(Redis)]
        API <-->|Long-term| SQLite[(SQLite)]
    end
    
    subgraph "Background Tasks"
        UI -.->|Tải lên| Upload[OCR / Document Ingestion]
        Upload --> Celery[Celery Worker]
        Celery --> Qdrant
    end
```

---

## 📂 Cấu trúc Repository

```text
Legal-RAG/
├── backend/            # Lõi xử lý FastAPI & RAG Agent
│   ├── agent/          # Logic 4 Flow (QA, Sector, Conflict, General Chat)
│   ├── api/            # API Endpoints & Session Management
│   ├── retrieval/      # Hybrid Search & Rerank logic
│   └── workers/        # Celery Background Worker (OCR/Ingestion)
├── frontend/           # Giao diện Next.js Web App (Premium UI)
├── scripts/            # Công cụ nạp dữ liệu (Ingest) & Snapshot
├── qdrant_snapshots/   # Nơi chứa file backup CSDL (.snapshot)
├── qdrant_storage/     # Dữ liệu Vector DB thực tế (Docker mount)
├── start_backend.ps1   # Script khởi động 1-click (Windows)
└── docker-compose.yml  # Triển khai Redis & Qdrant Containers
```

---

## 🚀 Hướng dẫn Cài đặt từ đầu (Zero to Hero)

### 1. Yêu cầu Hệ thống
- **Docker Desktop**: Chạy Redis và Qdrant.
- **Python 3.10+**: Cho Backend.
- **Node.js 18+**: Cho Frontend.
- **Dung lượng ổ đĩa**: Khoảng 5-10GB (để chứa Model Embedding bge-m3 và Vector DB).

### 2. Các bước triển khai

**Bước 1: Clone Repository**
```bash
git clone https://github.com/ngnam1104/Legal-RAG.git
cd Legal-RAG
```

**Bước 2: Cấu hình Môi trường (.env)**
Copy file mẫu và điền thông tin API Key (ưu tiên Gemini để có hiệu suất tốt nhất):
```bash
cp .env.example .env
```
*Lưu ý: Đảm bảo các đường dẫn CACHE (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`) trong `.env` trỏ về ổ đĩa có dung lượng trống.*

**Bước 3: Khởi động Database (Docker)**
Mở Docker Desktop, sau đó chạy:
```bash
docker-compose up -d
```

**Bước 4: Khôi phục Dữ liệu (Qdrant Snapshot)**
Nếu bạn có file snapshot (.snapshot) của CSDL Luật Việt Nam:
1. Copy file snapshot vào thư mục `./qdrant_snapshots/`.
2. Truy cập Dashboard Qdrant tại: `http://localhost:6335/dashboard`.
3. Chọn **Collections** -> Tạo collection mới (nếu chưa có) -> **Snapshots** -> **Restore from snapshot**.
4. (Hoặc dùng script nạp dữ liệu từ đầu):
   ```bash
   # Tạo venv và cài dependencies trước khi chạy script
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python scripts/ingest_local.py
   ```

**Bước 5: Khởi động toàn bộ Hệ thống**
Sử dụng script tự động (tốt nhất trên Windows):
```powershell
.\start_backend.ps1
```
Script sẽ tự động:
- Dọn dẹp các cổng 3000, 8000, 8001.
- Kích hoạt Python Venv và cài thư viện.
- Mở 4 cửa sổ riêng biệt cho: **Embedding Server**, **FastAPI Backend**, **Celery Worker**, và **Next.js Frontend**.

---

## 💻 Truy cập Ứng dụng

- **Giao diện người dùng**: `http://localhost:3000` (Giao diện Next.js Premium Dark Theme).
- **Backend API Docs**: `http://localhost:8000/docs`.
- **Qdrant Dashboard**: `http://localhost:6335/dashboard`.

---

## ⚙️ Cấu hình (Environment Variables)

| Biến | Mô tả |
| :--- | :--- |
| `LLM_PROVIDER` | `groq` \| `gemini` \| `ollama` |
| `QDRANT_URL` | Địa chỉ Qdrant (mặc định localhost:6335) |
| `REDIS_URL` | Địa chỉ Redis cho memory |
| `LEGAL_DENSE_MODEL`| Model embedding (mặc định `BAAI/bge-m3`) |

---

## 🛠️ Công nghệ Sử dụng

- **Models**: BGE-M3 (Embedding), Llama-3 (LLM), Gemini (Fallback).
- **Backend**: FastAPI, Celery, Redis.
- **Frontend**: Streamlit.
- **Vector DB**: Qdrant.
- **ORM/Storage**: SQLite (History), Redis (Cache/Queue).

