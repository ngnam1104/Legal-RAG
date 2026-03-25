# Legal-RAG
# Chatbot Văn Bản Pháp Luật Việt Nam (RAG)

  ## Báo cáo ngày 24/03/2026

  ### 1. Xây dựng Vector Database (Qdrant) — ĐÃ HOÀN THÀNH ✅

  #### 1.1. Nguồn dữ liệu
  - Dataset: **`th1nhng0/vietnamese-legal-documents`** trên Hugging Face (518,255 văn bản).
  - Lọc ra **2,000 văn bản** loại "Quyết định" bằng phương pháp **Stratified Sampling** (lấy mẫu phân tầng đều theo 1,063 lĩnh vực pháp luật), đảm bảo tính đa dạng và đại diện.
  - Tách riêng **200 văn bản** làm tập test cho RAG.

  #### 1.2. Quy trình xử lý dữ liệu (Pipeline)
  ```
  Raw Text → AdvancedLegalChunker → Embedding (bge-m3) → Qdrant Upsert
  ```

  **Chi tiết từng bước:**

  | Bước | Mô tả | Công cụ |
  |------|--------|---------|
  | **Chunking** | Chia văn bản theo cấu trúc pháp lý: tách riêng Phần căn cứ, từng Điều khoản, và Phụ lục. Mỗi chunk được gắn header metadata (số hiệu, tiêu đề, cơ quan ban hành) | `AdvancedLegalChunker` (tự viết) |
  | **Embedding** | Chuyển text → vector 1024 chiều. Chạy batch trên GPU T4 của Kaggle | `BAAI/bge-m3` (SentenceTransformer) |
  | **Upsert** | Đẩy lên Qdrant Cloud theo batch 100 points/lần | `qdrant-client` |
  | **Indexing** | Tạo Payload Index cho các trường: `is_appendix`, `legal_type`, `document_number`, `issuance_date`, `legal_sectors`, `title` | Qdrant Payload Index |
  | **Quantization** | Nén vector bằng Scalar Quantization (int8) để giảm RAM | `ScalarQuantization` |

  #### 1.3. Kết quả Database
  - **Collection**: `legal_vn_200_docs`
  - **Tổng số points (chunks)**: 3,048 (từ 200 văn bản)
  - **Vector dimension**: 1024 (BAAI/bge-m3)
  - **Distance metric**: Cosine
  - **Quantization**: int8 (giảm ~75% RAM)
  - **Hosting**: Qdrant Cloud (US-West-1)

  #### 1.4. Cấu trúc Payload mỗi Point
  ```json
  {
    "document_id": "680040",
    "document_number": "1415/QĐ-UBND",
    "title": "Quyết định 1415/QĐ-UBND năm 2025...",
    "legal_type": "Quyết định",
    "legal_sectors": "Bộ máy hành chính, Xây dựng - Đô thị",
    "issuance_date": "07/11/2025",
    "issuing_authority": "Tỉnh Đồng Tháp",
    "article_ref": "Điều 1.",
    "is_appendix": false,
    "chunk_text": "[THÔNG TIN TRÍCH DẪN]..."
  }
  ```

  #### 1.5. Kiểm thử Retrieval
  Đã test thành công trên Kaggle Notebook với nhiều kịch bản:
  - Truy vấn theo lĩnh vực (đường bộ, giao thông) → Score 0.62
  - Truy vấn theo mã hồ sơ TTHC → Score 0.54
  - Truy vấn theo căn cứ pháp lý → Score 0.58

  Notebook đầy đủ: [`icomm-qdrant-vecdb.ipynb`](icomm-qdrant-vecdb.ipynb)

  ---

  ### 2. Kiến trúc Chatbot — ĐÃ HOÀN THÀNH ✅

  Tái cấu trúc repo thành 5 package:

  ```
  ChatbotVBPL/
  ├── api/          # FastAPI endpoints (/chat, /upload-document)
  ├── core/         # Config, DB singleton, NLP (Chunker + Embedder)
  ├── rag/          # RAGEngine (3 modes), Retriever, DocumentManager
  ├── workers/      # Celery background tasks (OCR + Ingestion)
  ├── ui/           # Streamlit Web Chatbot
  ├── Dockerfile
  ├── docker-compose.yml
  └── run_local.ps1
  ```

  ### 3. Tính năng Chatbot — ĐÃ HOÀN THÀNH ✅
  - **3 chế độ hỏi đáp**: Q&A, Tìm VBPL liên quan, Phát hiện xung đột
  - **Conversation memory**: Lưu 7 lượt chat gần nhất
  - **Upload on-demand**: Hỗ trợ PDF (scan OCR), DOCX, DOC
  - **Conflict Detection**: Tự động phát hiện xung đột khi thêm văn bản mới

  ### 4. LLM Engine — ĐÃ HOÀN THÀNH ✅
  - Sử dụng **Groq API** (model `llama3-8b-8192`, mã nguồn mở)
  - Giao tiếp qua chuẩn OpenAI-compatible API
  - Embedding local bằng `BAAI/bge-m3` (SentenceTransformer, chạy CPU)

  ### 5. Triển khai — ĐÃ HOÀN THÀNH ✅
  - **Local**: `run_local.ps1` (tự tạo venv, cấu hình .env, chạy Streamlit)
  - **Docker**: `docker-compose up -d --build` (5 services: Redis, Qdrant, API, Worker, UI)
