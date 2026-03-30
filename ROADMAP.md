Gap lớn nhất: app runtime hiện vẫn là dense-only và chunk chủ yếu theo mức `Điều`; phần parent-child sâu, hybrid dense+sparse và benchmark chi tiết đang nằm chủ yếu trong notebook, chưa đi vào luồng API/UI/worker production. Ngoài ra còn thiếu long-term memory, self-verification, local LLM adapter và benchmark RAM đúng spec.

### Giai đoạn 1: Foundation & Data Processing
* [x] **Bước 1.1: Tách repo thành kiến trúc module rõ ràng**
  Phương pháp: Đã chia thành các lớp `api`, `core`, `rag`, `ui`, `workers`, đủ để tách config, DB client, LLM adapter, retriever, worker OCR và UI.
  Output: Các package nguồn và entrypoints hiện có trong repo; trọng tâm là [api/main.py](D:/iCOMM/Legal-RAG/api/main.py), [core/config.py](D:/iCOMM/Legal-RAG/core/config.py), [rag/chat_engine.py](D:/iCOMM/Legal-RAG/rag/chat_engine.py), [ui/app.py](D:/iCOMM/Legal-RAG/ui/app.py).

* [x] **Bước 1.2: Kết nối dữ liệu từ HuggingFace và dựng pipeline ingest mẫu**
  Phương pháp: Đã dùng `load_dataset()` từ `th1nhng0/vietnamese-legal-documents`, lọc `Quyết định`, lấy mẫu phân tầng theo `legal_sectors`, rồi chunk/embed/upsert.
  Output: Hàm `prepare_dataset()` và `ingest_data()` trong [ingest.py](D:/iCOMM/Legal-RAG/ingest.py).

* [ ] **Bước 1.3: Nâng pipeline ingest từ mẫu 200 docs lên full-scale ~500k docs**
  Phương pháp: Cần viết ingestion job idempotent, chạy nền theo batch lớn, có resume/checkpoint, phân shard theo `document_id`, và tách offline indexing khỏi app runtime.
  Expected Output: `workers/ingestion.py` hoặc `scripts/build_index.py`, manifest/checkpoint ingest, lệnh chạy background cho full corpus.

* [ ] **Bước 1.4: Chuẩn hóa config và dọn artifact dev/legacy**
  Phương pháp: Hợp nhất đường cấu hình env, sửa template env đang đặt sai tên, và xử lý script cũ còn import module không tồn tại.
  Expected Output: file `.env.example` chuẩn, một ingestion entrypoint thống nhất, và dọn/repair [scripts/ingest_local.py](D:/iCOMM/Legal-RAG/scripts/ingest_local.py).

### Giai đoạn 2: Chunking, Embedding & Vector Index
* [x] **Bước 2.1: Xây dựng legal chunker mức Điều + Phụ lục trong app runtime**
  Phương pháp: Dùng regex tách `Căn cứ` / `Điều` / `Phụ lục`, gắn metadata header vào từng chunk trước khi embed.
  Output: Class `AdvancedLegalChunker` trong [core/nlp.py](D:/iCOMM/Legal-RAG/core/nlp.py).

* [x] **Bước 2.2: Có PoC chunking parent-child sâu đến mức Điều -> Khoản -> Điểm**
  Phương pháp: Notebook và script vá notebook đã có logic tách `chapter/article/clause/point`, vector hóa unit nhỏ nhưng vẫn giữ `parent_article_text`.
  Output: PoC trong [icomm-qdrant-vecdb.ipynb](D:/iCOMM/Legal-RAG/icomm-qdrant-vecdb.ipynb) và [patch_notebook_points.py](D:/iCOMM/Legal-RAG/patch_notebook_points.py).

* [x] **Bước 2.3: Tạo Qdrant client, collection và payload index cơ bản**
  Phương pháp: Đã có singleton Qdrant client, tự tạo collection nếu chưa có, và index các field filter cơ bản như `document_number`, `legal_type`, `issuance_date`.
  Output: `get_qdrant_client()` và `ensure_qdrant_collection()` trong [core/db.py](D:/iCOMM/Legal-RAG/core/db.py).

* [x] **Bước 2.4: Có PoC Hybrid Retrieval dense+sparse với Qdrant Native RRF**
  Phương pháp: Notebook đã encode dense + sparse bằng BGE-M3, dùng `prefetch` và `Fusion.RRF`, kèm payload filtering và quantization.
  Output: Hybrid cells trong [icomm-qdrant-vecdb.ipynb](D:/iCOMM/Legal-RAG/icomm-qdrant-vecdb.ipynb), cùng artefact benchmark/patch ở [patch_notebook_points.py](D:/iCOMM/Legal-RAG/patch_notebook_points.py).

* [ ] **Bước 2.5: Productionize parent-child + hybrid retrieval vào app runtime**
  Phương pháp: Hiện app còn dense-only trong `SentenceTransformer`; cần đưa chunker sâu, schema dense+sparse, one-pass encoder thật sự, và retriever hybrid vào `core/` + `rag/`.
  Expected Output: `core/hybrid_encoder.py`, cập nhật [rag/retriever.py](D:/iCOMM/Legal-RAG/rag/retriever.py), [rag/document_manager.py](D:/iCOMM/Legal-RAG/rag/document_manager.py), [core/db.py](D:/iCOMM/Legal-RAG/core/db.py).

### Giai đoạn 3: RAG Runtime, LLM & Memory
* [x] **Bước 3.1: Dựng adapter LLM cho Groq và Gemini**
  Phương pháp: Đã trừu tượng hóa lớp gọi LLM theo provider, cùng API shape chung `chat_completion(messages, provider, model)`.
  Output: Adapter trong [core/llm.py](D:/iCOMM/Legal-RAG/core/llm.py).

* [x] **Bước 3.2: Triển khai 3 mode nghiệp vụ QA / Related / Conflict**
  Phương pháp: `RAGEngine` đã map mode sang prompt/system behavior riêng và trả về answer + references.
  Output: Class `RAGEngine` trong [rag/chat_engine.py](D:/iCOMM/Legal-RAG/rag/chat_engine.py).

* [x] **Bước 3.3: Có query rewriting và short-term memory**
  Phương pháp: Đã lưu lịch sử chat ngắn hạn trong memory in-process và rewrite query trước khi retrieve ở mode QA.
  Output: `ChatSessionManager` và `rewrite_query()` trong [rag/chat_engine.py](D:/iCOMM/Legal-RAG/rag/chat_engine.py).

* [ ] **Bước 3.4: Nâng short-term memory thành long-term memory với SQLite/Redis**
  Phương pháp: Cần tách memory store bền vững theo `session_id`, có TTL/retention, load lại lịch sử giữa các process và giữa các lần restart.
  Expected Output: `rag/memory_store.py`, bảng SQLite hoặc Redis schema, và integration vào `RAGEngine`.

* [ ] **Bước 3.5: Bổ sung self-verification chống hallucinated citations**
  Phương pháp: Thêm một verifier LLM nhỏ đọc `answer + references + context` rồi reject/repair nếu bịa điều khoản/số hiệu.
  Expected Output: `rag/verifier.py` và một verification stage trong luồng `chat()`.

* [ ] **Bước 3.6: Mở rộng adapter để hỗ trợ Local LLM (vLLM/Ollama)**
  Phương pháp: Chuẩn hóa provider interface, tách config per backend, và thêm adapter local không phụ thuộc web search.
  Expected Output: `core/llm_local.py` hoặc mở rộng [core/llm.py](D:/iCOMM/Legal-RAG/core/llm.py) với backend `ollama`/`vllm`.

### Giai đoạn 4: Upload, OCR & Legal Workflows
* [x] **Bước 4.1: Có backend upload on-demand cho PDF/DOCX/DOC**
  Phương pháp: FastAPI nhận file, lưu temp file, đẩy Celery task để OCR/chunk/embed/upsert bất đồng bộ.
  Output: Endpoint `/upload-document` trong [api/main.py](D:/iCOMM/Legal-RAG/api/main.py), task `process_document_task` trong [workers/tasks.py](D:/iCOMM/Legal-RAG/workers/tasks.py), Celery app trong [workers/celery_app.py](D:/iCOMM/Legal-RAG/workers/celery_app.py).

* [x] **Bước 4.2: Có workflow phát hiện xung đột pháp lý khi thêm văn bản mới**
  Phương pháp: `DocumentManager` search tài liệu cũ, gọi LLM để so xung đột, rồi set payload `conflicted_by` trên Qdrant.
  Output: Class `DocumentManager` trong [rag/document_manager.py](D:/iCOMM/Legal-RAG/rag/document_manager.py).

* [ ] **Bước 4.3: Hoàn thiện UX upload file ở UI và trạng thái job**
  Phương pháp: Hiện UI mới cho nhập text trực tiếp; cần thêm `file_uploader`, polling `task_id`, progress/error surface, và hiển thị kết quả ingest/conflict.
  Expected Output: cập nhật [ui/app.py](D:/iCOMM/Legal-RAG/ui/app.py) và thêm API status endpoint trong [api/main.py](D:/iCOMM/Legal-RAG/api/main.py).

### Giai đoạn 5: Benchmark, Testing & Production Deploy
* [x] **Bước 5.1: Có artefact benchmark/smoke-test sơ bộ**
  Phương pháp: Repo đã có notebook hybrid, báo cáo Qdrant và script test end-to-end để kiểm tra DB/query/LLM pipeline.
  Output: [icomm-qdrant-vecdb.ipynb](D:/iCOMM/Legal-RAG/icomm-qdrant-vecdb.ipynb), [QDRANT_REPORT.md](D:/iCOMM/Legal-RAG/QDRANT_REPORT.md), [test_pipeline.py](D:/iCOMM/Legal-RAG/test_pipeline.py).

* [ ] **Bước 5.2: Chuẩn hóa benchmark đúng spec: index speed, query latency, RAM trên Qdrant Docker**
  Phương pháp: Cần benchmark script tái lập được trên Docker Qdrant với tập dữ liệu cố định, đo `chunks/s`, `queries/s`, p50/p95 latency, RAM/CPU, disk footprint, recall.
  Expected Output: `benchmarks/benchmark_qdrant.py`, file CSV/JSON kết quả, dashboard hoặc markdown report benchmark chuẩn.

* [x] **Bước 5.3: Có skeleton Docker/Compose để demo app**
  Phương pháp: Đã có `Dockerfile`, `docker-compose.yml`, Redis, healthcheck và wiring env cho app; đồng thời có profile `local-db` cho Qdrant.
  Output: [Dockerfile](D:/iCOMM/Legal-RAG/Dockerfile) và [docker-compose.yml](D:/iCOMM/Legal-RAG/docker-compose.yml).

* [ ] **Bước 5.4: Hardening để lên production thật**
  Phương pháp: Cần bổ sung test tự động, retry/timeout chuẩn, structured logging, metrics, secrets handling, CI/CD, backup/snapshot Qdrant, và chốt topology Docker Qdrant đúng spec thay vì chỉ dừng ở demo.
  Expected Output: `tests/`, workflow CI, logging/metrics config, playbook deploy production, runbook backup/restore Qdrant.

Kết luận: nền tảng chatbot, upload OCR, conflict detection, Groq/Gemini adapter và Docker demo đã có. Phần còn thiếu lớn nhất để khớp spec là: `500k ingestion production`, `parent-child chunking trong app`, `one-pass dense+sparse hybrid retrieval`, `long-term memory`, `self-verification`, và `benchmark Docker Qdrant có RAM/latency chuẩn`.
