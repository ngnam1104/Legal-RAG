# Kế hoạch Triển khai: Đánh giá Độ chính xác QA & Node Upload Tài liệu

Tài liệu này trình bày chi tiết kế hoạch thực hiện cho hai yêu cầu trong `todo.md`:
1. Test độ chính xác hỏi đáp trên 2 DB cho 3 văn bản cụ thể.
2. Thêm logic ưu tiên tài liệu tải lên khi hỏi đáp.

> [!IMPORTANT]
> **Yêu cầu Phản hồi từ Người dùng:**
> 1. Về Node Upload: Tôi đề xuất **không nạp (ingest) thẳng tài liệu chat vào DB chính** để tránh rác DB. Thay vào đó, tài liệu tải lên sẽ được chunk, embed và lưu vào RAM tạm thời của phiên chat (In-memory Vector Search). Bạn có đồng ý với hướng tiếp cận này không?
> 2. Về Dataset Test: Tôi sẽ viết script dùng LLM tự động đọc nội dung 3 văn bản từ DB và sinh ra các câu hỏi/đáp án (Ground Truth) để tạo file JSON test. Bạn có muốn bổ sung thêm kịch bản test đặc biệt nào không?

---

## 1. Task 1: Đánh giá Độ chính xác QA cho 3 Văn bản

**Mục tiêu:** Khai phá toàn bộ nội dung của 3 văn bản (`55/2010/QH12`, `1620/QĐ-UBND`, `3192/2000/QĐ-BYT`), tạo bộ test dataset và đánh giá độ chính xác của 2 mode (`GENERAL_CHAT`, `LEGAL_CHAT`).

### Các bước thực hiện:
1. **Tạo Test Dataset (Synthetic Data Generation):**
   - Tạo script `scripts/generate_test_dataset.py`.
   - Script sẽ truy vấn Qdrant/Neo4j để lấy toàn bộ chunks của 3 văn bản trên.
   - Sử dụng LLM (Groq) để tự động sinh ra khoảng 15-20 câu hỏi & đáp án mẫu (Ground Truth) cho mỗi văn bản, bao phủ các phân lớp:
     - Siêu dữ liệu & Căn cứ
     - Phạm vi & Thời gian
     - Nội dung thực chất
     - Logic xử lý xung đột
     - Liên kết văn bản
   - Lưu kết quả vào `tests/qa_evaluation/Chatbot_test_2mode_3docs.json`.

2. **Chạy & Đánh giá (LLM-as-a-Judge):**
   - Viết lại hoặc mở rộng test runner thành `tests/qa_evaluation/evaluate_accuracy.py`.
   - Script sẽ chạy từng câu hỏi qua pipeline `LegalRAGWorkflow` hiện tại.
   - LLM Thẩm phán (LLM Judge) sẽ so sánh câu trả lời của RAG với Ground Truth và chấm điểm `✅ ĐẠT` hoặc `❌ KHÔNG ĐẠT`.
   - Xuất báo cáo tổng hợp tỷ lệ Accuracy và Latency cho 2 mode.

---

## 2. Task 2: Node Upload Tài liệu & Ưu tiên Hỏi đáp

**Mục tiêu:** Xử lý tài liệu người dùng tải lên, ưu tiên trả lời dựa trên tài liệu này, sau đó mới fallback hoặc bổ sung từ DB.

### Quyết định Thiết kế (Thiết kế In-Memory Search):
- **Không ném vào DB chính**: Tài liệu tải lên trong phiên chat thường là tài liệu cá nhân, dự thảo, hoặc chỉ dùng một lần. Việc ném vào DB Neo4j/Qdrant chính sẽ làm rác DB và tốn thời gian ingest.
- **Giải pháp**: Xây dựng **In-Memory Vector Search** theo chunk. Khi người dùng upload:
  1. API `/api/upload` sẽ Parse -> Chunk -> **Embed (sinh vector)** các chunk này.
  2. Lưu danh sách (Text + Vector) vào bộ nhớ tạm của Session thông qua `rag_engine.memory.set_temp_chunks()`.

### Các bước thực hiện:
1. **Cập nhật Upload API (`backend/api/main.py`):**
   - Bổ sung bước gọi `embedder.encode_documents()` để lấy vector cho từng chunk sau khi parse.
   - Gắn vector vào dict của mỗi chunk trước khi gọi `set_temp_chunks()`.

2. **Cập nhật Node Retrieve (`backend/agent/legal_chat.py`):**
   - Khi có `file_chunks` trong `state`:
     - Embed câu hỏi (query) của người dùng.
     - Tính Cosine Similarity giữa vector câu hỏi và vector của các `file_chunks` đang nằm trong RAM.
     - Lấy Top-K chunks phù hợp nhất từ file upload.
   - Vẫn tiến hành Phase 1 (Hybrid Search Qdrant) và Phase 2 (Neo4j) để lấy thêm tài liệu đối chiếu từ hệ thống.

3. **Cập nhật Context Builder (`backend/agent/utils_legal.py`):**
   - Hàm `build_legal_context` sẽ nhận Top-K chunks của file tải lên và đặt lên **đầu ngữ cảnh** trong thẻ `<tai_lieu_tam>`.
   - Các hits từ hệ thống DB sẽ đặt dưới thẻ `<tai_lieu_db>`.
   - Cập nhật Prompt (`GRAPHRAG_PROMPT` và `ANSWER_PROMPT`) để chỉ định LLM: *"Hãy ưu tiên tìm câu trả lời trong <tai_lieu_tam>. Nếu không đủ, mới dùng đến <tai_lieu_db>."*

---

## Mức độ Ảnh hưởng (Proposed File Changes)

#### [MODIFY] [backend/api/main.py](file:///d:/iCOMM/Legal-RAG/backend/api/main.py)
- Thêm bước sinh vector (embedding) cho tài liệu tải lên tại endpoint `/api/upload`.

#### [MODIFY] [backend/agent/legal_chat.py](file:///d:/iCOMM/Legal-RAG/backend/agent/legal_chat.py)
- Thêm logic tính toán Cosine Similarity cục bộ cho `file_chunks` trong hàm `retrieve()`.

#### [MODIFY] [backend/agent/utils_legal.py](file:///d:/iCOMM/Legal-RAG/backend/agent/utils_legal.py)
- Refactor `build_legal_context` để truyền chính xác các chunk được lọc vào thẻ ưu tiên.

#### [NEW] [scripts/generate_test_dataset.py](file:///d:/iCOMM/Legal-RAG/scripts/generate_test_dataset.py)
- Script truy vấn Neo4j/Qdrant và gọi LLM để sinh 60 cặp QA cho 3 văn bản mục tiêu.

#### [NEW] [tests/qa_evaluation/evaluate_accuracy.py](file:///d:/iCOMM/Legal-RAG/tests/qa_evaluation/evaluate_accuracy.py)
- Test runner dùng LLM-as-a-Judge đánh giá RAG pipeline cho 2 mode.

---

## Verification Plan

### Automated Tests
1. Chạy `python scripts/generate_test_dataset.py` để đảm bảo sinh thành công file JSON với nội dung bám sát 3 văn bản chỉ định.
2. Chạy `python tests/qa_evaluation/evaluate_accuracy.py` để verify tỷ lệ Đạt (Accuracy) > 80% trên bộ test tự sinh.

### Manual Verification
1. Dùng Postman hoặc frontend upload 1 file PDF mới (chưa có trong DB).
2. Chat hỏi một câu có trong file PDF đó. Check terminal log để đảm bảo hệ thống sử dụng In-memory Vector Search để trích xuất chunk từ thẻ `<tai_lieu_tam>` thay vì quét toàn bộ file.
3. Chat hỏi một câu cần đối chiếu giữa file tải lên và luật hiện hành trong DB, kiểm tra xem LLM có kết hợp được cả `<tai_lieu_tam>` và `<tai_lieu_db>` không.
