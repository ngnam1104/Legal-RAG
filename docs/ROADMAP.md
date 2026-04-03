# Nâng cấp Legal-RAG với Pipeline Chuẩn, Conflict Analyzer & Advanced Mode 1 Features

Hệ thống Legal-RAG sẽ được nâng cấp kiến trúc để phục vụ đồng thời **2 Chế độ (Modes)** hoạt động chính thông qua một bộ Định tuyến (QueryRouter). 

## 1. Môi trường và Cấu hình (Environment & Configuration)
- **Qdrant**: Dữ liệu DB load từ Kaggle, có thể chạy qua `Local Path` hoặc `Docker Container`.
- **LLM**: Kết hợp đa mô hình qua API `Groq` (Agents tốc độ cao/HyDE/Clustering), `Gemini`, `Ollama` (Suy luận phức tạp/Chat).
- **Thư viện PDF**: Sử dụng `PyMuPDF` hoặc `pdfplumber` để trích xuất cấu trúc văn bản thuần.

---

## 2. Nâng cấp Tiền xử lý & Truy xuất Dữ liệu (Core RAG Engine)

Áp dụng cho CẢ HAI luồng hệ thống.

### [NEW] `backend/retrieval/chunker.py`
Tích hợp `AdvancedLegalChunker` bóc tách văn bản theo Regex Cây thứ bậc (Chương > Điều > Khoản), kèm Breadcrumb chi tiết.
- 🔗 **Cải tiến Citation Graph (GraphRAG Ingestion)**: Trích xuất thêm Metadata "Căn cứ pháp lý" (Ví dụ: `Căn cứ Luật Đất Đai năm 2024...`). Gắn ID của văn bản gốc (Parent Law) vào metadata của Nghị định/Thông tư làm cầu nối mạng lưới.

### [MODIFY] `backend/retrieval/hybrid_search.py` & `reranker.py`
- Tích hợp model `BAAI/bge-reranker-v2-m3` để rà soát ngữ nghĩa.
- Xây dựng mạng lưới **LegalHybridRetriever** 4 Bước mới:
  1. **Broad Retrieval**: Qdrant Hybrid RRF Search.
  2. **Cross-Encoder Reranking**: Lọc Top-K qua mô hình BGE.
  3. **Context Expansion**: Scroll API gọi thêm ngữ cảnh lân cận trong cùng một Điều.
  4. 🔗 **Citation Graph Retrieval**: Nếu văn bản lọt vào Top-K là Nghị định/Quyết định, tự động truy vấn ngược để kéo theo "Luật Gốc" (Dựa vào Metadata Căn cứ pháp lý) đưa vào Context.

---

## 3. Kiến trúc Đa Luồng (Query Router)

Cửa ngõ sẽ định tuyến Request qua 3 hệ thống độc lập:

### Luồng 1: Tìm kiếm & Tổng hợp Văn bản Liên quan (Sector / Document Search)
Mục tiêu: Trả lời cho các câu hỏi mang tính chất xin danh sách, tra cứu khung pháp lý tổng thể (VD: "Liệt kê cho tôi các văn bản hiện hành về PCCC").
- **Bước 1 - Định tuyến**: Router nhận diện intent tìm kiếm văn bản (Dựa vào từ khóa như "liệt kê", "danh sách văn bản", "tổng hợp").
- **Bước 2 - Truy xuất mở rộng**: Hybrid Retrieval lấy ra Top 15 - 20 văn bản liên quan nhất từ CSDL.
- **Bước 3 - Xử lý Context (Post-Retrieval)**:
  - **Document Clustering (Phân cụm)**: LLM quét nhanh tiêu đề và nội dung của Top 20 văn bản, tự động gom chúng thành các nhóm chủ đề (Ví dụ: Nhóm Tiêu chuẩn thiết kế, Nhóm Chế tài xử phạt).
  - **Regulatory Timeline (Dòng thời gian)**: Hệ thống trích xuất Metadata (Ngày ban hành, Trạng thái hiệu lực), dùng LLM sắp xếp các văn bản theo trục thời gian để vạch ra vòng đời của luật (Luật A sinh ra năm 2013 -> Bị sửa đổi bởi Nghị định B năm 2015 -> Bị thay thế hoàn toàn bởi Luật C năm 2024).
- **Bước 4 - Trả kết quả**: Xuất ra màn hình một bản báo cáo toàn cảnh gồm: Các cụm chủ đề văn bản + Dòng thời gian biến động, giúp user nắm bắt khung pháp lý mà không cần tự đọc từng file.

### Luồng 2: Trả lời Câu hỏi Kiến thức (Legal QA / Knowledge Traversal)
Mục tiêu: Trả lời trực tiếp các câu hỏi nghiệp vụ, tình huống pháp lý cụ thể (VD: "Công ty có được giữ bản gốc bằng đại học của nhân viên không?").
- **Bước 1 - Định tuyến**: Router nhận diện intent hỏi đáp nghiệp vụ.
- **Bước 2 - Tối ưu Câu hỏi (Query Rewrite)**: Dùng LLM viết lại câu hỏi dựa trên lịch sử chat để câu query mang đầy đủ ngữ cảnh nhất.
- **Bước 3 - Truy xuất Sâu (Hybrid Graph Retrieval)**: Tìm kiếm chính xác đến cấp độ Điều/Khoản/Điểm. Dùng Graph để kéo theo các văn bản hướng dẫn liên đới (VD: Tìm thấy Luật thì tự động kéo thêm Nghị định hướng dẫn chi tiết cho Điều luật đó).
- **Bước 4 - Suy luận & Tổng hợp**: LLM đọc các Điều/Khoản vừa lấy được, tổng hợp thành một câu trả lời mang tính tư vấn: Trực tiếp trả lời câu hỏi (Được hay Không được) và trích dẫn chính xác (Căn cứ tại Khoản x, Điều y, Luật z).

### Luồng 3: Phát hiện Xung đột & Rà soát Tài liệu (Agentic Conflict Analyzer)
Mục tiêu: Đóng vai trò là một "Pháp chế AI", rà soát file người dùng tải lên (Nội quy, Quy chế, Hợp đồng) để đối chiếu với Pháp luật hiện hành nhằm tìm ra điểm trái luật.
- **Bước 1 - Tiền xử lý File (Hierarchical Chunking)**: Dùng công cụ (như PyMuPDF) bóc tách file tải lên thành các chunk, duy trì nghiêm ngặt cấu trúc cây văn bản (Chương > Điều > Khoản).
- **Bước 2 - Trích xuất Mệnh đề (Structured IE)**: LLM đọc các chunk và ép kiểu dữ liệu xuất ra JSON (Pydantic) gồm `[Chủ Thể] + [Hành Vi] + [Điều Kiện] + [Hệ Quả]`. Đồng thời trích xuất `[Ngày ban hành]` & `[Cơ quan ban hành]` của file.
- **Bước 3 - Dịch thuật Pháp lý (HyDE Enhancement)**: LLM biên dịch các "Mệnh đề đời thường" vừa bóc tách thành "Ngôn ngữ pháp lệnh" để tối ưu hóa việc truy xuất vector (tăng Recall) trong CSDL Luật.
- **Bước 4 - Hội đồng AI Xét duyệt (LangGraph Workflow)**:
  - **Judge Agent (Trọng tài)**: Dùng kỹ thuật Chain-of-Thought (CoT) để suy luận phản biện dựa trên 3 trụ cột: Lex Superior, Lex Posterior, và Deontic Logic (Bắt buộc vs. Cấm đoán). Gắn nhãn Hợp pháp / Trung lập / Mâu thuẫn.
  - **Reviewer Agent (Kiểm duyệt)**: Đọc lại phán quyết của Judge Agent, đối chiếu chéo với nguyên bản luật để chống Ảo giác (Hallucination).
- **Bước 5 - Trả kết quả**: Xuất ra "Ma trận chứng cứ" (Markdown Table) hiển thị rõ ràng các điểm `conflict_text` (phần quy định bị trái luật) kèm giải thích.

---

## 4. Verification Plan
1. **Citation Graph Check**: Test luồng Ingestion xem các Nghị định có bắt dính và lưu chuẩn xác tên các Luật gốc vào metadata `base_laws` không. Khi search một Nghị định, Luật gốc có bị kéo theo không.
2. **Timeline & Clustering Check**: Đặt câu hỏi rộng (VD: "Quy định Đất Đai") vào Mode 1, kỳ vọng LLM trả về Timeline từ 2013 -> 2024 và phân cụm tài liệu thành Hướng dẫn vs Xử phạt.
3. **Conflict Agent Check**: Thử một văn bản Nội quy vi phạm luật, đảm bảo Agent bắt lỗi trúng đích và ra đúng bảng Markdown Table.

# Benchmarking & Evaluation

### PHẦN 1: BỘ CHỈ SỐ ĐO LƯỜNG LÕI (CORE METRICS)

#### 1. Đánh giá tốc độ truy vấn DB (Database Performance)
Với Qdrant hay ChromaDB, bạn không chỉ đo trung bình mà phải đo các "điểm đuôi" (tail latency) khi hệ thống chịu tải.
* **Latency (Độ trễ):** Đo bằng giây/milliseconds cho mỗi query. Phải theo dõi **P90** và **P99** (90% hoặc 99% các truy vấn phải hoàn thành dưới X giây). Trong hệ thống của bạn, mức lý tưởng cho Hybrid + Rerank là < 1.5s/query.
* **QPS/Throughput (Truy vấn/giây):** Khả năng chịu tải đồng thời. Số lượng query hệ thống xử lý được trong 1 giây mà không bị nghẽn.

#### 2. Độ liên quan giữa Câu hỏi & Tài liệu (Retrieval Metrics)
Đây là bài toán Information Retrieval (IR) truyền thống. Cần một tập dữ liệu Test (Ground Truth) gồm: *Câu hỏi* -> *Danh sách ID Điều luật mong đợi*.
* **Recall@K (Độ bao phủ):** Trong K tài liệu trả về, bạn tóm được bao nhiêu tài liệu đúng? *(Trong ngành Luật, Recall quan trọng hơn Precision vì tuyệt đối không được bỏ sót căn cứ pháp lý).*
* **Precision@K (Độ chính xác truy xuất):** Trong K tài liệu trả về, có bao nhiêu cái là hữu ích thật sự? (Tránh đưa rác vào LLM).
* **MRR (Mean Reciprocal Rank):** Đo lường xem tài liệu đúng (relevant) xuất hiện ở vị trí thứ mấy. Vị trí số 1 điểm cao nhất, số 2 giảm phân nửa, v.v.
* **NDCG@K:** Đánh giá chất lượng của mô hình Reranker (bge-reranker của bạn). Nếu tài liệu quan trọng nhất nằm ở top 1, điểm sẽ là 1.0.

#### 3. Độ chính xác của câu trả lời (Generation Metrics)
Sử dụng phương pháp **LLM-as-a-judge** (Dùng GPT-4 hoặc Claude 3.5 làm giám khảo để chấm điểm câu trả lời của hệ thống). Thường dùng framework **RAGAS** hoặc **TruLens**:
* **Faithfulness (Độ trung thành / Chống ảo giác):** Kiểm tra xem mọi nhận định trong câu trả lời của AI có thể được tìm thấy trong Context (văn bản luật) không. Với Legal RAG, điểm này bắt buộc phải tiệm cận **1.0**.
* **Answer Relevance (Độ đi sát vấn đề):** Câu trả lời có giải quyết trực tiếp câu hỏi không, hay đang lan man, dài dòng?
* **Answer Correctness (Độ chính xác so với đáp án mẫu):** Đo lường ngữ nghĩa giữa câu trả lời của AI và câu trả lời mẫu của Luật sư.

---

### PHẦN 2: CHIẾN LƯỢC ĐÁNH GIÁ 3 MODE CHUYÊN BIỆT CỦA BẠN

Mỗi Mode trong hệ thống của bạn phục vụ một mục đích khác nhau, do đó trọng số các metrics cũng phải khác nhau:

#### MODE 1: Tìm tài liệu liên quan (Sector/Document Search)
* **Mục tiêu:** Đóng vai trò thư viện viên, không được sót luật.
* **Metrics quan trọng nhất:** **Recall@10** và **NDCG@10**.
* **Cách test:** Tạo 50 tình huống tìm kiếm (VD: *"Danh sách văn bản về thuế GTGT"*). Kiểm tra xem Qdrant có lôi ra đủ các Luật, Nghị định, Thông tư cốt lõi trong top 10 không. Nếu thiếu văn bản gốc, điểm Recall sẽ thấp -> Cần tinh chỉnh lại trọng số `sparse` vs `dense` trong Hybrid Search.

#### MODE 2: Hỏi đáp hiểu biết (Knowledge QA)
* **Mục tiêu:** Đóng vai trò tư vấn viên, đưa ra câu trả lời chuẩn xác kèm trích dẫn.
* **Metrics quan trọng nhất:** **Faithfulness** và **Context Precision**.
* **Cách test:** Cho hệ thống trả lời 50 câu hỏi nghiệp vụ. Nếu AI tự bịa ra một mức phạt (Hallucination) hoặc trích dẫn sai số hiệu Điều khoản, điểm Faithfulness sẽ rớt thê thảm. Bạn có thể dùng RAGAS để tự động hóa việc chấm điểm này. Điểm số mục tiêu cho Legal-QA là `Faithfulness > 0.95`.

#### MODE 3: Phát hiện xung đột (Conflict Detection)
* **Mục tiêu:** Đóng vai trò thanh tra pháp chế (NLI - Natural Language Inference).
* **Metrics quan trọng nhất:** **F1-Score**, **Precision**, **Recall** cho từng nhãn (Entailment, Neutral, Contradiction).
* **Cách test:** Đây là bài toán phân loại (Classification). Bạn cần lập một **Confusion Matrix (Ma trận nhầm lẫn)**:
    * *False Positive (Báo động giả):* Nội quy không sai luật nhưng AI bảo sai. Gây phiền toái cho người dùng.
    * *False Negative (Bỏ lọt rủi ro):* Nội quy sai luật rành rành nhưng AI bảo Hợp pháp. **Đây là lỗi nghiêm trọng nhất**.
    * Do đó, trong Mode này, bạn phải tối ưu prompt sao cho **Recall của nhãn Contradiction** đạt mức cao nhất có thể.

---

Để đo lường chuẩn xác, trong thực tế các kỹ sư AI thường sử dụng thư viện **RAGAS (Retrieval Augmented Generation Assessment)** vì nó được thiết kế sẵn để tính các chỉ số như *Faithfulness, Context Recall, Answer Relevance* một cách tự động.



Memory: Dài hạn trong 1 phiên chat
Nhiều phiên chat
