# BÁO CÁO KỸ THUẬT HỆ THỐNG LEGAL-RAG

## 1. Kiến trúc Hệ thống (System Architecture)

### 1.1. Các thành phần chính
- **Frontend (Next.js/React):** Giao diện tương tác người dùng, quản lý phiên chat (Session) và xử lý hiển thị trích dẫn (Citations) từ văn bản luật gốc.
- **Backend (FastAPI):** Đóng vai trò là máy chủ API.
- **Thành phần Xử lý Tài liệu (Document Processing Pipeline):**
    - **Document Parser:** Sử dụng thư viện `PyMuPDF (fitz)` để xử lý tệp PDF và `python-docx` cho tệp Word. Module này chịu trách nhiệm trích xuất văn bản thô (Raw Text) từ các tài liệu được tải lên.
    - **Structure-Aware Chunking:** Áp dụng logic từ `AdvancedLegalChunker` để tự động nhận diện cấu trúc phân cấp (Chương, Điều, Khoản) ngay cả trên các tài liệu người dùng tải lên, đảm bảo tính toàn vẹn ngữ nghĩa.
    - **Celery Workers:** Đảm nhận việc xử lý các tác vụ nặng như OCR (nếu cần), trích xuất đặc trưng và nạp dữ liệu vào Vector DB một cách bất đồng bộ để tránh gây nghẽn hệ thống.
- **Agent Orchestrator (LangGraph):** "Bộ não" điều phối toàn bộ vòng đời của một yêu cầu. Sử dụng đồ thị trạng thái (State Graph) để quản trị các bước: Viết lại câu hỏi -> Tìm kiếm -> Rerank -> Tổng hợp -> Phản hồi.
- **Hệ thống Lưu trữ & Cơ sở dữ liệu:**
    - **Lưu trữ Dài hạn (Persistent Storage):**
        - **SQLite:** Lưu trữ toàn bộ lịch sử hội thoại (Chat History), thông tin phiên chat (Sessions), và tiêu đề phiên. Đây là nơi lưu giữ dữ liệu người dùng để có thể khôi phục các cuộc hội thoại cũ.
        - **Qdrant:** Lưu trữ bền vững các Vector nhúng (Dense & Sparse) cùng toàn bộ Metadata của 5.000 văn bản pháp luật (242k chunks). Đây là "kho tri thức" chính của hệ thống.
        - **File System:** Lưu trữ các tệp văn bản pháp luật gốc (PDF/Docx) và các bản snapshot của cơ sở dữ liệu để phục vụ việc sao lưu.
    - **Lưu trữ Ngắn hạn & Đệm (In-memory Storage):**
        - **Redis:** Đóng vai trò vừa là **Message Broker** cho Celery, vừa là bộ đệm tốc độ cao cho toàn bộ hệ thống.
        - **Celery:** Quản lý và thực thi hàng đợi các tác vụ nền (Background Jobs).
        - **Vai trò cụ thể trong hệ thống:**
            - **Indexing & Ingestion:** Xử lý nạp văn bản quy mô lớn (Batch Ingestion). Khi người dùng nạp dữ liệu, Celery sẽ nhận nhiệm vụ từ Redis và thực hiện Chunking -> Embedding -> Upsert Qdrant một cách bất đồng bộ để không gây treo giao diện.
            - **Quản lý Task Status:** Lưu trữ trạng thái của các tác vụ đang xử lý (Pending, Processing, Completed) giúp người dùng theo dõi tiến độ nạp liệu theo thời gian thực.
            - **Tối ưu hóa Điều phối:** Giảm độ trễ bằng cách giải phóng tài nguyên cho Backend chính (FastAPI), chỉ tập trung xử lý yêu cầu HTTP và giao nhiệm vụ nặng cho các Workers.
- **Vector Database (Qdrant):** Lưu trữ và tìm kiếm vector đa nhiệm (Hybrid Search). Hỗ trợ lọc (Filtering) dựa trên siêu dữ liệu (Metadata) như Số hiệu văn bản, Loại văn bản.
- **Mô hình AI (Core AI Models):**
    - **Embedding Model:** **BGE-M3** (BAAI) - Đóng vai trò then chốt trong việc trích xuất vector ngữ nghĩa và từ khóa.
    - **Rerank Model:** **cross-encoder/ms-marco-MiniLM-L-6-v2** - Tối ưu hóa thứ hạng kết quả tìm kiếm, đảm bảo độ chính xác pháp lý cao nhất.
    - **Large Language Models (LLMs):** Cơ chế **Lazy-loading LLM Clients** linh hoạt hỗ trợ:
        - **Groq (Llama 3):** Ưu tiên tốc độ phản hồi cực nhanh (Đang sử dụng: LLaMA 3.1 8B Instant - Context window: lên đến 128K tokens).
        - **Gemini (Flash/Pro):** Xử lý các tài liệu có độ dài ngữ cảnh lớn.
        - **Ollama:** Đảm bảo quyền riêng tư khi chạy các mô hình nội bộ (Local).

- **Hạ tầng (Infrastructure):**
    - **Docker & Docker Compose:** Đóng gói các dịch vụ cơ sở hạ tầng thiết yếu bao gồm **Qdrant DB** (Vector Database) và **Redis** (Message Broker/Cache). Việc sử dụng Docker giúp đảm bảo các dịch vụ lưu trữ này luôn chạy ổn định trên mọi môi trường mà không cần cấu hình phức tạp.
    - **Triển khai ứng dụng:** Các thành phần khác như **Frontend (Next.js)** và **Backend (FastAPI/Celery)** được triển khai trực tiếp thông qua quản lý mã nguồn (Git) và cài đặt môi trường (Environment setup) tương ứng sau khi clone dự án, giúp tối ưu hóa khả năng can thiệp trực tiếp vào mã nguồn trong quá trình phát triển.

## 2. Quy trình Xử lý Dữ liệu 

Trái tim của hệ thống là quy trình tìm kiếm 5 bước, được tối ưu hóa đặc biệt cho dữ liệu pháp luật Việt Nam.

### 2.1. Ingestion & Chunking (Xử lý đầu vào)
Hệ thống sử dụng module `AdvancedLegalChunker` (đã được tinh chỉnh trong [notebook/legal_rag_qdrant_kaggle.ipynb](notebook/legal_rag_qdrant_kaggle.ipynb)) để chuyển đổi văn bản thô thành các đơn vị tri thức có cấu trúc.

- Đầu vào: 5000 văn bản - 242.000 Chunks: Đây là quy mô dữ liệu tối ưu cho phiên bản hiện tại. Do giới hạn về tài nguyên phần cứng (GPU VRAM) và dung lượng lưu trữ vector (Vector Storage), hệ thống hiện dừng lại ở mức 242.000 chunks để đảm bảo tốc độ phản hồi và độ chính xác cao nhất (không thực hiện nâng cấp lên quy mô 500.000 văn bản như dự kiến ban đầu để tránh gây treo hệ thống).

- **Hierarchical Regex Chunking:** Thay vì cắt theo độ dài cố định (Fixed-size), hệ thống sử dụng các biểu thức chính quy (Regex) để nhận diện cấu trúc cây của văn bản luật: `Chương > Điều > Khoản`, hay từng phần 1, 2, 3 trong Phụ Lục. Điều này đảm bảo mỗi "chunk" là một đơn vị pháp lý độc lập, toàn vẹn về ý nghĩa. 
- **Smart Metadata Enrichment:** Mỗi đơn vị văn bản được làm giàu bằng bộ siêu dữ liệu (Metadata) chi tiết, hỗ trợ lọc chính xác tại tầng Vector DB:
    - **Cấu trúc:** `article_id`, `reference_citation`.
    - **Pháp lý:** `legal_type`, `document_number`, `legal_sectors`.
- **Đầu ra: Payload dữ liệu thực tế**
    ```json
    {
      "document_id": "493941",
      "document_uid": "doc::quyetdinh::445-qd-bnnmt::2026",
      "chunk_id": "493941::article::2::clause::1",
      "document_number": "445/QĐ-BNNMT",
      "title": "Quyết định 445/QĐ-BNNMT về việc công bố thủ tục hành chính bị bãi bỏ...",
      "legal_type": "Quyết định",
      "legal_sectors": ["Đất đai", "Thủ tục hành chính"],
      "article_ref": "Điều 2",
      "clause_ref": "Khoản 1",
      "reference_citation": "445/QĐ-BNNMT | Điều 2 | Khoản 1",
      "chunk_text": "[LEGAL HEADER]\n- Title: Quyết định 445/QĐ-BNNMT...\n- Breadcrumb: Điều 2 > Khoản 1\n[NOI DUNG DIEU/KHOAN]\n1. Bãi bỏ các nội dung công bố tại Quyết định số 2304/QĐ-BNNMT...",
      "legal_basis_refs": [
        {
          "doc_number": "35/2025/NĐ-CP",
          "doc_title": "Nghị định số 35/2025/NĐ-CP...",
          "parent_law_id": "parent::decree::35-2025-nd-cp::2025"
        }
      ],
      "is_active": true
    }
    ```
- **Contextual Embedding:** Nội dung đưa vào model Vector không chỉ có văn bản thuần mà bao gồm cả `[LEGAL HEADER]` chứa Tiêu đề và reference_citation. Kỹ thuật này giúp model "hiểu" rõ đoạn văn đang thuộc văn bản nào và vị trí pháp lý nào trong hệ thống.

- **High-Precision Indexing (Float32):** Hệ thống sử dụng cấu hình vector `float32` nguyên bản (không nén/quantization). Mặc dù tiêu tốn nhiều bộ nhớ hơn, nhưng điều này đảm bảo giữ lại 100% đặc trưng ngữ nghĩa từ model BGE-M3, giúp đạt độ chính xác **Precision@10 = 1.0** trong môi trường thử nghiệm.
- **Payload Indexing & Filtering:** Để tối ưu hóa tốc độ truy vấn trên 242.000 chunks, hệ thống thiết lập các chỉ mục (Indexes) trên các trường thuộc tính quan trọng:
    - **Keyword Index:** Áp dụng cho `document_id`, `document_number`, `legal_sectors`, `article_ref`, giúp lọc (Filtering) tức thì khi người dùng yêu cầu tra cứu văn bản cụ thể.
    - **Full-Text Index:** Áp dụng cho trường `title` và `reference_citation` (sử dụng tokenizer `word`), cho phép tìm kiếm từ khóa linh hoạt ngay trong metadata của văn bản.
    - **On-Disk Optimization:** Cấu hình `on_disk=True` cho Dense Vectors để giảm tải RAM, đồng thời duy trì hiệu năng cao cho các tác vụ Hybrid Search.

#### Hiệu năng Xử lý & Đánh chỉ mục (Indexing Performance)
Dựa trên kết quả thực tế từ các bài kiểm tra hiệu năng ([docs/time_delay.md](docs/time_delay.md)):

- **Tổng thời gian Indexing:** ~1.19 giờ cho 5.000 văn bản (~242.245 chunks).
- **Phân bổ thời gian (Bottleneck):**
    - **Upsert Qdrant:** Chiếm **46.1%** (do kết nối mạng/disk I/O khi đẩy lượng lớn vector).
    - **Embedding (Dense + Sparse):** Chiếm **53.2%** tổng thời gian xử lý trên GPU T4.
- **Tốc độ truy xuất (Search Latency):**
    - **Search Regular:** ~22ms.
    - **Hybrid Search (bao gồm Embedding trên CPU):** 1.5s - 1.9s.
    - **Reranking (CPU):** ~8.6s (Đây là khâu tốn thời gian nhất nếu không có GPU hỗ trợ).

### 2.3. Chiến lược Embedding & Hybrid Search

Hệ thống sử dụng mô hình đa nhiệm **BGE-M3** làm nhân tố cốt lõi cho cả hai luồng tìm kiếm, cho phép tối ưu hóa tài nguyên và tăng cường độ chính xác.

- **Mô hình BGE-M3 (BAAI/bge-m3):** Một mô hình "all-in-one" hỗ trợ đa ngôn ngữ, đặc biệt mạnh mẽ với tiếng Việt. 
    - **Dense Vector:** Trích xuất vector 1024 chiều (1024-dim), đại diện cho toàn bộ ngữ nghĩa của đoạn văn.
    - **Sparse Vector (Lexical):** Trích xuất trọng số Lexical cho từng token, tương tự BM25 nhưng có khả năng học được trọng số từ dữ liệu. 
    - **Cấu trúc Tokenization:** Điểm đặc biệt của BGE-M3 là khả năng sinh đồng thời cả Dense và Sparse trong một lần xử lý duy nhất (Single Forward Pass). Hệ thống cấu hình xử lý tối ưu với độ dài mỗi đoạn văn (chunk) khoảng 1024 ký tự, đảm bảo model bao quát được toàn bộ ngữ cảnh mà không bị mất mát thông tin do vượt ngưỡng cửa sổ ngữ cảnh (Context Window).
- **Cơ chế Hybrid Search:**
    1. **Dense Retrieval (Semantic):** Trích xuất vector 1024 chiều. Sử dụng khoảng cách Cosine trên Qdrant để tìm các đoạn văn bản có sự tương đồng về ý nghĩa (ví dụ: truy vấn "sai phạm" khớp với kết quả "vi phạm").
    2. **Sparse Retrieval (Keyword):** Trích xuất trọng số Lexical cho từng token. Kỹ thuật này bù đắp cho Dense search bằng cách tìm chính xác các thực thể pháp lý (Số hiệu văn bản: `445/QĐ-BNNMT`, Thuật ngữ chuyên môn: `tranh chấp đất đai`).
- **Triển khai kỹ thuật (Code implementation):**
    Việc mã hóa và tìm kiếm được thực hiện thông qua lớp `LocalBGEHybridEncoder`, tích hợp trực tiếp với tính năng `prefetch` và `Fusion.RRF` của Qdrant:
    ```python
    # Thực hiện mã hóa đồng thời (Speed Optimization)
    dense_vecs, sparse_vecs = model.encode(
        texts,
        return_dense=True,
        return_sparse=True
    )

    # Truy vấn Hybrid trên Qdrant với thuật toán RRF
    raw_hits = qdrant_client.query_points(
        collection_name="legal_rag",
        prefetch=[
            models.Prefetch(query=dense_vecs, using="dense", limit=50),
            models.Prefetch(query=sparse_vecs, using="sparse", limit=50),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    )
    ```
- **Reciprocal Rank Fusion (RRF):** Kết hợp kết quả từ hai nguồn trên với hằng số $k=60$. 
    - **Công thức:** Điểm số cuối cùng được tính theo công thức: 
    $$score(d) = \sum_{r \in R} \frac{1}{k + rank(d, r)}$$
    Trong đó $R$ là tập hợp các bảng xếp hạng (Dense và Sparse), $rank(d, r)$ là thứ hạng của tài liệu $d$ trong bảng xếp hạng $r$.
    - **Ý nghĩa:** Thuật toán này giúp cân bằng giữa độ chính xác về từ khóa (Sparse) và chiều sâu về ngữ nghĩa (Dense) vốn có của văn bản pháp luật, đảm bảo các tài liệu xuất hiện ở vị trí cao trong cả hai phương pháp tìm kiếm sẽ được ưu tiên lên đầu.


### 2.4. Tối ưu hóa Sau Tìm kiếm (Post-Retrieval)
Để đảm bảo chất lượng ngữ cảnh đưa vào LLM, hệ thống thực hiện một chuỗi các thao tác hậu xử lý tinh vi:

- **Reranking (Cross-Encoder):**
    - **Cơ chế:** Sau khi có kết quả từ bước Hybrid Search, hệ thống sử dụng điểm số **RRF** làm cơ sở để chọn ra Top 40 ứng viên tiềm năng nhất. Sau đó, mô hình **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) sẽ thực hiện đánh giá lại (re-score) các ứng viên này bằng cách tính toán tương quan trực tiếp giữa Câu hỏi và từng Đoạn văn --> Top 10 ứng viên tốt nhất
    - **Hiệu quả:** Việc rerank lại dựa trên điểm RRF giúp tinh lọc các kết quả "suýt soát" về mặt từ khóa hoặc ngữ nghĩa, đẩy các văn bản có tính liên quan thực sự lên đầu. Điều này cực kỳ quan trọng trong luật pháp vì chỉ cần sai một thuật ngữ (ví dụ: "có quyền" vs "có nghĩa vụ") cũng làm thay đổi hoàn toàn bản chất câu trả lời.

- **Context Expansion (Small-to-Big):**
    - **Cơ chế:** Khi tìm thấy một `Khoản` (đơn vị nhỏ), hệ thống sử dụng siêu dữ liệu `article_id` và `document_uid` để truy vấn ngược lại Qdrant, lấy thêm các `Khoản` lân cận và `Tiêu đề Điều` liên quan.
    - **Mục tiêu:** Khôi phục lại tính toàn vẹn của Điều luật. Việc cung cấp cho LLM một Điều luật đầy đủ giúp nó hiểu rõ các điều kiện loại trừ hoặc các trường hợp áp dụng đặc thù thường nằm ở các Khoản khác nhau trong cùng một Điều.
    - **Hướng phát triển:** Mở rộng khả năng truy vấn sang các văn bản có liên quan trực tiếp (Cross-document retrieval) dựa trên sự tương đồng của các chunk tri thức, giúp xây dựng mạng lưới tri thức pháp luật đa chiều.

- **Lost-in-the-Middle Reordering:**
    - **Cơ chế:** Dựa trên nghiên cứu về việc LLM thường chú ý tốt nhất vào nội dung ở đầu và cuối Prompt, hệ thống sắp xếp lại các đoạn ngữ cảnh.
    - **Thứ tự:** Các đoạn có Score Rerank cao nhất được đặt ở vị trí số 1 và vị trí cuối cùng trong danh sách ngữ cảnh, các đoạn trung bình nằm ở giữa. Điều này giúp tối ưu hóa khả năng trích xuất thông tin của mô hình ngôn ngữ lớn (đặc biệt là các model như Llama 3 hoặc Gemini).

- **Legal Citation Formatting:** Tự động chuẩn hóa các trích dẫn theo định dạng `[Số hiệu văn bản | Điều X | Khoản Y]` ngay trong ngữ cảnh đưa vào Prompt. Điều này ép LLM phải sử dụng đúng định dạng trích dẫn khi trả lời, giúp người dùng dễ dàng đối soát với văn bản gốc.

## 3. Các Luồng Thực thi Agent (Agentic Flows)

Hệ thống được thiết kế theo kiến trúc **Multi-Agent**, chuyển dịch dần sang mô hình **LangGraph** để quản lý trạng thái và điều phối các tác vụ phức tạp. Việc sử dụng LangGraph cho tầng điều hướng giúp hệ thống có khả năng tự phục hồi (Self-correction) và lập luận đa bước.

### 3.1. Kiến trúc Điều hướng với LangGraph (Orchestration Layer)
Thay vì dùng các câu lệnh rẽ nhánh `if-else` cứng nhắc, hệ thống áp dụng một State Graph (Đồ thị trạng thái) để quản lý luồng đi của câu hỏi:

- **State (Trạng thái):** Một đối tượng chung lưu trữ xuyên suốt quá trình xử lý bao gồm: `question`, `condensed_query`, `documents`, `answer`, `is_legal_related`, và `retry_count`.
- **Node (Nút xử lý):**
    - **`QuestionRewriter`:** Cô đọng câu hỏi dựa trên lịch sử.
    - **`IntentRouter`:** Phân tích ý định để quyết định đi tiếp vào `LegalRetriever` hay `GeneralChat`.
    - **`LegalRetriever`:** Thực hiện Hybrid Search + Rerank.
    - **`AnswerGenerator`:** Tổng hợp câu trả lời từ ngữ cảnh.
    - **`Reflector`:** Kiểm tra tính chính xác của câu trả lời so với ngữ cảnh.
- **Edge (Cạnh điều hướng):** 
    - Các cạnh có điều kiện (Conditional Edges) quyết định luồng đi. Ví dụ: Nếu `Reflector` phát hiện ảo giác (Hallucination), nó sẽ đẩy trạng thái quay lại nút `LegalRetriever` với một gợi ý tìm kiếm mới thay vì trả kết quả sai cho người dùng.

### 3.2. Luồng Hỏi đáp Pháp luật (Legal QA Flow)
Đây là luồng phức tạp nhất, được thiết kế để triệt tiêu tối đa hiện tượng "ảo giác" của AI:
-   **Bước 1: Query Condensation (Cô đọng truy vấn):** Phân tích lịch sử trò chuyện để viết lại câu hỏi mới nhất thành một câu truy vấn độc lập cho RAG (ví dụ: "Nó quy định thế nào?" -> "Nghị định 125 quy định thế nào về mức phạt chậm nộp thuế?").
-   **Bước 2: Hybrid Retrieval & Reranking:** Thực hiện quy trình tìm kiếm 5 bước đã mô tả ở Mục 2 để lấy ra ngữ cảnh pháp lý chuẩn xác nhất.
-   **Bước 3: Suy luận và Trả lời:** LLM tổng hợp thông tin từ các Điều/Khoản đã trích xuất để đưa ra câu trả lời kèm trích dẫn nguồn cụ thể.
-   **Bước 4: Reflection Agent (Agent Phản hồi & Kiểm duyệt):** 
    -   Một Agent độc lập sẽ đối soát câu trả lời vừa sinh ra với văn bản luật gốc. 
    -   Nếu phát hiện AI "tự sáng tác" số hiệu văn bản hoặc nội dung không có trong ngữ cảnh, Agent này sẽ yêu cầu hệ thống thực hiện lại hoặc đưa ra cảnh báo từ chối trả lời để đảm bảo an toàn pháp lý.

### 3.2. Luồng Phân tích Xung đột (Conflict Analyzer)
Dành cho việc đối soát các quy định nội bộ hoặc hợp đồng với hệ thống pháp luật nhà nước:
-   **Extraction:** LLM quét văn bản người dùng cung cấp (Upload) để trích xuất các khẳng định, nghĩa vụ hoặc quy định cốt lõi.
-   **Legal Lookup:** Tự động tìm kiếm các quy định pháp luật nhà nước có liên quan đến các khẳng định vừa trích xuất.
-   **Contradiction Detection:** So sánh đa chiều để chỉ ra các điểm: **Tương thích**, **Rủi ro** hoặc **Trái luật**. Luồng này tuân thủ nguyên tắc ưu tiên văn bản có hiệu lực pháp lý cao hơn.

### 3.3. Luồng Tra cứu Chuyên sâu (Sector Search)
Tối ưu hóa cho việc quét diện rộng theo lĩnh vực để tạo lập các báo cáo danh mục văn bản:

- **Quy trình Xử lý (Logic Flow):**
    - **Bước 1: Sector Identification:** LLM nhận diện các ngành liên quan trong câu hỏi (ví dụ: Thuế, Đất đai, Ngân hàng).
    - **Bước 2: MetaData Filtering & Contextual Retrieval:** Áp dụng bộ lọc `legal_sectors` lên không gian Search. Sau đó, hệ thống thực hiện một lượt **Hybrid Search** chuyên sâu dựa trên ngữ nghĩa của câu hỏi trong phạm vi ngành đã lọc. Điều này đảm bảo kết quả trả về không chỉ đúng ngành mà còn khớp chính xác với nội dung chi tiết mà người dùng đang quan tâm.
    - **Bước 3: De-duplication & Sorting:** Hệ thống gom nhóm các chunk thuộc cùng một văn bản (Document-level grouping) và sắp xếp chúng theo thứ tự thời gian (Chronological order) từ cũ đến mới.
    - **Bước 4: LLM Synthesis:** LLM đọc danh sách văn bản và viết tóm tắt mục tiêu cho từng văn bản theo yêu cầu của người dùng.

- **Cấu trúc Ngữ cảnh LLM (Context Construction):**
    Thay vì gửi hàng trăm chunk lẻ tẻ, hệ thống gửi một biến `{docs_context}` được chuẩn hóa:
    ```text
    Danh sách văn bản thô (Tổng hợp theo năm):
    [1] Năm: 2020 | Số hiệu: 125/2020/NĐ-CP | Loại: Nghị định | Tiêu đề: ...
    [2] Năm: 2024 | Số hiệu: 01/2024/TT-BTC | Loại: Thông tư | Tiêu đề: ...
    ```
- **Giá trị mang lại:** Giúp người dùng có cái nhìn toàn cảnh về hệ thống văn bản quy phạm theo ngành dọc, hỗ trợ tốt cho công tác tra cứu hệ thống hơn là hỏi đáp đơn lẻ.

### 3.4. Luồng Trò chuyện Thông thường (General Chat Flow)
Luồng này được kích hoạt khi người dùng có các câu hỏi mang tính chất chào hỏi, giải thích thuật ngữ chung hoặc không yêu cầu trích dẫn pháp lý cụ thể:
-   **Trải nghiệm người dùng:** Tương tác mượt mà, thân thiện, không áp dụng các quy trình kiểm duyệt khắt khe như luồng QA để đảm bảo tốc độ phản hồi nhanh nhất.
-   **Phạm vi:** Trả lời dựa trên kiến thức nền của LLM nhưng vẫn giữ thái độ trung lập và khách quan.

### 3.5. Cơ chế Giao diện & Trải nghiệm (UI/UX Orchestration)
Giao diện người dùng (Next.js) cung cấp một thanh chọn chế độ (`ModeSelector`) trực quan. Việc cho phép người dùng chủ động chọn "Ý định" giúp:
-   **Tăng tính minh bạch:** Người dùng biết rõ AI đang xử lý theo logic nào.
-   **Tối ưu tài nguyên:** Tránh việc chạy các Agent nặng (như Conflict Analyzer) cho các câu hỏi đơn giản.
-   **Độ tin cậy cao:** Loại bỏ sai số từ bước nhận diện ý định tự động của LLM.

### 3.6. Tầm nhìn: "LangGraph-first" Orchestration
Việc áp dụng **LangGraph** thay thế cho bộ điều hướng thủ công hiện tại mang lại những lợi ích vượt trội:

| Đặc điểm | Hiện tại (Custom Scripts) | LangGraph (Target) |
| :--- | :--- | :--- |
| **Quản lý Trạng thái** | Dùng biến tạm trong hàm, khó theo dõi khi luồng dài. | **State persistent.** Trạng thái được lưu lại, cho phép "quay ngược thời gian" (Time-travel) để debug. |
| **Cơ chế Phục hồi** | Thử lại (Retry) bằng vòng lặp `while`, dễ gây treo. | **Cyclic Graphs.** Tự động quay lại các node trước đó (Loop back) nếu điều kiện chưa đạt. |
| **Tính Module hóa** | Các Flow (`qa`, `sector`) bị tách rời hoàn toàn. | **Sub-graphs.** Các flow lớn có thể chứa các flow nhỏ, dễ dàng tái sử dụng logic. |
| **Xử lý Song song** | Phải dùng `asyncio` thủ công cho từng phần. | **Parallel execution.** Tự động chạy song song các node không phụ thuộc nhau. |
| **Giám sát (Tracing)** | Phải viết log thủ công vào terminal. | **LangSmith Integration.** Theo dõi thời gian thực từng bước nhảy trên đồ thị, chi phí token và độ trễ. |

Cơ chế này giúp hệ thống Legal-RAG không chỉ là một bộ máy tìm kiếm (Searching) mà thực sự là một trợ lý ảo có khả năng **Tư duy (Reasoning)** và **Tự sửa lỗi (Self-improving)**.
