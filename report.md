# Báo cáo Tiến độ Dự án Legal-RAG

## 1. Tiền xử lý và phân rã (Chunking)
* **Đầu vào:** Lấy mẫu 500 văn bản pháp luật tiêu biểu, phủ đều 10 chủ đề/lĩnh vực khác nhau.
* **Phương pháp:** Áp dụng kỹ thuật Structure-Aware Chunking bằng Regex để bóc tách văn bản theo cấu trúc phân cấp pháp lý (Căn cứ, Chương, Điều, Khoản, Điểm, Phụ lục). Áp dụng thêm Fallback Chunking cắt theo độ dài cho các phần quá giới hạn.
* **Đầu ra:** Các đơn vị văn bản nhỏ (chunks) chứa trọn vẹn ngữ nghĩa của một quy định. Mỗi chunk đi kèm bộ metadata cực kỳ chi tiết (`document_id`, `legal_sectors`, `chapter_ref`, `article_ref`, `clause_ref`, `point_ref`) để hỗ trợ truy xuất chính xác sau này.

## 2. Biểu diễn dữ liệu (Embedding)
* **Đầu vào:** Các chunks văn bản đã được phân rã từ bước 1.
* **Phương pháp:** Đưa qua mô hình ngôn ngữ `BAAI/bge-m3` để trích xuất đặc trưng.
* **Đầu ra:** Hai loại vector song song: Dense Vectors (1024 chiều để biểu diễn ngữ nghĩa tổng quát) và Sparse Vectors (trọng số từ vựng để bắt chính xác các keyword pháp lý).

## 3. Lập chỉ mục (Indexing)
* **Đầu vào:** Dense Vectors, Sparse Vectors và Metadata từ bước 2.
* **Phương pháp:**
  * **Small-to-Big Retrieval:** Nhúng (embed) các đoạn text nhỏ để tìm kiếm vector chính xác nhất, nhưng lưu trữ kèm đoạn văn bản cha lớn hơn để cung cấp ngữ cảnh đầy đủ cho LLM đọc.
  * **Payload Indexing:** Trích xuất và lập chỉ mục các trường siêu dữ liệu (Metadata) trên Qdrant để tối ưu tốc độ lọc (Pre-filtering).
  * **Hybrid Indexing:** Kết hợp chỉ mục Vector (ngữ nghĩa) và Từ vựng (keyword).
  * **Quantization:** Lượng tử hóa vector (Scalar Quantization int8) giúp nén dung lượng, tiết kiệm RAM.
* **Đầu ra:** Cơ sở dữ liệu Vector (Qdrant) được tối ưu hóa dung lượng, có khả năng search chéo (Hybrid + RRF) và lọc siêu dữ liệu tốc độ cao.

## 4. Tích hợp LLM & Sinh câu trả lời (RAG Engine)
* **Đầu vào:** Câu hỏi gốc của người dùng + Lịch sử hội thoại (Memory) + Top-K Context (Ngữ cảnh) truy xuất từ Qdrant.
* **Phương pháp:**
  * Viết lại câu hỏi (Query Rewrite) dựa trên lịch sử chat để biến thành một truy vấn độc lập, rõ nghĩa.
  * Đưa ngữ cảnh rộng (nguyên đoạn văn bản gốc) vào Prompt để LLM đọc và tổng hợp thông tin. Hệ thống hiện tại đang sử dụng tạm mô hình **Llama 3.3** (không dùng tính năng search web - Groq API) để xử lý logic này.
* **Đầu ra:** Câu trả lời trực tiếp, logic, phân tách rõ Căn cứ pháp lý (trích dẫn điều, khoản) hoặc trả về danh sách tài liệu tổng hợp tùy theo kịch bản (Hỏi đáp QA hoặc Tìm kiếm theo lĩnh vực - Sector Search).


##  Kế hoạch tiếp theo

**1. Chuẩn hóa & Mở rộng Hạ tầng**
* **OOP & Docker:** Tái cấu trúc code sang Hướng đối tượng và đóng gói toàn bộ hệ thống (DB, API, Web) bằng Docker Compose.
* **Scale-up:** Nâng cấp tải 500.000 văn bản và kiểm thử (benchmark) tốc độ truy vấn của Qdrant.

**2. Nâng cấp Lõi RAG**
* **Tối ưu Pipeline:** Dịch chuyển sang **Ollama Cloud**, tích hợp thêm **Rerank** (chấm điểm lại context) và **Reflection** (tự kiểm duyệt để chống ảo giác).
* **Indexing nâng cao:** Triển khai **HyDE** (query bằng câu trả lời giả định), **Summary Indexing** (lập chỉ mục tóm tắt) và **Graph Indexing** (đồ thị trích dẫn chéo).

**3. Phát triển Ứng dụng**
* **Phát hiện xung đột (Conflict Detection):** Tính năng tự động phân tích và đối chiếu tài liệu upload (nội quy, hợp đồng...) với quy định pháp luật để tìm ra các điểm trái luật hoặc đã hết hiệu lực.

