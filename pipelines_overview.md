# Hệ thống Chuyên gia Pháp lý (Legal-RAG) - Phân tích Pipeline 3 Chế độ

Hệ thống của bạn được thiết kế dựa trên một **LangGraph Workflow** chung, với 5 Node bắt buộc (`Understand` ➡️ `Retrieve` ➡️ `Grade` ➡️ `Generate` ➡️ `Reflect`). 

Tuy nhiên, tùy thuộc vào chế độ được kích hoạt, các hành vi bên trong 5 Node này sẽ được cấu hình hoàn toàn khác nhau thông qua các **Strategy Classes**. Dưới đây là bức tranh toàn cảnh về cách mỗi Strategy hoạt động.

---

## 1. Chế độ Tư vấn Pháp lý (Legal QA Mode)
*Được tối ưu hóa để trả lời chính xác, trực tiếp vào các điều khoản phạt, quy định cụ thể của pháp luật.*

1. **`Understand` (Query Rewriter & Filter Extractor)**
   * Sinh ra **Văn bản giả định (HyDE)** bằng mô hình LLM để nhại lại văn phong luật pháp dựa trên câu hỏi của user (giúp Vector Search bắt từ chuẩn hơn).
   * Rút trích chặt chẽ các trường siêu dữ liệu (Metadata Filters) như: `doc_number` (Số hiệu), `legal_type` (Loại văn bản), `is_appendix` (Có phải phụ lục không). Bóc tách JSON an toàn bằng Regex (chống trả về Markdown rác từ LLM).
   * **Cơ chế Router An Toàn:** Nếu quá trình nhận diện Intent bị lỗi, hệ thống có cơ chế Fallback tự động ép về `RouteIntent.LEGAL_QA` để đảm bảo RAG luôn chạy thay vì văng lỗi.
   * **Cơ chế Retry:** Nếu vòng RAG đầu tiên thất bại, khi vòng lặp quay lại Node này, nó sẽ viết lại câu hỏi theo một *góc độ hoàn toàn khác* để ép Qdrant trả về không gian Vector mới.
2. **`Retrieve` (Hybrid + Dual Graph Expansion)**
   * Dùng Keyword + Vector tìm kiếm trên **Qdrant** qua thuật toán Hash BM25 nội bộ (giúp tránh lỗi API) kết hợp Dense Embedding, ID trên Qdrant được map chéo với Neo4j thông qua trường `chunk_id` độc lập trong Payload thay vì UUID gốc.
   * Yêu cầu **Neo4j** thực hiện thủ công các Cypher khép kín (Bypass Langchain wrapper lỗi):
     * *Bottom-Up Expansion:* Lấy TOC (Mục lục cha) và các văn bản "anh em" lân cận của chunk tìm được để LLM hiểu ngữ cảnh bao quát.
     * *Lateral Expansion:* Trích xuất các văn bản và thủ tục thuộc tài liệu cùng ngành (nhánh ngang).
3. **`Grade` (Truncation Grader)**
   * Thu nhỏ độ dài của mỗi chunk xuống còn 300 ký tự (Truncation) để tiết kiệm Token.
   * Gửi cho LLM một luồng batch nhẹ để chấm điểm `Có/Không` xem ngữ cảnh này có thực sự trả lời được câu hỏi không.
   * Nếu không, nó bật chế độ **Best-Effort** (Cố gắng hết sức / Trả lời tham khảo).
4. **`Generate` (Answer Synthesizer)**
   * Sinh câu trả lời dựa trên LLM. Nếu ở chế độ Best-Effort, nó tự động chèn **Cảnh báo (Disclaimer)** cho người dùng.
   * Gắn danh sách "📚 Tài liệu tham khảo thêm" từ chuỗi Lateral Expansion.
5. **`Reflect` (Hallucination Checker)**
   * LLM thứ hai được gọi lên để đọc lại câu trả lời vừa sinh.
   * Rà soát đối chiếu chéo (Cross-check) với context xem AI có vừa tự bịa ra điều luật (Ảo giác) không.
   * Nếu có ảo giác, kích hoạt **Correction Prompt** bắt AI tự sửa lại câu trả lời và đóng dấu chất lượng.

---

## 2. Chế độ Tra cứu Lĩnh vực (Sector Search Mode)
*Được tối ưu hóa cho dân nghiên cứu/Kế toán trưởng cần gom toàn bộ văn bản trong 1 lĩnh vực (VD: Đầu tư, Đấu thầu) và tổng hợp thành Ma trận Báo cáo.*

1. **`Understand` (Sector Query Planner)**
   * Nhận diện ngành luật (`legal_sectors`), dải thời gian hiệu lực (`date_range`) và các từ khóa cốt lõi.
   * Đọc phân tích file tải lên (nếu có) để nhận diện vùng tập trung và gộp từ khóa.
2. **`Retrieve` (Hierarchical Drill-Down)**
   * Tìm kiếm thông thái qua Neo4j theo 3 bậc (Thay vì đâm thẳng vào Vector Search như QA):
     * **Bậc 1:** Khớp Ngành (Nắm các Luật thuộc Ngành đó).
     * **Bậc 2:** Khớp Mục lục (Đưa Mục lục cho LLM xem xét để LLM chỉ điểm đâu là Điều/Khoản cần đào sâu).
     * **Bậc 3:** Vector Search Nhắm mục tiêu (Targeted Search) chọc thẳng vào các tọa độ Điều/Khoản mà LLM vừa chỉ.
   * MapReduce Graph: Query Neo4j lấy thống kê chung toàn ngành.
3. **`Grade` (Pipeline Lọc đa tầng)**
   * Không dùng LLM tốn kém ngay từ đầu, mà dùng Python thuần:
     * Khử trùng lặp (`Dedup`).
     * Lọc bằng thuật toán thời gian (`Heuristic Date Filter`).
   * Gửi batch đã được thu gọn cho LLM chấm điểm mức độ liên quan.
   * Biến các văn bản đậu (Passed Docs) thành định dạng **Bảng Markdown** (Table Matrix).
4. **`Generate` (Executive Summarizer)**
   * Dùng 1 lượt LLM cực ngọn (~100 tokens) để viết "Tóm tắt Điều hành" (Executive Summary).
   * Lắp ráp Tóm tắt này với Bảng Markdown từ Node Grade và các insight từ File đính kèm để thành 1 bản Report hoàn chỉnh.
5. **`Reflect` (Coverage Bias Checker)**
   * Kỹ thuật cực hay: Kiểm tra độ chênh lệch (Coverage Bias).
   * *Ví dụ:* Nếu danh sách trả về 100% là "Luật", nó sẽ hiểu rằng người dùng sẽ thiếu các văn bản hướng dẫn chi tiết. Nó lập tức lấy các Luật đó chọc vào Neo4j (qua quan hệ `has_basis`) để moi ra các Nghị định/Thông tư.
   * Nối kết quả bổ sung vòng Bảng Markdown và báo cáo cho người dùng.

---

## 3. Chế độ Rà soát Xung đột (Conflict Analyzer Mode)
*Được thiết kế để nạp Nội quy Công ty vào và xem nó có chửi lộn với Luật ban hành của Nhà nước hay không, và xem xét update CSDL.*

1. **`Understand` (Deontic Extractor & Loop Queuer)**
   * Rerank nội dung File bằng `Internal Reranker API`.
   * LLM phân rã văn bản Nội quy thành các hạt nguyên tử (Mệnh đề - Claims) với cấu trúc Logics: `[Chủ Thể] + [Hành Vi] + [Điều Kiện] + [Hệ Quả]`.
   * Đẩy các hạt này vào một hàng đợi (Queue) để tạo vòng lặp (Looping Batch).
2. **`Retrieve` (HyDE + Time-Travel Search)**
   * Dùng HyDE để biến "Nội quy" thành "Văn phong Nhà Nước".
   * Tìm Vector (Broad Search) nhưng **cố tình lấy cả văn bản Hết hiệu lực** (`include_inactive=True`).
   * Gọi cơ chế **Time-Travel** trên Neo4j: "Văn bản lỗi thời này vừa bị Nghị định nào thay thế (AMENDS/REPLACES)?". 
3. **`Grade` (API Reranker Pruner)**
   * Vứt 15 kết quả vào **Internal Reranker API**.
   * Cắt tỉa (Prune) cực kỳ sắt máu, chỉ giữ lại đúng 3 văn bản có điểm số tiệm cận nhất với Mệnh đề đang xét để tiết kiệm Token.
4. **`Generate` (Judge Agent)**
   * LLM đóng vai trò Thẩm phán Sơ thẩm.
   * Gắn ngữ cảnh Luật Khung (Lex Superior) + Luật Cấp sau (Lex Posterior) vào Prompt.
   * Thẩm phán dán nhãn: `Contradiction` (Trái luật), `Entailment` (Tuyệt vời), `Neutral` (Vùng xám / Tự do).
   * Sinh JSON đề xuất cập nhật cơ sở dữ liệu (`proposed_db_update`) nếu phát hiện luật mới đã lật đổ luật cũ.
5. **`Reflect` (Reviewer & Assembly Controller)**
   * LLM thứ hai (Hội đồng Giám đốc) kiểm tra lại xem Thẩm phán (Judge) có bị ngáo đá (Hallucinate) không. Lưu ý hiện trạng hệ thống tối ưu API Call bằng cách thiết lập cờ bypass nếu LLM quá tải.
   * Là một **Loop Controller**: Xét xem Hàng Đợi (Queue) còn mệnh đề nội quy nào chưa xử lý không? Nếu còn cờ `pass_flag=False`, nó kích hoạt tự động node trung gian `node_reset_for_batch` (để clear triệt để cờ `retry_count` và vòng lặp vô hạn) rồi ép hệ thống LangGraph quay đầu lại Node `Understand` để chạy mẻ tiếp theo.
   * State độc lập: Mọi xử lý mảng (array) đều dùng `pruned_batch` và `conflict_draft` riêng biệt cho Conflict Mode để không ghi đè chéo (Collision) sang biến `filtered_context` của chế độ QA thông thường.
   * Khi Queue đã cạn, lắp ráp toàn bộ thành 1 bảng báo cáo kiểm toán rực rỡ và đính kèm bộ mã HTML `<!-- DB_UPDATE_PROPOSAL -->` để UI kích hoạt tính năng Replace DB.
