Dưới đây là tóm tắt những điểm chính về phương pháp (method) của mô hình LightRAG được trình bày trong bài báo[cite: 1]:

**1. Lập chỉ mục văn bản dựa trên cấu trúc đồ thị (Graph-based Text Indexing)**
*   Hệ thống phân chia tài liệu thành các phần nhỏ gọn hơn và sử dụng Mô hình ngôn ngữ lớn (LLM) để trích xuất các thực thể (như tên, ngày tháng, địa điểm, sự kiện) cùng các mối quan hệ giữa chúng nhằm xây dựng một đồ thị tri thức toàn diện[cite: 1].
*   Mô hình sử dụng chức năng lập hồ sơ (profiling) bằng LLM để tạo ra một cặp khóa - giá trị (key-value) bằng văn bản cho mỗi nút thực thể và cạnh quan hệ trên đồ thị[cite: 1]. 
*   Khóa (key) là các từ hoặc cụm từ ngắn giúp truy xuất hiệu quả, trong khi giá trị (value) là một đoạn văn bản tóm tắt các đoạn trích có liên quan từ dữ liệu gốc[cite: 1].
*   LightRAG sử dụng chức năng loại bỏ trùng lặp để xác định và hợp nhất các thực thể cũng như quan hệ giống nhau từ các phân đoạn văn bản khác nhau, giúp tối ưu hóa kích thước đồ thị và quá trình xử lý[cite: 1].
*   Phương pháp này có thuật toán cập nhật gia tăng, cho phép tích hợp dữ liệu mới vào đồ thị tri thức hiện tại một cách liền mạch mà không cần phải xây dựng lại toàn bộ chỉ mục đồ thị, giúp giảm thiểu đáng kể chi phí tính toán[cite: 1].

**2. Mô hình truy xuất hai cấp độ (Dual-level Retrieval Paradigm)**
*   LightRAG trích xuất các từ khóa truy vấn ở cả mức độ chi tiết (thực thể cụ thể) và mức độ trừu tượng (khái niệm bao quát) để xử lý đa dạng các loại câu hỏi của người dùng[cite: 1].
*   **Truy xuất cấp độ thấp (Low-Level Retrieval):** Phục vụ cho các truy vấn chi tiết nhằm trích xuất thông tin chính xác gắn liền với các nút thực thể hoặc cạnh cụ thể trong đồ thị[cite: 1].
*   **Truy xuất cấp độ cao (High-Level Retrieval):** Phục vụ cho các truy vấn có chủ đề rộng hơn bằng cách tổng hợp thông tin để nắm bắt các khái niệm tổng quan thay vì đi sâu vào chi tiết[cite: 1].
*   Quá trình truy xuất được thực hiện thông qua việc kết hợp cơ sở dữ liệu vector để khớp từ khóa với các cấu trúc đồ thị, sau đó thu thập thêm các nút lân cận trong đồ thị con (thuộc tính liên quan bậc cao) để làm phong phú thêm ngữ cảnh truy xuất[cite: 1].

**3. Tạo câu trả lời tăng cường truy xuất (Retrieval-Augmented Answer Generation)**
*   Thông tin được truy xuất sẽ bao gồm các dữ liệu đã được ghép nối lại với nhau như: tên, mô tả của các thực thể và quan hệ, cùng với các đoạn trích từ văn bản gốc[cite: 1].
*   Hệ thống sử dụng một LLM đa năng để kết hợp truy vấn của người dùng với các đoạn văn bản đa nguồn này nhằm tạo ra câu trả lời chính xác, giàu thông tin và phù hợp với ngữ cảnh[cite: 1].