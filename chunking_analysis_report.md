# Báo cáo Phân tích Chất lượng Chunking

## Van ban: 49/2025/TT-BYT

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **49/2025/TT-BYT** dựa trên dữ liệu Dump từ Qdrant và Neo4j:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:**
**Kém**
Dữ liệu hiện tại có cấu trúc chunking rất rời rạc, thiếu tính logic và chứa nhiều lỗi kỹ thuật nghiêm trọng. Việc tách đoạn (chunking) không tuân thủ đúng cấu trúc văn bản pháp luật, dẫn đến nguy cơ cao hệ thống RAG sẽ trả về kết quả sai lệch hoặc thiếu thông tin khi người dùng truy vấn.

#### **2. Các lỗi cụ thể tìm thấy:**

*   **Lỗi định danh (Article Ref) - Chunk 3:**
    *   **Chunk ID:** `9cadcd7c-ef27-5240-9c07-d3b06f662788`
    *   **Mô tả:** Chunk chứa phần "Header" quan trọng (Tên văn bản, Số, Ngày ban hành, Quốc hiệu) nhưng trường `Article Ref` bị để **trống**.
    *   **Hậu quả:** Khi người dùng hỏi về thông tin chung của văn bản (ví dụ: "Thông tư 49 ban hành ngày nào?"), hệ thống có thể không tìm thấy chunk này hoặc trả về kết quả không chính xác do thiếu metadata định danh.

*   **Lỗi gộp/Split sai (Swallowing & Fragmentation) - Chunk 2 & Chunk 4:**
    *   **Chunk ID:** `481e1292-688b-521e-b6ab-1a25b7179b52` (Chunk 2) và `af7a4b9e-1a33-5a93-b044-dac383024426` (Chunk 4).
    *   **Mô tả:**
        *   **Chunk 2** được gán nhãn `Điều 1 > Khoản 1, 2, 3, 4, 5` nhưng nội dung chỉ mới bắt đầu giới thiệu tiêu đề Điều 1 và chưa đi vào chi tiết các khoản.
        *   **Chunk 4** lại chứa nội dung cụ thể của **Khoản 5** (bắt đầu bằng `[5. tiếp theo]`) nhưng lại bị tách rời khỏi các khoản 1-4.
        *   **Mâu thuẫn:** Chunk 2 tuyên bố chứa đủ 5 khoản nhưng nội dung bị cắt cụt, trong khi Chunk 4 lại chứa nội dung của Khoản 5 nhưng bị tách biệt. Điều này cho thấy thuật toán chunking đã cắt ngang dòng chảy nội dung của Điều 1 một cách tùy tiện.
    *   **Hậu quả:** Người dùng hỏi về "Khoản 1, 2, 3, 4" của Điều 1 có thể nhận được Chunk 2 (chỉ có tiêu đề) mà không có nội dung chi tiết, hoặc nhận được Chunk 4 (chỉ có Khoản 5) gây hiểu nhầm.

*   **Lỗi rác kỹ thuật (Technical Noise):**
    *   **Chunk ID:** `af7a4b9e-1a33-5a93-b044-dac383024426` (Chunk 4).
    *   **Mô tả:** Nội dung chứa ký tự lạ và nhãn kỹ thuật không thuộc văn bản gốc: **`[5. tiếp theo]`**.
    *   **Hậu quả:** Đây là dấu hiệu của việc xử lý text thô (raw text) chưa sạch hoặc lỗi trong quá trình gộp lại các đoạn văn bản. Ký tự này sẽ làm nhiễu mô hình ngôn ngữ (LLM) khi sinh câu trả lời.

*   **Lỗi nuốt tin (Potential Missing Content):**
    *   **Phân tích:** Theo Neo4j, Điều 1 có các Khoản 1, 2, 3, 4, 5.
    *   **Thực tế Qdrant:**
        *   Chunk 2: Chỉ có tiêu đề Điều 1.
        *   Chunk 4: Chỉ có nội dung Khoản 5.
    *   **Kết luận:** Nội dung chi tiết của **Khoản 1, 2, 3, 4** của Điều 1 dường như **bị mất** hoặc bị gộp vào một chunk khác không được hiển thị trong danh sách 4 chunks này (hoặc bị cắt mất hoàn toàn). Đây là lỗi nghiêm trọng nhất vì nội dung sửa đổi cụ thể (thường nằm ở các khoản này) không có mặt trong vector DB.

#### **3. Đề xuất khắc phục:**

1.  **Sửa lại chiến lược Chunking (Re-chunking):**
    *   Áp dụng chiến lược **Semantic Chunking** hoặc **Structure-based Chunking** dựa trên thẻ HTML/XML của văn bản pháp luật (thẻ `<article>`, `<clause>`).
    *   Đảm bảo mỗi chunk tương ứng với một **Điều** hoặc một **Khoản** hoàn chỉnh, không được cắt ngang giữa các khoản.
    *   Loại bỏ việc gán nhãn "Khoản 1-5" cho một chunk nếu chunk đó không chứa đầy đủ nội dung của 5 khoản đó.

2.  **Làm sạch dữ liệu (Data Cleaning):**
    *   Loại bỏ các ký tự kỹ thuật như `[5. tiếp theo]`, các dấu ngoặc vuông không cần thiết, và các đoạn header/footer bị chèn vào giữa nội dung.
    *   Đảm bảo phần Header (Chunk 3) được gán một `Article Ref` đặc biệt (ví dụ: `Header` hoặc `General Info`) thay vì để trống, để hệ thống vẫn có thể truy xuất được thông tin chung.

3.  **Kiểm tra tính toàn vẹn (Integrity Check):**
    *   Chạy script so sánh giữa danh sách các Khoản (Clauses) trong Neo4j và nội dung thực tế trong Qdrant.
    *   Nếu phát hiện Khoản 1, 2, 3, 4 của Điều 1 bị thiếu nội dung, cần tái xử lý (re-process) toàn bộ văn bản từ nguồn gốc (PDF/HTML) để đảm bảo không bị mất dữ liệu.

4.  **Tối ưu Metadata:**
    *   Đảm bảo trường `Article Ref` luôn chính xác và khớp với nội dung thực tế trong `Content`. Nếu một chunk chứa nhiều Điều, cần tách ra thành nhiều chunk riêng biệt.

---

## Van ban: 53/2025/TT-BYT

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **53/2025/TT-BYT** dựa trên dữ liệu Dump từ Qdrant và Neo4j:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:**
**Kém**
Dữ liệu hiện tại có cấu trúc chunking rời rạc, thiếu tính liên kết ngữ cảnh và chứa lỗi định danh nghiêm trọng. Việc tách chunk không tuân thủ đúng logic pháp lý (cắt ngang giữa các khoản/diểm) và Chunk chứa phần đầu văn bản (Header) bị mất định danh Điều khoản, gây khó khăn cho việc truy xuất chính xác.

#### **2. Các lỗi cụ thể tìm thấy:**

*   **Lỗi định danh (Article Ref) - Nghiêm trọng:**
    *   **Chunk 4 (ID: `ddafcd19...`):** Chứa phần đầu văn bản (Quốc hiệu, Số, Ngày ban hành, Tên Thông tư) nhưng trường `Article Ref` bị **để trống**. Đây là lỗi phổ biến khiến hệ thống không thể trả lời các câu hỏi về thông tin chung của văn bản (ví dụ: "Thông tư này ban hành ngày nào?").
    *   **Chunk 3 (ID: `b0b94f2e...`):** Nội dung bắt đầu bằng `[2. tiếp theo]` và liệt kê các điểm `c, d, đ, e`. Tuy nhiên, `Article Ref` chỉ ghi chung chung là `Điều 1`. Trong khi đó, Chunk 1 cũng ghi `Điều 1` nhưng chứa Khoản 1, 2. Việc thiếu chỉ dẫn rõ ràng về việc đây là phần tiếp theo của Khoản 2 (đã bị cắt ở Chunk 1) gây mất ngữ cảnh.

*   **Lỗi gộp (Swallowing) / Cắt ngang ngữ cảnh:**
    *   **Giữa Chunk 1 và Chunk 3:**
        *   Chunk 1 kết thúc ở điểm `b) Dân số, trẻ em...`.
        *   Chunk 3 bắt đầu ngay bằng `c) Đăng ký khám bệnh...`.
        *   **Vấn đề:** Chunk 1 ghi `Điều khoản: Điều 1 > Khoản 1, 2`. Chunk 3 ghi `Điều khoản: Điều 1 > Khoản 2. > Điểm c, d, đ, e)`.
        *   **Phân tích:** Chunk 1 đã bị cắt cụt ngay giữa danh sách các điểm (a, b) của Khoản 2. Chunk 3 là phần tiếp theo nhưng lại được gán nhãn là một chunk độc lập với `Chunk Index: 3` (trong khi Chunk 1 là `Index: 2`). Điều này cho thấy thuật toán chunking đã cắt ngang dòng văn bản pháp lý quan trọng, khiến người dùng không thấy được danh sách đầy đủ các chức năng của Trạm Y tế nếu chỉ truy xuất một chunk.

*   **Lỗi rác kỹ thuật / Định dạng:**
    *   **Chunk 3:** Chứa ký tự `[2. tiếp theo]`. Đây là dấu vết của quá trình xử lý văn bản (text processing) chưa được làm sạch hoàn toàn trước khi đưa vào Vector DB. Trong văn bản pháp luật gốc không có ký tự này, nó gây nhiễu cho mô hình ngôn ngữ.
    *   **Chunk 4:** Chứa các ký tự đặc biệt của Header (`|`, `...`) và dòng `./....` ở cuối Chunk 2 (dấu kết thúc văn bản) bị chèn vào nội dung chính, có thể gây hiểu nhầm về nội dung điều khoản.

*   **Lỗi nuốt tin (Kiểm tra chéo Neo4j - Qdrant):**
    *   **Neo4j (Mục lục):** Liệt kê `Điều 1` và `Điều 2`.
    *   **Qdrant:** Có chunk cho `Điều 1` (chia làm 2 phần) và `Điều 2`.
    *   **Kết luận:** Không có Điều khoản nào bị "nuốt" hoàn toàn (mất tích). Tuy nhiên, nội dung của `Điều 1` bị phân mảnh quá mức (chia thành 2 chunk rời rạc mà không có cơ chế liên kết `next_chunk_id` rõ ràng trong metadata).

#### **3. Đề xuất khắc phục:**

1.  **Sửa lỗi định danh Header:**
    *   Cập nhật lại **Chunk 4**: Gán `Article Ref` là `Lời mở đầu` hoặc `Thông tin chung` (hoặc để rỗng nhưng thêm tag `type: header` trong metadata) để hệ thống biết đây là phần thông tin văn bản, không phải điều khoản luật.

2.  **Tối ưu hóa chiến lược Chunking (Sliding Window hoặc Semantic Chunking):**
    *   **Không cắt ngang danh sách:** Cần điều chỉnh thuật toán để đảm bảo các danh sách (a, b, c, d...) nằm trọn vẹn trong một chunk hoặc sử dụng kỹ thuật *overlap* (chồng lấn) đủ lớn để Chunk 3 chứa lại phần đầu của danh sách (a, b) để đảm bảo ngữ cảnh đầy đủ.
    *   **Gộp Chunk 1 và Chunk 3:** Nếu độ dài cho phép, nên gộp Chunk 1 và Chunk 3 thành một chunk duy nhất cho `Khoản 2 Điều 1` để tránh việc người dùng phải truy xuất 2 lần mới có đủ thông tin.

3.  **Làm sạch dữ liệu (Data Cleaning):**
    *   Loại bỏ các ký tự giả tạo như `[2. tiếp theo]` trong nội dung Chunk 3.
    *   Chuẩn hóa lại phần kết thúc văn bản (loại bỏ `./....` nếu không cần thiết cho ngữ cảnh).

4.  **Cải thiện Metadata:**
    *   Thêm trường `parent_clause` hoặc `sequence_order` vào metadata của Chunk 3 để chỉ rõ nó là phần tiếp theo của Chunk 1 (ví dụ: `Chunk 1` có `next_chunk_id` trỏ đến `Chunk 3`).
    *   Đảm bảo `Chunk Index` phản ánh đúng thứ tự logic trong văn bản (hiện tại Chunk 4 là Index 1, Chunk 1 là Index 2, Chunk 3 là Index 3, Chunk 2 là Index 4 - thứ tự này có vẻ đảo lộn so với logic văn bản: Header -> Điều 1 -> Điều 2). Cần sắp xếp lại `Chunk Index` theo thứ tự xuất hiện thực tế trong văn bản.

---

## Van ban: 3398/QĐ-UBND

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **3398/QĐ-UBND** dựa trên dữ liệu Dump bạn cung cấp:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:** **Kém**
Dữ liệu hiện tại có cấu trúc chunking rời rạc, thiếu tính logic về thứ tự văn bản và bị mất mát thông tin quan trọng (phần căn cứ pháp lý). Việc gán nhãn `Article Ref` chưa chính xác cho các phần mở đầu, gây khó khăn cho việc truy xuất ngữ cảnh (context) khi người dùng hỏi về cơ sở ban hành.

#### **2. Các lỗi cụ thể tìm thấy:**

*   **Lỗi định danh (Article Ref) - Chunk 1 & Chunk 5:**
    *   **Chunk 1 (ID: 306c...):** Chứa phần đầu văn bản (Số, Ngày, Tên Quyết định, Căn cứ...) nhưng trường `Article Ref` bị **trống**. Đây là lỗi nghiêm trọng vì phần này chứa thông tin định danh quan trọng nhất của văn bản.
    *   **Chunk 5 (ID: e790...):** Chứa nội dung "Căn cứ Quyết định số 2302/QĐ-UBND..." (phần căn cứ pháp lý chi tiết) nhưng `Article Ref` cũng bị **trống**.
    *   *Hệ quả:* Khi người dùng hỏi "Quyết định này ban hành dựa trên căn cứ nào?", hệ thống có thể trả về kết quả không chính xác hoặc không tìm thấy chunk liên quan do thiếu thẻ bài (tag).

*   **Lỗi nuốt tin / Mất dữ liệu (Swallowing/Missing Data):**
    *   **Mất nội dung Danh mục (Phụ lục):** Theo `document_toc` của Neo4j và nội dung `Điều 1` trong Chunk 4, văn bản này công bố một "Danh mục thủ tục hành chính" (gồm 31 thủ tục). Tuy nhiên, trong 5 chunks của Qdrant, **không có chunk nào chứa nội dung chi tiết của danh mục này** (chỉ có dòng dẫn "Công bố kèm theo...").
    *   *Hệ quả:* RAG sẽ không thể trả lời các câu hỏi cụ thể về tên thủ tục, mã số, hoặc thẩm quyền giải quyết của từng thủ tục trong danh mục.

*   **Lỗi thứ tự logic (Logical Order):**
    *   Dữ liệu Qdrant không được sắp xếp theo thứ tự văn bản:
        *   Chunk 1: Phần đầu (Header).
        *   Chunk 5: Phần Căn cứ (nên nằm ngay sau Header).
        *   Chunk 4: Điều 1.
        *   Chunk 3: Điều 2.
        *   Chunk 2: Điều 3.
    *   *Hệ quả:* Nếu hệ thống RAG sử dụng `Chunk Index` để tái tạo văn bản hoặc tìm kiếm ngữ cảnh liền kề, nó sẽ bị đảo lộn (Ví dụ: Chunk 5 nằm giữa Chunk 1 và Chunk 4 về mặt logic, nhưng ID và Index lại không phản ánh đúng trình tự này).

*   **Lỗi rác kỹ thuật / Cắt cụt:**
    *   **Chunk 2 (ID: 55f5...):** Nội dung kết thúc bằng `./....`. Dấu `....` thường là dấu hiệu của việc cắt cụt dữ liệu (truncation) hoặc lỗi OCR/parse, không phải là ký tự kết thúc chuẩn của văn bản pháp luật.
    *   **Chunk 3 (ID: 8fdc...):** Nội dung bị cắt cụt ở dòng "Ban Quản l..." (thiếu từ "lý").

#### **3. Đề xuất khắc phục:**

1.  **Sửa lại quy tắc gán nhãn (Labeling Strategy):**
    *   Đối với các chunk chứa phần "Căn cứ" (Căn cứ...) và phần "Đầu văn bản" (Số, Ngày, Tên), cần gán `Article Ref` là **"Phần mở đầu"** hoặc **"Căn cứ"** thay vì để trống.
    *   Đảm bảo `Article Ref` của Chunk 5 được gán là "Căn cứ" để dễ dàng truy xuất.

2.  **Tái xử lý Chunking (Re-chunking) cho Danh mục:**
    *   Cần kiểm tra lại quy trình trích xuất văn bản gốc. Phần "Danh mục kèm theo" (Phụ lục) đang bị bỏ sót hoàn toàn.
    *   Nếu văn bản gốc có bảng danh mục, cần tách riêng các dòng/bảng này thành các chunk riêng biệt và gán `Article Ref` là **"Phụ lục"** hoặc **"Danh mục kèm theo Điều 1"**.

3.  **Sửa lỗi cắt cụt và ký tự lạ:**
    *   Điều chỉnh tham số `chunk_size` hoặc `overlap` trong quá trình băm nhỏ để tránh cắt ngang từ (ví dụ: "Ban Quản l...").
    *   Thêm bước làm sạch (cleaning) để loại bỏ các ký tự `....` thừa thãi ở cuối chunk.

4.  **Sắp xếp lại thứ tự (Re-indexing):**
    *   Đảm bảo `Chunk Index` trong Qdrant phản ánh đúng thứ tự xuất hiện trong văn bản gốc (Header -> Căn cứ -> Điều 1 -> Điều 2 -> Điều 3 -> Phụ lục). Hiện tại Chunk 5 (Căn cứ) đang bị đẩy xuống vị trí Index 2 (sau Header) nhưng lại nằm sau Chunk 4 (Điều 1) trong danh sách dump, gây nhầm lẫn về logic.

5.  **Kiểm tra lại Neo4j:**
    *   Đảm bảo nút `document_toc` trong Neo4j được cập nhật đầy đủ thông tin về Phụ lục nếu có, để đồng bộ với dữ liệu vector.

---

## Van ban: 3493/QĐ-UBND

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **3493/QĐ-UBND** dựa trên dữ liệu Dump từ Qdrant và Neo4j:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:** **Kém**
Dữ liệu hiện tại có cấu trúc cơ bản nhưng tồn tại nhiều lỗi nghiêm trọng về định danh (metadata) và logic phân mảnh (chunking), đặc biệt là việc xử lý các Phụ lục (bảng biểu). Các chunk quan trọng chứa danh mục thủ tục hành chính bị mất định danh Điều khoản, gây khó khăn cho việc truy xuất chính xác (Retrieval) và trích dẫn nguồn (Citation).

#### **2. Các lỗi cụ thể tìm thấy:**

*   **Lỗi định danh (Article Ref) nghiêm trọng:**
    *   **Chunk 1 (ID: 0852a622...)** và **Chunk 3 (ID: 54ce81e6...)**: Chứa nội dung danh mục các quy trình (STT 1-2, 20-22...). Đây là nội dung thực tế của **Điều 1** (theo Chunk 6 và Neo4j), nhưng trường `Article Ref` bị để **trống**.
    *   **Chunk 2 (ID: 474d4310...)**: Chứa phần đầu văn bản (Căn cứ, Lời mở đầu) nhưng `Article Ref` bị **trống**. Đây thường là phần "Căn cứ" hoặc "Lời mở đầu" không thuộc Điều cụ thể, nhưng cần được gán nhãn rõ ràng (ví dụ: "Căn cứ" hoặc "Header") thay vì để trống hoàn toàn.

*   **Lỗi cấu trúc Phụ lục/Bảng biểu (Swallowing & Fragmentation):**
    *   **Chunk 1 & Chunk 3**: Bị cắt rời một bảng danh mục duy nhất (36 quy trình) thành nhiều mảnh rời rạc.
        *   Chunk 1 chứa STT 1, 2.
        *   Chunk 3 chứa STT 20, 21, 22.
        *   **Hệ quả:** Người dùng không thể lấy được danh sách đầy đủ các quy trình trong một lần truy vấn. Nếu người dùng hỏi "Có những môn thể thao nào được cấp giấy chứng nhận?", hệ thống có thể chỉ trả về một phần nhỏ (Bóng đá, Quần vợt) mà thiếu các môn khác do dữ liệu bị băm vụn.
    *   **Thiếu Chunk trung gian:** Không thấy Chunk nào chứa các STT từ 3 đến 19. Có khả năng các chunk này bị mất hoặc chưa được dump trong dữ liệu mẫu, dẫn đến **Lỗi nuốt tin** (Missing content).

*   **Lỗi gộp (Swallowing) tiềm ẩn:**
    *   **Chunk 6 (ID: c536df1b...)**: Gán nhãn `Article Ref: Điều 1`. Nội dung bắt đầu bằng "Ban hành kèm theo... gồm: 1. Quy trình...". Tuy nhiên, nội dung này dường như chỉ liệt kê tiêu đề chung và bắt đầu danh sách, trong khi danh sách chi tiết lại nằm ở Chunk 1 và 3 (không có nhãn Điều 1). Điều này tạo ra sự đứt gãy ngữ nghĩa: Chunk 6 nói "gồm", nhưng nội dung "gồm" lại nằm ở các chunk khác không liên kết metadata.

*   **Lỗi rác kỹ thuật:**
    *   **Chunk 1 & 3**: Có dòng `|||` và các ký tự bảng biểu bị lỗi định dạng (`| STT | ... | | |`). Đây là dấu hiệu của việc parse bảng biểu (table) sang text chưa tốt, gây nhiễu cho mô hình vector.
    *   **Chunk 7**: Kết thúc bằng `./....` (dấu chấm lửng và dấu gạch chéo), có thể là lỗi cắt xén cuối văn bản hoặc ký tự thừa.

*   **Mất mát dữ liệu (So sánh Qdrant vs Neo4j):**
    *   Neo4j ghi nhận **Điều 1** có nội dung "Ban hành kèm theo... 36 quy trình...".
    *   Qdrant có Chunk 6 (Điều 1) nhưng nội dung bị cắt cụt ở dòng "1. Quy trình nội bộ...".
    *   Qdrant **thiếu hẳn** các chunk chứa nội dung chi tiết của 34 quy trình còn lại (từ STT 3 đến 19 và 23 đến 36).

#### **3. Đề xuất khắc phục:**

1.  **Sửa lại Metadata (Article Ref):**
    *   Gán lại `Article Ref: Điều 1` cho **Chunk 1** và **Chunk 3** (và các chunk danh mục còn lại).
    *   Gán `Article Ref: Căn cứ` hoặc `Header` cho **Chunk 2** để phân biệt với các Điều khoản chính.

2.  **Tối ưu hóa chiến lược Chunking cho Bảng biểu:**
    *   **Không cắt ngang bảng:** Cần điều chỉnh thuật toán chunking để giữ nguyên vẹn một hàng (row) hoặc một nhóm logic trong bảng. Nếu bảng quá dài, hãy chia theo nhóm (ví dụ: 10 quy trình/chunk) nhưng đảm bảo mỗi chunk đều có tiêu đề bảng và `Article Ref` chính xác.
    *   **Làm sạch dữ liệu bảng:** Loại bỏ các ký tự thừa `|||` và chuẩn hóa định dạng bảng sang Markdown hoặc JSON trước khi đưa vào Vector DB để mô hình hiểu cấu trúc tốt hơn.

3.  **Kiểm tra lại quy trình trích xuất (Extraction Pipeline):**
    *   Kiểm tra tại sao các STT từ 3 đến 19 và 23 đến 36 lại không xuất hiện trong Qdrant. Cần chạy lại quá trình OCR/Parse để đảm bảo toàn bộ 36 quy trình được đưa vào cơ sở dữ liệu.

4.  **Liên kết ngữ nghĩa (Context Linking):**
    *   Nếu bắt buộc phải chia nhỏ bảng, hãy thêm thông tin "Tiếp theo" hoặc "Tiếp tục danh mục" vào metadata của các chunk con để mô hình LLM hiểu rằng đây là một danh sách liên tục thuộc cùng một Điều.

5.  **Xử lý ký tự kết thúc:**
    *   Làm sạch các ký tự lạ như `./....` ở cuối Chunk 7.

---

## Van ban: 2586/QĐ-UBND

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **2586/QĐ-UBND** dựa trên dữ liệu Dump từ Qdrant và Neo4j:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:**
**KÉM**
Dữ liệu hiện tại có chất lượng rất thấp, không đảm bảo yêu cầu cơ bản cho hệ thống RAG. Tồn tại lỗi nghiêm trọng về việc gán nhãn (labeling), dữ liệu bị trùng lặp vô nghĩa, và thiếu hụt hoàn toàn nội dung của một Điều khoản quan trọng trong cơ sở dữ liệu vector.

#### **2. Các lỗi cụ thể tìm thấy:**

*   **Lỗi nuốt tin (Missing Content - Critical):**
    *   **Điều 3:** Trong Neo4j (Mục lục) có ghi rõ nội dung của **Điều 3** ("Giao Sở Khoa học và Công nghệ chủ trì..."). Tuy nhiên, khi quét toàn bộ 15 chunks trong Qdrant, **không có chunk nào** chứa nội dung hoặc nhãn "Điều 3". Điều này khiến hệ thống RAG không thể trả lời các câu hỏi liên quan đến trách nhiệm của Sở KH&CN.

*   **Lỗi định danh (Article Ref Missing/Mislabeling):**
    *   **Các Chunk chứa nội dung quan trọng bị mất nhãn:** Các Chunk **1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15** đều có `Article Ref` để trống hoặc gán sai thành "Dữ liệu Bảng biểu" / "PHẦN II".
    *   **Cụ thể:**
        *   Chunk 3, 11, 12 chứa nội dung chi tiết về "Lĩnh vực Dân số, Bà mẹ - Trẻ em" (thuộc Phụ lục/Quy trình) nhưng bị gán nhãn chung chung là "PHẦN II" thay vì gắn với Mã TTHC cụ thể hoặc Điều khoản quy định.
        *   Chunk 1, 5, 6, 7, 8, 9, 13, 14, 15 chứa các bước quy trình (Bước 1, 6, 8...) nhưng bị gán nhãn "Dữ liệu Bảng biểu", làm mất ngữ cảnh pháp lý.

*   **Lỗi rác kỹ thuật & Trùng lặp (Technical Noise & Duplication):**
    *   **Trùng lặp nội dung (Redundancy):** Có tới **6 chunks** (Chunk 1, 7, 8, 9, 13, 14) chứa nội dung gần như y hệt nhau về "Bước 1" của quy trình. Điều này gây nhiễu cho mô hình vector, làm giảm độ chính xác khi tìm kiếm (recall) do các vector tương tự nhau cạnh tranh nhau.
    *   **Lỗi định dạng văn bản:** Trong Chunk 1, 7, 8, 9, 13, 14 xuất hiện đoạn văn bản lặp lại vô nghĩa: *"TCCN có nhu cầu thực hiện TTHC. TCCN có nhu cầu thực hiện TTHC. Khi có nhu cầu..."*. Đây là lỗi parsing (băm nhỏ) hoặc lỗi OCR/copy-paste từ nguồn gốc.
    *   **Cấu trúc bảng bị vỡ:** Các chunk chứa bảng biểu (Chunk 1, 5, 6...) bị cắt ngang dòng, khiến nội dung trong các cột bị méo mó, khó đọc và mất ngữ cảnh (ví dụ: Chunk 5 bị cắt ở dòng "* Trường hợp hồ sơ quá hạn...").

*   **Lỗi gộp (Swallowing) - Nghi ngờ:**
    *   Chunk 2 chứa nội dung của **Điều 4** nhưng ngay sau đó lại xuất hiện dòng *"QUY TRÌNH NỘI BỘ TIẾPNHẬN VÀ GIẢI QUYẾT..."*. Có khả năng chunk này đã gộp phần cuối của Điều 4 và phần đầu của Phụ lục mà không tách biệt rõ ràng, gây nhiễu ngữ cảnh.

#### **3. Đề xuất khắc phục:**

1.  **Tái xử lý (Re-chunking) toàn bộ văn bản:**
    *   Cần chạy lại quy trình trích xuất và băm nhỏ (chunking) để đảm bảo **Điều 3** được tạo thành chunk riêng biệt và gán đúng nhãn `Article Ref: Điều 3`.
    *   Sử dụng chiến lược chunking dựa trên cấu trúc (structure-aware) thay vì chỉ dựa trên độ dài ký tự, để tách biệt rõ ràng giữa các "Điều" (Articles) và các "Bước" (Steps) trong quy trình.

2.  **Sửa lỗi gán nhãn (Metadata Correction):**
    *   Đối với các chunk thuộc Phụ lục/Quy trình: Cần gán `Article Ref` là mã TTHC cụ thể (ví dụ: `1.004946.000.00.00.H01`) hoặc tên thủ tục thay vì để trống hoặc gán chung chung là "Dữ liệu Bảng biểu".
    *   Loại bỏ các chunk trùng lặp (giữ lại 1 bản duy nhất cho mỗi bước quy trình).

3.  **Vệ sinh dữ liệu (Data Cleaning):**
    *   Loại bỏ các đoạn văn bản lặp lại vô nghĩa (như "TCCN có nhu cầu... TCCN có nhu cầu...").
    *   Chuẩn hóa lại định dạng bảng biểu: Nếu không thể giữ nguyên bảng, hãy chuyển đổi nội dung bảng thành văn bản mô tả (narrative text) để đảm bảo ngữ nghĩa không bị mất khi cắt chunk.

4.  **Kiểm tra lại quy trình ETL:**
    *   Rà soát lại logic trích xuất từ nguồn gốc (PDF/HTML) để đảm bảo không bỏ sót các điều khoản nằm giữa các phần phụ lục.
    *   Thêm bước kiểm tra (validation) tự động: So sánh danh sách Điều trong Neo4j với danh sách `Article Ref` trong Qdrant để phát hiện sớm các trường hợp "nuốt tin".

---

## Van ban: 57/2025/TT-BYT

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **57/2025/TT-BYT** dựa trên dữ liệu Dump từ Qdrant và Neo4j:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:** **KÉM**
Dữ liệu hiện tại có nhiều lỗi nghiêm trọng về cấu trúc, nội dung và tính toàn vẹn, sẽ gây ra tình trạng "ảo giác" (hallucination) hoặc trả lời sai lệch khi hệ thống RAG hoạt động. Đặc biệt là sự thiếu hụt Điều 1, lỗi gộp nội dung và sự xuất hiện của dữ liệu rác (spam).

---

#### **2. Các lỗi cụ thể tìm thấy:**

**A. Lỗi nuốt tin (Missing Data) - Nghiêm trọng**
*   **Điều 1 bị mất hoàn toàn:**
    *   **Mục lục (Neo4j):** Có liệt kê "Điều 1. Phạm vi điều chỉnh".
    *   **Qdrant:** Không có Chunk nào có `Article Ref: Điều 1`.
    *   **Hệ quả:** Người dùng hỏi về "Phạm vi điều chỉnh" sẽ không tìm thấy thông tin chính xác hoặc hệ thống sẽ trả lời sai dựa trên các điều khác.

**B. Lỗi gộp (Swallowing) & Định danh sai**
*   **Chunk 7 (ID: 98b55b11...):**
    *   **Lỗi:** Gộp 2 nội dung khác nhau vào 1 chunk nhưng chỉ gán nhãn `Article Ref: Điều 6`.
    *   **Chi tiết:** Chunk chứa cả nội dung "Hiệu lực thi hành" (thường là Điều 5 hoặc 6) VÀ "Lộ trình thực hiện" (thường là Điều 6 hoặc 7). Trong khi đó, Mục lục Neo4j tách biệt "Điều 5. Hiệu lực thi hành" và "Điều 6. Lộ trình thực hiện".
    *   **Mâu thuẫn:** Chunk này gán nhãn là Điều 6 nhưng nội dung lại bao trùm cả Điều 5 (theo logic mục lục).
*   **Chunk 8 (ID: a495868b...):**
    *   **Lỗi:** Gán nhãn `Article Ref: Điều 2` nhưng nội dung lại chứa cả "Phạm vi điều chỉnh" (thường là Điều 1) và "Căn cứ để xác định..." (Điều 2).
    *   **Chi tiết:** Dòng đầu tiên trong nội dung là "Điều 2 / Phạm vi điều chỉnh". Tuy nhiên, theo Mục lục Neo4j, "Phạm vi điều chỉnh" là **Điều 1**. Chunk này đang gộp Điều 1 và Điều 2 nhưng chỉ dán nhãn Điều 2.

**C. Lỗi rác kỹ thuật (Technical Noise)**
*   **Chunk 1, 7, 8:**
    *   **Lỗi:** Chứa đoạn text spam/copy từ website nguồn: *"Hãy đăng nhập hoặc đăng ký Thành viên Pro tại đây để xem toàn bộ văn bản tiếng Anh."*
    *   **Hệ quả:** Làm giảm chất lượng vector embedding, gây nhiễu khi tìm kiếm ngữ nghĩa.
*   **Chunk 1:**
    *   **Lỗi:** Nội dung bị cắt cụt và chứa dữ liệu lạ: *"Cục Quản lý Thực phẩm và Dược phẩm Hoa Kỳ..."* (Có vẻ là nội dung của một văn bản khác hoặc phần dịch tiếng Anh bị chèn sai vào Phụ lục).

**D. Lỗi định danh (Article Ref) & Cấu trúc**
*   **Chunk 1 (Phụ lục):**
    *   **Lỗi:** `Article Ref` để trống (` `) dù `Is Appendix: True`. Cần gán rõ ràng là "PHỤ LỤC" hoặc "Bảng phân nhóm" để hệ thống truy xuất chính xác.
*   **Chunk 5 (Căn cứ):**
    *   **Lỗi:** `Article Ref` để trống. Đây là phần "Căn cứ ban hành" (thường nằm ở đầu văn bản, trước Điều 1). Việc không gán nhãn khiến hệ thống không biết đây là phần mở đầu hay một điều khoản cụ thể.

**E. Lỗi logic số thứ tự Điều (Inconsistency)**
*   **Mục lục Neo4j:** Liệt kê: Điều 1, 2, 3, **5**, 6, 7, 8. (Thiếu Điều 4).
*   **Qdrant:** Có Chunk cho Điều 2, 3, 6, 7, 8.
*   **Vấn đề:** Cần xác minh xem văn bản gốc có thực sự bỏ qua Điều 4 hay không, hoặc đây là lỗi trong quá trình trích xuất (parsing) khiến Điều 4 bị mất.

---

#### **3. Đề xuất khắc phục:**

1.  **Tái xử lý (Re-chunking) với quy tắc tách biệt:**
    *   Bắt buộc tách **Điều 1** (Phạm vi điều chỉnh) thành một chunk riêng biệt.
    *   Tách **Điều 5** (Hiệu lực) và **Điều 6** (Lộ trình) thành các chunk riêng biệt, không gộp chung.
    *   Tách phần "Căn cứ ban hành" (Chunk 5) thành một chunk riêng với nhãn `Article Ref: Căn cứ` hoặc `Phần mở đầu`.

2.  **Làm sạch dữ liệu (Data Cleaning):**
    *   Loại bỏ hoàn toàn các đoạn text: *"Hãy đăng nhập hoặc đăng ký Thành viên Pro..."* và các đoạn text tiếng Anh không liên quan (FDA) trước khi đưa vào Vector DB.
    *   Chuẩn hóa lại `Article Ref` cho Chunk 1 (Phụ lục) thành "PHỤ LỤC".

3.  **Kiểm tra lại logic số thứ tự:**
    *   Đối chiếu lại văn bản gốc để xác nhận xem Điều 4 có tồn tại không. Nếu có, cần thêm chunk cho Điều 4. Nếu không, cần cập nhật lại Metadata trong Neo4j để ghi chú rõ "Bỏ qua Điều 4" nhằm tránh nhầm lẫn cho người dùng.

4.  **Cải thiện Metadata:**
    *   Đảm bảo mọi chunk đều có `Article Ref` rõ ràng. Nếu là phần mở đầu, gán là "Mở đầu". Nếu là Phụ lục, gán là "Phụ lục". Không để trống.

**Kết luận:** Dữ liệu hiện tại **không thể sử dụng** cho mục đích RAG pháp luật chính thống do thiếu Điều 1 và chứa nhiều lỗi gộp/nhiễu. Cần chạy lại quy trình ETL (Extract, Transform, Load) với các quy tắc tách đoạn (splitting rules) chặt chẽ hơn dựa trên cấu trúc "Điều - Khoản - Điểm".

---

## Van ban: 51/2025/TT-BYT

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **51/2025/TT-BYT** dựa trên dữ liệu Dump từ Qdrant và Neo4j.

---

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát: KÉM**
Dữ liệu hiện tại có **nhiều lỗi nghiêm trọng** về cấu trúc và định danh, khiến hệ thống RAG không thể trả lời chính xác các câu hỏi pháp lý.
*   **Mất mát thông tin:** Thiếu hoàn toàn nội dung của **Điều 1** (Điều khoản quan trọng nhất về việc ban hành).
*   **Định danh sai lệch:** Các chunk chứa nội dung quan trọng (Mã HS, Lộ trình, Giải thích từ ngữ) bị để trống `Article Ref` hoặc gán sai nội dung.
*   **Dữ liệu rác:** Tồn tại nhiều chunk chứa Header/Footer, nội dung lặp lại và bảng biểu bị cắt cụt.
*   **Mâu thuẫn cấu trúc:** Neo4j ghi nhận có Điều 1 nhưng Qdrant không có chunk tương ứng.

---

#### **2. Các lỗi cụ thể tìm thấy**

**A. Lỗi nuốt tin (Missing Content)**
*   **Điều 1 (Ban hành Quy chuẩn kỹ thuật quốc gia):**
    *   *Dữ liệu Neo4j:* Có trong Mục lục (`document_toc`).
    *   *Dữ liệu Qdrant:* **Không có Chunk nào** chứa nội dung "Điều 1".
    *   *Hệ quả:* Người dùng hỏi "Thông tư này ban hành quy chuẩn gì?" sẽ không tìm thấy câu trả lời chính xác từ chunk nội dung.

**B. Lỗi định danh (Article Ref Errors)**
*   **Chunk 3, 7, 9:**
    *   *Lỗi:* `Article Ref` bị **trống**.
    *   *Nội dung thực tế:* Chứa "Mã HS của thuốc lá điếu", "Giải thích từ ngữ", "Việc ghi nhãn...". Đây là nội dung thuộc **Điều 3** (hoặc các điều khoản kỹ thuật trong QCVN đính kèm) nhưng bị mất nhãn.
    *   *Đặc biệt:* Chunk 3 và Chunk 7 có nội dung **giống hệt nhau** (lặp lại) nhưng đều bị mất nhãn.
*   **Chunk 1:**
    *   *Lỗi:* Gán nhãn là `Điều 2` nhưng nội dung lại chứa cả "Hiệu lực thi hành" (thường là Điều 2) và phần "Ban hành Quy chuẩn" (thường là Điều 1). Có dấu hiệu gộp sai hoặc nội dung bị cắt xén không đúng chỗ.

**C. Lỗi gộp (Swallowing) & Cấu trúc**
*   **Chunk 1:** Gộp nội dung của việc "Ban hành QCVN" (Điều 1) và "Hiệu lực thi hành" (Điều 2) vào một chunk duy nhất với nhãn `Điều 2`. Điều này làm sai lệch ngữ nghĩa pháp lý.
*   **Chunk 6 (Bảng biểu):**
    *   *Lỗi:* `Article Ref` trống.
    *   *Nội dung:* Chứa bảng "Lộ trình áp dụng" (Tar/Nicotine). Đây là nội dung cốt lõi của **Điều 3** nhưng bị tách rời và mất nhãn.
    *   *Kỹ thuật:* Bảng bị cắt cụt (`|...`), thiếu phần kết thúc, gây khó khăn cho việc đọc hiểu của LLM.

**D. Lỗi rác kỹ thuật (Technical Noise)**
*   **Chunk 4:**
    *   *Lỗi:* Chứa toàn bộ Header của văn bản ("BỘ Y TẾ | CỘNG HÒA...", "Số: 51/2025...", "Căn cứ...").
    *   *Vấn đề:* Đây là phần "Lời mở đầu" (Preamble), không phải là một Điều khoản cụ thể. Việc gán `Article Ref` trống là đúng, nhưng chunk này không mang giá trị tìm kiếm nội dung luật (Retrieval value thấp) và làm loãng vector space.
*   **Chunk 5:**
    *   *Lỗi:* Nội dung bị cắt cụt ở giữa câu ("...Bộ Khoa học và Công nghệ sửa đổi..."). Có vẻ như đây là phần trích dẫn văn bản khác bị chèn vào sai chỗ hoặc lỗi cắt chunk.
*   **Chunk 3 & 7:**
    *   *Lỗi:* Nội dung **lặp lại 100%** (Duplicate). Đây là lỗi nghiêm trọng trong quá trình băm (chunking) hoặc tải dữ liệu.

**E. Cấu trúc Phụ lục (QCVN)**
*   Văn bản này ban hành kèm theo QCVN 16-1:2025/BYT.
*   *Vấn đề:* Các chunk (3, 6, 7, 9) chứa nội dung kỹ thuật của QCVN (Mã HS, Lộ trình, Giải thích từ ngữ) nhưng **không được đánh dấu `Is Appendix: True`** và bị gán sai hoặc mất `Article Ref`. Hệ thống sẽ không phân biệt được đâu là nội dung Thông tư và đâu là nội dung Quy chuẩn kỹ thuật đính kèm.

---

#### **3. Đề xuất khắc phục**

1.  **Tái tạo Chunk cho Điều 1:**
    *   Cần trích xuất lại nội dung "Điều 1. Ban hành Quy chuẩn kỹ thuật quốc gia" từ văn bản gốc và tạo một chunk mới với `Article Ref: Điều 1`.

2.  **Sửa lại logic gán nhãn (Labeling Logic):**
    *   **Chunk 3, 7, 9:** Cần gán lại `Article Ref` chính xác (có thể là `Điều 3` hoặc `Phụ lục I` tùy thuộc vào cấu trúc QCVN).
    *   **Chunk 6:** Gán `Article Ref: Điều 3` (vì nói về Lộ trình áp dụng) và sửa lại định dạng bảng để không bị cắt cụt.
    *   **Chunk 1:** Tách nội dung "Ban hành QCVN" sang một chunk riêng (gán Điều 1) và giữ lại phần "Hiệu lực" cho Điều 2.

3.  **Xử lý dữ liệu rác và trùng lặp:**
    *   **Xóa Chunk 7:** Vì nội dung trùng lặp hoàn toàn với Chunk 3.
    *   **Tối ưu Chunk 4:** Nếu không cần thiết cho việc tìm kiếm nội dung luật, có thể loại bỏ hoặc gán thẻ `Is Header: True` để hệ thống RAG ưu tiên thấp hơn.
    *   **Sửa Chunk 5:** Kiểm tra lại văn bản gốc để đảm bảo nội dung không bị cắt cụt giữa chừng.

4.  **Cải thiện cấu trúc Phụ lục:**
    *   Đánh dấu `Is Appendix: True` cho các chunk chứa nội dung kỹ thuật của QCVN 16-1:2025/BYT (Mã HS, Lộ trình, Giải thích từ ngữ).
    *   Đảm bảo các chunk này có `Article Ref` rõ ràng (ví dụ: `Phụ lục I - Điều 1`, `Phụ lục I - Điều 2`...).

5.  **Kiểm tra lại quy trình Chunking:**
    *   Cần rà soát lại thuật toán chia nhỏ văn bản để tránh việc cắt ngang giữa các bảng biểu (Chunk 6) và tránh tạo ra các chunk trùng lặp (Chunk 3 & 7).

---

## Van ban: 2587/QĐ-UBND

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **2587/QĐ-UBND** dựa trên dữ liệu Dump từ Qdrant và Neo4j:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:** **Kém**
Dữ liệu hiện tại có nhiều lỗi nghiêm trọng về định danh (metadata) và cấu trúc chunking, đặc biệt là việc xử lý các bảng biểu (Phụ lục) và phần đầu văn bản. Các chunk quan trọng bị mất nhãn "Điều khoản", gây khó khăn cho việc truy xuất chính xác (Retrieval) và có nguy cơ trả về kết quả sai lệch (Hallucination) khi người dùng hỏi về nội dung cụ thể.

---

#### **2. Các lỗi cụ thể tìm thấy:**

**A. Lỗi định danh (Article Ref) - Nghiêm trọng**
*   **Chunk 1 & Chunk 6 (Bảng biểu/Phụ lục):**
    *   **Lỗi:** `Article Ref` bị để trống (`""`).
    *   **Mô tả:** Đây là nội dung quan trọng nhất của văn bản (Danh sách các thủ tục hành chính được phê duyệt). Việc không gán nhãn (ví dụ: "Phụ lục" hoặc "Điều 1 - Phụ lục") khiến hệ thống không thể liên kết các thủ tục này với quyết định phê duyệt.
    *   **Chunk 1:** Chứa bảng STT 1-3.
    *   **Chunk 6:** Chứa bảng STT 1-2 (có vẻ là phần tiếp theo của bảng).
*   **Chunk 2 (Phần đầu văn bản):**
    *   **Lỗi:** `Article Ref` bị để trống.
    *   **Mô tả:** Chứa phần "Số: 2587/QĐ-UBND", ngày tháng và tiêu đề chính. Đây là chunk định danh văn bản, cần được gán nhãn đặc biệt (ví dụ: "Header" hoặc "Thông tin chung") để phân biệt với các Điều khoản.

**B. Lỗi gộp (Swallowing) & Cấu trúc**
*   **Chunk 5 (Điều 2):**
    *   **Lỗi:** Gán nhãn `Điều 2 > Khoản 1, 2, 3, 4` nhưng nội dung chỉ hiển thị dòng "Quyết định này có hiệu lực thi hành kể từ ngày ký" và bắt đầu liệt kê "1. Thay thế...".
    *   **Phân tích:** Có dấu hiệu chunk bị cắt cụt hoặc gộp sai. Nội dung "1. Thay thế..." thực chất là nội dung của **Điều 2** (về hiệu lực và thay thế), nhưng việc gán nhãn "Khoản 1, 2, 3, 4" trong khi nội dung chưa hiển thị đủ các khoản này là không chính xác. Cần kiểm tra xem các khoản 2, 3, 4 có bị rơi vào chunk khác hay không.

**C. Lỗi nuốt tin (Missing Content)**
*   **So sánh Neo4j vs Qdrant:**
    *   **Neo4j (Mục lục):** Liệt kê đầy đủ Điều 1, 2, 3, 4.
    *   **Qdrant:** Có chunk cho Điều 1, 2, 3, 4.
    *   **Kết luận:** Không có lỗi "nuốt tin" ở cấp độ Điều khoản chính. Tuy nhiên, **nội dung chi tiết của Phụ lục (Bảng biểu)** trong Neo4j (nếu có) không được ánh xạ rõ ràng sang các Chunk 1 và 6 do thiếu `Article Ref`.

**D. Lỗi rác kỹ thuật & Dữ liệu lặp**
*   **Lỗi lặp Header:** Tất cả 7 chunks đều chứa đoạn text lặp lại: `Văn bản: 2587/QĐ-UBND - Quyết định 2587/QĐ-UBND năm 2025 phê duyệt...`.
    *   **Tác động:** Làm loãng vector embedding, giảm độ chính xác khi tìm kiếm nội dung thực tế.
*   **Lỗi định dạng bảng (Chunk 1 & 6):**
    *   Có các dòng `|||||` và `||` rỗng giữa các dòng dữ liệu. Đây là dấu hiệu của lỗi parse Markdown/HTML khi chuyển đổi văn bản sang text, gây nhiễu dữ liệu.

---

#### **3. Đề xuất khắc phục:**

1.  **Sửa Metadata (Article Ref):**
    *   Gán nhãn `Article Ref` cho **Chunk 1** và **Chunk 6** là `"Phụ lục"` hoặc `"Điều 1 - Danh mục TTHC"`.
    *   Gán nhãn `Article Ref` cho **Chunk 2** là `"Thông tin chung"` hoặc `"Header"`.
    *   Đảm bảo `Is Appendix: True` cho các chunk chứa bảng biểu (Chunk 1, 6).

2.  **Tối ưu hóa Chunking (Băm nhỏ):**
    *   **Loại bỏ Header lặp:** Cấu hình lại pipeline để loại bỏ đoạn tiêu đề văn bản lặp lại ở đầu mỗi chunk, chỉ giữ lại trong chunk đầu tiên (Chunk 2).
    *   **Xử lý Bảng biểu:** Đảm bảo các dòng `|||||` rỗng được làm sạch (cleaning) trước khi đưa vào vector DB. Nếu bảng quá dài, hãy chia nhỏ theo từng nhóm thủ tục nhưng vẫn giữ nguyên cấu trúc tiêu đề cột.

3.  **Kiểm tra lại Chunk 5 (Điều 2):**
    *   Kiểm tra xem nội dung "Khoản 2, 3, 4" có bị cắt vào chunk khác không. Nếu Điều 2 chỉ có 1 đoạn văn bản duy nhất, hãy sửa nhãn thành `Điều 2` (bỏ phần "Khoản 1, 2, 3, 4" nếu không có sự phân tách rõ ràng trong nội dung).

4.  **Cải thiện Neo4j (Graph DB):**
    *   Đảm bảo nút (Node) cho "Phụ lục" hoặc "Bảng biểu" được tạo ra trong Neo4j và liên kết (Relationship) với các Chunk 1 và 6 trong Qdrant để hỗ trợ truy xuất ngữ cảnh tốt hơn.

**Kết luận:** Cần chạy lại quy trình tiền xử lý (Preprocessing) để làm sạch văn bản, gán lại metadata chính xác cho các phần Phụ lục và Header trước khi đưa vào Vector DB.

---

## Van ban: 54/NQ-HĐND

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **Nghị quyết 54/NQ-HĐND năm 2025** dựa trên dữ liệu Dump bạn cung cấp:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:**
**KÉM (Critical)**
Dữ liệu hiện tại không thể sử dụng cho hệ thống RAG pháp luật do thiếu nghiêm trọng các tham chiếu định danh (Article Ref) và mất kết nối giữa cấu trúc văn bản (Neo4j) với nội dung chi tiết (Qdrant). Hệ thống sẽ không thể trả lời chính xác các câu hỏi về "Điều khoản nào quy định..." hoặc trích xuất đúng bảng giá cụ thể.

#### **2. Các lỗi cụ thể tìm thấy:**

*   **Lỗi định danh (Article Ref) - Mức độ nghiêm trọng: Cao**
    *   **Tất cả 15 Chunks (ID từ 00574399... đến 05301f2f...)** đều có trường `Article Ref` bị **TRỐNG**.
    *   **Hậu quả:** Khi người dùng hỏi "Giá phẫu thuật cắt tử cung là bao nhiêu?", hệ thống tìm thấy Chunk 5 nhưng không biết nội dung này thuộc Điều 2 hay Phụ lục III, dẫn đến việc trích dẫn nguồn sai hoặc không trích dẫn được.
    *   **Lưu ý:** Chunk 14 có nội dung "PHỤ LỤC III" trong phần `Điều khoản`, nhưng trường `Article Ref` vẫn trống.

*   **Lỗi nuốt tin (Missing Content) - Mức độ nghiêm trọng: Cao**
    *   **Mâu thuẫn dữ liệu:**
        *   **Neo4j (Mục lục):** Liệt kê rõ 5 Điều khoản (Điều 1 đến Điều 5) với nội dung chi tiết (Khoản 1, 2, 3...).
        *   **Qdrant (Chunks):** **Không có Chunk nào** chứa nội dung văn bản của Điều 1, 2, 3, 4, 5.
        *   **Thực tế:** Toàn bộ 15 Chunks trong Qdrant đều là dữ liệu bảng biểu (Phụ lục) hoặc tiêu đề chung. Nội dung chính của Nghị quyết (các Điều khoản quy định) đã bị "nuốt" hoặc không được băm nhỏ (chunking) vào Vector DB.
    *   **Hậu quả:** Hệ thống không thể trả lời các câu hỏi về phạm vi điều chỉnh, nguyên tắc áp dụng hay tổ chức thực hiện.

*   **Lỗi cấu trúc Phụ lục (Appendix Structure)**
    *   **Chunk 14:** Được gán nhãn `Is Appendix: True` và nội dung là tiêu đề "PHỤ LỤC III". Tuy nhiên, các Chunk chứa dữ liệu bảng (Chunk 1-13, 15) cũng được gán `Is Appendix: True` nhưng không có tham chiếu rõ ràng đến Phụ lục I, II hay III.
    *   **Thiếu ngữ cảnh:** Các bảng giá (Chunk 1, 2, 3...) bị tách rời hoàn toàn khỏi tiêu đề Phụ lục (Chunk 14). Nếu người dùng hỏi về "Bảng giá tại Bệnh viện Đa khoa tỉnh", hệ thống có thể trả về một đoạn bảng ngẫu nhiên mà không biết đó là của Phụ lục nào.

*   **Lỗi rác kỹ thuật & Định dạng**
    *   **Lặp lại Header:** Tất cả các Chunk đều chứa đoạn text lặp lại: `Văn bản: 54/NQ-HĐND - Nghị quyết 54/NQ-HĐND năm 2025...`. Đây là "rác" không cần thiết làm loãng vector embedding.
    *   **Cắt dòng bảng biểu:** Một số Chunk (ví dụ Chunk 1, 2, 4) bị cắt ngang giữa dòng nội dung bảng (`...vật liệ...`, `...ngón chân trên ...`), làm mất tính toàn vẹn của dữ liệu bảng.

#### **3. Đề xuất khắc phục:**

1.  **Sửa quy trình Chunking (Quan trọng nhất):**
    *   **Tách biệt nội dung chính và Phụ lục:** Cần tạo các Chunk riêng cho nội dung của Điều 1, 2, 3, 4, 5 (hiện đang thiếu).
    *   **Gán Article Ref động:** Khi tạo Chunk cho Phụ lục, bắt buộc phải điền `Article Ref` là tên Phụ lục tương ứng (ví dụ: "Phụ lục III") hoặc tham chiếu đến Điều khoản quy định về giá (thường là Điều 2).
    *   **Xử lý bảng biểu:** Sử dụng thuật toán chunking chuyên biệt cho bảng (Table-aware chunking) để đảm bảo không cắt ngang dòng hoặc cột quan trọng. Nếu bắt buộc cắt, cần thêm ngữ cảnh (context) ở đầu và cuối chunk.

2.  **Làm sạch dữ liệu (Data Cleaning):**
    *   Loại bỏ đoạn text lặp lại `Văn bản: 54/NQ-HĐND...` ở đầu mỗi Chunk.
    *   Chuẩn hóa định dạng số tiền (thống nhất dấu chấm/phẩy) nếu cần thiết cho việc so sánh giá.

3.  **Cập nhật Neo4j:**
    *   Đảm bảo các Node `Article` trong Neo4j có liên kết (Relationship) `HAS_CHUNK` hoặc `CONTAINS` đến các Chunk ID tương ứng trong Qdrant. Hiện tại mối liên hệ này đang bị đứt gãy hoàn toàn.

4.  **Kiểm tra lại logic "Is Appendix":**
    *   Chỉ gán `Is Appendix: True` cho các chunk thuộc phần Phụ lục. Các chunk thuộc Điều 1-5 phải gán `Is Appendix: False` và có `Article Ref` cụ thể (ví dụ: "Điều 1").

**Kết luận:** Cần phải chạy lại quy trình ETL (Extract, Transform, Load) cho văn bản này. Dữ liệu hiện tại chỉ là một tập hợp các đoạn bảng rời rạc không có "hồn" (thiếu định danh và thiếu nội dung chính).

---

## Van ban: 55/2025/TT-BYT

Dưới đây là báo cáo kiểm định dữ liệu RAG cho văn bản **55/2025/TT-BYT** dựa trên dữ liệu Dump từ Qdrant và Neo4j:

### **Báo cáo Kiểm định Dữ liệu RAG**

#### **1. Đánh giá tổng quát:** **KÉM**
Dữ liệu hiện tại có nhiều lỗi nghiêm trọng về định danh, cấu trúc và tính toàn vẹn, sẽ gây ra tình trạng "Hallucination" (ảo giác) hoặc trả lời sai lệch khi hệ thống RAG hoạt động. Đặc biệt, việc thiếu các Điều khoản quan trọng và gán nhãn sai cho các mẫu biểu là lỗi chí mạng.

---

#### **2. Các lỗi cụ thể tìm thấy:**

**A. Lỗi nuốt tin (Missing Data - Critical)**
*   **Điều 1, 2, 4, 6, 10, 11, 12, 14:** Xuất hiện đầy đủ trong Mục lục (Neo4j) nhưng **hoàn toàn không có Chunk tương ứng** trong Qdrant.
    *   *Ví dụ:* Điều 1 (Phạm vi điều chỉnh), Điều 2 (Giải thích từ ngữ), Điều 10 (Kê đơn điện tử) là các nội dung cốt lõi nhưng bị mất.
    *   *Nguyên nhân:* Quy trình Chunking có thể đã bỏ qua các chương đầu hoặc các điều khoản ngắn, hoặc chỉ trích xuất một phần nhỏ của văn bản.

**B. Lỗi định danh (Article Ref Errors)**
*   **Chunk 5 (ID: 2f9c27d7...):**
    *   **Lỗi:** `Article Ref` để trống (`Is Appendix: False`).
    *   **Nội dung:** Là phần "Hướng dẫn sử dụng" (Cách sắc, cách uống...) thuộc về Mẫu đơn thuốc.
    *   **Hệ quả:** Hệ thống không biết đoạn này thuộc Điều nào hay Phụ lục nào, gây nhiễu khi tìm kiếm.
*   **Chunk 2, 4, 9 (ID: 0f57a2b1..., 297c8402..., 4cdb4a03...):**
    *   **Lỗi:** `Article Ref` gán là "ĐƠN THUỐC" thay vì tên cụ thể của Phụ lục (ví dụ: "Mẫu 1", "Mẫu 2" hoặc "Phụ lục I").
    *   **Hệ quả:** Khó phân biệt giữa các loại đơn thuốc khác nhau (Đơn thuốc thang vs Đơn thuốc cổ truyền).

**C. Lỗi rác kỹ thuật & Định dạng (Technical Noise)**
*   **Chunk 5:** Chứa các ký tự lặp lại vô nghĩa (`………………………………………………………………………………`) và lỗi chính tả trong nội dung ("Lời dặn8" thay vì "Lời dặn").
*   **Chunk 6 (ID: 30660527...):**
    *   **Lỗi:** Nội dung là danh sách thuốc (Thục địa, Trạch tả...) nhưng `Article Ref` là "Điều 8".
    *   **Phân tích:** Đây rõ ràng là ví dụ minh họa trong một Mẫu đơn hoặc Phụ lục, không phải là nội dung quy định pháp lý của Điều 8. Việc gán vào Điều 8 là sai logic.
*   **Chunk 7 & 15:** Chứa các ký tự lạ như `[- tiếp theo]` hoặc `[4. tiếp theo]` trong nội dung chính, cho thấy việc cắt đoạn (splitting) chưa làm sạch các marker nối đoạn.

**D. Lỗi gộp/Swallowing (Potential)**
*   **Chunk 13 (ID: 8f6575bb...):**
    *   Nội dung bắt đầu bằng "Ngày 01 thang x 07 ngày..." (ví dụ minh họa) ngay sau tiêu đề Điều 8.
    *   Có dấu hiệu gộp nội dung quy định (Khoản 6, 7) với ví dụ minh họa hoặc nội dung của Mẫu đơn vào cùng một chunk mà không tách biệt rõ ràng, gây khó khăn cho việc trích xuất chính xác quy định pháp lý.

**E. Cấu trúc Phụ lục (Appendix Structure)**
*   Các mẫu đơn (Chunk 2, 4, 5, 6, 9) bị băm vụn và không có thứ tự logic rõ ràng.
*   Chunk 5 (Hướng dẫn sử dụng) bị tách rời khỏi Mẫu đơn mà nó thuộc về (Chunk 4/9) và bị gán nhãn sai.

---

#### **3. Đề xuất khắc phục:**

1.  **Tái xử lý (Re-chunking) toàn bộ văn bản:**
    *   Chạy lại quy trình trích xuất để đảm bảo **tất cả** các Điều từ 1 đến 16 đều có mặt trong Qdrant.
    *   Sử dụng chiến lược chunking dựa trên cấu trúc (Semantic/Structural chunking) thay vì chỉ dựa trên độ dài ký tự, để đảm bảo mỗi Điều khoản là một đơn vị logic hoàn chỉnh.

2.  **Sửa quy tắc gán nhãn (Metadata Tagging):**
    *   **Đối với Điều khoản:** Gán chính xác `Article Ref` (ví dụ: "Điều 1", "Điều 10").
    *   **Đối với Phụ lục/Mẫu:** Gán `Article Ref` là tên Phụ lục cụ thể (ví dụ: "Phụ lục I - Mẫu đơn thuốc thang") và đặt `Is Appendix: True`.
    *   **Đối với nội dung ví dụ trong mẫu:** Cần có tag riêng biệt (ví dụ: `Type: Example`) hoặc gộp chung vào chunk của Mẫu đó nhưng không gán nhầm vào Điều luật.

3.  **Làm sạch dữ liệu (Data Cleaning):**
    *   Loại bỏ các ký tự lặp lại (`...`), các marker nối đoạn (`[- tiếp theo]`, `[4. tiếp theo]`).
    *   Sửa lỗi chính tả ("Lời dặn8" -> "Lời dặn").
    *   Loại bỏ các dòng Header/Footer lặp lại nếu có.

4.  **Kiểm tra tính toàn vẹn (Validation):**
    *   Viết script so sánh danh sách `Article` trong Neo4j với danh sách `Article Ref` trong Qdrant. Nếu có Điều nào trong Neo4j mà không có trong Qdrant -> Báo lỗi và yêu cầu nạp lại.

5.  **Tối ưu hóa Chunk cho Mẫu đơn:**
    *   Giữ nguyên cấu trúc của Mẫu đơn (Mẫu 1, Mẫu 2...) thành các chunk riêng biệt hoặc gộp toàn bộ một Mẫu vào một chunk duy nhất nếu độ dài cho phép, thay vì cắt rời các phần "Họ tên", "Cách sắc" ra làm nhiều chunk không liên quan.

---

