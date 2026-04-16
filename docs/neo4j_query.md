Dưới đây là bộ sưu tập các câu lệnh Cypher "hạng nặng" được thiết kế đặc biệt để bạn "khám sức khỏe" toàn diện cho Database Neo4j vừa được nâng cấp. Bạn có thể copy từng khối lệnh và dán trực tiếp vào **Neo4j Browser** để xem kết quả.

---
MATCH (n) DETACH DELETE n;

### 1. Kiểm tra Tổng quan (Health Check)
**Mục đích:** Xem xét quy mô dữ liệu và xác nhận tất cả các loại Node (Nhãn) đã được tạo đầy đủ chưa.

```cypher
// Đếm tổng số lượng Node theo từng loại (Document, Article, Clause, Chunk, Sector, v.v.)
MATCH (n) 
RETURN labels(n)[0] AS Loai_Node, count(n) AS Tong_So_Luong 
ORDER BY Tong_So_Luong DESC;
```

---

### 2. Kiểm tra Cấu trúc Phân cấp (Dynamic Tree Hierarchy)
**Mục đích:** Đảm bảo luồng `Document <- Article <- Clause <- Chunk` đã được móc nối chính xác.

```cypher
// Lấy 5 văn bản có đầy đủ cấu trúc từ Điều -> Khoản để vẽ Đồ thị
MATCH p=(d:Document)<-[:BELONGS_TO|PART_OF]-(a:Article)<-[:PART_OF]-(cl:Clause)
RETURN p
LIMIT 10;
```
*(💡 **Mẹo:** Chạy lệnh này xong, bạn hãy chọn góc nhìn Graph (Đồ thị) để thấy các nhánh cây mọc ra từ một văn bản chính).*

---

### 3. Kiểm tra Luồng "Sửa đổi, Bổ sung" (Bóc tách Text Cũ/Mới)
**Mục đích:** Test "Vũ khí cốt lõi" của GraphRAG - kiểm tra xem các mũi tên có lưu trữ thành công tọa độ và bối cảnh (old_text, new_text) hay không.

```cypher
// Trích xuất chi tiết dữ liệu trên mũi tên AMENDS (Dạng Bảng)
MATCH (new:Document)-[r:AMENDS|REPLACES|REPEALS]->(old:Document)
WHERE 
    r.old_text IS NULL OR trim(r.old_text) = "" OR
    r.new_text IS NULL OR trim(r.new_text) = "" OR
    r.clause_ref <> "TOAN_BO"
RETURN 
    new.document_number AS Van_Ban_Sua_Doi,
    old.document_number AS Van_Ban_Goc,
    r.target_article AS Dieu_Bi_Sua,
    r.target_clause AS Khoan_Bi_Sua,
    r.old_text AS Noi_Dung_Cu,
    r.new_text AS Noi_Dung_Moi
LIMIT 10;
```

---

### 4. Kiểm tra Cơ chế "Tự Thanh Lọc" (Trạng thái Hiệu lực)
**Mục đích:** Xác minh xem logic tự động gán `doc_status = 'Hết hiệu lực'` khi bị bãi bỏ/thay thế toàn bộ đã hoạt động đúng chưa.

```cypher
// Tìm các văn bản đã bị đánh dấu "Hết hiệu lực" và nguyên nhân (Ai bãi bỏ/thay thế nó)
MATCH (d:Document {doc_status: 'Hết hiệu lực'})
OPTIONAL MATCH (new_doc)-[r:REPLACES|REPEALS]->(d)
RETURN d.document_number AS Van_Ban_Het_Hieu_Luc,
       d.title AS Tieu_De,
       new_doc.document_number AS Van_Ban_Thu_Tieu,
       type(r) AS Hanh_Dong
LIMIT 15;
```

---

### 5. Kiểm tra Mạng lưới Căn cứ Pháp lý (Multi-hop Reasoning)
**Mục đích:** Xem văn bản A căn cứ vào văn bản B, văn bản B lại căn cứ vào văn bản C.

```cypher
// Vẽ sơ đồ mạng nhện Căn cứ pháp lý đa tầng
MATCH p=(d1:Document)-[:BASED_ON*1..2]->(d2:Document)
RETURN p
LIMIT 15;
```

---

### 6. Kiểm tra Nút Ảo & Fallback Title (Enrichment Logic)
**Mục đích:** Xem các văn bản "Tham chiếu" (chưa có full text nhưng được lôi vào qua phần Căn cứ hoặc Xung đột) có được bơm đủ `title` và dán nhãn chính xác không.

```cypher
// Truy xuất các Node tham chiếu (Ghost Nodes)
MATCH (d:Document)
WHERE d.is_full_text = false OR d.id STARTS WITH 'REF_'
RETURN d.id AS ID_Node,
       d.document_number AS So_Hieu,
       d.title AS Tieu_De_Fallback,
       d.year AS Nam_Ban_Hanh,
       d.doc_status AS Trang_Thai
LIMIT 10;
```

---

### 7. Kiểm tra Bộ lọc Đa chiều (Filter Matrix)
**Mục đích:** Giả lập truy vấn của LLM khi muốn gom nhóm văn bản theo Lĩnh vực và Cơ quan ban hành.

```cypher
// Thống kê số lượng văn bản theo Lĩnh vực và Cơ quan ban hành
MATCH (d:Document)-[:HAS_SECTOR]->(s:Sector)
MATCH (a:Authority)-[:ISSUED]->(d)
RETURN s.name AS Linh_Vuc, 
       a.name AS Co_Quan_Ban_Hanh, 
       count(d) AS So_Luong_Van_Ban
ORDER BY So_Luong_Van_Ban DESC
LIMIT 15;
```

### 8. Hiển thị "Vũ trụ" xung quanh văn bản (Tổng quan nhất)
Câu lệnh này sẽ lấy nốt văn bản làm trung tâm và hiển thị tất cả các nốt liên quan trực tiếp như: Người ký, Cơ quan ban hành, Lĩnh vực, và các văn bản Căn cứ/Xung đột.

```cypher
MATCH (d:Document)
WHERE d.document_number = '2349/QĐ-UBND'
MATCH path = (d)-[*1..2]-(neighbor)
RETURN path;
```
* **Kết quả:** Bạn sẽ thấy các bong bóng Metadata (Sector, Authority, Signer) và các Document khác nối vào văn bản mục tiêu qua các mũi tên `AMENDS`, `REPLACES`, `BASED_ON`.

---

### 9. Hiển thị cấu trúc nội bộ (Cây Điều khoản & Chunks)
Nếu bạn muốn xem văn bản đó được "chẻ" nhỏ thành các Điều, Khoản và Chunk như thế nào trên đồ thị:

```cypher
MATCH (d:Document {document_number: '02400/QĐ-UBND'})
OPTIONAL MATCH p=(d)<-[:BELONGS_TO|PART_OF*1..3]-(leaf)
RETURN p;
```
* **Mẹo:** Trong Neo4j Browser, cấu trúc này sẽ hiện ra như một cái cây với gốc là `Document`, cành là `Article`, nhánh là `Clause` và lá là `Chunk`.

---

### 10. Kiểm tra sâu các Quan hệ Xung đột (Có kèm nội dung cũ/mới)
Để xem văn bản này đang "tác động" đến các văn bản khác như thế nào, hoặc văn bản nào đang tác động đến nó, kèm theo nội dung cụ thể trên mũi tên:

```cypher
MATCH (d:Document {document_number: '02400/QĐ-UBND'})-[r:AMENDS|REPLACES|REPEALS]-(other:Document)
RETURN d.document_number AS Van_Ban_Goc,
       type(r) AS Loai_Quan_He,
       other.document_number AS Van_Ban_Lien_Quan,
       r.target_article AS Dieu_Bi_Tac_Dong,
       substring(r.old_text, 0, 100) + "..." AS Noi_Dung_Cu,
       substring(r.new_text, 0, 100) + "..." AS Noi_Dung_Moi;
```

---

### 11. Truy vấn "Gia phả" (Căn cứ của Căn cứ)
Kiểm tra xem hệ thống có bắt được luồng dẫn chiếu đa tầng không (văn bản này căn cứ vào luật nào, luật đó lại căn cứ vào hiến pháp nào):

```cypher
MATCH p=(d:Document {document_number: '02400/QĐ-UBND'})-[:BASED_ON*1..2]->(ancestor:Document)
RETURN p;
```
MATCH (d:Document)-[r]-()
RETURN d.document_number AS So_Hieu,
       d.title AS Tieu_De,
       count(DISTINCT type(r)) AS So_Loai_Lien_Ket,
       collect(DISTINCT type(r)) AS Danh_Sach_Loai_Lien_Ket
ORDER BY So_Loai_Lien_Ket DESC, d.document_number ASC
LIMIT 10;

Câu hỏi của bạn vô cùng thú vị! Việc tìm ra các văn bản có "nhiều loại liên kết nhất" hoặc "nhiều liên kết nhất" chính là đi tìm các **"Siêu nút" (Hub Nodes)** trong lý thuyết Đồ thị. 

Trong hệ thống pháp luật, đây thường là các **Đạo luật gốc** (được hàng trăm văn bản khác căn cứ vào) hoặc các **Nghị định/Thông tư cồng kềnh** (đi sửa đổi, bãi bỏ hàng chục văn bản khác cùng lúc).

Dưới đây là 3 cấp độ truy vấn Cypher để bạn khám phá các "Siêu văn bản" này trong database của mình:

### 1. Tìm văn bản có ĐA DẠNG LOẠI liên kết nhất (Most Relationship Types)
Câu lệnh này đếm xem một văn bản sở hữu bao nhiêu **kiểu** mũi tên khác nhau (VD: Vừa có `AMENDS`, vừa có `REPEALS`, vừa `BASED_ON`, vừa `HAS_SECTOR`...).

```cypher
MATCH (d:Document)-[r]-()
RETURN d.document_number AS So_Hieu,
       d.title AS Tieu_De,
       count(DISTINCT type(r)) AS So_Loai_Lien_Ket,
       collect(DISTINCT type(r)) AS Danh_Sach_Loai_Lien_Ket
ORDER BY So_Loai_Lien_Ket DESC, d.document_number ASC
LIMIT 10;
```

### 2. Tìm "Trùm Sổ" - Văn bản có TỔNG SỐ liên kết pháp lý nhiều nhất
Câu lệnh này loại bỏ các liên kết cấu trúc (như Lĩnh vực, Người ký...) và chỉ tập trung đếm số lượng "Đạn" mà nó bắn ra hoặc bị bắn vào (Căn cứ, Sửa đổi, Thay thế, Bãi bỏ). Đây chính là văn bản "quyền lực" nhất hoặc phức tạp nhất.

```cypher
MATCH (d:Document)-[r:BASED_ON|AMENDS|REPLACES|REPEALS]-(other:Document)
RETURN d.document_number AS So_Hieu,
       d.title AS Tieu_De,
       count(r) AS Tong_So_Lien_Ket_Phap_Ly,
       count(DISTINCT type(r)) AS Da_Dang_Loai,
       collect(DISTINCT type(r)) AS Cac_Loai_Giao_Tiep
ORDER BY Tong_So_Lien_Ket_Phap_Ly DESC
LIMIT 10;
```

### 3. Phân rã "Quyền lực" (Chiều ra vs Chiều vào)
Bạn muốn biết văn bản này là "Kẻ đi săn" (đi sửa/bãi bỏ văn bản khác) hay là "Kẻ bị săn" (bị nhiều văn bản khác xúm vào sửa/bãi bỏ)? Câu lệnh này sẽ bóc tách hướng của mũi tên:

```cypher
MATCH (d:Document)
// Đếm số mũi tên bắn RA (Nó đi tác động/căn cứ văn bản khác)
OPTIONAL MATCH (d)-[out_rel:BASED_ON|AMENDS|REPLACES|REPEALS]->()
WITH d, count(out_rel) AS So_Mui_Ten_Ban_Ra
// Đếm số mũi tên cắm VÀO (Nó bị văn bản khác tác động/căn cứ)
OPTIONAL MATCH (d)<-[in_rel:BASED_ON|AMENDS|REPLACES|REPEALS]-()
WITH d, So_Mui_Ten_Ban_Ra, count(in_rel) AS So_Mui_Ten_Cam_Vao
WHERE So_Mui_Ten_Ban_Ra + So_Mui_Ten_Cam_Vao > 0
RETURN d.document_number AS So_Hieu,
       So_Mui_Ten_Ban_Ra AS Di_Tac_Dong_Ke_Khac,
       So_Mui_Ten_Cam_Vao AS Bi_Ke_Khac_Tac_Dong,
       (So_Mui_Ten_Ban_Ra + So_Mui_Ten_Cam_Vao) AS Tong_Tuong_Tac
ORDER BY Tong_Tuong_Tac DESC
LIMIT 10;
```

💡 **Mẹo phân tích:**
* Nếu `Bi_Ke_Khac_Tac_Dong` cao ngất ngưởng với mũi tên `BASED_ON`: Đó là một đạo Luật/Bộ Luật gốc xương sống.
* Nếu `Bi_Ke_Khac_Tac_Dong` cao với mũi tên `AMENDS`: Văn bản đó đã chắp vá nát bươm (bị sửa đổi nhiều lần), LLM khi đọc văn bản này rất dễ dính Lex Posterior (đọc nhầm luật cũ).
* Nếu `Di_Tac_Dong_Ke_Khac` cao: Đây thường là các Nghị định/Thông tư "dọn dẹp", ra đời để bãi bỏ hàng loạt văn bản đã lỗi thời.

Bạn hãy thử copy lệnh số 3 dán vào Neo4j Browser, kết quả trả về sẽ cho bạn một cái nhìn cực kỳ "Data Science" về hệ thống pháp luật đấy!