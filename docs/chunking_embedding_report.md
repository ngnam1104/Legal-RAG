# 📋 Báo Cáo Kỹ Thuật: Pipeline Xử Lý Văn Bản Pháp Luật

**Notebook:** [legal_rag_build_qdrant_2.ipynb](file:///d:/iCOMM/Legal-RAG/notebook/legal_rag_build_qdrant_2.ipynb)
**Ngày tạo báo cáo:** 2026-04-16

---

## Mục lục

1. [Phần 1: Kỹ thuật Chunking, Regex & Phát hiện cấu trúc](#phần-1-kỹ-thuật-chunking-regex--phát-hiện-cấu-trúc)
2. [Phần 2: Kỹ thuật dựng QdrantDB](#phần-2-kỹ-thuật-dựng-qdrantdb)
3. [Phần 3: Kỹ thuật dựng Neo4jDB](#phần-3-kỹ-thuật-dựng-neo4jdb)

---

# Phần 1: Kỹ thuật Chunking, Regex & Phát hiện cấu trúc

## 1.1 Tổng quan kiến trúc Chunker

Toàn bộ logic chunking nằm trong class `AdvancedLegalChunker`, được thiết kế theo mô hình **Finite State Machine (FSM)** — máy trạng thái hữu hạn duyệt từng dòng văn bản từ trên xuống dưới.

```text
[BẮT ĐẦU]
   |
   v
[HeaderZone] (Dòng đầu tiên)
   |-- Regex Chương match --> [ChapterDetected] -- Regex Điều match --> [ArticleDetected]
   |-- Regex Điều match ----> [ArticleDetected]
   |-- Không match gì ------> [FreeText]
   
[ArticleDetected]
   |-- Regex Khoản match ----> [ClauseDetected] ---> Vượt CHUNK_LIMIT ---> [FlushChunk] (Tạo chunk mới)
   |                                |-- Regex Điểm match --> [PointDetected] ---> Vượt CHUNK_LIMIT ---> [FlushChunk]
   |                                |-- Khoản mới ---------> [ClauseDetected]
   |                                
   |-- Regex Phụ lục match --> [AppendixZone]
   |                                |-- I, II, III ---> [AppxLevel1]
   |                                                      |-- 1., 2. ---> [AppxLevel2]
   |                                                                        |-- 1.1, a) ---> [AppxLevel3]
   |
   |-- Dòng có >=2 ký tự "|" -> [TableZone] ---> Hết bảng ---> [ArticleDetected]

[FreeText] ---> Vượt TEXT_LIMIT ---> [FlushChunk] (Tạo chunk mới)
```

> [!NOTE]
> **Giải thích thuật toán FSM trong mã nguồn (`legal_rag_build_qdrant_2.ipynb`):**
> Lớp `AdvancedLegalChunker` xử lý tuần tự theo các bước:
> - **Bước 1 (Khởi tạo):** Đọc văn bản theo dòng, bắt đầu ở vùng `HeaderZone`.
> - **Bước 2 (Phân phối cấu trúc):** Áp dụng Regex (`chapter_pattern`, `article_pattern`...) để chuyển mốc trạng thái từ cấp Chương nhảy xuống Điều.
> - **Bước 3 (Khoản/Điểm):** Khi đang ở trong Điều, hệ thống rẽ nhánh kiểm tra `clause_pattern` và `point_pattern`. Tính kế thừa được duy trì qua biến lưu trạng thái.
> - **Bước 4 (Ngoại lệ - Phụ lục/Bảng):** Cùng lúc dò tìm Phụ Lục (`appendix_title_pattern`) hoặc Bảng biểu (phát hiện ký tự `|`).
> - **Bước 5 (Flush):** Tại bất kỳ trạng thái nào, nếu `current_text` phình to quá `CHUNK_LIMIT` (1200 ký tự) hoặc `TEXT_LIMIT`, hàm `flush_article()` sẽ được gọi để chốt hạ và đóng gói chunk lại một cách an toàn.

## 1.2 Hệ thống Regex Patterns — Bộ nhận diện phân cấp (Hierarchy Patterns)

### 1.2.1 Cấp 1: Chương / Phần (`chapter_pattern`)

```python
re.compile(r"(?im)^\s*(Chương|Phần)(?:\s+thứ)?\s+([a-zA-Z0-9]+|\d+)\b\s*[\.:\-]?\s*(.*)$")
```

| Thành phần | Ý nghĩa |
|---|---|
| `(?im)` | Bật chế độ case-insensitive (`i`) + multiline (`m`) |
| `^\s*` | Bắt đầu dòng, cho phép khoảng trắng đầu |
| `(Chương\|Phần)` | **Group 1**: Loại cấp (Chương hoặc Phần) |
| `(?:\s+thứ)?` | Hỗ trợ syntax "Phần thứ nhất" (optional) |
| `([a-zA-Z0-9]+\|\d+)` | **Group 2**: Mã số (I, II, 1, 2, hoặc chữ cái La Mã) |
| `[\.:\-]?` | Dấu phân cách tùy chọn (`.`, `:`, `-`) |
| `(.*)$` | **Group 3**: Tiêu đề Chương (phần còn lại của dòng) |

**Ví dụ match:** `Chương I: Quy định chung` → Group1=`Chương`, Group2=`I`, Group3=`Quy định chung`

---

### 1.2.2 Cấp 2: Điều / Mục (`article_pattern`)

```python
re.compile(r"(?im)^\s*(Điều|Mục)\s+(\d+[A-Za-z0-9\/\-]*)\s*[\.:\-]?\s*(.*)$")
```

| Thành phần | Ý nghĩa |
|---|---|
| `(Điều\|Mục)` | **Group 1**: Loại đơn vị pháp lý (Điều hoặc Mục) |
| `(\d+[A-Za-z0-9\/\-]*)` | **Group 2**: Mã số Điều, hỗ trợ `Điều 1a`, `Điều 2/3`, `Điều 5-bis` |
| `(.*)$` | **Group 3**: Tiêu đề Điều |

**Ví dụ match:** `Điều 5. Phạm vi điều chỉnh` → Group1=`Điều`, Group2=`5`, Group3=`Phạm vi điều chỉnh`

---

### 1.2.3 Cấp 3: Khoản (`clause_pattern`) — Regex phức tạp nhất

```python
re.compile(
    r"(?im)^\s*(Khoản\s+\d+[\.:\-]?)\s*(.*)$|"   # Khoản 1
    r"^\s*(\d+(?:\.\d+)*[\.\)])\s*(.*)$|"           # 1., 1.1., 1)
    r"^\s*(\(\d+\))\s*(.*)$|"                        # (1), (2)
    r"^\s*([-+•])\s+(.*)$"                           # -, +, •
)
```

Đây là regex **multi-branch** (sử dụng toán tử `|`) để nhận diện 4 dạng khoản khác nhau:

| Nhánh | Pattern | Ví dụ | Group ref / Group text |
|---|---|---|---|
| Nhánh 1 | `Khoản\s+\d+` | `Khoản 1: Nội dung...` | Group 1 / Group 2 |
| Nhánh 2 | `\d+(?:\.\d+)*[\.\)]` | `1.`, `1.1.`, `1)` | Group 3 / Group 4 |
| Nhánh 3 | `\(\d+\)` | `(1)`, `(2)` | Group 5 / Group 6 |
| Nhánh 4 | `[-+•]` | `- Nội dung`, `+ Bổ sung` | Group 7 / Group 8 |

> [!IMPORTANT]
> Khoản chỉ được nhận diện khi đã có `current_article_ref` (đang ở trong Điều) **VÀ** không ở trong vùng Phụ lục (`not in_appendix`).

---

### 1.2.4 Cấp 4: Điểm (`point_pattern`)

```python
re.compile(r"(?im)^\s*([a-zđ]\s*[\)\.)])\s*(.*)$")
```

| Thành phần | Ý nghĩa |
|---|---|
| `[a-zđ]` | Ký tự đơn (a-z hoặc đ) — điểm trong luật Việt Nam |
| `[\)\.)]` | Dấu đóng `)` hoặc `.` sau ký tự |

**Ví dụ match:** `a) Cơ quan nhà nước...`, `đ. Trường hợp đặc biệt...`

> [!NOTE]
> Điểm chỉ được nhận diện khi đã có `current_active_clause` (đang ở trong Khoản) **VÀ** `not in_appendix`.

---

## 1.3 Hệ thống Regex — Bộ nhận diện Phụ lục & Cấu trúc tự do

### 1.3.1 Nhận diện tiêu đề Phụ lục (`appendix_title_pattern`)

```python
re.compile(
    r"(?im)^\s*("
    r"(?:PHỤ\s+LỤC|PHU\s+LUC)(?:\s+(?:SỐ\s+)?(?:[IVXLCDM]+|\d+)|[A-Z])?(?:\s*[:\-\.]|\s+BAN\s+HÀNH|\s+KÈM\s+THEO)?|"
    r"(?:MẪU|MẪU\s+SỐ|BIỂU\s+MẪU)\s*[A-Za-z0-9\.\-\/]*(?:\s*[:\-\.]|\s+BAN\s+HÀNH|\s+KÈM\s+THEO)?|"
    r"DANH\s+MỤC(?:\s+(?:CHI\s+TIẾT|KÈM\s+THEO|CÁC|DỰ\s+ÁN|TÀI\s+SẢN|VẬT\s+TƯ|HÀNG\s+HÓA|QUỐC\s+GIA|MÃ))?"
    r")\b.*$"
)
```

Regex này có **3 nhánh chính**:

| Nhánh | Mục tiêu nhận diện | Ví dụ |
|---|---|---|
| 1 | PHỤ LỤC kèm số La Mã/Ả Rập | `PHỤ LỤC I`, `PHỤ LỤC SỐ 3`, `PHỤ LỤC A BAN HÀNH` |
| 2 | MẪU / BIỂU MẪU | `MẪU SỐ 01-A`, `BIỂU MẪU KÈM THEO` |
| 3 | DANH MỤC | `DANH MỤC CHI TIẾT`, `DANH MỤC HÀNG HÓA` |

> [!WARNING]
> Có bộ lọc `len(line) < 200` để chống nhận diện lỗi — nếu dòng dài hơn 200 ký tự thì đó là đoạn văn, không phải tiêu đề phụ lục.

---

### 1.3.2 Nhận diện tiêu đề "Hướng dẫn chuyên môn" (`substantive_title_pattern`)

```python
re.compile(
    r"(?im)^\s*(QUY ĐỊNH|QUY CHẾ|PHƯƠNG ÁN|ĐIỀU LỆ|CHƯƠNG TRÌNH|HƯỚNG DẪN|NỘI QUY|KẾ HOẠCH|CHIẾN LƯỢC|ĐỀ ÁN|DỰ ÁN)\b(?!\s*CHUNG\b).*$"
)
```

- Nhận diện các tài liệu chuyên môn kèm theo văn bản chính
- **Negative lookahead** `(?!\s*CHUNG\b)` để loại trừ "QUY ĐỊNH CHUNG" (đây thường là tiêu đề Chương)

---

### 1.3.3 Cấp phân cấp trong Phụ lục

| Pattern | Tương đương | Regex | Ví dụ |
|---|---|---|---|
| `appx_lvl1_pattern` | Điều (Article) | `([IVXLCDM]+\|[A-Z])\s*[\.:‐]` | `I. Giới thiệu`, `A. Phạm vi` |
| `appx_lvl2_pattern` | Khoản (Clause) | `(\d+)\s*[\.:‐]` | `1. Nội dung`, `2. Yêu cầu` |
| `appx_lvl3_pattern` | Điểm (Point) | `(\d+(?:\.\d+)+)\s*[\.:‐]?` | `1.1 Chi tiết`, `1.2.1 Mục nhỏ` |
| `part_lesson_pattern` | Tập/Bài | `(Phần\|Tập\|Bài)\s+([IVXLCDM0-9]+)` | `Bài 1. Tổng quan`, `Phần II. Hướng dẫn` |

---

## 1.4 Hệ thống Regex — Bóc tách Metadata trong văn bản

### 1.4.1 Căn cứ pháp lý (`legal_basis_line_pattern` + `legal_ref_pattern`)

```python
# Dòng bắt đầu bằng "Căn cứ"
legal_basis_line_pattern = re.compile(r"(?im)^\s*căn cứ\b.*$")

# Trích xuất loại văn bản và nội dung tham chiếu
legal_ref_pattern = re.compile(
    r"(?i)\b(Hiến pháp|Bộ luật|Luật|Nghị quyết|Pháp lệnh|Nghị định|Thông tư)\b([^.;\n]*)"
)
```

**Luồng xử lý:**
1. Quét 80 dòng đầu tìm dòng `Căn cứ...`
2. Trong mỗi dòng, dùng `legal_ref_pattern.finditer()` tìm tất cả tham chiếu
3. Cho mỗi tham chiếu, trích xuất: `doc_type`, `doc_number`, `doc_year`, `parent_law_id`
4. Deduplicate bằng `(parent_law_id, doc_title)` set

---

### 1.4.2 Ngày hiệu lực (`effective_date_pattern` + `effective_from_sign_pattern`)

```python
# Ngày cụ thể: "có hiệu lực từ ngày 01/01/2024"
effective_date_pattern = re.compile(
    r"(?i)có\s+hiệu\s+lực\s+(?:từ|kể\s+từ)\s+ngày\s+(\d{1,2})[/\- ](\d{1,2})[/\- ](\d{4})"
)

# Hiệu lực từ ngày ký: "có hiệu lực thi hành kể từ ngày ký"
effective_from_sign_pattern = re.compile(
    r"(?i)(có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày\s+ký|"
    r"chịu\s+trách\s+nhiệm\s+thi\s+hành\s+(?:quyết\s+định|nghị\s+định|thông\s+tư|văn\s+bản)\s+này)"
)
```

**Logic ưu tiên:**
1. Tìm ngày cụ thể → trả dạng `YYYY-MM-DD`
2. Nếu có cụm "từ ngày ký" → trả `promulgation_date`
3. Nếu không tìm thấy → trả chuỗi rỗng

> [!TIP]
> Chỉ quét phần **main body** (trước Phụ lục) để tránh lấy nhầm ngày tháng trong bảng biểu.

---

### 1.4.3 Quan hệ giữa các văn bản (`relationship_pattern`)

```python
relationship_pattern = re.compile(
    r"(sửa đổi(?:,\s*bổ sung)?|bổ sung|thay thế|bãi bỏ|huỷ bỏ)"  # Group 1: Action
    r"(.{0,150}?)"                                                    # Group 2: Scope (non-greedy, max 150 chars)
    r"(?:của\s+)?"                                                    # Bỏ chữ "của" nếu có
    r"(Hiến pháp|Bộ luật|Luật|...|Quyết định|Chỉ thị)"             # Group 3: Doc Type
    r"(?:\s+số)?\s+([0-9]+/[0-9]{4}/[A-Z0-9Đ\-]+|[0-9]+/[A-Z0-9Đ\-]+)", # Group 4: Doc Number
    re.IGNORECASE
)
```

| Group | Nội dung | Ví dụ |
|---|---|---|
| 1 | Hành động (Action) | `sửa đổi, bổ sung`, `thay thế`, `bãi bỏ` |
| 2 | Phạm vi tác động (Scope) | `điều 5 khoản 2` |
| 3 | Loại văn bản bị tác động | `Nghị định`, `Thông tư` |
| 4 | Số hiệu văn bản bị tác động | `142/2024/NĐ-CP` |

**Bộ lọc quan trọng (Strict Validation):**
- **Preamble Filter (Anchor):** Chỉ quét từ `Điều 1` hoặc `QUYẾT ĐỊNH:` trở xuống, bỏ phần Lời dẫn ở trên để tránh bắt nhầm "Căn cứ Luật số..." thành "sửa đổi"
- **Amended Validation:** Nếu hành động là `amended` nhưng không chỉ đích danh Điều/Khoản nào → **loại bỏ** (chống liên kết rác)

---

### 1.4.4 Regex trích xuất tọa độ chính xác (bổ trợ cho Relationship)

```python
extract_article_pattern = re.compile(r"(điều\s+\d+[a-zA-ZđĐ]*)", re.IGNORECASE)
extract_clause_pattern  = re.compile(r"(khoản\s+\d+[a-zA-ZđĐ]*)", re.IGNORECASE)
```

Dùng để bóc ra tất cả `Điều X`, `Khoản Y` từ khối `Scope` + khối `new_text` (dynamic context block).

---

### 1.4.5 Nhận diện Footer / Ký tên (`footer_pattern`)

```python
footer_pattern = re.compile(
    r"(?i)^\s*(nơi\s+nhận|kính\s+gửi)[\:\.]?|"
    r"^\s*(TM\.|KT\.|Q\.|TL\.|TUQ\.)?\s*"
    r"(CHÍNH\s+PHỦ|UBND|...|BỘ\s+TRƯỞNG|CHỦ\s+TỊCH|...)\b"
)
```

- Nhận diện phần ký tên cuối văn bản (Nơi nhận, TM. BỘ TRƯỞNG, v.v.)
- **Mục đích:** Loại bỏ hoàn toàn, không đưa vào chunk

---

### 1.4.6 Nhận diện Điều cuối cùng (`final_article_trigger`)

```python
final_article_trigger = re.compile(
    r"(?i)(có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày|"
    r"chịu\s+trách\s+nhiệm\s+thi\s+hành|"
    r"tổ\s+chức\s+thực\s+hiện)"
)
```

Phát hiện Điều cuối (thường là Điều "Thi hành") bằng các cụm từ đặc trưng.

---

## 1.5 Cơ chế cắt Chunk — FSM chi tiết

### 1.5.1 Cấu trúc bộ nhớ (Buffer System)

```text
+---------------------------------------------------------------------------------+
| BUFFER SYSTEM - BỘ NHỚ TRẠNG THÁI                                               |
|                                                                                 |
| current_chapter (string)                                                        |
|        |                                                                        |
|        +---> current_article_ref (string - 'Điều 5')                            |
|                     |                                                           |
|                     +---> current_article_preamble (list[str] - Lời dẫn Điều)   |
|                     |                                                           |
|                     +---> current_clauses_buffer (list[dict] - Các Khoản)       |
|                     |                                                           |
|                     +---> current_active_clause (dict - Khoản đang xây)         |
|                                     +---> clause.ref (string)                   |
|                                     +---> clause.text (string)                  |
|                                     +---> clause.points (list[str])             |
+---------------------------------------------------------------------------------+

+-------------------------------+
| CỜ TRẠNG THÁI                 |
|                               |
| * in_appendix (bool)          |
| * in_table (bool)             |
+-------------------------------+
```

> [!NOTE]
> **Giải thích cơ chế Buffer System trong mã nguồn (`legal_rag_build_qdrant_2.ipynb`):**
> Trong hàm `process_document()`, bộ nhớ hoạt động theo các bước:
> - **Bước 1 (Lưu vết cấp cao):** Khi quét thấy một Chương/Phần mới, tên của nó tự động lưu đè lên bộ nhớ tạm `current_chapter`.
> - **Bước 2 (Mở bộ đệm trung gian):** Với Điều con, `current_article_ref` sẽ lưu lại nhãn "Điều X", trong khi `current_article_preamble` sẽ hứng các đoạn text lời dẫn vào Điều này.
> - **Bước 3 (Xây dựng Khoản/Điểm):** Khi bắt vào Khoản, biến `current_active_clause` là một dict sẽ hứng ID Khoản và mạo danh một chuỗi list riêng `clause.points` (để hứng các Điểm nhỏ).
> - **Bước 4 (Đóng băng):** Khi một thẻ cấu trúc mới nhảy tới, dữ liệu con đang rải rác bên trong sẽ được nạp gộp mảng tổng `current_clauses_buffer` chờ duyệt xả, giúp breadcrumb không bao giờ bị hổng.

### 1.5.2 Giới hạn kích thước

| Hằng số | Giá trị | Mục đích |
|---|---|---|
| `CHUNK_LIMIT` | 1200 chars | Ngưỡng tổng buffer Khoản + Điểm. Vượt qua → flush |
| `TEXT_LIMIT` | 1500 chars | Ngưỡng text tự do (preamble, đoạn nối tiếp). Vượt qua → flush |
| Bảng biểu: rows | 20 dòng | Ngưỡng số dòng bảng tối đa trong 1 chunk |
| Bảng biểu: size | 3000 chars | Ngưỡng kích thước bảng tối đa |
| Min letters | 30 chữ cái | Chunk có < 30 ký tự alphabet → **loại bỏ** (rác) |
| Lời dẫn ngắn | 500 chars | Chunk "Lời dẫn" thuần túy < 500 chars → loại bỏ |

### 1.5.3 Cơ chế "Tiếp theo" (Continuation Chunks)

Khi một Khoản quá dài vượt `CHUNK_LIMIT`:

1. Flush chunk hiện tại
2. Tạo chunk nối tiếp với `ref = "Khoản X (tiếp theo)"`
3. Text mới bắt đầu bằng `[Khoản X tiếp theo]` (Context Injection)

Điều này giúp LLM biết chunk đang đọc là phần nối tiếp của Khoản nào.

### 1.5.4 Cơ chế lọc rác

| Bộ lọc | Điều kiện | Hành động |
|---|---|---|
| Min letters | `re.sub(r'[\W_0-9]', '', text)` < 30 | Bỏ qua chunk |
| Tiêu đề mồ côi | Không có article_ref, toàn VIẾT HOA | Bỏ qua chunk |
| Mào đầu ngắn | Không có article_ref/clause, < 500 chars | Bỏ qua chunk |
| Header zone | Dòng < 50, chứa "ĐỘC LẬP"/"TỰ DO" | Skip dòng |
| Footer | Match `footer_pattern` | Skip dòng |
| Căn cứ rải rác | `line.lower().startswith("căn cứ")` khi chưa vào nội dung | Skip dòng |

### 1.5.5 Cơ chế xử lý Bảng biểu

```text
[Dòng có >= 2 ký tự '|' ?]
   |
   |---(CÓ)---> [in_table đang true?]
   |                 |
   |                 |---(KHÔNG)---> [flush_article() hiện tại] ---> [Bắt đầu table mới (table_header = [line])]
   |                 |
   |                 |---(CÓ)------> [Dòng toàn -/:| ?]
   |                                     |
   |                                     |---(CÓ)------> [Thêm vào table_header]
   |                                     |
   |                                     |---(KHÔNG)---> [rows >= 20 hoặc size > 3000?]
   |                                                          |
   |                                                          |---(CÓ)------> [flush_table() cũ] ---> [table_rows = [line]]
   |                                                          |
   |                                                          |---(KHÔNG)---> [table_rows.append(line)]
   |
   |---(KHÔNG)--> [in_table đang true?]
                     |
                     |---(CÓ)------> [flush_table() nếu có rows] ---> [Reset in_table = false]
```

> [!NOTE]
> **Giải thích cơ chế bắt bảng trong mã nguồn (`legal_rag_build_qdrant_2.ipynb`):**
> Quá trình dò bảng tuân thủ trình tự sau:
> - **Bước 1 (Nhận diện row đầu tiên):** Nếu dòng (`line`) đang xét chứa `>= 2` ký hiệu `|`, cờ khởi dựng `in_table` sẽ bật thành True.
> - **Bước 2 (Chuyển giao):** Trước khi tạo một bảng mới hoàn toàn, nếu đang có buffer Điều/Khoản trống lửng lơ trên đó, code gọi `flush_article()` để chốt khối vùng text cũ.
> - **Bước 3 (Gom hàng):** Tất cả dòng thỏa mãn điều kiện bảng tiếp sau đó sẽ liên tục bị append vào array `table_header` hoặc phần thân `table_rows`.
> - **Bước 4 (Giới hạn tràn - Flush):** Code thiết lập màng lọc tự động kiểm tra `len(table_rows) >= 20` hoặc size > 3000 chars, `flush_table()` buộc phải xả bảng ngắt quãng rồi mới buffer tiếp phần bảng còn lại.

---

## 1.6 Hàm `flush_article()` — Đóng gói Chunk cuối cùng

### 1.6.1 Xây dựng Breadcrumb

Breadcrumb được tạo bằng cách nối:
```
Chapter > Article > Clause(s) > Point(s)
```

**Ví dụ:** `Chương IV > Điều 8 > Khoản 6, 7.`

### 1.6.2 Hàm `group_refs()` — Gom nhóm tham chiếu

Hàm tự động nhận diện dạng tham chiếu và thêm tiền tố phù hợp:

| Input | Dạng nhận diện | Output |
|---|---|---|
| `["Khoản 1", "Khoản 3"]` | Match `khoản\s+(.+)` | `Khoản 1, 3` |
| `["a)", "b)", "c)"]` | Match `[^()]+\)` | `Điểm a, b, c)` |
| `["1.", "2.", "3."]` | Match `[^.]+\.` | `Khoản 1, 2, 3.` |
| `["(1)", "(2)"]` | Match `\([^()]+\)` | `Khoản (1, 2)` |

### 1.6.3 Cấu trúc Output cho mỗi chunk

Mỗi chunk output là một dictionary gồm 4 trường:

```python
{
    "chunk_id":       "doc_id::article::c1",       # UUID duy nhất
    "chunk_text":     "Văn bản: ... \n Nội dung: ...", # Full text cho LLM đọc
    "text_to_embed":  "[doc_number] breadcrumb\n...", # Text tinh gọn để embed
    "qdrant_metadata": { ... },                      # Payload nhẹ cho Qdrant
    "neo4j_metadata":  { ... }                       # Payload đầy đủ cho Neo4j
}
```

> [!IMPORTANT]
> **Tách biệt `chunk_text` vs `text_to_embed`:** `chunk_text` chứa metadata header (Văn bản, Lĩnh vực, Điều khoản) để LLM đọc hiểu ngữ cảnh. `text_to_embed` chỉ chứa `[doc_number] breadcrumb` + nội dung gốc để embedding không bị loãng ngữ nghĩa bởi metadata lặp lại.

---

## 1.7 Hàm tiện ích xử lý chuỗi (Static Helpers)

| Hàm | Mục đích | Logic |
|---|---|---|
| `compact_whitespace()` | Dọn khoảng trắng | `re.sub(r"[ \t]+", " ", text).strip()` |
| `_slugify()` | Tạo slug URL-safe | Lowercase + thay ký tự đặc biệt bằng `-` |
| `_canonical_doc_type()` | Chuẩn hóa loại VB | Map "nghị định"→"decree", "thông tư"→"circular" |
| `_extract_year()` | Trích năm từ text | `re.search(r"\b(19\|20)\d{2}\b", text)` |
| `_extract_doc_number()` | Trích số hiệu | Regex: `\d+\/\d+(?:\/[A-Z0-9Đ\-]+)?` |
| `_parse_signer()` | Parse tên + ID ký | Tách `"Nguyễn Văn A:12345"` → `("Nguyễn Văn A", 12345)` |
| `normalize_doc_key()` | Chuẩn hóa key tra cứu | Xóa whitespace, `.`, `-`, `/` → UPPER |

---

## 1.8 Hàm `_extract_exact_article()` — Trích nội dung Điều chính xác

Mục đích: Khi phát hiện văn bản A sửa đổi Điều 5 của văn bản B, hàm này sẽ trích **đúng và đủ** nội dung Điều 5 từ văn bản B (`old_text`).

**Thuật toán:**
1. Tạo regex động từ `article_name` (chia thành từ, nối bằng `\s+` để bất chấp số khoảng trắng)
2. Tìm vị trí bắt đầu Điều target
3. Tìm vị trí kết thúc = Điều/Chương/Phần tiếp theo
4. Cắt + trả tối đa 1500 ký tự

---

# Phần 2: Kỹ thuật dựng QdrantDB

## 2.1 Kiến trúc tổng quan

```text
[100 Văn bản từ HuggingFace]
           |
           v
[AdvancedLegalChunker process_document()] (Chunking - CPU)
           |
           v
[BGE-M3 Dense 1024d + Sparse] (Embedding - GPU)
           |
           v
[Collection: legal_rag_3sectors_test] (Qdrant Cloud)
   |-- Vector: dense (1024d, Cosine)
   |-- Vector: sparse (BM25-like)
   +-- Payload: metadata
```

> [!NOTE]
> **Giải thích luồng Data đến đám mây Qdrant (`legal_rag_build_qdrant_2.ipynb`):**
> - **Bước 1 (Phân thân dữ liệu):** Lấy Dataset từ HuggingFace qua lọc CSV, đưa danh sách đi qua class `AdvancedLegalChunker` để cưa tách thành các chunk nhỏ độc lập mang trọn vẹn ngữ nghĩa.
> - **Bước 2 (Mã hoá Vector):** Bơm list string qua encoder `BAAI/bge-m3` bằng sức mạnh của GPU. Kết quả sinh ra cặp bài trùng: 1 array Dense Vector (1024 chiều tọa độ cosine không gian) và 1 array Sparse Vector (Bao từ vựng dạng BM25-like).
> - **Bước 3 (Cấu thành Payload):** Đóng gói UUID, hai trường Vector trên và siêu dữ liệu Metadata thật gọn gàng vào Payload PointStruct (Loại bỏ các trường metadata dư thừa đồ sộ để tránh lãng phí RAM của VectorDB).
> - **Bước 4 (Upsert):** Bắn thẳng gói dữ liệu đã nén qua đường HTTP API để lưu lại trong Collection `legal_rag_3sectors_test` trên Qdrant Cloud.

## 2.2 Mô hình Embedding: BGE-M3 Hybrid

### 2.2.1 Class `LocalBGEHybridEncoder`

| Thuộc tính | Giá trị |
|---|---|
| Model | `BAAI/bge-m3` |
| Dense dimension | **1024** |
| Sparse encoding | Lexical weights (BM25-like) |
| FP16 | Bật khi chạy CUDA |
| Max length | 2048 tokens |
| Batch size | 256 (GPU) / 16 (CPU) |
| ColBERT | **Tắt** (`return_colbert_vecs=False`) |

### 2.2.2 Phương thức `encode_hybrid()`

```python
def encode_hybrid(self, texts, batch_size=16):
    out = self.model.encode(
        texts,
        batch_size=256 if cuda else batch_size,
        max_length=2048,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    # Trả về: (List[List[float]], List[SparseVector])
```

### 2.2.3 Chuyển đổi Sparse Vector

```python
@staticmethod
def _to_sparse_vector(weights: Dict[str, float]) -> models.SparseVector:
    # Lọc bỏ giá trị 0.0
    # Ép key sang int, value sang float
    # Sắp xếp theo index tăng dần
    # Trả về SparseVector(indices=[], values=[])
```

## 2.3 Cấu hình Collection Qdrant

```python
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=1024, distance=Distance.COSINE, on_disk=True)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
    },
)
```

| Config | Giá trị | Lý do |
|---|---|---|
| `on_disk=True` (dense) | Lưu vector trên đĩa | Tránh tràn RAM khi collection lớn |
| `on_disk=True` (sparse) | Lưu sparse index trên đĩa | Tương tự, tránh tràn RAM |
| Distance metric | `COSINE` | Phù hợp với BGE-M3 normalized vectors |

## 2.4 Qdrant Payload Schema

Mỗi point được upsert lên Qdrant với payload **siêu nhẹ** (chỉ giữ những gì cần cho filtering & display):

```json
{
    "chunk_id":        "doc_id::article::c1",
    "document_id":     "123456",
    "document_number": "142/2024/NĐ-CP",
    "year":            "2024",
    "legal_sectors":   ["Thể thao - Y tế"],
    "is_table":        false,
    "breadcrumb":      "Chương I > Điều 1 > Khoản 1.",
    "chunk_text":      "Văn bản: 142/2024/NĐ-CP - ...\nNội dung:\n...",
    "title":           "Nghị định 142/2024/NĐ-CP về..."
}
```

> [!TIP]
> Payload Qdrant **không chứa** `legal_basis_refs`, `amended_refs`, `document_toc`, `signer_name`, v.v. — những metadata nặng này chỉ đẩy vào Neo4j.

## 2.5 Pipeline Upsert — Batch Processing

### 2.5.1 Cấu hình Batch

| Hằng số | Giá trị | Mục đích |
|---|---|---|
| `DOC_BATCH_SIZE` | 250 | Số văn bản xử lý mỗi lần lặp |
| `EMBED_BATCH_SIZE` | 128 | Batch size cho model embedding |
| `UPSERT_BATCH_SIZE` | 512 | Batch size cho Qdrant upsert |

### 2.5.2 Luồng xử lý chính

```text
+---> [ Lấy batch 250 văn bản ]
|             |
|             v
|     [ 1. CHUNKING - chunker.process_document() -> batch_chunks ]
|             |
|             v
|     [ 2. EMBEDDING - hybrid_encoder.encode_hybrid() -> dense_vecs, sparse_vecs ]
|             |
|             v
|     [ 3. BUILD POINTS - Tạo PointStruct: id=UUID5, vector={dense,sparse}, payload ]
|             |
|             v
|     [ 4. UPSERT - qdrant_client.upsert() batch 512 points/lần ]
|             |
|             v
|     [ 5. NEO4J PUSH - build_neo4j(driver, batch_chunks) ]
|             |
|             v
+---- [ gc.collect() - Dọn RAM ]
```

> [!NOTE]
> **Giải thích Pipeline Upsert theo lô (`legal_rag_build_qdrant_2.ipynb`):**
> - **Bước 1 (Chia Batch):** Dòng loop While ngoài cùng phân chia mảng chứa 100 tài liệu làm các Batch nhỏ, quy định cứng ở ngưỡng `DOC_BATCH_SIZE = 250` (Tương đương 250 file trọn vẹn).
> - **Bước 2 (Flatten chunk):** Ép FSM Chunker sinh ra chuỗi nối tiếp hàng ngàn mảng List[dict] để dàn trải ngữ cảnh thành `batch_chunks`.
> - **Bước 3 (Encode):** GPU tính toán list text này để sinh Vector Arrays (Dense, Sparse).
> - **Bước 4 (Đẩy song song Cloud):** Tiêm thẳng batch Vector lên Qdrant thông qua module client SDK. Tiếp ngay sau đó, kéo array chunk kia ném vào hàm `build_neo4j()` kích hoạt query Cypher để đẩy lên mạn GraphDB Aura.
> - **Bước 5 (Clear RAM/VRAM):** Cuối lô `gc.collect()` ép xóa các instance tensor rác và giải phóng bộ nhớ của Node đồ thị. Tránh sự cố sập memory leak cực kỳ nan giải trong môi trường Colab giới hạn 15GB VRAM.

### 2.5.3 UUID Generation

```python
qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
```

UUID5 đảm bảo:
- **Deterministic**: Cùng `chunk_id` → cùng UUID → tránh duplicate khi chạy lại
- **Unique**: Khác `chunk_id` → khác UUID

### 2.5.4 Chế độ Preview (Mode 2)

Nếu `PROCEED_UPSERT = False`:
- Chỉ lấy 1 văn bản đầu tiên
- Embed chunk thứ 3 (index 2) để tránh phần mào đầu rỗng
- In ra JSON payload mẫu cho cả Qdrant và Neo4j
- **Không ghi DB**

## 2.6 Global Doc Lookup — Truy xuất chéo văn bản

Trước vòng lặp chính, hệ thống xây 2 bảng tra cứu toàn cục:

| Lookup | Key | Value | Kích thước |
|---|---|---|---|
| `meta_by_docnum_lookup` | `normalize_doc_key(doc_number)` | Row metadata từ HF | 238,206 VB |
| `global_doc_lookup` | `doc_number.upper()` | Full text content | 240,119 VB |

**Mục đích:**
- `meta_by_docnum_lookup`: Dùng cho `enrich_reference_nodes()` — bổ sung metadata cho nốt tham chiếu trên Neo4j
- `global_doc_lookup`: Dùng cho `_extract_exact_article()` — lấy `old_text` khi xung đột

## 2.7 Thống kê Pipeline

```json
{
    "documents": 100,
    "chunks": 3352,
    "chunk_seconds": 1.20,
    "embed_seconds": 69.17,
    "point_build_seconds": 0.15,
    "upsert_seconds": 9.76
}
```

| Metric | Giá trị | Ghi chú |
|---|---|---|
| Trung bình chunks/doc | 33.52 | Khá cao do nhiều bảng biểu |
| Tỷ lệ bảng biểu | 40.39% (1354/3352) | ⚠️ Cao — có thể làm loãng search |
| Tỷ lệ nội dung chính | 59.61% (1998/3352) | |
| Thời gian embedding | 69.17s | Bottleneck chính (GPU) |
| Thời gian upsert | 9.76s | Nhanh do cloud Qdrant |

---

# Phần 3: Kỹ thuật dựng Neo4jDB

## 3.1 Schema Đồ thị (Graph Schema)

```text
[ THỰC THỂ CHÍNH ]
 * Document (id, document_number, title, promulgation_date, effective_date, year, url, doc_status, document_toc)
 * Article  (id, name, chapter_ref, qdrant_id, text, is_table)
 * Clause   (id, name, qdrant_id, text, is_table)
 * Chunk    (id, chunk_index, text, is_table, qdrant_id)

[ THỰC THỂ METADATA ]
 * Authority (name)
 * Signer    (name, signer_id)
 * LegalType (name)
 * Sector    (name)

[ QUAN HỆ ]
 Document --- BASED_ON / AMENDS / REPLACES / REPEALS ---> Document (tham chiếu)
 
 Article  --- BELONGS_TO ---> Document
 Clause   --- PART_OF ------> Article
 Chunk    --- PART_OF ------> Clause / Article
 Chunk    --- BELONGS_TO ---> Document
 
 Authority -- ISSUED -------> Document
 Signer   --- SIGNED -------> Document
 Document --- HAS_TYPE -----> LegalType
 Document --- HAS_SECTOR ---> Sector
```

> [!NOTE]
> **Giải thích kiến trúc rẽ nhánh mạng Neo4j Schema (`legal_rag_build_qdrant_2.ipynb`):**
> - **Bước 1 (Xây xương sống Hành chính):** Nhóm các lệnh khởi tạo sẽ chia Đồ thị ra, Core nhất là văn bản `Document`. Từ `Document` lại bóc ra nhánh phân cấp đi xuống các phần tử chứa Text là: `Document -> Article -> Clause -> Chunk`. 
> - **Bước 2 (Chỏm râu Metadata):** Tạo thêm các node vệ tinh bao quanh như Cơ quan ban hành `Authority`, Người ký (`Signer`), hay mảng Lĩnh vực `Sector`. Móc nối Edge chĩa vào thân `Document`.
> - **Bước 3 (Mạng Đồ Thị Xung Đột):** Khi tài liệu A cập nhật tài liệu B, đường viền quan hệ sẽ nối thẳng `Document A` qua `Document B` mang nhãn `AMENDS / REPLACES` cùng với trích xuất văn bản đối sánh.
> - **Bước 4 (Cầu nối Data Hybrid):** Neo4j không chứa vector đồ sộ. Nó chỉ nhận trường khóa ngoại `qdrant_id` đại diện đính lên các block Node lá (như Article, Clause). Khi cần RAG sẽ trượt song song cả hai nền tảng.

## 3.2 Hệ thống Constraints & Index

```python
init_neo4j_constraints(driver)
```

| Constraint | Loại | Mục đích |
|---|---|---|
| `Document.id` | UNIQUE | Đảm bảo không trùng Document |
| `Article.id` | UNIQUE | Đảm bảo không trùng Article |
| `Clause.id` | UNIQUE | Đảm bảo không trùng Clause |
| `Chunk.id` | UNIQUE | Đảm bảo không trùng Chunk |
| `Document.document_number` | INDEX | Tra cứu nhanh theo số hiệu (không UNIQUE vì có thể có nhiều VB "N/A") |
| `Authority.name` | UNIQUE | |
| `Sector.name` | UNIQUE | |
| `LegalType.name` | UNIQUE | |

## 3.3 Đồ thị Động (Dynamic Tree) — Thuật toán phân loại Leaf Level

Đây là **trái tim kỹ thuật** của Neo4j pipeline. Thay vì luôn tạo node Chunk, hệ thống quyết định **mức nào là leaf** (lá) để gắn `qdrant_id` và `text`:

```text
[Chunk đầu vào] ---> Có article_ref?
                         |
  +----------------------+--------------------+
  | (KHÔNG)                                   | (CÓ)
  v                                           v
[🔴 Chunk mồ côi (is_chunk_leaf)]          Có clause_ref?
                                              |
      +---------------------------------------+----------------------------------+
      | (KHÔNG)                                                                  | (CÓ)
      v                                                                          v
Node Article đã có text?                                        Clause '(tiếp theo)' hoặc có '[' ?
      |                                                                          |
      |---(KHÔNG)---> [🟢 Article là Leaf (is_article_leaf)]                       |
      |               (Gán text trực tiếp vào Article)                           |
      |                                                                          |---(CÓ)------> [🔴 Chunk mảnh (is_chunk_leaf)]
      |---(CÓ hoặc                                                               |               (Tạo node Chunk -> PART_OF -> Clause)
           is_table)                                                             |
              |                                                                  |---(KHÔNG)---> Node Clause đã có text?
              v                                                                                     |
    [🔴 Chunk con (is_chunk_leaf)]                                                                  |---(KHÔNG)---> [🟡 Clause là Leaf (is_clause_leaf)]
    (Tạo node Chunk -> PART_OF -> Article)                                                          |               (Gán text trực tiếp vào Clause)
                                                                                                    |
                                                                                                    |---(CÓ hoặc
                                                                                                         is_table)---> [🔴 Chunk mảnh (is_chunk_leaf)]
```

> [!NOTE]
> **Giải thích Giải thuật cây động ghim nốt lá - "Leaf Level" (`legal_rag_build_qdrant_2.ipynb`):**
> Luồng build Cypher trong Neo4j luôn tự động phân loại để neo Text + ID đúng chỗ:
> - **Bước 1 (Dò Article Level):** Nếu chunk này trực thuộc một Điều gốc, và cái Điều này chưa từng nạp Text (Node Article trắng dữ liệu), code rẽ nhánh gán cứng text vào Article và gắn lá cờ `is_article_leaf`. Mức sâu dừng ở đây.
> - **Bước 2 (Dò Clause Level):** Tương tự bước trên nhưng sâu hơn, nếu chunk thụt dật xuống cấp Khoản, mà Khoản đó còn trống, code sẽ nạp text vào Node Clause và gắn cờ `is_clause_leaf`.
> - **Bước 3 (Mảnh vỡ tái sinh - Chunk con):** Nếu Article hoặc Clause ấy đã từng nhận text trước đó (ví dụ bị cắt nhỏ ra làm 2 đoạn do quá 1500 chars, đoạn sau có chữ `[tiếp theo]`...), tất cả chunk dư địa này sẽ phải gom nhóm làm node phụ con `Chunk` rồi gắn quan hệ mỏ neo `PART_OF` trỏ ngược lên Node Article/Clause cha đã khởi tạo.
> - **Bước 4 (Mồ Côi):** Nếu dòng chunk chẳng thuộc Điều khoản nào (lời mở chặng, căn cứ), nó làm một node `Chunk` mồ côi phi thẳng `BELONGS_TO` lên rễ trên cùng là `Document`.

**3 Kịch bản (Scenario):**

| Kịch bản | Điều kiện | Leaf Node | Ví dụ |
|---|---|---|---|
| **A** | Điều không có Khoản | `Article` | Điều 1 chỉ có 3 dòng text |
| **B** | Khoản nguyên vẹn (không bị cắt) | `Clause` | Khoản 1. trong Điều 5 |
| **C** | Khoản bị tách mảnh hoặc bảng biểu | `Chunk` | "Khoản 2 (tiếp theo)" |

> [!IMPORTANT]
> Cơ chế `seen_nodes` (set) đảm bảo:
> - Mỗi Article chỉ nhận `qdrant_id` + `text` **một lần** (lần đầu tiên)
> - Nếu Article đã có text → chunk tiếp theo ép thành `Chunk` con
> - Tương tự cho Clause

## 3.4 Cypher Query — Phân tích chi tiết

### 3.4.1 Merge Document & Metadata (Nhóm A–D, F)

```cypher
// A. Tạo/Cập nhật Document
MERGE (d:Document {id: row.doc_id})
SET d.document_number = row.doc_num,
    d.title = row.title,
    d.promulgation_date = row.p_date,
    d.effective_date = row.eff_date,
    d.year = row.year,
    d.url = row.url,
    d.doc_status = row.doc_status,
    d.document_toc = row.doc_toc,
    d.is_full_text = true

// B. Authority → ISSUED → Document
MERGE (a:Authority {name: row.auth_name})
MERGE (a)-[:ISSUED]->(d)

// C. Signer → SIGNED → Document
MERGE (s:Signer {name: row.signer_name})
MERGE (s)-[:SIGNED]->(d)

// D. Document → HAS_TYPE → LegalType
MERGE (lt:LegalType {name: row.l_type})
MERGE (d)-[:HAS_TYPE]->(lt)

// F. Document → HAS_SECTOR → Sector
FOREACH (sec_name IN row.sectors |
    MERGE (sec:Sector {name: sec_name})
    MERGE (d)-[:HAS_SECTOR]->(sec)
)
```

### 3.4.2 Cấu trúc Đồ thị Động (Nhóm E — Dynamic Tree)

```cypher
// Nhánh 1: XỬ LÝ ARTICLE
FOREACH (... WHEN row.art_ref IS NOT NULL ...)
    MERGE (art:Article {id: row.doc_id + '_' + row.art_ref})
    ON CREATE SET art.name = row.art_ref, art.chapter_ref = row.chap_ref
    MERGE (art)-[:BELONGS_TO]->(d)
    
    // Nếu Article là leaf → gán qdrant_id + text
    FOREACH (... WHEN row.is_art_leaf ...)
        SET art.qdrant_id = row.chunk_id,
            art.text = row.text
    )

// Nhánh 2: XỬ LÝ CLAUSE
FOREACH (... WHEN row.base_cl_ref IS NOT NULL ...)
    MERGE (cl:Clause {id: ...})
    MERGE (cl)-[:PART_OF]->(art)
    
    // Nếu Clause là leaf → gán qdrant_id + text
    FOREACH (... WHEN row.is_cl_leaf ...)
        SET cl.qdrant_id = row.chunk_id, cl.text = row.text
    )

// Nhánh 3: XỬ LÝ CHUNK MẢNH
FOREACH (... WHEN row.is_chk_leaf ...)
    MERGE (c:Chunk {id: row.chunk_id})
    SET c.text = row.text, c.qdrant_id = row.chunk_id
    
    // Nếu con của Clause → PART_OF Clause
    // Nếu con của Article → PART_OF Article  
    // Nếu mồ côi → BELONGS_TO Document
)
```

### 3.4.3 Quản lý Xung đột (Nhóm G–J)

| Nhóm | Relationship | Thuộc tính Edge | Tự động đánh dấu |
|---|---|---|---|
| **G** | `BASED_ON` | _(không có thuộc tính edge)_ | – |
| **H** | `AMENDS` | `target_article`, `target_clause`, `is_entire_doc`, `old_text`, `new_text` | – |
| **I** | `REPLACES` | `target_article`, `target_clause`, `is_entire_doc`, `old_text`, `new_text` | Nếu `is_entire_doc=true` → SET `doc_status='Hết hiệu lực'` |
| **J** | `REPEALS` | `target_article`, `target_clause`, `is_entire_doc`, `old_text`, `new_text` | Nếu `is_entire_doc=true` → SET `doc_status='Hết hiệu lực'` |

**Ví dụ Edge AMENDS:**
```json
{
    "target_article": "Điều 5, Điều 7",
    "target_clause": "Khoản 2",
    "is_entire_doc": false,
    "old_text": "[Nội dung Điều 5 gốc từ global_doc_lookup]",
    "new_text": "[Nội dung sửa đổi mới, lấy dynamic context block ~1500 chars]"
}
```

## 3.5 Enrichment — Bổ sung metadata cho Node tham chiếu

Hàm `enrich_reference_nodes()` chạy **trước** `build_neo4j()`:

```text
[Scan batch_chunks] ---> [Gom doc_number từ legal_basis_refs + amended/replaced/repealed_refs]
                                               |
                                               v
                                    [Tra cứu meta_by_docnum_lookup]
                                               |
                                               v
                          [MERGE (p:Document) WHERE p.title IS NULL 
                          SET p.title, p.url, p.promulgation_date, ...]
```

> [!NOTE]
> **Giải thích thuật toán cập nhật vòng đệm Enrichment (`legal_rag_build_qdrant_2.ipynb`):**
> Để tránh tình trạng khi Văn bản A trỏ tới Văn bản B nhưng trên Neo4j văn bản B bị khuyết thông tin:
> - **Bước 1 (Tìm dấu vết Quan hệ):** Khi lô Chunk duyệt thấy các tham chiếu (nằm trong `legal_basis_refs, amended_refs`...), thuật toán chủ động sinh ra node Document cho Văn Bản B mang số hiệu tương ứng.
> - **Bước 2 (Lookup Dictionary offline):** Neo4j quét list `doc_number` ảo này móc vào từ điển Python tĩnh `meta_by_docnum_lookup` (thứ đang buffer sẵn 238K+ danh mục tải ở cache HuggingFace).
> - **Bước 3 (Lấp vùng tối):** Gọi script Cypher phụ chạy trước dùng lệnh `MERGE (Node Ảo)` rồi `SET` đè các thuộc tính bóc ra từ Dictionary (bao gồm Title, URL, Effective Date...) dán lên Node Ảo.
> - **Bước 4 (Kết quả):** Đồ thị RAG hoàn hảo 100%. Node B lúc nãy trống lốc thì giờ đã đầy đủ giao diện, dù bản gốc của nó không nằm trong tệp CSV nạp kho ban đầu đi chăng nữa.

**Mục đích:** Khi văn bản A tham chiếu văn bản B (qua số hiệu), node Document B có thể chỉ là "nốt ảo" (chỉ có `document_number`). Hàm này bổ sung `title`, `url`, `promulgation_date`, v.v. cho nốt ảo bằng cách tra cứu từ toàn bộ CSDL HuggingFace (238,206 văn bản).

> [!NOTE]
> Chỉ cập nhật nếu nốt đang thiếu dữ liệu: `WHERE p.title IS NULL OR p.title = '' OR p.id STARTS WITH 'REF_'`

## 3.6 Neo4j Payload Schema

Mỗi chunk gửi lên Neo4j chứa payload **đầy đủ** (nặng hơn Qdrant nhiều lần):

```json
{
    "document_id": "123456",
    "chunk_index": 3,
    "document_number": "142/2024/NĐ-CP",
    "title": "Nghị định 142/2024/NĐ-CP...",
    "legal_type": "Nghị định",
    "legal_sectors": ["Thể thao - Y tế"],
    "issuing_authority": "Chính phủ",
    "signer_name": "Nguyễn Văn A",
    "signer_id": 12345,
    "url": "https://...",
    "promulgation_date": "2024-06-15",
    "effective_date": "2024-08-01",
    "is_active": true,
    "chapter_ref": "Chương I",
    "article_ref": "Điều 5",
    "clause_ref": "Khoản 1, 2.",
    "is_table": false,
    "reference_citation": "142/2024/NĐ-CP | Chương I | Điều 5 | Khoản 1, 2.",
    "chunk_text": "...",
    "document_toc": "Chương I: Quy định chung\n  Điều 1. Phạm vi...\n  ...",
    "legal_basis_refs": [{"doc_type": "law", "doc_number": "..."}],
    "amended_refs": [...],
    "replaced_refs": [...],
    "repealed_refs": [...]
}
```

> [!IMPORTANT]
> `legal_basis_refs`, `amended_refs`, `replaced_refs`, `repealed_refs` chỉ được gắn vào **chunk đầu tiên** (`chunk_index == 1`) của mỗi văn bản. Các chunk sau đều nhận `[]` để tránh trùng lặp quan hệ trên đồ thị.

## 3.7 Tổng kết so sánh Qdrant vs Neo4j Payload

| Thuộc tính | Qdrant | Neo4j |
|---|---|---|
| `chunk_id` | ✅ | ✅ (dùng làm `qdrant_id` liên kết) |
| `document_id` | ✅ | ✅ |
| `document_number` | ✅ | ✅ |
| `year` | ✅ | ✅ |
| `legal_sectors` | ✅ | ✅ |
| `is_table` | ✅ | ✅ |
| `breadcrumb` | ✅ | – (dùng `reference_citation`) |
| `chunk_text` | ✅ | ✅ |
| `title` | ✅ | ✅ |
| `legal_type` | ❌ | ✅ |
| `issuing_authority` | ❌ | ✅ |
| `signer_name` / `signer_id` | ❌ | ✅ |
| `promulgation_date` / `effective_date` | ❌ | ✅ |
| `chapter_ref` / `article_ref` / `clause_ref` | ❌ | ✅ |
| `legal_basis_refs` | ❌ | ✅ (chunk 1 only) |
| `amended_refs` / `replaced_refs` / `repealed_refs` | ❌ | ✅ (chunk 1 only) |
| `document_toc` | ❌ | ✅ |
| `url` | ❌ | ✅ |

---

## Phụ lục: Data Flow tổng quan end-to-end

```text
+-----------------------------------------------------------+
| 1. DATA SOURCE                                            |
|  * 🤗 HuggingFace Dataset (nhn309261/vietnamese-legal-docs) |
|  * 📄 CSV Target IDs (metadata_the_thao_y_te_100.csv)       |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
| 2. DATA LOADING                                           |
|  * Load metadata + content                                |
|  * Filter by CSV IDs                                      |
|  * Chuẩn hóa columns                                      |
+-----------------------------------------------------------+
                              |
                              v
+-----------------------------------------------------------+
| 4. CHUNKING                                               |         +---------------------------------------+
|  * AdvancedLegalChunker                                   |<--------| 3. GLOBAL LOOKUPS                     |
|  * FSM duyệt từng dòng                                    |         |  * meta_by_docnum_lookup (238K VB)    |
|  * -> 3,352 chunks                                        |<--------|  * global_doc_lookup (240K full text) |
+-----------------------------------------------------------+         +---------------------------------------+
                |                           |                                         |
                |                           |                                         | (vào Neo4j)
                v                           +------------------------+                |
+-------------------------------+                                    |                |
| 5. EMBEDDING                  |                                    |                |
|  * BGE-M3 Hybrid              |                                    |                |
|  * Dense: 1024d               |                                    |                |
|  * Sparse: Lexical            |                                    v                v
+-------------------------------+                   +-----------------------------------------------+
                |                                   | 🔵 Neo4j Aura                                 |
                v                                   |  * Payload đầy đủ                             |
+-------------------------------+                   |  * Dynamic Tree + Conflict Graph              |
| ☁️ Qdrant Cloud                |                   |  * 6.77s push                                 |
|  * Payload nhẹ                |                   +-----------------------------------------------+
|  * 9.76s upsert               |
+-------------------------------+
```

> [!NOTE]
> **Giải thích Tổng thể luồng E2E Pipeline RAG (`legal_rag_build_qdrant_2.ipynb`):**
> - **Bước 1 (Mạp Meta Dataset):** Tải trọn bộ DataFrame gốc từ HuggingFace `nhn309261/vietnamese-legal-docs`, lọc CSV chỉ giữ tệp mục tiêu mảng Thể thao, Y tế.
> - **Bước 2 (Chỉ mục Toàn Cục):** Bật hai hệ thống Dictionary cache siêu tốc (`meta_by_docnum_lookup` dùng tạo node ảo và `global_doc_lookup` lưu trữ 240 ngàn mảng text rút trích Exact Xung đột).
> - **Bước 3 (Cắt Chunk Layer):** Ném data rẽ sóng đi qua `AdvancedLegalChunker`. Mặc sức chia tách cấu trúc logic ngữ nghĩa thành 3,352 Blocks độc lập.
> - **Bước 4 (Vector hóa Dense/Sparse Layer):** 3,352 array string này chui vào hàm encode của LLM Encoder `BGE-M3`.
> - **Bước 5 (Database Injecting Layer):** Tách 2 ngả đường. Đường mỏng đẩy JSON nhẹ nhàng qua Endpoint của Qdrant Cloud cho tốc độ search dưới 20ms. Mảng Metadata đồ sộ kèm viền Edge tham chiếu nhồi vào Cổng Driver Cypher của Neo4j Aura Graph. Khép kín vòng đời tái chỉ mục RAG.
