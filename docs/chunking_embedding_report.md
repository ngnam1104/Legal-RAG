# 📋 Báo Cáo Kỹ Thuật: Pipeline Xử Lý Văn Bản Pháp Luật

**Mã nguồn chính:**
- [core.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/core.py) — Chunking Orchestrator
- [fsm.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/fsm.py) — Finite State Machine duyệt dòng
- [metadata.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/metadata.py) — Regex Patterns & Helper Functions
- [heuristics.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/heuristics.py) — Relation hint detection
- [payload.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/payload.py) — Đóng gói payload Qdrant & Neo4j
- [toc.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/toc.py) — Table-of-Contents extraction
- [relations.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/extractor/relations.py) — Ontology Relation Extraction (10 nhãn)
- [entities.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/extractor/entities.py) — Unified LLM Entity Extraction
- [embedder.py](file:///d:/iCOMM/Legal-RAG/backend/models/embedder.py) — Dense (REST API) + Sparse (BM25) Encoder
- [reranker.py](file:///d:/iCOMM/Legal-RAG/backend/models/reranker.py) — Cross-Encoder Reranker (REST API)
- [hybrid_search.py](file:///d:/iCOMM/Legal-RAG/backend/retrieval/hybrid_search.py) — Tiered Prefetch + Rerank + Context Expansion
- [neo4j_client.py](file:///d:/iCOMM/Legal-RAG/backend/database/neo4j_client.py) — Neo4j Graph Build, Constraints & Entity Enrichment
- [chunking_embedding.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunking_embedding.py) — 🚀 Script chạy chính: 6-Phase Pipeline

**Ngày tạo báo cáo:** 2026-04-16
**Cập nhật lần cuối:** 2026-04-29

---

## Mục lục

1. [Phần 1: Kỹ thuật Chunking, Regex & Phát hiện cấu trúc](#phần-1-kỹ-thuật-chunking-regex--phát-hiện-cấu-trúc)
2. [Phần 2: Kỹ thuật dựng QdrantDB](#phần-2-kỹ-thuật-dựng-qdrantdb)
3. [Phần 3: Kỹ thuật dựng Neo4jDB](#phần-3-kỹ-thuật-dựng-neo4jdb)

---

# Phần 1: Kỹ thuật Chunking, Regex & Phát hiện cấu trúc

## 1.1 Tổng quan kiến trúc Chunker

Toàn bộ logic chunking nằm trong class `AdvancedLegalChunker` (file [core.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/core.py)), được thiết kế theo mô hình **Finite State Machine (FSM)** — máy trạng thái hữu hạn duyệt từng dòng văn bản từ trên xuống dưới. FSM engine tách ra module [fsm.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/fsm.py), regex patterns tại [metadata.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/metadata.py), logic đóng gói payload tại [payload.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/payload.py), và logic bóc tách quan hệ nằm trong [relations.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/extractor/relations.py).

**Quy trình máy trạng thái (FSM):**
1. **Bước Khởi đầu (HeaderZone):** Duyệt dòng đầu tiên. Nếu khớp regex Chương → chuyển trạng thái sang [ChapterDetected]. Nếu khớp regex Điều → chuyển trạng thái sang [ArticleDetected]. Nếu không khớp gì → thiết lập [FreeText].
2. **Xử lý Điều khoản (ArticleDetected):** Khi hệ thống đang đánh dấu trong một Điều, nếu phát hiện Khoản → hệ thống chuyển sang [ClauseDetected]. Nếu có Điểm trực thuộc Khoản → chuyển sang [PointDetected]. Trong bất kỳ bước nào, nếu bộ đệm kích thước chữ vượt ngưỡng `CHUNK_LIMIT`, hệ thống sẽ kích hoạt ép chốt văn bản và khởi tạo một log chunk tách rời nối tiếp [FlushChunk].
3. **Xử lý Phụ lục (AppendixZone):** Nếu rẽ nhánh bằng mẫu regex Phụ lục, hệ thống chia sâu theo 3 cấp độ thụt đầu dòng cấu trúc: I, II (Cấp 1) → 1., 2. (Cấp 2) → a), b) (Cấp 3).
4. **Xử lý Bảng biểu (TableZone):** Dòng bất kỳ chứa từ 2 ký tự `|` trở lên sẽ thay đổi cờ trạng thái bảng và gom hàng loạt thành block bảng.
5. **Xử lý Text tự do (FreeText):** Gom văn bản rải rác ngoại vi, chịu sự kềm tỏa ngặt của `TEXT_LIMIT` gây ngắt đoạn chunk [FlushChunk].

> [!NOTE]
> **Giải thích thuật toán FSM trong mã nguồn ([core.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/core.py) + [fsm.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/fsm.py)):**
> Lớp `AdvancedLegalChunker` xử lý tuần tự theo các bước:
> - **Bước 1 (Khởi tạo):** Đọc văn bản theo dòng, bắt đầu ở vùng `HeaderZone`.
> - **Bước 2 (Phân phối cấu trúc):** Áp dụng Regex (`chapter_pattern`, `article_pattern`...) để chuyển mốc trạng thái từ cấp Chương nhảy xuống Điều.
> - **Bước 3 (Khoản/Điểm):** Khi đang ở trong Điều, hệ thống rẽ nhánh kiểm tra `clause_pattern` và `point_pattern`. Tính kế thừa được duy trì qua biến lưu trạng thái.
> - **Bước 4 (Ngoại lệ - Phụ lục/Bảng):** Cùng lúc dò tìm Phụ Lục (`appendix_title_pattern`) hoặc Bảng biểu (phát hiện ký tự `|`).
> - **Bước 5 (Flush):** Tại bất kỳ trạng thái nào, nếu `current_text` phình to quá `CHUNK_LIMIT` (1800 ký tự) hoặc `TEXT_LIMIT` (2500 ký tự), hàm `flush_article()` sẽ được gọi để chốt hạ và đóng gói chunk lại một cách an toàn.
> - **Bước 6 (Footer):** Phần Nơi nhận và Chữ ký được giữ lại thành chunk riêng (thay vì bị loại bỏ) để chatbot có thể trả lời câu hỏi về người ký, nơi nhận.

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
    r"(?im)^\s*(QUY ĐỊNH|QUY CHẾ|QUY CHUẨN|QCVN|TIÊU CHUẨN|TCVN|PHƯƠNG ÁN|ĐIỀU LỆ|CHƯƠNG TRÌNH|HƯỚNG DẪN|NỘI QUY|KẾ HOẠCH|CHIẾN LƯỢC|ĐỀ ÁN|DỰ ÁN)\b(?!\s*CHUNG\b).*$"
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
- **Mục đích (hiện tại):** **Giữ lại** thành chunk riêng với `article_ref = "Phần Nơi nhận và Chữ ký"` để chatbot có thể trả lời câu hỏi về người ký, nơi nhận. Trước đây bị loại bỏ hoàn toàn → gây mất thông tin.

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

Hệ thống lưu giữ logic phân cấp nội bộ thông qua các cờ trạng thái kết hợp bộ nhớ tĩnh:

**Biến lưu trữ phân cấp:**
- `current_chapter` (string): Lưu tên phân khu / Chương hiện hành.
- `current_article_ref` (string): Đóng đinh con trỏ Điều hiện tại (vd: "Điều 5").
- `current_article_preamble` (list[str]): Mảng bảo vệ chứa Lời dẫn/Đoạn văn tự do của Điều đó.
- `current_clauses_buffer` (list[dict]): Danh sách đã đóng băng các khối Khoản.
- `current_active_clause` (dict): Khoản đang rải rác đắp xây thêm, chứa ref, text, points.

**Cờ trạng thái:**
- `in_appendix` (bool): Quy chuẩn logic thụt bậc ở vị thế Phụ lục.
- `in_table` (bool): Đang mở giao diện bảng dữ liệu dạng ô Markdown.

> [!NOTE]
> **Giải thích cơ chế Buffer System trong mã nguồn ([fsm.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/fsm.py)):**
> Trong hàm `process_document()`, bộ nhớ hoạt động theo các bước:
> - **Bước 1 (Lưu vết cấp cao):** Khi quét thấy một Chương/Phần mới, tên của nó tự động lưu đè lên bộ nhớ tạm `current_chapter`.
> - **Bước 2 (Mở bộ đệm trung gian):** Với Điều con, `current_article_ref` sẽ lưu lại nhãn "Điều X", trong khi `current_article_preamble` sẽ hứng các đoạn text lời dẫn vào Điều này.
> - **Bước 3 (Xây dựng Khoản/Điểm):** Khi bắt vào Khoản, biến `current_active_clause` là một dict sẽ hứng ID Khoản và mạo danh một chuỗi list riêng `clause.points` (để hứng các Điểm nhỏ).
> - **Bước 4 (Đóng băng):** Khi một thẻ cấu trúc mới nhảy tới, dữ liệu con đang rải rác bên trong sẽ được nạp gộp mảng tổng `current_clauses_buffer` chờ duyệt xả, giúp breadcrumb không bao giờ bị hổng.

### 1.5.2 Giới hạn kích thước

| Hằng số | Giá trị | Mục đích |
|---|---|---|
| `CHUNK_LIMIT` | 1800 chars | Ngưỡng tổng buffer Khoản + Điểm. Vượt qua → flush |
| `TEXT_LIMIT` | 2500 chars | Ngưỡng text tự do (preamble, đoạn nối tiếp). Vượt qua → flush |
| Bảng biểu: rows | 20 dòng | Ngưỡng số dòng bảng tối đa trong 1 chunk |
| Bảng biểu: size | 3000 chars | Ngưỡng kích thước bảng tối đa |
| Min letters | 30 chữ cái | Chunk có < 30 ký tự alphabet → **loại bỏ** (rác), trừ các section quan trọng (`Lời dẫn`, `Phần Nơi nhận và Chữ ký`) |
| Min bảng biểu | 15 ký tự | Bảng biểu có < 15 ký tự chữ+số → loại bỏ |

### 1.5.3 Cơ chế "Tiếp theo" (Continuation Chunks)

Khi một Khoản quá dài vượt `CHUNK_LIMIT` (1800 chars):

1. Flush chunk hiện tại
2. Tạo chunk nối tiếp với `ref = "Khoản X (tiếp theo)"`
3. Text mới bắt đầu bằng `[Khoản X tiếp theo]` (Context Injection)

Điều này giúp LLM biết chunk đang đọc là phần nối tiếp của Khoản nào.

### 1.5.4 Cơ chế lọc rác

| Bộ lọc | Điều kiện | Hành động |
|---|---|---|
| Min letters | `re.sub(r'[\W_0-9]', '', text)` < 30 | Bỏ qua chunk (trừ `Lời dẫn`, `Phần Nơi nhận và Chữ ký`) |
| Tiêu đề mồ côi | Không có article_ref, toàn VIẾT HOA | Bỏ qua chunk (trừ critical sections) |
| Header zone | Dòng < 50, chứa "ĐỘC LẬP"/"TỰ DO" | Skip dòng |
| Footer | Match `footer_pattern` | **Giữ lại** thành chunk riêng `article_ref = "Phần Nơi nhận và Chữ ký"` |
| Căn cứ | `legal_basis_line_pattern` | **Giữ lại** (để chatbot trả lời cơ sở pháp lý) |
| Paywall/Quảng cáo | Regex PAYWALL_PATTERNS | **Loại bỏ** text rác quảng cáo website trước khi chunking |

### 1.5.5 Cơ chế xử lý Bảng biểu

Quy tắc thuật toán xử lý và ngắt block nhịp Bảng biểu hoạt động bằng việc quét ký tự `|` (mô phỏng định dạng markdown lưới table):

1. **Giai đoạn khởi phát:** Tìm ra 1 dòng mang >= 2 ký tự viền `|`.
   - Nếu `in_table` đang tắt: Lập tức gọi lệnh `flush_article()` để giải thoát vùng text tĩnh tồn đọng vừa đọc. Bật cờ thành ON, gán dòng hiện thời đóng vai trò `table_header`.
2. **Giai đoạn thu nạp nòng cốt:** Tiếp tục chạy với `in_table` True.
   - Nếu dòng trắng đặc gồm -, :... thì gán đuôi `table_header`.
   - Nếu dòng dữ liệu, kiểm soát nghiêm dung lượng. Vượt ngưỡng hàng row >= 20 dòng hoặc khối phình > 3000 chars, sẽ ngắt chốt lệnh `flush_table()` rồi đúc mảng lưới bảng kế, tái sử dụng thanh title cột.
3. **Giai đoạn thoát nhịp:**
   - Dòng hiện tiếp không chứa `|`, nếu `in_table` True thì gọi xả kết quả cho phần thân còn lưu đọng trong bộ đệm `table_rows`, rồi thiết lập OFF cờ thu bảng.

> [!NOTE]
> **Giải thích cơ chế bắt bảng trong mã nguồn ([fsm.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/fsm.py)):**
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
    "text_to_embed":  "[doc_number] short_title\nbreadcrumb\n...", # Text tinh gọn có chèn cả tiêu đề ngắn để embed
    "qdrant_metadata": { ... },                      # Payload cho Qdrant (Đồng bộ 17 trường giống Neo4j)
    "neo4j_metadata":  { ... }                       # Payload đầy đủ cho Neo4j
}
```

> [!IMPORTANT]
> **Tách biệt `chunk_text` vs `text_to_embed`:** `chunk_text` chứa metadata header (Văn bản, Lĩnh vực, Điều khoản) để LLM đọc hiểu ngữ cảnh. `text_to_embed` chứa `[doc_number] short_title` + `breadcrumb` + nội dung gốc để không bị loãng ngữ nghĩa nhưng vẫn giữ được độ liên kết Semantic sâu nhờ Title của văn bản.

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

## 1.8 Khai thác Old Text Thông Minh (`extract_old_content_via_llm`)

Mục đích: Khi phát hiện văn bản A sửa đổi bổ sung Điểm a Khoản 2 Điều 5 của văn bản B, module RelationExtractor không thể chỉ dùng regex để cắt bù 2500 ký tự vì sẽ dễ lẫn lộn sang khoản khác.

**Thuật toán kết hợp Regex Boundary + LLM Drill-down:**
1. **Tìm Boundary (Regex)**: Sử dụng hàm `extract_exact_article()` tìm đúng vùng không gian Điều 5 bằng các Pattern Regex (Bắt đầu bằng `Điều 5.` và kết thúc khi sang `Điều 6.`).
2. **On-Premise LLM Rút trích (Drill-down)**: Cắt khối text thu được đẩy cho Server nội bộ (`InternalLLMClient` gọi Llama-3 8B tại `10.9.3.75:30028`) kèm tham số `target_clause`. Prompt yêu cầu: *"Trích xuất nguyên văn nội dung của 'Khoản 2... Điều 5' từ đoạn văn bản gốc"*. Việc có Server On-Premise dùng Free/Unlimited call giúp chạy Batch Offline hàng chục ngàn docs mà không tốn chi phí Cloud API.
3. Fallback: Nếu nội bộ lỗi thì trả về toàn bộ text của Điều đã được bảo vệ bằng Boundary Regex.

---

## 1.9 Khai Thác Mối Quan Hệ (`Local LLM Fallback Extraction`)

Hệ thống khai thác quan hệ hiện nay sử dụng kiến trúc **Unified Single-Pass Extraction** trong module [relations.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/extractor/relations.py) kết hợp [entities.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/extractor/entities.py).

**Cơ chế Extraction logic (phiên bản hiện tại):**
- **Heuristics Detection**: Module [heuristics.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/heuristics.py) quét hint quan hệ trong text, gắn cờ `relation_hints` vào metadata chunk.
- **Unified LLM Prompt**: Mỗi batch đoạn văn quan trọng được gom lại và gửi qua **1 prompt duy nhất** (`build_unified_prompt()` trong `entities.py`). Response LLM trả về đồng thời: `doc_relations` (ontology 10 nhãn), `entities` (thực thể tự do), và `node_relations` (graph triplets).
- **Passive Chain Detection**: Phát hiện quan hệ bị động trong vùng Căn cứ ("Luật A đã được sửa đổi theo Luật B") → tự động sinh cross-doc relations (B→AMENDS→A).
- **Regex Sweep**: Quét bù các số hiệu VB mà LLM bỏ sót, sử dụng Zone-aware Keywords (bộ từ khóa khác nhau cho vùng Preamble vs Article).
- **Old Content Drill-down**: Riêng với trường hợp `AMENDS`, hệ thống sử dụng `extract_exact_article()` (Regex Boundary) kết hợp LLM để trích xuất đúng nội dung nguyên bản.
- **Entity Enrichment**: Kết quả entities & node_relations được gắn vào `neo4j_metadata` rồi đẩy lên Neo4j qua `enrich_chunk_entities()` ([neo4j_client.py](file:///d:/iCOMM/Legal-RAG/backend/database/neo4j_client.py)).

# Phần 2: Kỹ thuật dựng QdrantDB

## 2.1 Kiến trúc tổng quan

Mô hình Indexing dữ liệu RAG và Embed lưu trữ trên VectorDB (Qdrant) đi qua pipeline 6 pha ([chunking_embedding.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunking_embedding.py)):
1. **Dữ liệu HF:** Tải toàn bộ dataset `nhn309261/vietnamese-legal-docs` từ HuggingFace, lọc qua CSV 8000 VB ưu tiên.
2. **Parser lõi:** Băm cắt cấu trúc qua module class `AdvancedLegalChunker` ([core.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunker/core.py)) + trích xuất quan hệ Unified LLM qua [relations.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/extractor/relations.py).
3. **Mã hoá nhúng:** Dense qua REST API `InternalAPIEmbedder` (BGE-M3 1024-d) + Sparse qua fastembed BM25 local.
4. **Hạ điểm tập kết:** Vector Data upsert vào Collection `legal_rag_docs_nam` của Qdrant, đồng thời build Neo4j Graph.

> [!NOTE]
> **Giải thích luồng Data đến Qdrant ([chunking_embedding.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunking_embedding.py)):**
> - **Bước 1 (Phân thân dữ liệu — Phase 1-3):** Tải Dataset từ HuggingFace, lọc qua CSV 8000 văn bản ưu tiên (Y tế). Chunker FSM xử lý tuần tự 3 tầng: VB gốc → VB tham chiếu depth=1 → VB tham chiếu depth=2.
> - **Bước 2 (Phase 4 — Ghost Nodes):** Tạo Ghost Node cho các VB được tham chiếu nhưng không có trong dataset.
> - **Bước 3 (Phase 5 — Mã hoá Vector):** Embed batch 64 chunk qua `InternalAPIEmbedder` (Dense BGE-M3 1024-d) + fastembed BM25 (Sparse local). Upsert Qdrant.
> - **Bước 4 (Phase 6 — Neo4j Build):** Đẩy payload `neo4j_metadata` lên Neo4j: Document Tree + Ghost Nodes + Graph Triplets + Entity Enrichment (10 loại entity từ Unified Extractor).

## 2.1.2 Kiến trúc Neo4j Graph DB (Dynamic Tree & Leaf Level Logic)

Đầu ra của Chunker chuyên trách đẩy `neo4j_metadata` vào file [neo4j_client.py](file:///d:/iCOMM/Legal-RAG/backend/database/neo4j_client.py). Thiết kế sử dụng cơ chế lá đa dạng giải quyết độ sâu của mỗi văn bản.

**Kịch bản phân bổ Leaf Level Logic (Mức độ Lá):**
- **Trường hợp A (Lá là Điều)**: Đọc ngang một Điều không thấy có chia Khoản/Điểm, FSM lập tức coi `Article` đó là lá cuối, gán thẳng ID vào Node `Article` và lưu lại Text.
- **Trường hợp B (Lá là Khoản)**: `Article` phân Khoản liền mạch, code coi `Clause` là lá cuối, gán Text + chunking metadata vào thẳng Node `Clause` (liên kết `PART_OF` tới mẹ `Article`).
- **Trường hợp C (Lá là Mảnh Chunk lẻ)**: Một Khoản quá dài, bị xé do quá 1200 char, code tạo hẳn node `Chunk` tách biệt, móc vào `Clause`. Đối với Table/bảng biểu sinh ở `flush_table` cũng tạo lập Node `Chunk`.

Cách này giúp Neo4j không bị kẹt cố định trong một cấu trúc tĩnh mà "co giãn" node map đúng theo đặc thù tài liệu VN. Toàn bộ `init_neo4j_constraints` khóa UNIQUE các nhánh node và dùng vòng lặp `UNWIND` chèn vào Graph tốc độ cao vút.

## 2.2 Mô hình Embedding: InternalAPIEmbedder (On-Premise REST API)

Mã nguồn: [embedder.py](file:///d:/iCOMM/Legal-RAG/backend/models/embedder.py)

### 2.2.1 Class `InternalAPIEmbedder`

| Thuộc tính | Giá trị |
|---|---|
| Endpoint (Dense) | `http://10.9.3.75:30010/api/v1/embedding` |
| Dense Model | **BAAI/bge-m3** (1024-d, Cosine) |
| Dense dimension | **1024** |
| Dense normalize | ✅ `True` (L2 normalization trên server) |
| Sparse Model | **fastembed BM25** (`Qdrant/bm25`, chạy local CPU) |
| Timeout | 60 giây |
| Singleton | `_instance` module-level guard |
| Dense Fallback | Zero-vectors `[0.0] × 1024` nếu REST API sập |
| Sparse Fallback | `SparseVector(indices=[], values=[])` nếu BM25 model lỗi |

### 2.2.2 Phương thức `encode()` — Dense

```python
def encode(self, texts: List[str]) -> List[List[float]]:
    """Gửi danh sách văn bản tới API và nhận về Dense vectors (1024-d)."""
    # Sanitize: loại bỏ null bytes và ký tự không hợp lệ
    texts = [t.replace('\x00', '').encode('utf-8', errors='ignore').decode('utf-8') for t in texts]
    payload = {"texts": texts, "normalize": True}  # L2 normalization
    resp = self._session.post(self.endpoint, json=payload, timeout=self.timeout)
    data = resp.json()
    embeddings: List[List[float]] = data.get("embeddings", [])
    return embeddings
    # Fallback: zero-vectors nếu API lỗi
```

### 2.2.3 Sparse Vector — fastembed BM25 (Qdrant/bm25)

Khác với phiên bản cũ sử dụng Deterministic Hash (MD5), hệ thống hiện tại đã chuyển sang **fastembed BM25** (`Qdrant/bm25`) — một mô hình BM25 chạy hoàn toàn local trên CPU, cho chất lượng sparse retrieval vượt trội hơn nhiều so với TF hash đơn giản.

**Hai phương thức sparse chính:**

```python
# 1. Encode documents (dùng cho ingestion/indexing)
def encode_sparse_documents(self, texts: List[str]) -> List[SparseVector]:
    embeddings = list(self.sparse_model.embed(texts, batch_size=batch_size))
    return [SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()) for emb in embeddings]

# 2. Encode query (dùng cho search/retrieval)
def encode_query_sparse(self, text: str) -> SparseVector:
    query_embeddings = list(self.sparse_model.query_embed(text))
    emb = query_embeddings[0]
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
```

> [!TIP]
> **fastembed BM25 vs Deterministic Hash (phiên bản cũ):**
> - **BM25** có trọng số IDF (Inverse Document Frequency) → từ hiếm có weight cao hơn, giúp match chính xác hơn.
> - **Deterministic Hash** chỉ dùng TF (Term Frequency) → mọi từ đều bình đẳng, dễ bị nhiễu.
> - BM25 tách biệt `embed()` (cho document indexing) và `query_embed()` (cho search) → tối ưu hóa cho từng use case.

## 2.2.4 Reranker: InternalAPIReranker (On-Premise REST API)

Mã nguồn: [reranker.py](file:///d:/iCOMM/Legal-RAG/backend/models/reranker.py)

Sau bước Hybrid Search (Dense + Sparse), hệ thống sử dụng **Cross-Encoder Reranker** để re-score chính xác hơn trước khi trả kết quả cho LLM.

| Thuộc tính | Giá trị |
|---|---|
| Endpoint | `http://10.9.3.75:30546/api/v1/reranking` |
| Timeout | 60 giây |
| Singleton | `_instance` module-level guard |
| Fallback | Giữ nguyên thứ tự gốc với `score = 0.0` nếu API sập |

**Luồng Rerank:**
1. Nhận `candidates` từ Tiered Prefetch (broad_retrieve).
2. Xây dựng text cho mỗi candidate: `title + reference_citation + chunk_text`.
3. Gửi `query + docs` tới REST API reranking.
4. Nhận `scores[]`, map lại vào candidates, sắp xếp giảm dần.
5. Áp dụng **Document Typology Boost** (tie-breaker nhỏ cho Nghị định/Luật) và **Appendix Penalty** (-0.002).

```python
# Document Typology Boost (tie-breaker)
if "NGHỊ ĐỊNH" in l_type: boost += 0.005
elif "NGHỊ QUYẾT" in l_type: boost += 0.003
if p.get("is_appendix"): boost -= 0.002
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
| `on_disk=True` (sparse) | Lưu sparse index trên đĩa | Sparse BM25 vectors từ fastembed |
| Distance metric | `COSINE` | Phù hợp với L2-normalized dense vectors (normalize=True) |

## 2.4 Qdrant Payload Schema

Mỗi point được upsert lên Qdrant với payload **đồng bộ tối đa** (bổ sung 8 trường filtering) để chuẩn bị cho RAG đa chiều:

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
    "title":           "Nghị định 142/2024/NĐ-CP về...",
    "article_ref":     "Điều 1",
    "is_active":       true,
    "is_appendix":     false,
    "legal_type":      "Nghị định",
    "effective_date":  "2024-01-01",
    "url":             "https://...",
    "chunk_index":     1,
    "reference_citation": "142/2024/NĐ-CP | Chương I | Điều 1 | Khoản 1"
}
```

> [!TIP]
> Việc bổ sung 8 trường (`article_ref`, `is_active`, `is_appendix`...) giúp các bộ lọc Semantic Filter (Vector Search) hoạt động trơn tru. Payload Qdrant **không chứa** `legal_basis_refs`, `amended_refs`, `signer_name`, v.v. — những quan hệ Graph edges này vẫn chỉ đẩy vào Neo4j.

## 2.5 Pipeline Upsert — Batch Processing

### 2.5.1 Cấu hình Batch

| Hằng số | Giá trị | Mục đích |
|---|---|---|
| `DOC_BATCH_SIZE` | 250 | Số văn bản xử lý mỗi lần lặp |
| `EMBED_BATCH_SIZE` | 128 | Batch size cho model embedding |
| `UPSERT_BATCH_SIZE` | 512 | Batch size cho Qdrant upsert |

### 2.5.2 Luồng xử lý chính

Quy trình vòng lặp chính của file Builder được vạch ra tuần tự nhằm giữ an toàn bộ nhớ:
- **Bước 1: Load Batch:** Chia tập danh sách văn bản theo khối lô nhỏ duyệt (ví dụ 250 VB/lần).
- **Bước 2: Phân tách (Chunking):** Chạy lệnh `chunker.process_document()` để băm nhỏ dữ liệu và xuất list array `batch_chunks`.
- **Bước 3: Mã hóa (Embedding):** Gọi `hybrid_encoder.encode_hybrid()` để nhận set tensor cực mịn chứa Dense_vecs, Sparse_vecs.
- **Bước 4: Sinh ID tĩnh:** Tạo PointStruct định danh payload Qdrant chốt bằng key sinh từ thuật toán hash `UUID5` qua chuỗi `chunk_id`.
- **Bước 5: Đồng bộ Database:** Lệnh SDK Qdrant nhồi batch 512 tọa độ/lần, riêng metadata đồ thị vươn cành lưu lên Neo4j thông qua `build_neo4j()`.
- **Bước 6: Thu dọn phiên:** Ép tiến trình Python nhả dung lượng rác `gc.collect()`.

> [!NOTE]
> **Giải thích Pipeline Upsert theo lô ([chunking_embedding.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunking_embedding.py)):**
> - **Bước 1 (Phase 1-3 Chunking):** Chia danh sách 8000 VB thành các batch nhỏ (`BATCH_SIZE = 4`). Mỗi VB được xử lý tuần tự: chunking → relation extraction → entity extraction.
> - **Bước 2 (Phase 5 Embedding):** Embed batch 64 chunk song song qua `InternalAPIEmbedder` (Dense BGE-M3) + fastembed BM25 (Sparse). Upsert lên Qdrant Collection.
> - **Bước 3 (Phase 6 Graph Build):** Chunk batch 5000 được đẩy vào `build_neo4j()` → tạo Document Tree + Ghost Nodes. Tiếp theo `enrich_chunk_entities()` đẩy entity/node_relation data lên Neo4j.
> - **Bước 4 (Clear RAM):** Cuối mỗi lô embed `gc.collect()` ép xóa tensor rác và giải phóng bộ nhớ. Checkpoint được lưu vào `.checkpoints/` cho phép resume khi bị gián đoạn.

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
    "documents": 90,
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
| Thời gian embedding | 69.17s | Bottleneck chính (REST API On-Premise) |
| Thời gian upsert | 9.76s | Nhanh do Qdrant local |

---

# Phần 3: Kỹ thuật dựng Neo4jDB

## 3.1 Schema Đồ thị (Graph Schema)

**1. THỰC THỂ CHÍNH (Core Entities):**
- **Document**: (`id`, `document_number`, `title`, `promulgation_date`, `effective_date`, `year`, `url`, `doc_status`, `document_toc`)
- **Article**: (`id`, `name`, `chapter_ref`, `qdrant_id`, `text`, `is_table`)
- **Clause**: (`id`, `name`, `qdrant_id`, `text`, `is_table`)
- **Chunk**: (`id`, `chunk_index`, `text`, `is_table`, `qdrant_id`)

**2. THỰC THỂ METADATA (Satellite Nodes):**
- **Authority**: Cơ quan ban hành (`name`)
- **Signer**: Người ký duyệt (`name`, `signer_id`)
- **LegalType**: Nhãn phân loại hình thức văn bản (`name`)
- **Sector**: Lĩnh vực áp dụng (`name`)

**3. MỐI QUAN HỆ CỐT LÕI (Edges):**
- Quan hệ Ontology (10 nhãn): `Document` —[ `BASED_ON`, `AMENDS`, `REPLACES`, `REPEALS`, `GUIDES`, `APPLIES`, `REFERENCES`, `ISSUED_WITH`, `ASSIGNS`, `CORRECTS` ]→ `Document`
- Phân đoạn nội hàm:
  - `Article` —[ `BELONGS_TO` ]→ `Document`
  - `Clause` —[ `PART_OF` ]→ `Article`
  - `Chunk` —[ `PART_OF` ]→ `Clause` / `Article` 
  - `Chunk lẻ` —[ `BELONGS_TO` ]→ `Document`
- Thuộc tính hành chính: 
  - `Authority` —[ `ISSUED` ]→ `Document`
  - `Signer` —[ `SIGNED` ]→ `Document`
  - `Document` —[ `HAS_TYPE` ]→ `LegalType`
  - `Document` —[ `HAS_SECTOR` ]→ `Sector`

> [!NOTE]
> **Giải thích kiến trúc rẽ nhánh mạng Neo4j Schema ([neo4j_client.py](file:///d:/iCOMM/Legal-RAG/backend/database/neo4j_client.py)):**
> - **Bước 1 (Xây xương sống Hành chính):** Nhóm các lệnh khởi tạo sẽ chia Đồ thị ra, Core nhất là văn bản `Document`. Từ `Document` lại bóc ra nhánh phân cấp đi xuống các phần tử chứa Text là: `Document -> Article -> Clause -> Chunk`. 
> - **Bước 2 (Chỏm râu Metadata):** Tạo thêm các node vệ tinh bao quanh như Cơ quan ban hành `Authority`, Người ký (`Signer`), hay mảng Lĩnh vực `Sector`. Móc nối Edge chĩa vào thân `Document`.
> - **Bước 3 (Mạng Đồ Thị Xung Đột):** Khi tài liệu A cập nhật tài liệu B, đường viền quan hệ sẽ nối thẳng `Document A` qua `Document B` mang nhãn `AMENDS / REPLACES` cùng với trích xuất văn bản đối sánh.
> - **Bước 4 (Entity Enrichment):** Hàm `enrich_chunk_entities()` đẩy 10 loại entity tự do (Organization, Person, Fee, v.v.) + node_relations lên Neo4j từ kết quả Unified Extractor.
> - **Bước 5 (Cầu nối Data Hybrid):** Neo4j không chứa vector đồ sộ. Nó chỉ nhận trường khóa ngoại `qdrant_id` đại diện đính lên các block Node lá (như Article, Clause). Khi cần RAG sẽ trượt song song cả hai nền tảng.

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

Đây là **trái tim kỹ thuật** của Neo4j pipeline. Thay vì luôn tạo node Chunk tĩnh, hệ thống sử dụng thuật toán phân cấp tự động xác định **mức (level) nào là leaf gốc (lá)** để gắn trường id ngoại `qdrant_id` và lưu chuỗi `text`.

**Quyết định mức Lá (Level Identification):**
1. **Kiểm tra Article Root**: Nếu chunk hoàn toàn khuyết `article_ref`, nó bị đẩy thành mộc *Chunk Mồ Côi*, trỏ trực tiếp lên `Document`.
2. **Có Article nhưng Khuyết Clause**: Hệ thống tra dấu `Article` gốc. Nếu chưa từng nhồi text, Node `Article` trở thành Lá. Nếu đã rải rác đắp text → tách Node `Chunk Con` móc `PART_OF` vào Article.
3. **Có cả Điều và Khoản**: Nhìn theo dấu `clause_ref`. Nếu là khối chia "(tiếp theo)" thì xé nốt Node `Chunk Mảnh` móc nối dưới Clause. Nếu rỗng trọn vẹn thì Node `Clause` gánh vai trò ghim text vào nhánh đồ thị.

> [!NOTE]
> **Giải thích Giải thuật cây động ghim nốt lá - "Leaf Level" ([neo4j_client.py](file:///d:/iCOMM/Legal-RAG/backend/database/neo4j_client.py)):**
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

Hàm `enrich_reference_nodes()` chạy **trước** `build_neo4j()` thông qua dòng lệnh Cypher bổ trợ:
1. Quét tệp khối lượng lô `batch_chunks` lớn và rút tập hợp toàn bộ `doc_number` xuất hiện rải rác qua thẻ `legal_basis_refs`, `amended_refs`...
2. Mang list các số hiệu đi tra bảng Hash nội địa `meta_by_docnum_lookup`.
3. Sinh mạng lưới Node Document bổ khuyết thông tin `title`, `url`, tạo node giả bằng mật kế `MERGE` kết hợp `SET` hoàn thiện lưới RAG.

> [!NOTE]
> **Giải thích thuật toán cập nhật vòng đệm Enrichment ([neo4j_client.py](file:///d:/iCOMM/Legal-RAG/backend/database/neo4j_client.py)):**
> Để tránh tình trạng khi Văn bản A trỏ tới Văn bản B nhưng trên Neo4j văn bản B bị khuyết thông tin:
> - **Bước 1 (Tìm dấu vết Quan hệ):** Khi lô Chunk duyệt thấy các tham chiếu (nằm trong `ontology_relations`...), thuật toán chủ động sinh ra node Document cho Văn Bản B mang số hiệu tương ứng.
> - **Bước 2 (Lookup Dictionary offline):** Neo4j quét list `doc_number` ảo này móc vào từ điển Python tĩnh `meta_by_docnum_lookup` (thứ đang buffer sẵn 238K+ danh mục tải ở cache HuggingFace).
> - **Bước 3 (Ghost Nodes — Phase 4):** Pipeline tự động tạo Ghost Node (`is_ghost=true`) cho mọi VB được tham chiếu nhưng không tồn tại trong dataset, sử dụng `MERGE` Cypher.
> - **Bước 4 (Entity Enrichment — Phase 6d):** Hàm `enrich_chunk_entities()` đẩy entities tự do + node_relations từ Unified Extractor lên Neo4j, tạo node Entity động với các label tùy ý (Organization, Fee, Condition, v.v.).

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
    "ontology_relations": [
        {"type": "AMENDS", "target": "...", "context": "..."},
        {"type": "GUIDES", "target": "...", "context": "..."}
    ]
}
```

> [!IMPORTANT]
> `legal_basis_refs` và `ontology_relations` chỉ được gắn vào **chunk đầu tiên** (`chunk_index == 1`) của mỗi văn bản. Các chunk sau đều nhận mảng rỗng `[]` để tối ưu tải trọng mốc và tránh lặp quan hệ trên đồ thị Neo4j.

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
| `legal_type` | ✅ | ✅ |
| `effective_date` | ✅ | ✅ |
| `url` | ✅ | ✅ |
| `article_ref` | ✅ | ✅ |
| `is_active` | ✅ | ✅ |
| `is_appendix` | ✅ | ✅ |
| `reference_citation` | ✅ | ✅ |
| `chunk_index` | ✅ | ✅ |
| `issuing_authority` | ❌ | ✅ |
| `signer_name` / `signer_id` | ❌ | ✅ |
| `promulgation_date` | ❌ | ✅ |
| `chapter_ref` / `clause_ref` | ❌ | ✅ |
| `legal_basis_refs` | ❌ | ✅ (chunk 1 only) |
| `ontology_relations` | ❌ | ✅ (chunk 1 only, 10 nhãn) |
| `document_toc` | ❌ | ✅ |

---

## Phụ lục: Data Flow tổng quan end-to-end

Quy trình xử lý hoàn chỉnh E2E tuân thủ thao tác lưu chuyển qua [chunking_embedding.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunking_embedding.py):

1. **DATA SOURCE:** Tải dataset HuggingFace `nhn309261/vietnamese-legal-docs` (offline cache). Lọc qua CSV 8000 VB ưu tiên.
2. **DATA LOADING:** Dựng `DynamicContentLookup` (lazy load nội dung theo doc_key) và `meta_by_docnum_lookup` ~238K records.
3. **CHUNKING + EXTRACTION (Phase 1-3):** FSM băm cấu trúc qua `AdvancedLegalChunker` → Unified LLM Extraction (quan hệ + entity) → Ghost Nodes (Phase 4).
4. **EMBEDDING & PERSISTENCE (Phase 5-6):**
   - Dense BGE-M3 (1024-d) + Sparse fastembed BM25 → Upsert Qdrant (batch 64).
   - `build_neo4j()` → Dynamic Tree + `enrich_chunk_entities()` → Entity Enrichment trên Neo4j.

> [!NOTE]
> **Giải thích Tổng thể luồng E2E Pipeline RAG ([chunking_embedding.py](file:///d:/iCOMM/Legal-RAG/backend/ingestion/chunking_embedding.py)):**
> - **Bước 1 (Phase 1 — Chunking VB gốc):** Tải dataset từ HuggingFace, lọc qua CSV 8000 VB ưu tiên lĩnh vực Y tế. Chunker FSM xử lý tuần tự batch 4 VB/lần. Quan hệ + Entity được trích xuất single-pass qua Unified LLM Prompt.
> - **Bước 2 (Phase 2-3 — VB tham chiếu):** Tự động phát hiện VB được nhắc đến trong quan hệ ontology → chunking tiếp depth=1 và depth=2.
> - **Bước 3 (Phase 4 — Ghost Nodes):** Tạo Ghost Node cho VB được tham chiếu nhưng không có trong dataset.
> - **Bước 4 (Phase 5 — Embedding):** Embed BGE-M3 Dense + fastembed BM25 Sparse → Upsert Qdrant Collection (`legal_rag_docs_nam`).
> - **Bước 5 (Phase 6 — Neo4j Build):** Build Document Tree + Ghost Nodes MERGE + Graph Triplets (free-form) + Entity Enrichment (10 loại entity từ Unified Extractor). Khép kín vòng đời tái chỉ mục RAG.

---

# Phần 4: Kiểm Định Chất Lượng Chunking (Quality Audit)


## Phụ lục B: Luồng Retrieval Runtime (Search Pipeline)

Khi user hỏi câu hỏi truy vấn, luồng tìm kiếm RAG đi qua các pha chuyển giao (mã nguồn: [hybrid_search.py](file:///d:/iCOMM/Legal-RAG/backend/retrieval/hybrid_search.py), [graph_search.py](file:///d:/iCOMM/Legal-RAG/backend/retrieval/graph_search.py)):

1. **Phân tích Query (SuperRouter):** Chạy Intent Detection + sinh HyDE + rút metadata filter.
2. **Encode Mảng Kép:** Gọi `Dense Encode` qua REST API kết hợp module tự vận hành BM25 `Sparse Encode` local.
3. **Mẫu Phân Tầng (Tiered Prefetch):** Gọi lấy Main Content Top K ưu tiên song hành mảng bổ trợ Phụ lục. Thuật toán RRF khử chênh lệch.
4. **Xếp Hạng (Reranker):** Gửi list hits gộp sang mô hình Cross-Encoder để thu core ranking. Tính bổ sung Typology weight.
5. **Cứu Hộ Lọc Rác (Keyword Rescue):** Giải thoát các Broad hits đang ôm sát 2-3 cụm từ khoá gốc nhưng bị mô hình Vector trảm rớt đài vô lý.
6. **Mở Rộng Context:** Nhóm Small-To-Big quét window rải rác 2 phía sườn chunk gốc.
7. **Trả Json (Format Results):** Sửa soạn payload RAG chu đáo để Agent LLM khởi sinh câu trả lời.

> [!IMPORTANT]
> **Critical Fallback Mechanisms:**
> - Nếu vector search trả 0 hits nhưng có `doc_number` → **Metadata Scroll** lấy toàn bộ chunks theo `document_number`.
> - Nếu vẫn 0 hits → **Emergency Regex** trích xuất số hiệu từ query rồi Scroll.
> - Nếu reranker bỏ sót chunks chứa keyword chính xác → **Keyword Rescue** inject lại.
