# Báo Cáo Pipeline Legal-RAG
**Thời điểm chạy:** 2026-04-29 16:59:15  
**Collection:** `legal_rag_docs_nam`  
**Chế độ:** TEST MODE — 2 văn bản gốc (SAMPLE_LIMIT = 2)

---

## Bước 1 — Lọc Chunk Bằng Regex (Tiền xử lý)

Trước khi gọi LLM, pipeline áp dụng một **bộ lọc heuristics hai tầng** để chỉ giữ lại các đoạn văn thực sự có giá trị:

### 1a. Lọc chunk tiềm năng chứa Quan hệ Văn bản (Doc-Doc)

Một chunk được đánh dấu **"tiềm năng chứa quan hệ"** khi **đồng thời** thỏa mãn:

| Điều kiện | Mô tả | Ví dụ từ khóa |
|---|---|---|
| **Có động từ quan hệ** | Regex `_RELATION_VERBS` | sửa đổi, bổ sung, thay thế, bãi bỏ, hết hiệu lực, căn cứ, ủy quyền... |
| **Có số hiệu văn bản** | Regex `_DOC_NUM_HINT` | `15/2020/NĐ-CP`, `80/2015/QH13`... |

> Yêu cầu **cả hai điều kiện** để giảm tối đa false-positive (tránh nhầm tham chiếu nội văn bản).

### 1b. Lọc chunk tiềm năng chứa Thực thể (Entity)

Regex `_ENTITY_KEYWORDS` tìm các chunk có nhắc đến:

- Cơ quan: bộ, sở, ban, ngành, ủy ban, tổng cục, cục, chi cục, đoàn, hội đồng
- Tổ chức: chính phủ, quốc hội, đảng, nhà nước, trung ương, doanh nghiệp, công ty, tập đoàn
- Chương trình: đề án, dự án, chương trình, kế hoạch, chiến lược, quỹ

### 1c. Phát hiện Passive Chain (Chuỗi Bị Động)

Regex `PASSIVE_BETWEEN` phát hiện mẫu:  
> *"Căn cứ [Văn bản A] đã được sửa đổi, bổ sung ... theo [Văn bản B]"*

→ Sinh ra **2 quan hệ đồng thời**:
- `X --BASED_ON--> A` (trực tiếp)
- `B --AMENDS--> A` (cross-doc, bị động)

---

## Bước 2 — PHASE 1: Chunking Văn Bản GỐC

### Nguồn văn bản
- **Số lượng:** 2 văn bản gốc (lấy từ CSV `top_8000_y_te_theo_quyen_luc.csv`, giới hạn SAMPLE_LIMIT=2)
- **Dataset:** HuggingFace `nhn309261/vietnamese-legal-docs` (offline cache)
- **Cách lấy:** Lấy 2 ID đầu tiên từ danh sách 8.000 văn bản ưu tiên lĩnh vực y tế

### Prompt LLM (tóm tắt)

Prompt `LEGAL_UNIFIED_EXTRACTOR_PROMPT` yêu cầu LLM thực hiện **đồng thời 3 nhiệm vụ** từ các đoạn văn bản pháp luật:

| Nhiệm vụ | Đầu ra | Ghi chú |
|---|---|---|
| **Nhiệm vụ 1** — Quan hệ văn bản | `doc_relations[]` | Trích xuất tất cả quan hệ pháp lý giữa các văn bản (BASED_ON, AMENDS, REPEALS, REPLACES, GUIDES, APPLIES_TO, ISSUED_WITH, ASSIGNS, CORRECTS) |
| **Nhiệm vụ 2** — Thực thể | `entities{}` | Trích xuất các thực thể: Organization, Person, Location, LegalArticle, Procedure, Condition, Fee, Penalty, Timeframe, Role, Concept |
| **Nhiệm vụ 3** — Quan hệ thực thể | `node_relations[]` | Trích xuất quan hệ giữa các thực thể (ISSUED_BY, SIGNED_BY, REGULATED_BY, MANAGED_BY, ASSIGNS_TO, REQUIRED_FOR, DEFINED_IN, LOCATED_IN, PART_OF...) |

**Quy tắc nghiêm ngặt trong prompt:**
- `source` và `target` phải là số hiệu chuẩn (VD: `44/2019/QH14`), **tuyệt đối không** dùng "Luật này", "Nghị định này"
- Dedup: mỗi bộ (source, target, edge_label) chỉ xuất hiện **một lần**
- Không trích xuất đại từ chỉ định mơ hồ: "Cơ quan này", "Tổ chức đó"
- Tên tổ chức phải viết đầy đủ: "Bộ GDĐT" → **"Bộ Giáo dục và Đào tạo"**

### Kết quả Phase 1

#### Lĩnh vực (2 văn bản gốc)
| Lĩnh vực | Số văn bản |
|---|---|
| Thể thao - Y tế | 2 |
| Bộ máy hành chính | 1 |

#### Quan hệ Văn bản (Doc Relations)
| Loại quan hệ | Số lượng | Dịch nghĩa |
|---|---|---|
| **DỰA_TRÊN** (BASED_ON) | 11 | Văn bản A căn cứ/dựa trên văn bản B |
| **SỬA_ĐỔI** (AMENDS) | 1 | Văn bản A sửa đổi, bổ sung văn bản B |
| **BÃI_BỎ** (REPEALS) | 1 | Văn bản A bãi bỏ/hết hiệu lực văn bản B |
| **HƯỚNG_DẪN** (GUIDES) | 1 | Văn bản A hướng dẫn thi hành văn bản B |
| **Tổng** | **14** | |

*Không có quan hệ cross-doc bị động; không có trùng lặp bị bỏ qua.*

#### Thực thể (Entities)
| Loại | Số lượng | Dịch nghĩa |
|---|---|---|
| **Tổ_chức** (Organization) | 45 | Bộ, Sở, Ủy ban, Cục... |
| **Khung_thời_gian** (Timeframe) | 16 | "30 ngày", "12 tháng"... |
| **Khái_niệm** (Concept) | 11 | Định nghĩa, thuật ngữ pháp lý... |
| **Vai_trò** (Role) | 7 | Giám đốc, Trưởng phòng... |
| **Phí_lệ_phí** (Fee) | 5 | "10.000.000 đồng"... |
| **Địa_điểm** (Location) | 3 | Tỉnh, thành phố... |
| **Điều_kiện** (Condition) | 2 | Điều kiện bắt buộc, tiêu chuẩn... |
| **Người** (Person) | 1 | Họ tên cụ thể |

#### Quan hệ Thực thể (Node Relations)
| Loại quan hệ | Số lượng | Dịch nghĩa |
|---|---|---|
| **GIAO_PHÂN_CÔNG** (ASSIGNS_TO) | 94 | A giao nhiệm vụ/phân công cho B |
| **BAN_HÀNH_BỞI** (ISSUED_BY) | 24 | Văn bản/quyết định được ban hành bởi cơ quan A |
| **ĐƯỢC_ĐIỀU_CHỈNH_BỞI** (REGULATED_BY) | 24 | Đối tượng A chịu sự điều chỉnh của B |
| **YÊU_CẦU_ĐỂ** (REQUIRED_FOR) | 12 | A là điều kiện bắt buộc để B |
| **ÁP_DỤNG_CHO** (APPLIES_TO) | 12 | Quy định A áp dụng cho đối tượng B |
| **KÝ_BỞI** (SIGNED_BY) | 1 | Văn bản được ký bởi người A |
| **GIỮ_VAI_TRÒ** (HOLDS_ROLE) | 1 | Người A giữ vai trò B |

---

## Bước 3 — PHASE 2: Chunking Văn Bản Tham Chiếu (Vệ Tinh 1)

### Nguồn văn bản
- **Số lượng:** Các văn bản được **tham chiếu bởi 2 văn bản gốc** trong Phase 1 và **tồn tại trong dataset** HuggingFace
- **Cách tìm:** Pipeline tự động thu thập `target_doc` từ các `doc_relations` ở Phase 1 → tra cứu ID trong `meta_docnum_to_id` → lọc bỏ đã xử lý

> Từ 2 văn bản gốc → **phát hiện thêm ~9 văn bản vệ tinh** 

### Prompt LLM
Giống hệt Phase 1 — cùng `LEGAL_UNIFIED_EXTRACTOR_PROMPT`, không thay đổi.

### Kết quả Phase 2 (lũy kế từ đầu)

#### Lĩnh vực (lũy kế)
| Lĩnh vực | Số văn bản |
|---|---|
| Thể thao - Y tế | 7 |
| Bộ máy hành chính | 5 |
| Tài chính nhà nước | 1 |
| Xây dựng - Đô thị | 1 |
| Lĩnh vực khác | 1 |

#### Quan hệ Văn bản (lũy kế)
| Loại quan hệ | Số lượng | Dịch nghĩa |
|---|---|---|
| **DỰA_TRÊN** (BASED_ON) | 57 | +46 so với Phase 1 |
| **BÃI_BỎ** (REPEALS) | 6 | +5 |
| **SỬA_ĐỔI** (AMENDS) | 2 | +1 |
| **HƯỚNG_DẪN** (GUIDES) | 2 | +1 |
| **PHÂN_CÔNG** (ASSIGNS) | 1 | Mới xuất hiện |
| **Tổng** | **68** | +54 so với Phase 1 |

#### Thực thể (lũy kế — tăng mạnh do văn bản vệ tinh phong phú hơn)
| Loại | Số lượng | Tăng so với Phase 1 |
|---|---|---|
| **Khái_niệm** (Concept) | 442 | +431 |
| **Tổ_chức** (Organization) | 243 | +198 |
| **Địa_điểm** (Location) | 197 | +194 |
| **Vai_trò** (Role) | 115 | +108 |
| **Khung_thời_gian** (Timeframe) | 96 | +80 |
| **Thủ_tục** (Procedure) | 50 | Mới |
| **Điều_kiện** (Condition) | 28 | +26 |
| **Người** (Person) | 13 | +12 |
| **Phí_lệ_phí** (Fee) | 5 | 0 |

#### Quan hệ Thực thể (lũy kế — top 10)
| Loại quan hệ | Số lượng | Dịch nghĩa |
|---|---|---|
| **GIAO_PHÂN_CÔNG** (ASSIGNS_TO) | 812 | |
| **ĐƯỢC_ĐIỀU_CHỈNH_BỞI** (REGULATED_BY) | 516 | |
| **YÊU_CẦU_ĐỂ** (REQUIRED_FOR) | 324 | |
| **QUẢN_LÝ_BỞI** (MANAGED_BY) | 288 | |
| **BAN_HÀNH** (ISSUES) | 180 | |
| **BAN_HÀNH_BỞI** (ISSUED_BY) | 179 | |
| **BÁO_CÁO_LÊN** (REPORTS_TO) | 128 | |
| **HỢP_TÁC_VỚI** (COOPERATES_WITH) | 99 | |
| **ĐIỀU_CHỈNH** (REGULATES) | 96 | |
| **DỰA_TRÊN** (BASED_ON) | 96 | |

> *Lưu ý: Phase 2 xuất hiện nhiều nhãn quan hệ mới như ISSUES, REPORTS_TO, APPROVES, FUNDS, HEADS, LEADS... do nội dung văn bản vệ tinh phong phú hơn (nghị định, thông tư chi tiết hơn).*

---

## Bước 4 — PHASE 3: Chunking Văn Bản Tham Chiếu Độ Sâu 2 (Vệ Tinh 2)

### Nguồn văn bản
- **Số lượng:** Các văn bản được **tham chiếu bởi các văn bản vệ tinh 1** trong Phase 2 và tồn tại trong dataset
- **Cách tìm:** Pipeline thu thập `target_doc` từ `doc_relations` Phase 2 → tra cứu ID → lọc đã xử lý
- **Tổng sau Phase 3:** 13 văn bản xử lý (cộng thêm ~13 văn bản vệ tinh 2)

### Prompt LLM
Giống hệt Phase 1 & 2 — cùng `LEGAL_UNIFIED_EXTRACTOR_PROMPT`.

### Kết quả Phase 3 (lũy kế cuối)

#### Lĩnh vực (lũy kế)
| Lĩnh vực | Số văn bản |
|---|---|
| Thể thao - Y tế | 17 |
| Bộ máy hành chính | 15 |
| Xây dựng - Đô thị | 5 |
| Tài chính nhà nước | 4 |
| Đầu tư | 3 |

#### Quan hệ Văn bản (lũy kế cuối)
| Loại quan hệ | Số lượng | Dịch nghĩa |
|---|---|---|
| **DỰA_TRÊN** (BASED_ON) | 144 | |
| **BÃI_BỎ** (REPEALS) | 16 | |
| **SỬA_ĐỔI** (AMENDS) | 14 | |
| **THAY_THẾ** (REPLACES) | 8 | Mới xuất hiện |
| **HƯỚNG_DẪN** (GUIDES) | 4 | |
| **ÁP_DỤNG** (APPLIES) | 2 | |
| **PHÂN_CÔNG** (ASSIGNS) | 1 | |
| **ÁP_DỤNG_CHO** (APPLIES_TO) | 1 | |
| **Tổng** | **190** | |

#### Thực thể (lũy kế cuối)
| Loại | Số lượng |
|---|---|
| **Khái_niệm** (Concept) | 937 |
| **Tổ_chức** (Organization) | 608 |
| **Địa_điểm** (Location) | 357 |
| **Vai_trò** (Role) | 267 |
| **Khung_thời_gian** (Timeframe) | 224 |
| **Thủ_tục** (Procedure) | 122 |
| **Điều_kiện** (Condition) | 70 |
| **Người** (Person) | 32 |
| **Phí_lệ_phí** (Fee) | 30 |
| **Hình_phạt** (Penalty) | 4 |
| **Tổng** | **2.651** |

#### Quan hệ Thực thể (lũy kế cuối — top 15)
| Loại quan hệ | Số lượng | Dịch nghĩa |
|---|---|---|
| **GIAO_PHÂN_CÔNG** (ASSIGNS_TO) | 1.579 | A giao nhiệm vụ cho B |
| **ĐƯỢC_ĐIỀU_CHỈNH_BỞI** (REGULATED_BY) | 1.078 | Đối tượng chịu sự điều chỉnh của cơ quan B |
| **QUẢN_LÝ_BỞI** (MANAGED_BY) | 715 | A được quản lý bởi B |
| **BAN_HÀNH_BỞI** (ISSUED_BY) | 630 | Văn bản do cơ quan B ban hành |
| **ÁP_DỤNG_CHO** (APPLIES_TO) | 411 | Quy định áp dụng cho đối tượng B |
| **YÊU_CẦU_ĐỂ** (REQUIRED_FOR) | 384 | A là điều kiện cần để B |
| **BÁO_CÁO_LÊN** (REPORTS_TO) | 338 | A báo cáo lên cơ quan B |
| **ĐIỀU_CHỈNH** (REGULATES) | 264 | A ban hành quy định điều chỉnh B |
| **PHÊ_DUYỆT** (APPROVES) | 252 | A phê duyệt/chấp thuận B |
| **BAN_HÀNH** (ISSUES) | 245 | A ban hành văn bản/quyết định |
| **KÝ_BỞI** (SIGNED_BY) | 213 | Văn bản được ký bởi A |
| **GIỮ_VAI_TRÒ** (HOLDS_ROLE) | 176 | Người A giữ vai trò B |
| **TÀI_TRỢ** (FUNDS) | 168 | A tài trợ/cấp ngân sách cho B |
| **THÀNH_PHẦN_CỦA** (PART_OF) | 132 | A là thành phần/bộ phận của B |
| **THÀNH_LẬP** (ESTABLISHES) | 132 | A quyết định thành lập B |

> Tổng số nhãn quan hệ thực thể khác nhau: **68 loại** — cho thấy độ phong phú của đồ thị tri thức pháp lý.

---

## Bước 5 — QUERY BENCHMARK

**Mục đích:** Kiểm tra khả năng truy vấn thực tế trên cả 2 cơ sở dữ liệu (Qdrant + Neo4j) sau khi ingestion xong.

**Câu truy vấn mẫu:** `"Luật số 80/2015/QH13 ban hành văn bản quy phạm pháp luật"`

### Kết quả Benchmark

| Kịch bản | Thời gian (ms) | Số kết quả | Trạng thái |
|---|---|---|---|
| Tìm kiếm Dense vector (top-5) | 113,4 | 5 | ✅ OK |
| Tìm kiếm Sparse/BM25 (top-5) | 18,7 | 5 | ✅ OK |
| Tìm kiếm Hybrid RRF (top-5) | 54,9 | 5 | ✅ OK |
| Lọc theo `legal_type='Luật'` (top-5) | 173,6 | 5 | ✅ OK |
| Lọc theo `year='2015'` (top-5) | 99,6 | 5 | ✅ OK |
| Scroll lọc `is_table=True` (10 bản) | 40,1 | 10 | ✅ OK |
| Đếm tổng Document nodes (Neo4j) | 47,5 | 1 | ✅ OK |
| Đếm tổng quan hệ (Neo4j) | 12,6 | 1 | ✅ OK |
| Đếm ghost nodes (`is_ghost=true`) | 27,5 | 1 | ✅ OK |
| Tìm văn bản `80/2015/QH13` | 24,8 | 0 | ✅ OK* |
| Lấy tất cả quan hệ của `80/2015/QH13` | 36,4 | 0 | ✅ OK* |
| Tìm cạnh REPEALS (limit 10) | 25,9 | 0 | ✅ OK* |
| Tìm cạnh GUIDES (limit 10) | 19,2 | 4 | ✅ OK |
| Đường đi ngắn nhất (2 hop) giữa 2 văn bản | 120,0 | 1 | ✅ OK |
| Đếm cạnh theo loại quan hệ | 54,9 | 80 | ✅ OK |

> *0 kết quả cho `80/2015/QH13` là bình thường trong TEST MODE (2 văn bản gốc), văn bản đó chưa được ingestion.

**Tổng kết:** 15/15 kịch bản thành công | 0 lỗi | Thời gian trung bình: **57,9 ms**

---

## Tổng Kết Pipeline

| Chỉ số | Giá trị |
|---|---|
| Văn bản gốc (Phase 1) | 2 |
| Văn bản tham chiếu Phase 2 (Vệ tinh 1) | ~11 |
| Văn bản Ghost Nodes (tham chiếu nhưng không có trong dataset) | 92 |
| Tổng chunks | 1.368 |
| Tổng quan hệ văn bản | 190 |
| Tổng thực thể | 2.651 |
| Tổng quan hệ thực thể | 9.028 |
