# Báo cáo Hiệu năng Hệ thống Legal-RAG

Báo cáo này tổng hợp các chỉ số hiệu năng về tốc độ phản hồi và tốc độ lập chỉ mục dữ liệu dựa trên các kết quả đo lường thực tế.

## 1. Tốc độ Phản hồi (QA Evaluation)
*Nguồn: [metrics_report.txt](file:///d:/iCOMM/Legal-RAG/tests/qa_evaluation/metrics_report.txt)*

| Chỉ số | Kết quả |
| :--- | :--- |
| **Tổng số test cases** | 114 |
| **Thời gian phản hồi trung bình** | **56.78s** |
| **Tỷ lệ chính xác nội dung** | 71.93% (82/114) |
| **Tỷ lệ đúng Intent** | 100.00% (114/114) |

### Phân rã thời gian Pipeline (Step Breakdown)
| Bước thực thi | Thời gian (Trung bình/Turn) |
| :--- | :--- |
| ⚡ Preprocess Memory/Files | 0.00s |
| ⚡ Detect Mode Only | 6.04s |
| ⚡ 1. Understand | 0.00s |
| ⚡ 2. Retrieve + Graph Expand | **33.26s** |
| ⚡ 3. Generate | 15.90s |

> [!NOTE]
> Phần lớn thời gian nằm ở bước **Retrieve + Graph Expand** (chiếm ~58% tổng thời gian). Trong đó, việc truy xuất Graph-Doc (`Graph_Doc_Fetch`) là thành phần tốn kém nhất.

---

## 2. Tốc độ và Quy mô Lập chỉ mục (Ingestion)
*Nguồn: [result_500.txt](file:///d:/iCOMM/Legal-RAG/result_500.txt)*

### Quy mô Dữ liệu (Cumulative Statistics - Phase 3)
Hệ thống đã hoàn tất nạp dữ liệu với quy mô:
- **Tổng số văn bản gốc**: 500 văn bản (Chủ yếu thuộc lĩnh vực Thể thao - Y tế)
- **Tổng số thực thể (Entities)**: ~22,600 (Tổ chức), ~18,300 (Khái niệm), ~9,600 (Vai trò)
- **Tổng số quan hệ (Relationships)**: ~17,381 quan hệ văn bản, ~33,259 quan hệ thực thể.

### Điểm chuẩn Truy vấn (Query Benchmark)
Đo lường tốc độ truy vấn trực tiếp trên cơ sở dữ liệu sau khi index:
- **Dense search (top-5)**: 39.3ms
- **Sparse (BM25) search (top-5)**: 5.9ms
- **Hybrid search RRF (top-5)**: 19.0ms
- **Find doc by number**: 94.8ms
- **Thời gian truy vấn trung bình (Avg Benchmark)**: **111.0ms**

---

## 3. Đánh giá chung
- **Tốc độ truy vấn (Database Level)**: Rất nhanh (trung bình ~111ms), cho thấy cấu trúc Index và Graph đang hoạt động hiệu quả.
- **Tốc độ phản hồi (End-to-End)**: Còn khá chậm (~56s), nguyên nhân chính do quá trình mở rộng đồ thị (Graph Expand) và gọi LLM nhiều lần trong pipeline.
- **Độ chính xác**: Intent đạt tuyệt đối, tuy nhiên nội dung cần tối ưu thêm để vượt ngưỡng 71%.
