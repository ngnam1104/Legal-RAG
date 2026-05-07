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

Hệ thống đã được kiểm thử với tập dữ liệu y tế quy mô lớn:
- **Tổng quy mô**: ~21,000 văn bản (PDF/DOCX).
- **Tổng số Points (Qdrant)**: ~672,185 points.
- **Thời gian xử lý trung bình**: **23.75s / văn bản**.
- **Công suất xử lý**: ~150 văn bản/giờ (với cấu hình LLM hiện tại).

---

## 3. Hiệu năng Cơ sở dữ liệu (Database Diagnostics)
*Nguồn: [test_results.txt](file:///d:/iCOMM/Legal-RAG/test_results.txt)*

Kết quả đo lường trực tiếp trên cụm Server (10.9.2.57) cho thấy tốc độ truy vấn ở mức Database cực kỳ tối ưu:

### 3.1. Vector Search (Qdrant)
| Kịch bản | Độ trễ (Latency) | Ghi chú |
| :--- | :--- | :--- |
| **Tìm kiếm ngữ nghĩa cơ bản** | **34.98 ms** | Top 5 tương đồng trên 670k points |
| **Tìm kiếm kèm Filter Metadata** | **2666.89 ms** | Lọc theo `document_number` cụ thể |

### 3.2. Graph Traversal (Neo4j)
| Kịch bản | Độ trễ (Latency) | Ý nghĩa |
| :--- | :--- | :--- |
| **Tra cứu Node (Lookup)** | 11.38 ms | Tìm Organization theo tên |
| **Phân cấp Văn bản (Hierarchy)** | 21.26 ms | Lấy cây: Doc -> Article -> Clause -> Chunk |
| **Bán kính 2 bước (2-hop)** | 444.15 ms | Phân tích quan hệ gián tiếp quy mô lớn |
| **Phân tích tác động sâu** | 702.19 ms | Lan truyền ảnh hưởng qua 3 bước nhảy |
| **Đường đi ngắn nhất (Shortest Path)** | **44.00 s** | Tìm liên kết giữa 2 khái niệm xa nhau |

> [!TIP]
> **Nhận xét**: 
> 1. Tốc độ truy vấn thô của DB rất nhanh (hầu hết < 50ms).
> 2. Sự chênh lệch giữa 44s (Shortest Path) và 56s (Tổng Pipeline) cho thấy các thuật toán đồ thị phức tạp là nhân tố chính ảnh hưởng đến trải nghiệm người dùng cuối. 
> 3. Cần tối ưu hóa việc cache kết quả Shortest Path cho các khái niệm phổ biến để giảm tải cho CPU Neo4j.
