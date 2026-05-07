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
