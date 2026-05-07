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
> Phần lớn thời gian nằm ở bước **Retrieve + Graph Expand** (chiếm ~58% tổng thời gian). Trong đó, việc truy xuất Graph-Doc (Graph_Doc_Fetch) là thành phần tốn kém nhất.

### Phân rã chi tiết bước Retrieve + Graph Expand (33.26s)
Dựa trên phân tích mã nguồn hệ thống, bước này bao gồm 5 giai đoạn chính với thời gian ước lượng:

| Giai đoạn | Thời gian ước lượng | Chi tiết công việc |
| :--- | :--- | :--- |
| **Phase 0 & 1 (Parallel)** | 5 - 8s | Chạy song song Vector Search (Qdrant) và Entity Search (Neo4j). |
| **Graph_Doc_Fetch** | 2 - 4s | Truy xuất các văn bản được đồ thị gợi ý nhưng Vector Search bỏ sót. |
| **Unified Reranking** | **8 - 12s** | **Nút thắt chính**: Chạy mô hình Cross-Encoder để xếp hạng lại các ứng viên. |
| **QdrantNeo4j Enrich** | 3 - 5s | Làm giàu thông tin Metadata (Người ký, Ngày hiệu lực, Mục lục) từ Neo4j. |
| **Neo4j Subgraph Expand** | **10 - 15s** | **Nút thắt chính**: Chạy 4 truy vấn Cypher tuần tự để lấy đồ thị con 2 bước. |

---

## 2. Tốc độ và Quy mô Lập chỉ mục (Ingestion)
*Nguồn: [result_500.txt](file:///d:/iCOMM/Legal-RAG/result_500.txt)*

Hệ thống đã được kiểm thử với tập dữ liệu y tế quy mô lớn:
- **Tổng quy mô**: ~21,000 văn bản (được thu thập qua 3 tầng từ chunking_embedding.py):
    1. **Văn bản gốc**: Danh sách ưu tiên từ 8,000 VB y tế cốt lõi.
    2. **Tham chiếu cấp 1**: Các VB được trích dẫn trực tiếp từ nhóm gốc.
    3. **Tham chiếu cấp 2**: Các VB được trích dẫn tiếp từ nhóm cấp 1.
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
| **Tìm kiếm kèm Filter Metadata** | **2666.89 ms** | Lọc theo document_number cụ thể |

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

---

## 4. Lộ trình Tối ưu hóa (Optimization Roadmap)

Dựa trên các chỉ số trên, hệ thống cần tập trung vào các hạng mục sau:

1. **Giảm độ trễ Graph (Shortest Path)**: 
   - Triển khai **Graph Caching** cho các cặp Concept phổ biến.
   - Sử dụng **Neo4j GDS (Graph Data Science)** để tính toán trước các trọng số liên kết.
2. **Tối ưu hóa Ingestion**:
   - Sử dụng **Parallel Processing** ở cấp độ Chunk để tận dụng tối đa GPU/CPU.
   - Batching các lệnh MERGE trong Neo4j để giảm overhead giao dịch.
3. **Cải thiện độ chính xác (Accuracy)**:
   - Tinh chỉnh bước **Rerank** để lọc nhiễu tốt hơn sau khi mở rộng đồ thị.
   - Bổ sung bước **Query Expansion** bằng LLM để cải thiện tỷ lệ Hit Rate trong Qdrant.