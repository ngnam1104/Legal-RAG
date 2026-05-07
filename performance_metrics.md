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
- **Tổng quy mô**: ~21,000 văn bản (được thu thập qua 3 tầng từ `chunking_embedding.py`):
    1. **Văn bản gốc**: Danh sách ưu tiên từ 8,000 VB y tế cốt lõi.
    2. **Tham chiếu cấp 1**: Các VB được trích dẫn trực tiếp từ nhóm gốc.
    3. **Tham chiếu cấp 2**: Các VB được trích dẫn tiếp từ nhóm cấp 1 để bao phủ toàn bộ mạng lưới quy định.
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

---

## 4. L? tr�nh T?i uu h�a (Optimization Roadmap)

D?a tr�n c�c ch? s? tr�n, h? th?ng c?n t?p trung v�o c�c h?ng m?c sau:

1. **Gi?m d? tr? Graph (Shortest Path)**: 
   - Tri?n khai **Graph Caching** cho c�c c?p Concept ph? bi?n.
   - S? d?ng **Neo4j GDS (Graph Data Science)** d? t�nh to�n tru?c c�c tr?ng s? li�n k?t.
2. **T?i uu h�a Ingestion**:
   - S? d?ng **Parallel Processing** ? c?p d? Chunk (hi?n dang ? c?p d? Document) d? t?n d?ng t?i da GPU/CPU khi nh�ng vector.
   - Batching c�c l?nh `MERGE` trong Neo4j d? gi?m overhead giao d?ch.
3. **C?i thi?n d? ch�nh x�c (Accuracy)**:
   - Tinh ch?nh bu?c **Rerank** (hi?n dang d�ng BGE-M3) d? l?c nhi?u t?t hon sau khi m? r?ng d? th?.
   - B? sung bu?c **Query Expansion** b?ng LLM d? c?i thi?n t? l? Hit Rate trong Qdrant.

### Ph�n r� chi ti?t bu?c Retrieve + Graph Expand (33.26s)
D?a tr�n ph�n t�ch m� ngu?n h? th?ng, bu?c n�y bao g?m 5 giai do?n ch�nh v?i th?i gian u?c lu?ng:

| Giai do?n | Th?i gian u?c lu?ng | Chi ti?t c�ng vi?c |
| :--- | :--- | :--- |
| **Phase 0 & 1 (Parallel)** | 5 - 8s | Ch?y song song Vector Search (Qdrant) v� Entity Search (Neo4j). |
| **Graph_Doc_Fetch** | 2 - 4s | Truy xu?t c�c van b?n du?c d? th? g?i � nhung Vector Search b? s�t. |
| **Unified Reranking** | **8 - 12s** | **N�t th?t ch�nh**: Ch?y m� h�nh Cross-Encoder d? x?p h?ng l?i d? ch�nh x�c c?a c�c ?ng vi�n. |
| **QdrantNeo4j Enrich** | 3 - 5s | L�m gi�u th�ng tin Metadata (Ngu?i k�, Ng�y hi?u l?c, M?c l?c) t? Neo4j. |
| **Neo4j Subgraph Expand** | **10 - 15s** | **N�t th?t ch�nh**: Ch?y 4 truy v?n Cypher tu?n t? d? l?y d? th? con 2 bu?c (2-hop) v� van b?n li�n quan (Sibling Docs). |

> [!IMPORTANT]
> **�i?m c?n t?i uu**: Bu?c **Unified Reranking** v� **Neo4j Subgraph Expand** dang chi?m t?i ~75% th?i gian c?a giai do?n Retrieve. Vi?c chuy?n c�c truy v?n Neo4j sang ch?y song song v� t?i uu h�a m� h�nh Rerank s? gi�p gi?m d�ng k? t?ng th?i gian ph?n h?i.
