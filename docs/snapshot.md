# 📸 Hướng dẫn Tạo Snapshot cho Qdrant (Legal-RAG)

Snapshot là bản sao chép vật lý của cơ sở dữ liệu vector, giúp bạn sao lưu và khôi phục dữ liệu một cách nhanh chóng mà không cần nạp (ingest) lại từ đầu.

---

## 1. Tạo Snapshot cho Collection cụ thể

Đây là cách phổ biến nhất để sao lưu một tập dữ liệu (ví dụ: `legal_rag_docs_5000`).

### Cách A: Sử dụng cURL (Command Line)
Chạy lệnh sau trong Terminal (đảm bảo Qdrant đang chạy):
```bash
curl.exe -X POST http://localhost:6335/collections/legal_rag_docs_5000/snapshots
```

### Cách B: Sử dụng Qdrant Dashboard (Giao diện)
1. Truy cập: `http://localhost:6335/dashboard`
2. Chọn collection: **legal_rag_docs_5000**.
3. Chuyển sang Tab **Snapshots**.
4. Nhấn nút **Create Snapshot**.

---

## 2. Tạo Snapshot cho Toàn bộ Storage
Nếu bạn muốn sao lưu tất cả collections và cấu hình hệ thống:
```bash
curl.exe -X POST http://localhost:6335/snapshots
```

---

## 3. Quản lý và Truy cập File Snapshot

- **Vị trí lưu trữ**: Các file `.snapshot` sẽ được lưu tại thư mục:
  `./qdrant_snapshots/legal_rag_docs_5000/` (đã được mount qua Docker).
- **Liệt kê danh sách snapshot hiện có**:
  ```bash
  curl.exe -s http://localhost:6335/collections/legal_rag_docs_5000/snapshots
  ```

---

## ⚠️ Giải quyết lỗi thường gặp

### Lỗi 500 Internal Server Error
Nếu bạn nhận được lỗi 500, có thể do Collection đang ở trạng thái **RED** (Lỗi/Mất đồng bộ). 
**Cách kiểm tra:**
```bash
curl.exe -s http://localhost:6335/collections/legal_rag_docs_5000
```
Nếu `status: "red"`, hãy thử:
1. **Khởi động lại Docker**: `docker-compose restart qdrant`.
2. **Kiểm tra dung lượng đĩa**: Đảm bảo ổ D: còn trống (ít nhất 2-5GB).
3. **Đợi Optimization**: Đôi khi Qdrant đang bận tối ưu hóa dữ liệu, hãy đợi vài phút rồi thử lại.

---

## 🔄 Cách Khôi phục (Restore)
1. Copy file `.snapshot` vào thư mục `./qdrant_snapshots/`.
2. Truy cập Dashboard -> Chọn Collection -> Snapshots.
3. Chọn file và nhấn **Restore**.
