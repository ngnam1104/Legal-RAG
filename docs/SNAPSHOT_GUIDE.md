# 📸 Hướng dẫn sử dụng Qdrant Snapshot (Backup & Di động)

Khi bạn đã có 242.000 points dữ liệu quý giá, việc sử dụng **Snapshot** là cách chuyên nghiệp nhất để đóng gói, sao lưu hoặc di chuyển toàn bộ DB sang một thiết bị khác chỉ với một file duy nhất.

---

## 1. Cách Tạo (Create) Snapshot
Sau khi dữ liệu đã được nạp xong và Qdrant đang chạy trong Docker Compose:

### **Sử dụng Terminal (PowerShell/Bash):**
Chạy lệnh `curl` sau để ra lệnh cho Qdrant tự nén dữ liệu:

```powershell
curl -X POST "http://localhost:6335/collections/legal_rag_docs_5000/snapshots"
```

### **Kết quả:**
*   File snapshot sẽ được tạo ra tại thư mục: `./qdrant_snapshots/` (trong thư mục gốc dự án `Legal-RAG`).
*   Tên file sẽ có dạng: `legal_rag_docs_5000-YYYY-MM-DD-HH-mm-ss.snapshot`.
*   File này chứa **toàn bộ** Vector, Payload và Cấu hình Indexing.

---

## 2. Cách Khôi phục (Restore) ở máy mới
Khi bạn chuyển sang một máy tính mới, hãy thực hiện các bước sau:

### **Bước 1: Chuẩn bị file**
*   Copy file `.snapshot` vừa tạo vào thư mục `d:\iCOMM\Legal-RAG\qdrant_snapshots` ở máy mới.

### **Bước 2: Khởi động Docker Qdrant trắng**
*   Chạy lệnh `docker-compose up -d`.

### **Bước 3: Thực hiện Restore**
Chạy lệnh sau trong Terminal máy mới (thay tên file đúng với thực tế):

```powershell
curl -X POST "http://localhost:6335/collections/legal_rag_docs_5000/snapshots/recover" `
     -H "Content-Type: application/json" `
     -d '{ "location": "http://localhost:6333/snapshots/legal_rag_docs_5000-TEN-FILE-CUA-BAN.snapshot" }'
```

> **Lưu ý**: Trong Docker, Qdrant tự hiểu cổng nội bộ là `6333` và đường dẫn file nằm trong volume mount.

---

## 3. Cách Khôi phục qua Giao diện (Dashboard)
Nếu bạn không quen dùng lệnh Terminal:

1.  Truy cập: [http://localhost:6335/dashboard](http://localhost:6335/dashboard)
2.  Chọn Collection `legal_rag_docs_5000`.
3.  Tìm mục **Snapshots**.
4.  Bạn có thể **Upload** trực tiếp file snapshot từ máy tính của mình lên và nhấn nút **Restore** (Phục hồi).

---

## 💡 Mẹo nhỏ (Best Practices):
*   **Dùng Git**: Chỉ đẩy các file `.md`, `.py`, `.yml` lên Git. Đừng bao giờ đẩy file `.snapshot` lên Git (vì nó quá nặng). Hãy lưu nó trên Google Drive hoặc ổ cứng ngoài.
*   **Định kỳ**: Mỗi tuần bạn nên tạo 1 snapshot để phòng trường hợp ổ cứng bị lỗi hoặc dữ liệu bị hỏng.
*   **Tên file**: Nên đổi tên file snapshot sao cho dễ nhớ (ví dụ: `legal_rag_v1_final.snapshot`).
