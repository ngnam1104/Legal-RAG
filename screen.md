## 1. Tạo session và chạy job (chuẩn nhất)

```bash
screen -S nam.nguyen3.index_kg
```

→ Trong session đó:

```bash
python your_script.py
```

👉 Thoát ra nhưng vẫn giữ chạy:

```bash
Ctrl + A + D
```

---

## 2. Attach lại session đang chạy

Bạn đã có:

```bash
screen -r 2466998.nam.nguyen3.index_kg
```

Hoặc:

```bash
screen -r
```

### Các case xảy ra:

### ✔ Case A: Session đang chạy

→ attach vào, thấy log chạy tiếp

---

### ⚠ Case B: Báo:

```
There is a screen on:
    2466998.nam.nguyen3.index_kg (Attached)
```

→ nghĩa là session đang mở ở nơi khác (hoặc bị treo)

Fix:

```bash
screen -d -r 2466998.nam.nguyen3.index_kg
```

---

### ❌ Case C: Không attach được

→ session **đã chết**

Check:

```bash
screen -ls
```

Nếu không thấy → job đã dừng → **không có gì để resume**

---

## 3. Hiểu sai phổ biến (của bạn đang gặp)

Bạn nói:

> “tắt nó, chạy lại cùng screen”

👉 Có 2 kiểu “tắt”:

### ❌ Nếu bạn:

* đóng SSH
* nhưng **không detach (Ctrl+A+D)**

→ session vẫn sống (OK)

---

### ❌ Nếu bạn:

* process Python crash / stop
* hoặc bạn `exit`

→ screen vẫn tồn tại nhưng **job bên trong đã chết**

👉 attach lại sẽ thấy **terminal trống**

---

## 4. Muốn chạy 30h an toàn (khuyến nghị thực tế)

### Cách chuẩn:

```bash
screen -S kg_job
```

Trong đó:

```bash
cd ~/nam.nguyen3/Legal-RAG/notebook
python run.py | tee log.txt
```

Detach:

```bash
Ctrl + A + D
```

---

## 5. Khi reconnect server

```bash
screen -ls
screen -r kg_job
```

---

## 6. Nếu muốn auto resume (quan trọng với bạn)

`screen` KHÔNG hỗ trợ resume job.

Bạn phải tự làm:

### ✔ Checkpoint đúng cách:

Trong code:

* load checkpoint nếu tồn tại
* không restart từ đầu

Ví dụ logic:

```python
if checkpoint_exists:
    load_state()
else:
    start_from_scratch()
```

---

## 7. Debug case của bạn (rất khả năng)

Bạn nói:

* có checkpoint
* nhưng chạy lại không dùng

👉 90% do:

* chạy sai working directory
* hoặc code không load checkpoint path đúng

Check nhanh:

```bash
pwd
ls checkpoints/
```

---

## 8. Tip nâng cao (nên dùng)

### Xem log mà không attach:

```bash
tail -f log.txt
```

---

### Kill session:

```bash
screen -X -S kg_job quit
```

---

## Kết luận ngắn gọn

* `screen` ≠ job manager
* nó chỉ giữ terminal
* job chết → screen không cứu được
* muốn chạy 30h:

  * dùng `screen`
  * * logging
  * * checkpoint load đúng

---

Nếu bạn muốn, gửi mình:

* command chạy script
* cấu trúc checkpoint

Mình sẽ chỉ ra **vì sao nó không resume** (khả năng cao nằm ở path hoặc logic load).
