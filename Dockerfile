FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho thư viện xử lý pdf và ocr
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    antiword \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Cấu hình cache cho Hugging Face
ENV HF_HOME=/app/.cache/huggingface

# Copy tệp requirements và cài đặt
COPY requirements.txt /app/

# Khắc phục lỗi Timeout/EOF: Cài trước PyTorch (bản CPU) để giảm ~2GB tải xuống
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện còn lại với Timeout cao
RUN pip install --default-timeout=2000 --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . /app/

# Khai báo các port sẽ dùng cho ứng dụng
EXPOSE 8000 8501
