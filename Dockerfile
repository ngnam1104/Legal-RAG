FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho thư viện xử lý pdf và ocr
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    antiword \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy tệp requirements và cài đặt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . /app/

# Khai báo các port sẽ dùng cho ứng dụng
EXPOSE 8000 8501
