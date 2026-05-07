#!/bin/bash

# Script tự động hóa quá trình tạo bộ test và đánh giá hội thoại dài hạn
echo "======================================================================"
echo "🚀 LEGAL-RAG: BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ LONG-TERM"
echo "======================================================================"

# Bước 1: Tạo bộ dữ liệu test mới (Long-term mode)
echo "Bước 1: Đang tạo bộ dữ liệu test mới (mode: long_term)..."
python utils/generate_test_dataset.py --mode long_term

if [ $? -eq 0 ]; then
    echo "✅ Đã tạo xong bộ dữ liệu test."
    echo ""
else
    echo "❌ LỖI: Quá trình tạo dữ liệu thất bại."
    exit 1
fi

# Bước 2: Chạy kiểm thử đánh giá ngữ cảnh dài hạn
echo "Bước 2: Đang chạy kiểm thử đánh giá (Multi-turn Evaluation)..."
echo "Lưu ý: Quá trình này có thể mất vài phút tùy vào tốc độ của LLM."
python tests/long_term_evaluation/test_long_term.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "🏁 QUY TRÌNH HOÀN TẤT THÀNH CÔNG."
    echo "Kết quả chi tiết tại: tests/long_term_evaluation/long_term_results.txt"
    echo "======================================================================"
else
    echo "❌ LỖI: Quá trình kiểm thử bị gián đoạn hoặc thất bại."
    exit 1
fi
