import sys
import os
import json

# Thêm thư mục gốc vào path để import backend
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import models
from backend.retrieval.vector_db import client as qdrant
from backend.config import settings

def inspect_payload():
    print("\n🔍 --- QDRANT PAYLOAD INSPECTOR ---")
    doc_number = input("1. Nhập số hiệu văn bản (document_number): ").strip()
    article_input = input("2. Nhập số hiệu Điều (ví dụ: 186) [Bỏ qua nếu muốn xem hết]: ").strip()
    
    if not doc_number:
        print("❌ Vui lòng nhập số hiệu văn bản hợp lệ.")
        return

    print(f"📡 Đang tìm kiếm trong collection: {settings.QDRANT_COLLECTION}...")
    
    # Xây dựng filter
    filter_conditions = [
        models.FieldCondition(
            key="document_number",
            match=models.MatchValue(value=doc_number),
        )
    ]
    
    if article_input:
        # Hỗ trợ tìm theo "Điều 186", "Điều 186a", v.v.
        # Chúng ta dùng MatchText sẽ linh hoạt hơn MatchValue
        article_query = f"Điều {article_input}"
        filter_conditions.append(
            models.FieldCondition(
                key="article_ref",
                match=models.MatchValue(value=article_query),
            )
        )

    try:
        # Sử dụng scroll với filter để lấy tất cả các điểm khớp
        points, next_page = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=models.Filter(must=filter_conditions),
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        if not points and article_input:
            # Thử lại một lần nữa với filter lỏng hơn (chỉ match text nếu MatchValue Điều X không ra)
            print(f"🔄 Không tìm thấy chính xác '{article_query}', thử tìm kiếm mở rộng...")
            filter_conditions[-1] = models.FieldCondition(
                key="article_ref",
                match=models.MatchText(text=article_input),
            )
            points, next_page = qdrant.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=models.Filter(must=filter_conditions),
                limit=100,
                with_payload=True,
                with_vectors=False
            )

        if not points:
            msg = f"số hiệu: {doc_number}"
            if article_input: msg += f" và Điều: {article_input}"
            print(f"⚠️ Không tìm thấy điểm nào khớp với {msg}")
            return
            
        print(f"✅ Tìm thấy {len(points)} điểm dữ liệu.\n")
        
        for i, point in enumerate(points, 1):
            print(f"--- [Điểm {i}] (ID: {point.id}) ---")
            print(json.dumps(point.payload, indent=4, ensure_ascii=False))
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Lỗi khi truy vấn Qdrant: {e}")

if __name__ == "__main__":
    inspect_payload()
