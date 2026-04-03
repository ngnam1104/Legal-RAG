"""
Kiểm tra số lượng point trong Qdrant chứa từng trường payload.
Phân nhóm theo: Định danh & Metadata | Cấu trúc & Tham chiếu | Nội dung & Căn cứ pháp lý.

Usage:
    python scripts/tests/check_payload_fields.py
"""

import os
import sys

# Allow importing from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, IsEmptyCondition, PayloadField

# ── Config ──────────────────────────────────────────────────────
# Try to import from backend settings first, fallback to env
try:
    from backend.config import settings
    QDRANT_URL = settings.QDRANT_URL
    QDRANT_API_KEY = settings.QDRANT_API_KEY
    COLLECTION = settings.QDRANT_COLLECTION
except ImportError:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6335")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_rag_docs_5000")

# ── Danh sách trường cần kiểm tra, chia theo nhóm ──────────────
FIELD_GROUPS = {
    "1️⃣  Nhóm Định danh & Metadata": [
        "document_id",
        "document_uid",
        "chunk_id",
        "document_number",
        "issuance_date",
        "title",
        "legal_type",
        "legal_sectors",
        "issuing_authority",
        "signer",
        "url",
        "is_active",
    ],
    "2️⃣  Nhóm Cấu trúc & Tham chiếu": [
        "article_id",
        "article_ref",
        "clause_ref",
        "is_appendix",
        "breadcrumb_path",
        "reference_citation",
        "parent_law_ids",
    ],
    "3️⃣  Nhóm Nội dung & Căn cứ pháp lý": [
        "parent_article_text",
        "matched_clause_text",
        "chunk_text",
        "legal_basis_texts",
        "legal_basis_refs",
    ],
}


def count_points_with_field(client: QdrantClient, collection: str, field_name: str) -> int:
    """
    Đếm số point CÓ chứa trường `field_name` trong payload.
    Logic: Tổng points - points mà trường đó rỗng/không tồn tại.
    """
    # Đếm số point KHÔNG có trường này (is_empty = True)
    empty_count = client.count(
        collection_name=collection,
        count_filter=Filter(
            must=[
                IsEmptyCondition(
                    is_empty=PayloadField(key=field_name)
                )
            ]
        ),
        exact=True,
    ).count

    return empty_count


def main():
    print(f"🔗 Kết nối Qdrant tại: {QDRANT_URL}")
    print(f"📦 Collection: {COLLECTION}")
    print("=" * 70)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    # Lấy tổng số points
    try:
        collection_info = client.get_collection(collection_name=COLLECTION)
        total_points = collection_info.points_count
    except Exception as e:
        print(f"❌ Không tìm thấy collection '{COLLECTION}': {e}")
        return

    print(f"📊 Tổng số points trong collection: {total_points:,}")
    print("=" * 70)

    all_fields_summary = []

    for group_name, fields in FIELD_GROUPS.items():
        print(f"\n{'─' * 70}")
        print(f"  {group_name}")
        print(f"{'─' * 70}")
        print(f"  {'Trường':<30} {'Có dữ liệu':>12} {'Trống/Thiếu':>12} {'Tỷ lệ (%)':>10}")
        print(f"  {'─' * 64}")

        for field in fields:
            try:
                empty_count = count_points_with_field(client, COLLECTION, field)
                has_data = total_points - empty_count
                pct = (has_data / total_points * 100) if total_points > 0 else 0

                # Icon theo tỷ lệ
                if pct == 100:
                    icon = "✅"
                elif pct >= 80:
                    icon = "🟡"
                elif pct > 0:
                    icon = "🟠"
                else:
                    icon = "❌"

                print(f"  {icon} {field:<28} {has_data:>10,} {empty_count:>12,} {pct:>9.1f}%")
                all_fields_summary.append((field, has_data, empty_count, pct))

            except Exception as e:
                print(f"  ⚠️  {field:<28} {'LỖI':>10} — {e}")
                all_fields_summary.append((field, -1, -1, -1))

    # ── Tổng kết ────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  📋 TỔNG KẾT")
    print(f"{'═' * 70}")

    full_fields = [f for f, h, e, p in all_fields_summary if p == 100]
    partial_fields = [f for f, h, e, p in all_fields_summary if 0 < p < 100]
    missing_fields = [f for f, h, e, p in all_fields_summary if p == 0]
    error_fields = [f for f, h, e, p in all_fields_summary if p == -1]

    print(f"  ✅ Đầy đủ (100%):  {len(full_fields)} trường")
    if full_fields:
        print(f"     → {', '.join(full_fields)}")

    print(f"  🟡 Có một phần:    {len(partial_fields)} trường")
    if partial_fields:
        for f in partial_fields:
            data = next((h, e, p) for fn, h, e, p in all_fields_summary if fn == f)
            print(f"     → {f}: {data[0]:,} points ({data[2]:.1f}%)")

    print(f"  ❌ Không có:       {len(missing_fields)} trường")
    if missing_fields:
        print(f"     → {', '.join(missing_fields)}")

    if error_fields:
        print(f"  ⚠️  Lỗi:           {len(error_fields)} trường")
        print(f"     → {', '.join(error_fields)}")

    print()


if __name__ == "__main__":
    main()
