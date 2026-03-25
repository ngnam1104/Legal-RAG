import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from core.config import settings


COLLECTION_NAME = settings.QDRANT_COLLECTION


def get_qdrant_client() -> QdrantClient:
    """Trả về client Qdrant (Cloud / HTTP / local path)."""
    if settings.QDRANT_URL.startswith("http"):
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=os.environ.get("QDRANT_API_KEY", None),
        )
    return QdrantClient(path=settings.QDRANT_URL)


# Singleton client dùng chung
client = get_qdrant_client()


def ensure_qdrant_collection(collection_name: str = COLLECTION_NAME, vector_size: int = 1024):
    """
    Tạo collection và index payload nếu chưa có.
    Gọi an toàn nhiều lần.
    """
    try:
        if client.collection_exists(collection_name):
            return

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        # Payload indexes phục vụ lọc nhanh (an toàn idempotent: Qdrant bỏ qua nếu đã tồn tại)
        for field_name, field_schema in [
            ("is_appendix", "bool"),
            ("document_number", "keyword"),
            ("legal_type", "keyword"),
            ("issuance_date", "keyword"),
            ("title", "text"),
        ]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                # Nếu index đã tồn tại hoặc schema không hỗ trợ, bỏ qua để không chặn khởi động
                pass
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant to ensure collection. Error: {e}")
