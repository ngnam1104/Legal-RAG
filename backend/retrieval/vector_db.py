from qdrant_client import QdrantClient
from qdrant_client import models
from backend.config import settings

def get_qdrant_client() -> QdrantClient:
    if settings.QDRANT_PATH:
        print(f"📡 Using Local Qdrant Storage: {settings.QDRANT_PATH}")
        return QdrantClient(path=settings.QDRANT_PATH)
        
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
        timeout=30
    )

def ensure_collection(client: QdrantClient, collection_name: str, dense_dim: int = 1024):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                )
            },
        )
        print(f"Created collection {collection_name}")

client = get_qdrant_client()
