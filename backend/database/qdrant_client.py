import os
from qdrant_client import QdrantClient
from qdrant_client import models

def get_qdrant_client() -> QdrantClient:
    qdrant_path = os.environ.get("QDRANT_PATH", "")
    if qdrant_path:
        print(f"📡 Using Local Qdrant Storage: {qdrant_path}")
        return QdrantClient(path=qdrant_path, check_compatibility=False)
        
    return QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        api_key=os.environ.get("QDRANT_API_KEY", "") or None,
        timeout=120,
        check_compatibility=False
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
