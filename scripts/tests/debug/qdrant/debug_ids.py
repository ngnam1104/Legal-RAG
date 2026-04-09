import os
import sys

# Add project root to path (scripts/tests/debug/file.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from backend.retrieval.vector_db import client as qdrant_client
from backend.config import settings

def debug_ids():
    collection = settings.QDRANT_COLLECTION
    print(f"Checking collection: {collection}")
    
    hits, _ = qdrant_client.scroll(
        collection_name=collection,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    
    for hit in hits:
        p = hit.payload
        print(f"ID: {p.get('document_id')} | Number: {p.get('document_number')}")

if __name__ == "__main__":
    debug_ids()
