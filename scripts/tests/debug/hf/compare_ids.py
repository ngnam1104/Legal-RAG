import os
import sys

# Add project root to path (scripts/tests/debug/file.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from backend.retrieval.vector_db import client as qdrant_client
from backend.config import settings
from datasets import load_dataset

def compare_ids():
    collection = settings.QDRANT_COLLECTION
    print(f"📡 Fetching IDs from Qdrant ({collection})...")
    hits, _ = qdrant_client.scroll(
        collection_name=collection,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    db_ids = {hit.payload.get("document_id") for hit in hits}
    print(f"DB Samples: {db_ids}")

    print(f"📥 Loading HF dataset stream...")
    ds = load_dataset('nhn309261/vietnamese-legal-documents', 'content', split='data', streaming=True)
    it = iter(ds)
    
    print("HF Samples:")
    for _ in range(5):
        try:
            record = next(it)
            print(f"HF ID: {record['id']}")
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    compare_ids()
