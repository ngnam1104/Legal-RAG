"""
Script nạp dữ liệu pháp lý từ HuggingFace vào Qdrant (Hybrid Search: Dense + Sparse).
Chạy: python scripts/ingest_local.py
"""
import os
import sys
import uuid
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

# Allow importing from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_db import client, ensure_collection
from retrieval.embedder import embedder
from backend.config import settings

COLLECTION_NAME = settings.QDRANT_COLLECTION

def setup_collection():
    print(f"⏳ Creating collection '{COLLECTION_NAME}'...")
    ensure_collection(client, COLLECTION_NAME, dense_dim=1024)
    
    print("⏳ Creating Payload Indexes...")
    indexes = ["is_appendix", "legal_type", "document_number", "issuance_date"]
    for field in indexes:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            print(f"  [+] Index created for: {field}")
        except Exception:
            print(f"  [=] Index already exists for: {field}")

def prepare_dataset(num_samples=200):
    print("⏳ Loading metadata and content datasets...")
    ds_metadata = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
    ds_content = load_dataset("th1nhng0/vietnamese-legal-documents", "content", split="data")
    
    df_meta = ds_metadata.to_pandas()
    df_qd = df_meta[(df_meta['legal_type'] == 'Quyết định') & (df_meta['legal_sectors'].notna())]
    
    print(f"📊 Filtering 'Quyết định' documents. Found: {len(df_qd)}")
    num_sectors = df_qd['legal_sectors'].nunique()
    samples_per_sector = (num_samples // num_sectors) + 2 

    df_sampled = df_qd.groupby('legal_sectors', group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_sector), random_state=42)
    )
    
    if len(df_sampled) > num_samples:
        df_sampled = df_sampled.sample(n=num_samples, random_state=42)
        
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_ids = set(df_sampled['id'])
    
    ds_content_filtered = ds_content.filter(lambda x: x['id'] in valid_ids)
    ds_metadata_filtered = ds_metadata.filter(lambda x: x['id'] in valid_ids)
    
    print(f"✅ Dataset prepared with {len(ds_content_filtered)} documents.")
    return ds_content_filtered, ds_metadata_filtered

def simple_chunk(text: str, metadata: dict, max_chunk_len: int = 1000):
    """Chunk văn bản đơn giản theo đoạn văn."""
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > max_chunk_len and current_chunk:
            chunks.append(current_chunk)
            current_chunk = p
        else:
            current_chunk = current_chunk + "\n" + p if current_chunk else p
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def ingest_data():
    setup_collection()
    
    ds_content, ds_metadata = prepare_dataset(num_samples=200)
    
    all_texts = []
    all_payloads = []
    
    print(f"⏳ Chunking documents...")
    for i in tqdm(range(len(ds_content)), desc="Chunking"):
        content = ds_content[i]['content']
        meta = ds_metadata[i]
        chunks = simple_chunk(content, meta)
        
        for idx, chunk_text in enumerate(chunks):
            all_texts.append(chunk_text)
            all_payloads.append({
                "document_id": str(meta.get('id', '')),
                "document_number": meta.get('document_number', 'Chưa rõ'),
                "title": meta.get('title', 'Không tiêu đề'),
                "legal_type": meta.get('legal_type', ''),
                "issuance_date": meta.get('issuance_date', ''),
                "signer": meta.get('signers', ''),
                "article_ref": f"Đoạn {idx + 1}",
                "is_appendix": False,
                "chunk_text": chunk_text
            })
        
    print(f"✅ Total chunks: {len(all_texts)}")
    
    print("🎨 Generating Dense Embeddings...")
    dense_vectors = embedder.encode_dense(all_texts, batch_size=32)
    
    print("🎨 Generating Sparse Embeddings...")
    sparse_vectors = embedder.encode_sparse_documents(all_texts, batch_size=32)
    
    print("🚀 Pushing points to Qdrant (Hybrid: Dense + Sparse)...")
    points = []
    for idx in range(len(all_texts)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vectors[idx],
            },
            payload=all_payloads[idx]
        ))
        
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Upserting"):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        
    print("🎉 Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
