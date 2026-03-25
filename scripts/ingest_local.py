import os
import sys
import uuid
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

# Allow importing from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.qdrant_manager import client
from embedding.embedder import get_embedder
from data.chunker import AdvancedLegalChunker

COLLECTION_NAME = "legal_vn_200_docs"

def setup_collection():
    print(f"⏳ Creating collection '{COLLECTION_NAME}'...")
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE, on_disk=True),
        )
        print("✅ Database created.")
    else:
        print("✅ Collection already exists.")

    print("⏳ Creating Payload Indexes...")
    indexes = ["is_appendix", "legal_type", "document_number", "issuance_date"]
    for field in indexes:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"  [+] Index created for: {field}")

def prepare_dataset(num_samples=200):
    print("⏳ Loading metadata and content datasets...")
    ds_metadata = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")
    ds_content = load_dataset("th1nhng0/vietnamese-legal-documents", "content", split="data")
    
    df_meta = ds_metadata.to_pandas()
    df_qd = df_meta[(df_meta['legal_type'] == 'Quyết định') & (df_meta['legal_sectors'].notna())]
    
    print(f"📊 Filtering 'Quyết định' documents. Found: {len(df_qd)}")
    num_sectors = df_qd['legal_sectors'].nunique()
    samples_per_sector = (num_samples // num_sectors) + 2 

    # Stratified Sampling
    df_sampled = df_qd.groupby('legal_sectors', group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_sector), random_state=42)
    )
    
    if len(df_sampled) > num_samples:
        df_sampled = df_sampled.sample(n=num_samples, random_state=42)
        
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_ids = set(df_sampled['id'])
    
    # Filter datasets based on exact IDs
    ds_content_filtered = ds_content.filter(lambda x: x['id'] in valid_ids)
    ds_metadata_filtered = ds_metadata.filter(lambda x: x['id'] in valid_ids)
    
    print(f"✅ Dataset prepared with {len(ds_content_filtered)} documents.")
    return ds_content_filtered, ds_metadata_filtered

def ingest_data():
    setup_collection()
    
    embedder = get_embedder()
    chunker = AdvancedLegalChunker()
    
    ds_content, ds_metadata = prepare_dataset(num_samples=200) # Mặc định làm 200 bản ghi để test nhanh
    
    all_chunks = []
    print(f"⏳ Chunking documents...")
    for i in tqdm(range(len(ds_content)), desc="Chunking"):
        content = ds_content[i]['content']
        metadata = ds_metadata[i]
        chunks = chunker.process_document(content, metadata)
        all_chunks.extend(chunks)
        
    print(f"✅ Total chunks: {len(all_chunks)}")
    
    print("🎨 Generating Embeddings (Batching)...")
    texts = [c['text_to_embed'] for c in all_chunks]
    vectors = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    
    print("🚀 Pushing points to Qdrant...")
    points = []
    for idx, chunk in enumerate(all_chunks):
        meta = chunk['metadata']
        payload = {
            "document_id": meta.get('id', ''),
            "document_number": meta.get('document_number', 'Chưa rõ'),
            "title": meta.get('title', 'Không tiêu đề'),
            "legal_type": meta.get('legal_type', ''),
            "issuance_date": meta.get('issuance_date', ''),
            "signer": meta.get('signers', ''),
            "article_ref": chunk['reference_tag'],
            "is_appendix": meta.get('is_appendix', False),
            "chunk_text": chunk['text_to_embed']
        }
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[idx],
            payload=payload
        ))
        
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Upserting"):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        
    print("🎉 Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
