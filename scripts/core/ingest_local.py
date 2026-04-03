"""
Script nạp dữ liệu pháp lý từ HuggingFace vào Qdrant (Hybrid Search: Dense + Sparse).
Phiên bản 2.0: Tinh gọn Payload và tối ưu hóa cho AdvancedLegalChunker.
Chạy: python scripts/core/ingest_local.py
"""
import os
import sys
import uuid
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (like HF_HOME) before other imports
load_dotenv()

from datasets import load_dataset
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType, Filter, TextIndexParams, TokenizerType

# Allow importing from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config import settings
from backend.retrieval.vector_db import client, ensure_collection
from backend.retrieval.embedder import embedder
from backend.retrieval.chunker import chunker

COLLECTION_NAME = settings.QDRANT_COLLECTION

def setup_collection():
    print(f"⏳ Đang chuẩn bị Collection '{COLLECTION_NAME}'...")
    ensure_collection(client, COLLECTION_NAME, dense_dim=1024)
    
    print("⏳ Đang thiết lập các Payload Indexes (Tối ưu)...")
    # Các trường quan trọng hỗ trợ lọc dữ liệu nhanh
    indexes = [
        "is_appendix", 
        "legal_type", 
        "document_number", 
        "issuance_date", 
        "document_id", 
        "document_uid", # Định danh chuẩn
        "article_id",   # Cho Small-to-Big
        "chapter_ref", 
        "article_ref", 
        "clause_ref", 
        "parent_law_ids", # Citation Graph
        "is_active"      # Hiệu lực văn bản
    ]
    
    for field in indexes:
        try:
            field_schema = PayloadSchemaType.BOOL if field in ["is_appendix", "is_active"] else PayloadSchemaType.KEYWORD
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=field_schema,
            )
            print(f"  [+] Index created for: {field}")
        except Exception as e:
            pass # Index existed
            
    # Index mảng cho ngành (Sector)
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="legal_sectors",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print("  [+] Index created for: legal_sectors")
    except Exception:
        pass

    # Index Full-text cho Title (Tìm kiếm từ khóa bổ trợ)
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="title",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )
        print("  [+] Text index created for: title")
    except Exception:
        pass


def prepare_dataset(num_samples=200):
    print("⏳ Đang tải dữ liệu từ HuggingFace (Metadata + Content)...")
    ds_metadata = load_dataset("nhn309261/vietnamese-legal-documents", "metadata", split="data")
    ds_content = load_dataset("nhn309261/vietnamese-legal-documents", "content", split="data")
    
    df_meta = ds_metadata.to_pandas()
    # Chỉ lấy các loại văn bản chính để demo
    df_qd = df_meta[(df_meta['legal_type'] == 'Luật') | (df_meta['legal_type'] == 'Nghị định') | (df_meta['legal_type'] == 'Thông tư')]
    
    num_sectors = df_qd['legal_sectors'].nunique()
    samples_per_sector = (num_samples // num_sectors) + 1 if num_sectors > 0 else num_samples

    print(f"📊 Phân bổ dữ liệu: ~{samples_per_sector} docs per sector.")
    df_sampled = df_qd.groupby('legal_sectors', group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_sector), random_state=42)
    )
    
    if len(df_sampled) > num_samples:
        df_sampled = df_sampled.sample(n=num_samples, random_state=42)
    
    valid_ids = set(df_sampled['id'])
    ds_content_filtered = ds_content.filter(lambda x: x['id'] in valid_ids)
    ds_metadata_filtered = ds_metadata.filter(lambda x: x['id'] in valid_ids)
    
    print(f"✅ Đã chuẩn bị {len(ds_content_filtered)} văn bản gốc.")
    return ds_content_filtered, ds_metadata_filtered

def ingest_data():
    setup_collection()
    
    # Nạp thử nghiệm 50 văn bản
    ds_content, ds_metadata = prepare_dataset(num_samples=50)
    
    all_chunks = []
    print(f"⏳ Đang thực hiện Chunking với AdvancedLegalChunker...")
    for i in tqdm(range(len(ds_content)), desc="Chunking"):
        content = ds_content[i]['content']
        meta = dict(ds_metadata[i])
        
        # AdvancedLegalChunker sẽ tự normalize metadata
        chunks = chunker.process_document(content, meta)
        all_chunks.extend(chunks)
        
    print(f"✅ Tổng cộng sinh ra: {len(all_chunks)} chunks.")
    
    all_texts = [c["chunk_text"] for c in all_chunks]
    
    # Tối ưu: Single forward pass cho cả Dense và Sparse
    print("🎨 Đang sinh Embedding (Hybrid: Dense + Sparse)...")
    dense_vectors, sparse_vectors = embedder.encode_hybrid(all_texts, batch_size=16)
    
    print("🚀 Đang đẩy dữ liệu lên Qdrant (Payload Tinh gọn)...")
    points = []
    for idx, chunk in enumerate(all_chunks):
        meta = chunk["metadata"]
        
        # TINH GỌN PAYLOAD: Xóa signer (giữ lại theo yeu cau), xóa base_laws, article_title, reference_tag
        payload = {
            "document_id": meta.get("document_id"),
            "document_uid": meta.get("document_uid"),
            "chunk_id": chunk.get("chunk_id"),
            "document_number": meta.get("document_number", "N/A"),
            "title": meta.get("title", "N/A"),
            "legal_type": meta.get("legal_type", "N/A"),
            "legal_sectors": meta.get("legal_sectors", []),
            "signer": meta.get("signer", "N/A"),
            "issuance_date": meta.get("issuance_date"),
            "is_active": meta.get("is_active", True),
            "is_appendix": meta.get("is_appendix", False),
            "chapter_ref": meta.get("chapter_ref"),
            "article_id": meta.get("article_id"),
            "article_ref": meta.get("article_ref"),
            "clause_ref": meta.get("clause_ref"),
            "point_ref": meta.get("point_ref"),
            "breadcrumb_path": meta.get("breadcrumb_path"),
            "reference_citation": meta.get("reference_citation"),
            "legal_basis_refs": meta.get("legal_basis_refs", []),
            "parent_law_ids": meta.get("parent_law_ids", []),
            "url": meta.get("url", ""),
            "chunk_text": chunk.get("chunk_text", "")
        }

        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, payload["chunk_id"])),
            vector={
                "dense": dense_vectors[idx],
                "sparse": sparse_vectors[idx],
            },
            payload=payload
        ))
        
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Upserting"):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        
    print("🎉 Hoàn tất! Dữ liệu đã được nạp thành công.")

if __name__ == "__main__":
    ingest_data()
