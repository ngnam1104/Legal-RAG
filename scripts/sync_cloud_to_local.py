import os
import time
import json
import re
import uuid
from typing import List, Optional, Any, Dict, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# ─── CONFIGURATION ───
load_dotenv()

# Cloud Qdrant (Source)
CLOUD_URL = os.getenv("CLOUD_QDRANT_URL", "https://bc0bb490-f62b-46b6-9863-5e413732dd93.us-west-1-0.aws.cloud.qdrant.io:6333")
CLOUD_API_KEY = os.getenv("CLOUD_QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NjZjNjMzNWEtOGI4NC00OTZhLTgxN2YtNzg5ZTlkZWZhN2FmIn0.F2ixBDFhiyw10MFYOaFyQOpfJh0oVt60XkhctvAsz9I")

# Local Qdrant (Destination)
LOCAL_URL = "http://localhost:6335" 
COLLECTION_NAME = "legal_rag_docs_5000"
STATE_FILE = "migrate_state.json"

# Optimization Settings
# Nhóm 1: KEYWORD (Lọc chính xác hoặc mảng)
INDEX_KEYWORDS = ["document_number", "document_id", "legal_type", "issuing_authority", "legal_sectors", "chapter_ref", "article_ref", "clause_ref"]

# Nhóm 2: BOOL (Lọc cờ Đúng/Sai)
INDEX_BOOLS = ["is_active", "is_appendix"]

# Nhóm 3: DATETIME (Lọc khoảng thời gian)
INDEX_DATETIMES = ["promulgation_date", "effective_date"]

# Text Indexes (Full-Text Search)
INDEX_TEXT = ["title", "breadcrumb_path", "article_title", "chunk_text"]

# ─── PHASE 1: CLOUD TO LOCAL MIGRATION ───
def migrate_from_cloud(cloud_client: QdrantClient, local_client: QdrantClient):
    print("\n" + "="*60)
    print("PHASE 1: BẮT ĐẦU DI CHUYỂN DỮ LIỆU CLOUD -> LOCAL DOCKER")
    print("="*60)
    
    if not local_client.collection_exists(COLLECTION_NAME):
        print(f"[*] Đang tạo Collection '{COLLECTION_NAME}' trên Local (Ép buộc Tắt Quantization)...")
        cloud_info = cloud_client.get_collection(COLLECTION_NAME)
        
        # CHÚ Ý: Ép buộc set quantization_config=None để dữ liệu local có Precision cao nhất (float32)
        local_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=cloud_info.config.params.vectors,
            sparse_vectors_config=cloud_info.config.params.sparse_vectors,
            quantization_config=None
        )

    # Load state for resume
    state = {"offset": None, "total_migrated": 0}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            
    next_page_offset = state["offset"]
    total_migrated = state["total_migrated"]
    batch_size = 1000

    if total_migrated > 0:
        print(f"[*] Tiếp tục từ mốc đã tải: {total_migrated} points...")

    while True:
        try:
            records, temp_next_page_offset = cloud_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_size,
                with_payload=True,
                with_vectors=True,
                offset=next_page_offset
            )
            if not records: break

            points = [models.PointStruct(id=r.id, payload=r.payload, vector=r.vector) for r in records]
            local_client.upsert(collection_name=COLLECTION_NAME, points=points)

            total_migrated += len(records)
            next_page_offset = temp_next_page_offset
            print(f"[Migration] Đã chuyển: {total_migrated} points...")
            
            with open(STATE_FILE, "w") as f:
                json.dump({"offset": next_page_offset, "total_migrated": total_migrated}, f)

            if next_page_offset is None: break
        except Exception as e:
            print(f"\n[!] Lỗi: {e}. Thử lại sau 5s...")
            time.sleep(5)
            continue

    print(f"[✓] HOÀN TẤT MIGRATION: {total_migrated} points.")
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)

# ─── PHASE 2: QDRANT OPTIMIZATION (INDEXING & PRECISION) ───
def optimize_qdrant_local(client: QdrantClient):
    print("\n" + "="*60)
    print("PHASE 2: BẮT ĐẦU TỐI ƯU HÓA INDEXING & PRECISION (FREEZE QUANTIZATION)")
    print("="*60)
    
    def safe_create_index(field_name, field_schema):
        try:
            client.create_payload_index(collection_name=COLLECTION_NAME, field_name=field_name, field_schema=field_schema, wait=False)
            print(f"[Index] Đã gửi lệnh tạo: {field_name}")
        except Exception as e:
            if "already exists" in str(e).lower(): print(f"[Index] Bỏ qua: {field_name} (Đã tồn tại)")
            else: print(f"[Index] Lỗi với {field_name}: {e}")

    # 1. Keyword Indexes (Important Metadata)
    for field in INDEX_KEYWORDS:
        safe_create_index(field, models.PayloadSchemaType.KEYWORD)

    # 2. Bool Indexes (True/False Search)
    for field in INDEX_BOOLS:
        safe_create_index(field, models.PayloadSchemaType.BOOL)

    # 3. Datetime Indexes (Time Range Filter)
    for field in INDEX_DATETIMES:
        safe_create_index(field, models.PayloadSchemaType.DATETIME)

    # 4. Text Indexes (Full-Text Search)
    text_params = models.TextIndexParams(type="text", tokenizer=models.TokenizerType.WORD, min_token_len=2, max_token_len=30, lowercase=True)
    for field in INDEX_TEXT:
        safe_create_index(field, text_params)

    # 3. Precision Management (No Quantization)
    print("[Quantization] BỎ QUA Quantization để duy trì độ chính xác (Precision) tuyệt đối (float32).")

    # Polling Status
    print("\nĐang chờ Qdrant hoàn tất xử lý Indexing...")
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > 600: break # Timeout 10 mins
        try:
            col_info = client.get_collection(COLLECTION_NAME)
            if col_info.status == models.CollectionStatus.GREEN and col_info.optimizer_status == models.OptimizersStatusOneOf.OK:
                print(f"\n[✓] THÀNH CÔNG! Indexing đã hoàn tất 100% sau {int(elapsed)}s.")
                break
            print(f"\rThời gian chờ: {int(elapsed)}s | Status: {col_info.status} | Opt: {col_info.optimizer_status}      ", end="", flush=True)
        except Exception as e: print(f"\n[!] Lỗi lấy status: {e}")
        time.sleep(5)

# ─── PHASE 3: ACCURACY & PERFORMANCE EVALUATION ───
class LocalBGEHybridEncoder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_name, use_fp16=(device == "cuda"), device=device)
    def encode_dense(self, texts: List[str]) -> List[List[float]]:
        out = self.model.encode(texts, batch_size=16, return_dense=True, return_sparse=False)
        return out["dense_vecs"].tolist()
    def encode_query_sparse(self, text: str) -> models.SparseVector:
        out = self.model.encode([text], batch_size=1, return_dense=False, return_sparse=True)
        weights = out["lexical_weights"][0]
        if not weights: return models.SparseVector(indices=[], values=[])
        pairs = sorted([(int(k), float(v)) for k, v in weights.items() if float(v) != 0.0])
        return models.SparseVector(indices=[p[0] for p in pairs], values=[p[1] for p in pairs])

def test_retrieval_accuracy(client: QdrantClient):
    print("\n" + "="*60)
    print("PHASE 3: KIỂM TRA ĐỘ CHÍNH XÁC VÀ HIỆU NĂNG TRUY XUẤT")
    print("="*60)
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Đang nạp mô hình Embedding trên thiết bị: {device}")
    encoder = LocalBGEHybridEncoder(device=device)
    
    test_queries = [
        # Test Query, Filter (Must)
        ("Tiêu chí và quyền nghĩa vụ của doanh nghiệp xã hội", [
            models.FieldCondition(key="legal_type", match=models.MatchValue(value="Luật"))
        ]),
        ("Quy định về người đại diện theo pháp luật của doanh nghiệp", [
            models.FieldCondition(key="document_number", match=models.MatchValue(value="59/2020/QH14"))
        ]),
        ("Điều kiện và hồ sơ đăng ký thành lập công ty cổ phần", [
             models.FieldCondition(key="promulgation_date", range=models.DatetimeRange(gte="2020-01-01T00:00:00Z"))
        ]),
        ("Các hành vi bị nghiêm cấm trong hoạt động doanh nghiệp", [
            models.FieldCondition(key="is_appendix", match=models.MatchValue(value=False))
        ])
    ]
    
    for query, filters in test_queries:
        print(f"\n🔎 Query: '{query}' | Filters: {len(filters)} conditions")
        t0 = time.perf_counter()
        
        # Search Step
        dense_vec = encoder.encode_dense([query])[0]
        sparse_vec = encoder.encode_query_sparse(query)
        
        filter_obj = models.Filter(must=filters) if filters else None
        
        # Hybrid Search with RRF Fusion
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=20, filter=filter_obj),
                models.Prefetch(query=sparse_vec, using="sparse", limit=20, filter=filter_obj),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=5,
            with_payload=True,
        ).points
        
        t1 = time.perf_counter()
        print(f"   [✓] Trả về {len(results)} kết quả trong {(t1-t0):.3f}s")
        for i, hit in enumerate(results, 1):
            p = hit.payload
            doc_type = p.get('legal_type', 'N/A')
            date = p.get('promulgation_date', 'N/A')
            is_app = p.get('is_appendix', False)
            print(f"       {i}. [{doc_type} | {date} | App:{is_app}] {p.get('title','N/A')[:60]}... Score: {hit.score:.4f}")

    print("\n[✓] HOÀN TẤT KIỂM TRA ACCURACY.")

# ─── MAIN EXECUTION ───
if __name__ == "__main__":
    cloud_c = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY, timeout=120)
    local_c = QdrantClient(url=LOCAL_URL, timeout=120)
    
    # 1. Migration
    # migrate_from_cloud(cloud_c, local_c)
    
    # 2. Optimization
    optimize_qdrant_local(local_c)
    
    # 3. Evaluation
    test_retrieval_accuracy(local_c)
    
    print("\n" + "="*60)
    print("TẤT CẢ QUY TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print("="*60)
