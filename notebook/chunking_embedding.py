import sys
import os
import pandas as pd
import uuid
import random
from datasets import load_dataset
from qdrant_client.models import PointStruct
from tqdm import tqdm

# --- CẤU HÌNH MÔI TRƯỜNG & ĐƯỜNG DẪN TƯƠNG ĐỐI ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
sys.path.append(".")

# ===================================================================
# CHẾ ĐỘ CHẠY: True = Test 8000 samples (DB tạm), False = Full pipeline
TEST_MODE = True
# ===================================================================

os.environ["QDRANT_URL"] = "http://localhost:6335"
os.environ["QDRANT_API_KEY"] = ""
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "u7aGQYEWeFJD-jyeHB4ATtoAud73PptW35M1RzFlT-0"

if TEST_MODE:
    os.environ["QDRANT_COLLECTION"] = "legal_hybrid_rag_docs_8K"
    NEO4J_LABEL_PREFIX = "TEST_"
    SAMPLE_LIMIT = 2
    SKIP_LLM = True
    print(f"--- TEST MODE: {SAMPLE_LIMIT} samples | SKIP_LLM={SKIP_LLM} | Collection: legal_hybrid_rag_docs_8K ---")
else:
    os.environ["QDRANT_COLLECTION"] = "legal_hybrid_rag_docs"
    NEO4J_LABEL_PREFIX = ""
    SAMPLE_LIMIT = None
    SKIP_LLM = False
    print("--- FULL PIPELINE MODE ---")

# Cache HuggingFace
hf_cache_dir = os.path.abspath(os.path.join(repo_root, ".cache", "huggingface"))
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_dir
os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(hf_cache_dir, "sentence_transformers")

print(f"Environment setup complete (Project Root: {repo_root})")

# --- IMPORT MODULES NỘI BỘ ---
from backend.retrieval.chunker.metadata import normalize_doc_key
from backend.retrieval.chunker.core import AdvancedLegalChunker
from backend.retrieval.embedder import embedder
from backend.retrieval.vector_db import client as qdrant, ensure_collection
from backend.retrieval.graph_db import build_neo4j, get_neo4j_driver

# --- CSV: Danh sách 8000 VB ưu tiên ---
csv_path = os.path.join(repo_root, "top_8000_y_te_theo_quyen_luc.csv")
target_ids = set()
if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    target_ids = set(df_csv["id"].astype(str).tolist())
    print(f"-> Đã đọc {len(target_ids)} IDs văn bản từ CSV.")
else:
    print(f"[WARN] Không tìm thấy CSV tại {csv_path}. Sẽ dùng toàn bộ dataset.")

# --- TẢI DATASET ---
print("-> Đang tải dataset từ HuggingFace...")
ds_content_all = load_dataset("nhn309261/vietnamese-legal-docs", "content", split="data")
ds_meta_all    = load_dataset("nhn309261/vietnamese-legal-docs", "metadata", split="data")

all_meta_records = [row for row in ds_meta_all]
meta_id_to_idx   = {str(row["id"]): i for i, row in enumerate(all_meta_records)}
meta_docnum_to_id = {}
for row in all_meta_records:
    key = normalize_doc_key(str(row.get("document_number", "")))
    if key:
        meta_docnum_to_id[key] = str(row["id"])

content_id_to_idx = {str(row["id"]): i for i, row in enumerate(ds_content_all.select_columns(["id"]))}
print(f"Loaded and mapped {len(all_meta_records)} metadata records.")

# --- DYNAMIC LOOKUP (Lazy load nội dung theo doc_key) ---
class DynamicContentLookup(dict):
    def __init__(self, ds_content, content_id_to_idx, meta_docnum_to_id):
        self.ds_content = ds_content
        self.content_id_to_idx = content_id_to_idx
        self.meta_docnum_to_id = meta_docnum_to_id
    def get(self, key, default=None):
        if key in self: return super().get(key)
        t_id_str = self.meta_docnum_to_id.get(key)
        if t_id_str:
            idx_cont = self.content_id_to_idx.get(t_id_str)
            if idx_cont is not None:
                text = self.ds_content[idx_cont].get("content", "")
                self[key] = text
                return text
        return default

global_doc_lookup = DynamicContentLookup(ds_content_all, content_id_to_idx, meta_docnum_to_id)
meta_by_docnum_lookup = {}

# --- KHỞI TẠO ---
chunker = AdvancedLegalChunker()
BATCH_SIZE = 4
MAX_RECURSIVE_DOCS = 100000

from collections import Counter

all_chunks        = []
processed_ids     = set()
total_rels        = 0
global_rel_counts       = Counter()  # Đếm theo nhãn quan hệ
global_cross_rel_count  = 0          # Đếm riêng cross-doc (passive chain)
global_overlap_skipped  = 0          # Đếm các cặp bị bỏ qua do already_captured
stats_lines       = []  # Để tích luỹ ghi vào report file cuối cùng

# Tập hợp để xây dựng ghost nodes ở Pha 3:
# target_doc_keys: tất cả các văn bản được tham chiếu bởi quan hệ trong toàn pipeline
all_referenced_target_keys = set()

def print_top_sectors(doc_ids, label_phase, meta_id_to_idx, all_meta_records):
    from collections import Counter
    sector_counts = Counter()
    for did in doc_ids:
        idx = meta_id_to_idx.get(did)
        if idx is not None:
            meta = all_meta_records[idx]
            sectors = meta.get("legal_sectors", [])
            if not sectors: sectors = meta.get("sectors", [])
            if not sectors: sectors = meta.get("fields", [])

            if isinstance(sectors, str):
                sectors = [s.strip() for s in sectors.split(",")]
            elif not isinstance(sectors, list):
                sectors = []

            for s in sectors:
                s_str = str(s).strip()
                if s_str and s_str.lower() not in ["none", "null", "n/a", ""]:
                    sector_counts[s_str] += 1

    header = f"\n--- TOP 5 LĨNH VỰC ({label_phase}) ---"
    print(header)
    stats_lines.append(header)

    if not sector_counts:
        line = "   (Không có dữ liệu lĩnh vực)"
        print(line)
        stats_lines.append(line)
    for s, count in sector_counts.most_common(5):
        line = f"   - {s}: {count} văn bản"
        print(line)
        stats_lines.append(line)

    rel_header = f"\n--- THỐNG KÊ QUAN HỆ ({label_phase}) ---"
    print(rel_header)
    stats_lines.append(rel_header)

    if not global_rel_counts:
        line = "   (Chưa có quan hệ nào)"
        print(line)
        stats_lines.append(line)
    for lbl, cnt in global_rel_counts.most_common():
        line = f"   - {lbl}: {cnt} relations"
        print(line)
        stats_lines.append(line)

    # --- THỐNG KÊ CHỒNG CHÉO ---
    total_normal = sum(global_rel_counts.values()) - global_cross_rel_count
    cross_header = f"\n--- THỐNG KÊ CHỒNG CHÉO / PASSIVE CHAIN ({label_phase}) ---"
    print(cross_header)
    stats_lines.append(cross_header)
    lines = [
        f"   - Tổng quan hệ thường (chủ động):     {total_normal}",
        f"   - Tổng cross-doc AMENDS (bị động): {global_cross_rel_count}",
        f"   - Bỏ qua do already_captured:          {global_overlap_skipped}",
    ]
    for l in lines:
        print(l)
        stats_lines.append(l)

    sep = "=" * 60
    print(sep)
    stats_lines.append(sep)

def process_doc_batch(doc_ids, pbar, discovered_set=None, newly_processed_list=None):
    """Chunk một batch văn bản, cập nhật pbar và trả về các doc_id tham chiếu mới."""
    global total_rels, global_cross_rel_count, global_overlap_skipped
    new_refs = set()
    for doc_id in doc_ids:
        if doc_id in processed_ids:
            continue
        idx_content = content_id_to_idx.get(doc_id)
        idx_meta    = meta_id_to_idx.get(doc_id)
        if idx_content is None or idx_meta is None:
            continue

        content = ds_content_all[idx_content].get("content", "")
        meta    = all_meta_records[idx_meta]
        doc_num = str(meta.get("document_number", ""))
        key = normalize_doc_key(doc_num)
        if key:
            meta_by_docnum_lookup[key] = meta

        doc_chunks = chunker.process_document(
            content=content,
            metadata=meta,
            global_doc_lookup=global_doc_lookup,
            skip_llm=SKIP_LLM
        )
        all_chunks.extend(doc_chunks)
        processed_ids.add(doc_id)

        if doc_chunks:
            first_meta = doc_chunks[0].get("neo4j_metadata", doc_chunks[0].get("metadata", {}))
            rels = first_meta.get("ontology_relations", [])
            total_rels += len(rels)

            # Theo dõi từng loại quan hệ: thường vs cross-doc
            seen_pairs = set()  # phát hiện chồng chéo nội bộ trong chunk này
            for r in rels:
                lbl = str(r.get("edge_label", r.get("label", "UNKNOWN"))).strip()
                is_cross = r.get("is_cross_doc", False)
                if is_cross:
                    global_cross_rel_count += 1
                if lbl:
                    global_rel_counts[lbl] += 1

                # Kiểm tra chồng chéo: cùng (source, target, label) xuất hiện >1 lần
                pair_key = (r.get("source_doc", ""), r.get("target_doc", ""), lbl)
                if pair_key in seen_pairs:
                    global_overlap_skipped += 1
                else:
                    seen_pairs.add(pair_key)

                t_key = normalize_doc_key(r.get("target_doc", ""))
                if t_key:
                    all_referenced_target_keys.add(t_key)  # gom cho Pha 3
                    t_id = meta_docnum_to_id.get(t_key)
                    if t_id and t_id not in processed_ids and discovered_set is not None:
                        discovered_set.add(t_id)

        pbar.update(1)
        if discovered_set is not None:
            pbar.set_postfix({"Chunks": len(all_chunks), "Rels": total_rels, "CrossRels": global_cross_rel_count, "RefDocsAdded": len(discovered_set)})
        else:
            pbar.set_postfix({"Chunks": len(all_chunks), "Rels": total_rels, "CrossRels": global_cross_rel_count, "RefDocsAdded": len(all_referenced_target_keys)})

# ============================================================
# PHASE 1: Chunk + Embed ORIGINAL documents (8000)
# ============================================================
msg = "\n" + "=" * 60 + "\nPHASE 1: Chunking ORIGINAL documents\n" + "=" * 60
print(msg)
stats_lines.append(msg)

target_ids_list = list(target_ids)
if SAMPLE_LIMIT:
    target_ids_list = target_ids_list[:SAMPLE_LIMIT]

discovered_ref_ids = set()  # Văn bản tham chiếu tìm thấy ở Pha 1

pbar1 = tqdm(total=len(target_ids_list), desc="PHASE 1: Chunking ORIGINAL", unit="doc")
for i in range(0, len(target_ids_list), BATCH_SIZE):
    process_doc_batch(target_ids_list[i:i+BATCH_SIZE], pbar1, discovered_set=discovered_ref_ids)
pbar1.close()

print_top_sectors(list(processed_ids), "TỔNG LUỸ KẾ SAU PHA 1 (GỐC)", meta_id_to_idx, all_meta_records)

# Giới hạn số văn bản tham chiếu
if len(discovered_ref_ids) > MAX_RECURSIVE_DOCS:
    discovered_ref_ids = set(random.sample(list(discovered_ref_ids), MAX_RECURSIVE_DOCS))

chunks_after_phase1 = len(all_chunks)
print(f"PHASE 1 complete: {chunks_after_phase1} chunks | {len(discovered_ref_ids)} reference docs to process.")

# ============================================================
# PHASE 2: Chunk + Embed REFERENCE documents (discovered in Phase 1, exist in dataset)
# ============================================================
msg = "\n" + "=" * 60 + "\nPHASE 2: Chunking REFERENCE documents (exist in dataset)\n" + "=" * 60
print(msg)
stats_lines.append(msg)

# Loại bỏ các ID đã được xử lý (tránh trùng lặp nếu nó nằm ở cuối Phase 1)
discovered_ref_ids = discovered_ref_ids - processed_ids
disco_ids_list = list(discovered_ref_ids)
discovered_depth2_ids = set()

pbar2 = tqdm(total=len(disco_ids_list), desc="PHASE 2: Chunking REFERENCE", unit="doc")
for i in range(0, len(disco_ids_list), BATCH_SIZE):
    process_doc_batch(disco_ids_list[i:i+BATCH_SIZE], pbar2, discovered_set=discovered_depth2_ids)
pbar2.close()

print_top_sectors(list(processed_ids), "TỔNG LUỸ KẾ SAU PHA 2 (+VỆ TINH 1)", meta_id_to_idx, all_meta_records)

chunks_after_phase2 = len(all_chunks)
ref_count = chunks_after_phase2 - chunks_after_phase1
print(f"PHASE 2 complete: {len(disco_ids_list)} reference docs -> {ref_count} additional chunks.")
print(f"SUMMARY: {len(processed_ids)} documents total -> {len(all_chunks)} chunks.")

# ============================================================
# PHASE 3: Chunk + Embed REFERENCE documents (Depth=2)
# ============================================================
msg = "\n" + "=" * 60 + "\nPHASE 3: Chunking REFERENCE documents (Depth=2)\n" + "=" * 60
print(msg)
stats_lines.append(msg)

# Loại bỏ các ID đã được xử lý ở Phase 1 và Phase 2
discovered_depth2_ids = discovered_depth2_ids - processed_ids
disco_depth2_list = list(discovered_depth2_ids)
pbar3 = tqdm(total=len(disco_depth2_list), desc="PHASE 3: Chunking DEPTH=2", unit="doc")
for i in range(0, len(disco_depth2_list), BATCH_SIZE):
    process_doc_batch(disco_depth2_list[i:i+BATCH_SIZE], pbar3)
pbar3.close()

print_top_sectors(list(processed_ids), "TỔNG LUỸ KẾ SAU PHA 3 (+VỆ TINH 2)", meta_id_to_idx, all_meta_records)

chunks_after_phase3 = len(all_chunks)
depth2_count = chunks_after_phase3 - chunks_after_phase2
print(f"PHASE 3 complete: {len(disco_depth2_list)} depth=2 docs -> {depth2_count} additional chunks.")
print(f"SUMMARY: {len(processed_ids)} documents total -> {len(all_chunks)} chunks.")

# ============================================================
# PHASE 4: Ghost nodes — văn bản được tham chiếu nhưng KHÔNG có trong dataset
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Creating GHOST NODES for referenced docs not in dataset")
print("=" * 60)

ghost_nodes = []
for ref_key in all_referenced_target_keys:
    ref_id = meta_docnum_to_id.get(ref_key)
    if ref_id and ref_id in processed_ids:
        continue  # Đã được xử lý đầy đủ ở Pha 1 hoặc Pha 2 rồi
    # Node ma: chỉ có doc_number, không có chunk, không có embedding
    ghost_nodes.append({
        "document_number": ref_key,
        "is_ghost": True,
        "title": "N/A",
        "legal_type": "N/A",
        "issuing_authority": "N/A",
    })

print(f"PHASE 4 complete: {len(ghost_nodes)} ghost nodes to create in Neo4j.")

# --- FLUSH DATABASE ---
collection_name = os.environ.get("QDRANT_COLLECTION", "legal_hybrid_rag_docs")
print("\n-> Flushing old database...")
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name=collection_name)
    print(f"   Deleted Qdrant Collection '{collection_name}'.")
driver = get_neo4j_driver()
if driver:
    with driver.session() as session:
        if TEST_MODE:
            session.run("MATCH (n) WHERE n.test_mode = true DETACH DELETE n")
            print("   Deleted test nodes in Neo4j.")
        else:
            session.run("MATCH (n) DETACH DELETE n")
            print("   Deleted all nodes and relations in Neo4j.")
    driver.close()
ensure_collection(qdrant, collection_name=collection_name)

# ============================================================
# PHASE 5: Embedding + Upload Qdrant
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Embedding & Upsert Qdrant")
print("=" * 60)

batch_size = 64
pbar5 = tqdm(total=len(all_chunks), desc="PHASE 5: Embed & Upsert", unit="chunk")

import gc

for i in range(0, len(all_chunks), batch_size):
    chunk_batch = all_chunks[i:i+batch_size]
    texts_to_embed = [c.get("text_to_embed", c.get("chunk_text", "")) for c in chunk_batch]
    
    dense_batch = embedder.encode_dense(texts_to_embed)
    sparse_batch = embedder.encode_sparse_documents(texts_to_embed)
    
    points = []
    for idx, chunk in enumerate(chunk_batch):
        q_payload = chunk.get("qdrant_metadata", chunk.get("metadata", {}))
        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
            vector={"dense": dense_batch[idx], "sparse": sparse_batch[idx]},
            payload=q_payload
        ))
        
    qdrant.upsert(collection_name=collection_name, points=points)
    pbar5.update(len(chunk_batch))
    
    # Ép dọn dẹp bộ nhớ mỗi nhịp để giải phóng hoàn toàn RAM
    del texts_to_embed, dense_batch, sparse_batch, points, chunk_batch
    if i % 5000 == 0:
        gc.collect()

pbar5.close()
print(f"   Upserted {len(all_chunks)} points to Qdrant.")

# ============================================================
# PHASE 6: Build Neo4j Graph (nodes đầy đủ + ghost nodes)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: Building Neo4j Graph")
print("=" * 60)

driver = get_neo4j_driver()
if driver:
    # 5a. Tạo nodes/edges đầy đủ từ all_chunks
    build_neo4j(driver, all_chunks, meta_by_docnum_lookup=meta_by_docnum_lookup)

    # 5b. Tạo ghost nodes (MERGE để không trùng nếu đã tồn tại)
    if ghost_nodes:
        with driver.session() as session:
            for gn in ghost_nodes:
                session.run(
                    """
                    MERGE (d:Document {document_number: $doc_num})
                    ON CREATE SET
                        d.title             = $title,
                        d.legal_type        = $legal_type,
                        d.issuing_authority = $issuing_authority,
                        d.is_ghost          = true,
                        d.test_mode         = $test_mode
                    """,
                    doc_num=gn["document_number"],
                    title=gn["title"],
                    legal_type=gn["legal_type"],
                    issuing_authority=gn["issuing_authority"],
                    test_mode=TEST_MODE
                )
        print(f"   Created {len(ghost_nodes)} ghost nodes in Neo4j.")
    driver.close()
    print("   Neo4j Graph update complete.")
else:
    print("   Err: Could not connect to Neo4j.")

# --- FINAL STATISTICS ---
import time
import datetime

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — STATISTICS")
print("=" * 60)

edge_stats = {}
for chunk in all_chunks:
    meta = chunk.get("neo4j_metadata", chunk.get("metadata", {}))
    for r in meta.get("ontology_relations", []):
        label = r.get("edge_label", "UNKNOWN")
        edge_stats[label] = edge_stats.get(label, 0) + 1

stats_lines.extend([
    f"Run timestamp       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Original docs       : {len(target_ids_list)}",
    f"Reference docs      : {len(disco_ids_list)}",
    f"Ghost nodes         : {len(ghost_nodes)}",
    f"Total chunks        : {len(all_chunks)}",
    f"Total relations     : {total_rels}",
    f"Qdrant collection   : {collection_name}",
    "",
    "Edge label breakdown:",
])
for label, count in sorted(edge_stats.items(), key=lambda x: -x[1]):
    stats_lines.append(f"  {label:15s}: {count}")

for line in stats_lines:
    print(" ", line)

# (stats_lines giữ trong bộ nhớ, sẽ gộp vào report_8k.txt cuối cùng)


# ============================================================
# QUERY BENCHMARK: Kiểm tra truy vấn thực tế trên cả 2 DB
# ============================================================
print("\n" + "=" * 60)
print("QUERY BENCHMARK — QDRANT + NEO4J")
print("=" * 60)

benchmark_results = []

def bench(label, fn):
    """Chạy fn(), đo thời gian, in kết quả."""
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        status = "OK"
    except Exception as e:
        result = None
        elapsed = (time.perf_counter() - t0) * 1000
        status = f"ERR: {e}"
    hit_count = len(result) if isinstance(result, list) else (1 if result else 0)
    print(f"  [{status:3s}] {label:50s} {elapsed:7.1f}ms  hits={hit_count}")
    benchmark_results.append({"label": label, "status": status, "ms": elapsed, "hits": hit_count})

# Lấy 1 chunk mẫu để dùng vector thật cho query
sample_text = "Luật số 80/2015/QH13 ban hành văn bản quy phạm pháp luật"
sample_dense  = embedder.encode_dense([sample_text])[0]
sample_sparse = embedder.encode_sparse_documents([sample_text])[0]

print("\n── QDRANT ──")

# 1. Dense vector search
bench(
    "Dense search (top-5)",
    lambda: qdrant.query_points(
        collection_name=collection_name,
        query=sample_dense,
        using="dense",
        limit=5
    ).points
)

# 2. Sparse vector search
bench(
    "Sparse (BM25) search (top-5)",
    lambda: qdrant.query_points(
        collection_name=collection_name,
        query=sample_sparse,
        using="sparse",
        limit=5
    ).points
)

# 3. Hybrid search (dense + sparse, bằng query_batch)
from qdrant_client.models import Prefetch, FusionQuery, Fusion
bench(
    "Hybrid search RRF (top-5)",
    lambda: qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=sample_dense,  using="dense",  limit=20),
            Prefetch(query=sample_sparse, using="sparse", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5,
    ).points
)

# 4. Filter search theo legal_type
from qdrant_client.models import Filter, FieldCondition, MatchValue
bench(
    "Filter: legal_type='Luật' (top-5)",
    lambda: qdrant.query_points(
        collection_name=collection_name,
        query=sample_dense,
        using="dense",
        query_filter=Filter(must=[FieldCondition(key="legal_type", match=MatchValue(value="Luật"))]),
        limit=5
    ).points
)

# 5. Filter search theo năm
bench(
    "Filter: year='2015' (top-5)",
    lambda: qdrant.query_points(
        collection_name=collection_name,
        query=sample_dense,
        using="dense",
        query_filter=Filter(must=[FieldCondition(key="year", match=MatchValue(value="2015"))]),
        limit=5
    ).points
)

# 6. Scroll (không dùng vector, chỉ filter)
bench(
    "Scroll filter: is_table=True (limit=10)",
    lambda: qdrant.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="is_table", match=MatchValue(value=True))]),
        limit=10
    )[0]
)

print("\n── NEO4J ──")
driver_bench = get_neo4j_driver()
if driver_bench:
    with driver_bench.session() as s:

        # 7. Đếm tổng số node
        bench("Count total Document nodes",
              lambda: s.run("MATCH (d:Document) RETURN count(d) AS cnt").data())

        # 8. Đếm tổng số edge
        bench("Count total relationships",
              lambda: s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").data())

        # 9. Đếm ghost nodes
        bench("Count ghost nodes (is_ghost=true)",
              lambda: s.run("MATCH (d:Document {is_ghost:true}) RETURN count(d) AS cnt").data())

        # 10. Tìm VB theo số hiệu
        bench("Find doc by number '80/2015/QH13'",
              lambda: s.run(
                  "MATCH (d:Document) WHERE d.document_number CONTAINS '80/2015' RETURN d LIMIT 1"
              ).data())

        # 11. Lấy tất cả quan hệ của 1 VB
        bench("Get all relations of doc '80/2015/QH13'",
              lambda: s.run(
                  "MATCH (d:Document)-[r]->(t) WHERE d.document_number CONTAINS '80/2015' RETURN type(r), t.document_number LIMIT 20"
              ).data())

        # 12. Tìm các VB bị bãi bỏ (REPEALS)
        bench("Find REPEALS edges (limit 10)",
              lambda: s.run(
                  "MATCH (a)-[r:REPEALS]->(b) RETURN a.document_number, b.document_number LIMIT 10"
              ).data())

        # 13. Tìm các VB hướng dẫn (GUIDES)
        bench("Find GUIDES edges (limit 10)",
              lambda: s.run(
                  "MATCH (a)-[r:GUIDES]->(b) RETURN a.document_number, b.document_number LIMIT 10"
              ).data())

        # 14. Đường đi ngắn nhất giữa 2 VB (2 hop)
        bench("Shortest path (2 hops) between any 2 docs",
              lambda: s.run(
                  "MATCH p=shortestPath((a:Document)-[*..2]-(b:Document)) "
                  "WHERE a <> b RETURN length(p) AS hops LIMIT 1"
              ).data())

        # 15. Stat: Đếm edges theo loại
        bench("Count edges grouped by relation type",
              lambda: s.run(
                  "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC"
              ).data())

    driver_bench.close()
else:
    print("  [WARN] Không kết nối được Neo4j — bỏ qua benchmark Neo4j.")

# Lưu stats + benchmark gộp vào 1 file
report_file = "notebook/report_8k.txt"
try:
    with open(report_file, "w", encoding="utf-8") as f:
        # Phần 1: Stats
        f.write("=" * 60 + "\n")
        f.write("PIPELINE STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write("\n".join(stats_lines) + "\n\n")
        # Phần 2: Benchmark
        f.write("=" * 60 + "\n")
        f.write("QUERY BENCHMARK\n")
        f.write("=" * 60 + "\n")
        f.write(f"Benchmark run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Collection   : {collection_name}\n\n")
        f.write(f"{'Scenario':<52} {'Time(ms)':>9}  {'Hits':>5}  Status\n")
        f.write("-" * 80 + "\n")
        for r in benchmark_results:
            f.write(f"{r['label']:<52} {r['ms']:>9.1f}  {r['hits']:>5}  {r['status']}\n")
        ok  = sum(1 for r in benchmark_results if r["status"] == "OK")
        err = sum(1 for r in benchmark_results if r["status"] != "OK")
        avg = sum(r["ms"] for r in benchmark_results) / len(benchmark_results) if benchmark_results else 0
        f.write("-" * 80 + "\n")
        f.write(f"Total: {len(benchmark_results)} | OK: {ok} | ERR: {err} | Avg: {avg:.1f}ms\n")
    print(f"\n-> Đã lưu report ra: {report_file}")
except Exception as e:
    print(f"[WARN] Không thể lưu report file: {e}")

print("\n[DONE] Ingestion process completed successfully!")

