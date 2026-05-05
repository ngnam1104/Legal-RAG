import sys
import os
import datetime

# --- LOGGER: Ghi log ra màn hình VÀ file tức thời ---
class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        # Bỏ qua các ký tự đè dòng (\r) của thanh tiến trình (tqdm)
        if "\r" not in message:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", ".debug"), exist_ok=True)
debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".debug")
log_filename = f"{debug_dir}/ingestion_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = DualLogger(log_filename)
print(f"📝 Logging tức thời vào file: {log_filename}")

import pandas as pd
import uuid
import random
from datasets import load_dataset
from qdrant_client.models import PointStruct
from tqdm import tqdm

# --- CẤU HÌNH MÔI TRƯỜNG & ĐƯỜNG DẪN TƯƠNG ĐỐI ---
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
sys.path.append(".")

# Load .env (ưu tiên giá trị từ file .env thay vì hardcode)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(repo_root, ".env"), override=False)
except ImportError:
    pass  # dotenv không bắt buộc nếu env đã được set sẵn

# ===================================================================
# CHẾ ĐỘ CHẠY: ĐẶT TEST_MODE = True để test 500 VB y tế với DB riêng
# Tất cả tham số được đọc từ .env để dễ thay đổi không cần sửa code
TEST_MODE         = True   # True = test 500 VB y tế | False = Full pipeline
USE_CHECKPOINT    = True  # Bật để resume nếu sập giữa chừng
FLUSH_DB_ON_START = True  # True = xóa sạch DB trước khi chạy (chỉ dùng khi reset)
# ===================================================================

# Đọc credentials từ env (ưu tiên giá trị server đã set, không ghi đè)
os.environ.setdefault("QDRANT_URL",     "http://localhost:6337")
os.environ.setdefault("QDRANT_API_KEY", "")
os.environ.setdefault("NEO4J_URI",      "bolt://localhost:7688")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "u7aGQYEWeFJD-jyeHB4ATtoAud73PptW35M1RzFlT-0")

if TEST_MODE:
    # Đọc từ .env TEST_ vars, fall back sang giá trị mặc định nếu chưa set
    _test_collection  = os.environ.get("TEST_QDRANT_COLLECTION", "legal_rag_test_yte_500")
    _test_neo4j_uri   = os.environ.get("TEST_NEO4J_URI",      "bolt://localhost:7689")
    _test_neo4j_user  = os.environ.get("TEST_NEO4J_USERNAME", "neo4j")
    _test_neo4j_pass  = os.environ.get("TEST_NEO4J_PASSWORD", "test_neo4j_pass")
    _test_limit       = int(os.environ.get("TEST_SAMPLE_LIMIT", "500"))

    # Override toàn bộ pipeline trỏ vào DB test — KHÔNG đụng production
    os.environ["QDRANT_COLLECTION"] = _test_collection
    os.environ["NEO4J_URI"]         = _test_neo4j_uri
    os.environ["NEO4J_USERNAME"]    = _test_neo4j_user
    os.environ["NEO4J_PASSWORD"]    = _test_neo4j_pass

    NEO4J_LABEL_PREFIX = ""   # Không cần prefix, đã tách DB hoàn toàn
    SAMPLE_LIMIT = _test_limit
    SKIP_LLM = False
    print(f"--- TEST MODE: {SAMPLE_LIMIT} VB y te ---")
    print(f"    Qdrant : {_test_collection}")
    print(f"    Neo4j  : {_test_neo4j_uri}  (container rieng, khong dung production)")
else:
    os.environ["QDRANT_COLLECTION"] = os.environ.get("QDRANT_COLLECTION", "legal_rag_docs_nam")
    NEO4J_LABEL_PREFIX = ""
    SAMPLE_LIMIT = None
    SKIP_LLM = False
    print(f"--- FULL PIPELINE MODE | Qdrant: {os.environ['QDRANT_COLLECTION']} ---")

# Cache HuggingFace
hf_cache_dir = os.path.abspath(os.path.join(repo_root, ".cache", "huggingface"))
os.makedirs(hf_cache_dir, exist_ok=True)
os.environ["HF_HOME"] = hf_cache_dir
os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(hf_cache_dir, "sentence_transformers")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"Environment setup complete (Project Root: {repo_root})")

# --- IMPORT MODULES NỘI BỘ (theo kiến trúc mới) ---
from backend.ingestion.chunker.metadata import normalize_doc_key
from backend.ingestion.chunker.core import AdvancedLegalChunker
from backend.ingestion.extractor.relations import extract_ontology_relationships_batch
from backend.models.embedder import embedder
from backend.database.qdrant_client import client as qdrant, ensure_collection
from backend.database.neo4j_client import (
    build_neo4j, get_neo4j_driver, init_neo4j_constraints, enrich_chunk_entities
)

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
# Dùng offline mode để load từ cache local, tránh re-download mỗi lần chạy
# Nếu chưa có cache, đặt HF_DATASETS_OFFLINE=0 để download lần đầu
os.environ.setdefault("HF_HUB_OFFLINE", "1")
print("-> Đang tải dataset từ HuggingFace (offline cache)...")
ds_content_all = load_dataset("nhn309261/vietnamese-legal-docs", "content", split="data", cache_dir=hf_cache_dir)
ds_meta_all    = load_dataset("nhn309261/vietnamese-legal-docs", "metadata", split="data", cache_dir=hf_cache_dir)

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

# --- THƯ MỤC CHECKPOINT ---
import pickle
import json
CHECKPOINT_DIR = os.path.abspath(os.path.join(repo_root, ".checkpoints"))
if USE_CHECKPOINT:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"-> Đang dùng Checkpoint. Phục hồi và cứu hộ trạng thái tại: {CHECKPOINT_DIR}")
else:
    print("-> KHÔNG sử dụng Checkpoint (sẽ chạy lại từ đầu).")

# --- KHỞI TẠO & DATABASE ---
driver = get_neo4j_driver()
collection_name = os.environ.get("QDRANT_COLLECTION", "legal_rag_docs_nam")

if FLUSH_DB_ON_START:
    print("\n-> Đang Flush toàn bộ database (FLUSH_DB_ON_START=True)...")
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name=collection_name)
        print(f"   Deleted Qdrant Collection '{collection_name}'.")
    if driver:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            mode_label = "test container" if TEST_MODE else "production"
            print(f"   Deleted all nodes in Neo4j ({mode_label}).")
    ensure_collection(qdrant, collection_name=collection_name)
else:
    print("\n-> BỎ QUA Flush Database (FLUSH_DB_ON_START=False). DB hiện tại sẽ được cập nhật (upsert/merge) tiếp.")
    if not qdrant.collection_exists(collection_name):
        ensure_collection(qdrant, collection_name=collection_name)

if driver:
    init_neo4j_constraints(driver)

chunker = AdvancedLegalChunker()
# BATCH_SIZE: số tài liệu xử lý mỗi làn (chunk + LLM extract).
# Tăng lên 8 để tăng throughput; LLM calls bên dưới đã song song hóa.
BATCH_SIZE = 8
MAX_RECURSIVE_DOCS = 100000

from collections import Counter

all_chunks        = []
processed_ids     = set()
total_rels        = 0
global_rel_counts       = Counter()  # Đếm theo nhãn quan hệ
global_cross_rel_count  = 0          # Đếm riêng cross-doc (passive chain)
global_overlap_skipped  = 0          # Đếm các cặp bị bỏ qua do already_captured

# --- THỐNG KÊ MỚI CHO UNIFIED EXTRACTOR ---
total_entities          = 0
global_entity_counts    = Counter()  # Đếm theo loại thực thể
total_node_rels         = 0
global_node_rel_counts  = Counter()  # Đếm theo nhãn quan hệ node

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

    # --- THỐNG KÊ NODE VÀ QUAN HỆ NODE ---
    node_header = f"\n--- THỐNG KÊ THỰC THỂ ({label_phase}) ---"
    print(node_header)
    stats_lines.append(node_header)
    if not global_entity_counts:
        print("   (Chưa có thực thể nào)")
        stats_lines.append("   (Chưa có thực thể nào)")
    for lbl, cnt in global_entity_counts.most_common():
        line = f"   - {lbl}: {cnt} entities"
        print(line)
        stats_lines.append(line)
        
    nrel_header = f"\n--- THỐNG KÊ QUAN HỆ THỰC THỂ ({label_phase}) ---"
    print(nrel_header)
    stats_lines.append(nrel_header)
    if not global_node_rel_counts:
        print("   (Chưa có quan hệ thực thể nào)")
        stats_lines.append("   (Chưa có quan hệ thực thể nào)")
    for lbl, cnt in global_node_rel_counts.most_common():
        line = f"   - {lbl}: {cnt} node relations"
        print(line)
        stats_lines.append(line)

    sep = "=" * 60
    print(sep)
    stats_lines.append(sep)

def process_doc_batch(doc_ids, pbar, discovered_set=None, newly_processed_list=None):
    """Chunk một batch văn bản bằng đa luồng, cập nhật pbar."""
    global total_rels, global_cross_rel_count, global_overlap_skipped, total_entities, total_node_rels
    import concurrent.futures

    batch_neo4j_chunks = []

    def process_single_doc(doc_id):
        if doc_id in processed_ids: return doc_id, None, None
        idx_content = content_id_to_idx.get(doc_id)
        idx_meta    = meta_id_to_idx.get(doc_id)
        if idx_content is None or idx_meta is None: return doc_id, None, None

        content = ds_content_all[idx_content].get("content", "")
        meta    = all_meta_records[idx_meta]
        doc_num = str(meta.get("document_number", ""))
        key = normalize_doc_key(doc_num)

        doc_chunks = None
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"doc_{doc_id}.pkl") if USE_CHECKPOINT else None
        
        if USE_CHECKPOINT and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "rb") as f: doc_chunks = pickle.load(f)
            except Exception: pass
                
        if doc_chunks is None:
            doc_chunks = chunker.process_document(
                content=content,
                metadata=meta,
                global_doc_lookup=global_doc_lookup,
                skip_llm=SKIP_LLM
            )
            if USE_CHECKPOINT and checkpoint_file:
                with open(checkpoint_file, "wb") as f: pickle.dump(doc_chunks, f)
                    
        return doc_id, doc_chunks, key

    # Chạy đa luồng song song các văn bản (max 8 luồng đồng thời)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(doc_ids)) as executor:
        futures = {executor.submit(process_single_doc, doc_id): doc_id for doc_id in doc_ids}
        
        for future in concurrent.futures.as_completed(futures):
            doc_id, doc_chunks, key = future.result()
            
            if doc_id in processed_ids:
                continue
                
            if key:
                idx_meta = meta_id_to_idx.get(doc_id)
                if idx_meta is not None:
                    meta_by_docnum_lookup[key] = all_meta_records[idx_meta]

            if doc_chunks:
                all_chunks.extend(doc_chunks)
                batch_neo4j_chunks.extend(doc_chunks)
                
                first_meta = doc_chunks[0].get("neo4j_metadata", doc_chunks[0].get("metadata", {}))
                rels = first_meta.get("ontology_relations", [])
                total_rels += len(rels)

                ents = first_meta.get("entities", {})
                for k, v in ents.items():
                    if isinstance(v, list):
                        cnt = len(v)
                        global_entity_counts[k] += cnt
                        total_entities += cnt

                nrels = first_meta.get("node_relations", [])
                total_node_rels += len(nrels)
                for nr in nrels:
                    lbl = nr.get("relationship", "UNKNOWN")
                    global_node_rel_counts[lbl] += 1

                seen_pairs = set()
                for r in rels:
                    lbl = str(r.get("edge_label", r.get("label", "UNKNOWN"))).strip()
                    is_cross = r.get("is_cross_doc", False)
                    if is_cross: global_cross_rel_count += 1
                    if lbl: global_rel_counts[lbl] += 1

                    pair_key = (r.get("source_doc", ""), r.get("target_doc", ""), lbl)
                    if pair_key in seen_pairs: global_overlap_skipped += 1
                    else: seen_pairs.add(pair_key)

                    t_key = normalize_doc_key(r.get("target_doc", ""))
                    if t_key:
                        all_referenced_target_keys.add(t_key)
                        t_id = meta_docnum_to_id.get(t_key)
                        if t_id and t_id not in processed_ids and discovered_set is not None:
                            discovered_set.add(t_id)

            processed_ids.add(doc_id)
            pbar.update(1)
            
            if discovered_set is not None:
                pbar.set_postfix({"Chunks": len(all_chunks), "Rels": total_rels, "CrossRels": global_cross_rel_count, "RefDocsAdded": len(discovered_set)})
            else:
                pbar.set_postfix({"Chunks": len(all_chunks), "Rels": total_rels, "CrossRels": global_cross_rel_count, "RefDocsAdded": len(all_referenced_target_keys)})

    # ---------------------------------------------------------
    # ĐƯA LÊN GRAPH DB NGAY SAU KHI TRÍCH RELATION CỦA BATCH NÀY
    # ---------------------------------------------------------
    if driver and batch_neo4j_chunks:
        # 1. Xây dựng Knowledge Graph động (Dynamic Tree)
        build_neo4j(driver, batch_neo4j_chunks, meta_by_docnum_lookup=meta_by_docnum_lookup)
        
        # 2. Xây dựng Knowledge Graph tự do (từ LLM Extractor - legacy fallback)
        with driver.session() as session:
            for chunk in batch_neo4j_chunks:
                meta = chunk.get("neo4j_metadata", {})
                if meta.get("node_relations"):
                    continue
                triplets = meta.get("graph_triplets", [])
                for t in triplets:
                    src_name = t.get("source_node")
                    src_type = str(t.get("source_type", "Entity")).replace(" ", "")
                    tgt_name = t.get("target_node")
                    tgt_type = str(t.get("target_type", "Entity")).replace(" ", "")
                    rel_name = str(t.get("relationship", "RELATED_TO")).replace(" ", "_").upper()
                    
                    if not src_name or not tgt_name: continue
                        
                    src_lbl = ''.join(e for e in src_type if e.isalnum()) or 'Entity'
                    tgt_lbl = ''.join(e for e in tgt_type if e.isalnum()) or 'Entity'
                    rel_lbl = ''.join(e for e in rel_name if e.isalnum() or e == '_') or 'RELATED_TO'

                    q = f"""
                    MERGE (a:{src_lbl} {{name: $src_name}})
                    MERGE (b:{tgt_lbl} {{name: $tgt_name}})
                    MERGE (a)-[r:{rel_lbl}]->(b)
                    """
                    try:
                        session.run(q, src_name=str(src_name).strip(), tgt_name=str(tgt_name).strip())
                    except Exception:
                        pass
        
        # 3. Entity Enrichment mới (10 loại Entity + Node Relations)
        enrich_params = []
        for chunk in batch_neo4j_chunks:
            meta = chunk.get("neo4j_metadata", {})
            ents = meta.get("entities")
            node_rels = meta.get("node_relations")
            if ents or node_rels:
                enrich_params.append({
                    "qdrant_id": chunk.get("chunk_id", ""),
                    "entities": ents or {},
                    "node_relations": node_rels or [],
                })
        if enrich_params:
            try:
                enrich_chunk_entities(driver, enrich_params, use_apoc=False)
            except Exception as e:
                print(f" Lỗi khi enrich entity: {e}")

# ============================================================
# PHASE 1: Chunk + Embed ORIGINAL documents (8000)
# ============================================================
msg = "\n" + "=" * 60 + "\nPHASE 1: Chunking ORIGINAL documents\n" + "=" * 60
print(msg)
stats_lines.append(msg)

target_ids_list = list(target_ids)
if SAMPLE_LIMIT:
    target_ids_list = target_ids_list[:SAMPLE_LIMIT]

# Giới hạn Phase 2+3 khi TEST_MODE để tránh VB vệ tinh vọt lên hàng nghìn doc
# Công thức: Phase 2 cap = 2x số gốc, Phase 3 cap = 1x số gốc
REF_LIMIT_PHASE2 = (SAMPLE_LIMIT * 2) if SAMPLE_LIMIT else MAX_RECURSIVE_DOCS
REF_LIMIT_PHASE3 = (SAMPLE_LIMIT * 1) if SAMPLE_LIMIT else MAX_RECURSIVE_DOCS

discovered_ref_ids = set()  # Văn bản tham chiếu tìm thấy ở Pha 1

pbar1 = tqdm(total=len(target_ids_list), desc="PHASE 1: Chunking ORIGINAL", unit="doc")
for i in range(0, len(target_ids_list), BATCH_SIZE):
    process_doc_batch(target_ids_list[i:i+BATCH_SIZE], pbar1, discovered_set=discovered_ref_ids)
pbar1.close()

print_top_sectors(list(processed_ids), "TỔNG LUỸ KẾ SAU PHA 1 (GỐC)", meta_id_to_idx, all_meta_records)

# Giới hạn số văn bản tham chiếu Phase 2
if len(discovered_ref_ids) > REF_LIMIT_PHASE2:
    discovered_ref_ids = set(random.sample(list(discovered_ref_ids), REF_LIMIT_PHASE2))
    print(f"[TEST_MODE] Cap Phase 2 refs tại {REF_LIMIT_PHASE2} docs")

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

# Giới hạn Phase 3 refs khi TEST_MODE
if len(discovered_depth2_ids) > REF_LIMIT_PHASE3:
    discovered_depth2_ids = set(random.sample(list(discovered_depth2_ids), REF_LIMIT_PHASE3))
    print(f"[TEST_MODE] Cap Phase 3 refs tại {REF_LIMIT_PHASE3} docs")

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

# Đưa ghost nodes lên Neo4j
if driver and ghost_nodes:
    print(f"-> Đang đẩy {len(ghost_nodes)} ghost nodes lên Neo4j...")
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

# ============================================================
# PHASE 5: Embedding + Upload Qdrant
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: Embedding & Upsert Qdrant")
print("=" * 60)

batch_size = 64
pbar5 = tqdm(total=len(all_chunks), desc="PHASE 5: Embed & Upsert", unit="chunk")

import gc

uploaded_chunks_file = os.path.join(CHECKPOINT_DIR, "uploaded_chunks_phase5.json") if USE_CHECKPOINT else None
uploaded_chunk_ids = set()
if uploaded_chunks_file and os.path.exists(uploaded_chunks_file):
    try:
        with open(uploaded_chunks_file, "r") as f:
            uploaded_chunk_ids = set(json.load(f))
        print(f"   [Phục hồi] Tìm thấy {len(uploaded_chunk_ids)} chunks đã embed từ lần chạy trước!")
    except:
        pass

actual_embed_count = 0
for i in range(0, len(all_chunks), batch_size):
    chunk_batch = all_chunks[i:i+batch_size]
    
    # Lọc những chunk chưa nhúng
    chunks_to_process = [c for c in chunk_batch if c["chunk_id"] not in uploaded_chunk_ids]
    
    if chunks_to_process:
        texts_to_embed = [c.get("text_to_embed", c.get("chunk_text", "")) for c in chunks_to_process]
        dense_batch = embedder.encode_dense(texts_to_embed)
        sparse_batch = embedder.encode_sparse_documents(texts_to_embed)
        
        points = []
        for idx, chunk in enumerate(chunks_to_process):
            q_payload = chunk.get("qdrant_metadata", chunk.get("metadata", {}))
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
                vector={"dense": dense_batch[idx], "sparse": sparse_batch[idx]},
                payload=q_payload
            ))
            
        qdrant.upsert(collection_name=collection_name, points=points)
        actual_embed_count += len(chunks_to_process)  # fix: đếm chunk thực sự embed
        
        if uploaded_chunks_file:
            uploaded_chunk_ids.update([c["chunk_id"] for c in chunks_to_process])
            if (i // batch_size) % 10 == 0:
                with open(uploaded_chunks_file, "w") as f:
                    json.dump(list(uploaded_chunk_ids), f)
                    
        del texts_to_embed, dense_batch, sparse_batch, points
    
    # fix: update đúng số chunk thực sự xử lý trong vòng này (kể cả đã skip)
    pbar5.update(len(chunk_batch))
    
    del chunk_batch, chunks_to_process
    if i % 5000 == 0:
        gc.collect()

if uploaded_chunks_file:
    with open(uploaded_chunks_file, "w") as f:
        json.dump(list(uploaded_chunk_ids), f)

pbar5.close()
skipped_embed = len(all_chunks) - actual_embed_count
print(f"   Embedded & upserted: {actual_embed_count} chunks | Skipped (already uploaded): {skipped_embed}")

if driver:
    driver.close()

# --- FINAL STATISTICS ---
import time
import datetime

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — STATISTICS")
print("=" * 60)

stats_lines.extend([
    f"Run timestamp       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Original docs       : {len(target_ids_list)}",
    f"Reference docs      : {len(disco_ids_list)}",
    f"Ghost nodes         : {len(ghost_nodes)}",
    f"Total chunks        : {len(all_chunks)}",
    f"Total doc relations : {total_rels}",
    f"Total entities      : {total_entities}",
    f"Total node relations: {total_node_rels}",
    f"Qdrant collection   : {collection_name}",
    "",
    "Document Edge label breakdown:",
])
for label, count in global_rel_counts.most_common():
    stats_lines.append(f"  {label:15s}: {count}")

stats_lines.append("\nEntity type breakdown:")
for label, count in global_entity_counts.most_common():
    stats_lines.append(f"  {label:15s}: {count}")

stats_lines.append("\nNode Relation label breakdown:")
for label, count in global_node_rel_counts.most_common():
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
report_file = os.path.join(repo_root, "result_8k.txt")
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

