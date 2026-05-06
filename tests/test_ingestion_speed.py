import os
import sys
import time
import uuid
import datetime

# Thiết lập đường dẫn
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# Tải env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(repo_root, ".env"), override=False)
except ImportError:
    pass

# Output file
RESULT_FILE = os.path.join(repo_root, "test_results.txt")

import backend.ingestion.extractor.relations as rel_module
import backend.ingestion.extractor.entities as ent_module
from backend.models.llm_factory import get_client

_original_extract = rel_module.extract_ontology_relationships_batch
_original_prompt_build = ent_module.build_unified_prompt
_original_parse = ent_module.parse_unified_response

# Timers
PRE_TIME = 0.0
INFERENCE_TOTAL_TIME = 0.0
NETWORK_TIME = 0.0
POST_TIME = 0.0

def timed_prompt_build(*args, **kwargs):
    global PRE_TIME
    start = time.time()
    res = _original_prompt_build(*args, **kwargs)
    PRE_TIME += (time.time() - start)
    return res

def timed_parse(*args, **kwargs):
    global POST_TIME
    start = time.time()
    res = _original_parse(*args, **kwargs)
    POST_TIME += (time.time() - start)
    return res

# Cần patch cả phương thức batch_chat_completion của client
llm_client = get_client()
_original_batch_chat = llm_client.batch_chat_completion
_original_generate = llm_client.llm_engine.generate

def timed_batch_chat(*args, **kwargs):
    global INFERENCE_TOTAL_TIME
    start = time.time()
    res = _original_batch_chat(*args, **kwargs)
    INFERENCE_TOTAL_TIME += (time.time() - start)
    return res

def timed_generate(*args, **kwargs):
    global NETWORK_TIME
    start = time.time()
    res = _original_generate(*args, **kwargs)
    NETWORK_TIME += (time.time() - start)
    return res

# Apply patches
ent_module.build_unified_prompt = timed_prompt_build
ent_module.parse_unified_response = timed_parse
llm_client.batch_chat_completion = timed_batch_chat
llm_client.llm_engine.generate = timed_generate

import pandas as pd
from datasets import load_dataset
from backend.ingestion.chunker.core import AdvancedLegalChunker
from backend.ingestion.chunker.metadata import normalize_doc_key
from backend.models.embedder import embedder
import json

def measure_time(func):
    """Decorator đo thời gian chạy."""
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"⏱️ Thời gian thực thi [{func.__name__}]: {(end - start):.4f} giây")
        return res, (end - start)
    return wrapper

@measure_time
def run_chunker_pipeline(chunker, doc_id, text, metadata, global_doc_lookup):
    # process_document already runs chunking AND llm extraction
    return chunker.process_document(content=text, metadata=metadata, global_doc_lookup=global_doc_lookup)

@measure_time
def run_embedding(texts):
    if not texts:
        return []
    return embedder.encode_dense(texts)

# Mock class cho global_doc_lookup
class MockLookup(dict):
    """
    Giả lập DynamicContentLookup (dict subclass) để tránh lỗi 
    'argument of type MockLookup is not iterable'.
    Trả về doc_text cho bất kỳ key nào chưa có trong dict.
    """
    def __init__(self, doc_text: str, doc_key: str = ""):
        super().__init__()
        self.doc_text = doc_text
        # Pre-populate với key chuẩn hoá của văn bản để `key in lookup` = True
        if doc_key:
            self[doc_key] = doc_text
    
    def get(self, key, default=None):
        if key in self:
            return super().get(key, default)
        return self.doc_text  # Fallback: trả về nội dung văn bản duy nhất

def test_single_doc_ingestion(target_doc_num="38/2020/QĐ-UBND"):
    print(f"==================================================")
    print(f"🚀 BẮT ĐẦU TEST INGESTION SPEED (1 VĂN BẢN: {target_doc_num})")
    print(f"==================================================")
    
    # 1. Load Dataset (offline cache)
    print("\n-> Đang tải dataset từ HuggingFace (offline cache)...")
    hf_cache_dir = os.path.abspath(os.path.join(repo_root, ".cache", "huggingface"))
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    try:
        ds_content = load_dataset("nhn309261/vietnamese-legal-docs", "content", split="data", cache_dir=hf_cache_dir)
        ds_meta = load_dataset("nhn309261/vietnamese-legal-docs", "metadata", split="data", cache_dir=hf_cache_dir)
    except Exception as e:
        print(f"❌ Không thể tải dataset offline: {e}")
        return

    # Tìm document theo document_number
    target_idx = None
    target_meta = None
    for i, meta in enumerate(ds_meta):
        if str(meta.get("document_number", "")).strip() == target_doc_num:
            target_idx = i
            target_meta = meta
            break
            
    if target_idx is None:
        print(f"❌ Không tìm thấy văn bản '{target_doc_num}' trong dataset.")
        # Fallback lấy văn bản đầu tiên
        target_idx = 0
        target_meta = ds_meta[0]
        print(f"⚠️ Fallback: Dùng văn bản {target_meta.get('document_number')} thay thế.")

    doc_id = target_meta.get("id", str(uuid.uuid4()))
    
    # Lấy nội dung tương ứng
    # dataset content mapping by id
    doc_text = ""
    for row in ds_content:
        if str(row.get("id")) == str(doc_id):
            doc_text = row.get("content", "")
            break
            
    if not doc_text:
        print(f"❌ Không tìm thấy nội dung (content) cho văn bản {doc_id}.")
        return

    print(f"\n✅ Đã tìm thấy văn bản: {target_meta.get('document_number')} (ID: {doc_id})")
    print(f"   Độ dài nội dung: {len(doc_text)} ký tự.")

    # 2 & 3. Khởi tạo Chunker & LLM Extraction
    print("\n" + "-"*50)
    print("▶️ BƯỚC 1 & 2: CHUNKING & LLM EXTRACTION (process_document)")
    print("-"*50)
    chunker = AdvancedLegalChunker()
    doc_key = normalize_doc_key(target_meta.get("document_number", ""))
    global_lookup = MockLookup(doc_text, doc_key=doc_key)
    print(f"   (doc_key chuẩn hoá: '{doc_key}')")
    
    global PRE_TIME, INFERENCE_TOTAL_TIME, NETWORK_TIME, POST_TIME
    PRE_TIME = 0.0
    INFERENCE_TOTAL_TIME = 0.0
    NETWORK_TIME = 0.0
    POST_TIME = 0.0
    
    extracted_chunks, t_total = run_chunker_pipeline(chunker, doc_id, doc_text, target_meta, global_lookup)
    
    # Tính toán overhead của client (chuyển đổi chuỗi, cấp phát thread)
    local_client_overhead = INFERENCE_TOTAL_TIME - NETWORK_TIME
    if local_client_overhead < 0: local_client_overhead = 0.0
    
    t_llm_total = PRE_TIME + INFERENCE_TOTAL_TIME + POST_TIME
    t_chunking_only = t_total - (t_llm_total)
    
    print(f"   -> Tạo ra và trích xuất thành công {len(extracted_chunks)} chunks.")
    print(f"   -> 🧩 Thời gian Chunking (Regex/FSM): {t_chunking_only:.4f}s")
    print(f"   -> 🧠 CHI TIẾT LLM EXTRACTION:")
    print(f"      + (1) Tiền xử lý (Build Prompt):     {PRE_TIME:.4f}s  (CPU local)")
    print(f"      + (2) Overhead của LLM Client:       {local_client_overhead:.4f}s  (CPU local - Threading/Formatting)")
    print(f"      + (3) THỜI GIAN GỌI MẠNG (Tới ICLLM): {NETWORK_TIME:.4f}s  (Network/GPU Server)")
    print(f"      + (4) Hậu xử lý (Parse JSON):        {POST_TIME:.4f}s  (CPU local)")
    
    # In kết quả post-processing relation
    if extracted_chunks:
        first_chunk_meta = extracted_chunks[0].get("neo4j_metadata", {})
        rels = first_chunk_meta.get("node_relations", [])
        doc_rels = first_chunk_meta.get("ontology_relations", [])
        entities = first_chunk_meta.get("entities", {})
        
        print("\n📊 [KẾT QUẢ POST-PROCESSING]")
        print(f"   - Số lượng Entities trích xuất: {sum(len(v) for v in entities.values() if isinstance(v, list))}")
        for k, v in entities.items():
            if v:
                print(f"     + {k}: {v[:3]}...")
                
        print(f"\n   - Số lượng Node Relations (Bên trong văn bản): {len(rels)}")
        if rels:
            # In cấu trúc thực tế của dict đầu tiên để xác nhận keys
            print(f"     [DEBUG keys]: {list(rels[0].keys())}")
        for r in rels[:5]:
            src  = r.get("source_node") or r.get("source") or r.get("src")
            tgt  = r.get("target_node") or r.get("target") or r.get("tgt")
            rel  = r.get("relationship") or r.get("relation") or r.get("rel")
            st   = r.get("source_type") or r.get("src_type") or "?"
            tt   = r.get("target_type") or r.get("tgt_type") or "?"
            print(f"     + ({st}) '{src}' -[{rel}]-> ({tt}) '{tgt}'")
            
        print(f"\n   - Số lượng Document Relations (Liên kết văn bản chéo): {len(doc_rels)}")
        for r in doc_rels[:5]:
            print(f"     + {r.get('source_doc')} -[{r.get('edge_label')}]-> {r.get('target_doc')}")
    else:
        print("   -> Không có chunk nào được trích xuất.")

    # 4. Embedding
    print("\n" + "-"*50)
    print("▶️ BƯỚC 3: EMBEDDING (HUGGINGFACE MODEL)")
    print("-"*50)
    texts_to_embed = [c["text_to_embed"] for c in extracted_chunks if c.get("text_to_embed")]
    print(f"   -> Số chunks sẽ embed: {len(texts_to_embed)}")
    embeddings, t_embed = run_embedding(texts_to_embed)
    print(f"   -> Đã embed {len(embeddings)} vectors (Kích thước: {len(embeddings[0]) if embeddings else 0} chiều).")

    # 5. Tong ket
    summary_lines = [
        "==================================================",
        f"TONG KET THOI GIAN (1 VAN BAN: {target_doc_num})",
        "==================================================",
        f"   1. Chunking (Regex):      {t_chunking_only:.4f}s",
        f"   2. LLM Pre-processing:    {PRE_TIME:.4f}s",
        f"   3. LLM Client Overhead:   {local_client_overhead:.4f}s",
        f"   4. LLM Network Request:   {NETWORK_TIME:.4f}s",
        f"   5. LLM Post-processing:   {POST_TIME:.4f}s",
        f"   6. Embedding (API):       {t_embed:.4f}s",
        f"   ----------------------------------",
        f"   Tong thoi gian:    {(t_total + t_embed):.4f}s",
        "==================================================",
    ]
    for line in summary_lines:
        print(line)

    # Ghi ket qua ra file
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"test_ingestion_speed.py -- {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\nKet qua da luu vao: {RESULT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=str, default="38/2020/QĐ-UBND", help="Document number de test")
    args = parser.parse_args()
    
    test_single_doc_ingestion(args.doc)
