import os
import sys
import asyncio
import json
import time
import re
from datetime import datetime
from typing import Dict, Any, List

# Cấu hình đường dẫn: 3 cấp lên tới Root (scripts/tests/file.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.retrieval.chunker import AdvancedLegalChunker
from backend.retrieval.embedder import get_embedder
from backend.agent.graph import app  # LangGraph entry point

REPORT_FILE = os.path.join(os.path.dirname(__file__), "verification_report.txt")

def log_to_report(message: str):
    print(message)
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

async def test_agent_mode(mode: str, query: str):
    log_to_report(f"\n" + "="*40)
    log_to_report(f"🤖 MODE: {mode.upper()}")
    log_to_report(f"❓ QUERY: {query}")
    log_to_report("="*40)
    
    start_total = time.perf_counter()
    
    initial_state = {
        "query": query,
        "session_id": f"test_session_{mode}_{int(time.time())}",
        "mode": mode,
        "file_path": None,
        "top_k": 3,
        "use_reflection": False,
        "use_rerank": True,
        "history": [],
        "condensed_query": "",
        "file_chunks": [],
        "rewritten_query": "",
        "filters": {},
        "hits": [],
        "context": "",
        "answer": "",
        "references": [],
        "retry_count": 0,
        "grade_retry_count": 0,
        "is_graded_pass": False,
        "metrics": {}
    }
    
    config = {"configurable": {"thread_id": f"test_{mode}"}}
    
    try:
        # Chúng ta sẽ thực thi LangGraph. 
        # Lưu ý: Các node trong LangGraph đã có logic tính thời gian riêng (metrics), 
        # nhưng ở đây chúng ta tính tổng quát từ bên ngoài.
        
        result = await app.ainvoke(initial_state, config=config)
        
        end_total = time.perf_counter()
        total_dur = end_total - start_total
        
        docs = result.get('hits', result.get('retrieved_docs', []))
        answer = result.get("answer", "")
        
        log_to_report(f"✅ Status: Completed in {total_dur:.2f}s")
        log_to_report(f"🔎 Retrieval Count: {len(docs)} chunks")
        
        if docs:
            # Ước tính thời gian retrieval (vì không thể can thiệp sâu vào node mà không sửa code agent)
            # Ở đây ta lấy từ metrics nếu agent có ghi lại
            metrics = result.get("metrics", {})
            for k, v in metrics.items():
                log_to_report(f"   ⏱️ {k}: {v:.2f}s")
                
        if answer:
            log_to_report("-" * 20)
            log_to_report(f"💬 Answer Snippet:\n{answer[:500]}...")
            log_to_report("-" * 20)
            
        return result
    except Exception as e:
        log_to_report(f"❌ Error in mode {mode}: {str(e)}")
        return None

async def main():
    # Khởi tạo file report mới
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"法 LEGAL RAG VERIFICATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Environment: Python {sys.version}\n")
        f.write(f"Cache Path: {os.environ.get('HF_HOME')}\n")
        f.write("="*60 + "\n\n")

    file_path = r"d:\iCOMM\Legal-RAG\legal_docs\txt\Luật 02-2011-QH13 Quốc hội.txt"
    
    # 1. SIMULATE INGESTION PIPELINE
    log_to_report("🚀 STEP 1: INGESTION PIPELINE SIMULATION")
    if os.path.exists(file_path):
        start_ingest = time.perf_counter()
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunker = AdvancedLegalChunker()
        chunks = chunker.process_document(content, {"title": "Luật Khiếu nại 2011"})
        chunk_time = time.perf_counter() - start_ingest
        log_to_report(f"✅ Chunking finished: {len(chunks)} chunks in {chunk_time:.2f}s")
        
        start_embed = time.perf_counter()
        embedder = get_embedder()
        model_load_time = time.perf_counter() - start_embed
        log_to_report(f"✅ Embedder Ready (Singleton/Cache check) in {model_load_time:.4f}s")
        
        start_enc = time.perf_counter()
        # Thử encode 5 chunks để đo stress test nhẹ
        test_texts = [c["chunk_text"] for c in chunks[:5]]
        dense, sparse = embedder.encode_hybrid(test_texts)
        enc_time = time.perf_counter() - start_enc
        log_to_report(f"✅ Embedding 5 chunks finished in {enc_time:.2f}s (Avg: {enc_time/5:.3f}s/chunk)")
    else:
        log_to_report(f"⚠️ Skip Ingestion: File {file_path} not found.")

    # 2. EXTENDED AGENT TESTING
    log_to_report("\n" + "🚀 STEP 2: MULTI-QUERY AGENT TESTING (3 MODES)")
    
    test_cases = [
        # LEGAL QA
        ("legal_qa", "Thủ tục khiếu nại lần đầu và thời gian giải quyết tối đa là bao lâu?"),
        ("legal_qa", "Người khiếu nại có những quyền và nghĩa vụ gì?"),
        
        # SECTOR SEARCH
        ("sector_search", "Tìm danh sách các văn bản quy định về khiếu nại hành chính"),
        
        # CONFLICT ANALYSIS
        ("conflict_analysis", "Luật khiếu nại 2011 có mâu thuẫn gì với Hiến pháp về quyền khiếu nại của công dân không?")
    ]
    
    for mode, q in test_cases:
        try:
            await test_agent_mode(mode, q)
            # Nghỉ 65 giây giữa các thao tác để hồi phục hoàn toàn 6000 TPM limit (Tokens Per Minute) của Groq Dev Tier.
            log_to_report("\n⏳ Resting for 65s to fully reset Groq TPM Limit (6000 TPM)...")
            await asyncio.sleep(65)
        except Exception as e:
            log_to_report(f"⚠️ Failed to complete task for {mode}: {str(e)}")

    log_to_report("\n" + "="*60)
    log_to_report("🏁 ALL TESTS COMPLETED. Check 'scripts/verification_report.txt' for full logs.")
    log_to_report("="*60)

if __name__ == "__main__":
    asyncio.run(main())
