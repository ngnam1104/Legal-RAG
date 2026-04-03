"""
Báo cáo chi tiết hiệu năng truy xuất RAG phân tách theo từng mode.
Chạy: python scripts/tests/retrieval_report.py
"""
import sys
import os
import time
import json
import asyncio
from typing import List, Dict, Any

# Thêm root vào sys.path để import được backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.retrieval.hybrid_search import retriever
from backend.agent.flow_legal_qa import rewrite_legal_query, build_legal_context, grade_documents
from backend.agent.flow_sector_search import transform_sector_query, strict_filter_docs
from backend.agent.flow_conflict_analyzer import IE_PROMPT, HYDE_PROMPT, extract_json_conflict
from backend.llm.factory import chat_completion

# --- CONFIGURATION ---
TEST_CASES = [
    {
        "mode": "LEGAL_QA",
        "query": "Mức phạt tiền đối với hành vi không đeo khẩu trang nơi công cộng theo Nghị định 117/2020/NĐ-CP",
        "description": "Kiểm tra khả năng truy xuất chính xác mức phạt hành chính y tế"
    },
    {
        "mode": "SECTOR_SEARCH",
        "query": "Các quy định về quản lý ngoại hối của Ngân hàng Nhà nước",
        "description": "Kiểm tra khả năng liệt kê danh mục văn bản ngành ngân hàng"
    },
    {
        "mode": "CONFLICT_ANALYZER",
        "query": "Quy định về thời điểm lập hóa đơn điện tử đối với dịch vụ viễn thông",
        "description": "Kiểm tra khả năng truy xuất quy định hóa đơn (Decree 123/2020)"
    }
]

# Save report in tests folder
REPORT_FILE = os.path.join(os.path.dirname(__file__), "retrieval_metrics_report.txt")

async def run_legal_qa_retrieval(query: str):
    """Mô phỏng luồng truy xuất của Legal QA"""
    t0 = time.perf_counter()
    
    # Phase 0: Rewrite
    rewrite_data = rewrite_legal_query(query)
    rewrite_query = rewrite_data.get("rewritten_query", query)
    filters = rewrite_data.get("filters", {})
    t_rewrite = time.perf_counter() - t0
    
    # Phase 1-3: Hybrid Search
    t1 = time.perf_counter()
    hits = retriever.search(
        query=rewrite_query,
        legal_type=filters.get("legal_type"),
        doc_number=filters.get("doc_number"),
        limit=5,
        expand_context=True
    )
    
    # Fallback if no hits (as in flow_legal_qa.py)
    if not hits and (filters.get("legal_type") or filters.get("doc_number")):
        hits = retriever.search(query=rewrite_query, limit=5, expand_context=True)
        
    t_search = time.perf_counter() - t1
    
    # Phase 4: Relevancy Grading
    t2 = time.perf_counter()
    relevancy = "NO"
    if hits:
        # Convert hits list to context string for grading
        context_str = build_legal_context(hits)
        is_relevant = grade_documents(rewrite_query, context_str)
        relevancy = "YES" if is_relevant else "NO"
    t_grade = time.perf_counter() - t2
    
    total_time = time.perf_counter() - t0
    
    return {
        "rewrite_output": rewrite_query,
        "filters": filters,
        "hits": hits,
        "relevancy": relevancy,
        "latencies": {
            "rewrite": t_rewrite,
            "search": t_search,
            "grade": t_grade
        },
        "total": total_time
    }

async def run_sector_search_retrieval(query: str):
    """Mô phỏng luồng truy xuất của Sector Search"""
    t0 = time.perf_counter()
    
    # Phase 0: Transform
    tf = transform_sector_query(query)
    kw = tf.get("keywords", query)
    fi = tf.get("filters", {})
    t_transform = time.perf_counter() - t0
    
    # Phase 1-3: Search
    t1 = time.perf_counter()
    hits = retriever.search(
        query=kw,
        limit=15,
        expand_context=False,
        legal_type=fi.get("legal_type"),
        doc_number=fi.get("doc_number")
    )
    t_search = time.perf_counter() - t1
    
    # Phase 4: Strict Filtering
    t2 = time.perf_counter()
    filtered_hits = strict_filter_docs(query, hits)
    t_filter = time.perf_counter() - t2
    
    total_time = time.perf_counter() - t0
    
    return {
        "rewrite_output": kw,
        "filters": fi,
        "hits": filtered_hits,
        "relevancy": "YES" if len(filtered_hits) > 0 else "NO",
        "latencies": {
            "transform": t_transform,
            "search": t_search,
            "filter": t_filter
        },
        "total": total_time
    }

async def run_conflict_analyzer_retrieval(query: str):
    """Mô phỏng luồng truy xuất của Conflict Analyzer"""
    t0 = time.perf_counter()
    
    # Phase 0: HyDE (Simulated for the query)
    hyde_prompt = HYDE_PROMPT.format(
        chu_the="Tổ chức, cá nhân",
        hanh_vi=query,
        dieu_kien="Trong điều kiện thực tế",
        he_qua="Theo quy định pháp luật"
    )
    hyde_query = chat_completion([{"role": "user", "content": hyde_prompt}], temperature=0.3)
    t_hyde = time.perf_counter() - t0
    
    # Phase 1-3: Search with include_inactive
    t1 = time.perf_counter()
    hits = retriever.search(query=hyde_query, limit=5, expand_context=True, include_inactive=True)
    t_search = time.perf_counter() - t1
    
    total_time = time.perf_counter() - t0
    
    return {
        "rewrite_output": hyde_query[:100] + "...",
        "filters": {"include_inactive": True},
        "hits": hits,
        "relevancy": "YES" if hits else "NO",
        "latencies": {
            "hyde": t_hyde,
            "search": t_search
        },
        "total": total_time
    }

async def generate_report():
    print(f"🚀 Starting Multi-Mode Retrieval Metrics Report...")
    
    results = []
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"  [{i}/{len(TEST_CASES)}] Processing Mode: {test['mode']} | Query: \"{test['query']}\"")
        
        if test['mode'] == "LEGAL_QA":
            res = await run_legal_qa_retrieval(test['query'])
        elif test['mode'] == "SECTOR_SEARCH":
            res = await run_sector_search_retrieval(test['query'])
        else: # CONFLICT_ANALYZER
            res = await run_conflict_analyzer_retrieval(test['query'])
            
        res['original_query'] = test['query']
        res['mode'] = test['mode']
        results.append(res)

    # Viết báo cáo
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("📊 BÁO CÁO HIỆU NĂNG TRUY XUẤT RAG (MULTI-MODE METRICS)\n")
        f.write(f"📅 Thời gian thực hiện: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 90 + "\n\n")
        
        for i, res in enumerate(results, 1):
            f.write(f"TEST {i}: [{res['mode']}] \"{res['original_query']}\"\n")
            f.write(f"  ├─ Rewrite/Transform: {res['rewrite_output']}\n")
            f.write(f"  ├─ Extracted Filters: {json.dumps(res['filters'], ensure_ascii=False)}\n")
            f.write(f"  ├─ Latency (Seconds):\n")
            for phase, val in res['latencies'].items():
                f.write(f"  │  - {phase.capitalize():<15}: {val:.3f}s\n")
            f.write(f"  │  => TOTAL PIPELINE   : {res['total']:.3f}s\n")
            
            hits = res['hits']
            top_score = hits[0].get("score", 0) if hits else 0
            
            f.write(f"  ├─ Accuracy & Quality:\n")
            f.write(f"  │  - Result Count      : {len(hits)}\n")
            f.write(f"  │  - Top-1 Rerank Score: {top_score:.4f}\n")
            f.write(f"  │  - LLM Relevancy     : {res['relevancy']}\n")
            
            if hits:
                f.write(f"  └─ Top Result: {hits[0].get('title', 'N/A')} ({hits[0].get('article_ref', 'N/A')})\n")
            else:
                f.write(f"  └─ Top Result: NONE\n")
            f.write("\n")

        # Thống kê tổng quan
        f.write("=" * 90 + "\n")
        f.write("📈 THỐNG KÊ TRUNG BÌNH (OVERALL SUMMARY)\n")
        f.write("=" * 90 + "\n")
        avg_time = sum(r['total'] for r in results) / len(results)
        avg_score = sum(h['hits'][0].get('score', 0) if h['hits'] else 0 for h in results) / len(results)
        relevancy_rate = sum(1 for r in results if r['relevancy'] == 'YES') / len(results) * 100
        
        f.write(f"  - Tổng thời gian TB: {avg_time:.3f}s\n")
        f.write(f"  - Điểm tự tin TB   : {avg_score:.4f}\n")
        f.write(f"  - Tỷ lệ Relevancy  : {relevancy_rate:.1f}%\n")
        f.write("=" * 90 + "\n")

    print(f"✅ Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    asyncio.run(generate_report())
