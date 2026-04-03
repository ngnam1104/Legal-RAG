"""
Benchmark tốc độ truy xuất tài liệu (Hybrid Search + Rerank)
Chạy: python scripts/tests/benchmark_retrieval.py
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.retrieval.hybrid_search import retriever

TEST_QUERIES = [
    "phòng cháy chữa cháy",
    "xử phạt vi phạm giao thông",
    "quyền lợi người lao động",
    "hợp đồng lao động thời vụ",
    "bảo vệ môi trường",
]

def benchmark():
    print("=" * 70)
    print("📊 BENCHMARK TRUY XUẤT TÀI LIỆU (Hybrid Search + Rerank)")
    print("=" * 70)
    
    # Warm-up (load model nếu chưa load)
    print("\n⏳ Warm-up (nạp model nếu cần)...")
    t0 = time.perf_counter()
    _ = retriever.search("test", limit=1, expand_context=False)
    print(f"   Warm-up xong: {time.perf_counter()-t0:.2f}s\n")
    
    results = []
    
    for query in TEST_QUERIES:
        print(f"🔎 Query: \"{query}\"")
        try:
            # Bước 1: Broad Retrieve (Hybrid Dense + Sparse)
            print(f"   [1/3] Đang lấy văn bản thô (Hybrid Search)...")
            t1 = time.perf_counter()
            broad_hits = retriever.broad_retrieve(query, top_k=40)  # Tăng lên 40 cho Sector Overview
            t2 = time.perf_counter()
            broad_time = t2 - t1
            print(f"   [1/3] ✅ Xong {len(broad_hits)} hits → {broad_time:.3f}s")
            
            # Bước 2: Rerank
            print(f"   [2/3] Đang sắp xếp lại với Reranker model ({len(broad_hits)} cặp)...")
            reranked = retriever.reranker.rerank(query, broad_hits, top_k=20)
            t3 = time.perf_counter()
            rerank_time = t3 - t2
            print(f"   [2/3] ✅ Xong {len(reranked)} hits → {rerank_time:.3f}s")
            
            # Bước 3: Context Expand
            print(f"   [3/3] Đang mở rộng ngữ cảnh (Context Expand)...")
            expanded = retriever.expand_context(reranked[:5], max_neighbors=8)
            t4 = time.perf_counter()
            expand_time = t4 - t3
            print(f"   [3/3] ✅ Xong {len(expanded)} hits → {expand_time:.3f}s")
            
            total = t4 - t1
            
            print(f"   ─────────────────────────────────")
            print(f"   TỔNG PIPELINE  :             → {total:.3f}s")
            print()
            
            results.append({
                "query": query,
                "broad_count": len(broad_hits),
                "broad_time": broad_time,
                "rerank_count": len(reranked),
                "rerank_time": rerank_time,
                "expand_count": len(expanded),
                "expand_time": expand_time,
                "total": total,
            })
        except Exception as e:
            import traceback
            print(f"\n❌ Lỗi chết pipeline ở câu: {query}")
            print(f"Chi tiết lỗi: {e}")
            traceback.print_exc()
            print("\nTiếp tục với câu tiếp theo...\n")
    
    # Summary
    print("=" * 70)
    print("📈 TỔNG KẾT")
    print("=" * 70)
    avg_broad = sum(r["broad_time"] for r in results) / len(results)
    avg_rerank = sum(r["rerank_time"] for r in results) / len(results)
    avg_expand = sum(r["expand_time"] for r in results) / len(results)
    avg_total = sum(r["total"] for r in results) / len(results)
    
    print(f"  {'Bước':<20} {'Min':>8} {'Avg':>8} {'Max':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'Broad Retrieve':<20} {min(r['broad_time'] for r in results):>7.3f}s {avg_broad:>7.3f}s {max(r['broad_time'] for r in results):>7.3f}s")
    print(f"  {'Rerank':<20} {min(r['rerank_time'] for r in results):>7.3f}s {avg_rerank:>7.3f}s {max(r['rerank_time'] for r in results):>7.3f}s")
    print(f"  {'Context Expand':<20} {min(r['expand_time'] for r in results):>7.3f}s {avg_expand:>7.3f}s {max(r['expand_time'] for r in results):>7.3f}s")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'TỔNG PIPELINE':<20} {min(r['total'] for r in results):>7.3f}s {avg_total:>7.3f}s {max(r['total'] for r in results):>7.3f}s")
    print()

if __name__ == "__main__":
    benchmark()
