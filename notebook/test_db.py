"""
test_db.py — Standalone benchmark script for Qdrant + Neo4j.
Chạy độc lập sau khi đã ingestion xong, không cần chạy lại pipeline.

Usage:
    python notebook/test_db.py
"""
import sys
import os
import time
import datetime

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

# ===================================================================
os.environ["QDRANT_URL"]        = "http://localhost:6335"
os.environ["QDRANT_API_KEY"]    = ""
os.environ["NEO4J_URI"]         = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"]    = "neo4j"
os.environ["NEO4J_PASSWORD"]    = "u7aGQYEWeFJD-jyeHB4ATtoAud73PptW35M1RzFlT-0"
COLLECTION_NAME = "legal_hybrid_rag_docs_8K"   # ← đổi nếu cần
# ===================================================================

from backend.retrieval.embedder import embedder
from backend.retrieval.vector_db import client as qdrant
from backend.retrieval.graph_db import get_neo4j_driver

benchmark_results = []

def bench(label, fn):
    """Chạy fn(), đo thời gian ms, in kết quả."""
    t0 = time.perf_counter()
    try:
        result  = fn()
        elapsed = (time.perf_counter() - t0) * 1000
        status  = "OK"
    except Exception as e:
        result  = None
        elapsed = (time.perf_counter() - t0) * 1000
        status  = f"ERR: {e}"
    hit_count = len(result) if isinstance(result, list) else (1 if result else 0)
    print(f"  [{status[:2]:2s}] {label:<52} {elapsed:7.1f}ms  hits={hit_count}")
    benchmark_results.append({"label": label, "status": status, "ms": elapsed, "hits": hit_count})

# --- Vector mẫu ---
SAMPLE_TEXT   = "Luật số 80/2015/QH13 ban hành văn bản quy phạm pháp luật"
sample_dense  = embedder.encode_dense([SAMPLE_TEXT])[0]
sample_sparse = embedder.encode_sparse_documents([SAMPLE_TEXT])[0]

print("=" * 60)
print(f"QUERY BENCHMARK — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Collection: {COLLECTION_NAME}")
print("=" * 60)

# ── QDRANT ──────────────────────────────────────────────────────────
print("\n── QDRANT ──")

bench(
    "Dense vector search (top-5)",
    lambda: qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense", sample_dense),
        limit=5
    )
)

bench(
    "Sparse (BM25) search (top-5)",
    lambda: qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=("sparse", sample_sparse),
        limit=5
    )
)

from qdrant_client.models import Prefetch, FusionQuery, Fusion
bench(
    "Hybrid RRF search (top-5)",
    lambda: qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=sample_dense,  using="dense",  limit=20),
            Prefetch(query=sample_sparse, using="sparse", limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5,
    ).points
)

from qdrant_client.models import Filter, FieldCondition, MatchValue
bench(
    "Filter: legal_type='Luật' + dense (top-5)",
    lambda: qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense", sample_dense),
        query_filter=Filter(must=[FieldCondition(key="legal_type", match=MatchValue(value="Luật"))]),
        limit=5
    )
)

bench(
    "Filter: year='2015' + dense (top-5)",
    lambda: qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense", sample_dense),
        query_filter=Filter(must=[FieldCondition(key="year", match=MatchValue(value="2015"))]),
        limit=5
    )
)

bench(
    "Scroll filter: is_table=True (limit=10)",
    lambda: qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="is_table", match=MatchValue(value=True))]),
        limit=10
    )[0]
)

bench(
    "Count total points in collection",
    lambda: [qdrant.count(collection_name=COLLECTION_NAME)]
)

# ── NEO4J ───────────────────────────────────────────────────────────
print("\n── NEO4J ──")
driver = get_neo4j_driver()
if driver:
    with driver.session() as s:

        bench("Count total LegalDocument nodes",
              lambda: s.run("MATCH (d:LegalDocument) RETURN count(d) AS cnt").data())

        bench("Count total relationships",
              lambda: s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").data())

        bench("Count ghost nodes (is_ghost=true)",
              lambda: s.run("MATCH (d:LegalDocument {is_ghost:true}) RETURN count(d) AS cnt").data())

        bench("Find doc by number contains '80/2015'",
              lambda: s.run(
                  "MATCH (d:LegalDocument) WHERE d.document_number CONTAINS '80/2015' RETURN d LIMIT 1"
              ).data())

        bench("Get all outgoing relations of '80/2015'",
              lambda: s.run(
                  "MATCH (d:LegalDocument)-[r]->(t) "
                  "WHERE d.document_number CONTAINS '80/2015' "
                  "RETURN type(r), t.document_number LIMIT 20"
              ).data())

        bench("Find REPEALS edges (limit 10)",
              lambda: s.run(
                  "MATCH (a)-[r:REPEALS]->(b) RETURN a.document_number, b.document_number LIMIT 10"
              ).data())

        bench("Find AMENDS edges (limit 10)",
              lambda: s.run(
                  "MATCH (a)-[r:AMENDS]->(b) RETURN a.document_number, b.document_number LIMIT 10"
              ).data())

        bench("Find GUIDES edges (limit 10)",
              lambda: s.run(
                  "MATCH (a)-[r:GUIDES]->(b) RETURN a.document_number, b.document_number LIMIT 10"
              ).data())

        bench("Find APPLIES edges (limit 10)",
              lambda: s.run(
                  "MATCH (a)-[r:APPLIES]->(b) RETURN a.document_number, b.document_number LIMIT 10"
              ).data())

        bench("Shortest path (up to 3 hops) between any 2 docs",
              lambda: s.run(
                  "MATCH p=shortestPath((a:LegalDocument)-[*..3]-(b:LegalDocument)) "
                  "WHERE a <> b RETURN length(p) AS hops LIMIT 1"
              ).data())

        bench("Count edges grouped by relation type",
              lambda: s.run(
                  "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC"
              ).data())

        bench("Top 10 most connected docs (out-degree)",
              lambda: s.run(
                  "MATCH (d:LegalDocument)-[r]->() RETURN d.document_number, count(r) AS deg "
                  "ORDER BY deg DESC LIMIT 10"
              ).data())

    driver.close()
else:
    print("  [WARN] Không kết nối được Neo4j.")

# ── KẾT QUẢ TỔNG HỢP ─────────────────────────────────────────────
print("\n" + "=" * 60)
total_ok  = sum(1 for r in benchmark_results if r["status"] == "OK")
total_err = sum(1 for r in benchmark_results if r["status"] != "OK")
avg_ms    = sum(r["ms"] for r in benchmark_results) / len(benchmark_results) if benchmark_results else 0
print(f"Total scenarios : {len(benchmark_results)}")
print(f"OK              : {total_ok}")
print(f"ERROR           : {total_err}")
print(f"Avg latency     : {avg_ms:.1f} ms")

# Lưu ra file
bench_file = "notebook/benchmark_8k.txt"
try:
    with open(bench_file, "w", encoding="utf-8") as f:
        f.write(f"Benchmark run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Collection: {COLLECTION_NAME}\n\n")
        f.write(f"{'Scenario':<52} {'Time(ms)':>9}  {'Hits':>5}  Status\n")
        f.write("-" * 80 + "\n")
        for r in benchmark_results:
            f.write(f"{r['label']:<52} {r['ms']:>9.1f}  {r['hits']:>5}  {r['status']}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total: {len(benchmark_results)} | OK: {total_ok} | ERR: {total_err} | Avg: {avg_ms:.1f}ms\n")
    print(f"\n-> Đã lưu benchmark ra: {bench_file}")
except Exception as e:
    print(f"[WARN] Không thể lưu benchmark file: {e}")
