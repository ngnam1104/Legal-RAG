import sys
import os
import time
import argparse
import asyncio
from typing import List, Dict

# Thiết lập đường dẫn import cho backend
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retrieval.chunker import AdvancedLegalChunker
from backend.retrieval.embedder import get_embedder
from backend.retrieval.hybrid_search import retriever
from backend.retrieval.vector_db import client as qdrant
from backend.config import settings
from backend.llm.factory import chat_completion
from backend.agent.utils_legal_qa import grade_documents

# Cấu hình đường dẫn dataset
MASTER_CSV_PATH = r"d:\iCOMM\Legal-RAG\Bộ test Hỏi đáp pháp luật về dữ liệu cá nhân.csv"

def load_diagnostic_queries(path: str, n: int = 3) -> List[tuple]:
    """Load n câu hỏi đầu tiên từ CSV làm test queries ẩn (chẩn đoán)."""
    try:
        import pandas as pd
        try:
            df = pd.read_csv(path, header=None, encoding='utf-8')
        except:
            df = pd.read_csv(path, header=None, encoding='latin-1')
        
        # Lấy cột thứ 2 (index 1) - nội dung câu hỏi
        queries = df[1].dropna().head(n).tolist()
        return [(q, {}) for q in queries]
    except Exception as e:
        print(f"⚠️ Cảnh báo: Không thể load {path}. Sử dụng queries mặc định. ({e})")
        return [("phòng cháy chữa cháy", {}), ("quyền lợi người lao động", {})]

TEST_QUERIES = load_diagnostic_queries(MASTER_CSV_PATH)

# Ttoggle để bật/tắt Rerank (Cross-Encoder) trong quá trình Benchmark
ENABLE_RERANK_BENCHMARK = False

def test_chunking():
    print("\n" + "="*50)
    print("--- [1] TESTING CHUNKING ---")
    chunker = AdvancedLegalChunker()
    
    # Mô phỏng một văn bản luật mẫu dài
    dummy_text = "Điều 1. Phạm vi điều chỉnh\nLuật này quy định về đối tượng, điều kiện, và quyền lợi.\n" * 100
    print(f"Kích thước văn bản thử nghiệm: {len(dummy_text)} chars")
    
    t0 = time.perf_counter()
    chunks = chunker.process_document(dummy_text, {"title": "Test Document"})
    t1 = time.perf_counter()
    
    duration = t1 - t0
    print(f"✅ Tạo ra {len(chunks)} chunks trong {duration:.4f}s")
    print(f"⚡ Tốc độ: {len(dummy_text)/duration:.0f} chars/s")

def test_embedding():
    print("\n" + "="*50)
    print("--- [2] TESTING EMBEDDING ---")
    embedder = get_embedder()
    dummy_texts = ["Đây là đoạn văn bản test hệ thống luật pháp." * 5] * 10
    
    t0 = time.perf_counter()
    dense = embedder.encode_dense(dummy_texts)
    sparse = embedder.encode_sparse_documents(dummy_texts)
    t1 = time.perf_counter()
    
    duration = t1 - t0
    print(f"✅ Encode {len(dummy_texts)} chunks xong trong {duration:.4f}s")
    print(f"⚡ Tốc độ: {len(dummy_texts)/duration:.2f} chunks/s")

def test_vectordb():
    print("\n" + "="*50)
    print("--- [3] TESTING VECTOR DB CONNECTION ---")
    col = settings.QDRANT_COLLECTION
    print(f"Kiểm tra kết nối Qdrant collection '{col}'...")
    
    t0 = time.perf_counter()
    try:
        info = qdrant.get_collection(col)
        t1 = time.perf_counter()
        print(f"✅ Kết nối thành công trong {t1 - t0:.4f}s")
        print(f"📊 Trạng thái: {info.status}")
        print(f"📊 Số lượng vector hiện có: {info.points_count}")
    except Exception as e:
        print(f"❌ Lỗi truy vấn VectorDB: {e}")

def test_retriever():
    print("\n" + "="*50)
    print("--- [4] TESTING RETRIEVER (BOTTLENECK) ---")
    print("Kiểm tra thời gian Broad Retrieve và Context Expand")
    
    for q, filters in TEST_QUERIES:
        print(f"\n🔎 Query: '{q}' | Filters: {filters}")
        try:
            t1 = time.perf_counter()
            broad_hits = retriever.broad_retrieve(q, top_k=40, **filters)
            t2 = time.perf_counter()
            
            # Đã loại bỏ Rerank theo yêu cầu
            expanded = retriever.expand_context(broad_hits[:10], max_neighbors=8)
            t3 = time.perf_counter()
            
            print(f"  ⏱ Broad Retrieve : {t2-t1:.4f}s (Thu về {len(broad_hits)} chunks)")
            print(f"  ⏱ Context Expand : {t3-t2:.4f}s (Mở rộng thành {len(expanded)} chunks)")
            print(f"  => Tổng Pipeline  : {t3-t1:.4f}s")
        except Exception as e:
            print(f"❌ Lỗi trong lúc truy xuất: {e}")

def test_generation():
    print("\n" + "="*50)
    print("--- [5] TESTING GENERATION (LLM MODEL) ---")
    messages = [{"role": "user", "content": "Hãy giải thích ngắn gọn 2 câu về lợi ích của RAG trong pháp luật."}]
    print("Gửi câu hỏi tới LLM sinh phản hồi...")
    
    t0 = time.perf_counter()
    try:
        resp = chat_completion(messages, temperature=0.1)
        t1 = time.perf_counter()
        duration = t1 - t0
        chars = len(resp)
        print(f"✅ Hoàn thành sinh {chars} ký tự trong {duration:.4f}s")
        print(f"⚡ Tốc độ: {chars/duration:.0f} chars/s")
        print(f"💬 Phản hồi: {resp[:150]}...")
    except Exception as e:
         print(f"❌ LLM Error: {e}")

def test_reflection():
    print("\n" + "="*50)
    print("--- [6] TESTING REFLECTION (GRADING) ---")
    print("Sử dụng LLM nhỏ để đánh giá xem văn bản có liên quan không.")
    
    dummy_query = "Trách nhiệm của người tham gia giao thông"
    dummy_context = "Luật giao thông quy định người đi bộ phải đi trên vỉa hè, lề đường."
    
    t0 = time.perf_counter()
    try:
        is_relevant = grade_documents(dummy_query, dummy_context)
        t1 = time.perf_counter()
        print(f"✅ Đánh giá xong trong {t1 - t0:.4f}s. Kết quả có liên quan: {is_relevant}")
    except Exception as e:
        print(f"❌ Reflection Error: {e}")



# Config params from eval/research_rag/config.py
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tests", "QA_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODE1_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_mode1_search.json")
MODE2_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_mode2_qa.json")
MODE3_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_mode3_conflict.json")
BENCHMARK_REPORT_JSON = os.path.join(OUTPUT_DIR, "benchmark_report.json")
BENCHMARK_REPORT_MD = os.path.join(OUTPUT_DIR, "benchmark_report.md")

"""
Core Metrics Module cho Legal RAG Benchmarking.
Bao gồm: IR Metrics, Generation Metrics (LLM-as-a-Judge), Classification Metrics, Latency.
"""
import math
import time
import json
import statistics
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

# Cấu hình đường dẫn: 5 cấp lên tới Root (scripts/tests/eval/research_rag/file.py)




# =====================================================================
# PHẦN 1: DATABASE PERFORMANCE (Latency & Throughput)
# =====================================================================

def measure_latency(query_fn: Callable, queries: List[str]) -> Dict[str, float]:
    """
    Đo độ trễ (latency) cho một hàm truy vấn.
    Trả về P50, P90, P99, avg, min, max tính bằng giây.
    """
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        query_fn(q)
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    n = len(latencies)
    if n == 0:
        return {"p50": 0, "p90": 0, "p99": 0, "avg": 0, "min": 0, "max": 0}

    return {
        "p50": latencies[int(n * 0.5)],
        "p90": latencies[int(n * 0.9)],
        "p99": latencies[min(int(n * 0.99), n - 1)],
        "avg": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
        "total_queries": n,
    }


def measure_throughput(query_fn: Callable, queries: List[str], concurrency: int = 5) -> Dict[str, float]:
    """
    Đo throughput (QPS) bằng cách chạy concurrent queries.
    """
    t0 = time.perf_counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(query_fn, q) for q in queries]
        for f in as_completed(futures):
            try:
                f.result()
                completed += 1
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    return {
        "total_queries": completed,
        "elapsed_seconds": round(elapsed, 3),
        "qps": round(completed / elapsed, 2) if elapsed > 0 else 0,
        "concurrency": concurrency,
    }


# =====================================================================
# PHẦN 2: RETRIEVAL METRICS (IR - Information Retrieval)
# =====================================================================

def recall_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 10) -> float:
    """
    Recall@K: Trong K tài liệu trả về, bao phủ bao nhiêu tài liệu đúng?
    recall = |retrieved ∩ expected| / |expected|
    """
    if not expected_ids:
        return 1.0  # Nếu không có expected, coi như đúng hết
    top_k = set(retrieved_ids[:k])
    relevant = set(expected_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 10) -> float:
    """
    Precision@K: Trong K tài liệu trả về, bao nhiêu cái là đúng?
    precision = |retrieved ∩ expected| / K
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(expected_ids)
    return len(top_k & relevant) / k


def mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """
    Mean Reciprocal Rank: Tài liệu đúng đầu tiên xuất hiện ở vị trí nào?
    MRR = 1 / rank_of_first_relevant
    """
    relevant = set(expected_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int = 10) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.
    Đánh giá chất lượng ranking (tài liệu quan trọng có nằm ở top không).
    Binary relevance: 1 nếu relevant, 0 nếu không.
    """
    relevant = set(expected_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 vì log2(1) = 0

    # Ideal DCG (tất cả relevant docs ở đầu)
    ideal_rels = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(retrieved_ids: List[str], expected_ids: List[str], k: int = 10) -> Dict[str, float]:
    """Tính toàn bộ IR metrics trong 1 lần gọi."""
    return {
        f"recall@{k}": round(recall_at_k(retrieved_ids, expected_ids, k), 4),
        f"precision@{k}": round(precision_at_k(retrieved_ids, expected_ids, k), 4),
        "mrr": round(mrr(retrieved_ids, expected_ids), 4),
        f"ndcg@{k}": round(ndcg_at_k(retrieved_ids, expected_ids, k), 4),
    }


# =====================================================================
# PHẦN 3: GENERATION METRICS (LLM-as-a-Judge) — Likert 1-4 + CoT
# =====================================================================

JUDGE_SYSTEM_PROMPT = """Bạn là giám khảo chuyên nghiệp đánh giá chất lượng hệ thống AI pháp lý Việt Nam.
BẮT BUỘC trả về JSON hợp lệ với đúng 2 trường sau:
{
    "reasoning": "(Chain-of-Thought) Phân tích chi tiết trước khi chấm điểm...",
    "score": <số nguyên từ 1 đến 4>
}
KHÔNG được trả về bất kỳ nội dung nào khác ngoài JSON trên."""


def _llm_judge(user_prompt: str) -> float:
    """
    Gọi LLM để chấm điểm theo thang Likert 1-4 với Chain-of-Thought.
    LLM phải trả về JSON {"reasoning": "...", "score": N}.
    Normalize: (score - 1) / 3.0 → float [0.0, 1.0].
    """
    try:
        from backend.llm.factory import chat_completion
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        response = chat_completion(messages, temperature=0.0)

        # Parse JSON từ response
        resp = response.strip()
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()

        data = json.loads(resp)
        score = int(data.get("score", 2))
        score = max(1, min(4, score))  # Clamp vào [1, 4]
        return (score - 1) / 3.0  # Normalize → [0.0, 1.0]
    except Exception as e:
        print(f"  [!] LLM Judge error: {e}")
        return 0.33  # Fallback = score 2/4


def faithfulness_score(answer: str, contexts: List[str]) -> float:
    """
    Faithfulness: Mọi nhận định trong câu trả lời có thể được xác minh từ Context không?
    Thang Likert 1-4 (ép chọn phe, không có mức trung lập).
    """
    context_text = "\n---\n".join(contexts)
    prompt = f"""Đánh giá mức độ TRUNG THÀNH (Faithfulness) của câu trả lời so với ngữ cảnh.
Faithfulness = Tỷ lệ các nhận định trong câu trả lời có thể được xác minh từ ngữ cảnh.

RUBRIC CHẤM ĐIỂM (BẮT BUỘC tuân theo):
- 1 điểm: Hoàn toàn ảo giác — Câu trả lời bịa đặt thông tin KHÔNG HỀ có trong ngữ cảnh.
- 2 điểm: Phần lớn ảo giác — Có trên 50% nội dung không tìm thấy trong ngữ cảnh.
- 3 điểm: Phần lớn trung thành — Trên 70% nội dung có căn cứ, chỉ sai sót nhỏ hoặc diễn giải hơi lệch.
- 4 điểm: Hoàn toàn trung thành — Mọi nhận định đều trích dẫn chính xác từ ngữ cảnh.

NGỮ CẢNH:
{context_text}

CÂU TRẢ LỜI CẦN ĐÁNH GIÁ:
{answer}

Trả về JSON: {{"reasoning": "...", "score": <1-4>}}"""
    return _llm_judge(prompt)


def answer_relevance_score(question: str, answer: str) -> float:
    """
    Answer Relevance: Câu trả lời có đi sát vấn đề được hỏi không?
    Thang Likert 1-4.
    """
    prompt = f"""Đánh giá mức độ LIÊN QUAN (Relevance) của câu trả lời so với câu hỏi.

RUBRIC CHẤM ĐIỂM (BẮT BUỘC tuân theo):
- 1 điểm: Hoàn toàn lạc đề — Câu trả lời không liên quan gì đến câu hỏi.
- 2 điểm: Liên quan mờ nhạt — Có nhắc đến chủ đề nhưng không giải quyết vấn đề cốt lõi.
- 3 điểm: Khá sát — Trả lời đúng hướng nhưng thiếu chi tiết hoặc hơi lan man.
- 4 điểm: Hoàn toàn sát — Trả lời trực tiếp, đầy đủ, súc tích đúng trọng tâm câu hỏi.

CÂU HỎI:
{question}

CÂU TRẢ LỜI:
{answer}

Trả về JSON: {{"reasoning": "...", "score": <1-4>}}"""
    return _llm_judge(prompt)


def answer_correctness_score(answer: str, ground_truth: str) -> float:
    """
    Answer Correctness: So sánh ngữ nghĩa giữa câu trả lời AI và đáp án mẫu.
    Thang Likert 1-4.
    """
    prompt = f"""Đánh giá mức độ CHÍNH XÁC (Correctness) của câu trả lời AI so với đáp án mẫu.

RUBRIC CHẤM ĐIỂM (BẮT BUỘC tuân theo):
- 1 điểm: Hoàn toàn sai — Kết luận trái ngược hoặc trích dẫn sai luật/điều khoản.
- 2 điểm: Sai nhiều hơn đúng — Có một phần đúng nhưng thiếu sót nghiêm trọng hoặc sai trọng tâm.
- 3 điểm: Đúng cơ bản — Kết luận đúng hướng, trích dẫn luật đúng nhưng thiếu chi tiết hoặc không đầy đủ.
- 4 điểm: Hoàn toàn chính xác — Cùng kết luận, cùng trích dẫn căn cứ, cùng mức độ chi tiết với đáp án mẫu.

ĐÁP ÁN MẪU (Ground Truth):
{ground_truth}

CÂU TRẢ LỜI AI CẦN ĐÁNH GIÁ:
{answer}

Trả về JSON: {{"reasoning": "...", "score": <1-4>}}"""
    return _llm_judge(prompt)


def compute_generation_metrics(question: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    """Tính toàn bộ Generation metrics đồng thời (concurrent) để giảm 3x latency."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_faith = executor.submit(faithfulness_score, answer, contexts)
        f_relev = executor.submit(answer_relevance_score, question, answer)
        f_corr = executor.submit(answer_correctness_score, answer, ground_truth)

    return {
        "faithfulness": round(f_faith.result(), 4),
        "answer_relevance": round(f_relev.result(), 4),
        "answer_correctness": round(f_corr.result(), 4),
    }


# =====================================================================
# PHẦN 4: CLASSIFICATION METRICS (NLI - Conflict Detection)
# =====================================================================

def confusion_matrix(predictions: List[str], ground_truths: List[str], labels: List[str] = None) -> Dict[str, Dict[str, int]]:
    """
    Tạo Confusion Matrix cho bài toán phân loại NLI.
    Returns: {actual_label: {predicted_label: count}}
    """
    if labels is None:
        labels = sorted(set(predictions + ground_truths))

    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}

    for pred, truth in zip(predictions, ground_truths):
        pred_norm = pred.lower().strip()
        truth_norm = truth.lower().strip()
        # Tìm label khớp nhất
        pred_label = next((l for l in labels if l.lower() == pred_norm), pred_norm)
        truth_label = next((l for l in labels if l.lower() == truth_norm), truth_norm)
        if truth_label in matrix and pred_label in matrix[truth_label]:
            matrix[truth_label][pred_label] += 1

    return matrix


def classification_report(predictions: List[str], ground_truths: List[str], labels: List[str] = None) -> Dict[str, Any]:
    """
    Tính Precision, Recall, F1-Score cho từng nhãn + macro average.
    """
    if labels is None:
        labels = sorted(set(predictions + ground_truths))

    cm = confusion_matrix(predictions, ground_truths, labels)
    report = {}

    for label in labels:
        tp = cm.get(label, {}).get(label, 0)
        fp = sum(cm.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(cm.get(label, {}).get(other, 0) for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": tp + fn,
        }

    # Macro average
    avg_precision = statistics.mean(r["precision"] for r in report.values()) if report else 0
    avg_recall = statistics.mean(r["recall"] for r in report.values()) if report else 0
    avg_f1 = statistics.mean(r["f1_score"] for r in report.values()) if report else 0

    report["macro_avg"] = {
        "precision": round(avg_precision, 4),
        "recall": round(avg_recall, 4),
        "f1_score": round(avg_f1, 4),
    }

    return {
        "per_label": report,
        "confusion_matrix": cm,
    }

"""
Benchmark Runner: Chạy đánh giá toàn diện cho cả 3 Mode của Legal RAG.
Mode 1: Search (IR Metrics) | Mode 2: QA (Generation Metrics) | Mode 3: Conflict (Classification Metrics)
"""
import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# Cấu hình đường dẫn: 5 cấp lên tới Root (scripts/tests/eval/research_rag/file.py)










def _init_retriever():
    """Lazy-load retriever để tránh import nặng khi chưa cần."""
    from backend.retrieval.hybrid_search import retriever
    return retriever


def _init_engine():
    """Lazy-load RAG engine."""
    from backend.agent.chat_engine import rag_engine
    return rag_engine


# =====================================================================
# MODE 1: SEARCH BENCHMARK (IR Metrics) — Chi tiết theo Pipeline Step
# Pipeline: Understand (HyDE) → Retrieve (Encode → Prefetch+RRF → Rerank → Expand) → Resolve References
# =====================================================================
def benchmark_mode1_search(dataset: List[Dict], limit: int = None) -> Dict[str, Any]:
    """
    Đánh giá khả năng tìm kiếm tài liệu (Recall, Precision, MRR, NDCG).
    Log chi tiết thời gian từng bước theo kiến trúc LangGraph Pipeline.
    """
    print("\n" + "="*60)
    print("📊 MODE 1: BENCHMARK TÌM KIẾM TÀI LIỆU (IR Metrics)")
    print("="*60)

    from backend.retrieval.hybrid_search import retriever
    from backend.agent.utils_legal_qa import rewrite_legal_query, resolve_recursive_references

    samples = dataset[:limit] if limit else dataset
    results = []

    for i, case in enumerate(tqdm(samples, desc="Mode 1 - Search")):
        query = case.get("query", "")
        expected_ids = case.get("expected_chunk_ids", [])
        if not query:
            continue

        step_times = {}
        t_total_start = time.perf_counter()

        # ── Step 1: Understand (HyDE Rewrite + Filter Extraction) ──
        t0 = time.perf_counter()
        rewrite_data = rewrite_legal_query(query)
        hypothetical = rewrite_data.get("hypothetical_answer") or query
        filters = rewrite_data.get("filters", {})
        step_times["1_understand_hyde_s"] = round(time.perf_counter() - t0, 4)

        # ── Step 2: Retrieve (Broad Retrieve) ──
        t0 = time.perf_counter()
        broad_hits = retriever.broad_retrieve(
            query=hypothetical, top_k=25, main_limit=20, appendix_limit=5,
            legal_type=filters.get("legal_type"),
            doc_number=filters.get("doc_number"),
        )
        step_times["2b_broad_retrieve_s"] = round(time.perf_counter() - t0, 4)

        # ── Step 2c: Rerank (Cross-Encoder vs RRF Sort) ──
        t0 = time.perf_counter()
        if ENABLE_RERANK_BENCHMARK:
            # Sử dụng Cross-Encoder để sắp xếp lại 25 kết quả đầu tiên
            reranked_hits = retriever.reranker.rerank(query=hypothetical, candidates=broad_hits, top_k=10)
        else:
            # Chỉ sắp xếp lại dựa trên RRF score (mặc định)
            reranked_hits = sorted(broad_hits, key=lambda x: x.get('rrf_score', 0), reverse=True)[:10]
            for item in reranked_hits:
                item['score'] = item.get('rrf_score', 0)
                item['rerank_score'] = item.get('rrf_score', 0)
        step_times["2c_rerank_s"] = round(time.perf_counter() - t0, 4)

        # 2d. Context Expansion (Small-to-Big / Window Retrieval)
        t0 = time.perf_counter()
        expanded_hits = retriever.expand_context(refined_hits=reranked_hits, max_neighbors=5)
        step_times["2d_context_expand_s"] = round(time.perf_counter() - t0, 4)

        # ── Step 3: Resolve References (Recursive Điều/Khoản/Phụ lục) ──
        t0 = time.perf_counter()
        # Format hits for resolve_recursive_references
        formatted_hits = []
        for item in expanded_hits:
            p = item.get("payload", {})
            formatted_hits.append({
                "id": item["id"], "score": item.get("rerank_score", 0),
                "document_number": p.get("document_number", ""),
                "article_ref": p.get("article_ref", ""),
                "title": p.get("title", ""),
                "text": p.get("expanded_context_text", p.get("chunk_text", "")),
                "chunk_id": p.get("chunk_id", ""),
                "is_appendix": p.get("is_appendix", False),
            })
        all_hits_with_refs = resolve_recursive_references(formatted_hits)
        step_times["3_resolve_refs_s"] = round(time.perf_counter() - t0, 4)

        step_times["total_pipeline_s"] = round(time.perf_counter() - t_total_start, 4)

        # ── Compute IR Metrics ──
        retrieved_ids = [h.get("chunk_id", "") for h in formatted_hits]
        metrics = compute_retrieval_metrics(retrieved_ids, expected_ids, k=10)
        metrics.update(step_times)
        metrics["query"] = query[:80]
        metrics["broad_count"] = len(broad_hits)
        metrics["rerank_count"] = len(reranked_hits)
        metrics["expand_count"] = len(expanded_hits)
        metrics["ref_count"] = len(all_hits_with_refs) - len(formatted_hits)
        results.append(metrics)

        # Log per-case
        print(f"\n  🔎 Q{i+1}: \"{query[:60]}...\"")
        print(f"     Step 1 Understand (HyDE)    : {step_times['1_understand_hyde_s']:.3f}s")
        print(f"     Step 2b Broad Retrieve (RRF) : {step_times['2b_broad_retrieve_s']:.3f}s → {len(broad_hits)} candidates")
        print(f"     Step 2c Rerank               : {step_times['2c_rerank_s']:.3f}s → {len(reranked_hits)} refined")
        print(f"     Step 2d Context Expand       : {step_times['2d_context_expand_s']:.3f}s → {len(expanded_hits)} expanded")
        print(f"     Step 3 Resolve References    : {step_times['3_resolve_refs_s']:.3f}s → +{len(all_hits_with_refs) - len(formatted_hits)} refs")
        print(f"     ═══ Total Pipeline           : {step_times['total_pipeline_s']:.3f}s")

        time.sleep(0.1)

    # Aggregate
    if not results:
        return {"error": "No valid test cases"}

    # Aggregate step times
    step_keys = ["1_understand_hyde_s", "2b_broad_retrieve_s",
                 "2c_rerank_s", "2d_context_expand_s", "3_resolve_refs_s", "total_pipeline_s"]
    ir_keys = ["recall@10", "precision@10", "mrr", "ndcg@10"]

    avg_metrics = {}
    for key in step_keys + ir_keys:
        values = [r[key] for r in results if key in r]
        avg_metrics[f"avg_{key}"] = round(sum(values) / len(values), 4) if values else 0

    print(f"\n{'='*60}")
    print(f"✅ Mode 1 Aggregated Results ({len(results)} queries):")
    print(f"{'─'*60}")
    print(f"  📏 IR Metrics:")
    for k in ir_keys:
        print(f"     avg_{k}: {avg_metrics[f'avg_{k}']}")
    print(f"  ⏱️ Pipeline Step Latencies (avg):")
    for k in step_keys:
        print(f"     avg_{k}: {avg_metrics[f'avg_{k}']:.4f}s")
    print(f"{'='*60}")

    return {
        "mode": "search",
        "total_queries": len(results),
        "aggregate": avg_metrics,
        "per_query": results,
    }


# =====================================================================
# MODE 2: QA BENCHMARK (Generation Metrics) — Chi tiết theo Pipeline Step
# Pipeline: Understand → Retrieve → Resolve Refs → Grade → Generate → Reflect
# =====================================================================
def benchmark_mode2_qa(dataset: List[Dict], limit: int = None) -> Dict[str, Any]:
    """
    Đánh giá chất lượng câu trả lời, log chi tiết từng bước LangGraph pipeline.
    Gọi trực tiếp Strategy methods thay vì black-box engine.chat().
    """
    print("\n" + "="*60)
    print("📊 MODE 2: BENCHMARK HỎI ĐÁP PHÁP LÝ (Generation Metrics)")
    print("="*60)

    from backend.agent.strategies.legal_qa import LegalQAStrategy
    from backend.retrieval.hybrid_search import retriever

    strategy = LegalQAStrategy()
    samples = dataset[:limit] if limit else dataset
    results = []

    for i, case in enumerate(tqdm(samples, desc="Mode 2 - QA")):
        question = case.get("question", "")
        ground_truth = case.get("ground_truth_answer", "")
        if not question:
            continue

        step_times = {}
        t_total_start = time.perf_counter()

        try:
            # Build state giống LangGraph
            state = {
                "query": question, "condensed_query": question,
                "session_id": "benchmark_qa", "mode": "LEGAL_QA",
                "use_reflection": True, "use_rerank": False,
                "llm_preset": None, "history": [],
                "file_chunks": [], "retry_count": 0,
                "raw_hits": [], "recursive_hits": [],
                "router_filters": {},
            }

            # ── Step 1: Understand (HyDE + Filter) ──
            t0 = time.perf_counter()
            understand_result = strategy.understand(state)
            step_times["1_understand_hyde_s"] = round(time.perf_counter() - t0, 4)
            state.update(understand_result)

            # ── Step 2a: Broad Retrieve ──
            rewritten_queries = state.get("rewritten_queries") or [state["query"]]
            kw = rewritten_queries[0]
            filters = state.get("metadata_filters", {})
            
            # Sanitize filters (LLM might return strings for booleans)
            is_appendix_filter = filters.get("is_appendix")
            if isinstance(is_appendix_filter, str):
                is_appendix_filter = "true" in is_appendix_filter.lower()

            t0 = time.perf_counter()
            broad_hits = retriever.broad_retrieve(
                query=kw, top_k=25, main_limit=20, appendix_limit=5,
                legal_type=filters.get("legal_type"),
                doc_number=filters.get("doc_number"),
                is_appendix=is_appendix_filter,
                article_ref=filters.get("article_ref")
            )
            step_times["2a_broad_retrieve_s"] = round(time.perf_counter() - t0, 4)

            # ── Step 2b: Rerank (Cross-Encoder vs Sort) ──
            t0 = time.perf_counter()
            if ENABLE_RERANK_BENCHMARK:
                reranked_hits = retriever.reranker.rerank(query=kw, candidates=broad_hits, top_k=15)
            else:
                reranked_hits = sorted(broad_hits, key=lambda x: x.get('rrf_score', 0), reverse=True)[:15]
                for h in reranked_hits:
                    h['score'] = h.get('rrf_score', 0)
            step_times["2b_rerank_s"] = round(time.perf_counter() - t0, 4)
            
            # Map back to state
            state["raw_hits"] = reranked_hits
            num_raw_hits = len(reranked_hits)

            # ── Step 3: Resolve References ──
            t0 = time.perf_counter()
            refs_result = strategy.resolve_references(state)
            step_times["3_resolve_refs_s"] = round(time.perf_counter() - t0, 4)
            state.update(refs_result)
            num_recursive = len(state.get("recursive_hits", []))

            # ── Step 4: Grade (CRAG) ──
            t0 = time.perf_counter()
            grade_result = strategy.grade(state)
            step_times["4_grade_crag_s"] = round(time.perf_counter() - t0, 4)
            state.update(grade_result)
            is_sufficient = state.get("is_sufficient", False)

            # ── Step 5: Generate (LLM Answer) ──
            t0 = time.perf_counter()
            gen_result = strategy.generate(state)
            step_times["5_generate_s"] = round(time.perf_counter() - t0, 4)
            state.update(gen_result)
            answer = state.get("draft_response", "")

            # ── Step 6: Reflect (Self-RAG Anti-Hallucination) ──
            t0 = time.perf_counter()
            reflect_result = strategy.reflect(state)
            step_times["6_reflect_s"] = round(time.perf_counter() - t0, 4)
            state.update(reflect_result)
            pass_flag = state.get("pass_flag", True)
            final_answer = state.get("final_response", answer)

            step_times["total_pipeline_s"] = round(time.perf_counter() - t_total_start, 4)

            # ── LLM-as-a-Judge Scoring ──
            contexts = [h.get("text", "") for h in state.get("raw_hits", []) if h.get("text")]
            gen_metrics = compute_generation_metrics(question, final_answer, contexts, ground_truth)
            gen_metrics.update(step_times)
            gen_metrics["question"] = question[:80]
            gen_metrics["is_sufficient"] = is_sufficient
            gen_metrics["pass_flag"] = pass_flag
            gen_metrics["num_raw_hits"] = num_raw_hits
            gen_metrics["num_recursive_refs"] = num_recursive
            results.append(gen_metrics)

            # Log per-case
            print(f"\n  💬 Q{i+1}: \"{question[:60]}...\"")
            print(f"     Step 1 Understand (HyDE)    : {step_times['1_understand_hyde_s']:.3f}s")
            print(f"     Step 2a Retrieve (Broad)     : {step_times['2a_broad_retrieve_s']:.3f}s → {len(broad_hits)} hits")
            print(f"     Step 2b Rerank               : {step_times['2b_rerank_s']:.3f}s")
            print(f"     Step 3 Resolve References    : {step_times['3_resolve_refs_s']:.3f}s → +{num_recursive} refs")
            print(f"     Step 4 Grade (CRAG)          : {step_times['4_grade_crag_s']:.3f}s → sufficient={is_sufficient}")
            print(f"     Step 5 Generate (LLM 70B)    : {step_times['5_generate_s']:.3f}s")
            print(f"     Step 6 Reflect (Self-RAG)    : {step_times['6_reflect_s']:.3f}s → pass={pass_flag}")
            print(f"     ═══ Total Pipeline           : {step_times['total_pipeline_s']:.3f}s")
            print(f"     📊 Faithfulness={gen_metrics.get('faithfulness','?')} | Relevance={gen_metrics.get('answer_relevance','?')} | Correctness={gen_metrics.get('answer_correctness','?')}")

        except Exception as e:
            print(f"  [!] Error on question {i}: {e}")
            import traceback; traceback.print_exc()
            results.append({"question": question[:80], "error": str(e)})

        time.sleep(5)  # Rate limit Groq

    # Aggregate
    if not results:
        return {"error": "No valid test cases"}

    step_keys = ["1_understand_hyde_s", "2a_broad_retrieve_s", "2b_rerank_s", "3_resolve_refs_s",
                 "4_grade_crag_s", "5_generate_s", "6_reflect_s", "total_pipeline_s"]
    gen_keys = ["faithfulness", "answer_relevance", "answer_correctness"]

    avg_metrics = {}
    for key in step_keys + gen_keys:
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        avg_metrics[f"avg_{key}"] = round(sum(values) / len(values), 4) if values else 0

    print(f"\n{'='*60}")
    print(f"✅ Mode 2 Aggregated Results ({len(results)} questions):")
    print(f"{'─'*60}")
    print(f"  📏 Generation Metrics (LLM-as-Judge, scale 0.0–1.0):")
    for k in gen_keys:
        print(f"     avg_{k}: {avg_metrics[f'avg_{k}']}")
    print(f"  ⏱️ Pipeline Step Latencies (avg):")
    for k in step_keys:
        print(f"     avg_{k}: {avg_metrics[f'avg_{k}']:.4f}s")
    print(f"{'='*60}")

    return {
        "mode": "qa",
        "total_questions": len(results),
        "aggregate": avg_metrics,
        "per_question": results,
    }


# =====================================================================
# MODE 3: CONFLICT BENCHMARK (Classification) — Chi tiết theo Pipeline Step
# Pipeline: Understand → Retrieve → Grade (Prune) → Generate (Judge) → Reflect (Review)
# =====================================================================
def benchmark_mode3_conflict(dataset: List[Dict], limit: int = None) -> Dict[str, Any]:
    """
    Đánh giá phát hiện xung đột, log chi tiết từng bước LangGraph pipeline.
    """
    print("\n" + "="*60)
    print("📊 MODE 3: BENCHMARK PHÁT HIỆN XUNG ĐỘT (NLI Classification)")
    print("="*60)

    from backend.retrieval.hybrid_search import retriever
    from backend.agent.utils_legal_qa import rewrite_legal_query, build_legal_context

    samples = dataset[:limit] if limit else dataset
    predictions = []
    ground_truths = []
    details = []

    for i, case in enumerate(tqdm(samples, desc="Mode 3 - Conflict")):
        claim = case.get("claim", "")
        expected_label = case.get("label", "").lower().strip()
        if not claim or not expected_label:
            continue

        step_times = {}
        t_total_start = time.perf_counter()

        try:
            # ── Step 1: Understand (Rewrite claim for search) ──
            t0 = time.perf_counter()
            rewrite_data = rewrite_legal_query(claim)
            hypothetical = rewrite_data.get("hypothetical_answer") or claim
            step_times["1_understand_s"] = round(time.perf_counter() - t0, 4)

            # ── Step 2a: Broad Retrieve ──
            t0 = time.perf_counter()
            broad_hits = retriever.broad_retrieve(query=hypothetical, top_k=15)
            step_times["2a_broad_retrieve_s"] = round(time.perf_counter() - t0, 4)

            # ── Step 2b: Rerank (Cross-Encoder vs Sort) ──
            t0 = time.perf_counter()
            if ENABLE_RERANK_BENCHMARK:
                reranked_hits = retriever.reranker.rerank(query=hypothetical, candidates=broad_hits, top_k=3)
            else:
                reranked_hits = sorted(broad_hits, key=lambda x: x.get('rrf_score', 0), reverse=True)[:3]
            step_times["2b_rerank_s"] = round(time.perf_counter() - t0, 4)
            
            hits = reranked_hits
            num_hits = len(hits)

            # ── Step 3: Grade / Build Context ──
            t0 = time.perf_counter()
            contexts = [h.get("text", "") for h in hits if h.get("text")]
            context_text = "\n---\n".join(contexts)
            step_times["3_grade_context_s"] = round(time.perf_counter() - t0, 4)

            # ── Step 4: Generate (LLM NLI Classification) ──
            t0 = time.perf_counter()
            from backend.llm.factory import chat_completion
            judge_prompt = f"""Bạn là hệ thống phát hiện xung đột pháp lý (NLI Classifier).
Đánh giá mệnh đề sau so với căn cứ pháp luật và trả về DUY NHẤT một nhãn:
- "contradiction" nếu mệnh đề trái ngược/vi phạm pháp luật
- "entailment" nếu mệnh đề hoàn toàn hợp pháp
- "neutral" nếu pháp luật không quy định rõ

MỆNH ĐỀ: {claim}

CĂN CỨ PHÁP LUẬT:
{context_text}

Chỉ trả về một từ duy nhất (contradiction/entailment/neutral):"""

            messages = [{"role": "user", "content": judge_prompt}]
            pred_raw = chat_completion(messages, temperature=0.0).strip().lower()
            step_times["4_generate_judge_s"] = round(time.perf_counter() - t0, 4)

            # Normalize prediction
            if "contradiction" in pred_raw:
                pred_label = "contradiction"
            elif "entailment" in pred_raw:
                pred_label = "entailment"
            elif "complex" in pred_raw:
                pred_label = "complex_reasoning"
            else:
                pred_label = "neutral"

            # Normalize ground truth
            if "complex" in expected_label:
                expected_label = "complex_reasoning"

            step_times["total_pipeline_s"] = round(time.perf_counter() - t_total_start, 4)

            predictions.append(pred_label)
            ground_truths.append(expected_label)
            is_correct = pred_label == expected_label
            details.append({
                "claim": claim[:100],
                "expected": expected_label,
                "predicted": pred_label,
                "correct": is_correct,
                "step_times": step_times,
            })

            # Log per-case
            icon = "✅" if is_correct else "❌"
            print(f"\n  {icon} C{i+1}: \"{claim[:60]}...\"")
            print(f"     Step 1 Understand (Rewrite)  : {step_times['1_understand_s']:.3f}s")
            print(f"     Step 2a Retrieve (Broad)     : {step_times['2a_broad_retrieve_s']:.3f}s → {len(broad_hits)} hits")
            print(f"     Step 2b Rerank               : {step_times['2b_rerank_s']:.3f}s")
            print(f"     Step 3 Grade/Context          : {step_times['3_grade_context_s']:.3f}s")
            print(f"     Step 4 Generate (NLI Judge)  : {step_times['4_generate_judge_s']:.3f}s")
            print(f"     ═══ Total Pipeline           : {step_times['total_pipeline_s']:.3f}s")
            print(f"     📊 Expected={expected_label} | Predicted={pred_label} → {icon}")

        except Exception as e:
            print(f"  [!] Error on claim {i}: {e}")

        time.sleep(3)  # Rate limit

    # Classification report
    if not predictions:
        return {"error": "No valid test cases"}

    labels = ["contradiction", "entailment", "neutral", "complex_reasoning"]
    report = classification_report(predictions, ground_truths, labels)

    accuracy = sum(1 for d in details if d["correct"]) / len(details) if details else 0

    # Aggregate step times
    step_keys = ["1_understand_s", "2a_broad_retrieve_s", "2b_rerank_s", "3_grade_context_s", "4_generate_judge_s", "total_pipeline_s"]
    avg_step_times = {}
    for k in step_keys:
        values = [d["step_times"][k] for d in details if "step_times" in d and k in d["step_times"]]
        avg_step_times[f"avg_{k}"] = round(sum(values) / len(values), 4) if values else 0

    print(f"\n{'='*60}")
    print(f"✅ Mode 3 Aggregated Results ({len(details)} claims):")
    print(f"{'─'*60}")
    print(f"  📏 Classification Metrics:")
    print(f"     Accuracy: {accuracy:.4f}")
    for label, metrics in report["per_label"].items():
        if label != "macro_avg":
            print(f"     {label}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1_score']:.3f}")
    print(f"  ⏱️ Pipeline Step Latencies (avg):")
    for k in step_keys:
        print(f"     avg_{k}: {avg_step_times[f'avg_{k}']:.4f}s")
    print(f"{'='*60}")

    return {
        "mode": "conflict",
        "total_claims": len(details),
        "accuracy": round(accuracy, 4),
        "classification_report": report["per_label"],
        "confusion_matrix": report["confusion_matrix"],
        "avg_step_times": avg_step_times,
        "per_claim": details,
    }


# =====================================================================
# REPORT GENERATOR
# =====================================================================
def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Tạo báo cáo Markdown đẹp từ kết quả benchmark."""
    lines = [
        "# 📊 Legal RAG - Benchmark Report",
        f"\n*Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "---\n",
    ]

    # Mode 1
    if "mode1" in results:
        m1 = results["mode1"]
        agg = m1.get("aggregate", {})
        lines.append("## 🔍 Mode 1: Tìm Kiếm Tài Liệu (Search/IR)")
        lines.append(f"**Tổng số queries:** {m1.get('total_queries', 0)}\n")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for k, v in agg.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Mode 2
    if "mode2" in results:
        m2 = results["mode2"]
        agg = m2.get("aggregate", {})
        lines.append("## 💬 Mode 2: Hỏi Đáp Pháp Lý (QA)")
        lines.append(f"**Tổng số câu hỏi:** {m2.get('total_questions', 0)}\n")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for k, v in agg.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    # Mode 3
    if "mode3" in results:
        m3 = results["mode3"]
        lines.append("## ⚖️ Mode 3: Phát Hiện Xung Đột (Conflict Detection)")
        lines.append(f"**Tổng số claims:** {m3.get('total_claims', 0)}")
        lines.append(f"**Accuracy:** {m3.get('accuracy', 0)}\n")
        report = m3.get("classification_report", {})
        lines.append("| Label | Precision | Recall | F1-Score | Support |")
        lines.append("|-------|-----------|--------|----------|---------|")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                lines.append(f"| {label} | {metrics.get('precision', '-')} | {metrics.get('recall', '-')} | {metrics.get('f1_score', '-')} | {metrics.get('support', '-')} |")

        # Confusion Matrix
        cm = m3.get("confusion_matrix", {})
        if cm:
            lines.append("\n### Confusion Matrix")
            labels = list(cm.keys())
            header = "| Actual \\ Predicted | " + " | ".join(labels) + " |"
            sep = "|" + "---|" * (len(labels) + 1)
            lines.append(header)
            lines.append(sep)
            for actual in labels:
                row = f"| **{actual}** | "
                row += " | ".join(str(cm[actual].get(pred, 0)) for pred in labels)
                row += " |"
                lines.append(row)

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Legal RAG Benchmark Runner")
    parser.add_argument("--mode", type=str, choices=["1", "2", "3", "all"], default="all",
                        help="Mode nào cần benchmark: 1 (Search), 2 (QA), 3 (Conflict), all (tất cả)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Giới hạn số test case cho mỗi mode (mặc định 5)")
    args = parser.parse_args()

    print("="*60)
    print("🏋️ Legal RAG - BENCHMARK RUNNER")
    print("="*60)

    all_results = {}

    # Mode 1
    if args.mode in ["1", "all"]:
        if os.path.exists(MODE1_OUTPUT_PATH):
            with open(MODE1_OUTPUT_PATH, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            all_results["mode1"] = benchmark_mode1_search(dataset, limit=args.limit)
        else:
            print(f"⚠️ Bỏ qua Mode 1: Không tìm thấy {MODE1_OUTPUT_PATH}")

    # Mode 2
    if args.mode in ["2", "all"]:
        if os.path.exists(MODE2_OUTPUT_PATH):
            with open(MODE2_OUTPUT_PATH, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            all_results["mode2"] = benchmark_mode2_qa(dataset, limit=args.limit)
        else:
            print(f"⚠️ Bỏ qua Mode 2: Không tìm thấy {MODE2_OUTPUT_PATH}")

    # Mode 3
    if args.mode in ["3", "all"]:
        if os.path.exists(MODE3_OUTPUT_PATH):
            with open(MODE3_OUTPUT_PATH, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            all_results["mode3"] = benchmark_mode3_conflict(dataset, limit=args.limit)
        else:
            print(f"⚠️ Bỏ qua Mode 3: Không tìm thấy {MODE3_OUTPUT_PATH}")

    # Save JSON report
    with open(BENCHMARK_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\n📁 JSON Report: {BENCHMARK_REPORT_JSON}")

    # Save Markdown report
    md_report = generate_markdown_report(all_results)
    with open(BENCHMARK_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"📁 Markdown Report: {BENCHMARK_REPORT_MD}")

    print("\n🎉 Benchmark hoàn tất!")



def main():

    parser = argparse.ArgumentParser(description="Legal-RAG Unified Performance Evaluator")
    parser.add_argument(
        "--mode", 
        default="all", 
        choices=["chunking", "embedding", "vectordb", "retriever", "generation", "reflection", "all", "evaluate_search", "evaluate_qa", "evaluate_conflict", "evaluate_all"], 
        help="Chọn module cần test hiệu năng (Mặc định: tất cả)"
    )
    args = parser.parse_args()
    
    print("="*60)
    print(f"🚀 BẮT ĐẦU CHẠY ĐÁNH GIÁ (Mode: {args.mode.upper()})")
    print("="*60)
    
    try:

        if args.mode in ["chunking", "all"]: test_chunking()
        if args.mode in ["embedding", "all"]: test_embedding()
        if args.mode in ["vectordb", "all"]: test_vectordb()
        if args.mode in ["retriever", "all"]: test_retriever()
        if args.mode in ["generation", "all"]: test_generation()
        if args.mode in ["reflection", "all"]: test_reflection()
        
        # New Benchmark Modes
        if args.mode in ["evaluate_search", "evaluate_qa", "evaluate_conflict", "evaluate_all"]:
            import pandas as pd
            csv_path = r"d:\iCOMM\Legal-RAG\Bộ test Hỏi đáp pháp luật về dữ liệu cá nhân.csv"
            print(f"\n[+] Đọc dataset từ CSV: {csv_path}")
            try:
                df = pd.read_csv(csv_path, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, header=None, encoding='latin-1')

            # Ánh xạ theo tên Mode trong CSV
            df_search = df[df[0].str.contains("tìm kiếm", na=False, case=False)].head(3)
            df_qa = df[df[0].str.contains("Hỏi đáp", na=False, case=False)].head(3)
            df_conflict = df[df[0].str.contains("xung đột", na=False, case=False)].head(3)

            print(f"[*] Đã chọn: {len(df_search)} Search, {len(df_qa)} QA, {len(df_conflict)} Conflict cases.")

            # MODE 1: Search (Dẫn dắt tìm kiếm)
            if args.mode in ["evaluate_search", "evaluate_all"]:
                dataset_m1 = []
                for _, row in df_search.iterrows():
                    dataset_m1.append({
                        "query": str(row[1]),
                        "expected_chunk_ids": [str(row[3])]
                    })
                benchmark_mode1_search(dataset_m1, limit=3)

            # MODE 2: QA (Hỏi đáp trực tiếp)
            if args.mode in ["evaluate_qa", "evaluate_all"]:
                dataset_m2 = []
                for _, row in df_qa.iterrows():
                    dataset_m2.append({
                        "question": str(row[1]),
                        "ground_truth_answer": str(row[2]),
                        "expected_chunk_ids": [str(row[3])]
                    })
                benchmark_mode2_qa(dataset_m2, limit=3)

            # MODE 3: Conflict (Phát hiện xung đột)
            if args.mode in ["evaluate_conflict", "evaluate_all"]:
                dataset_m3 = []
                for _, row in df_conflict.iterrows():
                    dataset_m3.append({
                        "claim": str(row[1]),
                        "label": "neutral", # Cột 2 trong CSV là giải thích, nên gán tạm nhãn neutral
                        "expected_chunk_ids": [str(row[3])]
                    })
                benchmark_mode3_conflict(dataset_m3, limit=3)

    except KeyboardInterrupt:
        print("\n⚠️ Người dùng dừng đột ngột.")
        
    print("\n" + "="*60)
    print("🏁 KẾT THÚC ĐÁNH GIÁ 🏁")
    print("="*60)

if __name__ == "__main__":
    main()
