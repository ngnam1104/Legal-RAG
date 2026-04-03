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

# Cấu hình đường dẫn: 4 cấp lên tới Root (scripts/tests/research/eval/file.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


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
