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

# Cấu hình đường dẫn: 4 cấp lên tới Root (scripts/tests/research/eval/file.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESEARCH_DIR = os.path.join(ROOT_DIR, "scripts", "tests", "research")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from config import (
    MODE1_OUTPUT_PATH, MODE2_OUTPUT_PATH, MODE3_OUTPUT_PATH,
    BENCHMARK_REPORT_JSON, BENCHMARK_REPORT_MD
)
from eval.metrics import (
    compute_retrieval_metrics, compute_generation_metrics,
    classification_report, measure_latency, measure_throughput
)


def _init_retriever():
    """Lazy-load retriever để tránh import nặng khi chưa cần."""
    from backend.retrieval.hybrid_search import retriever
    return retriever


def _init_engine():
    """Lazy-load RAG engine."""
    from backend.agent.chat_engine import rag_engine
    return rag_engine


# =====================================================================
# MODE 1: SEARCH BENCHMARK (IR Metrics)
# =====================================================================
def benchmark_mode1_search(dataset: List[Dict], limit: int = None) -> Dict[str, Any]:
    """
    Đánh giá khả năng tìm kiếm tài liệu (Recall, Precision, MRR, NDCG).
    Dataset format: [{"query": "...", "expected_chunk_ids": ["id1","id2"]}]
    """
    print("\n" + "="*60)
    print("📊 MODE 1: BENCHMARK TÌM KIẾM TÀI LIỆU (IR Metrics)")
    print("="*60)

    retriever = _init_retriever()
    samples = dataset[:limit] if limit else dataset
    results = []
    all_latencies = []

    for i, case in enumerate(tqdm(samples, desc="Mode 1 - Search")):
        query = case.get("query", "")
        expected_ids = case.get("expected_chunk_ids", [])
        if not query:
            continue

        t0 = time.perf_counter()
        hits = retriever.search(query=query, limit=10, expand_context=False)
        latency = time.perf_counter() - t0
        all_latencies.append(latency)

        retrieved_ids = [h.get("chunk_id", "") for h in hits]
        metrics = compute_retrieval_metrics(retrieved_ids, expected_ids, k=10)
        metrics["latency_s"] = round(latency, 4)
        metrics["query"] = query[:80]
        results.append(metrics)

        time.sleep(0.1)  # Tránh quá tải Qdrant

    # Aggregate
    if not results:
        return {"error": "No valid test cases"}

    avg_metrics = {}
    for key in ["recall@10", "precision@10", "mrr", "ndcg@10", "latency_s"]:
        values = [r[key] for r in results if key in r]
        avg_metrics[f"avg_{key}"] = round(sum(values) / len(values), 4) if values else 0

    all_latencies.sort()
    n = len(all_latencies)
    avg_metrics["p90_latency_s"] = round(all_latencies[int(n * 0.9)] if n > 0 else 0, 4)
    avg_metrics["p99_latency_s"] = round(all_latencies[min(int(n * 0.99), n - 1)] if n > 0 else 0, 4)

    print(f"\n✅ Mode 1 Results ({len(results)} queries):")
    for k, v in avg_metrics.items():
        print(f"   {k}: {v}")

    return {
        "mode": "search",
        "total_queries": len(results),
        "aggregate": avg_metrics,
        "per_query": results,
    }


# =====================================================================
# MODE 2: QA BENCHMARK (Generation Metrics - LLM-as-a-Judge)
# =====================================================================
def benchmark_mode2_qa(dataset: List[Dict], limit: int = None) -> Dict[str, Any]:
    """
    Đánh giá chất lượng câu trả lời (Faithfulness, Relevance, Correctness).
    Dataset format: [{"question": "...", "ground_truth_answer": "...", "expected_chunk_ids": [...]}]
    """
    print("\n" + "="*60)
    print("📊 MODE 2: BENCHMARK HỎI ĐÁP PHÁP LÝ (Generation Metrics)")
    print("="*60)

    engine = _init_engine()
    retriever = _init_retriever()
    samples = dataset[:limit] if limit else dataset
    results = []

    for i, case in enumerate(tqdm(samples, desc="Mode 2 - QA")):
        question = case.get("question", "")
        ground_truth = case.get("ground_truth_answer", "")
        if not question:
            continue

        try:
            # Gọi RAG engine thật
            t0 = time.perf_counter()
            response = engine.chat(
                session_id="benchmark_session",
                query=question,
                use_reflection=False  # Tắt reflection để đo thuần chất lượng RAG
            )
            latency = time.perf_counter() - t0

            answer = response.get("answer", "")

            # Lấy contexts từ retriever (cho LLM judge)
            hits = retriever.search(query=question, limit=5, expand_context=False)
            contexts = [h.get("text", "") for h in hits if h.get("text")]

            # LLM-as-a-Judge scoring
            gen_metrics = compute_generation_metrics(question, answer, contexts, ground_truth)
            gen_metrics["latency_s"] = round(latency, 4)
            gen_metrics["question"] = question[:80]
            results.append(gen_metrics)

        except Exception as e:
            print(f"  [!] Error on question {i}: {e}")
            results.append({"question": question[:80], "error": str(e)})

        time.sleep(5)  # Rate limit Groq (3 LLM judge calls + 1 RAG call)

    # Aggregate
    if not results:
        return {"error": "No valid test cases"}

    avg_metrics = {}
    for key in ["faithfulness", "answer_relevance", "answer_correctness", "latency_s"]:
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        avg_metrics[f"avg_{key}"] = round(sum(values) / len(values), 4) if values else 0

    print(f"\n✅ Mode 2 Results ({len(results)} questions):")
    for k, v in avg_metrics.items():
        print(f"   {k}: {v}")

    return {
        "mode": "qa",
        "total_questions": len(results),
        "aggregate": avg_metrics,
        "per_question": results,
    }


# =====================================================================
# MODE 3: CONFLICT BENCHMARK (Classification Metrics)
# =====================================================================
def benchmark_mode3_conflict(dataset: List[Dict], limit: int = None) -> Dict[str, Any]:
    """
    Đánh giá phát hiện xung đột (Confusion Matrix, F1-Score, Recall per label).
    Dataset format: [{"claim": "...", "label": "contradiction|entailment|complex_reasoning", "expected_chunk_ids": [...]}]
    """
    print("\n" + "="*60)
    print("📊 MODE 3: BENCHMARK PHÁT HIỆN XUNG ĐỘT (NLI Classification)")
    print("="*60)

    retriever = _init_retriever()
    samples = dataset[:limit] if limit else dataset
    predictions = []
    ground_truths = []
    details = []

    for i, case in enumerate(tqdm(samples, desc="Mode 3 - Conflict")):
        claim = case.get("claim", "")
        expected_label = case.get("label", "").lower().strip()
        if not claim or not expected_label:
            continue

        try:
            # Tìm context pháp luật liên quan tới claim
            hits = retriever.search(query=claim, limit=3, expand_context=False)
            contexts = [h.get("text", "") for h in hits if h.get("text")]
            context_text = "\n---\n".join(contexts)

            # Dùng LLM để phân loại claim
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

            predictions.append(pred_label)
            ground_truths.append(expected_label)
            details.append({
                "claim": claim[:100],
                "expected": expected_label,
                "predicted": pred_label,
                "correct": pred_label == expected_label,
            })

        except Exception as e:
            print(f"  [!] Error on claim {i}: {e}")

        time.sleep(3)  # Rate limit

    # Classification report
    if not predictions:
        return {"error": "No valid test cases"}

    labels = ["contradiction", "entailment", "neutral", "complex_reasoning"]
    report = classification_report(predictions, ground_truths, labels)

    accuracy = sum(1 for d in details if d["correct"]) / len(details) if details else 0

    print(f"\n✅ Mode 3 Results ({len(details)} claims):")
    print(f"   Accuracy: {accuracy:.4f}")
    for label, metrics in report["per_label"].items():
        if label != "macro_avg":
            print(f"   {label}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1_score']:.3f}")

    return {
        "mode": "conflict",
        "total_claims": len(details),
        "accuracy": round(accuracy, 4),
        "classification_report": report["per_label"],
        "confusion_matrix": report["confusion_matrix"],
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


if __name__ == "__main__":
    main()
