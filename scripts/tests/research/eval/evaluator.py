"""
Evaluator Entry Point: Gọi Benchmark Runner để đánh giá hệ thống Legal RAG.
Đây là wrapper script cho benchmark_runner.py, cung cấp interface đơn giản.
"""
import os
import sys

# Cấu hình đường dẫn: 4 cấp lên tới Root (scripts/tests/research/eval/file.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESEARCH_DIR = os.path.join(ROOT_DIR, "scripts", "tests", "research")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from eval.benchmark_runner import benchmark_mode1_search, benchmark_mode2_qa, benchmark_mode3_conflict


def run_quick_evaluation(mode: str = "all", limit: int = 5):
    """
    Chạy đánh giá nhanh với số lượng test case giới hạn.
    mode: "1" (Search), "2" (QA), "3" (Conflict), "all"
    """
    import json
    from tests.config import (
        MODE1_OUTPUT_PATH, MODE2_OUTPUT_PATH, MODE3_OUTPUT_PATH,
        BENCHMARK_REPORT_JSON
    )

    results = {}

    if mode in ["1", "all"] and os.path.exists(MODE1_OUTPUT_PATH):
        with open(MODE1_OUTPUT_PATH, "r", encoding="utf-8") as f:
            results["mode1"] = benchmark_mode1_search(json.load(f), limit=limit)

    if mode in ["2", "all"] and os.path.exists(MODE2_OUTPUT_PATH):
        with open(MODE2_OUTPUT_PATH, "r", encoding="utf-8") as f:
            results["mode2"] = benchmark_mode2_qa(json.load(f), limit=limit)

    if mode in ["3", "all"] and os.path.exists(MODE3_OUTPUT_PATH):
        with open(MODE3_OUTPUT_PATH, "r", encoding="utf-8") as f:
            results["mode3"] = benchmark_mode3_conflict(json.load(f), limit=limit)

    # Save
    with open(BENCHMARK_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n📁 Kết quả đã lưu tại: {BENCHMARK_REPORT_JSON}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick Evaluator")
    parser.add_argument("--mode", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    run_quick_evaluation(mode=args.mode, limit=args.limit)
