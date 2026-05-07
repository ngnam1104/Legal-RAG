import sys
import os
import json
import asyncio
import uuid
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.agent.chat_engine import rag_engine
from backend.models.llm_factory import chat_completion
from backend.config import settings

# 1. Cấu hình Logger
class Logger(object):
    def __init__(self, filename="tests/qa_evaluation/evaluate_accuracy_results.txt"):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Hàm đánh giá
def evaluate_answer(question, expected_answer, expected_citation, generated_answer):
    prompt = f"""Bạn là một Thẩm phán AI đánh giá câu trả lời của hệ thống RAG.
Nhiệm vụ: So sánh CÂU TRẢ LỜI CỦA RAG với CÂU TRẢ LỜI MẪU xem tính chính xác và đầy đủ. Yêu cầu CHẤM ĐIỂM NỚI LỎNG (Tolerant/Flexible):
1. ✅ ĐẠT: Nếu RAG trả lời đúng hướng, chứa thông tin chính xác tương đương hoặc RỘNG HƠN câu trả lời mẫu.
2. ✅ ĐẠT: RAG có trích dẫn đúng hoặc gần đúng các số hiệu văn bản trọng tâm. Không bắt bẻ sai số toán học / ngày tháng cực nhỏ (lệch 1 ngày).
3. ❌ KHÔNG ĐẠT: RAG trả lời sai bản chất, báo 'không tìm thấy' (khi mẫu có thông tin), hoặc dẫn sai hoàn toàn số hiệu văn bản cốt lõi.

Câu hỏi: {question}
Câu trả lời mẫu: {expected_answer}
Trích dẫn mẫu mong đợi: {expected_citation}
Câu trả lời từ RAG (bao gồm cả trích dẫn): {generated_answer}

ĐÁNH GIÁ CỦA BẠN: Bắt buộc trả về đúng MỘT DÒNG duy nhất bắt đầu bằng "✅ ĐẠT" hoặc "❌ KHÔNG ĐẠT", kèm theo một câu giải thích ngắn gọn lý do.
"""
    try:
        from backend.utils.text_utils import strip_thinking_tags
        res = chat_completion([{"role": "user", "content": prompt}], temperature=0.0)
        return strip_thinking_tags(str(res)).strip()
    except Exception as e:
        return f"❌ LỖI ĐÁNH GIÁ LLM: {str(e)}"

async def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    sys.stdout = Logger("tests/qa_evaluation/evaluate_accuracy_results.txt")

    # 3. Đọc dữ liệu
    test_cases = []
    try:
        json_path = os.path.join(os.path.dirname(__file__), "Chatbot_test_2mode_3docs.json")
        with open(json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
            for item in test_data:
                # We expect intent to be LEGAL_CHAT for all these legal complex queries
                test_cases.append({
                    "document_id": item["document_id"],
                    "question": item["question"],
                    "intent": "LEGAL_CHAT",
                    "answer": item["answer"],
                    "citation": item["citation"]
                })
    except Exception as e:
        print(f"Lỗi đọc file json: {e}")
        return

    print("STARTING TEST: QA ACCURACY (3 DOCS - GRAPH MULTI-HOP)\n" + "="*60)
    print(f"Logging to: {os.path.abspath('tests/qa_evaluation/evaluate_accuracy_results.txt')}")
    print(f"Logging failures to: {os.path.abspath('tests/qa_evaluation/test_false_result.txt')}")

    total_tests = len(test_cases)
    passed_count = 0
    intent_hits = 0
    all_turn_times = []
    mode_stats = {}
    all_step_times = {}

    with open(os.path.join(os.path.dirname(__file__), "test_false_result.txt"), "w", encoding="utf-8") as f_fail:
        f_fail.write("=== LIST OF FAILED TEST CASES ===\n\n")

        for idx, item in enumerate(test_cases):
            document_id = item["document_id"]
            question = item["question"]
            expected_intent = item["intent"]
            expected_answer = item["answer"]
            expected_citation = item["citation"]

            print(f"\n--- [Case {idx+1}/{total_tests}] [{document_id.upper()}]")
            print(f"🔹 Query: {question}")
            
            t0 = time.perf_counter()
            session_id = f"eval_{uuid.uuid4().hex[:6]}"
            
            answer = "[No Answer Generated]"
            detected_mode = "Unknown"
            standalone_query = question

            try:
                # Streaming execution via RAGEngine
                async for event in rag_engine.chat(
                    session_id=session_id,
                    query=question,
                    mode="AUTO",
                    llm_preset="internal", # Sử dụng preset local theo project setup
                    top_k=5,
                    use_reflection=False,
                    use_grading=False,
                    use_rerank=True
                ):
                    if event["type"] == "final":
                        answer = event["content"].get("answer", answer)
                        detected_mode = event["content"].get("detected_mode", detected_mode)
                        standalone_query = event["content"].get("standalone_query", standalone_query)
                        metrics = event["content"].get("metrics", {})
                        for step_name, duration in metrics.items():
                            if step_name not in all_step_times:
                                all_step_times[step_name] = []
                            all_step_times[step_name].append(duration)
                    elif event["type"] == "error":
                        answer = f"ERROR: {event['content']}"

            except Exception as e:
                print(f"FAILED Execution: {e}")
                answer = f"Exception: {e}"

            turn_time = time.perf_counter() - t0
            all_turn_times.append(turn_time)

            # Intent Check
            intent_match = "✅ KHỚP" if (detected_mode and expected_intent and detected_mode.upper() == expected_intent.upper()) else "❌ SAI"
            if intent_match == "✅ KHỚP": intent_hits += 1

            print(f"   [Intent] Target: {expected_intent} | Detected: {detected_mode} -> {intent_match}")
            print(f"   [Rewrite]: {standalone_query}")
            print(f"   [Answer]: {answer}")

            # Mode stats update
            mode_name = detected_mode if detected_mode else "UNKNOWN"
            if mode_name not in mode_stats:
                mode_stats[mode_name] = {"total": 0, "passed": 0, "times": []}
            mode_stats[mode_name]["total"] += 1
            mode_stats[mode_name]["times"].append(turn_time)

            # Đánh giá bằng LLM Judge
            judge_result = evaluate_answer(question, expected_answer, expected_citation, answer)
            print(f"   [JUDGE]: {judge_result}")
            print(f"   ⏱️ [Time]: {turn_time:.2f}s")

            if "✅ ĐẠT" in judge_result:
                passed_count += 1
                mode_stats[mode_name]["passed"] += 1
            else:
                f_fail.write(f"--- FAILED: {document_id.upper()} ---\n")
                f_fail.write(f"Query: {question}\n")
                f_fail.write(f"Intent: {expected_intent} (Expected) vs {detected_mode} (Actual)\n")
                f_fail.write(f"Judge: {judge_result}\n")
                f_fail.write(f"Answer: {answer}\n")
                f_fail.write("-" * 50 + "\n\n")
                f_fail.flush()
            
            print("-" * 60)

    # Calculate metrics
    accuracy = (passed_count / total_tests * 100) if total_tests > 0 else 0
    intent_accuracy = (intent_hits / total_tests * 100) if total_tests > 0 else 0
    avg_turn = (sum(all_turn_times) / len(all_turn_times)) if all_turn_times else 0

    # Báo cáo
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("         BÁO CÁO ĐÁNH GIÁ ĐỘ CHÍNH XÁC QA         ")
    report_lines.append("==================================================")
    report_lines.append("\n1. Chỉ số Tổng quan (Overall Metrics):")
    report_lines.append(f"- Tổng số test cases: {total_tests}")
    report_lines.append(f"- Tỷ lệ chính xác nội dung (Content Acc): {accuracy:.2f}% ({passed_count}/{total_tests})")
    report_lines.append(f"- Tỷ lệ đúng Intent (Intent Acc): {intent_accuracy:.2f}% ({intent_hits}/{total_tests})")
    report_lines.append(f"- Thời gian phản hồi trung bình: {avg_turn:.2f}s")

    report_lines.append("\n2. Chi tiết Hiệu suất theo Mode:")
    for mode, stats in mode_stats.items():
        if stats["total"] > 0:
            m_acc = (stats["passed"] / stats["total"]) * 100
            m_avg_t = sum(stats["times"]) / len(stats["times"])
            report_lines.append(f"  [{mode.upper()}]")
            report_lines.append(f"    + Số lượt kích hoạt: {stats['total']} lượt")
            report_lines.append(f"    + Tỷ lệ Đạt: {m_acc:.2f}% ({stats['passed']}/{stats['total']})")
            report_lines.append(f"    + Tốc độ sinh: {m_avg_t:.2f}s")

    if all_step_times:
        report_lines.append("\n3. Phân rã thời gian Pipeline (Step Breakdown/Turn):")
        order = [
            "Preprocess Memory/Files",
            "Detect Mode Only", 
            "Condense & Route",
            "Understand",
            "Retrieve + Graph Expand",
            "Generate"
        ]
        
        def get_order_index(x):
            for i, prefix in enumerate(order):
                if x.startswith(prefix):
                    return i
            return 999
            
        sorted_steps = sorted(all_step_times.keys(), key=get_order_index)

        for step_name in sorted_steps:
            avg_step = sum(all_step_times[step_name]) / len(all_step_times[step_name])
            clean_name = step_name.replace("_time", "")
            report_lines.append(f"  ⚡ {clean_name:30}: {avg_step:.2f}s")

    report_text = "\n".join(report_lines)
    report_file = os.path.join(os.path.dirname(__file__), "metrics_report.txt")
    with open(report_file, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text)

    print("\n\n" + report_text)
    print("="*60 + "\n")
    print(f"👉 Chi tiết báo cáo lưu tại: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
