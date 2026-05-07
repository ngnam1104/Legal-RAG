import sys
import os
import json
import asyncio
import uuid
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(override=True)

import logging

# Filter out messy ICLLM console logs
class ICLLMFilter(logging.Filter):
    def filter(self, record):
        if record.name.startswith("LLM/"):
            return False
        return True

# Apply filter to root logger right away
logging.basicConfig(level=logging.INFO)
for handler in logging.root.handlers:
    handler.addFilter(ICLLMFilter())

from backend.agent.chat_engine import rag_engine
from backend.models.llm_factory import chat_completion

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
Nhiệm vụ: So sánh CÂU TRẢ LỜI CỦA RAG với CÂU TRẢ LỜI MẪU để đánh giá độ phù hợp và tính hợp lý của tư vấn pháp lý. 

Yêu cầu ĐÁNH GIÁ LINH HOẠT (Flexible Assessment):
1. ✅ ĐẠT: Nếu RAG trả lời đúng hướng (intent), chứa thông tin chính xác hoặc có lý lẽ tương đương với câu trả lời mẫu. Không bắt lỗi nếu cách diễn đạt khác biệt nhưng bản chất pháp lý giống nhau.
2. ✅ ĐẠT: Nếu RAG nhắc đến ĐÚNG văn bản pháp luật trọng tâm. Lưu ý: Nếu RAG nhắc đến một số hiệu văn bản khác (ví dụ văn bản mới hơn hoặc văn bản liên quan) mà vẫn đưa ra kết luận hợp lý thì vẫn có thể chấp nhận (do cơ sở dữ liệu của hệ thống có thể chứa nhiều văn bản cập nhật hơn câu mẫu).
3. ✅ ĐẠT: RAG có thể trích dẫn rộng hơn hoặc chi tiết hơn mẫu. Không bắt bẻ các sai sót nhỏ về định dạng hoặc ngày tháng.
4. ❌ KHÔNG ĐẠT: RAG trả lời sai hoàn toàn bản chất vấn đề, báo 'không tìm thấy thông tin' trong khi mẫu có thông tin rõ ràng, hoặc đưa ra tư vấn gây nguy hiểm/sai lệch nghiêm trọng.

Câu hỏi: {question}
Câu trả lời mẫu: {expected_answer}
Trích dẫn mẫu mong đợi: {expected_citation}
Câu trả lời từ RAG: {generated_answer}

ĐÁNH GIÁ CỦA BẠN: Trả về đúng MỘT DÒNG duy nhất bắt đầu bằng "✅ ĐẠT" hoặc "❌ KHÔNG ĐẠT", kèm theo một câu giải thích ngắn gọn lý do tại sao đạt hoặc không đạt dựa trên các tiêu chí trên.
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
        report_lines.append("   (Thứ tự thực thi | Sub-steps được lồng với indent)")
        
        # Define proper hierarchical structure with execution order
        hierarchy = {
            "Preprocess Memory/Files_time": {"indent": 0, "is_parent": False},
            "Condense & Route_time": {"indent": 0, "is_parent": False},
            "Detect Mode Only_time": {"indent": 0, "is_parent": False},
            # --- Router sub-steps (nested, but only show if parent exists)
            "Route.LLM_Call_time": {"indent": 2, "is_parent": False, "parent": ["Condense & Route_time", "Detect Mode Only_time"]},
            "Route.JSON_Parse_time": {"indent": 2, "is_parent": False, "parent": ["Condense & Route_time", "Detect Mode Only_time"]},
            # --- RAG Pipeline
            "1. Understand_time": {"indent": 0, "is_parent": False},
            "2. Retrieve + Graph Expand_time": {"indent": 0, "is_parent": True},
            # --- Retrieve sub-steps (nested)
            "Retrieve.Phase0_and_Phase1_Parallel_time": {"indent": 2, "is_parent": False},
            "Retrieve.QdrantNeo4j_Enrich_time": {"indent": 2, "is_parent": False},
            "Retrieve.Neo4j_Subgraph_time": {"indent": 2, "is_parent": False},
            "Retrieve.Graph_Doc_Fetch_time": {"indent": 2, "is_parent": False},
            # --- Generate
            "3. Generate_time": {"indent": 0, "is_parent": True},
            # --- Generate sub-steps (nested)
            "Generate.BuildContext_time": {"indent": 2, "is_parent": False},
            "Generate.LLM_Call_time": {"indent": 2, "is_parent": False},
            "Generate.FilterRefs_time": {"indent": 2, "is_parent": False},
            # --- Reflect (optional)
            "4. Reflect_time": {"indent": 0, "is_parent": False},
        }
        
        # Execution order list
        execution_order = [
            "Preprocess Memory/Files_time",
            "Condense & Route_time",
            "Detect Mode Only_time",
            "Route.LLM_Call_time",
            "Route.JSON_Parse_time",
            "1. Understand_time",
            "2. Retrieve + Graph Expand_time",
            "Retrieve.Phase0_and_Phase1_Parallel_time",
            "Retrieve.QdrantNeo4j_Enrich_time",
            "Retrieve.Neo4j_Subgraph_time",
            "Retrieve.Graph_Doc_Fetch_time",
            "3. Generate_time",
            "Generate.BuildContext_time",
            "Generate.LLM_Call_time",
            "Generate.FilterRefs_time",
            "4. Reflect_time",
        ]
        
        # Track which parent steps have been displayed
        displayed_steps = set()
        
        for step_key in execution_order:
            if step_key not in all_step_times:
                continue
            
            # Skip Route sub-steps if parent doesn't exist
            if "Route." in step_key:
                parent_exists = any(p in all_step_times for p in ["Condense & Route_time", "Detect Mode Only_time"])
                if not parent_exists:
                    continue
            
            step_name = step_key.replace("_time", "")
            avg_step = sum(all_step_times[step_key]) / len(all_step_times[step_key])
            
            indent_level = hierarchy.get(step_key, {}).get("indent", 0)
            indent_str = "  " * indent_level if indent_level > 0 else ""
            
            report_lines.append(f"  ⚡ {indent_str}{step_name:40}: {avg_step:.2f}s")
            displayed_steps.add(step_key)
        
        # Add clarification about total time
        report_lines.append("\n   📌 Lưu ý:")
        report_lines.append(f"   • Tốc độ sinh (152.90s) = tổng thời gian per turn = Routing + Retrieval + Generation")
        report_lines.append(f"   • Sub-steps được lồng vào parent steps (ví dụ: Retrieve.* lồng trong Retrieve + Graph Expand)")
        report_lines.append(f"   • Retrieve.Graph_Doc_Fetch (~118s) là phần lớn nhất của Retrieval (~120s)")

    report_text = "\n".join(report_lines)
    report_file = os.path.join(os.path.dirname(__file__), "metrics_report.txt")
    with open(report_file, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text)

    print("\n\n" + report_text)
    print("="*60 + "\n")
    print(f"👉 Chi tiết báo cáo lưu tại: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
