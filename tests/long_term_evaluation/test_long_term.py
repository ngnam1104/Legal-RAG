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

# 1. Cấu hình Logger chuyên sâu
class Logger(object):
    def __init__(self, filename="tests/long_term_evaluation/long_term_results.txt"):
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

# Hàm đánh giá Multi-turn chuyên sâu (Judge)
def evaluate_multi_turn_answer(session_history, question, expected_answer, generated_answer):
    history_context = ""
    if session_history:
        history_context = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in session_history[-4:]])

    prompt = f"""Bạn là một Thẩm phán AI đánh giá hệ thống Legal-RAG trong hội thoại dài hạn.
Nhiệm vụ: So sánh CÂU TRẢ LỜI CỦA RAG với CÂU TRẢ LỜI MẪU trong ngữ cảnh hội thoại đã qua.

LỊCH SỬ HỘI THOẠI GẦN ĐÂY:
{history_context}

CÂU HỎI HIỆN TẠI: {question}
CÂU TRẢ LỜI MẪU: {expected_answer}
CÂU TRẢ LỜI TỪ RAG: {generated_answer}

TIÊU CHÍ ĐÁNH GIÁ:
1. ✅ ĐẠT: RAG trả lời đúng bản chất pháp lý VÀ hiểu đúng các tham chiếu ngữ cảnh (VD: "nó", "văn bản này", "ông ấy") từ lịch sử.
2. ✅ ĐẠT: Cách diễn đạt có thể khác mẫu nhưng thông tin cốt lõi và căn cứ pháp luật phải chính xác.
3. ❌ KHÔNG ĐẠT: Trả lời sai luật, báo không tìm thấy thông tin trong khi mẫu có, hoặc bị mất ngữ cảnh (hallucination về thực thể đã nhắc ở lượt trước).

ĐÁNH GIÁ CỦA BẠN: Trả về đúng MỘT DÒNG duy nhất bắt đầu bằng "✅ ĐẠT" hoặc "❌ KHÔNG ĐẠT", kèm theo lý giải ngắn gọn.
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

    results_file = "tests/long_term_evaluation/long_term_results.txt"
    failures_file = "tests/long_term_evaluation/long_term_false_result.txt"
    metrics_file = "tests/long_term_evaluation/long_term_metrics_summary.txt"
    
    sys.stdout = Logger(results_file)

    # Đọc dữ liệu test
    json_path = os.path.join(os.path.dirname(__file__), "long_term_test_data.json")
    if not os.path.exists(json_path):
        print(f"❌ Không tìm thấy file dữ liệu test: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            sessions_data = json.load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc file JSON: {e}")
        return

    print("STARTING TEST: LONG-TERM CONTEXT EVALUATION (MULTI-TURN)\n" + "="*70)
    print(f"Logging to: {os.path.abspath(results_file)}")
    print(f"Failures logged to: {os.path.abspath(failures_file)}")

    total_turns = 0
    passed_turns = 0
    intent_hits = 0
    all_turn_times = []
    mode_stats = {}
    all_step_times = {}
    comparison_results = []

    # Mở file ghi lỗi
    with open(failures_file, "w", encoding="utf-8") as f_fail:
        f_fail.write("=== LIST OF FAILED LONG-TERM TURNS ===\n\n")

        for s_idx, session_data in enumerate(sessions_data):
            session_id = f"long_eval_{uuid.uuid4().hex[:6]}"
            session_name = session_data.get("session_name", f"Session {s_idx+1}")
            doc_id = session_data.get("document_id", "Unknown")
            turns = session_data.get("turns", [])
            
            print(f"\n🎬 [SESSION {s_idx+1}/{len(sessions_data)}] {session_name} (Document: {doc_id})")
            print("-" * 70)
            
            current_history = []
            
            current_session_answers = []
            
            for turn in turns:
                total_turns += 1
                query = turn["query"]
                expected_answer = turn["expected_answer"]
                expected_intent = "LEGAL_CHAT" # Mặc định cho bộ test pháp lý dài hạn

                print(f"\n🔹 [Turn {turn['turn']}] Query: {query}")
                
                t0 = time.perf_counter()
                answer = "[No Answer Generated]"
                detected_mode = "Unknown"
                standalone_query = query

                try:
                    # Gọi RAG Engine với chế độ Streaming
                    async for event in rag_engine.chat(
                        session_id=session_id,
                        query=query,
                        mode="AUTO",
                        llm_preset="internal",
                        top_k=5,
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
                    print(f"   ❌ FAILED Execution: {e}")
                    answer = f"Exception: {e}"

                turn_time = time.perf_counter() - t0
                all_turn_times.append(turn_time)

                # Kiểm tra Intent
                intent_match = "✅ KHỚP" if (detected_mode and detected_mode.upper() == expected_intent.upper()) else "❌ SAI"
                if intent_match == "✅ KHỚP": intent_hits += 1

                print(f"   [Intent]: Target: {expected_intent} | Detected: {detected_mode} -> {intent_match}")
                print(f"   [Rewrite]: {standalone_query}")
                print(f"   [Answer]: {answer}")

                # Đánh giá bằng LLM Judge (Cung cấp context history)
                judge_result = evaluate_multi_turn_answer(current_history, query, expected_answer, answer)
                print(f"   [JUDGE]: {judge_result}")
                print(f"   ⏱️ [Time]: {turn_time:.2f}s")

                # Cập nhật thống kê theo Mode
                mode_name = detected_mode if detected_mode else "UNKNOWN"
                if mode_name not in mode_stats:
                    mode_stats[mode_name] = {"total": 0, "passed": 0, "times": []}
                mode_stats[mode_name]["total"] += 1
                mode_stats[mode_name]["times"].append(turn_time)

                # Lưu lịch sử hội thoại cho lượt kế tiếp (RAG Engine đã tự lưu trong memory, nhưng Judge cần bản copy này)
                current_history.append({"role": "user", "content": query})
                current_history.append({"role": "assistant", "content": answer})

                if "✅ ĐẠT" in judge_result:
                    passed_turns += 1
                    mode_stats[mode_name]["passed"] += 1
                else:
                    f_fail.write(f"--- FAILED: Session '{session_name}' | Turn {turn['turn']} ---\n")
                    f_fail.write(f"Original Query: {query}\n")
                    f_fail.write(f"Standalone Query: {standalone_query}\n")
                    f_fail.write(f"Judge: {judge_result}\n")
                    f_fail.write(f"Answer: {answer}\n")
                    f_fail.write("-" * 50 + "\n\n")
                    f_fail.flush()
                
                # Lưu kết quả lượt này (Giữ nguyên cấu trúc từ file gốc nhưng bỏ expected_answer)
                turn_result = turn.copy()
                if "expected_answer" in turn_result:
                    del turn_result["expected_answer"]

                turn_result.update({
                    "answer": answer,
                    "judge": judge_result,
                    "detected_mode": detected_mode,
                    "standalone_query": standalone_query
                })
                current_session_answers.append(turn_result)

                # Cập nhật vào danh sách tổng và lưu file ngay lập tức (Real-time update)
                existing_s = next((s for s in comparison_results if s["session_name"] == session_name), None)
                if existing_s:
                    existing_s["turns"] = current_session_answers
                else:
                    comparison_results.append({
                        "session_name": session_name,
                        "document_id": doc_id,
                        "turns": current_session_answers
                    })

                comparison_file = "tests/long_term_evaluation/legal_rag_answers.json"
                with open(comparison_file, "w", encoding="utf-8") as f_comp:
                    json.dump(comparison_results, f_comp, ensure_ascii=False, indent=4)

                print("-" * 40)
            


    # 4. Tính toán Metrics tổng thể
    accuracy = (passed_turns / total_turns * 100) if total_turns > 0 else 0
    intent_accuracy = (intent_hits / total_turns * 100) if total_turns > 0 else 0
    avg_turn = (sum(all_turn_times) / len(all_turn_times)) if all_turn_times else 0

    # 5. Xây dựng Báo cáo phân cấp (Giống evaluate_accuracy.py)
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("       BÁO CÁO ĐÁNH GIÁ HỘI THOẠI DÀI HẠN       ")
    report_lines.append("==================================================")
    report_lines.append("\n1. Chỉ số Tổng quan (Overall Metrics):")
    report_lines.append(f"- Tổng số lượt chat (Total Turns): {total_turns}")
    report_lines.append(f"- Tỷ lệ chính xác nội dung: {accuracy:.2f}% ({passed_turns}/{total_turns})")
    report_lines.append(f"- Tỷ lệ đúng Intent: {intent_accuracy:.2f}% ({intent_hits}/{total_turns})")
    report_lines.append(f"- Thời gian phản hồi trung bình: {avg_turn:.2f}s")

    report_lines.append("\n2. Chi tiết Hiệu suất theo Mode:")
    for mode, stats in mode_stats.items():
        if stats["total"] > 0:
            m_acc = (stats["passed"] / stats["total"]) * 100
            m_avg_t = sum(stats["times"]) / len(stats["times"])
            report_lines.append(f"  [{mode.upper()}]")
            report_lines.append(f"    + Số lượt kích hoạt: {stats['total']} lượt")
            report_lines.append(f"    + Tỷ lệ Đạt: {m_acc:.2f}% ({stats['passed']}/{stats['total']})")
            report_lines.append(f"    + Tốc độ trung bình: {m_avg_t:.2f}s")

    if all_step_times:
        report_lines.append("\n3. Phân rã thời gian Pipeline (Step Breakdown/Turn):")
        
        hierarchy = {
            "Preprocess Memory/Files_time": {"indent": 0},
            "Condense & Route_time": {"indent": 0},
            "Detect Mode Only_time": {"indent": 0},
            "Route.LLM_Call_time": {"indent": 2},
            "Route.JSON_Parse_time": {"indent": 2},
            "1. Understand_time": {"indent": 0},
            "2. Retrieve + Graph Expand_time": {"indent": 0},
            "Retrieve.Phase0_and_Phase1_Parallel_time": {"indent": 2},
            "Retrieve.QdrantNeo4j_Enrich_time": {"indent": 2},
            "Retrieve.Neo4j_Subgraph_time": {"indent": 2},
            "Retrieve.Graph_Doc_Fetch_time": {"indent": 2},
            "3. Generate_time": {"indent": 0},
            "Generate.BuildContext_time": {"indent": 2},
            "Generate.LLM_Call_time": {"indent": 2},
            "4. Reflect_time": {"indent": 0},
        }
        
        execution_order = [
            "Preprocess Memory/Files_time", "Condense & Route_time", "Detect Mode Only_time",
            "Route.LLM_Call_time", "Route.JSON_Parse_time", "1. Understand_time",
            "2. Retrieve + Graph Expand_time", "Retrieve.Phase0_and_Phase1_Parallel_time",
            "Retrieve.QdrantNeo4j_Enrich_time", "Retrieve.Neo4j_Subgraph_time",
            "Retrieve.Graph_Doc_Fetch_time", "3. Generate_time", "Generate.BuildContext_time",
            "Generate.LLM_Call_time", "4. Reflect_time"
        ]
        
        for step_key in execution_order:
            if step_key in all_step_times:
                avg_step = sum(all_step_times[step_key]) / len(all_step_times[step_key])
                indent = "  " * hierarchy.get(step_key, {}).get("indent", 0)
                report_lines.append(f"  ⚡ {indent}{step_key.replace('_time',''):40}: {avg_step:.2f}s")

    report_lines.append("\n📌 Ghi chú: Kết quả dựa trên khả năng duy trì Context qua nhiều lượt chat.")
    report_text = "\n".join(report_lines)
    
    # Lưu báo cáo metrics summary
    with open(metrics_file, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text)

    print("\n" + report_text)
    print("\n" + "="*70)
    print(f"👉 Báo cáo chi tiết: {results_file}")
    print(f"👉 Danh sách lỗi: {failures_file}")
    print(f"👉 File JSON so sánh: tests/long_term_evaluation/legal_rag_answers.json")

if __name__ == "__main__":
    asyncio.run(main())
