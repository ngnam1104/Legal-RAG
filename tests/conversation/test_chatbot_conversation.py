import sys
import os
import json
import asyncio
import uuid
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.agent.chat_engine import rag_engine
from backend.llm.factory import chat_completion
from backend.config import settings

# 1. Cấu hình Logger để vừa in ra màn hình vừa ghi file
class Logger(object):
    def __init__(self, filename="tests/conversation/test_chatbot_conversation_results.txt"):
        self.terminal = sys.stdout
        # Đảm bảo thư mục kết quả tồn tại
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Hàm đánh giá bằng LLM Judge (Tolerant version)
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
        res = chat_completion([{"role": "user", "content": prompt}], temperature=0.0, model=settings.LLM_ROUTING_MODEL)
        return str(res).strip()
    except Exception as e:
        return f"❌ LỖI ĐÁNH GIÁ LLM: {str(e)}"

async def main():
    # Đảm bảo encoding chuẩn cho terminal
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # Chuyển hướng stdout vào file
    sys.stdout = Logger("tests/conversation/test_chatbot_conversation_results.txt")

    # 3. Chạy test từ file Chatbot_test_conversation.json
    try:
        json_path = os.path.join(os.path.dirname(__file__), "Chatbot_test_conversation.json")
        with open(json_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Lỗi đọc file json: {e}")
        return

    print("STARTING TEST: CONVERSATION MEMORY (MULTI-TURN)\n" + "="*60)
    print(f"Logging to: {os.path.abspath('tests/conversation/test_chatbot_conversation_results.txt')}")
    print(f"Logging failures to: {os.path.abspath('tests/conversation/test_conversation_false_result.txt')}")
    
    conversations = test_data.get("conversations", [])
    total_turns = 0
    passed_turns = 0
    intent_hits = 0
    all_turn_times = []
    mode_stats = {} # Tracks total, passed, times per intent/mode
    all_step_times = {} # Tracks average step execution times across all turns

    with open(os.path.join(os.path.dirname(__file__), "test_conversation_false_result.txt"), "w", encoding="utf-8") as f_fail:
        f_fail.write("=== LIST OF FAILED CONVERSATION TURNS ===\n\n")

        for conv in conversations:
            # Create a unique session ID for testing
            session_id = f"test_{conv['id']}_{uuid.uuid4().hex[:6]}"
            role = conv.get("role_description", "N/A")

            print(f"\n" + "="*80)
            print(f"📂 HỘI THOẠI: {conv['id']}")
            print(f"👤 VAI TRÒ: {role}")
            print("="*80)
            
            for msg in conv["messages"]:
                total_turns += 1
                turn = msg.get("turn", 0)
                question = msg["question"]
                expected_intent = msg.get("intent", "")
                target_rewritten = msg.get("rewritten_query", "")
                expected_answer = msg["answer"]
                expected_citation = msg.get("citation", "")
                
                print(f"\n🔹 [Turn {turn}] Query: {question}")
                t0 = time.perf_counter()
                
                # Streaming execution via RAGEngine
                answer = "[No Answer Generated]"
                detected_mode = "Unknown"
                standalone_query = question

                try:
                    async for event in rag_engine.chat(
                        session_id=session_id,
                        query=question,
                        mode="AUTO",
                        llm_preset="groq_8b",
                        top_k=5,
                        use_reflection=False,
                        use_grading=True,
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

                # --- Intent & Rewrite check ---
                intent_match = "✅ KHỚP" if (detected_mode and expected_intent and detected_mode.upper() == expected_intent.upper()) else "❌ SAI"
                if intent_match == "✅ KHỚP": intent_hits += 1

                print(f"   [Intent] Target: {expected_intent} | Detected: {detected_mode} -> {intent_match}")
                print(f"   [Rewrite] Target: {target_rewritten}")
                print(f"   [Rewrite] Actual: {standalone_query}")
                print(f"   [Answer]: {answer}")
                
                # Update mode stats
                mode_name = detected_mode if detected_mode else "UNKNOWN"
                if mode_name not in mode_stats:
                    mode_stats[mode_name] = {"total": 0, "passed": 0, "times": []}
                mode_stats[mode_name]["total"] += 1
                mode_stats[mode_name]["times"].append(turn_time)
                
                # Đánh giá kết quả
                judge_result = evaluate_answer(question, expected_answer, expected_citation, answer)
                print(f"   [JUDGE]: {judge_result}")
                print(f"   ⏱️ [Time]: {turn_time:.2f}s")
                
                if "✅ ĐẠT" in judge_result:
                    passed_turns += 1
                    mode_stats[mode_name]["passed"] += 1
                else:
                    f_fail.write(f"--- FAILED: {conv['id']} | Turn {turn} (Role: {role}) ---\n")
                    f_fail.write(f"Query: {question}\n")
                    f_fail.write(f"Intent: {expected_intent} (Expected) vs {detected_mode} (Actual)\n")
                    f_fail.write(f"Rewrite: {standalone_query}\n")
                    f_fail.write(f"Judge: {judge_result}\n")
                    f_fail.write(f"Answer: {answer}\n")
                    f_fail.write("-" * 50 + "\n\n")
                    f_fail.flush()
                
                print("-" * 40)

    accuracy = (passed_turns / total_turns * 100) if total_turns > 0 else 0
    intent_accuracy = (intent_hits / total_turns * 100) if total_turns > 0 else 0
    avg_turn = (sum(all_turn_times) / len(all_turn_times)) if all_turn_times else 0

    # Viết file báo cáo
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("   BÁO CÁO ĐÁNH GIÁ ĐA LƯỢT (CONVERSATION RAG)    ")
    report_lines.append("==================================================")
    report_lines.append("\n1. Phương pháp Đánh giá:")
    report_lines.append("- Trọng tài: LLM Judge hỗ trợ đánh giá ngữ cảnh đa lượt hội thoại.")
    report_lines.append("- Tiêu chí: Đánh giá nới lỏng (Tolerant). Kiểm tra độ khớp intent và hỗ trợ vai trò người dùng.")
    report_lines.append("\n2. Chỉ số Tổng quan (Overall Metrics):")
    report_lines.append(f"- Tổng số lượt hội thoại test (Total Turns): {total_turns}")
    report_lines.append(f"- Tỷ lệ chính xác nội dung (Content Acc): {accuracy:.2f}% ({passed_turns}/{total_turns})")
    report_lines.append(f"- Tỷ lệ đúng Intent (Intent Acc): {intent_accuracy:.2f}% ({intent_hits}/{total_turns})")
    report_lines.append(f"- Thời gian phản hồi trung bình (Per turn): {avg_turn:.2f}s")

    report_lines.append("\n3. Chi tiết Hiệu suất theo Mode (Detected intent):")
    for mode, stats in mode_stats.items():
        if stats["total"] > 0:
            m_acc = (stats["passed"] / stats["total"]) * 100
            m_avg_t = sum(stats["times"]) / len(stats["times"])
            report_lines.append(f"  [{mode.upper()}]")
            report_lines.append(f"    + Số lượt kích hoạt: {stats['total']} lượt")
            report_lines.append(f"    + Tỷ lệ Đạt: {m_acc:.2f}% ({stats['passed']}/{stats['total']})")
            report_lines.append(f"    + Tốc độ sinh: {m_avg_t:.2f}s")


    if all_step_times:
        report_lines.append("\n4. Phân rã thời gian Pipeline (Step Breakdown/Turn):")

        # Sắp xếp theo đúng trình tự pipeline thay vì Alphabetical
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
    print(f"👉 Chi tiết báo cáo đã được lưu vào: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
