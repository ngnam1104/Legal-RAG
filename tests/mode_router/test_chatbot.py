import sys
import os
import json
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.agent.graph import LegalRAGWorkflow
from backend.llm.factory import chat_completion
from backend.config import settings

# 1. Cấu hình Logger để vừa in ra màn hình vừa ghi file
class Logger(object):
    def __init__(self, filename="tests/mode_router/test_chatbot_results.txt"):
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

# Đảm bảo encoding chuẩn cho terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Chuyển hướng stdout vào file
sys.stdout = Logger("tests/mode_router/test_chatbot_results.txt")

# 2. Khởi tạo RAG Graph
workflow = LegalRAGWorkflow()
graph = workflow.build()

# 3. Chạy test từ file Chatbot_test.json
test_cases = []
try:
    json_path = os.path.join(os.path.dirname(__file__), "Chatbot_test_3mode_1docs.json")
    with open(json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        for item in test_data:
            test_cases.append((item["document_id"], item["question"], item["category"], item["answer"], item["citation"]))
except Exception as e:
    print(f"Lỗi đọc file json: {e}")

print("STARTING TEST: 3 RAG MODES (ON-PREMISE)\n" + "="*60)
print(f"Logging to: {os.path.abspath('tests/mode_router/test_chatbot_results.txt')}")
print(f"Logging failures to: {os.path.abspath('tests/mode_router/test_false_result.txt')}")

# 4. Hàm đánh giá bằng LLM Judge
def evaluate_answer(question, expected_answer, expected_citation, generated_answer):
    prompt = f"""Bạn là một Thẩm phán AI đánh giá câu trả lời của hệ thống RAG.
Nhiệm vụ: So sánh CÂU TRẢ LỜI CỦA RAG với CÂU TRẢ LỜI MẪU xem tính chính xác và đầy đủ. Yêu cầu CHẤM ĐIỂM NỚI LỎNG (Tolerant/Flexible):
1. ✅ ĐẠT: Nếu RAG trả lời đúng hướng, chứa thông tin chính xác tương đương hoặc RỘNG HƠN câu trả lời mẫu (VD: mẫu chỉ nêu 1 tỉnh nhưng RAG nêu 2 tỉnh bao gồm cả tỉnh đó).
2. ✅ ĐẠT: RAG có trích dẫn đúng hoặc gần đúng các số hiệu văn bản trọng tâm. Không bắt bẻ sai số toán học / ngày tháng cực nhỏ (lệch 1 ngày).
3. ❌ KHÔNG ĐẠT: RAG trả lời sai bản chất, báo 'không tìm thấy' (khi mẫu có thông tin), hoặc dẫn sai hoàn toàn số hiệu văn bản cốt lõi.

Câu hỏi: {question}
Câu trả lời mẫu: {expected_answer}
Trích dẫn mẫu mong đợi: {expected_citation}
Câu trả lời từ RAG (bao gồm cả trích dẫn): {generated_answer}

ĐÁNH GIÁ CỦA BẠN: Bắt buộc trả về đúng MỘT DÒNG duy nhất bắt đầu bằng "✅ ĐẠT" hoặc "❌ KHÔNG ĐẠT", kèm theo một câu giải thích ngắn gọn lý do.
"""
    try:
        from backend.config import settings
        res = chat_completion([{"role": "user", "content": prompt}], temperature=0.0, model=settings.LLM_ROUTING_MODEL)
        return str(res).strip()
    except Exception as e:
        return f"❌ LỖI ĐÁNH GIÁ LLM: {str(e)}"

# 5. Vòng lặp thực thi test duy nhất
total_tests = len(test_cases)
passed_count = 0
failed_results = []
all_total_times = []
all_step_times = {} # Dictionary of lists for step-level averages
mode_stats = {} # Tracks total, passed, times per intent

with open(os.path.join(os.path.dirname(__file__), "test_false_result.txt"), "w", encoding="utf-8") as f_fail:
    f_fail.write("=== LIST OF FAILED TEST CASES ===\n\n")

    for document_id, question, intent, expected_answer, expected_citation in test_cases:
        print(f"\n--- [{document_id.upper()}] Query: {question}")
        t0 = time.perf_counter()
        
        initial_state = {
            "query": question, 
            "mode": "AUTO",
            "session_id": f"test_{intent}",
            "history": [],
            "use_grading": True,
            "use_reflection": False,
            "use_rerank": True,
            "top_k": 5
        }
        
        try:
            # invoke là synchronous call
            res = graph.invoke(initial_state)
            
            # Thu thập metrics từ Graph
            metrics = res.get("metrics", {})
            for step_name, duration in metrics.items():
                if step_name not in all_step_times:
                    all_step_times[step_name] = []
                all_step_times[step_name].append(duration)

            # Lấy kết quả từ answer (mode thường) hoặc final_response (mode reflect)
            answer = res.get('answer') or res.get('final_response') or res.get('draft_response') or "[No Answer]"
            print(f"\n[Generated Answer]:\n{answer}")
            
            # Đánh giá kết quả
            print(f"\n[Expected Answer]: {expected_answer}")
            print(f"[Expected Citation]: {expected_citation}")
            judge_result = evaluate_answer(question, expected_answer, expected_citation, answer)
            print(f"\n[LLM JUDGE]: {judge_result}")
            
            total_time = time.perf_counter() - t0
            all_total_times.append(total_time)
            print(f"⏱️ [Total Turnaround Time]: {total_time:.2f}s")
            
            # Lọc mode_name
            mode_name = intent if intent else "UNKNOWN"
            if mode_name not in mode_stats:
                mode_stats[mode_name] = {"total": 0, "passed": 0, "times": []}
            mode_stats[mode_name]["total"] += 1
            mode_stats[mode_name]["times"].append(total_time)

            if "✅ ĐẠT" in judge_result:
                passed_count += 1
                mode_stats[mode_name]["passed"] += 1
            else:
                # Ghi vào file câu lỗi
                f_fail.write(f"--- FAILED: {document_id.upper()} ---\n")
                f_fail.write(f"Query: {question}\n")
                f_fail.write(f"Judge: {judge_result}\n")
                f_fail.write(f"Answer: {answer}\n")
                f_fail.write("-" * 40 + "\n\n")
                f_fail.flush()
            
        except Exception as e:
            print(f"FAILED Execution: {e}")
            f_fail.write(f"ERROR executing query: {question} -> {e}\n")
        
        print("-" * 60)

accuracy = (passed_count / total_tests * 100) if total_tests > 0 else 0
avg_total = (sum(all_total_times) / len(all_total_times)) if all_total_times else 0

# Viết file báo cáo
report_lines = []
report_lines.append("==================================================")
report_lines.append("         BÁO CÁO ĐÁNH GIÁ CHẤT LƯỢNG RAG          ")
report_lines.append("==================================================")
report_lines.append("\n1. Phương pháp Đánh giá:")
report_lines.append("- Trọng tài: LLM Judge tự động so sánh Câu trả lời của RAG với Ground Truth.")
report_lines.append("- Tiêu chí: Đánh giá nới lỏng (Tolerant). Hệ thống đạt điểm nếu bao hàm đủ ý chỉ đạo hoặc rộng hơn tài liệu gốc.")
report_lines.append("\n2. Chỉ số Tổng quan (Overall Metrics):")
report_lines.append(f"- Tổng số test cases: {total_tests}")
report_lines.append(f"- Tỷ lệ chính xác chung (Accuracy): {accuracy:.2f}% ({passed_count}/{total_tests})")
report_lines.append(f"- Thời gian trung bình 1 chu kỳ: {avg_total:.2f}s")

report_lines.append("\n3. Chi tiết Hiệu suất theo Mode (Intent):")
for mode, stats in mode_stats.items():
    if stats["total"] > 0:
        m_acc = (stats["passed"] / stats["total"]) * 100
        m_avg_t = sum(stats["times"]) / len(stats["times"])
        report_lines.append(f"  [{mode.upper()}]")
        report_lines.append(f"    + Số lượng: {stats['total']} queries")
        report_lines.append(f"    + Tỷ lệ Đạt: {m_acc:.2f}% ({stats['passed']}/{stats['total']})")
        report_lines.append(f"    + Tốc độ sinh: {m_avg_t:.2f}s")

if all_step_times:
    report_lines.append("\n4. Phân rã thời gian Pipeline (Step Breakdown):")
    for step_name in sorted(all_step_times.keys()):
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
