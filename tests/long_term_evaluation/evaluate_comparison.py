import sys
import os
import json
import asyncio
import time

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.models.llm_factory import chat_completion
from backend.utils.text_utils import strip_thinking_tags

DATA_FILE = "tests/long_term_evaluation/long_term_test_data.json"
RAG_FILE = "tests/long_term_evaluation/legal_rag_answers.json"
GPT_FILE = "tests/long_term_evaluation/chatgpt_answers.json"
REPORT_FILE = "tests/long_term_evaluation/accuracy_comparison_report.json"

def create_chatgpt_template():
    """Tạo file JSON mẫu để người dùng điền câu trả lời từ ChatGPT UI."""
    if not os.path.exists(DATA_FILE):
        print(f"❌ Không tìm thấy {DATA_FILE}. Hãy chạy utils/generate_test_dataset.py trước.")
        return False

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    template = []
    for sess in data:
        template_sess = {
            "session_name": sess["session_name"],
            "document_id": sess["document_id"],
            "turns": []
        }
        for turn in sess["turns"]:
            turn_template = turn.copy()
            if "expected_answer" in turn_template:
                del turn_template["expected_answer"] # Xóa đáp án mẫu để điền đáp án ChatGPT
            
            turn_template["answer"] = "DÁN CÂU TRẢ LỜI CỦA CHATGPT VÀO ĐÂY"
            template_sess["turns"].append(turn_template)
        template.append(template_sess)

    with open(GPT_FILE, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Đã tạo file mẫu tại: {GPT_FILE}")
    print("💡 Hướng dẫn: Mở file trên, dán câu trả lời từ ChatGPT vào rồi chạy lại script này để so sánh.")
    return True

def evaluate_answer_vs_ground_truth(history, query, ground_truth, answer, system_name):
    """Sử dụng LLM Judge chấm điểm Đạt/Không Đạt so với Ground Truth."""
    history_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-2:]])
    
    prompt = f"""Bạn là một chuyên gia Pháp lý đánh giá AI. 
Nhiệm vụ: Chấm điểm câu trả lời của {system_name} dựa trên ĐÁP ÁN MẪU (GROUND TRUTH).

LỊCH SỬ GẦN ĐÂY:
{history_str}

CÂU HỎI: {query}
ĐÁP ÁN MẪU: {ground_truth}
CÂU TRẢ LỜI CỦA {system_name}: {answer}

TIÊU CHÍ:
- ✅ ĐẠT: Đúng bản chất pháp lý, đúng căn cứ (Điều/Khoản) và giữ được ngữ cảnh multi-turn.
- ❌ KHÔNG ĐẠT: Sai luật, thiếu căn cứ hoặc bị mất ngữ cảnh (quên nhân vật/văn bản đã nhắc).

TRẢ VỀ: Một dòng bắt đầu bằng "✅ ĐẠT" hoặc "❌ KHÔNG ĐẠT" kèm lý do ngắn gọn.
"""
    try:
        res = chat_completion([{"role": "user", "content": prompt}], temperature=0.0)
        return strip_thinking_tags(str(res)).strip()
    except Exception as e:
        return f"❌ LỖI: {str(e)}"

async def run_comparison():
    """Chạy đánh giá so sánh Accuracy."""
    if not os.path.exists(RAG_FILE):
        print(f"❌ Thiếu file {RAG_FILE}. Hãy chạy tests/long_term_evaluation/test_long_term.py trước.")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f: gt_data = json.load(f)
    with open(RAG_FILE, "r", encoding="utf-8") as f: rag_results = json.load(f)
    with open(GPT_FILE, "r", encoding="utf-8") as f: gpt_results = json.load(f)

    # Kiểm tra xem người dùng đã điền data chưa
    if "DÁN CÂU TRẢ LỜI" in json.dumps(gpt_results):
        print(f"⚠️ Bạn chưa điền đầy đủ câu trả lời vào {GPT_FILE}. Hãy điền rồi chạy lại.")
        return

    print("="*80)
    print("📈 SO SÁNH ĐỘ CHÍNH XÁC (ACCURACY): LEGAL-RAG VS CHATGPT")
    print("="*80)

    total_turns = 0
    rag_passed = 0
    gpt_passed = 0
    comparison_details = []

    for gt_session in gt_data:
        s_name = gt_session["session_name"]
        print(f"\nEvaluating Session: {s_name}")
        
        rag_sess = next((s for s in rag_results if s["session_name"] == s_name), None)
        gpt_sess = next((s for s in gpt_results if s["session_name"] == s_name), None)

        if not rag_sess or not gpt_sess:
            print(f"   ⚠️ Thiếu dữ liệu đối soát cho session: {s_name}")
            continue

        history_gpt = []

        for turn in gt_session["turns"]:
            total_turns += 1
            query = turn["query"]
            gt_ans = turn["expected_answer"]

            # 1. Lấy kết quả Legal-RAG (đã có judge từ test_long_term.py)
            rag_turn = next((t for t in rag_sess["turns"] if t["turn"] == turn["turn"]), None)
            rag_ans = rag_turn["answer"] if rag_turn else "N/A"
            rag_judge = rag_turn["judge"] if rag_turn else "❌ KHÔNG ĐẠT"
            if "✅ ĐẠT" in rag_judge: rag_passed += 1

            # 2. Chấm điểm cho ChatGPT
            gpt_turn = next((t for t in gpt_sess["turns"] if t["turn"] == turn["turn"]), None)
            gpt_ans = gpt_turn["answer"] if gpt_turn else "N/A"
            
            print(f"   Turn {turn['turn']}: Judging ChatGPT vs Ground Truth...", end="\r")
            gpt_judge = evaluate_answer_vs_ground_truth(history_gpt, query, gt_ans, gpt_ans, "ChatGPT")
            if "✅ ĐẠT" in gpt_judge: gpt_passed += 1

            # In kết quả so sánh trực diện
            rag_status = "✅" if "✅ ĐẠT" in rag_judge else "❌"
            gpt_status = "✅" if "✅ ĐẠT" in gpt_judge else "❌"
            print(f"   Turn {turn['turn']}: RAG [{rag_status}] | ChatGPT [{gpt_status}]")

            history_gpt.append({"role": "user", "content": query})
            history_gpt.append({"role": "assistant", "content": gpt_ans})

            comparison_details.append({
                "session": s_name,
                "turn": turn["turn"],
                "query": query,
                "legal_rag": {"status": rag_judge, "answer": rag_ans},
                "chatgpt": {"status": gpt_judge, "answer": gpt_ans}
            })

    # Summary
    print("\n" + "="*80)
    print("📊 TỔNG KẾT ĐỐI CHIẾU ĐỘ CHÍNH XÁC")
    print("="*80)
    print(f"Tổng số lượt kiểm thử: {total_turns}")
    print(f"- Accuracy Legal-RAG: {(rag_passed/total_turns)*100:.2f}% ({rag_passed}/{total_turns})")
    print(f"- Accuracy ChatGPT  : {(gpt_passed/total_turns)*100:.2f}% ({gpt_passed}/{total_turns})")
    print("="*80)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(comparison_details, f, ensure_ascii=False, indent=4)
    print(f"👉 Báo cáo chi tiết lưu tại: {REPORT_FILE}")

async def main():
    if not os.path.exists(GPT_FILE):
        create_chatgpt_template()
    else:
        await run_comparison()

if __name__ == "__main__":
    asyncio.run(main())
