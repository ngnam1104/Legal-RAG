import sys
import os
import json
import asyncio
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.models.llm_factory import chat_completion
from backend.utils.text_utils import strip_thinking_tags

DATA_FILE = "tests/long_term_evaluation/long_term_test_data.json"
RAG_FILE = "tests/long_term_evaluation/legal_rag_answers.json"
GPT_FILE = "tests/long_term_evaluation/chatgpt_answers.json"
GEMINI_FILE = "tests/long_term_evaluation/gemini_answers.json"
REPORT_FILE = "tests/long_term_evaluation/accuracy_comparison_report.json"

def create_templates():
    """Tạo file JSON mẫu để người dùng điền câu trả lời từ ChatGPT và Gemini UI."""
    if not os.path.exists(DATA_FILE):
        print(f"❌ Không tìm thấy {DATA_FILE}. Hãy chạy utils/generate_test_dataset.py trước.")
        return False

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    gpt_template = []
    gemini_template = []
    for sess in data:
        gpt_sess = {
            "session_name": sess["session_name"],
            "document_id": sess["document_id"],
            "turns": []
        }
        gemini_sess = {
            "session_name": sess["session_name"],
            "document_id": sess["document_id"],
            "turns": []
        }
        for turn in sess["turns"]:
            t_gpt = turn.copy()
            t_gemini = turn.copy()
            if "expected_answer" in t_gpt:
                del t_gpt["expected_answer"]
                del t_gemini["expected_answer"]
            if "citation" in t_gpt:
                del t_gpt["citation"]
                del t_gemini["citation"]
            
            t_gpt["answer"] = "DÁN CÂU TRẢ LỜI CỦA CHATGPT VÀO ĐÂY"
            t_gemini["answer"] = "DÁN CÂU TRẢ LỜI CỦA GEMINI VÀO ĐÂY"
            
            gpt_sess["turns"].append(t_gpt)
            gemini_sess["turns"].append(t_gemini)
            
        gpt_template.append(gpt_sess)
        gemini_template.append(gemini_sess)

    if not os.path.exists(GPT_FILE):
        with open(GPT_FILE, "w", encoding="utf-8") as f:
            json.dump(gpt_template, f, ensure_ascii=False, indent=4)
        print(f"✅ Đã tạo file mẫu ChatGPT tại: {GPT_FILE}")

    if not os.path.exists(GEMINI_FILE):
        with open(GEMINI_FILE, "w", encoding="utf-8") as f:
            json.dump(gemini_template, f, ensure_ascii=False, indent=4)
        print(f"✅ Đã tạo file mẫu Gemini tại: {GEMINI_FILE}")
    
    print("💡 Hướng dẫn: Mở các file trên, dán câu trả lời từ ChatGPT/Gemini vào rồi chạy lại script này để so sánh.")
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

TIÊU CHÍ (Nới lỏng):
- ✅ ĐẠT: Đúng bản chất pháp lý và giữ được ngữ cảnh multi-turn. KHÔNG BẮT BUỘC phải có số Điều/Khoản chính xác nếu nội dung pháp lý đã đúng.
- ✅ ĐẠT: Nếu {system_name} diễn giải bằng ngôn ngữ tự nhiên nhưng khớp với ý chính của ĐÁP ÁN MẪU.
- ❌ KHÔNG ĐẠT: Sai lệch hoàn toàn về bản chất pháp lý, đưa ra thông tin trái luật, hoặc bị mất hoàn toàn ngữ cảnh hội thoại.

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
    with open(GEMINI_FILE, "r", encoding="utf-8") as f: gemini_results = json.load(f)

    # Kiểm tra xem người dùng đã điền data chưa
    if "DÁN CÂU TRẢ LỜI" in json.dumps(gpt_results) or "DÁN CÂU TRẢ LỜI" in json.dumps(gemini_results):
        print(f"⚠️ Bạn chưa điền đầy đủ câu trả lời vào {GPT_FILE} hoặc {GEMINI_FILE}. Hãy điền rồi chạy lại.")
        return

    print("="*80)
    print("📈 SO SÁNH ĐỘ CHÍNH XÁC (ACCURACY): LEGAL-RAG VS CHATGPT VS GEMINI")
    print("="*80)

    total_turns = 0
    rag_passed = 0
    gpt_passed = 0
    gemini_passed = 0
    comparison_details = []

    for gt_session in gt_data:
        s_name = gt_session["session_name"]
        print(f"\nEvaluating Session: {s_name}")
        
        rag_sess = next((s for s in rag_results if s["session_name"] == s_name), None)
        gpt_sess = next((s for s in gpt_results if s["session_name"] == s_name), None)
        gemini_sess = next((s for s in gemini_results if s["session_name"] == s_name), None)

        if not rag_sess or not gpt_sess or not gemini_sess:
            print(f"   ⚠️ Thiếu dữ liệu đối soát cho session: {s_name}")
            continue

        history_gpt = []
        history_gemini = []

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
            
            # 3. Chấm điểm cho Gemini
            gemini_turn = next((t for t in gemini_sess["turns"] if t["turn"] == turn["turn"]), None)
            gemini_ans = gemini_turn["answer"] if gemini_turn else "N/A"
            
            print(f"   Turn {turn['turn']}: Judging ChatGPT vs Ground Truth...", end="\r")
            gpt_judge = evaluate_answer_vs_ground_truth(history_gpt, query, gt_ans, gpt_ans, "ChatGPT")
            if "✅ ĐẠT" in gpt_judge: gpt_passed += 1

            print(f"   Turn {turn['turn']}: Judging Gemini vs Ground Truth... ", end="\r")
            gemini_judge = evaluate_answer_vs_ground_truth(history_gemini, query, gt_ans, gemini_ans, "Gemini")
            if "✅ ĐẠT" in gemini_judge: gemini_passed += 1

            # In kết quả so sánh trực diện
            rag_status = "✅" if "✅ ĐẠT" in rag_judge else "❌"
            gpt_status = "✅" if "✅ ĐẠT" in gpt_judge else "❌"
            gemini_status = "✅" if "✅ ĐẠT" in gemini_judge else "❌"
            print(f"   Turn {turn['turn']}: RAG [{rag_status}] | ChatGPT [{gpt_status}] | Gemini [{gemini_status}]")

            history_gpt.append({"role": "user", "content": query})
            history_gpt.append({"role": "assistant", "content": gpt_ans})
            
            history_gemini.append({"role": "user", "content": query})
            history_gemini.append({"role": "assistant", "content": gemini_ans})

            comparison_details.append({
                "session": s_name,
                "turn": turn["turn"],
                "query": query,
                "legal_rag": {"status": rag_judge, "answer": rag_ans},
                "chatgpt": {"status": gpt_judge, "answer": gpt_ans},
                "gemini": {"status": gemini_judge, "answer": gemini_ans}
            })

    # Summary
    print("\n" + "="*80)
    print("📊 TỔNG KẾT ĐỐI CHIẾU ĐỘ CHÍNH XÁC")
    print("="*80)
    print(f"Tổng số lượt kiểm thử: {total_turns}")
    if total_turns > 0:
        print(f"- Accuracy Legal-RAG: {(rag_passed/total_turns)*100:.2f}% ({rag_passed}/{total_turns})")
        print(f"- Accuracy ChatGPT  : {(gpt_passed/total_turns)*100:.2f}% ({gpt_passed}/{total_turns})")
        print(f"- Accuracy Gemini   : {(gemini_passed/total_turns)*100:.2f}% ({gemini_passed}/{total_turns})")
    print("="*80)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(comparison_details, f, ensure_ascii=False, indent=4)
    print(f"👉 Báo cáo chi tiết lưu tại: {REPORT_FILE}")

async def main():
    if not os.path.exists(GPT_FILE) or not os.path.exists(GEMINI_FILE):
        create_templates()
    else:
        await run_comparison()

if __name__ == "__main__":
    asyncio.run(main())
