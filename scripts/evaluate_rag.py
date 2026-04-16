import os
import sys
import time
import json
import pandas as pd
from tabulate import tabulate
from datetime import datetime

# Đảm bảo import được backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.graph import workflow
from backend.llm.factory import chat_completion
from backend.config import settings

# =====================================================================
# 1. GOLDEN DATASET
# =====================================================================
GOLDEN_DATASET = [
    {
        "id": "TC_01",
        "description": "LEGAL_QA - Cần tìm Điều khoản cụ thể và xem cả Mục lục",
        "query": "Thủ tục đăng ký khai sinh cho người đi bồi dưỡng ở nước ngoài quy định ra sao? Cho xin mục lục văn bản.",
        "expected_mode": "LEGAL_QA",
        "expected_keywords_in_context": ["<muc_luc_van_ban>", "khai sinh"]
    },
    {
        "id": "TC_02",
        "description": "SECTOR_SEARCH - Tìm báo cáo tổng quan diện rộng theo lĩnh vực",
        "query": "Tổng hợp các văn bản quy phạm pháp luật thuộc lĩnh vực Y tế ban hành năm 2024",
        "expected_mode": "SECTOR_SEARCH",
        "expected_keywords_in_context": ["Y tế"]
    },
    {
        "id": "TC_03",
        "description": "CONFLICT_ANALYZER - Phân tích hiệu lực và luật sửa đổi",
        "query": "Quy định về thuế giá trị gia tăng ở điều 5 Luật Thuế có bị sửa đổi gần đây không?",
        "expected_mode": "CONFLICT_ANALYZER",
        "expected_keywords_in_context": ["AMENDS", "REPLACES", "bị tác động bởi", "Nội dung cũ", "Nội dung hiện hành"]
    },
    {
        "id": "TC_04",
        "description": "UPLOAD_MOCK - Kiểm tra bypass DB để lấy thông tin từ File Upload tạm thời",
        "query": "Quy định bảo mật nào được nêu rõ trong tài liệu đính kèm của tôi?",
        "expected_mode": "LEGAL_QA",
        "expected_keywords_in_context": ["<tai_lieu_tam>", "quy định bảo mật"],
        "mock_file_upload": [
            {
                "chunk_id": "mock_chunk_1",
                "unit_text": "Quy định bảo mật dữ liệu phòng ban IT: Mọi nhân viên phải thay đổi mật khẩu 30 ngày một lần...",
                "metadata": {"source": "upload_file.pdf"}
            }
        ]
    }
]


# =====================================================================
# 2. EXTRACT METRICS (Performance Profiling)
# =====================================================================
def extract_pipeline_metrics(state_metrics: list) -> dict:
    """Bóc tách các chỉ số đo đạc từ list metrics của state."""
    perf = {
        "Total_Time": 0.0,
        "Understand_Time": 0.0,
        "Retrieve_Neo4j_Time": 0.0,
        "Grade_Time": 0.0,
        "Generate_Time": 0.0
    }
    
    for m in state_metrics:
        node = m.get("node", "")
        t = m.get("time", 0.0)
        perf["Total_Time"] += t
        
        if "1. Understand" in node:
            perf["Understand_Time"] = t
        elif "2. Retrieve" in node:
            perf["Retrieve_Neo4j_Time"] = t
        elif "3. Grade" in node:
            perf["Grade_Time"] = t
        elif "4. Generate" in node:
            perf["Generate_Time"] = t
            
    # Làm tròn để hiển thị đẹp hơn
    return {k: round(v, 3) for k, v in perf.items()}


# =====================================================================
# 3. LLM-AS-A-JUDGE (Quality Evaluation)
# =====================================================================
def evaluate_quality_with_llm(query: str, context: str, answer: str) -> dict:
    """Dùng Routing Model (8B) hoặc Core Model (nhỏ) để chấm điểm Faithfulness & Relevance."""
    prompt = f"""
Bạn là một AI Judge độc lập chuyên chấm điểm hệ thống RAG (Retrieval-Augmented Generation).
Xin hãy chấm điểm (từ 1 đến 5) cho câu trả lời sau dựa trên 2 tiêu chí.
Chỉ trả về JSON hợp lệ.

[Context Excerpt]
{context[:2000]}

[Query]
{query}

[Answer]
{answer[:1000]}

Tiêu chí:
1. "faithfulness" (1-5): Câu trả lời có bám vào Context không? 5 là hoàn toàn dùng thông tin Context, 1 là bịa đặt hoặc lạc đề.
2. "relevance" (1-5): Câu trả lời có đáp ứng đúng và đủ Query không? 5 là cực tốt, 1 là vô dụng.

Trả về JSON ĐÚNG cấu trúc sau (không giải thích thêm):
{{"faithfulness": 4, "relevance": 5, "reason": ""}}
"""
    try:
        # Sử dụng API chat_completion có sẵn, ép trả về temperature 0 cho ổn định
        resp = chat_completion(
            [{"role": "user", "content": prompt}], 
            model=settings.LLM_ROUTING_MODEL, # Model nhẹ để tiết kiệm chi phí
            temperature=0.0
        )
        
        # Parse JSON fallback
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].strip()
            
        data = json.loads(resp)
        return {
            "faithfulness": float(data.get("faithfulness", 0)),
            "relevance": float(data.get("relevance", 0)),
            "judge_reason": data.get("reason", "N/A")[:50]
        }
    except Exception as e:
        print(f"⚠️ [Judge Error]: {e}")
        return {"faithfulness": 0.0, "relevance": 0.0, "judge_reason": "JSON Parse Error"}


# =====================================================================
# 4. KIỂM THỬ ĐẶC THÙ THEO MODE (Mode-Specific Assertions)
# =====================================================================
def assert_mode_specifics(test_case: dict, state: dict) -> list:
    """Truy vấn sâu vào State để xác thực các logic đặc thù (MapReduce, Neo4j, XML Tags)."""
    flags = []
    mode = state.get("detected_mode", "")
    context_str = state.get("filtered_context", "")
    
    if test_case["expected_mode"] == "LEGAL_QA":
        # Check Neo4j Bottom-Up TOC injection
        if "<muc_luc_van_ban>" in context_str:
            flags.append("✅ TOC Injected")
        else:
            flags.append("❌ Missing TOC")
            
    if test_case["expected_mode"] == "SECTOR_SEARCH":
        # Check Neo4j Sector MapReduce
        graph_ctx = state.get("graph_context", {})
        if graph_ctx.get("sector_mapreduce"):
            flags.append("✅ MapReduce Data Present")
        else:
            flags.append("❌ Missing MapReduce")
            
    if test_case["expected_mode"] == "CONFLICT_ANALYZER":
        # Check Neo4j Time-Travel
        meta_filters = state.get("metadata_filters", {})
        deontic_txt = meta_filters.get("deontic_context", "")
        if "CẢNH BÁO HIỆU LỰC" in deontic_txt or "AMENDS" in str(state.get("graph_context")):
            flags.append("✅ Time-Travel/Deontic Detected")
        else:
            flags.append("❌ Missing Time-Travel")
            
    if "mock_file_upload" in test_case:
        # Check Session-scoped Upload XML Taging
        if "<tai_lieu_tam>" in context_str:
            flags.append("✅ RAM Upload Bypassed")
        else:
            flags.append("❌ Missing File Upload Tags")
            
    return flags


# =====================================================================
# 5. KHỞI CHẠY RUNNER VÀ XUẤT BÁO CÁO (Reporting)
# =====================================================================
def run_evaluation():
    print("="*60)
    print("🚀 BẮT ĐẦU: RAG PIPELINE EVALUATION (VECTOR-TO-GRAPH)")
    print("="*60)
    
    results = []
    
    for i, tc in enumerate(GOLDEN_DATASET):
        print(f"\n▶ Đang chạy Test Case {i+1}: {tc['id']}")
        print(f"   Query: {tc['query']}")
        print(f"   Expected Mode: {tc['expected_mode']}")
        
        # Init base state
        state = {
            "query": tc['query'],
            "history": [],
            "mode": "rag_pipeline", 
            "file_chunks": tc.get("mock_file_upload", []),
            "use_rerank": False
        }
        
        try:
            # Chạy LangGraph
            final_state = workflow.invoke(state)
            
            # Extract Metrics
            metrics = extract_pipeline_metrics(final_state.get("metrics", []))
            
            # Trích xuất Answer và Context
            ans = final_state.get("final_response", final_state.get("draft_response", ""))
            
            # Do Conflict Analyzer trả draft_response dưới dạng Array của JSON object, cần xử lý
            if isinstance(ans, list):
                ans = json.dumps(ans, ensure_ascii=False)
                
            ctx = str(final_state.get("filtered_context", ""))
            
            # Check Expected Keyword
            keyword_pass = True
            for kw in tc["expected_keywords_in_context"]:
                # Tìm trong context hoặc extra metadata
                meta_dump = json.dumps(final_state.get("metadata_filters", {}), ensure_ascii=False)
                graph_dump = json.dumps(final_state.get("graph_context", {}), ensure_ascii=False)
                
                if kw not in ctx and kw not in meta_dump and kw not in graph_dump:
                    keyword_pass = False
                    break
                    
            # 3. LLM QA Judge
            judge_score = evaluate_quality_with_llm(tc['query'], ctx, ans)
            
            # 4. Mode-Specific Constraints
            mode_flags = assert_mode_specifics(tc, final_state)
            
            # Lưu Data
            row = {
                "Test_ID": tc["id"],
                "Detected_Mode": final_state.get("detected_mode", "N/A"),
                "Expected_Keywords": "✅ Match" if keyword_pass else "❌ Fail",
                "Faithfulness": judge_score["faithfulness"],
                "Relevance": judge_score["relevance"],
                "Time(s)": metrics["Total_Time"],
                "Neo4j_Time": metrics["Retrieve_Neo4j_Time"],
                "Mode_Asserts": ", ".join(mode_flags),
                "Judge_Reason": judge_score["judge_reason"]
            }
            results.append(row)
            print(f"   ✓ Xong. Total Time: {metrics['Total_Time']}s | Score: Faith ({judge_score['faithfulness']}) Rel ({judge_score['relevance']})")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"   ❌ LỖI FATAL khi chạy Test Case {i+1}: {e}")
            results.append({
                "Test_ID": tc["id"],
                "Detected_Mode": "ERROR",
                "Expected_Keywords": "ERROR",
                "Faithfulness": 0,
                "Relevance": 0,
                "Time(s)": 0,
                "Neo4j_Time": 0,
                "Mode_Asserts": "ERROR",
                "Judge_Reason": str(e)
            })

    # ========================================
    # XUẤT BÁO CÁO (REPORTING)
    # ========================================
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (RAG EVALUATION REPORT)")
    print("="*80)
    
    # In dạng tabulate đẹp mắt ra console
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
    
    # Tính điểm trung bình Performance & Quality
    avg_faith = df["Faithfulness"].mean()
    avg_rel = df["Relevance"].mean()
    avg_time = df["Time(s)"].mean()
    
    print(f"\n📈 TRUNG BÌNH CHẤT LƯỢNG:")
    print(f"  - Faithfulness: {avg_faith:.1f}/5.0")
    print(f"  - Relevance   : {avg_rel:.1f}/5.0")
    print(f"  - Avg Latency : {avg_time:.2f} s/query")
    
    # Lưu ra CSV file (Thư mục scripts/tests/reports nếu có)
    os.makedirs(os.path.join(os.path.dirname(__file__), "tests", "reports"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(os.path.dirname(__file__), "tests", "reports", f"eval_report_{timestamp}.csv")
    
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Tệp báo cáo chi tiết được lưu tại: {report_path}")
    print("="*80)


if __name__ == "__main__":
    # Chỉ chạy eval khi được gọi trực tiếp bằng python evaluate_rag.py
    run_evaluation()
