import operator
import time
import asyncio
import functools
from typing import Annotated, Dict, List, Any, Optional, TypedDict, Union

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from backend.llm.factory import chat_completion
from backend.agent.query_router import RouteIntent, router
from backend.utils.document_parser import parser
from backend.config import settings

# Tích hợp hệ thống Framework mới
from backend.agent.state import AgentState
from backend.agent.strategies.base import BaseRAGStrategy
from backend.agent.strategies.legal_qa import LegalQAStrategy
from backend.agent.strategies.sector_search import SectorSearchStrategy
from backend.agent.strategies.conflict_analyzer import ConflictAnalyzerStrategy
from backend.agent.utils_general_chat import execute_general_chat

# --- PROMPTS ---
CONDENSE_PROMPT = """Bạn là chuyên gia phân tích ngữ cảnh (Context Analyzer) cho một Trợ lý Pháp lý AI.
Nhiệm vụ của bạn là đọc LỊCH SỬ HỘI THOẠI và CÂU HỎI MỚI để tạo ra một "Câu hỏi độc lập" (Standalone Query) hoàn chỉnh.

QUY TẮC BẮT BUỘC:
1. TRÁNH MẤT MÁT NGỮ CẢNH: Nếu câu hỏi mới ngầm ám chỉ đến một văn bản, đạo luật, nghị định, hoặc chủ đề đã được đề cập trong lịch sử, BẠN PHẢI đưa tên/số hiệu của văn bản/chủ đề đó vào câu viết lại.
2. PHỤC HỒI ĐẠI TỪ: Thay thế hoàn toàn các từ "nó", "điều đó", "luật này", "văn bản trên", "khoản đó" bằng danh từ cụ thể.
3. CHUYỂN HƯỚNG CHỦ ĐỀ (TOPIC SHIFT): Nếu câu hỏi mới hoàn toàn không liên quan đến lịch sử hội thoại, HÃY BỎ QUA lịch sử và chỉ làm rõ câu hỏi mới.
4. ĐỊNH DẠNG ĐẦU RA: TUYỆT ĐỐI KHÔNG giải thích, KHÔNG trả lời câu hỏi, KHÔNG dùng từ ngữ giao tiếp (như "Câu hỏi là:", "Dạ đây là..."). CHỈ TRẢ VỀ DUY NHẤT một chuỗi văn bản là câu hỏi đã được viết lại.

--- VÍ DỤ MẪU ---
Lịch sử:
User: Tìm hiểu về Luật Đất đai 2024.
AI: Luật Đất đai 2024 có nhiều điểm mới về bồi thường, hỗ trợ tái định cư...
Câu hỏi mới: Mức bồi thường khi thu hồi đất nông nghiệp theo nó là bao nhiêu?
Viết lại: Mức bồi thường khi thu hồi đất nông nghiệp theo Luật Đất đai 2024 là bao nhiêu?

Lịch sử:
User: Điều kiện để thành lập công ty TNHH 1 thành viên?
AI: Cần có đủ vốn, trụ sở, người đại diện pháp luật...
Câu hỏi mới: Thế thủ tục xin giấy phép an toàn vệ sinh thực phẩm thì sao?
Viết lại: Thủ tục xin giấy phép an toàn vệ sinh thực phẩm như thế nào? (Lưu ý: Chủ đề đã thay đổi, không kế thừa lịch sử).

Lịch sử:
User: Tôi đang đọc Nghị định 100/2019 về giao thông.
AI: Nghị định này quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ và đường sắt.
Câu hỏi mới: Khoản 3 điều 5 phạt bao nhiêu?
Viết lại: Khoản 3 Điều 5 của Nghị định 100/2019/NĐ-CP phạt bao nhiêu tiền?
----------------

LỊCH SỬ HỘI THOẠI GẦN NHẤT:
{history}

CÂU HỎI MỚI NHẤT CỦA NGƯỜI DÙNG:
{query}

CÂU HỎI ĐỘC LẬP ĐÃ VIẾT LẠI (Chỉ xuất kết quả, không giải thích):"""

# --- HELPERS (DELEGATE FACTORY) ---

def get_strategy(mode: str) -> BaseRAGStrategy:
    if mode == RouteIntent.LEGAL_QA:
        return LegalQAStrategy()
    elif mode == RouteIntent.SECTOR_SEARCH:
        return SectorSearchStrategy()
    elif mode == RouteIntent.CONFLICT_ANALYZER:
        return ConflictAnalyzerStrategy()
    else:
        return LegalQAStrategy() # Fallback

def node_timer(name: str):
    """Decorator to measure and log node execution time."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(state: AgentState):
            t_start = time.perf_counter()
            print(f"  [LangGraph] Node: {name} starting...")
            
            if asyncio.iscoroutinefunction(func):
                result = await func(state)
            else:
                result = func(state)
            
            duration = time.perf_counter() - t_start
            print(f"       ⏱️ {name} finished in {duration:.2f}s")
            
            if result is None: result = {}
            if "metrics" not in result:
                result["metrics"] = {}
            result["metrics"][f"{name}_time"] = duration
            return result
        return wrapper
    return decorator


# --- SYSTEM PRE-PROCESSING NODES ---

@node_timer("Preprocess Memory/Files")
def node_preprocess(state: AgentState):
    """Tải dữ liệu từ RAM hoặc Disk trước khi vào Pipeline."""
    session_id = state.get("session_id")
    from backend.agent.chat_engine import rag_engine
    
    chunks = rag_engine.memory.get_temp_chunks(session_id)
    if chunks:
        return {"file_chunks": chunks}
        
    file_path = state.get("file_path")
    if file_path:
        try:
            chunks = parser.parse_and_chunk(file_path)
            rag_engine.memory.set_temp_chunks(session_id, chunks)
            return {"file_chunks": chunks}
        except:
            pass
    return {"file_chunks": []}

@node_timer("Condense Query")
def node_condense(state: AgentState):
    """Làm rõ câu hỏi dựa trên lịch sử hội thoại."""
    history = state.get("history", [])
    query = state["query"]
    
    if not history:
        condensed = query
    else:
        # Lấy tối đa 10 lượt tương tác gần nhất (5 lượt trao đổi)
        history_lines = []
        for m in history[-10:]:
            role = "User" if m["role"] == "user" else "AI"
            # Lược bỏ các khoảng trắng thừa, giới hạn 1000 ký tự mỗi tin nhắn
            content_snippet = " ".join(m["content"].split())[:1000] 
            history_lines.append(f"{role}: {content_snippet}")
            
        prompt = CONDENSE_PROMPT.format(history="\n".join(history_lines), query=query)
        condensed = chat_completion(
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.0, 
            model=settings.LLM_ROUTING_MODEL, 
            llm_preset=state.get("llm_preset")
        )
        condensed = (condensed or query).strip().strip('"').strip("'")
            
    # Auto Intent Detection
    detected_mode = state.get("mode")
    router_filters = {}
    if detected_mode == RouteIntent.AUTO:
         detected_mode, router_filters = router.route_query(condensed, has_file_attachment=bool(state.get("file_path")))
         
    return {"condensed_query": condensed, "detected_mode": detected_mode, "router_filters": router_filters}

@node_timer("General Chat")
def node_general_chat(state: AgentState):
    """Bypass pipeline cho câu hỏi thường."""
    answer = execute_general_chat(state["query"], history=state.get("history", []), file_chunks=state.get("file_chunks", []), llm_preset=state.get("llm_preset"))
    return {"answer": answer, "draft_response": answer, "final_response": answer}


# --- UNIVERSAL 5-STAGE RAG NODES (Vector-to-Graph) ---

@node_timer("1. Understand")
def node_understand(state: AgentState):
    mode = state.get("detected_mode") or state.get("mode")
    strategy = get_strategy(mode)
    
    # Track retry to avoid infinite loop
    retry_count = state.get("retry_count", 0)
    result = strategy.understand(state)
    result["retry_count"] = retry_count + 1
    return result

@node_timer("2. Retrieve + Graph Expand")
def node_retrieve(state: AgentState):
    mode = state.get("detected_mode") or state.get("mode")
    strategy = get_strategy(mode)
    return strategy.retrieve(state)

@node_timer("3. Grade")
def node_grade(state: AgentState):
    mode = state.get("detected_mode") or state.get("mode")
    strategy = get_strategy(mode)
    return strategy.grade(state)

@node_timer("4. Generate")
def node_generate(state: AgentState):
    mode = state.get("detected_mode") or state.get("mode")
    strategy = get_strategy(mode)
    return strategy.generate(state)

@node_timer("5. Reflect")
def node_reflect(state: AgentState):
    mode = state.get("detected_mode") or state.get("mode")
    strategy = get_strategy(mode)
    
    # Đồng bộ output universal sang legacy output cho chat_engine
    result = strategy.reflect(state)
    
    # Backward compatibility mapping
    answer = result.get("final_response", state.get("draft_response", ""))
    
    # Update for current state
    result["answer"] = answer
    return result


# --- ROUTERS ---

def router_dispatcher(state: AgentState):
    """Điều hướng sau khi condense xong."""
    mode = state.get("detected_mode") or state.get("mode")
    if mode == RouteIntent.GENERAL_CHAT:
        return "general"
    return "rag_pipeline"

def router_grade(state: AgentState):
    """Grade -> Generate hoặc quay lại Understand để search từ khóa khác."""
    if state.get("is_sufficient", False):
        return "generate"
    # Giới hạn retry
    if state.get("retry_count", 0) >= 2:
        return "generate" # Dù fail vẫn ráng giải thích "Tôi không tìm thấy"
    return "retry_understand"

def router_reflect(state: AgentState):
    """Điều hướng vòng lặp Batching hoặc kết thúc."""
    # Xử lý Batch Processing (Ví dụ Conflict Analyzer)
    pending = state.get("pending_tasks", [])
    if pending and state.get("pass_flag") is False:
        return "loop_next_batch"
        
    # Legal QA đã tự sửa lỗi (self-correction) ngay bên trong node reflect,
    # thay vì loop lại generate và gây vòng lặp vô hạn.
    return "end"


# --- ASSEMBLE GRAPH ---

workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("preprocess", node_preprocess)
workflow.add_node("condense", node_condense)
workflow.add_node("general", node_general_chat)

workflow.add_node("understand", node_understand)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("grade", node_grade)
workflow.add_node("generate", node_generate)
workflow.add_node("reflect", node_reflect)

# Edges
workflow.add_edge(START, "preprocess")
workflow.add_edge("preprocess", "condense")
workflow.add_conditional_edges(
    "condense",
    router_dispatcher,
    {
        "general": "general",
        "rag_pipeline": "understand"
    }
)
workflow.add_edge("general", END)

# RAG Pipeline Flow: Understand → Retrieve(+Graph) → Grade → Generate → Reflect
workflow.add_edge("understand", "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    router_grade,
    {
        "generate": "generate",
        "retry_understand": "understand"
    }
)

workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    router_reflect,
    {
        "end": END,
        "retry_generate": "generate",
        "loop_next_batch": "understand"
    }
)

# Compile
app = workflow.compile()
