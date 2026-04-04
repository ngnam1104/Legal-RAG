import operator
import time
import asyncio
from typing import Annotated, Dict, List, Any, Optional, TypedDict, Union

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from backend.llm.factory import chat_completion
from backend.retrieval.hybrid_search import retriever
from backend.agent.query_router import RouteIntent, router
from backend.utils.document_parser import parser

# --- PROMPTS ---
CONDENSE_PROMPT = """Bạn là hệ thống viết lại câu hỏi (Query Condensation) cho hệ thống RAG pháp lý.

LỊCH SỬ HỘI THOẠI GẦN NHẤT (có thể chứa nhiều chủ đề, nhiều chế độ khác nhau):
{history}

CÂU HỎI MỚI NHẤT CỦA NGƯỜI DÙNG:
{query}

NHIỆM VỤ:
- Nếu câu hỏi mới ĐỘC LẬP và đã đầy đủ ngữ cảnh → Trả lại nguyên văn câu hỏi.
- Nếu câu hỏi mới là câu tiếp nối (chứa đại từ mơ hồ: "nó", "điều đó", "văn bản trên", "luật này"..., hoặc thiếu chủ ngữ) → Viết lại thành câu hỏi ĐỘC LẬP, HOÀN CHỈNH bằng cách thay thế đại từ bằng thực thể cụ thể từ lịch sử.

QUY TẮC BẮT BUỘC:
1. GIỮ NGUYÊN tất cả số hiệu văn bản, tên riêng, ngày tháng, địa danh.
2. KHÔNG thêm thông tin mà người dùng chưa đề cập.
3. KHÔNG giải thích, KHÔNG trả lời câu hỏi. CHỈ TRẢ VỀ CÂU HỎI ĐÃ VIẾT LẠI.

CÂU HỎI ĐÃ VIẾT LẠI:"""

# Import refactored functions
from backend.agent.flow_legal_qa import (
    rewrite_legal_query, build_legal_context, reflect_on_answer, 
    grade_documents, transform_query,
    ANSWER_PROMPT, CORRECTION_PROMPT
)
from backend.agent.flow_sector_search import transform_sector_query, strict_filter_docs, sort_and_group_docs, generate_sector_report
from backend.agent.flow_conflict_analyzer import IE_PROMPT, extract_json_conflict, analyze_single_claim
from backend.agent.flow_general_chat import execute_general_chat

# --- STATE DEFINITION ---

class AgentState(TypedDict):
    # Inputs
    query: str
    session_id: str
    mode: str
    file_path: Optional[str]
    top_k: int
    use_reflection: bool
    use_rerank: bool
    
    # Internal State
    history: List[Dict[str, str]]
    condensed_query: str
    file_chunks: List[Dict[str, Any]]
    detected_mode: Optional[str]
    
    # Retrieval
    rewritten_query: str
    filters: Dict[str, Any]
    hits: List[Dict[str, Any]]
    context: str
    
    # Outputs
    answer: str
    references: Annotated[List[Dict[str, Any]], operator.add]
    
    # Control
    reflection_result: Dict[str, Any]
    retry_count: int
    grade_retry_count: int
    is_graded_pass: bool
    metrics: Dict[str, float]

# --- NODES ---

def node_condense(state: AgentState):
    """Viết lại câu hỏi dựa trên lịch sử để có câu truy vấn độc lập."""
    history = state.get("history", [])
    query = state["query"]
    
    if not history:
        return {"condensed_query": query}
    
    # Format history
    history_lines = []
    for msg in history[-4:]:
        role = "Người dùng" if msg["role"] == "user" else "Trợ lý AI"
        content = msg["content"][:100] + "..." if msg["role"] == "assistant" and len(msg["content"]) > 100 else msg["content"]
        history_lines.append(f"{role}: {content}")
        
    prompt = CONDENSE_PROMPT.format(history="\n".join(history_lines), query=query)
    condensed = chat_completion([{"role": "user", "content": prompt}], temperature=0.0).strip()
    
    if not condensed or len(condensed) > len(query) * 3:
        condensed = query
        
    print(f"  [LangGraph] Node: Condense -> {condensed[:50]}...")
    
    # Intent Detection if mode is AUTO
    detected_mode = state.get("mode")
    if detected_mode == RouteIntent.AUTO:
         print(f"  [LangGraph] Attempting Auto-Intent Detection...")
         detected_mode = router.route_query(condensed, has_file_attachment=bool(state.get("file_path")))
         print(f"       ✅ Detected Intent: {detected_mode}")
         
    return {"condensed_query": condensed, "detected_mode": detected_mode}

def node_preprocess_files(state: AgentState):
    """Xử lý file context nếu có. Ưu tiên lấy từ bộ nhớ RAM của session."""
    session_id = state.get("session_id")
    from backend.agent.chat_engine import rag_engine
    
    # 1. Thử lấy từ RAM (đã được parse lúc upload)
    chunks = rag_engine.memory.get_temp_chunks(session_id)
    if chunks:
        print(f"  [LangGraph] Chunks found in Session RAM ({len(chunks)} chunks)")
        return {"file_chunks": chunks}
        
    # 2. Fallback: Parse từ disk nếu có file_path nhưng chưa có trong RAM
    file_path = state.get("file_path")
    if file_path:
        try:
            print(f"  [LangGraph] Parsing file from disk: {file_path}")
            chunks = parser.parse_and_chunk(file_path)
            # Lưu vào RAM để lần sau không phải parse lại
            rag_engine.memory.set_temp_chunks(session_id, chunks)
            return {"file_chunks": chunks}
        except Exception as e:
            print(f"  [LangGraph] File processing error: {e}")
            
    return {"file_chunks": []}

def node_general_chat(state: AgentState):
    """Luồng trò chuyện tự do."""
    print(f"  [LangGraph] Node: General Chat")
    # Sử dụng history từ state để truyền cho LLM
    answer = execute_general_chat(state["query"], history=state.get("history", []))
    # Return condensed_query bằng query gốc để giữ nhất quán state
    return {"answer": answer, "condensed_query": state["query"]}

def node_legal_qa_retrieve(state: AgentState):
    """Bước tìm kiếm cho Legal QA."""
    print(f"  [LangGraph] Node: Legal QA Retrieve")
    query = state.get("condensed_query", state["query"])
    
    if state.get("grade_retry_count", 0) == 0:
        rewrite_data = rewrite_legal_query(query)
        rewritten = rewrite_data.get("hypothetical_answer", query)
        filters = rewrite_data.get("filters", {})
    else:
        rewritten = query
        filters = state.get("filters", {})
    
    # 2. Hybrid Search
    hits = retriever.search(
        query=rewritten, 
        use_rerank=state.get("use_rerank", True),
        legal_type=filters.get("legal_type"),
        doc_number=filters.get("doc_number")
    )
    
    # Fallback if no hits with filters
    if (filters.get("legal_type") or filters.get("doc_number")) and not hits:
        hits = retriever.search(query=rewritten, use_rerank=state.get("use_rerank", True))

    # 3. Phân loại và Ưu tiên References
    final_references = []
    # Thêm gợi ý từ file upload trước
    file_chunks = state.get("file_chunks", [])
    if file_chunks:
        # Nếu file quá lớn, chúng ta có thể thực hiện vector search mini ở đây
        # Để đơn giản và ưu tiên, chúng ta lấy Top-5 từ file hoặc tất cả nếu < 5
        for idx, fc in enumerate(file_chunks[:5]):
            final_references.append({
                "title": f"[Tài liệu tải lên] {fc['metadata'].get('title', 'File')}",
                "article": fc['metadata'].get('article_ref', 'Nội dung'),
                "score": 1.0, # High priority
                "chunk_id": fc.get("chunk_id", f"temp_{idx}"),
                "text_preview": fc.get("chunk_text", "")[:200],
                "document_number": fc['metadata'].get('document_number', ""),
                "url": fc['metadata'].get('url', ""),
                "is_staged": True
            })

    for h in hits:
        final_references.append({
            "title": h.get("title", ""),
            "article": h.get("article_ref", h.get("document_number", "")),
            "score": h.get("score", 0),
            "chunk_id": h.get("chunk_id", ""),
            "text_preview": h.get("text", "")[:200],
            "document_number": h.get("document_number", ""),
            "url": h.get("url", "")
        })

    return {
        "rewritten_query": rewritten,
        "filters": filters,
        "hits": hits,
        "references": final_references
    }

def node_legal_qa_grade(state: AgentState):
    """CrAG: Đánh giá NGỮ CẢNH có chứa câu trả lời không."""
    print(f"  [LangGraph] Node: Legal QA Grade")
    context = build_legal_context(state.get("hits", []), file_chunks=state.get("file_chunks", []))
    is_pass = grade_documents(state["query"], context)
    print(f"       ✅ Grade pass: {is_pass}")
    return {"is_graded_pass": is_pass, "context": context}

def node_legal_qa_transform(state: AgentState):
    """CrAG: Đổi góc độ truy vấn (Query Transform)."""
    print(f"  [LangGraph] Node: Legal QA Transform")
    new_query = transform_query(state["query"])
    print(f"       ✅ New Query: {new_query}")
    return {"condensed_query": new_query, "grade_retry_count": state.get("grade_retry_count", 0) + 1}

def node_legal_qa_generate(state: AgentState):
    """Bước sinh câu trả lời cho Legal QA."""
    print(f"  [LangGraph] Node: Legal QA Generate")
    
    if not state.get("is_graded_pass", False) and state.get("grade_retry_count", 0) >= 2:
         # Refusal
         return {"answer": "Xin lỗi, dữ liệu hiện tại của hệ thống không chứa thông tin hoặc quy định trực tiếp về vấn đề này."}
    
    context = state.get("context", "")
    if not context:
        return {"answer": "Xin lỗi, dữ liệu hiện tại của hệ thống không chứa thông tin hoặc quy định trực tiếp về vấn đề này."}
        
    prompt = ANSWER_PROMPT.format(context=context, query=state["query"])
    # If we are in a retry/correction phase from Reflection
    if state.get("reflection_result") and not state["reflection_result"].get("pass", True):
        prompt = CORRECTION_PROMPT.format(
            feedback=state["reflection_result"].get("feedback", ""),
            context=context,
            query=state["query"]
        )
        
    answer = chat_completion([{"role": "user", "content": prompt}], temperature=0.3)
    return {"answer": answer}

def node_legal_qa_reflect(state: AgentState):
    """Bước kiểm duyệt ảo giác cho Legal QA."""
    if not state.get("use_reflection"):
        return {"reflection_result": {"pass": True}}
        
    # Skip reflection if it's a standard refusal
    if "không chứa thông tin" in state.get("answer", ""):
        return {"reflection_result": {"pass": True}}
        
    print(f"  [LangGraph] Node: Legal QA Reflect")
    res = reflect_on_answer(state["query"], state.get("context", ""), state["answer"])
    return {"reflection_result": res, "retry_count": state.get("retry_count", 0) + 1}


def node_sector_search(state: AgentState):
    """Luồng Tìm kiếm danh mục kết hợp với lọc cứng."""
    print(f"  [LangGraph] Node: Sector Search")
    query = state["condensed_query"]
    
    tf = transform_sector_query(query)
    kw = tf.get("keywords", query)
    fi = tf.get("filters", {})
    
    hits = retriever.search(query=kw, expand_context=False, legal_type=fi.get("legal_type"), doc_number=fi.get("doc_number"))
    filtered_hits = strict_filter_docs(query, hits)
    
    if not filtered_hits:
        return {"answer": "Không tìm thấy văn bản nào liên quan đến yêu cầu của bạn.", "references": []}
        
    docs_context = sort_and_group_docs(filtered_hits, state.get("file_chunks"))
    answer = generate_sector_report(query, docs_context)
    
    references = [{
        "title": h.get("title", ""),
        "article": h.get("article_ref", h.get("document_number", "")),
        "score": h.get("score", 0),
        "chunk_id": h.get("chunk_id", ""),
        "document_number": h.get("document_number", ""),
        "url": h.get("url", "")
    } for h in filtered_hits]
    
    return {"answer": answer, "references": references}

def node_conflict_analyze(state: AgentState):
    """Luồng Phát hiện xung đột (Parallelized)."""
    print(f"  [LangGraph] Node: Conflict Analyze")
    file_chunks = state.get("file_chunks", [])
    if not file_chunks:
        # Fallback to QA if no file
        return {"answer": "Chế độ NÀY yêu cầu tải lên một file hợp đồng/nội quy.", "references": []}

    sample_chunks = file_chunks[:3]
    
    async def run_parallel_analysis():
        all_results = []
        for c in sample_chunks:
            text = c.get("text_to_embed", c.get("unit_text", ""))
            ie_resp = chat_completion([{"role": "user", "content": IE_PROMPT.format(text=text)}], temperature=0.1)
            ie_data = extract_json_conflict(ie_resp)
            metadata = ie_data.get("metadata", {})
            claims = ie_data.get("claims", [])
            
            tasks = [analyze_single_claim(cl, metadata) for cl in claims]
            chunk_results = await asyncio.gather(*tasks)
            all_results.extend(chunk_results)
        return all_results

    # Run async logic in sync node (or use a loop)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(run_parallel_analysis())
    
    # Format report
    md_table = "### ⚠️ Kết Quả Phân Tích Xung Đột Pháp Lý\n\n"
    md_table += "| Mệnh đề Nội quy (Claim) | Phán quyết | Căn cứ Pháp lý | Giải thích chi tiết |\n"
    md_table += "| :--- | :---: | :--- | :--- |\n"
    
    references = []
    seen_ids = set()
    for r in results:
        icon = "❌" if "contradiction" in str(r['label']).lower() else ("✅" if "entailment" in str(r['label']).lower() else "⚪")
        md_table += f"| {r['claim']} | {icon} **{r['label']}** | {r['reference_law']} | {r['conflict_reasoning']} |\n"
        for h in r.get("hits", []):
            if h.get("chunk_id") not in seen_ids:
                references.append({
                    "title": h.get("title", ""),
                    "article": h.get("article_ref", h.get("document_number", "")),
                    "score": h.get("score", 0),
                    "chunk_id": h.get("chunk_id", ""),
                    "document_number": h.get("document_number", ""),
                    "url": h.get("url", "")
                })
                seen_ids.add(h.get("chunk_id"))
                
    return {"answer": md_table, "references": references}

# --- EDGES / ROUTING ---

def router_entry(state: AgentState):
    """Router tại điểm bắt đầu (START)."""
    # Nếu là AUTO, vẫn cần đi qua 'condense' để LLM xử lý ý định dựa trên query đã làm sạch
    if state["mode"] == RouteIntent.GENERAL_CHAT:
        return "general"
    return "condense"

def router_main(state: AgentState):
    # Ưu tiên giá trị đã được detect (hoặc giá trị thủ công từ state)
    mode = state.get("detected_mode") or state.get("mode")
    
    if mode == RouteIntent.SECTOR_SEARCH:
        return "sector"
    elif mode == RouteIntent.CONFLICT_ANALYZER:
        return "conflict"
    elif mode == RouteIntent.GENERAL_CHAT:
        return "general"
    else:
        return "legal_qa"

def router_grade(state: AgentState):
    if state.get("is_graded_pass", False):
        return "generate"
    if state.get("grade_retry_count", 0) >= 2:
        return "fail"
    return "transform"

def router_reflection(state: AgentState):
    res = state.get("reflection_result", {})
    if res.get("pass", True) or state.get("retry_count", 0) >= 2:
        return "end"
    return "retry"

# --- ASSEMBLE GRAPH ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("condense", node_condense)
workflow.add_node("preprocess", node_preprocess_files)
workflow.add_node("general", node_general_chat)

workflow.add_node("legal_qa_retrieve", node_legal_qa_retrieve)
workflow.add_node("legal_qa_grade", node_legal_qa_grade)
workflow.add_node("legal_qa_transform", node_legal_qa_transform)
workflow.add_node("legal_qa_generate", node_legal_qa_generate)
workflow.add_node("legal_qa_reflect", node_legal_qa_reflect)

workflow.add_node("sector", node_sector_search)
workflow.add_node("conflict", node_conflict_analyze)

# Set Entry Point
workflow.add_conditional_edges(
    START,
    router_entry,
    {
        "general": "general",
        "condense": "condense"
    }
)
workflow.add_edge("condense", "preprocess")

# Conditional Router
workflow.add_conditional_edges(
    "preprocess",
    router_main,
    {
        "sector": "sector",
        "conflict": "conflict",
        "general": "general",
        "legal_qa": "legal_qa_retrieve"
    }
)

# Individual Flow Edges
workflow.add_edge("general", END)
workflow.add_edge("sector", END)
workflow.add_edge("conflict", END)

# Legal QA Cycle
workflow.add_edge("legal_qa_retrieve", "legal_qa_grade")

workflow.add_conditional_edges(
    "legal_qa_grade",
    router_grade,
    {
        "generate": "legal_qa_generate",
        "transform": "legal_qa_transform",
        "fail": "legal_qa_generate"
    }
)
workflow.add_edge("legal_qa_transform", "legal_qa_retrieve")

# Reflect Cycle
workflow.add_edge("legal_qa_generate", "legal_qa_reflect")
workflow.add_conditional_edges(
    "legal_qa_reflect",
    router_reflection,
    {
        "end": END,
        "retry": "legal_qa_generate"
    }
)

# Compile
app = workflow.compile()
