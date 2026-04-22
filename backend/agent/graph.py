import operator
import time
import asyncio
import functools
from typing import Annotated, Dict, List, Any, Optional, TypedDict, Union
import os

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

import logging
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
from backend.agent.utils.utils_general_chat import execute_general_chat

logger = logging.getLogger("graph")



# --- HELPERS (DELEGATE FACTORY) ---

def get_strategy(mode: str) -> BaseRAGStrategy:
    mode_upper = str(mode).upper() if mode else ""
    if mode_upper == RouteIntent.LEGAL_QA:
        return LegalQAStrategy()
    elif mode_upper == RouteIntent.SECTOR_SEARCH:
        return SectorSearchStrategy()
    elif mode_upper == RouteIntent.CONFLICT_ANALYZER:
        return ConflictAnalyzerStrategy()
    else:
        return LegalQAStrategy() # Fallback

def node_timer(name: str):
    """Decorator to measure and log node execution time.
    
    IMPORTANT: Wrapper MUST be synchronous (def, not async def) so that
    LangGraph accepts both graph.invoke() and graph.ainvoke().
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state: AgentState):
            t_start = time.perf_counter()
            logger.info(f"  [LangGraph] Node: {name} starting...")
            
            result = func(state)
            
            duration = time.perf_counter() - t_start
            logger.info(f"       ⏱️ {name} finished in {duration:.2f}s")
            
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
    file_path = state.get("file_path")
    doc_id = os.path.basename(file_path) if file_path else "default"
    
    from backend.agent.chat_engine import rag_engine
    
    chunks = rag_engine.memory.get_temp_chunks(session_id, document_id=doc_id)
    if chunks:
        logger.debug(f"       📚 Cache hit for doc: {doc_id}")
        return {"file_chunks": chunks}
        
    if file_path:
        try:
            chunks = parser.parse_and_chunk(file_path)
            rag_engine.memory.set_temp_chunks(session_id, chunks, document_id=doc_id)
            return {"file_chunks": chunks}
        except Exception as e:
            logger.error(f"       [!] Error parsing file: {e}")
            pass
    return {"file_chunks": []}

@node_timer("Condense & Route")
def node_condense(state: AgentState):
    """Sử dụng SuperRouter: Vừa viết lại, vừa lấy Intent, vừa trích xuất Filters."""
    history = state.get("history", [])
    query = state["query"]
    
    # 1 Lần gọi LLM duy nhất giải quyết 3 tác vụ
    detected_mode, standalone_query, hypo_query, router_filters = router.super_route_query(
        query, 
        history=history, 
        conv_state=state.get("conversation_state"),
        has_file_attachment=bool(state.get("file_path")), 
        llm_preset=state.get("llm_preset")
    )
    
    if detected_mode == RouteIntent.GENERAL_CHAT:
        hypo_query = query # Trả lại nguyên gốc nếu hỏi thăm thông thường
        standalone_query = query
         
    return {
        "standalone_query": standalone_query,
        "condensed_query": hypo_query, 
        "detected_mode": detected_mode, 
        "router_filters": router_filters
    }


@node_timer("Detect Mode Only")
def node_detect_mode_only(state: AgentState):
    """Sử dụng SuperRouter khi KHÔNG có history."""
    query = state["query"]
    
    # Kể cả không lịch sử, SuperRouter vẫn rất tốt để sinh HyDE và Filters
    detected_mode, standalone_query, hypo_query, router_filters = router.super_route_query(
        query, 
        history=[], 
        conv_state=state.get("conversation_state"),
        has_file_attachment=bool(state.get("file_path")), 
        llm_preset=state.get("llm_preset")
    )
    
    if detected_mode == RouteIntent.GENERAL_CHAT:
        hypo_query = query
        standalone_query = query
    
    return {
        "standalone_query": standalone_query,
        "condensed_query": hypo_query, 
        "detected_mode": detected_mode, 
        "router_filters": router_filters
    }

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
    
    if state.get("retry_count", 0) > 0:
        logger.info(f"       🧠 [Understand] Nới lỏng bộ lọc do Retry {state.get('retry_count')}")
        state["router_filters"] = {}
        
    # Fix 2.2: retry_count is now managed by router_grade, not here.
    # This prevents premature increment on first pass.
    result = strategy.understand(state)
    return result

@node_timer("2. Retrieve + Graph Expand")
def node_retrieve(state: AgentState):
    mode = state.get("detected_mode") or state.get("mode")
    strategy = get_strategy(mode)
    return strategy.retrieve(state)

@node_timer("4. Generate")
def node_generate(state: AgentState):
    mode = state.get("detected_mode") or state.get("mode")
    strategy = get_strategy(mode)
    
    # generate handles final formatting without reflect
    result = strategy.generate(state)
    
    # Backward compatibility mapping
    answer = result.get("final_response", result.get("draft_response", ""))
    
    # Update for current state
    result["answer"] = answer
    return result


# --- ROUTERS ---

def router_dispatcher(state: AgentState):
    """Điều hướng sau khi condense xong."""
    mode = state.get("detected_mode") or state.get("mode")
    if mode == RouteIntent.GENERAL_CHAT:
        logger.info(f"       🔀 Dispatcher: Routing to 'general'")
        return "general"
    logger.info(f"       🔀 Dispatcher: Routing to 'rag_pipeline' ({mode})")
    return "rag_pipeline"

def router_generate(state: AgentState):
    """Điều hướng vòng lặp Batching hoặc kết thúc."""
    pending = state.get("pending_tasks", [])
    if pending:
        logger.info(f"       🔄 Generator: {len(pending)} pending tasks remaining. Looping back to 'understand'")
        return "loop_next_batch"
        
    logger.info(f"       🏁 Generator: No pending tasks. Ending workflow")
    return "end"


# --- ASSEMBLE GRAPH ---

# Fix 2.1: Router to bypass condense when no history
def router_preprocess(state: AgentState):
    """Skip condense node if no chat history exists."""
    history = state.get("history", [])
    if history:
        return "condense"
    return "detect_only"

# Fix 2.2: Increment retry_count atomically when grade triggers retry
def node_increment_retry(state: AgentState):
    """Bump retry_count before re-entering understand on CRAG retry."""
    return {"retry_count": state.get("retry_count", 0) + 1}

def node_reset_for_batch(state: AgentState):
    """Reset retry_count and is_sufficient when starting a new batch."""
    return {"retry_count": 0, "is_sufficient": None}


class LegalRAGWorkflow:
    def __init__(self):
        self.workflow = StateGraph(AgentState)
        self._add_nodes()
        self._add_edges()

    def _add_nodes(self):
        self.workflow.add_node("preprocess", node_preprocess)
        self.workflow.add_node("condense", node_condense)
        self.workflow.add_node("detect_only", node_detect_mode_only)  # Fix 2.1
        self.workflow.add_node("general", node_general_chat)
        self.workflow.add_node("understand", node_understand)
        self.workflow.add_node("retrieve", node_retrieve)
        self.workflow.add_node("generate", node_generate)
        self.workflow.add_node("bump_retry", node_increment_retry)  # Fix 2.2
        self.workflow.add_node("reset_for_batch", node_reset_for_batch)

    def _add_edges(self):
        self.workflow.add_edge(START, "preprocess")
        
        # Fix 2.1: Conditional bypass of condense when no history
        self.workflow.add_conditional_edges(
            "preprocess",
            router_preprocess,
            {
                "condense": "condense",
                "detect_only": "detect_only"
            }
        )
        
        # Both condense and detect_only feed into dispatcher
        self.workflow.add_conditional_edges(
            "condense",
            router_dispatcher,
            {
                "general": "general",
                "rag_pipeline": "understand"
            }
        )
        self.workflow.add_conditional_edges(
            "detect_only",
            router_dispatcher,
            {
                "general": "general",
                "rag_pipeline": "understand"
            }
        )
        self.workflow.add_edge("general", END)

        # RAG Pipeline Flow: Understand → Retrieve(+Graph) → Generate
        self.workflow.add_edge("understand", "retrieve")
        self.workflow.add_edge("retrieve", "generate")

        self.workflow.add_conditional_edges(
            "generate",
            router_generate,
            {
                "end": END,
                "loop_next_batch": "reset_for_batch"
            }
        )

        self.workflow.add_edge("reset_for_batch", "understand")

    def build(self):
        return self.workflow.compile()

# Backward compatibility for chat_engine.py
workflow_instance = LegalRAGWorkflow()
app = workflow_instance.build()

