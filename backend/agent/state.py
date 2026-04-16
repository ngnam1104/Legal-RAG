import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated

class AgentState(TypedDict):
    # --- Framework Core (Inputs) ---
    query: str
    session_id: str
    mode: str
    file_path: Optional[str]
    top_k: int
    use_reflection: bool
    use_rerank: bool
    llm_preset: Optional[str]  # e.g., 'groq_8b', 'groq_70b', 'gemini', 'ollama'
    
    # --- Trace & Memory ---
    history: List[Dict[str, str]]
    condensed_query: str
    file_chunks: List[Dict[str, Any]]
    detected_mode: Optional[str]
    router_filters: Optional[Dict[str, Any]]
    metrics: Annotated[Dict[str, float], operator.ior]
    
    # --- Legacy Variables (gradually deprecate) ---
    rewritten_query: str
    filters: Dict[str, Any]
    hits: List[Dict[str, Any]]
    context: str
    supplemental_context: str
    answer: str
    references: Annotated[List[Dict[str, Any]], operator.add]
    reflection_result: Dict[str, Any]
    retry_count: int
    grade_retry_count: int
    is_graded_pass: bool
    
    # --- Universal 5-Stage RAG States (New Framework) ---
    # 1. Understand (Query Rewrite / Extraction)
    rewritten_queries: List[str]      # Hỗ trợ search đa luồng / đa mệnh đề
    metadata_filters: Dict[str, Any]  # Tham số filter Qdrant (vd: {"legal_type": "Luật", "is_appendix": False})
    
    # 2. Retrieve (Vector/Hybrid DB Fetch + Graph Expansion)
    raw_hits: List[Dict[str, Any]]    # Kết quả thô từ Qdrant
    graph_context: Dict[str, Any]     # Kết quả từ Neo4j (lateral_docs, document_toc, sibling_texts, time_travel)
    
    # 3. Grade (Kiểm định)
    filtered_context: str             # Context sau khi loại bỏ nhiễu
    is_sufficient: bool               # cờ hiệu cho router: đủ context để Generate?
    
    # 4. Generate
    draft_response: str
    
    # 5. Reflect
    final_response: str
    pass_flag: bool
    feedback: str
    
    # --- Iterators (Batching/MapReduce) ---
    pending_tasks: List[Any]          # Hàng đợi, VD: danh sách Claims chưa xử lý
    completed_results: List[Any]      # Lưu trữ kết quả sau từng vòng loop
