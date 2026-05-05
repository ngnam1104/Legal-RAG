import os
import uuid
import shutil
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import json
import logging
from qdrant_client import models
from fastapi.responses import StreamingResponse

# Silence redundant logs
logging.getLogger("httpx").setLevel(logging.WARNING)

from backend.agent.chat_engine import rag_engine
from backend.models.embedder import embedder
from backend.models.reranker import reranker
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- WARMUP STRATEGY ---
    print("\n🔥 [Startup] Warming up local models (Embedder & Reranker)...")
    try:
        from backend.models.embedder import get_embedder
        from backend.models.reranker import reranker
        get_embedder()
        reranker._lazy_load()
        print(f"✅ [Startup] Local models are ready.")
    except Exception as e:
        print(f"⚠️ [Startup] Warmup failed: {e}")
    
    yield
    print("\n💤 [Shutdown] Cleaning up...")

app = FastAPI(
    title="Chatbot VBPL API",
    description="API hệ thống RAG VBPL với FastAPI - Bao gồm Chat, Sessions, Upload.",
    version="3.1.0",
    lifespan=lifespan
)

# Thêm CORS Middleware để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể đổi thành ["http://localhost:3000"] trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================================
# MODELS
# =====================================================================
class ChatRequest(BaseModel):
    session_id: str = None  # Nếu None, tự tạo session mới
    query: str
    mode: str = "LEGAL_CHAT"  # Mặc định: Trợ lý Pháp luật (GraphRAG)
    file_path: Optional[str] = None  # Đường dẫn file tạm nếu cần context riêng
    provider: Optional[str] = None
    model: Optional[str] = None
    llm_preset: Optional[str] = "groq_8b" # Mặc định 8B để tiết kiệm
    top_k: int = 3
    use_reflection: Optional[bool] = None
    use_grading: Optional[bool] = None
    use_rerank: Optional[bool] = None

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None

class UpdateTitleRequest(BaseModel):
    title: str

class IngestRequest(BaseModel):
    session_id: str
    file_id: str
    filename: str

# =====================================================================
# HEALTH
# =====================================================================
@app.get("/health")
def health_check():
    return {"status": "ok"}

# =====================================================================
# CHAT ENDPOINT
# =====================================================================
@app.post("/api/chat")
async def chat_endpoint(request_data: ChatRequest, fastapi_request: Request):
    """
    Endpoint Hỏi đáp API. Tự động tạo session nếu chưa có.
    Hỗ trợ ngắt kết nối (Cancellation): Nếu FE hủy request, BE sẽ dừng xử lý.
    """
    # Tự tạo session nếu client không gửi session_id
    session_id = request_data.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        rag_engine.memory.create_session(session_id=session_id)

    # Đảm bảo session tồn tại trong DB
    if not rag_engine.memory.get_session(session_id):
        rag_engine.memory.create_session(session_id=session_id)

    print(f"\n--- [API] Incoming Chat Request ---")
    print(f"    Session ID: {session_id} | Mode: {request_data.mode}")
    print(f"    Query: {request_data.query[:100]}...")

    # Chuyển đổi tên file từ frontend gửi thành đường dẫn tuyệt đối an toàn
    absolute_file_path = None
    if request_data.file_path:
        filename = os.path.basename(request_data.file_path)
        absolute_file_path = os.path.join(UPLOAD_DIR, filename)

    # LUỒNG STREAMING: Chuyển tiếp sự kiện từ LangGraph tới Client
    async def event_generator():
        try:
            async for event in rag_engine.chat(
                session_id=session_id,
                query=request_data.query,
                mode=request_data.mode,
                file_path=absolute_file_path,
                llm_preset=request_data.llm_preset,
                top_k=request_data.top_k,
                use_reflection=request_data.use_reflection,
                use_grading=request_data.use_grading,
                use_rerank=request_data.use_rerank
            ):
                # Check for disconnect
                if await fastapi_request.is_disconnected():
                    print(f"    ⚠️ [API] Client disconnected during stream.")
                    break
                
                # Format as SSE
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"    ❌ [API] Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =====================================================================
# SESSION MANAGEMENT ENDPOINTS
# =====================================================================
@app.get("/api/sessions")
async def list_sessions():
    """Lấy danh sách tất cả phiên chat (mới nhất trước)."""
    sessions = rag_engine.memory.list_sessions(limit=50)
    return {"sessions": sessions}


@app.post("/api/sessions")
async def create_session(request: CreateSessionRequest = None):
    """Tạo phiên chat mới."""
    title = request.title if request else None
    session_id = rag_engine.memory.create_session(title=title)
    return {"session_id": session_id, "title": title or "Phiên chat mới"}


@app.put("/api/sessions/{session_id}/title")
async def update_session_title(session_id: str, request: UpdateTitleRequest):
    """Cập nhật tiêu đề phiên chat."""
    session = rag_engine.memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    rag_engine.memory.update_session_title(session_id, request.title)
    return {"message": "Title updated", "session_id": session_id, "title": request.title}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Lấy toàn bộ lịch sử chat của một phiên (long-term từ SQLite)."""
    session = rag_engine.memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = rag_engine.memory.get_full_history(session_id)
    return {
        "session_id": session_id,
        "title": session.get("title", ""),
        "messages": messages
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Xóa phiên chat và toàn bộ lịch sử."""
    session = rag_engine.memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rag_engine.memory.delete_session(session_id)
    return {"message": "Session deleted", "session_id": session_id}
@app.delete("/api/sessions/{session_id}/last-turn")
async def delete_session_last_turn(session_id: str):
    """Xóa lượt chat cuối cùng (cả câu hỏi và câu trả lời) để cập nhật mới."""
    session = rag_engine.memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    rag_engine.memory.delete_last_turn(session_id)
    return {"message": "Last turn deleted", "session_id": session_id}


class SyncConflictRequest(BaseModel):
    document_numbers_to_disable: List[str]
    new_file_id: Optional[str] = None
    new_filename: Optional[str] = None

# =====================================================================
# UPLOAD & INGEST ENDPOINTS
# =====================================================================
@app.post("/api/documents/sync-conflict")
async def sync_conflict_database(request: SyncConflictRequest):
    """
    Xác nhận đồng bộ cơ sở dữ liệu sau khi nhận kết quả phân tích Xung đột.
    Sẽ đánh dấu is_active = False đối với danh sách document_number cung cấp.
    """
    try:
        from backend.retrieval.hybrid_search import retriever
        total_disabled = 0
        for doc_num in request.document_numbers_to_disable:
            disabled_chunks = retriever.deactivate_document_by_number(doc_num)
            total_disabled += disabled_chunks
            print(f"    [API] Diactivated {disabled_chunks} chunks for document: {doc_num}")
            
        ingest_task_id = None
        if request.new_file_id and request.new_filename:
            # Trigger Ingestion
            ext = os.path.splitext(request.new_filename)[1].lower()
            file_path = os.path.join(UPLOAD_DIR, f"{request.new_file_id}{ext}")
            
            if os.path.exists(file_path):
                from backend.ingestion.pipeline import process_document_task
                result = process_document_task(file_path)
                print(f"    -> [API] Synchronous Ingestion Completed: {result.get('status')}")
            
        return {
            "status": "success",
            "message": f"Cập nhật thành công. Đã vô hiệu hoá {total_disabled} record thuộc về {len(request.document_numbers_to_disable)} văn bản cũ.",
            "disabled_chunks": total_disabled
        }
    except Exception as e:
        print(f"    ❌ Sync conflict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...), 
    session_id: Optional[str] = Form(None)
):
    if os.environ.get("QDRANT_READ_ONLY", "false").lower() == "true":
        raise HTTPException(status_code=403, detail="Qdrant configured as read-only. Ingestion disabled.")

    if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ Upload định dạng PDF, DOCX, và DOC.")

    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    print(f"--- [API] Receiving File Upload: {file.filename} -> {temp_file_path} ---")

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Xử lý trích xuất RAM ngay lập tức nếu có session_id
    chunks = []
    if session_id:
        try:
            from backend.utils.document_parser import parser
            print(f"    -> [API] Parsing file into session RAM: {session_id}")
            chunks = parser.parse_and_chunk(temp_file_path)
            rag_engine.memory.set_temp_chunks(session_id, chunks)
        except Exception as e:
            print(f"    ⚠️ [API] Failed to parse file for RAM storage: {e}")

    return {
        "message": "Đã tải lên tạm thời và sẵn sàng truy vấn.",
        "file_id": file_id,
        "filename": file.filename,
        "session_id": session_id,
        "chunks_count": len(chunks)
    }


@app.post("/api/ingest")
async def ingest_document(request: IngestRequest):
    """
    Nạp dữ liệu vĩnh viễn vào DB.
    Kiểm tra trùng lặp số hiệu văn bản trước khi nạp.
    """
    ext = os.path.splitext(request.filename)[1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{request.file_id}{ext}")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File không tồn tại trên server.")

    # 1. Trích xuất metadata để kiểm tra trùng lặp
    try:
        from backend.utils.document_parser import parser
        from backend.database.qdrant_client import client as qdrant
        
        print(f"--- [API] Ingest Request for {request.filename} ---")
        # Chỉ parse metadata (thường là ở đầu file)
        metadata = parser.extract_metadata(file_path)
        doc_number = metadata.get("document_number")

        if doc_number:
            print(f"    -> Checking duplicate for: {doc_number}")
            search_result = qdrant.scroll(
                collection_name=os.environ.get("QDRANT_COLLECTION", "legal_hybrid_rag_docs"),
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_number",
                            match=models.MatchValue(value=doc_number),
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            if search_result[0]:
                existing_doc = search_result[0][0].payload.get("title", request.filename)
                return {
                    "status": "duplicate",
                    "message": f"Tài liệu có số hiệu '{doc_number}' đã tồn tại trong DB (Tên: {existing_doc}).",
                    "document_number": doc_number
                }

        # 2. Execute Task Synchronously
        from backend.ingestion.pipeline import process_document_task
        print(f"    -> Running Ingestion synchronously...")
        result = process_document_task(file_path)

        if result.get("status") == "success":
            return {
                "status": "success",
                "message": "Nạp dữ liệu vào DB thành công.",
                "result": result
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error during ingestion"))

    except Exception as e:
        print(f"    ❌ Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
