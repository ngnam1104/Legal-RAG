import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from qdrant_client import models

from backend.agent.chat_engine import rag_engine
from backend.config import settings

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Chatbot VBPL API",
    description="API hệ thống RAG VBPL với FastAPI - Bao gồm Chat, Sessions, Upload.",
    version="3.0.0"
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
    mode: str = "LEGAL_QA"  # Mặc định: Hỏi đáp Pháp luật
    file_path: Optional[str] = None  # Đường dẫn file tạm nếu cần context riêng
    provider: str = None
    model: str = None
    top_k: int = 3
    use_reflection: bool = True
    use_rerank: bool = None  # None = dùng config mặc định, True/False = override

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
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint Hỏi đáp API. Tự động tạo session nếu chưa có.
    """
    # Tự tạo session nếu client không gửi session_id
    session_id = request.session_id
    if not session_id:
        session_id = str(uuid.uuid4())
        rag_engine.memory.create_session(session_id=session_id)

    # Đảm bảo session tồn tại trong DB
    if not rag_engine.memory.get_session(session_id):
        rag_engine.memory.create_session(session_id=session_id)

    print(f"\n--- [API] Incoming Chat Request ---")
    print(f"    Session ID: {session_id}")
    print(f"    Mode: {request.mode}")
    print(f"    Query: {request.query[:100]}...")
    print(f"    Top-K: {request.top_k} | Rerank: {request.use_rerank} | Reflection: {request.use_reflection}")

    response = rag_engine.chat(
        session_id=session_id,
        query=request.query,
        mode=request.mode,
        file_path=request.file_path,
        top_k=request.top_k,
        use_reflection=request.use_reflection,
        use_rerank=request.use_rerank
    )
    
    print(f"--- [API] Request Handled Successfully ---\n")
    return response


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


# =====================================================================
# UPLOAD ENDPOINT (giữ nguyên logic cũ)
# =====================================================================
# =====================================================================
# UPLOAD & INGEST ENDPOINTS
# =====================================================================
@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...), 
    session_id: Optional[str] = Form(None)
):
    if settings.QDRANT_READ_ONLY:
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
        from backend.retrieval.vector_db import client as qdrant
        
        print(f"--- [API] Ingest Request for {request.filename} ---")
        # Chỉ parse metadata (thường là ở đầu file)
        metadata = parser.extract_metadata(file_path)
        doc_number = metadata.get("document_number")

        if doc_number:
            print(f"    -> Checking duplicate for: {doc_number}")
            search_result = qdrant.scroll(
                collection_name=settings.QDRANT_COLLECTION,
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

        # 2. Trigger Background Task
        from backend.workers.tasks import process_document_task
        task = process_document_task.delay(file_path)
        print(f"    -> Ingestion Task Triggered: {task.id}")

        return {
            "status": "success",
            "message": "Bắt đầu nạp dữ liệu vào DB...",
            "task_id": task.id
        }

    except Exception as e:
        print(f"    ❌ Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    try:
        from backend.workers.tasks import process_document_task
        task_result = process_document_task.AsyncResult(task_id)

        if task_result.ready():
            result = task_result.result
            if isinstance(result, dict) and result.get("status") == "success":
                return {
                    "status": "completed",
                    "result": result
                }
            else:
                return {
                    "status": "failed",
                    "error": str(result.get("error") if isinstance(result, dict) else result)
                }
        return {"status": "pending"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
