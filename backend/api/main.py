import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, APIRouter, HTTPException
from pydantic import BaseModel
import uvicorn

from backend.agent.chat_engine import rag_engine
from backend.config import settings

app = FastAPI(
    title="Chatbot VBPL API",
    description="API hệ thống RAG VBPL với FastAPI mới.",
    version="2.0.0"
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    session_id: str
    query: str
    provider: str = None
    model: str = None
    top_k: int = 3
    use_reflection: bool = True

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint Hỏi đáp API sử dụng RAG Engine (Memory + Hybrid Search + Rerank + ReAct + Reflection)
    """
    response = rag_engine.chat(
        session_id=request.session_id, 
        query=request.query, 
        provider=request.provider, 
        # model=request.model,
        top_k=request.top_k,
        use_reflection=request.use_reflection
    )
    return response

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if settings.QDRANT_READ_ONLY:
        raise HTTPException(status_code=403, detail="Qdrant configured as read-only. Ingestion disabled.")
        
    try:
        from backend.workers.tasks import process_document_task
    except ImportError:
        raise HTTPException(status_code=501, detail="Background workers not configured properly.")

    if not file.filename.lower().endswith(('.pdf', '.docx', '.doc')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ Upload định dạng PDF, DOCX, và DOC.")
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    task = process_document_task.delay(temp_file_path)
    
    return {
        "message": "Upload thành công, file đang được xử lý ngầm.",
        "task_id": task.id,
        "file_id": file_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
