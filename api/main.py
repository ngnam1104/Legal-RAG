import os
import sys
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workers.tasks import process_document_task

app = FastAPI(
    title="Chatbot VBPL API",
    description="API cho Upload tài liệu và Query hệ thống RAG VBPL.",
    version="1.0.0"
)

# Thư mục chứa file tạm
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

from pydantic import BaseModel
from rag.chat_engine import rag_engine

class ChatRequest(BaseModel):
    session_id: str
    query: str
    provider: str = None
    model: str = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint Hỏi đáp API sử dụng RAG Mode 1 (Memory + Qdrant Top-5)
    """
    response = rag_engine.chat(
        request.session_id, 
        request.query, 
        provider=request.provider, 
        model=request.model
    )
    return response

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint nhận file (PDF hoặc Docx) rồi đưa vào background Celery Workers để xử lý
    """
    if not file.filename.endswith(('.pdf', '.docx', '.doc', '.PDF', '.DOCX', '.DOC')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ Upload định dạng PDF, DOCX, và DOC.")
    
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    temp_file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    # Ghi file vào ổ đĩa chờ Celery xử lý
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Gọi task Celery bất đồng bộ (chỉ truyền vào đường dẫn file)
    task = process_document_task.delay(temp_file_path)
    
    return {
        "message": "Upload thành công, file đang được xử lý ngầm (OCR, Chunking, Embedding, Lưu Qdrant).",
        "task_id": task.id,
        "file_id": file_id
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
