import os
import sys
import uuid
import docx
from pdf2image import convert_from_path
import pytesseract
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workers.celery_app import celery_app
from core.config import settings
from rag.document_manager import document_manager
from core.db import client as qdrant

@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, file_path: str):
    """
    Task xử lý bất đồng bộ: Đọc file PDF/Docx -> text -> chunking -> vector -> qdrant.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")

        ext = os.path.splitext(file_path)[1].lower()
        full_text = ""

        if ext == ".docx":
            # Xử lý text file docx
            doc = docx.Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".doc":
            # Xử lý file .doc cũ (Word 97-2003)
            try:
                # 1. Thử dùng antiword (cho Docker / Linux)
                import subprocess
                result = subprocess.run(['antiword', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                full_text = result.stdout
            except Exception:
                # 2. Thử dùng win32com nếu đang chạy Local trên Windows có cài MS Word
                try:
                    import win32com.client
                    word = win32com.client.DispatchEx("Word.Application")
                    word.Visible = False
                    # win32com requires absolute paths
                    abs_path = os.path.abspath(file_path)
                    doc = word.Documents.Open(abs_path)
                    full_text = doc.Content.Text
                    doc.Close(False)
                    word.Quit()
                except Exception as e:
                    raise ValueError(f"Hệ thống không cài đặt phần mềm hỗ trợ đọc .doc (yêu cầu antiword trên Linux hoặc MS Word trên Windows): {str(e)}")
        elif ext == ".pdf":
            # Xử lý PDF và OCR với pytesseract
            # Cần cài đặt poppler và tesseract-ocr (ngôn ngữ vie) trên hệ điều hành nhé
            images = convert_from_path(file_path)
            text_pages = []
            for img in images:
                text = pytesseract.image_to_string(img, lang="vie")
                text_pages.append(text)
            full_text = "\n".join(text_pages)
        else:
            raise ValueError(f"Định dạng không được hỗ trợ: {ext}")

        # Xử lý file rác hoặc quá ngắn
        if len(full_text.strip()) < 50:
            raise ValueError("Nội dung file quá ngắn hoặc không có chữ (OCR thất bại / File rác).")

        # Cố gắng bóc tách tiêu đề/số hiệu từ tên file
        filename = os.path.basename(file_path)
        doc_number = filename.split('.')[0]
        
        metadata = {
            "id": str(uuid.uuid4()),
            "document_number": doc_number,
            "title": f"Tài liệu OCR từ {filename}",
            "is_appendix": False
        }

        # Gọi DocumentManager để chunk, check conflict và lưu Qdrant tự động
        result = document_manager.add_document(content=full_text, metadata=metadata)
        
        if result.get("status") == "error":
            raise ValueError(result.get("message", "Lỗi xử lý DocumentManager."))

        # Remove temp file after successful processing
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "status": "success", 
            "doc_number": doc_number, 
            "chunks_inserted": result.get("chunks_inserted", 0),
            "conflicts_found": result.get("conflicts_found", [])
        }

    except Exception as exc:
        print(f"Error processing file {file_path}: {exc}")
        # Could retry upon API limit errors or transient Qdrant errors
        # self.retry(exc=exc, countdown=60)
        return {"status": "error", "error": str(exc)}
