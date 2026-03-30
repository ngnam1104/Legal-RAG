import os
import uuid
import docx
from pdf2image import convert_from_path
import pytesseract
from qdrant_client.models import PointStruct

from backend.workers.celery_app import celery_app
from backend.config import settings
from retrieval.vector_db import client as qdrant
from retrieval.embedder import embedder

@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, file_path: str):
    """
    Task xử lý bất đồng bộ: Đọc file PDF/Docx -> text -> chunking -> vector -> qdrant.
    """
    try:
        if settings.QDRANT_READ_ONLY:
            return {
                "status": "error",
                "error": "Qdrant is configured as read-only. Ingestion task is disabled.",
            }

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")

        ext = os.path.splitext(file_path)[1].lower()
        full_text = ""

        if ext == ".docx":
            doc = docx.Document(file_path)
            full_text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".doc":
            try:
                import subprocess
                result = subprocess.run(['antiword', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                full_text = result.stdout
            except Exception:
                try:
                    import win32com.client
                    word = win32com.client.DispatchEx("Word.Application")
                    word.Visible = False
                    abs_path = os.path.abspath(file_path)
                    doc = word.Documents.Open(abs_path)
                    full_text = doc.Content.Text
                    doc.Close(False)
                    word.Quit()
                except Exception as e:
                    raise ValueError(f"Không thể đọc .doc: {str(e)}")
        elif ext == ".pdf":
            images = convert_from_path(file_path)
            text_pages = []
            for img in images:
                text = pytesseract.image_to_string(img, lang="vie")
                text_pages.append(text)
            full_text = "\n".join(text_pages)
        else:
            raise ValueError(f"Định dạng không được hỗ trợ: {ext}")

        if len(full_text.strip()) < 50:
            raise ValueError("Nội dung file quá ngắn hoặc không có chữ.")

        filename = os.path.basename(file_path)
        doc_number = filename.split('.')[0]
        
        # Simple chunking: split by paragraphs
        paragraphs = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 20]
        if not paragraphs:
            raise ValueError("Không tìm thấy đoạn văn bản hợp lệ.")

        # Encode and upsert
        dense_vectors = embedder.encode_dense(paragraphs, batch_size=32)
        sparse_vectors = embedder.encode_sparse_documents(paragraphs, batch_size=32)

        points = []
        for idx, text in enumerate(paragraphs):
            payload = {
                "document_number": doc_number,
                "title": f"Tài liệu OCR từ {filename}",
                "article_ref": f"Đoạn {idx + 1}",
                "is_appendix": False,
                "chunk_text": text,
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense_vectors[idx],
                    },
                    payload=payload,
                )
            )
            # Note: sparse vectors need separate upsert or named vector support

        BATCH_SIZE = 100
        collection_name = settings.QDRANT_COLLECTION
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            qdrant.upsert(collection_name=collection_name, points=batch)

        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "status": "success", 
            "doc_number": doc_number, 
            "chunks_inserted": len(points),
        }

    except Exception as exc:
        print(f"Error processing file {file_path}: {exc}")
        return {"status": "error", "error": str(exc)}
