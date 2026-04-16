import os
from typing import List, Dict, Any
import docx
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from backend.retrieval.chunker import chunker

class DocumentParser:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, file_path: str) -> str:
        if not fitz:
            raise ImportError("PyMuPDF (fitz) is not installed. Run `pip install pymupdf`")
        text_content = []
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text_content.append(page.get_text())
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF {file_path}: {str(e)}")
        return "\n".join(text_content)

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text_content = [para.text for para in doc.paragraphs]
            return "\n".join(text_content)
        except Exception as e:
            raise RuntimeError(f"Failed to read DOCX {file_path}: {str(e)}")

    def parse_and_chunk(self, file_path: str, base_metadata: Dict[str, Any] = None) -> List[Dict]:
        """
        Extract text from file and apply Hierarchical Chunking (AdvancedLegalChunker).
        Ensures breadcrumbs like (Chương > Điều > Khoản) are preserved.
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            content = self.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            content = self.extract_text_from_docx(file_path)
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        metadata = base_metadata or {}
        if "title" not in metadata:
            metadata["title"] = os.path.basename(file_path)

        # AdvancedLegalChunker handles Regex-based tree splitting and Breadcrumbs
        chunks = chunker.process_document(content, metadata)
        return chunks

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Trích xuất metadata cơ bản (số hiệu văn bản, tiêu đề) để kiểm tra trùng lặp."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            content = self.extract_text_from_pdf(file_path)
        elif ext == ".docx":
            content = self.extract_text_from_docx(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        from backend.retrieval.chunker import metadata as md
        
        # Lấy 100 dòng đầu tiên để tìm số hiệu
        preamble = "\n".join(content.splitlines()[:100])
        doc_number = md.extract_doc_number(preamble)
        
        return {
            "document_number": doc_number,
            "title": os.path.basename(file_path)
        }

parser = DocumentParser()
