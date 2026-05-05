import os
import subprocess
from typing import List, Dict, Any
import docx
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    import docx2txt
except ImportError:
    docx2txt = None

from backend.ingestion.chunker.core import chunker

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

    def extract_text_from_doc(self, file_path: str) -> str:
        """Extract text from legacy .doc (Word 97-2003) format.
        Tries: 1) docx2txt, 2) python-docx (some .doc files work), 3) raw binary extraction.
        """
        # Strategy 1: docx2txt (handles some .doc files)
        if docx2txt:
            try:
                text = docx2txt.process(file_path)
                if text and len(text.strip()) > 50:
                    return text
            except Exception:
                pass
        
        # Strategy 2: python-docx (works if the .doc is actually docx-compat)
        try:
            doc = docx.Document(file_path)
            text_content = [para.text for para in doc.paragraphs]
            text = "\n".join(text_content)
            if text and len(text.strip()) > 50:
                return text
        except Exception:
            pass

        # Strategy 3: Raw binary text extraction (last resort)
        try:
            with open(file_path, 'rb') as f:
                raw = f.read()
            # Extract readable text chunks from binary
            import re
            # Try utf-16-le first (common in .doc)
            try:
                decoded = raw.decode('utf-16-le', errors='ignore')
                # Filter to keep only printable Vietnamese text
                lines = decoded.split('\n')
                clean_lines = [l.strip() for l in lines if len(l.strip()) > 2 and not all(c in '\x00\x01\x02\x03\x04\x05' for c in l[:5])]
                text = "\n".join(clean_lines)
                if len(text.strip()) > 100:
                    return text
            except Exception:
                pass
            
            # utf-8 fallback
            decoded = raw.decode('utf-8', errors='ignore')
            lines = decoded.split('\n')
            clean_lines = [l.strip() for l in lines if len(l.strip()) > 3]
            text = "\n".join(clean_lines)
            if len(text.strip()) > 100:
                return text
        except Exception as e:
            pass

        raise RuntimeError(f"Failed to read .doc file {file_path}. Install docx2txt: pip install docx2txt")

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
        elif ext == ".doc":
            content = self.extract_text_from_doc(file_path)
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

        from backend.ingestion.chunker import metadata as md
        
        # Lấy 100 dòng đầu tiên để tìm số hiệu
        preamble = "\n".join(content.splitlines()[:100])
        doc_number = md.extract_doc_number(preamble)
        
        return {
            "document_number": doc_number,
            "title": os.path.basename(file_path)
        }

parser = DocumentParser()
