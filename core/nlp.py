import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from core.config import settings

# ---------------------------------------------------------
# 1. EMBEDDING SINGLETON
# ---------------------------------------------------------
class LocalEmbedder:
    """Tạo embedding cho văn bản bằng model chạy local qua CPU/GPU"""
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        """Encode 1 hoặc nhiều chuỗi thành vectors"""
        if isinstance(texts, str):
            texts = [texts]
        vectors = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar
        )
        return vectors.tolist()

_embedder_instance = None

def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        print("Loading Embedding Model...")
        _embedder_instance = LocalEmbedder()
    return _embedder_instance

# ---------------------------------------------------------
# 2. CHUNKING LOGIC FOR INGESTION & WORKERS
# ---------------------------------------------------------
class AdvancedLegalChunker:
    """Phiên bản mới mạnh mẽ hơn, hỗ trợ chia metadata chi tiết."""
    def __init__(self):
        self.dieu_pattern = re.compile(r'(?m)^[\s]*(Điều\s+\d+[\.\:])')
        self.phuluc_pattern = re.compile(r'(?m)^[\s]*(PHỤ LỤC|DANH MỤC).*$')
        
    def _build_metadata_header(self, metadata, is_appendix=False, ref_tag=""):
        header_lines = ["[THÔNG TIN TRÍCH DẪN]"]
        for key, value in metadata.items():
            if key not in ['id', 'is_appendix'] and pd.notna(value):
                header_lines.append(f"- {str(key).capitalize()}: {value}")
        
        header_lines.append(f"- Tham chiếu: {ref_tag}")
        
        if is_appendix:
            header_lines.append("\n[LOẠI NỘI DUNG: PHỤ LỤC/DANH MỤC CHI TIẾT]")
        else:
            header_lines.append("\n[LOẠI NỘI DUNG: NỘI DUNG CHÍNH / ĐIỀU KHOẢN]")
            
        return "\n".join(header_lines) + "\n"

    def process_document(self, content, metadata):
        content = str(content).strip()
        chunks = []
        
        match_pl = self.phuluc_pattern.search(content)
        if match_pl:
            main_text = content[:match_pl.start()].strip()
            app_text = content[match_pl.start():].strip()
        else:
            main_text = content
            app_text = ""

        if main_text:
            parts = self.dieu_pattern.split(main_text)
            intro_text = parts[0].strip()
            if intro_text:
                header = self._build_metadata_header(metadata, is_appendix=False, ref_tag="Phần mở đầu & Căn cứ")
                chunks.append({
                    "text_to_embed": f"{header}[PHẦN CĂN CỨ PHÁP LÝ]\n{intro_text}",
                    "reference_tag": "Căn cứ",
                    "metadata": {**metadata, "is_appendix": False}
                })
            
            for i in range(1, len(parts), 2):
                dieu_name = parts[i].strip()
                dieu_content = parts[i+1].strip() if i+1 < len(parts) else ""
                header = self._build_metadata_header(metadata, is_appendix=False, ref_tag=dieu_name)
                
                chunks.append({
                    "text_to_embed": f"{header}{dieu_name}\n{dieu_content}",
                    "reference_tag": dieu_name,
                    "metadata": {**metadata, "is_appendix": False}
                })

        if app_text:
            header_app = self._build_metadata_header(metadata, is_appendix=True, ref_tag="Phụ lục")
            app_lines = app_text.split('\n')
            current_chunk = ""
            part_idx = 1
            
            for line in app_lines:
                current_chunk += line + "\n"
                if len(current_chunk) > 1000:
                    chunks.append({
                        "text_to_embed": f"{header_app}[PHẦN {part_idx}]\n{current_chunk.strip()}",
                        "reference_tag": f"Phụ lục - P{part_idx}",
                        "metadata": {**metadata, "is_appendix": True}
                    })
                    current_chunk = ""
                    part_idx += 1
            
            if current_chunk:
                chunks.append({
                    "text_to_embed": f"{header_app}[PHẦN {part_idx}]\n{current_chunk.strip()}",
                    "reference_tag": f"Phụ lục - P{part_idx}",
                    "metadata": {**metadata, "is_appendix": True}
                })
                
        return chunks
