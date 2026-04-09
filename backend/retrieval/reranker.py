from typing import List, Dict, Any
from backend.config import settings

class DocumentReranker:
    """Cross-Encoder reranker siêu xịn (Hỗ trợ đa ngôn ngữ/Tiếng Việt).
    Sử dụng mô hình BAAI/bge-reranker-v2-m3."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        self.model_name = model_name
        self.model = None
        self.use_fp16 = use_fp16
        self._load_failed = False

    def _lazy_load(self):
        if self.model is not None or self._load_failed:
            return
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            print(f"⏳ Đang nạp Reranker model (BAAI/bge-reranker-v2-m3)...")
            model_kwargs = {}
            if self.use_fp16:
                model_kwargs['torch_dtype'] = torch.float16
                
            self.model = CrossEncoder(self.model_name, max_length=4096, default_activation_function=None, model_kwargs=model_kwargs)
            print(f"✅ Reranker model đã sẵn sàng.")
        except Exception as e:
            print(f"⚠️ Không thể tải Reranker model: {e}")
            print(f"   → Hệ thống sẽ dùng RRF score thay thế.")
            self._load_failed = True

    def score(self, query: str, docs: List[str]) -> List[float]:
        self._lazy_load()
        if self.model is None:
            return [0.0] * len(docs)
        if not docs:
            return []
        # Guard: ensure query and all docs are non-empty strings for CrossEncoder tokenizer
        query = str(query) if query is not None else "N/A"
        if not query.strip():
            query = "N/A"
        docs = [str(d) if d is not None else "N/A" for d in docs]
        docs = [d if d.strip() else "N/A" for d in docs]
        pairs = [[query, d] for d in docs]
        # CrossEncoder sử dụng hàm predict
        scores = self.model.predict(pairs)
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]

    def _build_rerank_text(self, payload: Dict[str, Any]) -> str:
        title = payload.get("title") or ""
        citation = payload.get("reference_citation") or ""
        # Gộp title, citation và nội dung chunk_text để Reranker đánh giá chính xác nhất
        content = payload.get("chunk_text") or payload.get("text") or ""
        return f"{title}\n{citation}\n{content}".strip()

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Sắp xếp lại các Documents.
        Nếu Reranker model chưa sẵn sàng → fallback về sắp xếp theo RRF score.
        """
        if not candidates:
            return []

        # Fallback: nếu model chưa load, dùng RRF score có sẵn
        if self.model is None and not self._load_failed:
            self._lazy_load()
        
        if self.model is None:
            # Fallback: sort theo rrf_score
            scored = []
            for item in candidates:
                enriched = dict(item)
                enriched["score"] = enriched.get("rrf_score", 0.0)
                enriched["rerank_score"] = enriched.get("rrf_score", 0.0)
                scored.append(enriched)
            scored.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
            return scored[:top_k]

        docs = [self._build_rerank_text(item.get("payload", item)) for item in candidates]
        rerank_scores = self.score(query, docs)

        scored = []
        for item, score in zip(candidates, rerank_scores):
            enriched = dict(item)
            enriched["score"] = float(score)
            enriched["rerank_score"] = float(score)
            scored.append(enriched)

        scored.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
        return scored[:top_k]

# Singleton instance
reranker = DocumentReranker()

