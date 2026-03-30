from typing import List, Dict, Union
from FlagEmbedding import BGEM3FlagModel
from qdrant_client.models import SparseVector
from backend.config import settings
from retrieval.base import BaseEmbedder
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LocalBGEHybridEncoder(BaseEmbedder):
    """Sử dụng BGE-M3 để tạo cả Dense và Sparse vectors."""
    def __init__(self, model_name: str = None, device: str = None):
        if not model_name:
            model_name = settings.LEGAL_DENSE_MODEL
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model_name = model_name
        self.device = device
        self.model = BGEM3FlagModel(model_name, use_fp16=False, device=device)

    @staticmethod
    def _to_sparse_vector(weights: Dict[str, float]) -> SparseVector:
        if not weights:
            return SparseVector(indices=[], values=[])
        pairs = [(int(k), float(v)) for k, v in weights.items() if float(v) != 0.0]
        pairs.sort(key=lambda x: x[0])
        return SparseVector(
            indices=[idx for idx, _ in pairs],
            values=[val for _, val in pairs],
        )

    def encode_dense(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        outputs = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vecs = outputs["dense_vecs"]
        return dense_vecs.tolist() if hasattr(dense_vecs, "tolist") else [list(vec) for vec in dense_vecs]

    def encode_sparse_documents(self, texts: Union[str, List[str]], batch_size: int = 8) -> List[SparseVector]:
        if isinstance(texts, str):
            texts = [texts]
        outputs = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=8192,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        lexical_weights = outputs["lexical_weights"]
        return [self._to_sparse_vector(weights) for weights in lexical_weights]

    def encode_query_sparse(self, text: str) -> SparseVector:
        return self.encode_sparse_documents([text], batch_size=1)[0]

    def encode_query_dense(self, text: str) -> List[float]:
        return self.encode_dense([text], batch_size=1)[0]

# Singleton instance
embedder = LocalBGEHybridEncoder()
