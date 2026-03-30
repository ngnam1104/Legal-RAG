from typing import List, Dict, Optional
from qdrant_client import models
from retrieval.vector_db import client
from retrieval.embedder import embedder
from backend.config import settings
from retrieval.base import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION

    def search(
        self,
        query: str,
        limit: int = 10,
        filter_conditions: Optional[List[models.FieldCondition]] = None
    ) -> List[Dict]:
        """Thực hiện Hybrid Search (Prefetch Dense + Sparse, sau đó Fusion bằng RRF)"""
        dense_vector = embedder.encode_query_dense(query)
        sparse_vector = embedder.encode_query_sparse(query)

        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        # Sử dụng API query_points mới với prefetch
        response = client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=limit * 2,
                    filter=query_filter
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=limit * 2,
                    filter=query_filter
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        formatted_results = []
        for hit in response.points:
            p = hit.payload
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "document_number": p.get("document_number", ""),
                "article_ref": p.get("article_ref", ""),
                "title": p.get("title", ""),
                "text": p.get("chunk_text", ""),
                "is_appendix": p.get("is_appendix", False),
            })

        return formatted_results

retriever = HybridRetriever()
