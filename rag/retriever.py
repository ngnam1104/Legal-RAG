import os
import sys
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import client, ensure_qdrant_collection, COLLECTION_NAME  # noqa: E402
from core.nlp import get_embedder  # noqa: E402
from qdrant_client.models import Filter, FieldCondition, MatchValue


class LegalRetriever:
    """Xử lý truy vấn CSDL Qdrant với các tham số lọc."""

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.embedder = get_embedder()
        ensure_qdrant_collection(self.collection_name)

    def search(
        self,
        query: str,
        limit: int = 3,
        filter_appendix: bool = False,
        doc_number: str | None = None,
    ) -> List[Dict]:
        """Truy vấn Top-K từ Qdrant DB"""
        print(f"🔍 Searching for: '{query}'")

        query_vector = self.embedder.encode(query, show_progress_bar=False)[0]

        filter_conditions = []
        if filter_appendix:
            filter_conditions.append(
                FieldCondition(key="is_appendix", match=MatchValue(value=False))
            )
        if doc_number:
            filter_conditions.append(
                FieldCondition(key="document_number", match=MatchValue(value=doc_number))
            )
        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        response = client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

        formatted_results = []
        for hit in response.points:
            p = hit.payload
            formatted_results.append(
                {
                    "score": hit.score,
                    "document_number": p.get("document_number", ""),
                    "article_ref": p.get("article_ref", ""),
                    "title": p.get("title", ""),
                    "is_appendix": p.get("is_appendix", False),
                    "text": p.get("chunk_text", ""),
                    "conflicted_by": p.get("conflicted_by", []),
                }
            )

        return formatted_results


retriever = LegalRetriever()
