import uuid
from typing import Dict, Any, List
from qdrant_client.models import PointStruct

from core.db import client as qdrant, ensure_qdrant_collection, COLLECTION_NAME
from core.nlp import AdvancedLegalChunker, get_embedder
from core.llm import chat_completion


class DocumentManager:
    """Xử lý việc thêm văn bản mới và rà soát xung đột tự động."""

    def __init__(self):
        ensure_qdrant_collection(COLLECTION_NAME)
        self.chunker = AdvancedLegalChunker()
        self.embedder = get_embedder()

    def _detect_conflicts_with_llm(self, new_text: str, retrieved_docs: List[Dict]) -> List[str]:
        """Dùng LLM phân tích xem new_text có xung đột hay bãi bỏ các tài liệu cũ không."""
        if not retrieved_docs:
            return []

        old_context = "\n\n---\n\n".join(
            [f"[{d['document_number']} - {d['article_ref']}]: {d['chunk_text']}" for d in retrieved_docs]
        )

        sys_prompt = (
            "Bạn là trợ lý pháp lý phân tích sự thay đổi luật pháp.\n"
            "Hãy kiểm tra xem 'Văn bản mới' có bãi bỏ, thay thế, hoặc xung đột mâu thuẫn trực tiếp với các 'Văn bản cũ' không.\n"
            "Chỉ trả lời bằng danh sách các SỐ HIỆU VĂN BẢN cũ bị xung đột. Nếu không có, hãy trả lời 'KHONG'."
        )
        user_prompt = (
            f"Văn bản cũ (từ Database):\n{old_context}\n\n=== VĂN BẢN MỚI ===\n"
            f"{new_text}\n\nCó văn bản cũ nào bị xung đột không?"
        )

        try:
            answer = chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            ).strip()

            if "KHONG" in answer:
                return []

            conflicting_doc_numbers = []
            for d in retrieved_docs:
                if d["document_number"] in answer:
                    conflicting_doc_numbers.append(d["document_number"])

            return list(set(conflicting_doc_numbers))
        except Exception as e:
            print(f"Error checking conflicts: {e}")
            return []

    def _mark_conflict_in_db(self, old_doc_numbers: List[str], new_doc_number: str):
        """Cập nhật payload của các văn bản bị mâu thuẫn/xung đột."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        for doc_num in old_doc_numbers:
            scroll_results, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_number", match=MatchValue(value=doc_num))]
                ),
                limit=1000,
            )

            for point in scroll_results:
                existing_conflicts = point.payload.get("conflicted_by", [])
                if new_doc_number not in existing_conflicts:
                    existing_conflicts.append(new_doc_number)

                    qdrant.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"conflicted_by": existing_conflicts},
                        points=[point.id],
                    )
            print(f"🚨 Đã đánh dấu văn bản cũ {doc_num} bị xung đột bởi {new_doc_number}.")

    def add_document(self, content: str, metadata: Dict[str, Any]) -> dict:
        """
        1. Chunk & Embed.
        2. Sinh vector cho toàn bộ văn bản.
        3. Dùng vector mở đầu để search văn bản liên quan cũ.
        4. Detect conflict.
        5. Đánh dấu văn bản cũ & Upsert văn bản mới.
        """
        doc_id = metadata.get("id", str(uuid.uuid4()))
        doc_number = metadata.get("document_number", "UNKNOWN")

        chunks = self.chunker.process_document(content, metadata)
        if not chunks:
            return {"status": "error", "message": "No valid text chunks found."}

        intro_text = chunks[0]["text_to_embed"] if len(chunks) > 0 else content[:1000]
        intro_vector = self.embedder.encode(intro_text, show_progress_bar=False)[0]

        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=intro_vector,
            limit=5,
        )

        retrieved_docs = []
        for hit in search_results:
            p = hit.payload
            if p.get("document_number") != doc_number:
                retrieved_docs.append(
                    {
                        "document_number": p.get("document_number"),
                        "article_ref": p.get("article_ref"),
                        "chunk_text": p.get("chunk_text"),
                    }
                )

        conflicting_docs = self._detect_conflicts_with_llm(intro_text, retrieved_docs)

        if conflicting_docs:
            self._mark_conflict_in_db(conflicting_docs, doc_number)

        texts = [c["text_to_embed"] for c in chunks]
        vectors = self.embedder.encode(texts, batch_size=32, show_progress_bar=False)

        points = []
        for idx, chunk in enumerate(chunks):
            meta = chunk["metadata"]

            payload = {
                "document_id": doc_id,
                "document_number": doc_number,
                "title": meta.get("title", "Không tiêu đề"),
                "legal_type": meta.get("legal_type", ""),
                "issuance_date": meta.get("issuance_date", ""),
                "signer": meta.get("signers", ""),
                "article_ref": chunk["reference_tag"],
                "is_appendix": meta.get("is_appendix", False),
                "chunk_text": chunk["text_to_embed"],
                "conflicted_by": [],
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vectors[idx],
                    payload=payload,
                )
            )

        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

        return {
            "status": "success",
            "doc_number": doc_number,
            "chunks_inserted": len(points),
            "conflicts_found": conflicting_docs,
        }


document_manager = DocumentManager()
