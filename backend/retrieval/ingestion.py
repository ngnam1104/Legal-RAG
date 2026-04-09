import os
import uuid
from qdrant_client.models import PointStruct

from backend.config import settings
from backend.retrieval.vector_db import client as qdrant
from backend.retrieval.embedder import embedder
from backend.utils.document_parser import parser
from backend.agent.utils_conflict_analyzer import (
    extract_claims_from_text, 
    judge_claim, 
    review_claim
) # Changed from flow_conflict_analyzer to utils_conflict_analyzer
from backend.llm.factory import chat_completion

def process_document_task(file_path: str):
    """
    Task xử lý thông minh (Đồng bộ): 
    1. Trích xuất & Chunks (Hierarchical)
    2. Tóm tắt nội dung (LLM Summary)
    3. Phân tích xung đột (Conflict Analysis)
    4. Lưu trữ Hybrid Vector (Dense + Sparse)
    """
    try:
        if settings.QDRANT_READ_ONLY:
            return {"status": "error", "error": "Qdrant is read-only."}

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")

        filename = os.path.basename(file_path)
        
        # 1. Trích xuất văn bản & Phân đoạn phân cấp (Hierarchical Chunking)
        print(f"⏳ Processing document: {filename}")
        chunks = parser.parse_and_chunk(file_path)
        if not chunks:
            raise ValueError("Không trích xuất được nội dung hợp lệ.")

        # 2. Tạo bản Tóm tắt nhanh (Quick Summary)
        full_text_sample = "\n".join([c.get("text_to_embed", "") for c in chunks[:5]])
        summary_prompt = f"Hãy viết một bản tóm tắt cực ngắn (khoảng 3-5 câu) nội dung chính của tài liệu sau:\n\n{full_text_sample}"
        summary = chat_completion([{"role": "user", "content": summary_prompt}], temperature=0.1)

        # 3. [FREE TIER SAFETY] Bỏ qua Phân tích xung đột tự động khi upload để tiết kiệm Token 70B
        # Phân tích này nên được kích hoạt thủ công trong Chat qua chế độ CONFLICT_ANALYZER
        conflict_report = "Tự động phân tích bị tắt để tiết kiệm API Quota. Vui lòng chat với file để phân tích."
        # conflict_report = analyze_document_conflicts(chunks)


        # 4. Lưu trữ vào Vector DB (Hybrid)
        all_texts = [c["text_to_embed"] for c in chunks]
        dense_vectors = embedder.encode_dense(all_texts)
        sparse_vectors = embedder.encode_sparse_documents(all_texts)

        points = []
        for idx, chunk in enumerate(chunks):
            p = chunk["metadata"]
            payload = {
                "document_id": str(uuid.uuid4()),
                "chunk_id": chunk.get("chunk_id", str(uuid.uuid4())),
                "title": p.get("title", filename),
                "is_appendix": bool(p.get("is_appendix", False)),
                "chunk_text": chunk.get("text_to_embed", ""),
                "is_user_upload": True,
                "summary": summary if idx == 0 else "" # Gắn summary vào chunk đầu tiên để dễ tra cứu
            }
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vectors[idx],
                    "sparse": sparse_vectors[idx],
                },
                payload=payload
            ))

        collection_name = settings.QDRANT_COLLECTION
        qdrant.upsert(collection_name=collection_name, points=points)

        # Cleanup file tạm (Để lại để chat_engine có thể parse context thô nhanh)
        # if os.path.exists(file_path):
        #     os.remove(file_path)

        return {
            "status": "success",
            "filename": filename,
            "summary": summary,
            "conflict_report": conflict_report,
            "chunks": len(points)
        }

    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return {"status": "error", "error": str(exc)}
