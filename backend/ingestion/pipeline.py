"""
Document ingestion pipeline for single-upload mode.
Đồng bộ với output schema mới từ AdvancedLegalChunker:
  chunk["qdrant_metadata"] — payload nhẹ cho Qdrant
  chunk["neo4j_metadata"] — payload đầy đủ cho Neo4j
  chunk["text_to_embed"]  — text tinh gọn để embed (không có metadata header)
"""
import os
import uuid
import time
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from backend.database.qdrant_client import get_qdrant_client
from backend.models.embedder import embedder
from backend.utils.document_parser import parser
from backend.models.llm_factory import chat_completion
from backend.database.neo4j_client import get_neo4j_driver, build_neo4j


def fetch_old_text_from_qdrant(doc_number: str, article_ref: str) -> str:
    """Truy vấn Qdrant để trích xuất nội dung cũ của một điều luật."""
    try:
        qdrant = get_qdrant_client()
        search_result = qdrant.scroll(
            collection_name=os.environ.get("QDRANT_COLLECTION", "legal_hybrid_rag_docs"),
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_number", match=MatchValue(value=doc_number)),
                    FieldCondition(key="article_ref", match=MatchValue(value=article_ref)),
                ]
            ),
            limit=1,
            with_payload=True
        )
        if search_result and search_result[0]:
            return search_result[0][0].payload.get("chunk_text", "")
    except Exception as e:
        print(f"Lỗi truy vấn old_text trên Qdrant: {e}")
    return ""


def process_document_task(file_path: str):
    """
    Task xử lý Single Upload toàn diện:
    1. Trích xuất văn bản & Phân rã Hierarchical Chunks + Graph Relations
    2. Lookup old_text từ Qdrant
    3. Tóm tắt nội dung
    4. Upsert Qdrant (Hybrid) — dùng qdrant_metadata + text_to_embed
    5. Build Neo4j (Graph) — dùng neo4j_metadata
    """
    try:
        if os.environ.get("QDRANT_READ_ONLY", "false").lower() == "true":
            return {"status": "error", "error": "Qdrant is read-only."}

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} không tồn tại.")

        filename = os.path.basename(file_path)
        
        # 1. Trích xuất Text & Chunks + Lấy Relation
        print(f"⏳ Processing document: {filename}")
        chunks = parser.parse_and_chunk(file_path)
        if not chunks:
            raise ValueError("Không trích xuất được nội dung hợp lệ.")

        # 2. Xử lý thiếu Cross-document `target_text` do chạy cục bộ
        for chunk in chunks:
            # Hỗ trợ cả output schema mới (neo4j_metadata) và cũ (metadata) 
            meta = chunk.get("neo4j_metadata", chunk.get("metadata", {}))
            if "ontology_relations" in meta:
                for rel in meta["ontology_relations"]:
                    target_doc = rel.get("target_doc")
                    target_art = rel.get("target_article")
                    if target_doc and target_art and not rel.get("target_text"):
                        old_txt = fetch_old_text_from_qdrant(target_doc, target_art)
                        rel["target_text"] = old_txt if old_txt else ""

        # 3. Tóm tắt (Summary)
        full_text_sample = "\n".join([c.get("chunk_text", "") for c in chunks[:5]])
        summary_prompt = f"Hãy viết một bản tóm tắt cực ngắn (khoảng 3-5 câu) nội dung chính của tài liệu sau:\n\n{full_text_sample}"
        summary = chat_completion([{"role": "user", "content": summary_prompt}], temperature=0.1)

        conflict_report = "Tự động phân tích xung đột đã tắt. Vui lòng hỏi trong Chat."

        # 4. Lưu trữ Qdrant Vector DB
        # Ưu tiên dùng text_to_embed (tinh gọn) nếu có, fallback về chunk_text
        all_texts = [c.get("text_to_embed", c.get("chunk_text", "")) for c in chunks]
        dense_vectors = embedder.encode_dense(all_texts)
        sparse_vectors = embedder.encode_sparse_documents(all_texts)

        points = []
        doc_id_override = str(uuid.uuid4())
        
        for idx, chunk in enumerate(chunks):
            # Lấy payload Qdrant (nhẹ) — fallback về metadata cũ nếu chưa có schema mới
            q_payload = chunk.get("qdrant_metadata", chunk.get("metadata", {}))
            q_payload["document_id"] = doc_id_override
            q_payload["is_user_upload"] = True
            q_payload["summary"] = summary if idx == 0 else ""
            
            # Cập nhật neo4j_metadata document_id nếu có
            n_payload = chunk.get("neo4j_metadata", {})
            if n_payload:
                n_payload["document_id"] = doc_id_override
            
            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
                vector={
                    "dense": dense_vectors[idx],
                    "sparse": sparse_vectors[idx],
                },
                payload=q_payload
            ))

        collection_name = os.environ.get("QDRANT_COLLECTION", "legal_hybrid_rag_docs")
        qdrant = get_qdrant_client()
        qdrant.upsert(collection_name=collection_name, points=points)
        print(f"✅ Đã Upsert {len(points)} chunks lên Qdrant.")

        # 5. Lưu trữ Neo4j Graph DB
        driver = get_neo4j_driver()
        if driver:
            build_neo4j(driver, chunks)
            print(f"✅ Đã Push Graph Nodes và Relations lên Neo4j.")
            driver.close()
        else:
            print("⚠️ Bỏ qua Neo4j vì config không hợp lệ hoặc không có kết nối.")

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
