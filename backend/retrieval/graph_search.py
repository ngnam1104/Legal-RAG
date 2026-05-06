import logging
import json
import re
from typing import List, Dict, Any, Optional
from backend.models.llm_factory import get_client
from backend.database.neo4j_client import get_neo4j_driver
from backend.utils.text_utils import strip_thinking_tags

logger = logging.getLogger(__name__)

# ── Labels pháp lý hợp lệ (dùng để thu hẹp scan, tận dụng label index) ──
_ENTITY_LABELS = (
    "Organization", "Person", "Location", "Procedure",
    "Condition", "Fee", "Penalty", "Timeframe", "Role",
    "Concept", "Term", "LegalArticle"
)
_LABEL_FILTER = "|".join(_ENTITY_LABELS)  # dùng trong Cypher: (e:Organization|Person|...)

QUERY_ENTITY_PROMPT = """Bạn là trợ lý phân tích truy vấn pháp lý.
Hãy trích xuất các Thực thể pháp lý quan trọng (như: Cơ quan, Tổ chức, Tên văn bản, Người, Khái niệm, Quy trình, Thuật ngữ) từ câu hỏi của người dùng.
Lưu ý: Chỉ trích xuất CỤM TỪ CỐT LÕI (VD: "Bộ Y tế", "Thông tư 19", "Quy trình cấp phép"). Không trích xuất các từ hỏi (ai, cái gì, thế nào).
Tối đa 5 thực thể. Chỉ trả về JSON thuần túy, không có văn bản nào khác.

Định dạng trả về:
{
  "entities": ["Thực thể 1", "Thực thể 2"]
}

Câu hỏi: {query}
JSON:
"""


class EntityGraphRetriever:
    """
    Retriever thực hiện truy xuất dựa trên Entity (Thực thể) từ Graph Database (Neo4j).
    Chiến lược (Entity-First):
      1. Regex: Bắt số hiệu văn bản, tên cơ quan quen thuộc không cần LLM (nhanh).
      2. LLM: Trích xuất Entity phức tạp nếu Regex không đủ (chậm hơn, chỉ gọi khi cần).
      3. Neo4j: Fuzzy match Entity → Chunk IDs + Graph Context.
    """
    def __init__(self):
        self._llm_client = None   # lazy init để tránh block khi import
        self._neo4j_driver = None

    @property
    def llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_client()
        return self._llm_client

    @property
    def neo4j_driver(self):
        if self._neo4j_driver is None:
            self._neo4j_driver = get_neo4j_driver()
        return self._neo4j_driver

    # ------------------------------------------------------------------
    # 1. EXTRACT ENTITIES từ query
    # ------------------------------------------------------------------
    def _regex_extract(self, query: str) -> List[str]:
        """Regex nhanh: bắt số hiệu văn bản, tên Bộ/UBND quen thuộc."""
        entities = []
        # Số hiệu văn bản (VD: 19/2021/TT-BYT)
        doc_ids = re.findall(
            r"\d+/\d{4}/(?:TT|NĐ|NĐ-CP|QH|L|QĐ|NQ)(?:-[A-ZĐa-zđÀ-ỹ]+)*",
            query, re.IGNORECASE
        )
        entities.extend(d.upper() for d in doc_ids)
        # Tên cơ quan hay gặp
        organs = re.findall(
            r"Bộ\s+[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ ]+|"
            r"UBND(?:\s+(?:tỉnh|huyện|thành phố|xã|phường)\s+[A-ZĐ][a-zàáảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ ]+)?",
            query
        )
        entities.extend(o.strip() for o in organs if o.strip())
        return list(dict.fromkeys(entities))  # dedup, giữ thứ tự

    def extract_entities(self, query: str) -> List[str]:
        """Regex trước, LLM fallback nếu không tìm được gì."""
        fast = self._regex_extract(query)
        if fast:
            return fast[:5]

        # LLM fallback (chỉ khi regex trắng tay)
        prompt = QUERY_ENTITY_PROMPT.format(query=query)
        try:
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            if not response:
                return []
            response = strip_thinking_tags(response)
            cleaned = re.sub(r'```json\n|\n```|```', '', response).strip()
            data = json.loads(cleaned)
            return data.get("entities", [])[:5]
        except Exception as e:
            logger.warning(f"[Graph Search] Entity extraction error: {e}")
            return []

    # ------------------------------------------------------------------
    # 2. SEARCH by entities trong Neo4j
    # ------------------------------------------------------------------
    def search_by_entities(self, entities: List[str]) -> Dict[str, Any]:
        """
        Quét Neo4j để lấy chunk_id + graph context liên kết với entities.

        FIX so với phiên bản cũ:
          - Chỉ MATCH trên các label pháp lý hợp lệ → dùng label index, nhanh hơn.
          - Traverse ĐÚNG HƯỚNG: Entity → ngược [:HAS_ENTITY] → Chunk → ngược [...] → Document.
          - Tách 2 query: (A) chunk_ids, (B) doc relations qua entity-to-entity relations.
        """
        if not entities or not self.neo4j_driver:
            return {"chunk_ids": [], "graph_context": "", "doc_numbers": [], "found_entities": []}

        # Query A: Tìm Chunk chứa Entity (HAS_ENTITY ngược chiều)
        # Dùng label filter để Neo4j dùng label index thay vì full scan
        chunk_query = f"""
        UNWIND $entities AS ent_name
        WITH ent_name,
             replace(replace(replace(replace(ent_name, '(', '\\\\('), ')', '\\\\)'), '[', '\\\\['), ']', '\\\\]') AS safe_ent
        MATCH (e:{_LABEL_FILTER})
        WHERE toLower(e.name) CONTAINS toLower(ent_name)
           OR toLower(ent_name) CONTAINS toLower(e.name)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
        WHERE c.qdrant_id IS NOT NULL
        OPTIONAL MATCH (c)-[:PART_OF|BELONGS_TO*1..3]->(doc:Document)
        RETURN DISTINCT
            e.name          AS entity_name,
            labels(e)[0]    AS entity_type,
            c.qdrant_id     AS chunk_id,
            doc.document_number AS doc_number
        LIMIT 150
        """

        # Query B: Quan hệ Entity–Entity (VD: Organization -[:ISSUED_BY]-> Organization)
        rel_query = f"""
        UNWIND $entities AS ent_name
        MATCH (e:{_LABEL_FILTER})
        WHERE toLower(e.name) CONTAINS toLower(ent_name)
        MATCH (e)-[r]->(target)
        WHERE type(r) <> 'HAS_ENTITY' AND target.name IS NOT NULL
        RETURN DISTINCT
            e.name          AS src_name,
            labels(e)[0]    AS src_type,
            type(r)         AS rel_type,
            target.name     AS tgt_name,
            labels(target)[0] AS tgt_type
        LIMIT 50
        """

        chunk_ids  = set()
        doc_numbers = set()
        entity_info = set()
        rel_lines   = []

        try:
            with self.neo4j_driver.session() as session:
                # A. Chunks + Docs
                for r in session.run(chunk_query, entities=entities).data():
                    if r.get("entity_name"):
                        entity_info.add(f"{r.get('entity_type','Entity')}: {r['entity_name']}")
                    if r.get("chunk_id"):
                        chunk_ids.add(str(r["chunk_id"]))
                    if r.get("doc_number"):
                        doc_numbers.add(r["doc_number"])

                # B. Entity–Entity relations
                for r in session.run(rel_query, entities=entities).data():
                    src = r.get("src_name", "?")
                    rel = r.get("rel_type", "?")
                    tgt = r.get("tgt_name", "?")
                    src_t = r.get("src_type", "Entity")
                    tgt_t = r.get("tgt_type", "Entity")
                    rel_lines.append(f"[{src_t}: {src}] --({rel})--> [{tgt_t}: {tgt}]")

            # Format Graph Context
            ctx_parts = []
            if entity_info:
                ctx_parts.append("Thực thể liên quan: " + ", ".join(sorted(entity_info)))
            if rel_lines:
                ctx_parts.append("Quan hệ Thực thể:\n" + "\n".join(rel_lines[:20]))

            graph_context = "\n".join(ctx_parts)

            logger.info(
                f"  [Graph Search] entities={len(entity_info)}, "
                f"chunk_ids={len(chunk_ids)}, docs={len(doc_numbers)}, rels={len(rel_lines)}"
            )

            return {
                "chunk_ids"     : list(chunk_ids),
                "doc_numbers"   : list(doc_numbers),
                "graph_context" : graph_context,
                "found_entities": list(entity_info),
            }

        except Exception as e:
            logger.error(f"[Graph Search] Neo4j query error: {e}")
            return {"chunk_ids": [], "graph_context": "", "doc_numbers": [], "found_entities": []}

    # ------------------------------------------------------------------
    # 3. END-TO-END
    # ------------------------------------------------------------------
    def search(self, query: str) -> Dict[str, Any]:
        """End-to-end: Extract → Neo4j search."""
        try:
            entities = self.extract_entities(query)
            if not entities:
                logger.debug("[Graph Search] No entities extracted, skipping Neo4j.")
                return {"chunk_ids": [], "doc_numbers": [], "graph_context": "", "found_entities": []}
            return self.search_by_entities(entities)
        except Exception as e:
            logger.error(f"[Graph Search] Unexpected error: {e}")
            return {"chunk_ids": [], "doc_numbers": [], "graph_context": "", "found_entities": []}


# Global instance — lazy-init, không block khi import
entity_retriever = EntityGraphRetriever()
