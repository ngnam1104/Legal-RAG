import logging
import json
import re
from typing import List, Dict, Any, Optional
from backend.models.llm_factory import get_client
from backend.database.neo4j_client import get_neo4j_driver
from backend.utils.text_utils import strip_thinking_tags

logger = logging.getLogger(__name__)

# ── Labels pháp lý hợp lệ (tận dụng label index, không scan toàn graph) ──
_ENTITY_LABELS = (
    "Organization", "Person", "Location", "Procedure",
    "Condition", "Fee", "Penalty", "Timeframe", "Role",
    "Concept", "Term", "LegalArticle"
)
_LABEL_FILTER = "|".join(_ENTITY_LABELS)

QUERY_ENTITY_PROMPT = """Bạn là trợ lý phân tích truy vấn pháp lý.
Hãy trích xuất các Thực thể pháp lý quan trọng (Cơ quan, Tổ chức, Tên văn bản, Người, Quy trình, Thuật ngữ) từ câu hỏi.
Tối đa 5 thực thể. Chỉ trả về JSON thuần túy.

Định dạng:
{"entities": ["Thực thể 1", "Thực thể 2"]}

Câu hỏi: {query}
JSON:
"""


class EntityGraphRetriever:
    """
    Entity-First Graph Retriever.
    1. Regex: bắt số hiệu văn bản + tên Bộ/UBND (không cần LLM).
    2. LLM fallback: khi Regex trắng tay.
    3. Neo4j: Fulltext Index (nếu có) → CONTAINS fallback → chunk_ids + graph context.
    Lazy-init: driver/LLM không block khi import module.
    """

    _FULLTEXT_INDEX_NAME = "entity_name_fulltext"
    _fulltext_index_ready: bool = False   # class-level, chia sẻ mọi instance

    def __init__(self):
        self._llm_client    = None
        self._neo4j_driver  = None

    # ── Lazy properties ──
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
    # Fulltext Index — tạo lần đầu, dùng mãi
    # ------------------------------------------------------------------
    def _ensure_fulltext_index(self):
        if EntityGraphRetriever._fulltext_index_ready or not self.neo4j_driver:
            return
        labels_str = "|".join(_ENTITY_LABELS)
        cypher = (
            f"CREATE FULLTEXT INDEX {self._FULLTEXT_INDEX_NAME} IF NOT EXISTS "
            f"FOR (n:{labels_str}) ON EACH [n.name] "
            f"OPTIONS {{ indexConfig: {{ `fulltext.analyzer`: 'standard' }} }}"
        )
        try:
            with self.neo4j_driver.session() as s:
                s.run(cypher)
            EntityGraphRetriever._fulltext_index_ready = True
            logger.info(f"[Graph Search] Fulltext index '{self._FULLTEXT_INDEX_NAME}' ready.")
        except Exception as e:
            logger.warning(f"[Graph Search] Fulltext index unavailable, using CONTAINS: {e}")

    # ------------------------------------------------------------------
    # 1. EXTRACT ENTITIES
    # ------------------------------------------------------------------
    def _regex_extract(self, query: str) -> List[str]:
        """Regex nhanh: số hiệu văn bản + tên Bộ/UBND."""
        entities = []
        doc_ids = re.findall(
            r"\d+/\d{4}/(?:TT|NĐ|NĐ-CP|QH|L|QĐ|NQ)(?:-[A-ZĐa-zđÀ-ỹ]+)*",
            query, re.IGNORECASE
        )
        entities.extend(d.upper() for d in doc_ids)
        organs = re.findall(
            r"Bộ\s+[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ ]+"
            r"|UBND(?:\s+(?:tỉnh|huyện|thành phố|xã|phường)\s+[A-ZĐ][a-zàáảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ ]+)?",
            query
        )
        entities.extend(o.strip() for o in organs if o.strip())
        return list(dict.fromkeys(entities))[:5]

    def extract_entities(self, query: str) -> List[str]:
        """Regex trước → LLM fallback nếu Regex trắng tay."""
        fast = self._regex_extract(query)
        if fast:
            return fast

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
    # 2. SEARCH IN NEO4J
    # ------------------------------------------------------------------
    def search_by_entities(self, entities: List[str]) -> Dict[str, Any]:
        """
        Tìm chunk_ids + graph context bằng Entity.
        Dùng Fulltext Index nếu sẵn sàng, fallback CONTAINS.
        """
        if not entities or not self.neo4j_driver:
            return {"chunk_ids": [], "graph_context": "", "doc_numbers": [], "found_entities": []}

        self._ensure_fulltext_index()

        # Lucene query: exact phrase, fuzzy edit-distance 1
        lucene_terms = " OR ".join(f'"{e}"~1' for e in entities)

        # Query A: Chunks chứa Entity — Fulltext hoặc CONTAINS
        if EntityGraphRetriever._fulltext_index_ready:
            chunk_query = """
            CALL db.index.fulltext.queryNodes($idx, $lucene_terms)
            YIELD node AS e, score
            WHERE score > 0.5
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
            WHERE c.qdrant_id IS NOT NULL
            OPTIONAL MATCH (c)-[:PART_OF|BELONGS_TO*1..3]->(doc:Document)
            RETURN DISTINCT
                e.name              AS entity_name,
                labels(e)[0]        AS entity_type,
                c.qdrant_id         AS chunk_id,
                doc.document_number AS doc_number
            LIMIT 150
            """
        else:
            chunk_query = f"""
            UNWIND $entities AS ent_name
            MATCH (e:{_LABEL_FILTER})
            WHERE toLower(e.name) CONTAINS toLower(ent_name)
               OR toLower(ent_name) CONTAINS toLower(e.name)
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
            WHERE c.qdrant_id IS NOT NULL
            OPTIONAL MATCH (c)-[:PART_OF|BELONGS_TO*1..3]->(doc:Document)
            RETURN DISTINCT
                e.name              AS entity_name,
                labels(e)[0]        AS entity_type,
                c.qdrant_id         AS chunk_id,
                doc.document_number AS doc_number
            LIMIT 150
            """

        # Query B: Entity–Entity relations (ISSUED_BY, MANAGED_BY, ...)
        rel_query = f"""
        UNWIND $entities AS ent_name
        MATCH (e:{_LABEL_FILTER})
        WHERE toLower(e.name) CONTAINS toLower(ent_name)
        MATCH (e)-[r]->(target)
        WHERE type(r) <> 'HAS_ENTITY' AND target.name IS NOT NULL
        RETURN DISTINCT
            e.name              AS src_name,
            labels(e)[0]        AS src_type,
            type(r)             AS rel_type,
            target.name         AS tgt_name,
            labels(target)[0]   AS tgt_type
        LIMIT 50
        """

        chunk_ids   = set()
        doc_numbers = set()
        entity_info = set()
        rel_lines   = []

        try:
            with self.neo4j_driver.session() as session:
                # A — Chunk IDs + doc numbers
                run_params = (
                    {"idx": self._FULLTEXT_INDEX_NAME, "lucene_terms": lucene_terms}
                    if EntityGraphRetriever._fulltext_index_ready
                    else {"entities": entities}
                )
                for r in session.run(chunk_query, **run_params).data():
                    if r.get("entity_name"):
                        entity_info.add(f"{r.get('entity_type','Entity')}: {r['entity_name']}")
                    if r.get("chunk_id"):
                        chunk_ids.add(str(r["chunk_id"]))
                    if r.get("doc_number"):
                        doc_numbers.add(r["doc_number"])

                # B — Entity–Entity relations
                for r in session.run(rel_query, entities=entities).data():
                    src_t = r.get("src_type", "Entity")
                    tgt_t = r.get("tgt_type", "Entity")
                    rel_lines.append(
                        f"[{src_t}: {r.get('src_name','?')}]"
                        f" --({r.get('rel_type','?')})--> "
                        f"[{tgt_t}: {r.get('tgt_name','?')}]"
                    )

            # Format Graph Context
            ctx_parts = []
            if entity_info:
                ctx_parts.append("Thực thể liên quan: " + ", ".join(sorted(entity_info)))
            if rel_lines:
                ctx_parts.append("Quan hệ Thực thể:\n" + "\n".join(rel_lines[:20]))

            logger.info(
                f"  [Graph Search] entities={len(entity_info)}, "
                f"chunk_ids={len(chunk_ids)}, docs={len(doc_numbers)}, rels={len(rel_lines)}"
            )

            return {
                "chunk_ids"     : list(chunk_ids),
                "doc_numbers"   : list(doc_numbers),
                "graph_context" : "\n".join(ctx_parts),
                "found_entities": list(entity_info),
            }

        except Exception as e:
            logger.error(f"[Graph Search] Neo4j query error: {e}")
            return {"chunk_ids": [], "graph_context": "", "doc_numbers": [], "found_entities": []}

    # ------------------------------------------------------------------
    # 3. END-TO-END
    # ------------------------------------------------------------------
    def search(self, query: str) -> Dict[str, Any]:
        try:
            entities = self.extract_entities(query)
            if not entities:
                return {"chunk_ids": [], "doc_numbers": [], "graph_context": "", "found_entities": []}
            return self.search_by_entities(entities)
        except Exception as e:
            logger.error(f"[Graph Search] Unexpected error: {e}")
            return {"chunk_ids": [], "doc_numbers": [], "graph_context": "", "found_entities": []}


# Global instance — lazy-init, không block khi import
entity_retriever = EntityGraphRetriever()
