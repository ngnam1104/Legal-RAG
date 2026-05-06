import logging
import json
import re
from typing import List, Dict, Any, Optional
from backend.models.llm_factory import get_client
from backend.database.neo4j_client import get_neo4j_driver
from backend.utils.text_utils import strip_thinking_tags

logger = logging.getLogger(__name__)

QUERY_ENTITY_PROMPT = """Bạn là trợ lý phân tích truy vấn pháp lý.
Hãy trích xuất các Thực thể pháp lý quan trọng (như: Cơ quan, Tổ chức, Tên văn bản, Người, Khái niệm, Quy trình, Thuật ngữ) từ câu hỏi của người dùng.
Lưu ý: Chỉ trích xuất CỤM TỪ CỐT LÕI (VD: "Bộ Y tế", "Thông tư 19", "Quy trình cấp phép"). Không trích xuất các từ hỏi (ai, cái gì, thế nào).
Chỉ trả về JSON thuần túy, không có văn bản nào khác.

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
      1. LLM trích xuất Entity từ User Query.
      2. Fuzzy match Entity trong Neo4j.
      3. Truy xuất các Chunk (qdrant_id) hoặc Document liên quan đến Entity.
    """
    def __init__(self):
        self.llm_client = get_client()
        self.neo4j_driver = get_neo4j_driver()

    def extract_entities(self, query: str) -> List[str]:
        """Dùng LLM (nhỏ/nhanh) để lấy entity từ câu hỏi."""
        prompt = QUERY_ENTITY_PROMPT.format(query=query)
        try:
            # Gửi LLM, tắt thinking để tốc độ nhanh nhất
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            if not response:
                return []
            
            response = strip_thinking_tags(response)
            cleaned = re.sub(r'```json\n|\n```|```', '', response).strip()
            data = json.loads(cleaned)
            return data.get("entities", [])
        except Exception as e:
            logger.warning(f"Error parsing entity extraction: {e}")
            return []

    def search_by_entities(self, entities: List[str]) -> Dict[str, Any]:
        """
        Quét Neo4j để lấy các chunk_id và graph context liên kết với các entities.
        """
        if not entities or not self.neo4j_driver:
            return {"chunk_ids": [], "graph_context": "", "doc_numbers": []}

        # Cypher: Tìm Entity bằng RegEx (Fuzzy/Substring) không phân biệt hoa thường
        # Sau đó lấy:
        #  1) Các chunk_id nối trực tiếp bằng [:HAS_ENTITY]
        #  2) Các doc_number nối bằng các quan hệ khác (ISSUED_BY, MANAGED_BY,...)
        cypher_query = """
        UNWIND $entities AS ent_name
        // Escape regex special chars
        WITH ent_name, replace(replace(replace(replace(ent_name, '(', '\\\\('), ')', '\\\\)'), '[', '\\\\['), ']', '\\\\]') AS safe_ent
        
        MATCH (e) 
        WHERE e.name =~ ('(?i).*' + safe_ent + '.*') 
           OR safe_ent =~ ('(?i).*' + e.name + '.*')
        
        // 1. Chunks chứa Entity
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
        WHERE c.qdrant_id IS NOT NULL
        
        // 2. Documents liên quan đến Entity qua quan hệ hành động (VD: ISSUED_BY)
        OPTIONAL MATCH (doc:Document)-[r]->(e)
        WHERE type(r) <> 'HAS_ENTITY'
        
        RETURN 
            e.name AS entity_name, 
            labels(e)[0] AS entity_type, 
            c.qdrant_id AS chunk_id,
            doc.document_number AS doc_number,
            type(r) AS doc_relation
        LIMIT 150
        """
        
        chunk_ids = set()
        doc_numbers = set()
        entity_info = set()
        doc_rels = set()
        
        try:
            with self.neo4j_driver.session() as session:
                results = session.run(cypher_query, entities=entities).data()
                
                for r in results:
                    ent_display = f"{r.get('entity_type', 'Entity')}: {r.get('entity_name', '?')}"
                    entity_info.add(ent_display)
                    
                    if r.get("chunk_id"):
                        chunk_ids.add(str(r["chunk_id"]))
                    
                    if r.get("doc_number"):
                        dnum = r["doc_number"]
                        doc_numbers.add(dnum)
                        if r.get("doc_relation"):
                            doc_rels.add(f"Văn bản [{dnum}] --({r['doc_relation']})--> [{ent_display}]")
                        
            # Format Graph Context bổ sung
            context_lines = []
            if entity_info:
                context_lines.append("Thực thể liên quan tìm thấy trong Graph: " + ", ".join(entity_info))
            if doc_rels:
                context_lines.append("Quan hệ Văn bản - Thực thể:")
                context_lines.extend(doc_rels)
                
            graph_context = "\n".join(context_lines) if context_lines else ""
            
            return {
                "chunk_ids": list(chunk_ids),
                "doc_numbers": list(doc_numbers),
                "graph_context": graph_context,
                "found_entities": list(entity_info)
            }
            
        except Exception as e:
            logger.error(f"Error executing graph entity search: {e}")
            return {"chunk_ids": [], "graph_context": "", "doc_numbers": []}

    def search(self, query: str) -> Dict[str, Any]:
        """
        End-to-end Entity Graph Search.
        """
        entities = self.extract_entities(query)
        logger.info(f"  [Graph Search] Extracted entities: {entities}")
        
        if not entities:
            return {"chunk_ids": [], "doc_numbers": [], "graph_context": ""}
            
        return self.search_by_entities(entities)

# Global instance
entity_retriever = EntityGraphRetriever()
