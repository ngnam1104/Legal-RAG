"""
utils_legal.py — Legal-specific utilities for the Agent pipeline.
==================================================================
Hợp nhất từ: utils_legal_qa.py + utils_pl.py (GraphRAG helpers)
"""

from typing import List, Dict, Any, Optional
import json
import re
import time

from backend.models.llm_factory import chat_completion
from backend.retrieval.hybrid_search import retriever
import os
from backend.utils.text_utils import extract_thinking_and_answer, strip_thinking_tags

# ANSWER_PROMPT is imported from prompt.py by consumers directly.
# Re-export here for backward compatibility.
from backend.prompt import ANSWER_PROMPT

from backend.config import _VI_TRANSLATION_MAP

def translate_schema(text: str) -> str:
    return _VI_TRANSLATION_MAP.get(text, text)



# =====================================================================
# BUILD LEGAL CONTEXT — Xây dựng ngữ cảnh pháp lý cho LLM
# =====================================================================

def build_legal_context(hits: List[Dict[str, Any]], file_chunks: List[Dict[str, Any]] = None, max_chars: int = None, graph_context: Dict[str, Any] = None) -> str:
    """
    Xây dựng Ngữ cảnh Pháp lý (Legal Context) sử dụng 5 chiến thuật:
    1. Document TOC Injection: Đưa Mục lục văn bản (từ Neo4j) vào đầu context chống Lost-in-the-Middle.
    2. XML Context Separation: Tách biệt <tai_lieu_tam> (upload) và <tai_lieu_db> (hệ thống).
    3. Lost-in-the-Middle Reordering: Đưa hits quan trọng nhất ra 2 đầu.
    4. Hard Character Limit: Đảm bảo không vượt quá cửa sổ ngữ cảnh LLM.
    5. Sibling Expansion: Ghép sibling texts từ Bottom-Up traversal (Neo4j).
    """
    if max_chars is None:
        max_chars = int(os.environ.get("MAX_CONTEXT_CHARS", 50000))

    context_parts = []
    current_chars = 0
    doc_counter = 0
    
    # --- Chiến thuật Đặc Biệt: Metadata Information (Signer/BasedOn/Year) ---
    if graph_context:
        meta_info = []
        if graph_context.get("year_info"):
            meta_info.append(graph_context["year_info"])
        if graph_context.get("signer_info"):
            meta_info.append(graph_context["signer_info"])
        if graph_context.get("based_on_info"):
            meta_info.append(graph_context["based_on_info"])
        if graph_context.get("admin_metadata"):
            meta_info.append(graph_context["admin_metadata"])
            
        if meta_info:
            meta_block = "<thong_tin_van_ban_chinh_xac>\n" + "\n".join(meta_info) + "\n</thong_tin_van_ban_chinh_xac>\n"
            context_parts.append(meta_block)
            current_chars += len(meta_block)
    
    # --- Chiến thuật 0: Document TOC (từ Neo4j graph_context) ---
    if graph_context and graph_context.get("document_toc"):
        toc_text = graph_context["document_toc"]
        toc_block = f"<muc_luc_van_ban>\n{toc_text}\n</muc_luc_van_ban>\n"
        context_parts.append(toc_block)
        current_chars += len(toc_block)

    # --- [TÀI LIỆU TẢI LÊN]: Bọc trong thẻ <tai_lieu_tam> ---
    if file_chunks:
        context_parts.append("<tai_lieu_tam>")
        for idx, f_chunk in enumerate(file_chunks, start=1):
            text = f_chunk.get("text_to_embed", f_chunk.get("unit_text", ""))
            # Nén trắng
            text = re.sub(r'\s*\n\s*', '\n', text)
            text = re.sub(r' {2,}', ' ', text).strip()
            chunk_info = f"[File Chunk {idx}]\n{text}\n"
            context_parts.append(chunk_info)
        context_parts.append("</tai_lieu_tam>\n")

    # --- [CƠ SỞ PHÁP LÝ TỪ HỆ THỐNG DB]: Bọc trong thẻ <tai_lieu_db> ---
    # Chiến thuật 1: Lost-in-the-Middle Reordering
    if hits and len(hits) > 2:
        sorted_hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
        reordered = [None] * len(sorted_hits)
        
        left = 0
        right = len(sorted_hits) - 1
        for i, hit in enumerate(sorted_hits):
            if i % 2 == 0:
                reordered[left] = hit
                left += 1
            else:
                reordered[right] = hit
                right -= 1
        hits = reordered

    context_parts.append("<tai_lieu_db>")
    
    # Chiến thuật 2: XML Context Injection
    for hit in hits:
        doc_counter += 1
        ref = hit.get("article_ref") or hit.get("reference_tag") or "N/A"
        doc_number = hit.get("document_number") or "N/A"
        title = hit.get("title") or "N/A"
        ref_citation = hit.get("reference_citation") or ref
        text = hit.get("text", "")
        # Nén trắng
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        is_appendix = hit.get("is_appendix", False)
        base_laws = hit.get("base_laws", [])

        if is_appendix:
            loai = "PHỤ LỤC / BẢNG BIỂU ĐÍNH KÈM - Đây là dữ liệu chi tiết, định mức hoặc biểu mẫu."
        else:
            loai = "NỘI DUNG CHÍNH - Quy định trực tiếp trong văn bản pháp luật."

        base_law_xml = f"\n    <can_cu_phap_ly>{', '.join(base_laws)}</can_cu_phap_ly>" if base_laws else ""
        
        chunk_xml = (
            f'<can_cu id="{doc_counter}">\n'
            f'  <metadata>\n'
            f'    <nguon>{doc_number} ({title})</nguon>\n'
            f'    <vi_tri>{ref_citation}</vi_tri>\n'
            f'    <loai_noi_dung>{loai}</loai_noi_dung>{base_law_xml}\n'
            f'  </metadata>\n'
            f'  <noi_dung>\n'
            f'    {text}\n'
            f'  </noi_dung>\n'
            f'</can_cu>\n'
        )

        context_parts.append(chunk_xml)
    
    # --- Sibling texts từ Neo4j Bottom-Up Expansion ---
    if graph_context and graph_context.get("sibling_texts"):
        for sib_text in graph_context["sibling_texts"][:10]:
            sib_text_clean = re.sub(r'\s*\n\s*', '\n', sib_text)
            sib_block = f'<can_cu_bo_sung>\n  {sib_text_clean}\n</can_cu_bo_sung>\n'
            context_parts.append(sib_block)
    
    context_parts.append("</tai_lieu_db>")

    return "\n".join(context_parts)


# =====================================================================
# LEGAL REFERENCE EXTRACTION
# =====================================================================

def extract_legal_references(text: str) -> List[str]:
    """
    Sử dụng Regex để phát hiện các Điều khoản hoặc Phụ lục được nhắc tới.
    """
    patterns = [
        r"Điều\s+\d+",
        r"Phụ\s+lục\s+[\d\w\-]+",
        r"Phụ\s+lục\s+[IVXLCDM]+",
        r"Mẫu\s+số\s+\d+",
        r"Khoản\s+\d+\s+Điều\s+\d+"
    ]
    
    found = []
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        for m in matches:
            norm = m.strip().replace("  ", " ").capitalize()
            if "Khoản" in norm and "Điều" in norm:
                norm = "Điều" + norm.split("Điều")[1]
            if norm not in found:
                found.append(norm)
    return found


def resolve_recursive_references(primary_hits: List[Dict[str, Any]], max_supplemental_chars: int = 15000) -> List[Dict[str, Any]]:
    """
    Duyệt qua các hits hiện tại, tìm references và truy xuất thêm.
    Fix 6A: Giới hạn tổng ký tự bổ sung để tránh Context Overflow.
    """
    all_hits = list(primary_hits)
    seen_refs = set()
    
    for h in primary_hits:
        doc_num = h.get("document_number", "")
        art_ref = h.get("article_ref", "")
        if doc_num and art_ref:
            seen_refs.add(f"{doc_num}::{art_ref}")

    new_hits = []
    total_supplemental_chars = 0
    
    for h in primary_hits:
        text = h.get("text", "")
        doc_num = h.get("document_number", "")
        if not doc_num: continue
        
        refs = extract_legal_references(text)
        for ref_id in refs:
            clean_ref = ref_id.strip().lower()
            ref_key = f"{doc_num}::{clean_ref}"
            
            if ref_key not in seen_refs:
                if total_supplemental_chars >= max_supplemental_chars:
                    print(f"      ⚠️ [Recursive Ref] Reached supplemental char limit ({max_supplemental_chars}). Stopping.")
                    break
                    
                print(f"      🔗 Found internal reference: {ref_id} in {doc_num}. Fetching...")
                supp_hits = retriever.retrieve_specific_reference(doc_num, ref_id)
                if supp_hits:
                    for sh in supp_hits:
                        sh["is_supplemental"] = True
                        sh["score"] = 1.0
                        total_supplemental_chars += len(sh.get("text", ""))
                        new_hits.append(sh)
                        seen_refs.add(ref_key)
                        
    return all_hits + new_hits[:10]


# =====================================================================
# FILTER CITED REFERENCES — Lọc references thực sự được cite
# =====================================================================

def filter_cited_references(answer_text: str, refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lọc danh sách references chỉ giữ lại các chunk thực sự được trích dẫn trong câu trả lời.
    Giúp UI chỉ hiển thị nguồn tham chiếu có ý nghĩa thay vì toàn bộ top-k.
    
    Logic khớp:
      1. document_number xuất hiện trong answer (VD: "51/2025/TT-BYT")
      2. article_ref xuất hiện trong answer (VD: "Điều 5")
      3. Partial match — chỉ phần số hiệu ngắn (VD: "51/2025")
    Fallback: Nếu không phát hiện citation nào → giữ top 3 theo score.
    """
    if not answer_text or not refs:
        return refs

    answer_lower = answer_text.lower()
    cited = []

    for ref in refs:
        doc_num = ref.get("document_number", "")
        article = ref.get("article", "") or ""
        title = ref.get("title", "")
        text_preview = ref.get("text_preview", "")

        is_cited = False

        # Check 1: Document match (VD: "51/2025/TT-BYT")
        has_doc_match = False
        if doc_num and len(doc_num) > 3:
            if doc_num.lower() in answer_lower:
                has_doc_match = True
            else:
                parts = doc_num.split("/")
                if len(parts) >= 2:
                    short_num = f"{parts[0]}/{parts[1]}"
                    if short_num.lower() in answer_lower:
                        has_doc_match = True

        # Check 2: Article reference match (VD: "Điều 5")
        has_article_match = False
        if article:
            article_clean = article.strip().lower()
            if article_clean and len(article_clean) > 2 and article_clean in answer_lower:
                has_article_match = True

        # Quyết định Is Cited (Strict mode)
        if has_article_match:
            is_cited = True
        elif has_doc_match and not article:
            is_cited = True
            
        # Check 3: Text snippet match
        if not is_cited and title and len(title) > 15:
            title_words = title.split()
            if len(title_words) > 3:
                title_fragment = " ".join(title_words[:5]).lower()
                if title_fragment in answer_lower:
                    is_cited = True

        if is_cited:
            # --- [SANITIZATION - LÀM SẠCH DỮ LIỆU UI] ---
            if " > " in article:
                parts = article.split(" > ")
                if len(parts[0]) > 40:
                    ref["article"] = parts[-1]
            
            clean_text = text_preview.strip()
            art_label = ref["article"].split(" > ")[-1].strip()
            if clean_text.lower().startswith(art_label.lower()):
                clean_text = re.sub(r'^' + re.escape(art_label) + r'[\.\s\:\-]+', '', clean_text, flags=re.IGNORECASE).strip()
            ref["text_preview"] = clean_text
            
            cited.append(ref)

    # Fallback: Nếu LLM tóm tắt hoàn toàn → giữ duy nhất 1 nguồn đáng tin nhất
    if not cited and refs:
        fallback_ref = sorted(refs, key=lambda x: x.get("score", 0), reverse=True)[0]
        if " > " in fallback_ref.get("article", ""):
            parts = fallback_ref["article"].split(" > ")
            if len(parts[0]) > 40: fallback_ref["article"] = parts[-1]
        return [fallback_ref]

    return cited


# =====================================================================
# GRAPHRAG HELPERS — Neo4j 2-hop Subgraph Fetch + Format
# =====================================================================

def fetch_related_graph(entity_ids: List[str]) -> Dict[str, Any]:
    """
    Kéo toàn bộ subgraph từ Neo4j cho danh sách entity_ids:
      - Quan hệ văn bản (AMENDS, REPLACES, BASED_ON ...): qua doc-level relations
      - Thực thể tự do (HAS_ENTITY → Organization, Person, Fee...): qua enrich_chunk_entities
      - Quan hệ thực thể động (RESPONSIBLE_FOR, SIGNED_BY, AFFECTS...): via node_relations

    Returns dict {"doc_relations": [...], "entities": [...], "node_relations": [...]}
    """
    if not entity_ids:
        return {}

    from backend.database.neo4j_client import get_neo4j_driver
    driver = get_neo4j_driver()
    if not driver:
        return {}

    # ── Query 1: Document-level Relations (AMENDS, REPLACES, BASED_ON, etc.) ──
    doc_rel_query = """
    UNWIND $ids AS cid
    MATCH (c) WHERE c.qdrant_id = cid OR c.id = cid
    // Lấy quan hệ thuộc về Document node cha
    OPTIONAL MATCH (c)-[:BELONGS_TO|PART_OF*1..3]->(doc:Document)
    OPTIONAL MATCH (doc)-[dr]->(other_doc:Document)
    WHERE type(dr) IN ['AMENDS','REPLACES','REPEALS','BASED_ON','GUIDES','APPLIES','ISSUED_WITH','ASSIGNS','CORRECTS']
    WITH c, cid, doc, other_doc,
         type(dr) AS rel_type,
         dr.target_article AS target_article,
         dr.chunk_text AS chunk_text
    WHERE rel_type IS NOT NULL
    RETURN
        coalesce(doc.document_number, cid) AS source,
        other_doc.document_number AS target,
        other_doc.title AS target_title,
        rel_type,
        target_article,
        chunk_text
    LIMIT 50
    """

    # ── Query 2: Free-form Entities linked to chunk nodes (HAS_ENTITY) ──
    entity_query = """
    UNWIND $ids AS cid
    MATCH (c) WHERE c.qdrant_id = cid OR c.id = cid
    OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
    WHERE e.name IS NOT NULL
    RETURN cid AS chunk_id,
           labels(e)[0] AS entity_type,
           e.name AS entity_name
    LIMIT 200
    """

    # ── Query 3: Free-form Node Relations (RESPONSIBLE_FOR, SIGNED_BY, etc.) ──
    node_rel_query = """
    UNWIND $ids AS cid
    MATCH (c) WHERE c.qdrant_id = cid OR c.id = cid
    OPTIONAL MATCH (c)-[:HAS_ENTITY]->(src_ent)
    OPTIONAL MATCH (src_ent)-[nr]->(tgt)
    WHERE nr IS NOT NULL AND type(nr) NOT IN ['HAS_ENTITY']
    RETURN
        labels(src_ent)[0] AS source_type,
        src_ent.name AS source_node,
        type(nr) AS relationship,
        labels(tgt)[0] AS target_type,
        tgt.name AS target_node,
        nr.chunk_text AS chunk_text
    LIMIT 100
    """

    # ── Query 4: Cross-Document Entity Context (Sibling Documents) ──
    # Tự động kết nối các văn bản thông qua Procedure, Term chung (Dù không có liên kết trực tiếp)
    sibling_query = """
    UNWIND $ids AS cid
    MATCH (c) WHERE c.qdrant_id = cid OR c.id = cid
    MATCH (c)-[:HAS_ENTITY]->(e)
    WHERE labels(e)[0] IN ['Procedure', 'Term', 'Condition', 'Organization'] AND e.name IS NOT NULL
    // Tìm các chunk khác (khác Document) cũng chứa entity này
    MATCH (other_c)-[:HAS_ENTITY]->(e)
    WHERE other_c.qdrant_id <> cid AND coalesce(other_c.qdrant_id, other_c.id) <> coalesce(c.qdrant_id, c.id)
    OPTIONAL MATCH (other_c)-[:PART_OF|BELONGS_TO*1..3]->(other_doc:Document)
    OPTIONAL MATCH (c)-[:PART_OF|BELONGS_TO*1..3]->(doc:Document)
    WHERE other_doc IS NOT NULL AND other_doc.document_number <> coalesce(doc.document_number, 'N/A')
    RETURN 
        e.name AS shared_entity,
        labels(e)[0] AS entity_type,
        other_doc.document_number AS sibling_doc,
        other_c.text AS sibling_text
    LIMIT 20
    """

    result = {"doc_relations": [], "entities": [], "node_relations": [], "sibling_texts": []}

    try:
        with driver.session() as session:
            # Doc relations
            for r in session.run(doc_rel_query, ids=entity_ids).data():
                if r.get("target"):
                    result["doc_relations"].append(r)

            # Free-form entities
            for r in session.run(entity_query, ids=entity_ids).data():
                if r.get("entity_name"):
                    result["entities"].append(r)

            # Node relations
            for r in session.run(node_rel_query, ids=entity_ids).data():
                if r.get("source_node") and r.get("target_node"):
                    result["node_relations"].append(r)
                    
            # Sibling texts
            for r in session.run(sibling_query, ids=entity_ids).data():
                if r.get("sibling_text") and r.get("sibling_doc"):
                    ent = r["shared_entity"]
                    doc_num = r["sibling_doc"]
                    text = r["sibling_text"]
                    formatted_sib = f"[Liên kết qua {r['entity_type']}: {ent}] - Nguồn: {doc_num}\nNội dung: {text[:1500]}"
                    if formatted_sib not in result["sibling_texts"]:
                        result["sibling_texts"].append(formatted_sib)

    except Exception as e:
        import logging
        logging.getLogger("utils_legal").warning(f"fetch_related_graph error: {e}")

    return result


def format_graph_context(subgraph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format subgraph dict (từ fetch_related_graph) thành:
      - nodes:          list[str] — mô tả các nút văn bản
      - edges:          list[str] — quan hệ pháp lý (AMENDS, REPLACES...)
      - entity_context: str       — thực thể tự do (Organization, Person, Fee...)
      - node_rel_lines: list[str] — quan hệ thực thể động (RESPONSIBLE_FOR...)
    """
    nodes = set()
    edges = []
    entity_lines = []
    node_rel_lines = []

    # ── 1. Doc-level relations ──
    for dr in subgraph.get("doc_relations", []):
        source = dr.get("source", "?")
        target = dr.get("target", "?")
        target_title = dr.get("target_title") or ""
        rel_type = translate_schema(dr.get("rel_type", "RELATED"))
        target_article = dr.get("target_article") or ""
        chunk_text = dr.get("chunk_text") or ""

        nodes.add(f"Văn bản: {source}")
        target_label = f"{target} ({target_title})" if target_title else target
        nodes.add(f"Văn bản: {target_label}")

        article_info = f" [{target_article}]" if target_article else ""
        evidence = f' — Bằng chứng: "{chunk_text[:120]}"' if chunk_text else ""
        edges.append(f"[{source}] --({rel_type})--> [{target}]{article_info}{evidence}")

    # ── 2. Free-form Entities (HAS_ENTITY) ──
    # Group by entity_type
    entity_by_type: Dict[str, List[str]] = {}
    for ent in subgraph.get("entities", []):
        etype = translate_schema(ent.get("entity_type") or "Entity")
        ename = ent.get("entity_name", "")
        if ename:
            entity_by_type.setdefault(etype, [])
            if ename not in entity_by_type[etype]:
                entity_by_type[etype].append(ename)

    for etype, names in entity_by_type.items():
        entity_lines.append(f"{etype}: {', '.join(names)}")

    # ── 3. Node Relations (RESPONSIBLE_FOR, SIGNED_BY, AFFECTS...) ──
    for nr in subgraph.get("node_relations", []):
        src = nr.get("source_node", "?")
        src_type = translate_schema(nr.get("source_type") or "Entity")
        tgt = nr.get("target_node", "?")
        tgt_type = translate_schema(nr.get("target_type") or "Entity")
        rel = translate_schema(nr.get("relationship", "RELATED"))
        chunk_text = nr.get("chunk_text") or ""
        evidence = f' — "{chunk_text[:100]}"' if chunk_text else ""
        node_rel_lines.append(f"[{src_type}: {src}] --({rel})--> [{tgt_type}: {tgt}]{evidence}")

    entity_context = "\n".join(entity_lines) if entity_lines else ""

    return {
        "nodes": list(nodes),
        "edges": edges,
        "entity_context": entity_context,
        "node_rel_lines": node_rel_lines,
    }


def _get_node_short(node):
    if not node: return "?"
    if "document_number" in node: return node["document_number"]
    if "name" in node: return node["name"]
    return str(node.get("id"))

