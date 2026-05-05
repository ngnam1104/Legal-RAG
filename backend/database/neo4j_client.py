"""
Neo4j Graph Database ingestion module.

Kiến trúc Dynamic Tree:
  Document → Article → Clause → Chunk
  Document → Authority, Signer, LegalType, Sector
  Document → BASED_ON/AMENDS/REPLACES/REPEALS → Document(ref)

Leaf Level Logic (3 kịch bản):
  A: Article là leaf (Điều không có Khoản) → gán qdrant_id + text vào Article
  B: Clause là leaf (Khoản nguyên vẹn) → gán qdrant_id + text vào Clause
  C: Chunk là leaf (Khoản bị tách/bảng biểu) → tạo node Chunk → PART_OF parent
"""
import re
from neo4j import GraphDatabase
import os
from backend.ingestion.chunker.metadata import normalize_doc_key


def get_neo4j_driver():
    try:
        neo4j_uri = os.environ.get("NEO4J_URI")
        if not neo4j_uri:
            print("⚠️ NEO4J_URI missing, Graph Feature Disabled.")
            return None
        
        return GraphDatabase.driver(
            neo4j_uri,
            auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
        )
    except Exception as e:
        print(f"Lỗi khởi tạo Neo4j: {e}")
        return None


def init_neo4j_constraints(driver):
    """Thiết lập các ràng buộc duy nhất để bảo vệ toàn vẹn dữ liệu Graph."""
    queries = [
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;",
        "CREATE CONSTRAINT legalarticle_id_unique IF NOT EXISTS FOR (a:LegalArticle) REQUIRE a.id IS UNIQUE;",
        "CREATE CONSTRAINT clause_id_unique IF NOT EXISTS FOR (cl:Clause) REQUIRE cl.id IS UNIQUE;",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;",
        "CREATE INDEX document_number_index IF NOT EXISTS FOR (d:Document) ON (d.document_number);",
        "CREATE CONSTRAINT sector_name_unique IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE;",
        "CREATE CONSTRAINT legaltype_name_unique IF NOT EXISTS FOR (lt:LegalType) REQUIRE lt.name IS UNIQUE;"
    ]
    # Constraints cho các loại entity chuẩn (Unified Extractor)
    entity_labels = [
        "Organization", "Person", "Location", "Procedure", "Condition",
        "Fee", "Penalty", "Timeframe", "Role", "Concept"
    ]
    for label in entity_labels:
        prop = "name"
        constraint_name = f"{label.lower()}_name_unique"
        queries.append(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE;"
        )
    with driver.session() as session:
        for query in queries:
            try:
                session.run(query)
            except Exception as e:
                print(f"⚠️ Constraint warning: {e}")
    print("🛡️ Hệ thống Constraints (bao gồm 10 Entity Label) đã sẵn sàng!")


def enrich_reference_nodes(driver, batch_chunks, meta_by_docnum_lookup=None):
    """Bổ sung metadata cho các nốt Document tham chiếu (nốt ảo chỉ có document_number)."""
    if not meta_by_docnum_lookup:
        return

    missing_docs = set()
    for chunk in batch_chunks:
        meta = chunk.get("neo4j_metadata", {})
        all_refs = (meta.get("legal_basis_refs", []) +
                    meta.get("amended_refs", []) +
                    meta.get("replaced_refs", []) +
                    meta.get("repealed_refs", []))
        for ref in all_refs:
            dnum = ref.get("doc_number")
            if dnum and dnum not in ["unknown", "N/A"]:
                missing_docs.add(dnum.strip())
                
        # Thêm target_doc từ ontology_relations để vét sạch các node REF_
        ontology_rels = meta.get("ontology_relations", [])
        for rel in ontology_rels:
            dnum = rel.get("target_doc")
            if dnum and dnum not in ["unknown", "N/A"]:
                missing_docs.add(dnum.strip())
                
        # Thêm cả cross-doc target_doc nếu có
        for rel in ontology_rels:
            if rel.get("is_cross_doc"):
                src_num = rel.get("source_doc")
                if src_num and src_num not in ["unknown", "N/A"]:
                    missing_docs.add(src_num.strip())

    enrich_payloads = []
    seen_ids = set()
    for dnum in missing_docs:
        key = normalize_doc_key(dnum)
        if key in meta_by_docnum_lookup:
            hf = meta_by_docnum_lookup[key]

            # Use canonical doc_number and id to avoid uniqueness constraint violation
            canonical_doc_num = hf.get("document_number") or dnum
            doc_id = str(hf.get("id", f"REF_{canonical_doc_num}"))
            
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            p_date = str(hf.get("issuance_date", "")).split("T")[0] if hf.get("issuance_date") else ""

            import re
            m_yr = re.search(r"\b((?:19|20)\d{2})\b", p_date)
            if not m_yr: m_yr = re.search(r"(\d{4})", p_date)
            year_val = m_yr.group(1) if m_yr else str(hf.get("year", ""))

            enrich_payloads.append({
                "document_number": canonical_doc_num,
                "id": doc_id,
                "title": hf.get("title", ""),
                "url": hf.get("url", ""),
                "p_date": p_date,
                "eff_date": str(hf.get("effective_date", p_date)),
                "doc_toc": hf.get("document_toc", ""),
                "year": year_val,
                "doc_status": "Đang có hiệu lực"
            })

    if not enrich_payloads:
        return

    query = """
    UNWIND $batch AS doc
    MERGE (p:Document {id: doc.id})
    WITH p, doc
    WHERE p.title IS NULL OR p.title = '' OR p.id STARTS WITH 'REF_'
    SET p.document_number = COALESCE(p.document_number, doc.document_number),
        p.title = doc.title,
        p.url = doc.url,
        p.promulgation_date = doc.p_date,
        p.effective_date = doc.eff_date,
        p.document_toc = doc.doc_toc,
        p.year = doc.year,
        p.doc_status = COALESCE(p.doc_status, doc.doc_status),
        p.is_full_text = false
    """
    with driver.session() as session:
        batch_size = 2000
        for i in range(0, len(enrich_payloads), batch_size):
            session.run(query, batch=enrich_payloads[i:i+batch_size])


def build_neo4j(driver, batch_chunks, meta_by_docnum_lookup=None):
    """
    Push batch of chunks into Neo4j using UNWIND for massive speed up với Đồ Thị Động.
    Đồng bộ 100% với Cypher query trong notebook.
    """
    if not batch_chunks or not driver:
        return

    # Chạy ENRICHMENT trước tiên
    enrich_reference_nodes(driver, batch_chunks, meta_by_docnum_lookup)

    # Bộ nhớ đánh dấu các Điều/Khoản đã nạp text
    seen_nodes = set()

    # 1. ĐÓNG GÓI DATA TẠI PYTHON: Xác định Vai Trò (Leaf Level)
    params = []
    for chunk in batch_chunks:
        meta = chunk.get("neo4j_metadata", chunk.get("metadata", {}))
        p_date = meta.get("promulgation_date", "")
        year = meta.get("year", "N/A")

        art_ref = meta.get("article_ref")
        cl_ref_raw = meta.get("clause_ref")
        is_table = meta.get("is_table", False)

        # Logic Đồ thị Động: Phân loại Mức độ Lá (Leaf Level)
        is_article_leaf = False
        is_clause_leaf = False
        is_chunk_leaf = False
        base_clause_ref = None

        if art_ref:
            if not cl_ref_raw:
                node_key = f"LA_{meta.get('document_id')}_{art_ref}"
                if is_table or node_key in seen_nodes:
                    is_chunk_leaf = True
                else:
                    is_article_leaf = True
                    seen_nodes.add(node_key)
            else:
                if "(tiếp theo)" in cl_ref_raw or "[" in cl_ref_raw:
                    is_chunk_leaf = True
                    base_clause_ref = cl_ref_raw.replace(" (tiếp theo)", "").replace("[", "").replace(" tiếp theo]", "").strip()
                else:
                    base_clause_ref = cl_ref_raw
                    node_key = f"CL_{meta.get('document_id')}_{art_ref}_{base_clause_ref}"
                    if is_table or node_key in seen_nodes:
                        is_chunk_leaf = True
                    else:
                        is_clause_leaf = True
                        seen_nodes.add(node_key)
        else:
            is_chunk_leaf = True


        # Normalize ontology relations
        ontology_relations = meta.get("ontology_relations") or []
        for rel in ontology_relations:
            # Normalize target_doc
            tdoc = rel.get("target_doc")
            if tdoc and meta_by_docnum_lookup:
                key = normalize_doc_key(tdoc)
                if key in meta_by_docnum_lookup:
                    rel["target_doc"] = meta_by_docnum_lookup[key].get("document_number") or tdoc
            # Normalize source_doc
            sdoc = rel.get("source_doc")
            if sdoc and meta_by_docnum_lookup:
                key = normalize_doc_key(sdoc)
                if key in meta_by_docnum_lookup:
                    rel["source_doc"] = meta_by_docnum_lookup[key].get("document_number") or sdoc

        # Normalize legal basis refs
        refs = []
        for r in (meta.get("legal_basis_refs") or []):
            rdoc = r.get("doc_number")
            if rdoc and meta_by_docnum_lookup:
                key = normalize_doc_key(rdoc)
                if key in meta_by_docnum_lookup:
                    r["doc_number"] = meta_by_docnum_lookup[key].get("document_number") or rdoc
            
            # Check if covered by ontology_relations
            if r.get("doc_number") and r["doc_number"] not in {
                rel.get("target_doc") for rel in ontology_relations if rel.get("edge_label") == "BASED_ON"
            }:
                refs.append(r)

        params.append({
            "doc_id": meta.get("document_id"),
            "doc_num": meta.get("document_number") or "N/A",
            "title": meta.get("title") or "",
            "l_type": meta.get("legal_type") or "N/A",
            "p_date": p_date or "",
            "year": year,
            "url": meta.get("url") or "",
            "auth_name": meta.get("issuing_authority") or "",
            "signer_name": meta.get("signer_name") or "",
            "signer_id": meta.get("signer_id"),
            "eff_date": meta.get("effective_date") or "",
            "doc_status": meta.get("doc_status") or "Đang có hiệu lực",
            "doc_toc": meta.get("document_toc") or "",
            "sectors": meta.get("legal_sectors") or [],
            "ontology_relations": ontology_relations,

            # legal_basis_refs: chỉ giữ các ref CHƯА được ontology_relations BASED_ON phủ đủ
            "refs": refs,

            # Thông tin Phân cấp và Lá
            "chunk_id": chunk.get("chunk_id"),
            "chunk_idx": meta.get("chunk_index"),
            "chap_ref": meta.get("chapter_ref") or "",
            "art_ref": art_ref or "",
            "base_cl_ref": base_clause_ref,
            "raw_cl_ref": cl_ref_raw or "",

            # Cờ lá
            "is_art_leaf": is_article_leaf,
            "is_cl_leaf": is_clause_leaf,
            "is_chk_leaf": is_chunk_leaf,

            "is_table": meta.get("is_table", False),
            "ref_cit": meta.get("reference_citation") or "",
            "text": chunk.get("chunk_text") or ""
        })

    # 2. CYPHER QUERY: Cấu trúc Đồ thị Động (Dynamic Tree)
    query = """
    UNWIND $batch AS row

    // A. Merge Document chính
    MERGE (d:Document {id: row.doc_id})
    SET d.document_number = COALESCE(row.doc_num, d.document_number, 'N/A'),
        d.title = COALESCE(row.title, d.title, ''),
        d.promulgation_date = COALESCE(row.p_date, d.promulgation_date, ''),
        d.effective_date = COALESCE(row.eff_date, d.effective_date, ''),
        d.year = COALESCE(row.year, d.year, 'N/A'),
        d.url = COALESCE(row.url, d.url, ''),
        d.doc_status = COALESCE(row.doc_status, d.doc_status, 'Đang có hiệu lực'),
        d.document_toc = COALESCE(row.doc_toc, d.document_toc, ''),
        d.is_full_text = true

    // B. Merge Authority (Organization with type Authority)
    FOREACH (auth IN CASE WHEN row.auth_name IS NOT NULL AND row.auth_name <> 'N/A' AND row.auth_name <> '' THEN [1] ELSE [] END |
        MERGE (a:Organization {name: row.auth_name})
        ON CREATE SET a.type = 'Authority'
        MERGE (d)-[:ISSUED_BY]->(a)
    )

    // C. Merge Signer (Person)
    FOREACH (signer IN CASE WHEN row.signer_name IS NOT NULL AND row.signer_name <> '' THEN [1] ELSE [] END |
        MERGE (s:Person {name: row.signer_name})
        SET s.signer_id = CASE WHEN row.signer_id IS NOT NULL THEN row.signer_id ELSE s.signer_id END
        MERGE (d)-[:SIGNED_BY]->(s)
    )

    // D. Merge LegalType
    FOREACH (ltype IN CASE WHEN row.l_type IS NOT NULL AND row.l_type <> 'N/A' AND row.l_type <> '' THEN [1] ELSE [] END |
        MERGE (lt:LegalType {name: row.l_type})
        MERGE (d)-[:HAS_TYPE]->(lt)
    )

    // F. Merge Sectors
    FOREACH (sec_name IN row.sectors |
        MERGE (sec:Sector {name: sec_name})
        MERGE (d)-[:HAS_SECTOR]->(sec)
    )

    // E. XÂY DỰNG CẤU TRÚC ĐỒ THỊ ĐỘNG (DYNAMIC TREE)

    // Nhánh 1: XỬ LÝ ĐIỀU (LEGAL ARTICLE)
    FOREACH (_o IN CASE WHEN row.art_ref IS NOT NULL AND row.art_ref <> '' THEN [1] ELSE [] END |
        MERGE (art:LegalArticle {id: row.doc_id + '_' + row.art_ref})
        ON CREATE SET art.name = row.art_ref, art.chapter_ref = COALESCE(row.chap_ref, '')
        MERGE (d)-[:HAS_ARTICLE]->(art)

        FOREACH (_leaf IN CASE WHEN row.is_art_leaf THEN [1] ELSE [] END |
            SET art.qdrant_id = row.chunk_id,
                art.text = COALESCE(row.text, ''),
                art.is_table = row.is_table,
                art.reference_citation = COALESCE(row.ref_cit, '')
        )
    )

    // Nhánh 2: XỬ LÝ KHOẢN (CLAUSE)
    FOREACH (_o IN CASE WHEN row.base_cl_ref IS NOT NULL THEN [1] ELSE [] END |
        MERGE (art:LegalArticle {id: row.doc_id + '_' + row.art_ref})
        MERGE (cl:Clause {id: row.doc_id + '_' + row.art_ref + '_' + row.base_cl_ref})
        ON CREATE SET cl.name = row.base_cl_ref
        MERGE (cl)-[:PART_OF]->(art)

        FOREACH (_leaf IN CASE WHEN row.is_cl_leaf THEN [1] ELSE [] END |
            SET cl.qdrant_id = row.chunk_id,
                cl.text = COALESCE(row.text, ''),
                cl.is_table = row.is_table,
                cl.reference_citation = COALESCE(row.ref_cit, '')
        )
    )

    // Nhánh 3: XỬ LÝ CHUNK MẢNH (SPLIT CHUNK)
    FOREACH (_o IN CASE WHEN row.is_chk_leaf THEN [1] ELSE [] END |
        MERGE (c:Chunk {id: row.chunk_id})
        SET c.chunk_index = row.chunk_idx,
            c.text = COALESCE(row.text, ''),
            c.is_table = row.is_table,
            c.reference_citation = COALESCE(row.ref_cit, ''),
            c.qdrant_id = row.chunk_id

        FOREACH (_c IN CASE WHEN row.base_cl_ref IS NOT NULL THEN [1] ELSE [] END |
            MERGE (cl:Clause {id: row.doc_id + '_' + row.art_ref + '_' + row.base_cl_ref})
            MERGE (c)-[:PART_OF]->(cl)
        )

        FOREACH (_a IN CASE WHEN row.base_cl_ref IS NULL AND row.art_ref IS NOT NULL AND row.art_ref <> '' THEN [1] ELSE [] END |
            MERGE (art:Article {id: row.doc_id + '_' + row.art_ref})
            MERGE (c)-[:PART_OF]->(art)
        )

        FOREACH (_orphan IN CASE WHEN row.art_ref IS NULL OR row.art_ref = '' THEN [1] ELSE [] END |
            MERGE (c)-[:BELONGS_TO]->(d)
        )
    )

    // F2. ONTOLOGY RELATIONS (10 NHÃN) VÀ PHANTOM HIERARCHY
    FOREACH (rel IN row.ontology_relations |
        MERGE (p:Document {document_number: rel.target_doc})
        ON CREATE SET p.id = 'REF_' + rel.target_doc, p.is_full_text = false
        
        FOREACH (_art IN CASE WHEN rel.target_article IS NOT NULL AND rel.target_article <> '' THEN [1] ELSE [] END |
            MERGE (art:Article {id: COALESCE(p.id, 'REF_' + rel.target_doc) + '_' + rel.target_article})
            ON CREATE SET art.name = rel.target_article
            // Always set target_text if present to overwrite ghost node dummy texts
            SET art.text = CASE WHEN rel.target_text <> '' THEN rel.target_text ELSE COALESCE(art.text, '') END
            MERGE (art)-[:BELONGS_TO]->(p)
            
            FOREACH (_cl IN CASE WHEN rel.target_clause IS NOT NULL AND rel.target_clause <> '' THEN [1] ELSE [] END |
                MERGE (cl:Clause {id: art.id + '_' + rel.target_clause})
                ON CREATE SET cl.name = rel.target_clause
                SET cl.text = CASE WHEN rel.target_text <> '' THEN rel.target_text ELSE COALESCE(cl.text, '') END
                MERGE (cl)-[:PART_OF]->(art)
            )
        )
    )

    // Tạo Edges với các nhãn động (Python string format sẽ điền vào đây)
    {dynamic_f2_blocks}

    // F4. CROSS-DOC RELATIONS — Passive Chain (nguồn là rel.source_doc ≠ d)
    {dynamic_f4_blocks}

    // G. CAN CU PHAP LY (BASED_ON) — FALLBACK cho truờng hợp LLM không trích xuất được
    // row.refs đã được lọc Python-side: chỉ gồm các ref CHƯА có trong ontology_relations BASED_ON
    FOREACH (ref IN row.refs |
        FOREACH (_o IN CASE WHEN ref.doc_number IS NOT NULL AND ref.doc_number <> '' AND ref.doc_number <> 'unknown' THEN [1] ELSE [] END |
            MERGE (p:Document {document_number: ref.doc_number})
            ON CREATE SET p.id = 'REF_' + ref.doc_number, p.is_full_text = false
            SET p.title = COALESCE(p.title, ref.doc_title, ''),
                p.year  = COALESCE(p.year,  ref.doc_year,  'N/A')
            MERGE (d)-[:BASED_ON]->(p)
        )
    )
    """

    from backend.ingestion.extractor.entities import FIXED_DOC_RELATIONS, DYNAMIC_DOC_RELATIONS
    all_doc_rels = FIXED_DOC_RELATIONS | DYNAMIC_DOC_RELATIONS
    
    f2_blocks = []
    f4_blocks = []
    for rel_type in all_doc_rels:
        # F2: Direct relations from the current document
        f2_blocks.append(f"""
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = '{rel_type}' AND (r.is_cross_doc IS NULL OR r.is_cross_doc = false)] |
        MERGE (p:Document {{document_number: rel.target_doc}})
        MERGE (d)-[r_edge:{rel_type}]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
        """)
        # F4: Cross-doc relations (passive chains where source is not d)
        f4_blocks.append(f"""
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.is_cross_doc = true AND r.edge_label = '{rel_type}'] |
        MERGE (src:Document {{document_number: rel.source_doc}})
        ON CREATE SET src.id = 'REF_' + rel.source_doc, src.is_full_text = false
        MERGE (tgt:Document {{document_number: rel.target_doc}})
        ON CREATE SET tgt.id = 'REF_' + rel.target_doc, tgt.is_full_text = false
        MERGE (src)-[r_edge:{rel_type}]->(tgt)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.source = 'passive_chain'
    )
        """)

    final_query = query.format(
        dynamic_f2_blocks="\\n".join(f2_blocks),
        dynamic_f4_blocks="\\n".join(f4_blocks)
    )

    with driver.session() as session:
        batch_size = 2000
        for i in range(0, len(params), batch_size):
            chunk_batch = params[i:i+batch_size]
            try:
                session.run(final_query, batch=chunk_batch)
            except Exception as e:
                print(f"Lỗi khi push lô {i} tới Neo4j: {e}")


# =====================================================================
# RETRIEVAL-SIDE: CYPHER QUERY TEMPLATES & HELPER FUNCTIONS
# =====================================================================

def run_cypher(query: str, params: dict = None):
    """Chạy một Cypher query và trả về danh sách records dạng dict."""
    driver = get_neo4j_driver()
    if not driver:
        return []
    try:
        with driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    except Exception as e:
        print(f"  [Neo4j] ⚠️ Cypher query failed: {e}")
        return []


# --- YÊU CẦU 1: Lateral Expansion (Mở rộng ngang) ---
LATERAL_EXPANSION_QUERY = """
MATCH (c)-[:PART_OF|BELONGS_TO*1..3]->(d:Document)-[:HAS_SECTOR]->(s:Sector)
WHERE c.qdrant_id = $chunk_id OR c.id = $chunk_id
WITH d, s
MATCH (s)<-[:HAS_SECTOR]-(related_doc:Document)
WHERE related_doc <> d
RETURN DISTINCT related_doc.title AS title, 
       related_doc.document_number AS document_number,
       s.name AS shared_sector
LIMIT 3
"""

# --- YÊU CẦU 2: Bottom-Up Expansion (Legal QA) ---
BOTTOM_UP_EXPANSION_QUERY = """
MATCH (c)-[:PART_OF|BELONGS_TO*1..2]->(art:Article)-[:BELONGS_TO]->(d:Document)
WHERE c.qdrant_id = $chunk_id OR c.id = $chunk_id
OPTIONAL MATCH (art)<-[:PART_OF]-(sibling)
WHERE sibling.qdrant_id IS NOT NULL AND (sibling.qdrant_id <> $chunk_id AND sibling.id <> $chunk_id)
RETURN d.title AS doc_title,
       d.document_number AS doc_number,
       d.document_toc AS doc_toc,
       art.name AS article_ref,
       collect(DISTINCT sibling.qdrant_id) AS sibling_chunk_ids,
       collect(DISTINCT sibling.text) AS sibling_texts
"""

# --- YÊU CẦU 3: Sector Search MapReduce ---
SECTOR_FULL_MAPREDUCE_QUERY = """
MATCH (d:Document)-[:HAS_SECTOR]->(s:Sector {name: $sector})
OPTIONAL MATCH (d)-[:HAS_TYPE]->(lt:LegalType)
RETURN lt.name AS Loai_VB, 
       count(DISTINCT d) AS So_Luong, 
       collect(DISTINCT d.title)[..15] AS Danh_Sach
ORDER BY So_Luong DESC
"""

# --- YÊU CẦU 5: Hierarchical Sector Search (Drill-down) ---
SECTOR_DOCS_BY_TITLE_QUERY = """
MATCH (d:Document)-[:HAS_SECTOR]->(s:Sector {name: $sector})
WHERE toLower(d.title) CONTAINS toLower($query) 
   OR toLower(d.document_number) CONTAINS toLower($query)
RETURN d.id AS id, 
       d.title AS title, 
       d.document_number AS document_number, 
       d.document_toc AS document_toc
LIMIT 10
"""

# --- YÊU CẦU 4: Conflict Analyzer Time-Travel ---
CONFLICT_TIME_TRAVEL_QUERY = """
MATCH (c)-[:PART_OF|BELONGS_TO*1..3]->(doc:Document)
WHERE c.qdrant_id = $chunk_id OR c.id = $chunk_id
OPTIONAL MATCH (new_doc:Document)-[r:AMENDS|REPLACES]->(doc)
RETURN doc.title AS original_title,
       doc.document_number AS original_doc_number,
       doc.doc_status AS doc_status,
       r.target_text AS target_text,
       r.context AS context,
       r.target_article AS target_article,
       new_doc.document_number AS amending_doc_number,
       new_doc.title AS amending_doc_title,
       type(r) AS relation_type
"""

# --- BLOCK 3: GRAPH CONFLICT DETECTION ---
CONFLICT_DETECTION_QUERY = """
MATCH (doc:Document)
WHERE doc.document_number CONTAINS $doc_number
MATCH (new_doc:Document)-[r:AMENDS|REPLACES|REPEALS]->(doc)
WHERE toLower(r.target_article) CONTAINS toLower($article_ref) OR r.target_article = ''
RETURN new_doc.document_number AS amending_doc,
       new_doc.title AS title,
       new_doc.doc_status AS doc_status,
       type(r) AS relation_type,
       r.target_article AS target_article,
       r.context AS context
"""

def detect_conflicting_documents(doc_number: str, article_ref: str) -> list:
    """
    Tìm kiếm các văn bản (qua GraphDB) có sửa đổi/thay thế/bãi bỏ một điều khoản cụ thể.
    """
    if not doc_number or not article_ref:
        return []
    records = run_cypher(CONFLICT_DETECTION_QUERY, {"doc_number": doc_number, "article_ref": article_ref})
    conflicts = []
    for r in records:
        conflicts.append({
            "amending_doc": r.get("amending_doc"),
            "title": r.get("title"),
            "doc_status": r.get("doc_status"),
            "relation_type": r.get("relation_type"),
            "target_article": r.get("target_article"),
            "context": r.get("context")
        })
    return conflicts


# =====================================================================
# HIGH-LEVEL HELPER FUNCTIONS (Được gọi từ strategies/)
# =====================================================================

def lateral_expand(chunk_ids):
    """
    YÊU CẦU 1: Tìm tài liệu cùng ngành (Sector) với các chunks đã retrieve.
    Trả về danh sách 'Tài liệu tham khảo thêm'.
    """
    all_related = []
    seen = set()
    for cid in chunk_ids[:5]:
        records = run_cypher(LATERAL_EXPANSION_QUERY, {"chunk_id": cid})
        for r in records:
            key = r.get("document_number", "")
            if key and key not in seen:
                all_related.append(r)
                seen.add(key)
    return all_related[:5]


def bottom_up_expand(chunk_ids):
    """
    YÊU CẦU 2: Bottom-Up Traversal cho Legal QA.
    Trả về dict chứa document_toc và danh sách sibling chunks.
    """
    toc = ""
    sibling_texts = []
    sibling_ids = set()
    
    for cid in chunk_ids[:5]:
        records = run_cypher(BOTTOM_UP_EXPANSION_QUERY, {"chunk_id": cid})
        for r in records:
            if not toc and r.get("doc_toc"):
                toc = r["doc_toc"]
            for sid in (r.get("sibling_chunk_ids") or []):
                if sid and sid not in sibling_ids:
                    sibling_ids.add(sid)
            for text in (r.get("sibling_texts") or []):
                if text:
                    sibling_texts.append(text)
    
    return {
        "document_toc": toc,
        "sibling_chunk_ids": list(sibling_ids),
        "sibling_texts": sibling_texts[:20]
    }


def sector_mapreduce(sector_name):
    """
    YÊU CẦU 3: Graph MapReduce cho Sector Search.
    Trả về JSON thống kê phân nhóm theo loại văn bản.
    """
    return run_cypher(SECTOR_FULL_MAPREDUCE_QUERY, {"sector": sector_name})


def find_docs_in_sector_by_title(sector_name, query):
    """
    YÊU CẦU 5: Tìm kiếm văn bản trong Sector dựa trên Title chứa từ khóa.
    """
    return run_cypher(SECTOR_DOCS_BY_TITLE_QUERY, {"sector": sector_name, "query": query})


def conflict_time_travel(chunk_ids):
    """
    YÊU CẦU 4: Time-Travel query cho Conflict Analyzer.
    Kiểm tra xem document có bị AMENDS hoặc REPLACES bởi văn bản mới không.
    Sweep 8 chunk_ids để bao phủ rộng hơn.
    """
    results = []
    seen = set()
    for cid in chunk_ids[:8]:
        records = run_cypher(CONFLICT_TIME_TRAVEL_QUERY, {"chunk_id": cid})
        for r in records:
            key = f"{r.get('original_doc_number')}::{r.get('amending_doc_number', '')}"
            if key not in seen:
                results.append(r)
                seen.add(key)
    return results


# --- Fix 5.1: Direct Entity-based Graph Traversal (Independent of Vector Search) ---
FULL_DOCUMENT_ARTICLES_QUERY = """
MATCH (d:Document)
WHERE d.document_number CONTAINS $doc_number
WITH d
MATCH (d)-[:HAS_ARTICLE]->(art:LegalArticle)
OPTIONAL MATCH (cl:Clause)-[:PART_OF]->(art)
RETURN d.document_number AS document_number,
       d.title AS title,
       art.name AS article_ref,
       art.text AS article_text,
       collect(DISTINCT {name: cl.name, text: cl.text}) AS clauses
ORDER BY art.name
LIMIT 150
"""

def fetch_full_document_articles(doc_number: str) -> list:
    """
    Fix 5.1: Traverse Document → Article → Clause độc lập với Vector Search.
    Khi Qdrant trả 0 hits nhưng doc_number đã biết, dùng Neo4j lấy trực tiếp nội dung Article/Clause.
    """
    if not doc_number:
        return []
    records = run_cypher(FULL_DOCUMENT_ARTICLES_QUERY, {"doc_number": doc_number})
    # Filter out records with None/empty article_text and None clauses
    clean = []
    for r in records:
        clauses = r.get("clauses") or []
        # Filter out None clauses
        real_clauses = [c for c in clauses if c and c.get("text")]
        clean.append({
            "document_number": r.get("document_number", ""),
            "title": r.get("title", ""),
            "article_ref": r.get("article_ref", ""),
            "article_text": r.get("article_text") or "",
            "clauses": real_clauses
        })
    return clean

SPECIFIC_ARTICLE_QUERY = """
MATCH (d:Document)
WHERE d.document_number CONTAINS $doc_number
WITH d
MATCH (d)-[:HAS_ARTICLE]->(art:LegalArticle)
WHERE toLower(art.name) CONTAINS toLower($article_keyword)
OPTIONAL MATCH (cl:Clause)-[:PART_OF]->(art)
RETURN d.document_number AS document_number,
       d.title AS title,
       art.name AS article_ref,
       art.text AS article_text,
       collect(DISTINCT {name: cl.name, text: cl.text}) AS clauses
LIMIT 10
"""

def fetch_specific_article(doc_number: str, article_keyword: str) -> list:
    """
    Kéo Đích danh một cụm Điều khoản/Phụ lục trên Neo4j bằng Full Text Match.
    Dành riêng cho các query chứa rõ 'Điều X', 'Phụ lục Y'.
    """
    if not doc_number or not article_keyword:
        return []
    records = run_cypher(SPECIFIC_ARTICLE_QUERY, {"doc_number": doc_number, "article_keyword": article_keyword})
    clean = []
    for r in records:
        clauses = r.get("clauses") or []
        real_clauses = [c for c in clauses if c and c.get("text")]
        clean.append({
            "document_number": r.get("document_number", ""),
            "title": r.get("title", ""),
            "article_ref": r.get("article_ref", ""),
            "article_text": r.get("article_text") or "",
            "clauses": real_clauses
        })
    return clean


def search_docs_by_keyword(query: str, limit: int = 5):
    """
    Search documents, articles, and clauses in Neo4j by full-text keyword.
    Returns ScoredPointMock to be compatible with RAG pipeline.
    """
    query_cypher = """
    CALL {
      MATCH (d:Document)
      WHERE toLower(d.title) CONTAINS toLower($query) 
         OR toLower(d.document_number) CONTAINS toLower($query)
      RETURN d.id AS id, d.title AS title, d.document_number AS doc_number, d.document_toc AS text, "Document" AS type
      
      UNION
      
      MATCH (a:Article)-[:BELONGS_TO]->(d:Document)
      WHERE toLower(a.name) CONTAINS toLower($query) 
         OR toLower(a.text) CONTAINS toLower($query)
      RETURN a.id AS id, d.title AS title, d.document_number AS doc_number, a.text AS text, "Article" AS type
      
      UNION
      
      MATCH (c:Clause)-[:PART_OF]->(a:Article)-[:BELONGS_TO]->(d:Document)
      WHERE toLower(c.name) CONTAINS toLower($query) 
         OR toLower(c.text) CONTAINS toLower($query)
      RETURN c.id AS id, d.title AS title, d.document_number AS doc_number, c.text AS text, "Clause" AS type
    }
    RETURN id, title, doc_number, text, type
    LIMIT $limit
    """
    records = run_cypher(query_cypher, {"query": query, "limit": limit})
    
    from backend.retrieval.hybrid_search import ScoredPointMock
    hits = []
    for r in records:
        hits.append(ScoredPointMock(
            id=r["id"],
            payload={
                "document_number": r["doc_number"],
                "title": r["title"],
                "chunk_text": r["text"] or f"MATCHED {r['type']} IN: {r['title']}",
                "is_active": True,
                "legal_type": "Document"
            },
            score=0.9
        ))
    return hits

# --- NEW METADATA FETCHER FOR SECTOR SEARCH/LEGAL QA (GRAPH-BASED) ---
def fetch_document_administrative_metadata(doc_number: str) -> str:
    """Lấy thông tin hành chính văn bản qua GraphDB."""
    query = """
    MATCH (d:Document {document_number: $doc_num})
    OPTIONAL MATCH (s:Signer)-[:SIGNED]->(d)
    OPTIONAL MATCH (a:Authority)-[:ISSUED]->(d)
    OPTIONAL MATCH (d)-[:HAS_TYPE]->(lt:LegalType)
    OPTIONAL MATCH (d)-[:HAS_SECTOR]->(sec:Sector)
    RETURN d.document_number AS doc_number,
           d.year AS year,
           d.effective_date AS effective_date,
           d.doc_status AS doc_status,
           collect(DISTINCT s.name) AS signers,
           collect(DISTINCT a.name) AS authorities,
           collect(DISTINCT lt.name) AS legal_types,
           collect(DISTINCT sec.name) AS sectors
    """
    records = run_cypher(query, {"doc_num": doc_number})
    if not records:
        return ""
    
    info = []
    for r in records:
        text = f"Thông tin hành chính của văn bản {r.get('doc_number', doc_number)}:"
        if r.get('year') and r.get('year') != 'N/A':
            text += f" Năm ban hành: {r['year']}."
        if r.get('effective_date') and r.get('effective_date') != 'N/A':
            text += f" Ngày hiệu lực: {r['effective_date']}."
        if r.get('signers') and any(r['signers']):
            text += f" Người ký (Signer): {', '.join(r['signers'])}."
        if r.get('authorities') and any(r['authorities']):
            text += f" Cơ quan ban hành (Authority): {', '.join(r['authorities'])}."
        if r.get('legal_types') and any(r['legal_types']):
            text += f" Loại văn bản (Legal Type): {', '.join(r['legal_types'])}."
        if r.get('sectors') and any(r['sectors']):
            text += f" Lĩnh vực (Sector): {', '.join(r['sectors'])}."
        if r.get('doc_status') and r.get('doc_status') != 'N/A':
            text += f" Tình trạng hiệu lực: {r['doc_status']}."
        info.append(text)
    return " ".join(info)


# --- 2-HOP CHAIN DETECTION (Conflict Analyzer Enhancement) ---
CONFLICT_CHAIN_QUERY = """
MATCH (doc:Document)
WHERE doc.document_number CONTAINS $doc_number
MATCH path = (newer:Document)-[:AMENDS|REPLACES|REPEALS*1..2]->(doc)
RETURN newer.document_number AS chain_doc,
       newer.title AS chain_title,
       newer.doc_status AS chain_status,
       [rel in relationships(path) | type(rel)] AS chain_types,
       [rel in relationships(path) | rel.target_article] AS chain_articles,
       [rel in relationships(path) | rel.context] AS chain_contexts,
       length(path) AS depth
ORDER BY depth ASC
LIMIT 15
"""

def conflict_chain_detect(doc_number: str, article_ref: str = "") -> list:
    """
    2-Hop Chain Detection: Tìm chuỗi sửa đổi gián tiếp.
    VD: NĐ-A sửa NĐ-B, NĐ-B sửa NĐ-C → A gián tiếp ảnh hưởng C.
    """
    if not doc_number:
        return []
    records = run_cypher(CONFLICT_CHAIN_QUERY, {"doc_number": doc_number})
    chains = []
    seen = set()
    for r in records:
        chain_doc = r.get("chain_doc", "")
        if not chain_doc or chain_doc in seen:
            continue
        # Nếu có article_ref, lọc chỉ lấy chain liên quan đến điều khoản đó
        if article_ref:
            chain_arts = r.get("chain_articles") or []
            if chain_arts and not any(article_ref.lower() in str(a).lower() for a in chain_arts if a):
                continue
        seen.add(chain_doc)
        chains.append({
            "chain_doc": chain_doc,
            "chain_title": r.get("chain_title", ""),
            "chain_status": r.get("chain_status", ""),
            "chain_types": r.get("chain_types", []),
            "chain_articles": r.get("chain_articles", []),
            "chain_contexts": r.get("chain_contexts", []),
            "depth": r.get("depth", 1)
        })
    return chains


# --- BASED_ON REVERSE TRAVERSAL (Sector Search Enhancement) ---
SECTOR_BASED_ON_REVERSE_QUERY = """
MATCH (d:Document)-[:HAS_SECTOR]->(s:Sector {name: $sector})
WITH collect(d) AS sector_docs
UNWIND sector_docs AS base_doc
MATCH (derived:Document)-[:BASED_ON]->(base_doc)
WHERE NOT derived IN sector_docs
RETURN DISTINCT derived.document_number AS derived_doc_number,
       derived.title AS derived_title,
       base_doc.document_number AS base_doc_number,
       base_doc.title AS base_doc_title
ORDER BY derived.document_number
LIMIT 20
"""

def sector_based_on_reverse(sector_name: str) -> list:
    """
    Reverse BASED_ON: Tìm tất cả văn bản phái sinh được ban hành
    DỰA TRÊN các văn bản thuộc sector này.
    Giúp Sector Search liệt kê đầy đủ hệ thống văn bản liên quan.
    """
    if not sector_name:
        return []
    records = run_cypher(SECTOR_BASED_ON_REVERSE_QUERY, {"sector": sector_name})
    results = []
    seen = set()
    for r in records:
        dnum = r.get("derived_doc_number", "")
        if dnum and dnum not in seen:
            results.append({
                "derived_doc_number": dnum,
                "derived_title": r.get("derived_title", ""),
                "base_doc_number": r.get("base_doc_number", ""),
                "base_doc_title": r.get("base_doc_title", "")
            })
            seen.add(dnum)
    return results


# =====================================================================
# ENTITY ENRICHMENT — Gắn 10 loại Entity vào Chunk/Article/Clause (HAS_ENTITY)
# =====================================================================

# Các nhãn entity hợp lệ trước đây.
# Hiện tại (Từ v2): Nhãn sẽ do LLM tự do quyết định, không còn bị giới hạn bởi mảng dưới đây.
_ENTITY_LABELS = [
    "Organization", "Subject", "LegalConcept", "Procedure", "Location",
    "Right_Obligation", "Sanction", "Timeframe", "Form", "Fee"
]

_MARK_ENRICHED_QUERY = """
UNWIND $ids AS chunk_id
MATCH (c)
WHERE c.qdrant_id = chunk_id OR c.id = chunk_id
SET c.enriched_v2 = true
"""

_ENRICH_ENTITY_QUERY = """
UNWIND $items AS item
// Tìm node lá chủ sỹ (Chunk / Clause / Article) theo qdrant_id
MATCH (leaf)
WHERE leaf.qdrant_id = item.qdrant_id
WITH leaf, item
// Upsert từng entity node theo label + name
CALL apoc.merge.node([item.label], {name: item.name}) YIELD node AS ent
// Tạo cạnh HAS_ENTITY nếu chưa tồn tại
MERGE (leaf)-[:HAS_ENTITY]->(ent)
"""

_ENRICH_NODE_REL_QUERY = """
UNWIND $nrels AS nr
CALL apoc.merge.node([nr.source_type], {name: nr.source_node}) YIELD node AS src
CALL apoc.merge.node([nr.target_type], {name: nr.target_node}) YIELD node AS tgt
CALL apoc.merge.relationship(src, nr.relationship, {}, {chunk_text: coalesce(nr.chunk_text, '')}, tgt, {}) YIELD rel
RETURN count(rel) AS created
"""


def _sanitize_neo4j_label(raw: str) -> str:
    """Biến raw string thành Neo4j label hợp lệ (chỉ chữ + số + _)."""
    import re
    label = re.sub(r'[^a-zA-Z0-9_]', '', str(raw).strip())
    # Đảm bảo bắt đầu bằng chữ (Neo4j bắt buộc)
    if label and not label[0].isalpha():
        label = 'E_' + label
    
    # ALIAS MAPPER (Schema Deduplication)
    if label in ["Article"]: label = "LegalArticle"
    if label in ["Authority", "Institution"]: label = "Organization"
    if label in ["Signer", "PersonRole"]: label = "Person"
    
    return label or 'Entity'


def _sanitize_rel_type(raw: str) -> str:
    """Biến raw string thành Neo4j relationship type hợp lệ (SCREAMING_SNAKE_CASE)."""
    import re
    rel = re.sub(r'[^a-zA-Z0-9_]', '_', str(raw).strip().upper())
    rel = re.sub(r'_+', '_', rel).strip('_')
    
    # ALIAS MAPPER: chỉ alias chủ động → bị động, KHÔNG đảo ngược bị động thành chủ động
    alias_map = {
        "ISSUED":          "ISSUED_BY",
        "SIGNED":          "SIGNED_BY",
        "APPLIES":         "APPLIES_TO",
        "SUPERVISED_BY":   "MANAGED_BY",
        "ACCOUNTABLE_TO":  "MANAGED_BY",
        "RESPONSIBLE_FOR": "MANAGED_BY",
        "IMPLEMENTED_BY":  "MANAGED_BY",
        "ASSIGNS":         "ASSIGNED_BY",
    }
    if rel in alias_map:
        rel = alias_map[rel]
        
    return rel or 'RELATED_TO'


def _ensure_entity_constraint(session, label: str) -> None:
    """Tạo constraint `name IS UNIQUE` cho entity label nếu chưa tồn tại."""
    if label in ["Document", "LegalArticle", "Article", "Clause", "Chunk"]:
        return  # Tuyệt đối không tạo name IS UNIQUE cho Structural Nodes!
        
    constraint_name = f"{label.lower()[:40]}_name_unique"
    try:
        session.run(
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.name IS UNIQUE;"
        )
    except Exception:
        pass  # bỏ qua nếu đã tồn tại hoặc Neo4j không hỗ trợ



def enrich_chunk_entities(
    driver,
    params_list: list,
    use_apoc: bool = True,
) -> None:
    """
    Batch-upsert free-form entity vào Neo4j và tạo cạnh HAS_ENTITY từ lá (Chunk/Clause/Article).
    Luôn đánh dấu lá là `enriched_v2 = true` (kể cả không có entity).

    params_list: list[
        {
            "qdrant_id": str,                      # ID của lá node
            "entities": {"TênNhãnTựDo": [...], "Fee": [...], ...},
            "node_relations": [{source_node, source_type, target_node, target_type,
                                relationship, chunk_text}, ...]
        }
    ]
    use_apoc: Nếu True dùng apoc.merge.node (đưa ra label động).
              Nếu False, dùng fallback Cypher per-label (chậm hơn nhưng không cần APOC).
    """
    if not driver or not params_list:
        return

    # Luôn ghi nhận đã xử lý đối với toàn bộ các chunk trong params
    qdrant_ids = [p["qdrant_id"] for p in params_list if p.get("qdrant_id")]
    if qdrant_ids:
        try:
            with driver.session() as session:
                session.run(_MARK_ENRICHED_QUERY, ids=qdrant_ids)
        except Exception as e:
            print(f"[enrich_chunk_entities] Lỗi khi đánh dấu enriched_v2: {e}")

    if use_apoc:
        _enrich_with_apoc(driver, params_list)
    else:
        _enrich_fallback(driver, params_list)


def fetch_enriched_v2_chunk_ids() -> set:
    """Lấy danh sách qdrant_id của các Chunk đã được LLM trích xuất Entity mở rộng (v2)."""
    query = "MATCH (c) WHERE c.enriched_v2 = true RETURN c.qdrant_id AS chunk_id"
    records = run_cypher(query)
    return {r["chunk_id"] for r in records if r.get("chunk_id")}


def _enrich_with_apoc(driver, params_list: list) -> None:
    """Phương án APOC: apoc.merge.node cho label động."""
    entity_items = []   # flatten: {qdrant_id, label, name}
    node_rel_items = [] # flatten: {source_node, source_type, target_node, target_type, relationship, chunk_text}

    for p in params_list:
        qdrant_id = p.get("qdrant_id")
        if not qdrant_id:
            continue
        entities = p.get("entities") or {}
        for label, names in entities.items():
            clean_label = _sanitize_neo4j_label(label)
            if not clean_label:
                continue
            for name in names:
                name = str(name).strip()
                if name:
                    entity_items.append({"qdrant_id": qdrant_id, "label": clean_label, "name": name})
        for nr in (p.get("node_relations") or []):
            if nr.get("source_node") and nr.get("target_node") and nr.get("relationship"):
                node_rel_items.append({
                    "source_node": str(nr["source_node"]).strip(),
                    "source_type": _sanitize_neo4j_label(nr.get("source_type", "Entity")),
                    "target_node": str(nr["target_node"]).strip(),
                    "target_type": _sanitize_neo4j_label(nr.get("target_type", "Entity")),
                    "relationship": _sanitize_rel_type(nr["relationship"]),
                    "chunk_text": str(nr.get("chunk_text", ""))[:300],
                })

    BATCH_SIZE = 500
    with driver.session() as session:
        for i in range(0, len(entity_items), BATCH_SIZE):
            try:
                session.run(_ENRICH_ENTITY_QUERY, items=entity_items[i:i+BATCH_SIZE])
            except Exception as e:
                print(f"[enrich_chunk_entities] Entity batch error i={i}: {e}")
        for i in range(0, len(node_rel_items), BATCH_SIZE):
            try:
                session.run(_ENRICH_NODE_REL_QUERY, nrels=node_rel_items[i:i+BATCH_SIZE])
            except Exception as e:
                print(f"[enrich_chunk_entities] NodeRel batch error i={i}: {e}")





def _enrich_fallback(driver, params_list: list) -> None:
    """
    Fallback không cần APOC: chạy 1 Cypher riêng cho mỗi loại entity/relation.
    Sử dụng MERGE toàn phần — đảm bảo node và edge đã có thì NỐI VÀO chứ không tạo mới.
    """
    per_label: dict = {}   # {label_str: [{qdrant_id, name}]}
    per_rel: dict   = {}   # {"SRC_TYPE|TGT_TYPE|REL_TYPE": [{src_node, src_type, tgt_node, tgt_type, rel, chunk_text}]}

    import re
    for p in params_list:
        qdrant_id = p.get("qdrant_id")
        if not qdrant_id:
            continue

        # --- Entities ---
        entities = p.get("entities") or {}
        for label, names in entities.items():
            clean_label = _sanitize_neo4j_label(label)
            if not clean_label:
                continue
            bucket = per_label.setdefault(clean_label, [])
            for name in names:
                name = str(name).strip()
                if name:
                    bucket.append({"qdrant_id": qdrant_id, "name": name})

        # --- Node Relations ---
        for nr in (p.get("node_relations") or []):
            src_node = str(nr.get("source_node", "")).strip()
            tgt_node = str(nr.get("target_node", "")).strip()
            rel_type = _sanitize_rel_type(nr.get("relationship", "RELATED_TO"))
            src_type = _sanitize_neo4j_label(nr.get("source_type", "Entity"))
            tgt_type = _sanitize_neo4j_label(nr.get("target_type", "Entity"))
            if not src_node or not tgt_node:
                continue
            key = f"{src_type}|{tgt_type}|{rel_type}"
            per_rel.setdefault(key, []).append({
                "src_node":  src_node,
                "tgt_node":  tgt_node,
                "chunk_text": str(nr.get("chunk_text", ""))[:300],
            })

    BATCH_SIZE = 500
    with driver.session() as session:

        # --- Ghi entity nodes + HAS_ENTITY edges ---
        for label, items in per_label.items():
            if not items:
                continue
            _ensure_entity_constraint(session, label)
            q = f"""
            UNWIND $items AS item
            MATCH (leaf) WHERE leaf.qdrant_id = item.qdrant_id
            MERGE (ent:{label} {{name: item.name}})
            MERGE (leaf)-[:HAS_ENTITY]->(ent)
            """
            for i in range(0, len(items), BATCH_SIZE):
                try:
                    session.run(q, items=items[i:i+BATCH_SIZE])
                except Exception as e:
                    print(f"[enrich_fallback] {label} entity batch error i={i}: {e}")

        # --- Ghi node_relations (MERGE src → MERGE tgt → MERGE rel) ---
        for key, items in per_rel.items():
            if not items:
                continue
            src_type, tgt_type, rel_type = key.split("|")
            _ensure_entity_constraint(session, src_type)
            _ensure_entity_constraint(session, tgt_type)
            q = f"""
            UNWIND $items AS item
            MERGE (src:{src_type} {{name: item.src_node}})
            MERGE (tgt:{tgt_type} {{name: item.tgt_node}})
            MERGE (src)-[r:{rel_type}]->(tgt)
            ON CREATE SET r.chunk_text = item.chunk_text
            ON MATCH  SET r.chunk_text = CASE
                WHEN item.chunk_text <> '' THEN item.chunk_text
                ELSE r.chunk_text
            END
            """
            for i in range(0, len(items), BATCH_SIZE):
                try:
                    session.run(q, items=items[i:i+BATCH_SIZE])
                except Exception as e:
                    print(f"[enrich_fallback] {rel_type} rel batch error i={i}: {e}")

