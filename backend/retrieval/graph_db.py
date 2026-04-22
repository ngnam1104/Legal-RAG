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
from backend.config import settings
from backend.retrieval.chunker.metadata import normalize_doc_key


def get_neo4j_driver():
    """Tạo kết nối tới Neo4j dựa trên settings."""
    if not settings.NEO4J_URI:
        return None
    try:
        return GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
        )
    except Exception as e:
        print(f"Lỗi khởi tạo Neo4j: {e}")
        return None


def init_neo4j_constraints(driver):
    """Thiết lập các ràng buộc duy nhất để bảo vệ toàn vẹn dữ liệu Graph."""
    queries = [
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;",
        "CREATE CONSTRAINT article_id_unique IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE;",
        "CREATE CONSTRAINT clause_id_unique IF NOT EXISTS FOR (cl:Clause) REQUIRE cl.id IS UNIQUE;",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;",
        "CREATE INDEX document_number_index IF NOT EXISTS FOR (d:Document) ON (d.document_number);",
        "CREATE CONSTRAINT authority_name_unique IF NOT EXISTS FOR (a:Authority) REQUIRE a.name IS UNIQUE;",
        "CREATE CONSTRAINT sector_name_unique IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE;",
        "CREATE CONSTRAINT legaltype_name_unique IF NOT EXISTS FOR (lt:LegalType) REQUIRE lt.name IS UNIQUE;"
    ]
    with driver.session() as session:
        for query in queries:
            try:
                session.run(query)
            except Exception as e:
                print(f"⚠️ Constraint warning: {e}")
    print("🛡️ Hệ thống Constraints (Bao gồm Node Clause) đã sẵn sàng!")


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

    enrich_payloads = []
    for dnum in missing_docs:
        key = normalize_doc_key(dnum)
        if key in meta_by_docnum_lookup:
            hf = meta_by_docnum_lookup[key]

            p_date = str(hf.get("issuance_date", "")).split("T")[0] if hf.get("issuance_date") else ""

            import re
            m_yr = re.search(r"\b((?:19|20)\d{2})\b", p_date)
            if not m_yr: m_yr = re.search(r"(\d{4})", p_date)
            year_val = m_yr.group(1) if m_yr else str(hf.get("year", ""))

            enrich_payloads.append({
                "document_number": dnum,
                "id": str(hf.get("id", f"REF_{dnum}")),
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
    MERGE (p:Document {document_number: doc.document_number})
    WITH p, doc
    WHERE p.title IS NULL OR p.title = '' OR p.id STARTS WITH 'REF_'
    SET p.id = doc.id,
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
        session.run(query, batch=enrich_payloads)


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
                node_key = f"ART_{meta.get('document_id')}_{art_ref}"
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
            "refs": meta.get("legal_basis_refs") or [],
            "ontology_relations": meta.get("ontology_relations") or [],

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

    // B. Merge Authority
    FOREACH (auth IN CASE WHEN row.auth_name IS NOT NULL AND row.auth_name <> 'N/A' AND row.auth_name <> '' THEN [1] ELSE [] END |
        MERGE (a:Authority {name: row.auth_name})
        MERGE (a)-[:ISSUED]->(d)
    )

    // C. Merge Signer
    FOREACH (signer IN CASE WHEN row.signer_name IS NOT NULL AND row.signer_name <> '' THEN [1] ELSE [] END |
        MERGE (s:Signer {name: row.signer_name})
        SET s.signer_id = CASE WHEN row.signer_id IS NOT NULL THEN row.signer_id ELSE s.signer_id END
        MERGE (s)-[:SIGNED]->(d)
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

    // Nhánh 1: XỬ LÝ ĐIỀU (ARTICLE)
    FOREACH (_o IN CASE WHEN row.art_ref IS NOT NULL AND row.art_ref <> '' THEN [1] ELSE [] END |
        MERGE (art:Article {id: row.doc_id + '_' + row.art_ref})
        ON CREATE SET art.name = row.art_ref, art.chapter_ref = COALESCE(row.chap_ref, '')
        MERGE (art)-[:BELONGS_TO]->(d)

        FOREACH (_leaf IN CASE WHEN row.is_art_leaf THEN [1] ELSE [] END |
            SET art.qdrant_id = row.chunk_id,
                art.text = COALESCE(row.text, ''),
                art.is_table = row.is_table,
                art.reference_citation = COALESCE(row.ref_cit, '')
        )
    )

    // Nhánh 2: XỬ LÝ KHOẢN (CLAUSE)
    FOREACH (_o IN CASE WHEN row.base_cl_ref IS NOT NULL THEN [1] ELSE [] END |
        MERGE (art:Article {id: row.doc_id + '_' + row.art_ref})
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

    // Tạo Edges với 10 nhãn
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'BASED_ON'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:BASED_ON]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'AMENDS'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:AMENDS]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'REPEALS'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:REPEALS]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'REPLACES'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:REPLACES]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'GUIDES'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:GUIDES]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'APPLIES'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:APPLIES]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'REFERENCES'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:REFERENCES]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'ISSUED_WITH'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:ISSUED_WITH]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'ASSIGNS'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:ASSIGNS]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )
    FOREACH (rel IN [r IN row.ontology_relations WHERE r.edge_label = 'CORRECTS'] |
        MERGE (p:Document {document_number: rel.target_doc})
        MERGE (d)-[r_edge:CORRECTS]->(p)
        SET r_edge.relation_phrase = COALESCE(rel.relation_phrase, ''), r_edge.context = COALESCE(rel.context, ''), r_edge.target_article = COALESCE(rel.target_article, ''), r_edge.target_clause = COALESCE(rel.target_clause, ''), r_edge.target_text = COALESCE(rel.target_text, '')
    )

    // G. CĂN CỨ PHÁP LÝ (BASED_ON) [MỨC ĐỘ CHUNK/GRANULAR CŨ - CHẠY SONG SONG]
    FOREACH (ref IN row.refs |
        FOREACH (_o IN CASE WHEN ref.doc_number IS NOT NULL AND ref.doc_number <> '' AND ref.doc_number <> 'unknown' THEN [1] ELSE [] END |
            MERGE (p:Document {document_number: ref.doc_number})
            ON CREATE SET p.id = 'REF_' + ref.doc_number, p.is_full_text = false
            SET p.title = COALESCE(p.title, ref.doc_title, ''),
                p.year = COALESCE(p.year, ref.doc_year, 'N/A')
            MERGE (d)-[:BASED_ON]->(p)
        )
    )
    """

    with driver.session() as session:
        session.run(query, batch=params)


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
       collect(DISTINCT d.title)[..10] AS Danh_Sach
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
    """
    results = []
    seen = set()
    for cid in chunk_ids[:5]:
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
MATCH (art:Article)-[:BELONGS_TO]->(d)
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
MATCH (art:Article)-[:BELONGS_TO]->(d)
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


