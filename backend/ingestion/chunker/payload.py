"""
payload.py — Đóng gói chunk payload cho Qdrant và Neo4j.

Exports:
  DocContext          — dataclass metadata bất biến của văn bản
  group_refs()        — nhóm danh sách tham chiếu thành chuỗi breadcrumb
  build_article_chunk() — xây dựng chunk Điều/Khoản/Điểm + entity/relation hints
  build_table_chunk()   — xây dựng chunk Bảng biểu + entity/relation hints

Các hàm build_* trả về None nếu chunk bị lọc rác,
giúp FSM chỉ tăng chunk_idx khi chunk được emit thực sự.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.ingestion.chunker import heuristics


# ==========================================
# DocContext — metadata cố định của văn bản
# Thay thế ~20 biến closure trong core.py cũ
# ==========================================

@dataclass
class DocContext:
    doc_id:              str
    doc_number:          str
    doc_title:           str
    year:                str
    sectors_list:        List[str]
    sectors_str:         str
    doc_status:          str
    issuing_auth:        str
    signer_name:         str
    signer_id:           Any
    promulgation_date:   str
    final_effective_date: str
    legal_type_meta:     str
    url_meta:            str
    basis_refs:          List[dict] = field(default_factory=list)
    document_toc:        str = ""
    ontology_rels:       List[dict] = field(default_factory=list)
    entities:            Dict[str, List[str]] = field(default_factory=dict)
    node_relations:      List[dict] = field(default_factory=list)



# ==========================================
# Tiện ích nội bộ
# ==========================================

def _compact(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).replace("\r", "").split())


# ==========================================
# group_refs — nhóm tham chiếu cho breadcrumb
# Tách từ closure flush_article trong core.py cũ
# ==========================================

def group_refs(refs: List[str]) -> str:
    """
    Nhóm danh sách tham chiếu (khoản/điểm) thành chuỗi ngắn gọn.
    Ví dụ: ["Khoản 1", "Khoản 2"] → "Khoản 1, 2"
             ["a)", "b)"]          → "Điểm a, b)"
    """
    refs = [r for r in refs if not re.match(r"^\s*[-+•]\s*$", str(r))]
    if not refs:
        return ""

    first = str(refs[0]).strip()

    # Khoản X, Y
    m_khoan = re.match(r"(?i)(khoản)\s+(.+)", first)
    if m_khoan:
        prefix = m_khoan.group(1).capitalize()
        nums = []
        for r in refs:
            m = re.match(r"(?i)khoản\s+(.+)", str(r).strip())
            nums.append(m.group(1) if m else str(r).strip())
        return f"{prefix} {', '.join(nums)}"

    # Điểm X, Y
    m_diem = re.match(r"(?i)(điểm)\s+(.+)", first)
    if m_diem:
        prefix = m_diem.group(1).capitalize()
        nums = []
        for r in refs:
            m = re.match(r"(?i)điểm\s+(.+)", str(r).strip())
            nums.append(m.group(1) if m else str(r).strip())
        return f"{prefix} {', '.join(nums)}"

    # Dạng "a)" → "Điểm a, b)"
    if re.match(r"^[^()]+\)$", first):
        nums = []
        for r in refs:
            clean_r = str(r).strip()
            m = re.match(r"^([^()]+)\)$", clean_r)
            nums.append(m.group(1) if m else clean_r.rstrip(")"))
        return f"Điểm {', '.join(nums)})"

    # Dạng "1." → "Khoản 1, 2."
    if re.match(r"^[^.]+\.$", first):
        nums = []
        for r in refs:
            clean_r = str(r).strip()
            m = re.match(r"^([^.]+)\.$", clean_r)
            nums.append(m.group(1) if m else clean_r.rstrip("."))
        return f"Khoản {', '.join(nums)}."

    # Dạng "(1)" → "Khoản (1, 2)"
    if re.match(r"^\([^()]+\)$", first):
        nums = []
        for r in refs:
            clean_r = str(r).strip()
            m = re.match(r"^\(([^()]+)\)$", clean_r)
            nums.append(m.group(1) if m else clean_r.strip("()"))
        return f"Khoản ({', '.join(nums)})"

    return ", ".join(str(r).strip() for r in refs)


# ==========================================
# build_article_chunk
# ==========================================

def build_article_chunk(
    ctx:             DocContext,
    chunk_idx:       int,
    chapter:         Optional[str],
    article_ref:     Optional[str],
    article_preamble: List[str],
    clauses:         List[dict],
    in_appendix:     bool,
) -> Optional[Dict[str, Any]]:
    """
    Xây dựng chunk dict đầy đủ cho một Điều/Khoản/Điểm.

    - Chạy bộ lọc rác; trả về None nếu không đủ nội dung.
    - Gắn nhãn has_potential_entities & has_potential_relations
      trực tiếp (single-pass, không quét lại).
    - Chunk đầu tiên (chunk_idx == 1) được gắn legal_basis_refs,
      document_toc, ontology_relations.
    """
    # --- Ghép nội dung text ---
    parts: List[str] = []
    if article_ref and not article_ref.isupper():
        parts.append(article_ref)
    if article_preamble:
        parts.append("\n".join(article_preamble))
    for cl in clauses:
        parts.append(cl["text"])
        for pt in cl.get("points", []):
            parts.append(pt)

    text_content = "\n".join(parts).strip()

    # --- Bộ lọc rác ---
    is_critical_section = article_ref in ["Lời dẫn", "Phần Nơi nhận và Chữ ký"]
    letters_only = re.sub(r"[\W_0-9]", "", text_content)
    if len(letters_only) < 30 and not is_critical_section:
        return None
    if not article_ref and not clauses and text_content.isupper() and not is_critical_section:
        return None

    # --- Breadcrumb assembly ---
    art_ref = article_ref or None
    bc_components = [chapter, art_ref]
    clause_ref_meta = ""

    if len(clauses) > 1:
        cl_refs: List[str] = []
        for cl in clauses:
            ref = _compact(cl["ref"].replace(" (tiếp theo)", ""))
            if ref and ref not in cl_refs:
                cl_refs.append(ref)
        if cl_refs:
            clause_ref_meta = group_refs(cl_refs)
            bc_components.append(clause_ref_meta)

    elif len(clauses) == 1:
        cl_ref = _compact(clauses[0]["ref"].replace(" (tiếp theo)", ""))
        if cl_ref:
            cl_ref_grouped   = group_refs([cl_ref])
            clause_ref_meta  = cl_ref_grouped
            bc_components.append(cl_ref_grouped)

        point_lines = clauses[0].get("points", [])
        if len(point_lines) > 1:
            pt_prefs: List[str] = []
            for pt in point_lines:
                m_pt = re.match(
                    r"^\s*([0-9]+(?:\.[0-9]+)*[\.)]|[a-zA-ZđĐ][\.)])", pt
                )
                if m_pt:
                    prf = _compact(m_pt.group(1))
                    if prf not in pt_prefs:
                        pt_prefs.append(prf)
            if pt_prefs:
                pt_grouped = group_refs(pt_prefs)
                clause_ref_meta += f" - {pt_grouped}" if clause_ref_meta else pt_grouped
                bc_components.append(pt_grouped)

        elif len(point_lines) == 1:
            m_pt = re.match(
                r"^\s*([0-9]+(?:\.[0-9]+)*[\.)]|[a-zA-ZđĐ][\.)])", point_lines[0]
            )
            if m_pt:
                pt_grouped = group_refs([_compact(m_pt.group(1))])
                clause_ref_meta += f" - {pt_grouped}" if clause_ref_meta else pt_grouped
                bc_components.append(pt_grouped)

    breadcrumb   = " > ".join([x for x in bc_components if x])
    chunk_id_val = f"{ctx.doc_id}::article::c{chunk_idx}"
    short_title  = ctx.doc_title[:100] + "..." if len(ctx.doc_title) > 100 else ctx.doc_title
    ref_citation = ctx.doc_number + (
        " | " + breadcrumb.replace(" > ", " | ") if breadcrumb else ""
    )

    chunk_text = (
        f"Văn bản: {ctx.doc_number} - {short_title}\n"
        f"Lĩnh vực: {ctx.sectors_str}\n"
        f"Điều khoản: {breadcrumb}\n"
        f"Nội dung:\n"
        f"{text_content}"
    )
    short_embed_title = ctx.doc_title[:80] if ctx.doc_title else ""
    text_to_embed = (
        f"[{ctx.doc_number}] {short_embed_title}\n{breadcrumb}\n{text_content}"
    )

    # --- Single-pass entity / relation hints ---
    has_ents = heuristics.has_potential_entities(text_content)
    has_rels = heuristics.has_potential_relations(text_content)
    is_first = chunk_idx == 1

    qdrant_payload = {
        "chunk_id":           chunk_id_val,
        "document_id":        ctx.doc_id,
        "document_number":    ctx.doc_number,
        "year":               ctx.year,
        "legal_sectors":      ctx.sectors_list,
        "is_table":           False,
        "breadcrumb":         breadcrumb,
        "chunk_text":         chunk_text,
        "title":              ctx.doc_title,
        "article_ref":        art_ref or "",
        "is_active":          True,
        "is_appendix":        in_appendix,
        "legal_type":         ctx.legal_type_meta or "",
        "effective_date":     ctx.final_effective_date or "",
        "url":                ctx.url_meta or "",
        "chunk_index":        chunk_idx,
        "reference_citation": ref_citation,
        "has_potential_entities": has_ents,
        "has_potential_relations": has_rels,
    }

    neo4j_payload = {
        "document_id":        ctx.doc_id,
        "chunk_index":        chunk_idx,
        "document_number":    ctx.doc_number or "N/A",
        "title":              ctx.doc_title or "N/A",
        "legal_type":         ctx.legal_type_meta or "N/A",
        "legal_sectors":      ctx.sectors_list or [],
        "issuing_authority":  ctx.issuing_auth or "N/A",
        "signer_name":        ctx.signer_name or "",
        "signer_id":          ctx.signer_id,
        "url":                ctx.url_meta or "",
        "year":               ctx.year or "N/A",
        "promulgation_date":  ctx.promulgation_date or "",
        "effective_date":     ctx.final_effective_date or "",
        "doc_status":         ctx.doc_status,
        "is_active":          True,
        "chapter_ref":        chapter or "",
        "article_ref":        art_ref or "",
        "clause_ref":         clause_ref_meta or "",
        "is_table":           False,
        "reference_citation": ref_citation,
        "chunk_text":         chunk_text or "",
        "legal_basis_refs":   ctx.basis_refs if is_first else [],
        "document_toc":       ctx.document_toc or "",
        "ontology_relations": ctx.ontology_rels if is_first else [],
        "entities":           ctx.entities if is_first else {},
        "node_relations":     ctx.node_relations if is_first else [],
        "has_potential_entities": has_ents,
        "has_potential_relations": has_rels,
    }

    return {
        "chunk_id":               chunk_id_val,
        "chunk_text":             chunk_text,
        "text_to_embed":          text_to_embed,
        "qdrant_metadata":        qdrant_payload,
        "neo4j_metadata":         neo4j_payload,
        "has_potential_entities": has_ents,
        "has_potential_relations": has_rels,
    }


# ==========================================
# build_table_chunk
# ==========================================

def build_table_chunk(
    ctx:           DocContext,
    chunk_idx:     int,
    article_ref:   Optional[str],
    current_chapter: Optional[str],
    header_lines:  List[str],
    row_lines:     List[str],
    in_appendix:   bool,
) -> Optional[Dict[str, Any]]:
    """
    Xây dựng chunk dict cho một Bảng biểu.
    Trả về None nếu bảng không đủ nội dung.
    Gắn nhãn has_potential_entities & has_potential_relations.
    """
    if not row_lines:
        return None

    parts = list(header_lines) + list(row_lines)
    text_content = "\n".join(parts).strip()

    # Bảng toàn ký tự đặc biệt / số → lọc bỏ
    letters_and_nums = re.sub(r"[\W_]", "", "\n".join(row_lines))
    if len(letters_and_nums) < 15:
        return None

    breadcrumb = "Dữ liệu Bảng biểu"
    if article_ref:
        breadcrumb = f"{article_ref} > {breadcrumb}"

    chunk_id_val = f"{ctx.doc_id}::table::c{chunk_idx}"
    short_title  = ctx.doc_title[:100] + "..." if len(ctx.doc_title) > 100 else ctx.doc_title
    ref_citation = ctx.doc_number + (
        " | " + breadcrumb.replace(" > ", " | ") if breadcrumb else ""
    )

    chunk_text = (
        f"Văn bản: {ctx.doc_number} - {short_title}\n"
        f"Lĩnh vực: {ctx.sectors_str}\n"
        f"Điều khoản: {breadcrumb}\n"
        f"Nội dung:\n"
        f"{text_content}"
    )
    short_embed_title = ctx.doc_title[:80] if ctx.doc_title else ""
    text_to_embed = (
        f"[{ctx.doc_number}] {short_embed_title}\n{breadcrumb}\n{text_content}"
    )

    has_ents = heuristics.has_potential_entities(text_content)
    has_rels = heuristics.has_potential_relations(text_content)
    is_first = chunk_idx == 1

    qdrant_payload = {
        "chunk_id":           chunk_id_val,
        "document_id":        ctx.doc_id,
        "document_number":    ctx.doc_number,
        "year":               ctx.year,
        "legal_sectors":      ctx.sectors_list,
        "is_table":           True,
        "breadcrumb":         breadcrumb,
        "chunk_text":         chunk_text,
        "title":              ctx.doc_title,
        "article_ref":        article_ref or "",
        "is_active":          True,
        "is_appendix":        in_appendix,
        "legal_type":         ctx.legal_type_meta or "",
        "effective_date":     ctx.final_effective_date or "",
        "url":                ctx.url_meta or "",
        "chunk_index":        chunk_idx,
        "reference_citation": ref_citation,
        "has_potential_entities": has_ents,
        "has_potential_relations": has_rels,
    }

    neo4j_payload = {
        "document_id":        ctx.doc_id,
        "chunk_index":        chunk_idx,
        "document_number":    ctx.doc_number or "N/A",
        "title":              ctx.doc_title or "N/A",
        "legal_type":         ctx.legal_type_meta or "N/A",
        "legal_sectors":      ctx.sectors_list or [],
        "issuing_authority":  ctx.issuing_auth or "N/A",
        "signer_name":        ctx.signer_name or "",
        "signer_id":          ctx.signer_id,
        "url":                ctx.url_meta or "",
        "year":               ctx.year or "N/A",
        "promulgation_date":  ctx.promulgation_date or "",
        "effective_date":     ctx.final_effective_date or "",
        "doc_status":         ctx.doc_status,
        "is_active":          True,
        "chapter_ref":        current_chapter or "",
        "article_ref":        article_ref or "",
        "clause_ref":         "",
        "is_table":           True,
        "reference_citation": ref_citation,
        "chunk_text":         chunk_text or "",
        "legal_basis_refs":   ctx.basis_refs if is_first else [],
        "document_toc":       ctx.document_toc or "",
        "ontology_relations": ctx.ontology_rels if is_first else [],
        "entities":           ctx.entities if is_first else {},
        "node_relations":     ctx.node_relations if is_first else [],
        "has_potential_entities": has_ents,
        "has_potential_relations": has_rels,
    }

    return {
        "chunk_id":               chunk_id_val,
        "chunk_text":             chunk_text,
        "text_to_embed":          text_to_embed,
        "qdrant_metadata":        qdrant_payload,
        "neo4j_metadata":         neo4j_payload,
        "has_potential_entities": has_ents,
        "has_potential_relations": has_rels,
    }
