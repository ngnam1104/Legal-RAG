"""
AdvancedLegalChunker — Trái tim hệ thống tách nhỏ văn bản pháp luật.
Đồng bộ 100% với logic trong notebook legal_rag_build_qdrant_2.ipynb (2026-04-16).

Kiến trúc FSM duyệt từng dòng:
  Header Zone → Chapter → Article → Clause → Point
                                  → Appendix (3 levels)
                                  → Table (flush_table riêng biệt)
"""
import re
import uuid
import datetime
from typing import Any, Dict, List

from backend.retrieval.chunker import metadata as md
from backend.retrieval.chunker import relations as rel


def _compact_whitespace(text: str) -> str:
    """Local compact (dùng trong process_document closure)."""
    if not text:
        return ""
    text = str(text).replace('\r', '')
    return ' '.join(text.split())


def _get_clause_size(clause_obj) -> int:
    """Tính kích thước tổng của 1 clause object (text + points)."""
    if not clause_obj:
        return 0
    return len(clause_obj["text"]) + sum(len(p) for p in clause_obj["points"])


class AdvancedLegalChunker:
    """
    Trái tim của hệ thống tách nhỏ văn bản pháp luật và phân rã các quan hệ đồ thị.
    Đồng bộ hoàn toàn với Notebook v2: Dynamic Tree, flush_article/flush_table,
    Point detection, Appendix 3 levels, TOC extraction, group_refs, 6 bộ lọc rác.
    """
    def __init__(self):
        pass

    # ==========================================
    # CÁC HÀM BÓC TÁCH METADATA (CĂN CỨ PHÁP LÝ & NGÀY)
    # ==========================================
    def _build_parent_law_id(self, doc_type: str, doc_number: str, year: str, doc_title: str) -> str:
        basis = doc_number or doc_title or "unknown"
        return f"parent::{doc_type}::{md.slugify(basis)}::{year or 'unknown'}"

    def _parse_legal_basis_line(self, raw_line: str):
        refs = []
        line = md.compact_whitespace(raw_line)
        for m in md.legal_ref_pattern.finditer(line):
            raw_type = md.compact_whitespace(m.group(1))
            tail = md.compact_whitespace(m.group(2))
            full_ref = md.compact_whitespace(f"{raw_type} {tail}")
            doc_type = md.canonical_doc_type(raw_type)
            year = md.extract_year(full_ref)
            doc_number = md.extract_doc_number(full_ref)
            refs.append({
                "basis_line": line, "doc_type": doc_type, "doc_number": doc_number,
                "doc_year": year, "doc_title": full_ref,
                "parent_law_id": self._build_parent_law_id(doc_type, doc_number, year, full_ref),
            })
        return refs

    def _extract_legal_basis_metadata(self, content: str) -> List[dict]:
        preamble = "\n".join((content or "").splitlines()[:80])
        all_refs = []
        for line in preamble.splitlines():
            if md.legal_basis_line_pattern.match(line):
                all_refs.extend(self._parse_legal_basis_line(line))
        dedup = []
        seen = set()
        for r in all_refs:
            key = (r.get("parent_law_id"), r.get("doc_title"))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        return dedup

    def _extract_effective_date(self, content: str, promulgation_date: str) -> str:
        lines = content.splitlines()
        appendix_start_idx = len(lines)

        for i, line in enumerate(lines):
            if md.appendix_title_pattern.match(line):
                appendix_start_idx = i
                break

        main_body_lines = lines[:appendix_start_idx]
        search_zone = "\n".join(main_body_lines)

        m = md.effective_date_pattern.search(search_zone)
        if m:
            day, month, year = m.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        if md.effective_from_sign_pattern.search(search_zone):
            return promulgation_date

        return ""

    # ==========================================
    # TRÁI TIM HỆ THỐNG: HÀM CẮT CHUNK (PROCESS_DOCUMENT)
    # Đồng bộ 100% với Notebook Cell lK4NO3WiVgaJ
    # ==========================================
    def process_document(self, content: str, metadata: Dict[str, Any], global_doc_lookup: dict = None) -> List[Dict[str, Any]]:
        # 1. Dọn dẹp nội dung thô
        content = str(content or "").replace("\r\n", "\n").strip()
        lines = content.splitlines()

        # 2. Khởi tạo Metadata cơ bản
        doc_id = str(metadata.get("id", uuid.uuid4()))
        doc_number = metadata.get("document_number", "N/A")
        doc_title = metadata.get("title", "N/A")
        issuing_auth = metadata.get("issuing_authority", "N/A")

        signer_name, signer_id = md.parse_signer(metadata.get("signers", ""))
        promulgation_date = metadata.get("promulgation_date", "")
        year = md.extract_year(promulgation_date) or str(metadata.get("year", "N/A"))

        eff_date = self._extract_effective_date(content, promulgation_date)
        final_effective_date = eff_date or metadata.get("effective_date", "") or promulgation_date
        basis_refs = self._extract_legal_basis_metadata(content)

        sectors_list = metadata.get("legal_sectors_list", [])
        sectors_str = ", ".join(sectors_list) if sectors_list else "Chung"

        rel_dict = rel.extract_relationship_metadata(content, global_doc_lookup=global_doc_lookup)
        amended_refs = rel_dict.get('amended', [])
        replaced_refs = rel_dict.get('replaced', [])
        repealed_refs = rel_dict.get('repealed', [])

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        if final_effective_date and final_effective_date > today_str:
            doc_status = "Chưa có hiệu lực"
        else:
            doc_status = "Còn hiệu lực"

        # ==========================================
        # BƯỚC 1: TIỀN XỬ LÝ - RÚT TRÍCH MỤC LỤC (TOC) & LỜI DẪN
        # ==========================================
        toc_list = []
        for line_idx, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            if md.appendix_title_pattern.match(line_clean):
                break
            m_ch = md.chapter_pattern.match(line_clean)
            m_ar = md.article_pattern.match(line_clean)
            if m_ch:
                toc_list.append(f"{m_ch.group(1)} {m_ch.group(2)}: {m_ch.group(3)}")
            elif m_ar:
                art_num = m_ar.group(2).strip()
                art_title = m_ar.group(3).strip()
                short_title = art_title[:100] + "..." if len(art_title) > 100 else art_title
                toc_list.append(f"  {m_ar.group(1)} {art_num}. {short_title}".strip())

        document_toc = "\n".join(toc_list)

        # Khởi tạo mảng chứa kết quả đầu ra
        chunks: List[Dict[str, Any]] = []

        # ==========================================
        # BƯỚC 2: KHỞI TẠO BIẾN TRẠNG THÁI (STATE)
        # ==========================================
        global_chunk_idx = 0
        CHUNK_LIMIT = 1200
        TEXT_LIMIT = 1500

        # ==========================================
        # BƯỚC 3: HÀM ĐÓNG GÓI CHUNK (FLUSH_ARTICLE & FLUSH_TABLE)
        # ==========================================
        def group_refs(refs):
            refs = [r for r in refs if not re.match(r"^\s*[-+•]\s*$", str(r))]
            if not refs:
                return ""
            first = str(refs[0]).strip()

            m_khoan = re.match(r"(?i)(khoản)\s+(.+)", first)
            if m_khoan:
                prefix = m_khoan.group(1).capitalize()
                nums = []
                for r in refs:
                    m = re.match(r"(?i)khoản\s+(.+)", str(r).strip())
                    nums.append(m.group(1) if m else str(r).strip())
                return f"{prefix} {', '.join(nums)}"

            if re.match(r"^[^()]+\)$", first):
                nums = []
                for r in refs:
                    clean_r = str(r).strip()
                    m = re.match(r"^([^()]+)\)$", clean_r)
                    nums.append(m.group(1) if m else clean_r.rstrip(')'))
                return f"Điểm {', '.join(nums)})"

            if re.match(r"^[^.]+\.$", first):
                nums = []
                for r in refs:
                    clean_r = str(r).strip()
                    m = re.match(r"^([^.]+)\.$", clean_r)
                    nums.append(m.group(1) if m else clean_r.rstrip('.'))
                return f"Khoản {', '.join(nums)}."

            if re.match(r"^\([^()]+\)$", first):
                nums = []
                for r in refs:
                    clean_r = str(r).strip()
                    m = re.match(r"^\(([^()]+)\)$", clean_r)
                    nums.append(m.group(1) if m else clean_r.strip('()'))
                return f"Khoản ({', '.join(nums)})"

            return ", ".join(str(r).strip() for r in refs)

        def flush_article(chapter, article_ref, article_preamble, clauses_buffer, active_clause):
            nonlocal global_chunk_idx
            _clauses = list(clauses_buffer)
            if active_clause:
                _clauses.append(active_clause)

            if not _clauses and not article_preamble:
                return

            parts = []
            if article_ref and not article_ref.isupper():
                parts.append(article_ref)

            if article_preamble:
                parts.append("\n".join(article_preamble))

            for cl in _clauses:
                parts.append(cl["text"])
                for pt in cl.get("points", []):
                    parts.append(pt)

            text_content = "\n".join(parts).strip()

            # Bộ lọc rác
            letters_only = re.sub(r'[\W_0-9]', '', text_content)
            if len(letters_only) < 30:
                return
            if not article_ref and not clauses_buffer and text_content.isupper():
                return
            if (not article_ref or article_ref == "Lời dẫn") and not clauses_buffer and len(text_content) < 500:
                return

            global_chunk_idx += 1

            art_ref = article_ref if article_ref else None

            bc_components = [chapter, art_ref]
            clause_ref_meta = ""

            if len(_clauses) > 1:
                cl_refs = []
                for cl in _clauses:
                    ref = _compact_whitespace(cl["ref"].replace(" (tiếp theo)", ""))
                    if ref and ref not in cl_refs:
                        cl_refs.append(ref)
                if cl_refs:
                    clause_ref_meta = group_refs(cl_refs)
                    bc_components.append(clause_ref_meta)
            elif len(_clauses) == 1:
                cl_ref = _compact_whitespace(_clauses[0]["ref"].replace(" (tiếp theo)", ""))
                if cl_ref:
                    cl_ref_grouped = group_refs([cl_ref])
                    clause_ref_meta = cl_ref_grouped
                    bc_components.append(cl_ref_grouped)

                point_lines = _clauses[0].get("points", [])
                if len(point_lines) > 1:
                    pt_prefs = []
                    for pt in point_lines:
                        m_pt = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*[\.)]|[a-zA-ZđĐ][\.)])", pt)
                        if m_pt:
                            prf = _compact_whitespace(m_pt.group(1))
                            if prf not in pt_prefs:
                                pt_prefs.append(prf)
                    if pt_prefs:
                        pt_grouped = group_refs(pt_prefs)
                        clause_ref_meta += f" - {pt_grouped}" if clause_ref_meta else pt_grouped
                        bc_components.append(pt_grouped)
                elif len(point_lines) == 1:
                    m_pt = re.match(r"^\s*([0-9]+(?:\.[0-9]+)*[\.)]|[a-zA-ZđĐ][\.)])", point_lines[0])
                    if m_pt:
                        pt_grouped = group_refs([_compact_whitespace(m_pt.group(1))])
                        clause_ref_meta += f" - {pt_grouped}" if clause_ref_meta else pt_grouped
                        bc_components.append(pt_grouped)

            breadcrumb = " > ".join([x for x in bc_components if x])
            chunk_id_val = f"{doc_id}::article::c{global_chunk_idx}"

            short_title = doc_title[:100] + "..." if len(doc_title) > 100 else doc_title

            chunk_text = (
                f"Văn bản: {doc_number} - {short_title}\n"
                f"Lĩnh vực: {sectors_str}\n"
                f"Điều khoản: {breadcrumb}\n"
                f"Nội dung:\n"
                f"{text_content}"
            )

            text_to_embed = f"[{doc_number}] {breadcrumb}\n{text_content}"

            qdrant_payload = {
                "chunk_id": chunk_id_val,
                "document_id": doc_id,
                "document_number": doc_number,
                "year": year,
                "legal_sectors": sectors_list,
                "is_table": False,
                "breadcrumb": breadcrumb,
                "chunk_text": chunk_text,
                "title": doc_title
            }

            neo4j_payload = {
                "document_id": doc_id,
                "chunk_index": global_chunk_idx,
                "document_number": doc_number,
                "title": doc_title,
                "legal_type": metadata.get("legal_type", "N/A"),
                "legal_sectors": sectors_list,
                "issuing_authority": issuing_auth,
                "signer_name": signer_name,
                "signer_id": signer_id,
                "url": metadata.get("url", ""),
                "promulgation_date": promulgation_date,
                "effective_date": final_effective_date,
                "doc_status": doc_status,
                "is_active": True,
                "chapter_ref": chapter,
                "article_ref": art_ref,
                "clause_ref": clause_ref_meta if clause_ref_meta else None,
                "is_table": False,
                "reference_citation": doc_number + (" | " + breadcrumb.replace(' > ', ' | ') if breadcrumb else ""),
                "chunk_text": chunk_text,
                "legal_basis_refs": basis_refs if global_chunk_idx == 1 else [],
                "document_toc": document_toc,
                "amended_refs": amended_refs if global_chunk_idx == 1 else [],
                "replaced_refs": replaced_refs if global_chunk_idx == 1 else [],
                "repealed_refs": repealed_refs if global_chunk_idx == 1 else []
            }

            chunks.append({
                "chunk_id": chunk_id_val,
                "chunk_text": chunk_text,
                "text_to_embed": text_to_embed,
                "qdrant_metadata": qdrant_payload,
                "neo4j_metadata": neo4j_payload
            })

        def flush_table(article_ref, header_lines, row_lines):
            nonlocal global_chunk_idx
            if not row_lines:
                return

            global_chunk_idx += 1

            parts = list(header_lines) + list(row_lines)
            text_content = "\n".join(parts).strip()

            letters_only = re.sub(r'[\W_0-9]', '', "\n".join(row_lines))
            if len(letters_only) < 30:
                return

            breadcrumb = "Dữ liệu Bảng biểu"
            if article_ref:
                breadcrumb = f"{article_ref} > {breadcrumb}"

            chunk_id_val = f"{doc_id}::table::c{global_chunk_idx}"

            short_title = doc_title[:100] + "..." if len(doc_title) > 100 else doc_title
            chunk_text = (
                f"Văn bản: {doc_number} - {short_title}\n"
                f"Lĩnh vực: {sectors_str}\n"
                f"Điều khoản: {breadcrumb}\n"
                f"Nội dung:\n"
                f"{text_content}"
            )
            text_to_embed = f"[{doc_number}] {breadcrumb}\n{text_content}"

            qdrant_payload = {
                "chunk_id": chunk_id_val,
                "document_id": doc_id,
                "document_number": doc_number,
                "year": year,
                "legal_sectors": sectors_list,
                "is_table": True,
                "breadcrumb": breadcrumb,
                "chunk_text": chunk_text,
                "title": doc_title
            }

            neo4j_payload = {
                "document_id": doc_id,
                "chunk_index": global_chunk_idx,
                "document_number": doc_number,
                "title": doc_title,
                "legal_type": metadata.get("legal_type", "N/A"),
                "legal_sectors": sectors_list,
                "issuing_authority": issuing_auth,
                "signer_name": signer_name,
                "signer_id": signer_id,
                "url": metadata.get("url", ""),
                "promulgation_date": promulgation_date,
                "effective_date": final_effective_date,
                "doc_status": doc_status,
                "is_active": True,
                "chapter_ref": current_chapter,
                "article_ref": article_ref,
                "clause_ref": None,
                "is_table": True,
                "reference_citation": doc_number + (" | " + breadcrumb.replace(' > ', ' | ') if breadcrumb else ""),
                "chunk_text": chunk_text,
                "legal_basis_refs": basis_refs if global_chunk_idx == 1 else [],
                "document_toc": document_toc,
                "amended_refs": amended_refs if global_chunk_idx == 1 else [],
                "replaced_refs": replaced_refs if global_chunk_idx == 1 else [],
                "repealed_refs": repealed_refs if global_chunk_idx == 1 else []
            }

            chunks.append({
                "chunk_id": chunk_id_val,
                "chunk_text": chunk_text,
                "text_to_embed": text_to_embed,
                "qdrant_metadata": qdrant_payload,
                "neo4j_metadata": neo4j_payload
            })

        # --- PHẦN 4: VÒNG LẶP DUYỆT TEXT CHÍNH ---
        lines = content.splitlines()

        # 1. KHỞI TẠO CÁC "KHAY CHỨA" (BUFFERS)
        in_table = False
        table_header = []
        table_rows = []

        current_chapter = None
        current_article_ref = None
        current_article_preamble = []
        current_clauses_buffer = []
        current_active_clause = None

        in_appendix = False

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # ==============================================================
            # BƯỚC 2: BẢO VỆ VÙNG AN TOÀN (Header Protection & Table Processing)
            # ==============================================================
            is_safe_zone = (line_idx < 50)
            if is_safe_zone and (line.isupper() and "ĐỘC LẬP" in line or "TỰ DO" in line):
                continue

            # Lọc bỏ hẳn các dòng Footer
            if md.footer_pattern.match(line):
                continue

            # --- XỬ LÝ BẢNG BIỂU ---
            if line.count('|') >= 2:
                if not in_table:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_clauses_buffer = []
                    current_active_clause = None

                    in_table = True
                    table_header = [line]
                    table_rows = []
                    continue
                else:
                    if set(line.replace('|', '').replace('-', '').replace(':', '').strip()) == set():
                        table_header.append(line)
                        continue

                    if len(table_header) < 2 and not table_rows:
                        table_header.append(line)
                        continue

                    if table_rows:
                        current_table_len = sum(len(r) for r in table_header) + sum(len(r) for r in table_rows)
                        if len(table_rows) >= 20 or (current_table_len + len(line)) > 3000:
                            flush_table(current_article_ref, table_header, table_rows)
                            table_rows = []

                    table_rows.append(line)

                    if len(table_rows) == 1 and len(line) > 3000:
                        flush_table(current_article_ref, table_header, table_rows)
                        table_rows = []
                continue
            else:
                if in_table:
                    if table_rows:
                        flush_table(current_article_ref, table_header, table_rows)
                    in_table = False
                    table_header, table_rows = [], []

            # ==============================================================
            # BƯỚC 3: NHẬN DIỆN CẤU TRÚC PHÂN CẤP LUẬT CHUẨN
            # ==============================================================

            # 3.1. NHẬN DIỆN CHƯƠNG
            m_ch = md.chapter_pattern.match(line)
            if m_ch:
                flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                current_chapter = _compact_whitespace(f"{m_ch.group(1)} {m_ch.group(2)}")
                current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause = None, [], [], None
                in_appendix = False
                continue

            # 3.2. NHẬN DIỆN ĐIỀU
            m_ar = md.article_pattern.match(line)
            if m_ar:
                buffer_len = sum(len(p) for p in current_article_preamble) + sum(_get_clause_size(c) for c in current_clauses_buffer)
                if not current_clauses_buffer and not current_active_clause and buffer_len < 150:
                    pass
                else:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_article_preamble = []

                current_article_ref = f"{m_ar.group(1).strip()} {m_ar.group(2).strip()}"
                article_title = m_ar.group(3).strip()

                if article_title:
                    current_article_preamble.append(article_title)
                current_clauses_buffer = []
                current_active_clause = None
                in_appendix = False
                continue

            # 3.3. NHẬN DIỆN KHOẢN
            m_cl = md.clause_pattern.match(line)
            if m_cl and current_article_ref and not in_appendix:
                if current_active_clause:
                    active_size = _get_clause_size(current_active_clause)
                    buffer_size = sum(_get_clause_size(c) for c in current_clauses_buffer)

                    if (buffer_size + active_size) > CHUNK_LIMIT:
                        if buffer_size > 0:
                            flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, None)
                            current_clauses_buffer = [current_active_clause]
                        else:
                            flush_article(current_chapter, current_article_ref, current_article_preamble, [], current_active_clause)
                            current_clauses_buffer = []

                        if current_article_preamble:
                            current_article_preamble = []
                    else:
                        current_clauses_buffer.append(current_active_clause)

                cl_ref = _compact_whitespace(m_cl.group(1) or m_cl.group(3) or m_cl.group(5) or m_cl.group(7))
                cl_text = _compact_whitespace(m_cl.group(2) or m_cl.group(4) or m_cl.group(6) or m_cl.group(8))

                current_active_clause = {
                    "ref": cl_ref,
                    "text": f"{cl_ref} {cl_text}".strip(),
                    "points": []
                }
                continue

            # 3.4. NHẬN DIỆN ĐIỂM
            m_pt = md.point_pattern.match(line)
            if m_pt and current_active_clause and not in_appendix:
                pt_ref = _compact_whitespace(m_pt.group(1))
                pt_text = _compact_whitespace(m_pt.group(2))
                new_point_text = f"{pt_ref} {pt_text}"

                buffer_sz = sum(_get_clause_size(c) for c in current_clauses_buffer)
                if buffer_sz + _get_clause_size(current_active_clause) + len(new_point_text) > CHUNK_LIMIT:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_clauses_buffer = []
                    if current_article_preamble:
                        current_article_preamble = []

                    cl_ref_base = current_active_clause["ref"].replace(" (tiếp theo)", "")
                    current_active_clause = {
                        "ref": f"{cl_ref_base} (tiếp theo)",
                        "text": f"[{cl_ref_base} tiếp theo]",
                        "points": [new_point_text]
                    }
                else:
                    current_active_clause["points"].append(new_point_text)
                continue

            # ==============================================================
            # BƯỚC 4: NHẬN DIỆN PHỤ LỤC & HƯỚNG DẪN CHUYÊN MÔN (KÈM THEO)
            # ==============================================================
            m_appx_title = md.appendix_title_pattern.match(line) or md.substantive_title_pattern.match(line)
            if m_appx_title and len(line) < 200:
                flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                in_appendix = True
                current_chapter = _compact_whitespace(line)
                current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause = None, [], [], None
                continue

            # NẾU ĐANG TRONG LÃNH THỔ HƯỚNG DẪN / PHỤ LỤC
            if in_appendix:
                # 4.A1 Nhận diện Tập / Phần / Bài
                m_part = md.part_lesson_pattern.match(line)
                if m_part:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_article_ref = f"{m_part.group(1)} {m_part.group(2)}"
                    title_part = m_part.group(3).strip()
                    current_article_preamble = [title_part] if title_part else []
                    current_clauses_buffer, current_active_clause = [], None
                    continue

                # 4.A2 Nhận diện Tiêu đề Khối VIẾT HOA
                if line.isupper() and 5 < len(line) < 150 and not re.match(r'^[IVXLCDM0-9]', line):
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_article_ref = line.strip()
                    current_article_preamble = []
                    current_clauses_buffer, current_active_clause = [], None
                    continue

                # 4.A3 CẤP 1 PHỤ LỤC: I, II, III hoặc A, B, C
                m_a1 = md.appx_lvl1_pattern.match(line)
                if m_a1:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_article_ref = _compact_whitespace(f"{m_a1.group(1)} {m_a1.group(2)}")
                    current_article_preamble = []
                    current_clauses_buffer, current_active_clause = [], None
                    continue

                # 4.B CẤP 2 PHỤ LỤC: 1, 2, 3
                m_a2 = md.appx_lvl2_pattern.match(line)
                if m_a2:
                    if current_active_clause:
                        active_size = _get_clause_size(current_active_clause)
                        buffer_size = sum(_get_clause_size(c) for c in current_clauses_buffer)

                        if (buffer_size + active_size) > CHUNK_LIMIT:
                            if buffer_size > 0:
                                flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, None)
                                current_clauses_buffer = [current_active_clause]
                            else:
                                flush_article(current_chapter, current_article_ref, current_article_preamble, [], current_active_clause)
                                current_clauses_buffer = []

                            if current_article_preamble:
                                current_article_preamble = []
                        else:
                            current_clauses_buffer.append(current_active_clause)

                    cl_ref = _compact_whitespace(m_a2.group(1))
                    cl_text = _compact_whitespace(m_a2.group(2))

                    current_active_clause = {
                        "ref": f"{cl_ref}.",
                        "text": f"{cl_ref}. {cl_text}".strip(),
                        "points": []
                    }
                    continue

                # 4.C CẤP 3 PHỤ LỤC: 1.1, 1.2.1 hoặc a), b)
                m_a3 = md.appx_lvl3_pattern.match(line)
                m_pt_appx = md.point_pattern.match(line)

                sub_match = m_a3 or m_pt_appx
                if sub_match and current_active_clause:
                    pt_ref = _compact_whitespace(sub_match.group(1))
                    pt_text = _compact_whitespace(sub_match.group(2))
                    new_point_text = f"{pt_ref} {pt_text}"

                    buffer_sz = sum(_get_clause_size(c) for c in current_clauses_buffer)
                    if buffer_sz + _get_clause_size(current_active_clause) + len(new_point_text) > CHUNK_LIMIT:
                        flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                        current_clauses_buffer = []
                        if current_article_preamble:
                            current_article_preamble = []
                        cl_ref_base = current_active_clause["ref"].replace(" (tiếp theo)", "")
                        current_active_clause = {
                            "ref": f"{cl_ref_base} (tiếp theo)",
                            "text": f"[{cl_ref_base} tiếp theo]",
                            "points": [new_point_text]
                        }
                    else:
                        current_active_clause["points"].append(new_point_text)
                    continue

            # ==============================================================
            # BƯỚC 5: TEXT TỰ DO (ĐOẠN VĂN TIẾP NỐI)
            # ==============================================================

            # Lọc rác mào đầu: Xóa "Căn cứ..." ở phần mở đầu
            if not current_chapter and not current_article_ref and line.lower().startswith("căn cứ"):
                continue

            if current_active_clause:
                preamble_size = sum(len(p) for p in current_article_preamble) if current_article_preamble else 0
                buffer_size = sum(_get_clause_size(c) for c in current_clauses_buffer)

                if buffer_size + preamble_size + _get_clause_size(current_active_clause) + len(line) > TEXT_LIMIT:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_clauses_buffer = []
                    if current_article_preamble:
                        current_article_preamble = []

                    cl_ref_base = current_active_clause["ref"].replace(" (tiếp theo)", "")

                    if current_active_clause["points"]:
                        current_active_clause = {
                            "ref": f"{cl_ref_base} (tiếp theo)",
                            "text": f"[{cl_ref_base} tiếp theo]",
                            "points": [line]
                        }
                    else:
                        current_active_clause = {
                            "ref": f"{cl_ref_base} (tiếp theo)",
                            "text": f"[{cl_ref_base} tiếp theo]\n{line}",
                            "points": []
                        }
                else:
                    if current_active_clause["points"]:
                        current_active_clause["points"][-1] += f"\n{line}"
                    else:
                        current_active_clause["text"] += f"\n{line}"
            elif current_article_ref:
                if sum(len(p) for p in current_article_preamble) + len(line) > TEXT_LIMIT:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_article_preamble = [line]
                    current_clauses_buffer = []
                    current_active_clause = None
                else:
                    current_article_preamble.append(line)
            elif current_chapter:
                if sum(len(p) for p in current_article_preamble) + len(line) > TEXT_LIMIT:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_article_preamble = [line]
                    current_clauses_buffer = []
                    current_active_clause = None
                else:
                    current_article_preamble.append(line)
            else:
                if sum(len(p) for p in current_article_preamble) + len(line) > TEXT_LIMIT:
                    flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)
                    current_article_preamble = [line]
                    current_clauses_buffer = []
                    current_active_clause = None
                else:
                    current_article_preamble.append(line)

        # --- KẾT THÚC VÒNG LẶP: FLUSH NỐT DỮ LIỆU CUỐI ---
        if in_table:
            flush_table(current_article_ref, table_header, table_rows)
        else:
            flush_article(current_chapter, current_article_ref, current_article_preamble, current_clauses_buffer, current_active_clause)

        return chunks
