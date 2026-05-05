"""
fsm.py — Máy trạng thái hữu hạn (FSM) duyệt từng dòng văn bản pháp luật.

Kiến trúc phân cấp:
  Header Zone → Chapter → Article → Clause → Point
                                   → Appendix (3 levels)
                                   → Table (flush riêng biệt)

Export duy nhất: scan_document(lines, ctx) → List[chunk_dict]

Mỗi chunk được gắn nhãn has_potential_entities và has_potential_relations
ngay trong lần quét đầu tiên — không cần quét lại sau.
"""
import re
from typing import Any, Dict, List, Optional

from backend.ingestion.chunker import metadata as md
from backend.ingestion.chunker.payload import (
    DocContext,
    build_article_chunk,
    build_table_chunk,
)

# ==========================================
# Hằng số kích thước chunk
# ==========================================
_CHUNK_LIMIT = 1500   # Kích thước tối đa buffer khoản trước khi flush
_TEXT_LIMIT  = 2000   # Kích thước tối đa đoạn văn tự do


def _compact(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).replace("\r", "").split())


def _clause_size(clause_obj: Optional[dict]) -> int:
    if not clause_obj:
        return 0
    return len(clause_obj["text"]) + sum(len(p) for p in clause_obj.get("points", []))


# ==========================================
# scan_document — vòng lặp FSM chính
# ==========================================

def scan_document(lines: List[str], ctx: DocContext) -> List[Dict[str, Any]]:
    """
    Duyệt toàn bộ danh sách dòng văn bản, áp dụng FSM để tách chunk.

    Parameters
    ----------
    lines : List[str]  — Các dòng đã splitlines() từ nội dung văn bản
    ctx   : DocContext — Metadata bất biến của văn bản

    Returns
    -------
    List[dict] — Danh sách chunk dict, mỗi chunk có:
        chunk_id, chunk_text, text_to_embed,
        qdrant_metadata, neo4j_metadata,
        has_potential_entities, has_potential_relations  ← single-pass tagged
    """
    chunks: List[Dict[str, Any]] = []

    # chunk_idx chỉ tăng khi chunk được emit thực sự (pass bộ lọc rác)
    chunk_idx: int = 0

    # ==== Buffers bảng biểu ====
    in_table:     bool       = False
    table_header: List[str]  = []
    table_rows:   List[str]  = []

    # ==== Buffers cấu trúc phân cấp ====
    current_chapter:           Optional[str]  = None
    current_article_ref:       Optional[str]  = None
    current_article_preamble:  List[str]      = []
    current_clauses_buffer:    List[dict]     = []
    current_active_clause:     Optional[dict] = None
    in_appendix:               bool           = False

    # ----------------------------------------------------------
    # Hàm nội bộ: emit article chunk
    # ----------------------------------------------------------
    def try_emit_article(clauses: List[dict], active: Optional[dict]) -> None:
        """
        Gộp clauses + active thành 1 chunk rồi emit nếu pass bộ lọc rác.
        chunk_idx chỉ tăng khi chunk được emit thực sự.
        Đọc chapter, article_ref, preamble, in_appendix từ closure.
        """
        nonlocal chunk_idx
        all_clauses = list(clauses) + ([active] if active else [])
        tentative   = chunk_idx + 1
        result = build_article_chunk(
            ctx              = ctx,
            chunk_idx        = tentative,
            chapter          = current_chapter,
            article_ref      = current_article_ref,
            article_preamble = current_article_preamble,
            clauses          = all_clauses,
            in_appendix      = in_appendix,
        )
        if result:
            chunk_idx = tentative
            chunks.append(result)

    def flush_article() -> None:
        """Flush toàn bộ state hiện tại (buffer + active clause)."""
        try_emit_article(current_clauses_buffer, current_active_clause)

    # ----------------------------------------------------------
    # Hàm nội bộ: emit table chunk
    # ----------------------------------------------------------
    def try_emit_table() -> None:
        nonlocal chunk_idx
        tentative = chunk_idx + 1
        result = build_table_chunk(
            ctx            = ctx,
            chunk_idx      = tentative,
            article_ref    = current_article_ref,
            current_chapter = current_chapter,
            header_lines   = table_header,
            row_lines      = table_rows,
            in_appendix    = in_appendix,
        )
        if result:
            chunk_idx = tentative
            chunks.append(result)

    # ==============================================================
    # VÒNG LẶP CHÍNH
    # ==============================================================
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # ==============================================================
        # BƯỚC 2: BẢO VỆ VÙNG AN TOÀN (Header Protection)
        # ==============================================================
        is_safe_zone = line_idx < 50
        if is_safe_zone and line.isupper() and ("ĐỘC LẬP" in line or "TỰ DO" in line):
            continue

        # ==============================================================
        # PHÁT HIỆN FOOTER (Nơi nhận / Chữ ký)
        # Không `continue` — để line rơi xuống BƯỚC 5 (text tự do)
        # ==============================================================
        if md.footer_pattern.match(line):
            in_appendix = False
            if current_article_ref != "Phần Nơi nhận và Chữ ký":
                flush_article()
                current_chapter          = None
                current_article_ref      = "Phần Nơi nhận và Chữ ký"
                current_article_preamble = []
                current_clauses_buffer   = []
                current_active_clause    = None

        # ==============================================================
        # XỬ LÝ BẢNG BIỂU (dòng chứa >= 2 dấu |)
        # ==============================================================
        if line.count("|") >= 2:
            if not in_table:
                flush_article()
                current_clauses_buffer = []
                current_active_clause  = None
                in_table      = True
                table_header  = [line]
                table_rows    = []
                continue
            else:
                # Dòng phân cách header (---) → vào header
                if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                    table_header.append(line)
                    continue

                # Hàng đầu tiên khi header chưa đủ 2 dòng → vào header
                if len(table_header) < 2 and not table_rows:
                    table_header.append(line)
                    continue

                # Bảng quá lớn → flush và bắt đầu batch mới
                if table_rows:
                    current_table_len = (
                        sum(len(r) for r in table_header) +
                        sum(len(r) for r in table_rows)
                    )
                    if len(table_rows) >= 15 or (current_table_len + len(line)) > 2500:
                        try_emit_table()
                        table_rows = []

                table_rows.append(line)

                if len(table_rows) == 1 and len(line) > 2500:
                    try_emit_table()
                    table_rows = []
                continue
        else:
            if in_table:
                if table_rows:
                    try_emit_table()
                in_table     = False
                table_header = []
                table_rows   = []

        # ==============================================================
        # BƯỚC 3: NHẬN DIỆN CẤU TRÚC PHÂN CẤP LUẬT CHUẨN
        # ==============================================================

        # 3.1. CHƯƠNG
        m_ch = md.chapter_pattern.match(line)
        if m_ch:
            flush_article()
            current_chapter          = _compact(f"{m_ch.group(1)} {m_ch.group(2)}")
            current_article_ref      = None
            current_article_preamble = []
            current_clauses_buffer   = []
            current_active_clause    = None
            in_appendix              = False
            continue

        # 3.2. ĐIỀU
        m_ar = md.article_pattern.match(line)
        if m_ar:
            if current_article_preamble or current_clauses_buffer or current_active_clause:
                flush_article()
                current_article_preamble = []

            current_article_ref = f"{m_ar.group(1).strip()} {m_ar.group(2).strip()}"
            article_title = m_ar.group(3).strip()
            if article_title:
                current_article_preamble.append(article_title)
            current_clauses_buffer = []
            current_active_clause  = None
            in_appendix            = False
            continue

        # 3.3. KHOẢN
        m_cl = md.clause_pattern.match(line)
        if m_cl and current_article_ref and not in_appendix:
            if current_active_clause:
                active_size = _clause_size(current_active_clause)
                buffer_size = sum(_clause_size(c) for c in current_clauses_buffer)

                if (buffer_size + active_size) > _CHUNK_LIMIT:
                    if buffer_size > 0:
                        # Flush buffer, giữ active → đầu buffer mới
                        try_emit_article(current_clauses_buffer, None)
                        current_clauses_buffer = [current_active_clause]
                    else:
                        # Flush chỉ active
                        try_emit_article([], current_active_clause)
                        current_clauses_buffer = []
                    if current_article_preamble:
                        current_article_preamble = []
                else:
                    current_clauses_buffer.append(current_active_clause)

            cl_ref  = _compact(
                m_cl.group(1) or m_cl.group(3) or m_cl.group(5) or m_cl.group(7)
            )
            cl_text = _compact(
                m_cl.group(2) or m_cl.group(4) or m_cl.group(6) or m_cl.group(8)
            )
            current_active_clause = {
                "ref":    cl_ref,
                "text":   f"{cl_ref} {cl_text}".strip(),
                "points": [],
            }
            continue

        # 3.4. ĐIỂM
        m_pt = md.point_pattern.match(line)
        if m_pt and current_active_clause and not in_appendix:
            pt_ref         = _compact(m_pt.group(1))
            pt_text        = _compact(m_pt.group(2))
            new_point_text = f"{pt_ref} {pt_text}"

            buffer_sz = sum(_clause_size(c) for c in current_clauses_buffer)
            if buffer_sz + _clause_size(current_active_clause) + len(new_point_text) > _CHUNK_LIMIT:
                try_emit_article(current_clauses_buffer, current_active_clause)
                current_clauses_buffer = []
                if current_article_preamble:
                    current_article_preamble = []
                cl_ref_base = current_active_clause["ref"].replace(" (tiếp theo)", "")
                current_active_clause = {
                    "ref":    f"{cl_ref_base} (tiếp theo)",
                    "text":   f"[{cl_ref_base} tiếp theo]",
                    "points": [new_point_text],
                }
            else:
                current_active_clause["points"].append(new_point_text)
            continue

        # ==============================================================
        # BƯỚC 4: PHỤ LỤC & HƯỚNG DẪN CHUYÊN MÔN (KÈM THEO)
        # ==============================================================
        m_appx_title = md.appendix_title_pattern.match(line)
        m_sub_title  = md.substantive_title_pattern.match(line)

        if (m_appx_title or m_sub_title) and len(line) < 200:
            flush_article()
            is_standard = bool(
                m_sub_title and re.search(
                    r"(QUY CHUẨN|QCVN|TIÊU CHUẨN|TCVN)", line, re.IGNORECASE
                )
            )
            in_appendix              = bool(m_appx_title) or is_standard
            current_chapter          = _compact(line)
            current_article_ref      = None
            current_article_preamble = []
            current_clauses_buffer   = []
            current_active_clause    = None
            continue

        if in_appendix:
            # 4.A1: Tập / Phần / Bài
            m_part = md.part_lesson_pattern.match(line)
            if m_part:
                flush_article()
                current_article_ref = f"{m_part.group(1)} {m_part.group(2)}"
                title_part = m_part.group(3).strip() if m_part.group(3) else ""
                current_article_preamble = [title_part] if title_part else []
                current_clauses_buffer   = []
                current_active_clause    = None
                continue

            # 4.A2: Tiêu đề khối VIẾT HOA
            if line.isupper() and 5 < len(line) < 150 and not re.match(r"^[IVXLCDM0-9]", line):
                flush_article()
                current_article_ref      = line.strip()
                current_article_preamble = []
                current_clauses_buffer   = []
                current_active_clause    = None
                continue

            # 4.A3: Cấp 1 phụ lục: I, II, III hoặc A, B, C
            m_a1 = md.appx_lvl1_pattern.match(line)
            if m_a1:
                flush_article()
                current_article_ref      = _compact(f"{m_a1.group(1)} {m_a1.group(2)}")
                current_article_preamble = []
                current_clauses_buffer   = []
                current_active_clause    = None
                continue

            # 4.B: Cấp 2 phụ lục: 1, 2, 3
            m_a2 = md.appx_lvl2_pattern.match(line)
            if m_a2:
                if current_active_clause:
                    active_size = _clause_size(current_active_clause)
                    buffer_size = sum(_clause_size(c) for c in current_clauses_buffer)
                    if (buffer_size + active_size) > _CHUNK_LIMIT:
                        if buffer_size > 0:
                            try_emit_article(current_clauses_buffer, None)
                            current_clauses_buffer = [current_active_clause]
                        else:
                            try_emit_article([], current_active_clause)
                            current_clauses_buffer = []
                        if current_article_preamble:
                            current_article_preamble = []
                    else:
                        current_clauses_buffer.append(current_active_clause)

                cl_ref  = _compact(m_a2.group(1))
                cl_text = _compact(m_a2.group(2))
                current_active_clause = {
                    "ref":    f"{cl_ref}.",
                    "text":   f"{cl_ref}. {cl_text}".strip(),
                    "points": [],
                }
                continue

            # 4.C: Cấp 3 phụ lục: 1.1, 1.2.1 hoặc a), b)
            m_a3       = md.appx_lvl3_pattern.match(line)
            m_pt_appx  = md.point_pattern.match(line)
            sub_match  = m_a3 or m_pt_appx
            if sub_match and current_active_clause:
                pt_ref         = _compact(sub_match.group(1))
                pt_text        = _compact(sub_match.group(2))
                new_point_text = f"{pt_ref} {pt_text}"

                buffer_sz = sum(_clause_size(c) for c in current_clauses_buffer)
                if buffer_sz + _clause_size(current_active_clause) + len(new_point_text) > _CHUNK_LIMIT:
                    try_emit_article(current_clauses_buffer, current_active_clause)
                    current_clauses_buffer = []
                    if current_article_preamble:
                        current_article_preamble = []
                    cl_ref_base = current_active_clause["ref"].replace(" (tiếp theo)", "")
                    current_active_clause = {
                        "ref":    f"{cl_ref_base} (tiếp theo)",
                        "text":   f"[{cl_ref_base} tiếp theo]",
                        "points": [new_point_text],
                    }
                else:
                    current_active_clause["points"].append(new_point_text)
                continue

        # ==============================================================
        # BƯỚC 5: TEXT TỰ DO (ĐOẠN VĂN TIẾP NỐI)
        # ==============================================================
        if current_active_clause:
            preamble_size = sum(len(p) for p in current_article_preamble)
            buffer_size   = sum(_clause_size(c) for c in current_clauses_buffer)

            if (
                buffer_size + preamble_size +
                _clause_size(current_active_clause) + len(line) > _TEXT_LIMIT
            ):
                try_emit_article(current_clauses_buffer, current_active_clause)
                current_clauses_buffer = []
                if current_article_preamble:
                    current_article_preamble = []

                cl_ref_base = current_active_clause["ref"].replace(" (tiếp theo)", "")
                if current_active_clause["points"]:
                    current_active_clause = {
                        "ref":    f"{cl_ref_base} (tiếp theo)",
                        "text":   f"[{cl_ref_base} tiếp theo]",
                        "points": [line],
                    }
                else:
                    current_active_clause = {
                        "ref":    f"{cl_ref_base} (tiếp theo)",
                        "text":   f"[{cl_ref_base} tiếp theo]\n{line}",
                        "points": [],
                    }
            else:
                if current_active_clause["points"]:
                    current_active_clause["points"][-1] += f"\n{line}"
                else:
                    current_active_clause["text"] += f"\n{line}"

        elif current_article_ref:
            if sum(len(p) for p in current_article_preamble) + len(line) > _TEXT_LIMIT:
                flush_article()
                current_article_preamble = [line]
                current_clauses_buffer   = []
                current_active_clause    = None
            else:
                current_article_preamble.append(line)

        elif current_chapter:
            if sum(len(p) for p in current_article_preamble) + len(line) > _TEXT_LIMIT:
                flush_article()
                current_article_preamble = [line]
                current_clauses_buffer   = []
                current_active_clause    = None
            else:
                current_article_preamble.append(line)

        else:
            if not current_article_ref:
                current_article_ref = "Lời dẫn"
            if sum(len(p) for p in current_article_preamble) + len(line) > _TEXT_LIMIT:
                flush_article()
                current_article_preamble = [line]
                current_clauses_buffer   = []
                current_active_clause    = None
            else:
                current_article_preamble.append(line)

    # ==============================================================
    # KẾT THÚC VÒNG LẶP: FLUSH NỐT DỮ LIỆU CUỐI
    # ==============================================================
    if in_table:
        if table_rows:
            try_emit_table()
    else:
        flush_article()

    return chunks
