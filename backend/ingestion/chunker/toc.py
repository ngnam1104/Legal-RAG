"""
toc.py — Trích xuất Mục lục (TOC) từ văn bản pháp luật.

Duyệt từng dòng cho đến khi gặp Phụ lục để thu thập tiêu đề
Chương và Điều thành chuỗi mục lục ngắn gọn.
"""
from typing import List

from backend.ingestion.chunker import metadata as md


def extract_toc(lines: List[str]) -> str:
    """
    Xây dựng chuỗi mục lục từ danh sách dòng văn bản.

    - Chỉ quét cho đến khi gặp tiêu đề Phụ lục.
    - Bỏ qua dòng trống.
    - Cắt ngắn title Điều nếu > 100 ký tự.

    Returns
    -------
    str — chuỗi mục lục, mỗi entry một dòng.
    """
    toc_list: List[str] = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        if md.appendix_title_pattern.match(line_clean):
            break  # Dừng tại Phụ lục đầu tiên

        m_ch = md.chapter_pattern.match(line_clean)
        m_ar = md.article_pattern.match(line_clean)

        if m_ch:
            toc_list.append(f"{m_ch.group(1)} {m_ch.group(2)}: {m_ch.group(3)}")
        elif m_ar:
            art_num   = m_ar.group(2).strip()
            art_title = m_ar.group(3).strip()
            short_title = art_title[:100] + "..." if len(art_title) > 100 else art_title
            toc_list.append(f"  {m_ar.group(1)} {art_num}. {short_title}".strip())

    return "\n".join(toc_list)
