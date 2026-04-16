"""
Bóc tách quan hệ giữa các văn bản pháp luật (sửa đổi, thay thế, bãi bỏ).
Đồng bộ 100% với notebook legal_rag_build_qdrant_2.ipynb (2026-04-16).

Các cải tiến so với phiên bản cũ:
- Deduplicate bằng seen_keys set
- Scan combined_scope (scope + new_text) cho findall Điều/Khoản
- normalize_doc_key() cho cross-lookup key
- Tách riêng start/end pattern trong extract_exact_article()
"""
import re
from typing import Dict, List
from backend.retrieval.chunker import metadata as md


def extract_exact_article(content: str, article_name: str) -> str:
    """Bóc đúng và đủ 100% nội dung của article_name (VD: 'Điều 5') từ content.
    Lấy từ tên Điều đó kéo dài đến sát trước Điều tiếp theo hoặc Chương tiếp theo.
    Trả về max 1500 ký tự."""
    if not content or not article_name:
        return ""

    # Chẻ từ và nối lại bằng \s+ để bất chấp số lượng khoảng trắng
    parts = article_name.strip().split()
    safe_name = r'\s+'.join([re.escape(p) for p in parts])

    start_pattern = re.compile(r'(?im)^\s*' + safe_name + r'\b[\.:\-]?\s*')
    start_match = start_pattern.search(content)
    if not start_match:
        return ""

    start_pos = start_match.start()
    remaining = content[start_match.end():]
    end_pattern = re.compile(r'(?im)^\s*(?:Điều\s+\d|Chương\s+[IVXLCDM0-9]|Phần\s+(?:thứ\s+)?[IVXLCDM0-9])')
    end_match = end_pattern.search(remaining)

    if end_match:
        result = content[start_pos: start_match.end() + end_match.start()]
    else:
        result = content[start_pos:]

    return result.strip()[:1500]


def extract_relationship_metadata(content: str, global_doc_lookup: dict = None) -> Dict[str, List[dict]]:
    """Bóc tách quan hệ văn bản (sửa đổi, thay thế, bãi bỏ) với đầy đủ tọa độ Điều/Khoản."""
    lines = (content or "").splitlines()

    # YC1: MÀNG LỌC PREAMBLE BẰNG ANCHOR
    anchor_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'(?i)^Điều\s+1\b', stripped) or re.match(r'^QUYẾT ĐỊNH\s*:', stripped):
            anchor_idx = i
            break

    search_lines = lines[anchor_idx:] if anchor_idx is not None else lines
    search_zone = "\n".join(search_lines)

    relationships = {"amended": [], "replaced": [], "repealed": []}
    seen_keys = set()

    for m in md.relationship_pattern.finditer(search_zone):
        raw_action = m.group(1).lower()
        raw_scope = m.group(2).strip()
        raw_type = m.group(3)
        doc_number = md.compact_whitespace(m.group(4))

        # YC2: LẤY CONTEXT BLOCK ĐỘNG (new_text)
        block_end_match = re.search(r'\n\s*Điều\s+\d', search_zone[m.start() + 1:])
        if block_end_match:
            new_text = search_zone[m.start(): m.start() + 1 + block_end_match.start()]
        else:
            new_text = search_zone[m.start():]
        new_text = new_text[:1500].strip()

        # QUÉT VÉT: Dùng khối text vừa lấy kết hợp raw_scope để findall Điều/Khoản
        combined_scope = raw_scope + " " + new_text

        # 1. Phân loại Action
        if "sửa" in raw_action or "bổ sung" in raw_action:
            rel_type = "amended"
        elif "thay thế" in raw_action:
            rel_type = "replaced"
        else:
            rel_type = "repealed"

        # 2. Bóc tách TỌA ĐỘ CHÍNH XÁC (Target Scope) + Deduplicate
        target_article = None
        target_clause = None

        if combined_scope:
            articles = md.extract_article_pattern.findall(combined_scope)
            if articles:
                dedup_arts = []
                for a in articles:
                    clean_a = md.compact_whitespace(a).capitalize()
                    if clean_a not in dedup_arts:
                        dedup_arts.append(clean_a)
                target_article = ", ".join(dedup_arts)

            clauses = md.extract_clause_pattern.findall(combined_scope)
            if clauses:
                dedup_cls = []
                for c in clauses:
                    clean_c = md.compact_whitespace(c).capitalize()
                    if clean_c not in dedup_cls:
                        dedup_cls.append(clean_c)
                target_clause = ", ".join(dedup_cls)

        # STRICT VALIDATION: Sửa đổi mà không chỉ đích danh → loại bỏ
        if rel_type == "amended" and target_article is None and target_clause is None:
            continue

        is_entire_doc = (target_article is None and target_clause is None)

        # Chống trùng lặp linh hoạt
        unique_key = f"{rel_type}_{doc_number}_{target_article}_{target_clause}"
        if unique_key in seen_keys:
            continue
        seen_keys.add(unique_key)

        # YC4: TRUY XUẤT CHÉO old_text TỪ global_doc_lookup
        old_text = ""
        doc_num_clean = md.normalize_doc_key(doc_number)

        if global_doc_lookup and doc_num_clean in global_doc_lookup and target_article:
            first_article = target_article.split(",")[0].strip()
            if first_article:
                old_text = extract_exact_article(global_doc_lookup[doc_num_clean], first_article)

        relationships[rel_type].append({
            "doc_type": md.canonical_doc_type(raw_type),
            "doc_number": doc_number,
            "target_article": target_article,
            "target_clause": target_clause,
            "is_entire_doc": is_entire_doc,
            "old_text": old_text,
            "new_text": new_text
        })

    return relationships
