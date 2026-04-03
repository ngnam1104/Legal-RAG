import re
import uuid
from typing import Any, List, Dict, Tuple, Optional

def compact_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()

def split_sector_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_text = ", ".join(str(x) for x in value if str(x).strip())
    else:
        raw_text = str(value).strip()
    if not raw_text or raw_text.lower() == "nan":
        return []
    # Tách theo hệ thống chuẩn: chỉ tách dấu phẩy
    parts = re.split(r"\s*,\s*", raw_text)
    return sorted(set(part.strip() for part in parts if part and part.strip()))

class AdvancedLegalChunker:
    def __init__(self, appendix_chunk_size: int = 1000, max_chunk_size: int = 1500):
        self.appendix_chunk_size = appendix_chunk_size
        self.max_chunk_size = max_chunk_size

        # Hierarchical tree regex
        self.appendix_pattern = re.compile(r"(?im)^\s*(PHU LUC|PHỤ LỤC|DANH MUC|DANH MỤC)\b.*$")
        self.chapter_pattern = re.compile(r"(?im)^\s*(Chương\s+[IVXLCDM0-9]+)\s*(.*)$")
        self.article_pattern = re.compile(r"(?im)^\s*(Điều\s+\d+[A-Za-z0-9\/\-]*)[\.\:\-]?\s*(.*)$")
        self.clause_pattern = re.compile(
            r"(?im)^\s*(Khoản\s+\d+[\.\:\-]?)\s*(.*)$|^\s*(\d+[\.\)])\s*(.*)$"
        )

        # Citation graph extraction regex
        self.legal_basis_line_pattern = re.compile(r"(?im)^\s*căn cứ\b.*$")
        self.legal_ref_pattern = re.compile(
            r"(?i)\b(Hiến pháp|Bộ luật|Luật|Nghị quyết|Pháp lệnh|Nghị định|Thông tư)\b([^.;\n]*)"
        )

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-") or "unknown"

    @staticmethod
    def _canonical_doc_type(raw: str) -> str:
        text = (raw or "").lower()
        if "hiến pháp" in text or "hien phap" in text:
            return "constitution"
        if "bộ luật" in text or "bo luat" in text:
            return "code"
        if "luật" in text or "luat" in text:
            return "law"
        if "pháp lệnh" in text or "phap lenh" in text:
            return "ordinance"
        if "nghị quyết" in text or "nghi quyet" in text:
            return "resolution"
        if "nghị định" in text or "nghi dinh" in text:
            return "decree"
        if "thông tư" in text or "thong tu" in text:
            return "circular"
        return "other"

    @staticmethod
    def _extract_year(text: str) -> str:
        match = re.search(r"\b(19|20)\d{2}\b", text or "")
        return match.group(0) if match else ""

    @staticmethod
    def _extract_doc_number(text: str) -> str:
        number_patterns = [
            r"(?i)(?:số\s*)?(\d+\/\d+(?:\/[A-ZĐ\-]+)?)",
            r"(?i)(\d+\/[A-ZĐ\-]+)",
        ]
        for pattern in number_patterns:
            match = re.search(pattern, text or "")
            if match:
                return compact_whitespace(match.group(1))
        return ""

    def _build_document_uid(self, metadata: dict) -> str:
        legal_type = self._canonical_doc_type(metadata.get("legal_type", ""))
        doc_number = metadata.get("document_number") or metadata.get("id") or "unknown"
        issue_date = str(metadata.get("issuance_date") or metadata.get("issue_date") or "")
        year = self._extract_year(issue_date) or self._extract_year(str(metadata.get("title", ""))) or "unknown"
        return f"doc::{legal_type}::{self._slugify(str(doc_number))}::{year}"

    def _build_parent_law_id(self, doc_type: str, doc_number: str, year: str, doc_title: str) -> str:
        basis = doc_number or doc_title or "unknown"
        return f"parent::{doc_type}::{self._slugify(basis)}::{year or 'unknown'}"

    def _parse_legal_basis_line(self, raw_line: str):
        references = []
        line = compact_whitespace(raw_line)
        for match in self.legal_ref_pattern.finditer(line):
            raw_type = compact_whitespace(match.group(1))
            tail = compact_whitespace(match.group(2))
            full_ref = compact_whitespace(f"{raw_type} {tail}")
            doc_type = self._canonical_doc_type(raw_type)
            year = self._extract_year(full_ref)
            doc_number = self._extract_doc_number(full_ref)
            parent_law_id = self._build_parent_law_id(
                doc_type=doc_type,
                doc_number=doc_number,
                year=year,
                doc_title=full_ref,
            )
            references.append(
                {
                    "basis_line": line,
                    "doc_type": doc_type,
                    "doc_number": doc_number,
                    "doc_year": year,
                    "doc_title": full_ref,
                    "parent_law_id": parent_law_id,
                }
            )
        return references

    def _extract_legal_basis_metadata(self, content: str, metadata: dict):
        preamble = "\n".join((content or "").splitlines()[:80])
        legal_basis_refs = []

        for line in preamble.splitlines():
            if self.legal_basis_line_pattern.match(line):
                legal_basis_refs.extend(self._parse_legal_basis_line(line))

        dedup_refs = []
        seen = set()
        for ref in legal_basis_refs:
            key = (ref.get("parent_law_id"), ref.get("doc_title"))
            if key in seen:
                continue
            seen.add(key)
            dedup_refs.append(ref)

        parent_law_ids = [r.get("parent_law_id") for r in dedup_refs if r.get("parent_law_id")]
        document_uid = self._build_document_uid(metadata)

        return {
            "document_uid": document_uid,
            "legal_basis_refs": dedup_refs,
            "parent_law_ids": sorted(set(parent_law_ids)),
        }

    def _normalize_metadata(self, metadata: Dict):
        data = dict(metadata)
        data["document_id"] = str(data.get("id", ""))
        data["legal_sectors"] = split_sector_list(data.get("legal_sectors"))
        data["signer"] = data.get("signer") or data.get("signers") or ""
        data["url"] = str(data.get("url") or data.get("link") or "")
        data["issuance_date"] = str(data.get("issuance_date") or data.get("issued_date") or data.get("issue_date") or "")
        data["is_active"] = data.get("is_active", True)
        data["document_uid"] = self._build_document_uid(data)
        return data

    def _make_breadcrumb(
        self,
        chapter_ref=None,
        article_ref=None,
        clause_ref=None,
        point_ref=None,
        is_appendix=False,
        section_label=None,
    ):
        parts = []
        if section_label:
            parts.append(section_label)
        if chapter_ref:
            parts.append(chapter_ref)
        if article_ref:
            parts.append(article_ref)
        if clause_ref:
            parts.append(clause_ref)
        if point_ref:
            parts.append(point_ref)
        if is_appendix and not section_label:
            parts.append("Phu luc")

        breadcrumb_path = " > ".join(parts) if parts else "Noi dung chung"
        return {"breadcrumb_path": breadcrumb_path}

    def _build_metadata_header(self, metadata, is_appendix, article_ref=None, clause_ref=None, chapter_ref=None, point_ref=None, breadcrumb_path=None):
        header_lines = [
            "[LEGAL HEADER]",
            f"- Title: {metadata.get('title', 'N/A')}",
            f"- Document number: {metadata.get('document_number', 'N/A')}",
            f"- Legal type: {metadata.get('legal_type', 'N/A')}",
            f"- Issuing authority: {metadata.get('issuing_authority', 'N/A')}",
            f"- Legal sectors: {'; '.join(metadata.get('legal_sectors', []))}",
            f"- Breadcrumb: {breadcrumb_path or 'N/A'}",
            f"- Chapter: {chapter_ref or 'N/A'}",
            f"- Article: {article_ref or 'N/A'}",
            f"- Clause: {clause_ref or 'N/A'}",
            f"- Point: {point_ref or 'N/A'}",
            f"- Is appendix: {is_appendix}",
            f"- Source URL: {metadata.get('url', 'N/A')}",
            "",
        ]
        return "\n".join(header_lines)

    def _build_citation(self, metadata, chapter_ref=None, article_ref=None, clause_ref=None, is_appendix=False, appendix_part=None, point_ref=None):
        pieces = [metadata.get("document_number") or metadata.get("title") or "Van ban"]
        if chapter_ref:
            pieces.append(chapter_ref)
        if article_ref:
            pieces.append(article_ref)
        if clause_ref:
            pieces.append(clause_ref)
        if point_ref:
            pieces.append(point_ref)
        if is_appendix and appendix_part:
            pieces.append(appendix_part)
        return " | ".join([piece for piece in pieces if piece])

    def _semantic_split_intro(self, text: str) -> List[str]:
        lines = text.splitlines()
        chunks = []
        cur_chunk = []
        for line in lines:
            if re.match(r"(?i)^\s*(căn cứ|chiếu theo|theo quy định|luật)\b", line):
                if cur_chunk:
                    chunks.append("\n".join(cur_chunk).strip())
                    cur_chunk = []
            cur_chunk.append(line)
        if cur_chunk:
            chunks.append("\n".join(cur_chunk).strip())
        return [c for c in chunks if c.strip()]

    def _semantic_split_appendix(self, text: str) -> List[str]:
        # Tách phụ lục theo các mục I., II., 1., 2., a), b)
        pattern = re.compile(r"(?im)^\s*(?:[IVXLCDM]+\.|[0-9]+\.|[a-z]\)|[-+])\s+")
        lines = text.splitlines()
        chunks = []
        cur_chunk = []
        for line in lines:
            if pattern.match(line):
                if cur_chunk:
                    chunks.append("\n".join(cur_chunk).strip())
                    cur_chunk = []
            cur_chunk.append(line)
        if cur_chunk:
            chunks.append("\n".join(cur_chunk).strip())
        return [c for c in chunks if c.strip()]

    def _split_main_and_appendix(self, content):
        match = self.appendix_pattern.search(content)
        if match:
            return content[: match.start()].strip(), content[match.start() :].strip()
        return content.strip(), ""

    def _split_articles(self, main_text):
        intro_lines = []
        articles = []
        current_article = None
        current_chapter = None

        for raw_line in main_text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                if current_article is None:
                    intro_lines.append("")
                else:
                    current_article["lines"].append("")
                continue

            chapter_match = self.chapter_pattern.match(line)
            if chapter_match:
                current_chapter = compact_whitespace(line)
                if current_article is None:
                    intro_lines.append(line)
                else:
                    current_article["lines"].append(line)
                continue

            article_match = self.article_pattern.match(line)
            if article_match:
                if current_article is not None:
                    articles.append(current_article)
                current_article = {
                    "chapter_ref": current_chapter,
                    "article_ref": compact_whitespace(article_match.group(1)),
                    "article_title": compact_whitespace(article_match.group(2)),
                    "lines": [line],
                }
                continue

            if current_article is None:
                intro_lines.append(line)
            else:
                current_article["lines"].append(line)

        if current_article is not None:
            articles.append(current_article)
        return "\n".join(intro_lines).strip(), articles

    def _split_clauses(self, article):
        full_article_text = "\n".join(article["lines"]).strip()
        body_lines = article["lines"][1:] if len(article["lines"]) > 1 else []
        clauses = []
        current_clause = None

        for raw_line in body_lines:
            line = raw_line.rstrip()
            clause_match = self.clause_pattern.match(line)
            if clause_match:
                if current_clause is not None:
                    clauses.append(current_clause)
                clause_ref = compact_whitespace(clause_match.group(1) or clause_match.group(3))
                clause_tail = compact_whitespace(clause_match.group(2) or clause_match.group(4))
                current_clause = {"clause_ref": clause_ref, "lines": [f"{clause_ref} {clause_tail}".strip()]}
                continue

            if current_clause is None:
                if line.strip():
                    current_clause = {"clause_ref": None, "lines": [line]}
            else:
                current_clause["lines"].append(line)

        if current_clause is not None:
            clauses.append(current_clause)
        if not clauses:
            clauses = [{"clause_ref": None, "lines": article["lines"]}]
        return full_article_text, clauses

    def _appendix_chunks(self, appendix_text, metadata, doc_id, graph_meta):
        chunks = []
        if not appendix_text:
            return chunks

        app_chunks = self._semantic_split_appendix(appendix_text)
        for part_idx, chunk_text in enumerate(app_chunks, start=1):
            breadcrumb = self._make_breadcrumb(
                is_appendix=True,
                section_label="Phu luc",
                article_ref="Phu luc",
                clause_ref=f"P{part_idx}",
            )
            header_app = self._build_metadata_header(
                metadata=metadata,
                is_appendix=True,
                article_ref="Phu luc",
                clause_ref=f"P{part_idx}",
                breadcrumb_path=breadcrumb["breadcrumb_path"],
            )
            citation = self._build_citation(
                metadata=metadata,
                chapter_ref=None,
                article_ref="Phu luc",
                clause_ref=None,
                is_appendix=True,
                appendix_part=f"P{part_idx}",
            )
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::appendix::{part_idx}",
                    "chunk_text": (
                        f"{header_app}[BREADCRUMB] {breadcrumb['breadcrumb_path']}\n"
                        f"[PHAN {part_idx}]\n{chunk_text.strip()}"
                    ),
                    "metadata": {
                        **metadata,
                        **graph_meta,
                        **breadcrumb,
                        "is_appendix": True,
                        "chapter_ref": None,
                        "article_id": f"{doc_id}::appendix::{part_idx}",
                        "article_ref": "Phu luc",
                        "clause_ref": f"P{part_idx}",
                        "reference_citation": citation,
                    },
                }
            )
        return chunks

    def process_document(self, content: str, metadata: dict) -> List[Dict]:
        metadata = self._normalize_metadata(metadata)
        # Chuẩn hóa văn bản: Chuyển \r\n thành \n, và thay thế các khoảng trắng lạ (như non-breaking space) thành dấu cách chuẩn
        content = str(content or "").replace("\r\n", "\n")
        content = re.sub(r"[^\S\n]+", " ", content).strip() 
        
        doc_id = metadata.get("document_id") or str(uuid.uuid4())
        graph_meta = self._extract_legal_basis_metadata(content=content, metadata=metadata)
        chunks = []

        main_text, appendix_text = self._split_main_and_appendix(content)
        intro_text, articles = self._split_articles(main_text)

        if intro_text:
            intro_sub_chunks = self._semantic_split_intro(intro_text)
            for idx, sub_text in enumerate(intro_sub_chunks):
                article_id = f"{doc_id}::preamble"
                ref_tag = "Noi dung" if len(intro_sub_chunks) == 1 else f"Noi dung (P{idx+1})"
                breadcrumb = self._make_breadcrumb(section_label="Mo dau", article_ref=ref_tag)
                citation = self._build_citation(metadata=metadata, article_ref=ref_tag, is_appendix=False)
                header = self._build_metadata_header(
                    metadata=metadata,
                    is_appendix=False,
                    article_ref=ref_tag,
                    breadcrumb_path=breadcrumb["breadcrumb_path"],
                )
                chunks.append(
                    {
                        "chunk_id": f"{article_id}::{idx}",
                        "chunk_text": (
                            f"{header}[BREADCRUMB] {breadcrumb['breadcrumb_path']}\n"
                            f"[NOI DUNG]\n{sub_text}"
                        ),
                        "metadata": {
                            **metadata,
                            **graph_meta,
                            **breadcrumb,
                            "is_appendix": False,
                            "chapter_ref": None,
                            "article_id": article_id,
                            "article_ref": ref_tag,
                            "clause_ref": None,
                            "reference_citation": citation,
                        },
                    }
                )

        for article_index, article in enumerate(articles, start=1):
            article_id = f"{doc_id}::article::{article_index}"
            full_article_text, clause_entries = self._split_clauses(article)

            for clause_index, clause in enumerate(clause_entries, start=1):
                clause_ref = clause["clause_ref"]
                unit_text = "\n".join(clause["lines"]).strip()
                if not unit_text:
                    continue

                breadcrumb = self._make_breadcrumb(
                    chapter_ref=article.get("chapter_ref"),
                    article_ref=article["article_ref"],
                    clause_ref=clause_ref,
                )
                citation = self._build_citation(
                    metadata=metadata,
                    chapter_ref=article.get("chapter_ref"),
                    article_ref=article["article_ref"],
                    clause_ref=clause_ref,
                    is_appendix=False,
                )
                header = self._build_metadata_header(
                    metadata=metadata,
                    is_appendix=False,
                    article_ref=article["article_ref"],
                    clause_ref=clause_ref,
                    chapter_ref=article.get("chapter_ref"),
                    breadcrumb_path=breadcrumb["breadcrumb_path"],
                )
                chunks.append(
                    {
                        "chunk_id": f"{article_id}::c{clause_index}",
                        "chunk_text": (
                            f"{header}[BREADCRUMB] {breadcrumb['breadcrumb_path']}\n"
                            f"[NOI DUNG DIEU/KHOAN]\n{unit_text}"
                        ),
                        "metadata": {
                            **metadata,
                            **graph_meta,
                            **breadcrumb,
                            "is_appendix": False,
                            "chapter_ref": article.get("chapter_ref"),
                            "article_id": article_id,
                            "article_ref": article["article_ref"],
                            "clause_ref": clause_ref,
                            "reference_citation": citation,
                        },
                    }
                )

        chunks.extend(self._appendix_chunks(appendix_text, metadata, doc_id, graph_meta))
        return chunks

chunker = AdvancedLegalChunker()
