import re
import uuid
from typing import Any, Dict, List

def compact_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()

class AdvancedLegalChunker:
    def __init__(self):
        # 1. NHẬN DIỆN PHỤ LỤC (Mở rộng)
        self.appendix_pattern = re.compile(
            r"(?im)^\s*("
            r"(?:Mẫu\s+số|Mẫu|Biểu\s+mẫu)[\s\d\w\.\-\:]*|" # Bắt mọi cụm bắt đầu bằng Mẫu số, Mẫu...
            r"PHỤ LỤC|PHU LUC|"
            r"DANH MỤC|DANH MUC|"
            r"BẢNG BIỂU|BANG BIEU|"
            r"PHƯƠNG ÁN|PHUONG AN|"
            r"QUY ĐỊNH|QUY DINH|"
            r"QUY CHẾ|QUY CHE"
            r")\b.*$"
        )

        self.chapter_pattern = re.compile(r"(?im)^\s*(Chương\s+[IVXLCDM0-9]+)\s*(.*)$")
        self.section_pattern = re.compile(r"(?im)^\s*((?:Mục|Phần)\s+[0-9A-ZĐ]+)\s*[\.\:\-]?\s*(.*)$")
        self.roman_pattern = re.compile(r"(?im)^\s*((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV)\.)\s*(.*)$")
        self.article_pattern = re.compile(r"(?im)^\s*((?:Điều|Dieu|Điêu)\s+\d+[A-Za-z0-9\/\-]*)\s*[\.\:\-]?\s*(.*)$")
        self.clause_pattern = re.compile(r"(?im)^\s*(Khoản\s+\d+[\.\:\-]?)\s*(.*)$|^\s*(\d+[\.\)])\s*(.*)$")

        self.legal_basis_line_pattern = re.compile(r"(?im)^\s*căn cứ\b.*$")
        self.legal_ref_pattern = re.compile(r"(?i)\b(Hiến pháp|Bộ luật|Luật|Nghị quyết|Pháp lệnh|Nghị định|Thông tư)\b([^.;\n]*)")
        self.effective_date_pattern = re.compile(r"(?i)có\s+hiệu\s+lực\s+(?:từ|kể\s+từ)\s+ngày\s+(\d{1,2})[/\- ](\d{1,2})[/\- ](\d{4})")
        self.effective_from_sign_pattern = re.compile(
            r"(?i)("
            r"có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày\s+ký|" # Loại 1: Hiệu lực từ ngày ký
            r"chịu\s+trách\s+nhiệm\s+thi\s+hành\s+(?:quyết\s+định|nghị\s+định|thông\s+tư|văn\s+bản)\s+này" # Loại 2: Chịu trách nhiệm thi hành
            r")"
        )

        # 2. NHẬN DIỆN ĐIỀU KHOẢN KẾT THÚC VÀ FOOTER
        self.final_article_trigger = re.compile(
            r"(?i)("
            r"có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày|"
            r"chịu\s+trách\s+nhiệm\s+thi\s+hành|"
            r"tổ\s+chức\s+thực\s+hiện"
            r")"
        )
        self.footer_pattern = re.compile(
            # Bắt chính xác cụm "Nơi nhận:" hoặc "Kính gửi:" ở đầu dòng
            r"(?i)^\s*(nơi\s+nhận|kính\s+gửi)[\:\.]?|"

            # Bắt các tiền tố chữ ký (Thay mặt, Ký thay, Thừa lệnh...)
            r"^\s*(TM\.|KT\.|Q\.|TL\.|TUQ\.)?\s*"

            # Bắt ĐẦY ĐỦ các chức danh lãnh đạo phổ biến nhất của bộ máy Nhà nước
            r"("
            r"CHÍNH\s+PHỦ|UBND|ỦY\s+BAN\s+NHÂN\s+DÂN|"
            r"BỘ\s+TRƯỞNG|CHỦ\s+TỊCH|THỨ\s+TRƯỞNG|GIÁM\s+ĐỐC|TỔNG\s+GIÁM\s+ĐỐC|"
            r"CỤC\s+TRƯỞNG|TỔNG\s+CỤC\s+TRƯỞNG|CHÁNH\s+VĂN\s+PHÒNG|"
            r"CHÁNH\s+ÁN|VIỆN\s+TRƯỞNG|TỔNG\s+KIỂM\s+TOÁN|CHỦ\s+NHIỆM|TỔNG\s+BÍ\s+THƯ"
            r")\b"
        )


    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-") or "unknown"

    @staticmethod
    def _canonical_doc_type(raw: str) -> str:
        text = (raw or "").lower()
        mapping = {
            "hiến pháp": "constitution", "hien phap": "constitution",
            "bộ luật": "code", "bo luat": "code",
            "luật": "law", "luat": "law",
            "pháp lệnh": "ordinance", "phap lenh": "ordinance",
            "nghị quyết": "resolution", "nghi quyet": "resolution",
            "nghị định": "decree", "nghi dinh": "decree",
            "thông tư": "circular", "thong tu": "circular"
        }
        for k, v in mapping.items():
            if k in text: return v
        return "other"

    @staticmethod
    def _extract_year(text: str) -> str:
        m = re.search(r"\b(19|20)\d{2}\b", text or "")
        return m.group(0) if m else ""

    @staticmethod
    def _extract_doc_number(text: str) -> str:
        patterns = [r"(?i)(?:số\s*)?(\d+\/\d+(?:\/[A-Z0-9Đ\-]+)?)", r"(?i)(\d+\/[A-Z0-9Đ\-]+)"]
        for p in patterns:
            m = re.search(p, text or "")
            if m: return compact_whitespace(m.group(1))
        return ""

    def _build_parent_law_id(self, doc_type: str, doc_number: str, year: str, doc_title: str) -> str:
        basis = doc_number or doc_title or "unknown"
        return f"parent::{doc_type}::{self._slugify(basis)}::{year or 'unknown'}"

    def _parse_legal_basis_line(self, raw_line: str):
        refs = []
        line = compact_whitespace(raw_line)
        for m in self.legal_ref_pattern.finditer(line):
            raw_type = compact_whitespace(m.group(1))
            tail = compact_whitespace(m.group(2))
            full_ref = compact_whitespace(f"{raw_type} {tail}")
            doc_type = self._canonical_doc_type(raw_type)
            year = self._extract_year(full_ref)
            doc_number = self._extract_doc_number(full_ref)
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
            if self.legal_basis_line_pattern.match(line):
                all_refs.extend(self._parse_legal_basis_line(line))
        dedup = []
        seen = set()
        for r in all_refs:
            key = (r.get("parent_law_id"), r.get("doc_title"))
            if key in seen: continue
            seen.add(key)
            dedup.append(r)
        return dedup

    def _extract_effective_date(self, content: str, promulgation_date: str) -> str:
        lines = content.splitlines()
        appendix_start_idx = len(lines)
        for i, line in enumerate(lines):
            if self.appendix_pattern.match(line):
                appendix_start_idx = i
                break
        main_body_lines = lines[:appendix_start_idx]
        search_lines = main_body_lines[:200]
        if len(main_body_lines) > 200:
            search_lines += main_body_lines[-200:]
        search_zone = "\n".join(search_lines)
        m = self.effective_date_pattern.search(search_zone)
        if m:
            day, month, year = m.groups()
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        if self.effective_from_sign_pattern.search(search_zone):
            return promulgation_date
        return ""

    @staticmethod
    def _parse_signer(signer_raw: str) -> tuple:
        if not signer_raw or ":" not in signer_raw:
            return signer_raw, None
        parts = signer_raw.split(":")
        name = parts[0].strip()
        try:
            return name, int(parts[1].strip())
        except:
            return name, None

    def process_document(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = str(content or "").replace("\r\n", "\n").strip()
        doc_id = str(metadata.get("id", uuid.uuid4()))
        doc_number = metadata.get("document_number", "N/A")
        signer_name, signer_id = self._parse_signer(metadata.get("signers", ""))
        promulgation_date = metadata.get("promulgation_date", "")
        eff_date = self._extract_effective_date(content, promulgation_date)
        final_effective_date = eff_date or metadata.get("effective_date", "") or promulgation_date
        basis_refs = self._extract_legal_basis_metadata(content)

        chunks: List[Dict[str, Any]] = []

        # --- State Tracking ---
        current_chapter = None
        current_section = None
        current_roman_head = None
        current_article_ref = None
        current_article_title = ""
        current_clauses_data = []
        article_preamble = []
        current_appendix_buffer = []

        is_in_appendix = False
        current_appendix_name = ""
        current_article_idx = 0
        current_appendix_idx = 0
        global_chunk_idx = 0
        found_final_article = False

        def flush_buffer():
            nonlocal global_chunk_idx, current_clauses_data, article_preamble, current_appendix_buffer
            if not current_clauses_data and not article_preamble and not current_appendix_buffer:
                return

            parts = []

            if is_in_appendix:
                parts.extend(current_appendix_buffer)
                cl_refs = []
            else:
                # [GIẢI QUYẾT YÊU CẦU 1]: Luôn đính kèm lời dẫn (preamble) vào mọi Point của Điều
                if article_preamble:
                    parts.append("\n".join(article_preamble))

                cl_refs = [c["ref"] for c in current_clauses_data]
                if current_clauses_data:
                    parts.extend([c["text"] for c in current_clauses_data])

            text_content = "\n".join(parts).strip()
            if not text_content: return

            global_chunk_idx += 1

            if is_in_appendix:
                breadcrumb = current_appendix_name if current_appendix_name else "Phụ lục"
                chunk_id_val = f"{doc_id}::appendix::{current_appendix_idx}::c{global_chunk_idx}"
                ref_citation = f"{doc_number} | {breadcrumb}"
                clause_label = "PHỤ LỤC"
                art_ref = None
                cl_ref_list = []
                current_appendix_buffer.clear()
            else:
                # [THAY ĐỔI]: Logic dự phòng Article Ref nếu không có chữ "Điều"
                primary_ref = current_article_ref
                if not primary_ref:
                    # Nếu không có Điều, lấy mốc cấu trúc cao nhất hiện có
                    primary_ref = current_section or current_roman_head
                
                art_full = f"{primary_ref}. {current_article_title}".strip(". ") if primary_ref else None

                if cl_refs:
                    cl_display = f"{cl_refs[0]} - {cl_refs[-1]}" if len(cl_refs) > 1 else cl_refs[0]
                    clause_label = cl_display.upper()
                    cl_ref_list = cl_refs
                else:
                    clause_label = current_article_ref.upper() if current_article_ref else "CHUNG"
                    cl_ref_list = []

                bc_parts = [current_chapter, current_section, current_roman_head, art_full, (cl_display if cl_refs else None)]
                breadcrumb = " > ".join([x for x in bc_parts if x])
                chunk_id_val = f"{doc_id}::article::{current_article_idx}::c{global_chunk_idx}"
                ref_citation = f"{doc_number} | {breadcrumb.replace(' > ', ' | ')}" if breadcrumb else doc_number
                art_ref = art_full # Sẽ chứa Mục I hoặc Phần A nếu không có Điều

            chunk_text = (
                f"[VĂN BẢN] {doc_number}\n"
                f"[VỊ TRÍ] {breadcrumb}\n"
                f"[NỘI DUNG {clause_label}]\n"
                f"{text_content}"
            )

            payload = {
                "document_id": doc_id,
                "chunk_index": global_chunk_idx,
                "document_number": doc_number,
                "title": metadata.get("title", "N/A"),
                "legal_type": metadata.get("legal_type", "N/A"),
                "legal_sectors": metadata.get("legal_sectors_list", []),
                "issuing_authority": metadata.get("issuing_authority", "N/A"),
                "signer_name": signer_name,
                "signer_id": signer_id,
                "url": metadata.get("url", ""),
                "promulgation_date": promulgation_date,
                "effective_date": final_effective_date,
                "is_active": True,
                "chapter_ref": current_chapter,
                "article_ref": art_ref,
                "clause_ref": cl_ref_list,
                "is_appendix": is_in_appendix,
                "reference_citation": ref_citation,
                "chunk_text": chunk_text,
                "legal_basis_refs": basis_refs
            }

            chunks.append({
                "chunk_id": chunk_id_val,
                "chunk_text": chunk_text,
                "metadata": payload
            })

        # --- XỬ LÝ TỪNG DÒNG TEXT ---
        lines = content.splitlines()
        for line in lines:
            line = line.strip()
            if not line: continue

            # [0] KIỂM TRA CÂU HIỆU LỰC NGAY LẬP TỨC TRƯỚC KHI BỊ LỆNH CONTINUE BỎ QUA
            if not is_in_appendix and not found_final_article:
                if self.final_article_trigger.search(line):
                    found_final_article = True

            # [1] KIỂM TRA PHỤ LỤC
            m_app = self.appendix_pattern.match(line)
            if m_app:
                flush_buffer()
                current_clauses_data = []
                article_preamble = []
                is_in_appendix = True
                current_appendix_idx += 1
                current_appendix_name = compact_whitespace(line)
                current_chapter = None
                current_article_ref = None
                current_article_title = ""
                continue

            # [2] KIỂM TRA FOOTER
            if not is_in_appendix and found_final_article:
                if self.footer_pattern.match(line) or (line.isupper() and len(line) > 5):
                    flush_buffer()
                    current_clauses_data = []
                    article_preamble = []
                    is_in_appendix = True
                    current_appendix_idx += 1
                    current_appendix_name = "Nơi nhận / Footer" if re.search(r"(?i)^(nơi\s+nhận|kính\s+gửi)", line) else "Chữ ký / Đóng dấu"
                    
                    # [TỐI ƯU]: Không xóa Art Ref nếu đang ở trong vùng Footer để giữ context (Inheritance)
                    # current_chapter = None 
                    # current_article_ref = None
                    # current_article_title = ""
                    current_appendix_buffer.append(line)
                    continue

            # [3] BẪY BẢNG BIỂU (MARKDOWN TABLE) SAU ĐIỀU CUỐI CÙNG
            if not is_in_appendix and found_final_article:
                if line.startswith("|"):
                    flush_buffer()
                    current_clauses_data = []
                    article_preamble = []
                    is_in_appendix = True
                    current_appendix_idx += 1
                    current_appendix_name = "Bảng biểu / Phụ lục đính kèm"
                    
                    # [GIẢI QUYẾT BẢNG GIÁ ĐẤT]: Giữ nguyên Article Ref của Điều ban hành bảng
                    # current_chapter = None
                    # current_article_ref = None
                    # current_article_title = ""
                    current_appendix_buffer.append(line)
                    continue

            # Xử lý Phụ lục
            if is_in_appendix:
                MAX_APP_LEN = 2000
                current_buffer_len = sum(len(s) for s in current_appendix_buffer)
                line_len = len(line)

                if current_buffer_len > 0 and (current_buffer_len + line_len) > MAX_APP_LEN:
                    flush_buffer()
                    current_buffer_len = 0

                if line_len > MAX_APP_LEN:
                    for i in range(0, line_len, MAX_APP_LEN):
                        sub_line = line[i : i + MAX_APP_LEN]
                        current_appendix_buffer.append(sub_line)
                        flush_buffer()
                else:
                    current_appendix_buffer.append(line)
                    if current_buffer_len + line_len > MAX_APP_LEN:
                        flush_buffer()
                continue

            # Phân tích Chương
            m_ch = self.chapter_pattern.match(line)
            if m_ch:
                flush_buffer()
                current_clauses_data = []
                article_preamble = []
                current_chapter = compact_whitespace(f"{m_ch.group(1)}. {m_ch.group(2)}")
                continue

            # Phân tích Mục/Phần
            m_sec = self.section_pattern.match(line)
            if m_sec:
                flush_buffer()
                current_clauses_data = []
                article_preamble = []
                current_section = compact_whitespace(f"{m_sec.group(1)}. {m_sec.group(2)}")
                current_roman_head = None # Reset Roman khi có Mục mới
                current_article_ref = None
                continue

            # Phân tích số La Mã (I, II, III...)
            m_rom = self.roman_pattern.match(line)
            if m_rom:
                flush_buffer()
                current_clauses_data = []
                article_preamble = []
                current_roman_head = compact_whitespace(f"{m_rom.group(1)}. {m_rom.group(2)}")
                current_article_ref = None
                continue

            # Phân tích Điều
            m_ar = self.article_pattern.match(line)
            if m_ar:
                flush_buffer()
                current_clauses_data = []
                article_preamble = []
                current_article_idx += 1
                current_article_ref = m_ar.group(1).strip()
                article_remainder = m_ar.group(2).strip()
                current_article_title = article_remainder[:300] + "..." if len(article_remainder) > 300 else article_remainder

                if article_remainder:
                    article_preamble.append(article_remainder)
                continue

            # Phân tích Khoản
            m_cl = self.clause_pattern.match(line)
            if m_cl and current_article_ref:
                # [GIẢI QUYẾT YÊU CẦU 2]: Xả thông minh TRƯỚC khi nạp Khoản mới
                current_len = sum(len(c["text"]) for c in current_clauses_data) + sum(len(p) for p in article_preamble)
                num_clauses = len(current_clauses_data)

                should_flush = False

                if num_clauses >= 3:
                    should_flush = True # Đã đủ 3 khoản -> Xả
                elif num_clauses == 2 and current_len > 1500:
                    should_flush = True # 2 khoản nhưng khá nặng -> Xả sớm
                elif num_clauses >= 1 and current_len > 2500:
                    should_flush = True # Dù mới có 1-2 khoản nhưng đã quá to (>2500) -> Xả để nhường chỗ cho Khoản mới

                if should_flush:
                    last_clause = current_clauses_data[-1]
                    flush_buffer()

                    if num_clauses > 1:
                        current_clauses_data = [last_clause] # Chồng lấn 1 khoản
                    else:
                        current_clauses_data = []

                cl_ref = compact_whitespace(m_cl.group(1) or m_cl.group(3))
                cl_text = compact_whitespace(m_cl.group(2) or m_cl.group(4))

                current_clauses_data.append({
                    "ref": cl_ref,
                    "text": f"{cl_ref} {cl_text}".strip() if cl_text else cl_ref
                })
                continue

            # ==========================================
            # TEXT BÌNH THƯỜNG & CƠ CHẾ TÁCH HỒI TỐ
            # ==========================================
            if len(line) > 3000 and not is_in_appendix:
                flush_buffer()
                is_in_appendix = True
                current_appendix_name = "Nội dung bảng biểu siêu dài"
                current_appendix_buffer.append(line)
                continue

            if current_clauses_data:
                current_len = sum(len(c["text"]) for c in current_clauses_data) + sum(len(p) for p in article_preamble)
                projected_len = current_len + len(line) + 1

                if projected_len > 4000:
                    if len(current_clauses_data) > 1:
                        active_monster = current_clauses_data.pop()
                        flush_buffer()
                        active_monster["text"] += "\n" + line
                        current_clauses_data = [active_monster]
                    else:
                        flush_buffer()
                        last_ref = current_clauses_data[0]["ref"] if current_clauses_data else "Khoản"
                        current_clauses_data = [{"ref": last_ref, "text": f"{last_ref} (tiếp theo)\n{line}"}]
                else:
                    current_clauses_data[-1]["text"] += "\n" + line

            # XỬ LÝ LỜI DẪN (PREAMBLE)
            else:
                projected_preamble_len = sum(len(p) for p in article_preamble) + len(line)
                # KỂ KIỂM TRA TRƯỚC KHI APPEND ĐỂ TRÁNH PHÌNH TO CHUNK
                if projected_preamble_len > 4000:
                    flush_buffer()
                    article_preamble = ["[Nội dung lời dẫn quá dài, phần trước đã được chuyển sang Chunk trước...]", line]
                else:
                    article_preamble.append(line)

            # --- Gắn cờ nếu phát hiện đây là Điều khoản hiệu lực ---


        flush_buffer()
        return chunks

chunker = AdvancedLegalChunker()
print("Advanced Chunker with Footer/Appendix Trap is ready.")