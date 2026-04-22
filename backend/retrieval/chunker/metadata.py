import re
from typing import Any

def compact_whitespace(text: str) -> str:
    """Hàm dọn dẹp khoảng trắng dư thừa, giữ nguyên từ file cũ"""
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()

def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-") or "unknown"

def canonical_doc_type(raw: str) -> str:
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

def extract_year(text: str) -> str:
    """Bóc tách năm 4 chữ số từ chuỗi. Ưu tiên 19xx/20xx, fallback bất kỳ 4 chữ số."""
    t = str(text or "")
    # Ưu tiên: tìm năm 19xx hoặc 20xx với word boundary
    m = re.search(r"\b((?:19|20)\d{2})\b", t)
    if m:
        return m.group(1)
    # Fallback: tìm bất kỳ chuỗi 4 chữ số nào
    m2 = re.search(r"(\d{4})", t)
    if m2:
        yr = int(m2.group(1))
        if 1900 <= yr <= 2100:
            return str(yr)
    return ""

VALID_DOC_STATUSES = [
    "Đang có hiệu lực",
    "Hết hiệu lực một phần",
    "Hết hiệu lực toàn bộ",
    "Chưa có hiệu lực",
]

def normalize_doc_status(raw: str) -> str:
    """
    Ép giá trị doc_status về 1 trong 4 trạng thái chuẩn.
    Xử lý mọi giá trị rác từ LLM hoặc HF metadata.
    """
    if not raw or str(raw).strip().lower() in ("", "nan", "none", "n/a"):
        return "Đang có hiệu lực"  # default an toàn

    text = str(raw).strip().lower()

    # Kiểm tra "hết hiệu lực" trước (vì nó chứa từ "hiệu lực")
    if "hết hiệu lực" in text or "het hieu luc" in text:
        if "một phần" in text or "mot phan" in text or "1 phần" in text:
            return "Hết hiệu lực một phần"
        if "toàn bộ" in text or "toan bo" in text:
            return "Hết hiệu lực toàn bộ"
        # "Hết hiệu lực" không rõ → mặc định toàn bộ
        return "Hết hiệu lực toàn bộ"

    if "chưa" in text or "chua" in text or "sắp" in text:
        return "Chưa có hiệu lực"

    if ("còn" in text or "đang" in text or "con hieu" in text
            or "dang co" in text or "có hiệu lực" in text or "hieu luc" in text):
        return "Đang có hiệu lực"

    # Fallback: không nhận ra → mặc định
    return "Đang có hiệu lực"


def extract_doc_number(text: str) -> str:
    patterns = [r"(?i)(?:số\s*)?(\d+\/\d+(?:\/[A-Z0-9Đ\-]+)?)", r"(?i)(\d+\/[A-Z0-9Đ\-]+)"]
    for p in patterns:
        m = re.search(p, text or "")
        if m: return compact_whitespace(m.group(1))
    return ""

def parse_signer(signer_raw: Any) -> tuple:
    signer_str = str(signer_raw) if signer_raw is not None else ""
    if signer_str.strip().lower() in ["nan", "", "none"]:
        return "", None
    if ":" not in signer_str:
        return signer_str, None
    parts = signer_str.split(":")
    name = parts[0].strip()
    try:
        return name, int(parts[1].strip())
    except ValueError:
        return name, None

# ==========================================
# REGEX PATTERNS (Cơ bản lấy từ code cũ)
# ==========================================
chapter_pattern = re.compile(r"(?im)^\s*(Chương|Phần)(?:\s+thứ)?\s+([a-zA-Z0-9]+|\d+)\b\s*[\.\:\-]?\s*(.*)$")
article_pattern = re.compile(r"(?im)^\s*(Điều|Mục)\s+(\d+[A-Za-z0-9\/\-]*)\s*[\.\:\-]?\s*(.*)$")
clause_pattern = re.compile(
    r"(?im)^\s*(Khoản\s+\d+[\.\:\-]?)\s*(.*)$|"  
    r"^\s*(\d+(?:\.\d+)*[\.\)])\s*(.*)$|"        
    r"^\s*(\(\d+\))\s*(.*)$|"                    
    r"^\s*([-+•])\s+(.*)$"                       
)
point_pattern = re.compile(r"(?im)^\s*([a-zđ]\s*[\)\.])\s*(.*)$")

appx_lvl1_pattern = re.compile(r"(?im)^\s*([IVXLCDM]+|[A-Z])\s*[\.\:\-]\s*(.*)$")
appx_lvl2_pattern = re.compile(r"(?im)^\s*(\d+)\s*[\.\:\-]\s*(.*)$")
appx_lvl3_pattern = re.compile(r"(?im)^\s*(\d+(?:\.\d+)+)\s*[\.\:\-]?\s*(.*)$")

substantive_title_pattern = re.compile(
    r"(?im)^\s*(QUY ĐỊNH|QUY CHẾ|QUY CHUẨN|QCVN|TIÊU CHUẨN|TCVN|PHƯƠNG ÁN|ĐIỀU LỆ|CHƯƠNG TRÌNH|HƯỚNG DẪN|NỘI QUY|KẾ HOẠCH|CHIẾN LƯỢC|ĐỀ ÁN|DỰ ÁN)\b(?!\s*CHUNG\b).*$"
)
appendix_title_pattern = re.compile(
    r"(?im)^\s*("
    r"(?:PHỤ\s+LỤC|PHU\s+LUC)(?:\s+(?:SỐ\s+)?(?:[IVXLCDM]+|\d+)|[A-Z])?(?:\s*[\:\-\.]|\s+BAN\s+HÀNH|\s+KÈM\s+THEO)?|"
    r"(?:MẪU|MẪU\s+SỐ|BIỂU\s+MẪU)\s*[A-Za-z0-9\.\-\/]*(?:\s*[\:\-\.]|\s+BAN\s+HÀNH|\s+KÈM\s+THEO)?|"
    r"DANH\s+MỤC(?:\s+(?:CHI\s+TIẾT|KÈM\s+THEO|CÁC|DỰ\s+ÁN|TÀI\s+SẢN|VẬT\s+TƯ|HÀNG\s+HÓA|QUỐC\s+GIA|MÃ))?"
    r")\b.*$"
)

legal_basis_line_pattern = re.compile(r"(?im)^\s*căn cứ\b.*$")
legal_ref_pattern = re.compile(r"(?i)\b(Hiến pháp|Bộ luật|Luật|Nghị quyết|Pháp lệnh|Nghị định|Thông tư)\b([^.;\n]*)")

effective_date_pattern = re.compile(r"(?i)có\s+hiệu\s+lực\s+(?:từ|kể\s+từ)\s+ngày\s+(\d{1,2})[/\- ](\d{1,2})[/\- ](\d{4})")
effective_from_sign_pattern = re.compile(
    r"(?i)("
    r"có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày\s+ký|"
    r"chịu\s+trách\s+nhiệm\s+thi\s+hành\s+(?:quyết\s+định|nghị\s+định|thông\s+tư|văn\s+bản)\s+này"
    r")"
)

final_article_trigger = re.compile(
    r"(?i)("
    r"có\s+hiệu\s+lực\s+(?:thi\s+hành\s+)?(?:kể\s+)?từ\s+ngày|"
    r"chịu\s+trách\s+nhiệm\s+thi\s+hành|"
    r"tổ\s+chức\s+thực\s+hiện"
    r")"
)

footer_pattern = re.compile(
    r"(?i)^\s*(nơi\s+nhận|kính\s+gửi)[\:\.]?|"
    r"^\s*(TM\.|KT\.|Q\.|TL\.|TUQ\.)?\s*"
    r"("
    r"CHÍNH\s+PHỦ|UBND|ỦY\s+BAN\s+NHÂN\s+DÂN|"
    r"BỘ\s+TRƯỞNG|CHỦ\s+TỊCH|THỨ\s+TRƯỞNG|GIÁM\s+ĐỐC|TỔNG\s+GIÁM\s+ĐỐC|"
    r"CỤC\s+TRƯỞNG|TỔNG\s+CỤC\s+TRƯỞNG|CHÁNH\s+VĂN\s+PHÒNG|"
    r"CHÁNH\s+ÁN|VIỆN\s+TRƯỞNG|TỔNG\s+KIỂM\s+TOÁN|CHỦ\s+NHIỆM|TỔNG\s+BÍ\s+THƯ"
    r")\b"
)

relationship_pattern = re.compile(
    r"(sửa đổi(?:,\s*bổ sung)?|bổ sung|thay thế|bãi bỏ|huỷ bỏ)"
    r"(.{0,150}?)"                                              
    r"(?:của\s+)?"                                              
    r"(Hiến pháp|Bộ luật|Luật|Pháp lệnh|Lệnh|Nghị quyết|Nghị định|Thông tư|Quyết định|Chỉ thị)" 
    r"(?:\s+số)?\s+([0-9]+/[0-9]{4}/[A-Z0-9Đ\-]+|[0-9]+/[A-Z0-9Đ\-]+)", 
    re.IGNORECASE
)

extract_article_pattern = re.compile(r"(điều\s+\d+[a-zA-ZđĐ]*)", re.IGNORECASE)
extract_clause_pattern = re.compile(r"(khoản\s+\d+[a-zA-ZđĐ]*)", re.IGNORECASE)
def normalize_doc_key(text: str) -> str:
    """Chuẩn hóa số hiệu VB thành key tra cứu: bỏ dấu cách, chấm, gạch, slash, viết hoa."""
    if not text:
        return ""
    return re.sub(r'[\s.\-/]', '', str(text)).strip().upper()

part_lesson_pattern = re.compile(
    r"(?im)^\s*(phần|tập|bài)\s+([ivxlcdm0-9]+)\s*[\.\:\-–—]?\s*(.+)?$",
    re.UNICODE
)
