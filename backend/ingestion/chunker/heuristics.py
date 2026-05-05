"""
heuristics.py — Bộ lọc heuristics cho chunker.

Cung cấp 3 hàm:
  has_potential_entities(text)  — chunk có thể chứa thực thể cần NER
  has_potential_relations(text) — chunk có thể chứa quan hệ liên văn bản [MỚI]
  is_meaningful_paragraph(text) — lọc cụm rác quá ngắn

Được gọi TRỰC TIẾP trong payload builders khi emit chunk,
đảm bảo chỉ cần 1 lần quét FSM là đã có nhãn entity/relation hint.
"""
import re


# ==========================================
# 1. ENTITY HINTS
# ==========================================
_ENTITY_KEYWORDS = re.compile(
    r"(?i)\b("
    r"bộ|sở|ban|ngành|ủy\s+ban|uy\s+ban|tổng\s+cục|cục|chi\s+cục|đoàn|hội\s+đồng|"
    r"chính\s+phủ|quốc\s+hội|đảng|nhà\s+nước|trung\s+ương|địa\s+phương|"
    r"doanh\s+nghiệp|công\s+ty|tập\s+đoàn|tổng\s+công\s+ty|tổ\s+chức|"
    r"đề\s+án|dự\s+án|chương\s+trình|kế\s+hoạch|chiến\s+lược|quỹ"
    r")\b"
)


def has_potential_entities(text: str) -> bool:
    """
    Regex lọc các chunk tiềm năng chứa thực thể (Cơ quan, Đề án, Tổ chức, Hội đồng...).
    Giúp giảm lượng token gọi LLM NER không cần thiết.
    """
    if not text:
        return False
    return bool(_ENTITY_KEYWORDS.search(text))


# ==========================================
# 2. RELATION HINTS  [MỚI]
# ==========================================
_RELATION_VERBS = re.compile(
    r"(?i)\b("
    r"sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|hết\s+hiệu\s+lực|hủy\s+bỏ|đình\s+chỉ|"
    r"đính\s+chính|giao\s+(?:cho|trách\s+nhiệm|nhiệm\s+vụ)|ủy\s+quyền|phân\s+công|"
    r"căn\s+cứ|áp\s+dụng|hướng\s+dẫn\s+thi\s+hành|quy\s+định\s+chi\s+tiết|"
    r"ban\s+hành\s+kèm\s+theo|thực\s+hiện\s+theo|ngưng\s+hiệu\s+lực|chấm\s+dứt\s+hiệu\s+lực"
    r")\b"
)

# Số hiệu văn bản chuẩn: XX/XXXX/XX-XX hoặc XX/XX-XX (tối thiểu 3 ký tự sau dấu /)
_DOC_NUM_HINT = re.compile(
    r"\b\d+[\/\-](?:20\d{2}|19\d{2})[\/\-][A-ZĐABCDEFGHIJKLMNOPQRSTUVWXYZ][A-Z0-9Đ\-\/]+\b"
    r"|\b\d+[\/\-][A-ZĐ][A-Z0-9Đ\-\/]{2,}\b",
    re.IGNORECASE,
)


def has_potential_relations(text: str) -> bool:
    """
    Chunk được đánh dấu 'tiềm năng chứa quan hệ liên văn bản' khi ĐỒNG THỜI:
      1. Có từ khóa hành động quan hệ (sửa đổi, bãi bỏ, căn cứ, thay thế...)
      2. Có số hiệu văn bản dạng xx/xxxx/XX-XX (đề phòng match nội-văn-bản)

    Yêu cầu cả hai điều kiện để giảm false-positive tối đa.
    """
    if not text:
        return False
    return bool(_RELATION_VERBS.search(text)) and bool(_DOC_NUM_HINT.search(text))


# ==========================================
# 3. GARBAGE FILTER
# ==========================================
_LETTER_ONLY_RE = re.compile(
    r"[^a-zA-Záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệ"
    r"íìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĐđ]+"
)


def is_meaningful_paragraph(text: str) -> bool:
    """
    Lọc bỏ các cụm rác quá ngắn hoặc toàn số, ký tự đặc biệt.
    """
    if not text:
        return False
    letters = _LETTER_ONLY_RE.sub("", text)
    return len(letters) > 20
