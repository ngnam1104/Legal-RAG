"""
entities.py — Trích xuất Thực thể + Quan hệ Node từ văn bản pháp luật.

Exports:
  FIXED_ENTITY_TYPES       — danh sách 12 loại entity cố định
  FIXED_NODE_RELATIONS     — danh sách 49 loại node relation chuẩn hoá (closed set)
  build_unified_prompt()   — tạo prompt unified từ batch contexts
  parse_unified_response() — parse JSON response thành {doc_relations, entities, node_relations}
"""
import json
import re
from typing import List, Dict, Any

from backend.models.llm_factory import get_client
from backend.utils.text_utils import extract_json_from_text
from backend.prompt import LEGAL_UNIFIED_EXTRACTOR_PROMPT

llm_client = get_client()

# ==========================================
# TẬP CỐ ĐỊNH + TẬP ĐỘNG (tích lũy trong pipeline run)
# ==========================================

FIXED_ENTITY_TYPES = {
    "Organization", "Person", "Location", "Document", "LegalArticle",
    "Procedure", "Condition", "Fee", "Penalty", "Timeframe", "Role", "Concept",
}

FIXED_NODE_RELATIONS = {
    # === BAN HÀNH / THẨM QUYỀN ===
    "ISSUED_BY",        # ban hành bởi
    "SIGNED_BY",        # ký bởi
    "APPROVED_BY",      # phê duyệt bởi
    "PUBLISHED_BY",     # công bố bởi
    "CREATED_BY",       # tạo ra bởi
    "ESTABLISHED_BY",   # thành lập bởi
    # === THỰC HIỆN / THI HÀNH ===
    "IMPLEMENTED_BY",   # thực hiện bởi
    "ENFORCED_BY",      # cưỡng chế/kiểm tra bởi
    "APPLIED_BY",       # áp dụng bởi
    "EXECUTED_BY",      # thực thi bởi
    # === QUẢN LÝ ===
    "MANAGED_BY",       # quản lý (gộp: governed, supervised, directed, operated_under)
    "REGULATED_BY",     # điều tiết/điều chỉnh
    "COORDINATED_BY",   # phối hợp bởi
    # === CHUYỂN GIAO / GỬI ===
    "TRANSFERRED_TO",   # chuyển đến
    "TRANSFERRED_FROM", # chuyển từ
    "SUBMITTED_TO",     # nộp/gửi đến
    "DELEGATED_TO",     # ủy quyền cho
    "ASSIGNED_TO",      # giao cho
    "ASSIGNED_BY",      # được giao bởi
    # === BÁO CÁO / THÔNG BÁO ===
    "REPORTED_TO",      # báo cáo đến (gộp: notified_to, informs)
    "NOTIFIED_TO",      # thông báo chính thức
    # === QUYỀN / CẤM / MIỄN ===
    "PERMITTED_TO",     # được phép
    "PROHIBITED_FROM",  # bị cấm
    "EXEMPT_FROM",      # được miễn
    "ENTITLED_TO",      # có quyền/được hưởng
    # === YÊU CẦU / TUÂN THỦ ===
    "REQUIRED_FOR",     # cần thiết cho
    "REQUIRED_BY",      # được yêu cầu bởi
    "COMPLIES_WITH",    # tuân thủ quy định
    "AFFECTED_BY",      # bị ảnh hưởng bởi
    # === PHÂN LOẠI / ĐỊNH NGHĨA ===
    "DEFINED_IN",       # được định nghĩa trong
    "CLASSIFIED_AS",    # được phân loại là
    "BELONGS_TO",       # thuộc về
    "PART_OF",          # là một phần của
    "LOCATED_IN",       # nằm trong/tại
    "MEMBER_OF",        # là thành viên của
    # === TÀI CHÍNH ===
    "FUNDED_BY",        # được tài trợ/cấp vốn bởi
    "PAID_TO",          # thanh toán cho
    "PAID_BY",          # được thanh toán bởi
    "COLLECTED_BY",     # được thu bởi
    # === VĂN BẢN PHÁP LÝ ===
    "REPLACED_BY",      # được thay thế bởi
    "AMENDED_BY",       # được sửa đổi bởi
    "REPEALED_BY",      # được bãi bỏ bởi
    "REFERENCED_BY",    # được tham chiếu bởi
    "BASED_ON",         # dựa trên/căn cứ
    # === CHUNG ===
    "APPLIES_TO",       # áp dụng cho
    "RELATED_TO",       # liên quan đến (fallback cuối)
}

# Quan hệ LLM hay sinh nhầm — fallback sang RELATED_TO
# Bao gồm: property-as-relationship, timestamp, số đo
BLACKLIST_RELATIONS = {
    "IS", "HAS", "DEADLINE", "EFFECTIVE_FROM", "HAS_MINIMUM_SIZE",
    "PUBLISHED_ON", "STARTS_AT", "ENDS_AT", "IS_EQUAL_TO", "NOT_EQUAL_TO",
    "MUST_NOT_BE_HIGHER_THAN", "MUST_NOT_BE_LOWER_THAN",
    "OCCURS_AT", "OCCURS_EVERY", "EXPIRES_ON", "MEETS_EVERY",
    "UPDATED_EVERY", "EXECUTED_AT", "EXECUTED_ON",
}
# Regex: bắt toàn bộ HAS_* property giả

FIXED_DOC_RELATIONS = {
    "BASED_ON", "AMENDS", "REPEALS", "REPLACES",
    "GUIDES", "APPLIES_TO", "ISSUED_WITH", "ASSIGNS", "CORRECTS",
    "AMENDED_BY", "REPEALED_BY", "REPLACED_BY", "GUIDED_BY",
    "REFERENCED_BY",
}

# Bảng chuyển đổi verb-root → canonical passive relation
# Được dùng trong fuzzy matching cuối cùng
_VERB_ROOT_CANONICAL = {
    "ISSUE":       "ISSUED_BY",
    "SIGN":        "SIGNED_BY",
    "APPROV":      "APPROVED_BY",
    "PUBLISH":     "PUBLISHED_BY",
    "CREAT":       "CREATED_BY",
    "ESTABLISH":   "ESTABLISHED_BY",
    "IMPLEMENT":   "IMPLEMENTED_BY",
    "ENFORC":      "ENFORCED_BY",
    "APPLY":       "APPLIED_BY",
    "APPLI":       "APPLIED_BY",
    "EXECUT":      "EXECUTED_BY",
    "PERFORM":     "IMPLEMENTED_BY",
    "CARRY":       "IMPLEMENTED_BY",
    "MANAG":       "MANAGED_BY",
    "GOVERN":      "MANAGED_BY",
    "SUPERVIS":    "MANAGED_BY",
    "DIRECT":      "MANAGED_BY",
    "REGULAT":     "REGULATED_BY",
    "COORDINAT":   "COORDINATED_BY",
    "TRANSFER":    "TRANSFERRED_TO",
    "SUBMIT":      "SUBMITTED_TO",
    "DELEGAT":     "DELEGATED_TO",
    "ASSIGN":      "ASSIGNED_TO",
    "REPORT":      "REPORTED_TO",
    "NOTIF":       "NOTIFIED_TO",
    "INFORM":      "NOTIFIED_TO",
    "PERMIT":      "PERMITTED_TO",
    "PROHIBIT":    "PROHIBITED_FROM",
    "EXEMPT":      "EXEMPT_FROM",
    "ENTITL":      "ENTITLED_TO",
    "REQUIR":      "REQUIRED_FOR",
    "AFFECT":      "AFFECTED_BY",
    "DEFIN":       "DEFINED_IN",
    "CLASSIF":     "CLASSIFIED_AS",
    "BELONG":      "BELONGS_TO",
    "FUND":        "FUNDED_BY",
    "PAY":         "PAID_TO",
    "COLLECT":     "COLLECTED_BY",
    "REPLAC":      "REPLACED_BY",
    "AMEND":       "AMENDED_BY",
    "REPEAL":      "REPEALED_BY",
    "REFERENC":    "REFERENCED_BY",
    "RELAT":       "RELATED_TO",
}

# Tập động: ghi nhận nhãn mới LLM tạo trong quá trình pipeline
# → các lần normalize sau sẽ khoanh ngắ luôn mà không force fallback
DYNAMIC_ENTITY_TYPES: set = set()
DYNAMIC_NODE_RELATIONS: set = set()
DYNAMIC_DOC_RELATIONS: set = set()

# Cấu trúc rỗng ban đầu (nếu cần fallback)
_EMPTY_ENTITIES: Dict[str, List[str]] = {}

# Regex kiểm tra format hợp lệ
_RE_PASCAL    = re.compile(r'^[A-Z][a-zA-Z]{1,29}$')       # PascalCase 2–30 ký tự
_RE_SCREAMING = re.compile(r'^[A-Z][A-Z0-9_]{1,39}$')      # SCREAMING_SNAKE_CASE

ENTITY_NAME_ALIASES = {
    "Organization": {
        "bộ gdđt": "Bộ Giáo dục và Đào tạo",
        "bộ gd&đt": "Bộ Giáo dục và Đào tạo",
        "bộ gd đt": "Bộ Giáo dục và Đào tạo",
        "bộ giáo dục đào tạo": "Bộ Giáo dục và Đào tạo",
        "bộ yt": "Bộ Y tế",
        "bộ y tế": "Bộ Y tế",
        "ubnd": "Ủy ban nhân dân",
        "uỷ ban nhân dân": "Ủy ban nhân dân",
        "hđnd": "Hội đồng nhân dân",
        "chính phủ": "Chính phủ",
        "nhà nước": "Nhà nước"
    }
}

def _normalize_entity_name(raw_name: str, ent_type: str) -> str:
    """Chuẩn hóa giá trị của thực thể dựa trên từ điển Alias (chống viết tắt)."""
    if not raw_name: return ""
    name = str(raw_name).strip()
    if not name: return ""
    
    lower_name = name.lower()
    
    # BỘ LỌC ĐỒNG THAM CHIẾU (COREFERENCE FILTER)
    # Chặn các đại từ chỉ định lấp lửng (garbage nodes)
    garbage_exact_matches = {"cơ quan", "tổ chức", "cá nhân", "đối tượng", "người", "văn bản", "điều", "khoản", "chương", "phần", "mục"}
    garbage_suffixes = (" này", " đó", " nêu trên", " dưới đây", " sau đây")
    
    if lower_name in garbage_exact_matches or lower_name.endswith(garbage_suffixes):
        return ""  # Trả về rỗng để luồng Parse vứt bỏ luôn
        
    if ent_type in ENTITY_NAME_ALIASES:
        if lower_name in ENTITY_NAME_ALIASES[ent_type]:
            return ENTITY_NAME_ALIASES[ent_type][lower_name]
            
    # Normalize khoảng trắng
    name = re.sub(r'\s+', ' ', name)
    # Capitalize đầu câu: chữ đầu tiên hoa, giữ nguyên phần còn lại
    if name:
        name = name[0].upper() + name[1:]
    return name


def _dedup_entity_values(values: list[str]) -> tuple[list[str], dict[str, str]]:
    """
    Khu trung entity values -- CHI exact case-insensitive dedup.
    Tra ve (deduped_list, alias_map) de redirect relations.

    LY DO KHONG dung substring containment:
    Trong domain phap ly, "Quyet dinh xu phat" va "Quyet dinh xu phat linh vuc y te"
    la 2 entity KHAC NHAU du cai sau chua cai truoc.
    Substring merge se gay sai noi dung (linh vuc A bi gan thanh linh vuc B).

    Chi merge khi 2 gia tri GIONG HET nhau sau khi lowercase.
    Uu tien giu form viet hoa dau cau (chuan hon).
    """
    alias_map: dict[str, str] = {}  # {removed_lower → canonical_value}

    if len(values) <= 1:
        return values, alias_map

    # Bước 1: Dedup exact case-insensitive
    seen_lower: dict[str, str] = {}  # lower → best form
    for v in values:
        lo = v.lower()
        if lo not in seen_lower:
            seen_lower[lo] = v
        else:
            existing = seen_lower[lo]
            # Ưu tiên form viết hoa đầu câu
            if v[0].isupper() and not existing[0].isupper():
                alias_map[existing.lower()] = v   # existing bị thay → redirect sang v
                seen_lower[lo] = v
            else:
                alias_map[v.lower()] = existing   # v bị bỏ → redirect sang existing
    return list(seen_lower.values()), alias_map


def _normalize_entity_type(raw_type: str) -> str:
    """
    ƯU TIÊN: FIXED → DYNAMIC → chấp nhận nhãn mới hợp lệ (PascalCase)
    Dự phòng cuối: 'Concept' khi nhãn thực sự vô nghĩa.
    """
    if not raw_type:
        return "Concept"
    s = raw_type.strip()

    # 0. Alias Mapper (Gộp các nhãn đồng nghĩa)
    if s in ["Article"]:                   s = "LegalArticle"
    if s in ["Authority", "Institution"]:  s = "Organization"
    if s in ["Signer", "PersonRole"]:      s = "Person"

    # 1. Exact match trong cả 2 tập
    if s in FIXED_ENTITY_TYPES or s in DYNAMIC_ENTITY_TYPES:
        return s

    # 2. Case-insensitive match (Fixed trước, sau đó Dynamic)
    for ft in (*FIXED_ENTITY_TYPES, *DYNAMIC_ENTITY_TYPES):
        if ft.lower() == s.lower():
            return ft

    # 3. Prefix fuzzy match trong Fixed (tái sử dụng nhãn gần giống)
    lower = s.lower()
    for ft in FIXED_ENTITY_TYPES:
        if lower.startswith(ft.lower()[:4]) or ft.lower().startswith(lower[:4]):
            return ft

    # 4. nhãn mới: chấp nhận nếu format PascalCase hợp lệ
    if _RE_PASCAL.match(s):
        DYNAMIC_ENTITY_TYPES.add(s)
        return s

    # 5. Fallback thực sự (nhãn rác)
    return "Concept"


def _normalize_relationship(raw_rel: str) -> str:
    """
    Pipeline chuẩn hoá quan hệ — CLOSED SET, không tạo DYNAMIC mới:
    0. HAS_* regex → RELATED_TO
    1. Blacklist → RELATED_TO
    2. Comprehensive alias_map (chủ động→bị động, đồng nghĩa→canonical)
    3. Exact match FIXED
    4. Verb-root fuzzy match → FIXED canonical
    5. Fallback RELATED_TO
    """
    if not raw_rel:
        return "RELATED_TO"
    s = raw_rel.strip().upper().replace(" ", "_")

    # 0. Ánh xạ cận nghĩa / dọn rác
    if s in _CROSS_VERB_MAPPING:
        s = _CROSS_VERB_MAPPING[s]

    # 0.5 HAS_* property check (wipe out hallucinated properties)
    if s.startswith("HAS_") and s not in {"HAS_ENTITY", "HAS_ARTICLE", "HAS_TYPE", "HAS_SECTOR"}:
        return "RELATED_TO"

    # 1. Blacklist
    if s in BLACKLIST_RELATIONS:
        return "RELATED_TO"

    # 2. Comprehensive alias map
    _ALIAS: dict = {
        # --- Chủ động → Bị động (Issuance / Acceptance) ---
        "ISSUES":              "ISSUED_BY",
        "ISSUED":              "ISSUED_BY",
        "SIGNS":               "SIGNED_BY",
        "SIGNED":              "SIGNED_BY",
        "APPROVES":            "APPROVED_BY",
        "APPROVED":            "APPROVED_BY",
        "PUBLISHES":           "PUBLISHED_BY",
        "PUBLISHED":           "PUBLISHED_BY",
        "PUBLISHES_IN":        "PUBLISHED_BY",
        "CREATES":             "CREATED_BY",
        "CREATED":             "CREATED_BY",
        "ESTABLISHES":         "ESTABLISHED_BY",
        "ESTABLISHED":         "ESTABLISHED_BY",
        "ACCEPTS":             "ACCEPTED_BY",
        "CONFIRMS":            "CONFIRMED_BY",
        "ACKNOWLEDGES":        "ACKNOWLEDGED_BY",
        # --- Implementation ---
        "IMPLEMENTS":          "IMPLEMENTED_BY",
        "PERFORMS":            "IMPLEMENTED_BY",
        "PERFORMED_BY":        "IMPLEMENTED_BY",
        "CARRIED_OUT_BY":      "IMPLEMENTED_BY",
        "DIRECTS_IMPLEMENTATION_OF": "IMPLEMENTED_BY",
        "GUIDES_IMPLEMENTATION_OF":  "IMPLEMENTED_BY",
        "ENFORCES":            "ENFORCED_BY",
        "APPLIES":             "APPLIED_BY",
        "EXECUTES":            "EXECUTED_BY",
        "EXECUTED":            "EXECUTED_BY",
        "EXECUTES_AT":         "EXECUTED_BY",
        # --- Management synonyms → MANAGED_BY ---
        "MANAGES":             "MANAGED_BY",
        "GOVERNS":             "MANAGED_BY",
        "GOVERNED_BY":         "MANAGED_BY",
        "SUPERVISED_BY":       "MANAGED_BY",
        "SUPERVISES":          "MANAGED_BY",
        "DIRECTED_BY":         "MANAGED_BY",
        "DIRECTS":             "MANAGED_BY",
        "LEADS":               "MANAGED_BY",
        "LEAD_BY":             "MANAGED_BY",
        "OPERATES_UNDER":      "MANAGED_BY",
        "SUBORDINATE_TO":      "MANAGED_BY",
        "IS_RESPONSIBLE_FOR":  "MANAGED_BY",
        "RESPONSIBLE_FOR":     "MANAGED_BY",
        "ACCOUNTABLE_TO":      "MANAGED_BY",
        "IS_HIGHEST_AUTHORITY_OF": "MANAGED_BY",
        "CHAIRMANED_BY":       "MANAGED_BY",
        "ASSESSES":            "MANAGED_BY",
        "DECIDES":             "DECIDED_BY",
        "DETERMINES":          "DETERMINED_BY",
        "ORGANIZES":           "ORGANIZED_BY",
        "INITIATES":           "INITIATED_BY",
        "GUIDES":              "GUIDED_BY",
        "HANDLES":             "HANDLED_BY",
        # --- Regulation ---
        "REGULATES":           "REGULATED_BY",
        "COORDINATES":         "COORDINATED_BY",
        "COORDINATED_WITH":    "COORDINATED_BY",
        "COORDINATES_WITH":    "COORDINATED_BY",
        "CONTRACTS_WITH":      "COOPERATES_WITH",
        "ORGANIZES_WITH":      "ORGANIZED_WITH",
        "PARTICIPATES_IN":     "PARTICIPATED_BY",
        "CONSULTED_WITH":      "CONSULTED_BY",
        # --- Transfer / Assignment ---
        "TRANSFERS":           "TRANSFERRED_TO",
        "TRANSFERS_TO":        "TRANSFERRED_TO",
        "TRANSFERRED_BY":      "TRANSFERRED_TO",
        "TRANSFERRED_VIA":     "TRANSFERRED_TO",
        "TRANSPORTS":          "TRANSFERRED_TO",
        "DELIVERS":            "TRANSFERRED_TO",
        "DELIVERED_TO":        "TRANSFERRED_TO",
        "SUBMITS":             "SUBMITTED_TO",
        "SUBMITS_TO":          "SUBMITTED_TO",
        "SUBMITTED_BY":        "SUBMITTED_TO",
        "DELEGATES":           "DELEGATED_TO",
        "DELEGATES_TO":        "DELEGATED_TO",
        "ASSIGNS":             "ASSIGNED_TO",
        "ASSIGNED":            "ASSIGNED_TO",
        "RETURNED_TO":         "TRANSFERRED_TO",
        "RETURNED_BY":         "TRANSFERRED_TO",
        "APPOINTS":            "APPOINTED_BY",
        "EMPLOYS":             "EMPLOYED_BY",
        "IMPORTS":             "IMPORTED_BY",
        "PROPOSED_TO":         "SUBMITTED_TO",
        # --- Reporting / Notification ---
        "REPORTS":             "REPORTED_TO",
        "REPORTS_TO":          "REPORTED_TO",
        "REPORTED_BY":         "REPORTED_TO",
        "REPORTS_ON":          "REPORTED_TO",
        "REPORTS_VIOLATIONS_TO": "REPORTED_TO",
        "NOTIFIES":            "NOTIFIED_TO",
        "NOTIFIED_BY":         "NOTIFIED_TO",
        "INFORMS":             "NOTIFIED_TO",
        "PROVIDES_INFO_TO":    "NOTIFIED_TO",
        "PUBLISHES_INFO_TO":   "NOTIFIED_TO",
        "TRANSMITS_DATA_TO":   "NOTIFIED_TO",
        "REQUESTS_INFO_FROM":  "NOTIFIED_TO",
        "RECEIVES_INFO_FROM":  "NOTIFIED_TO",
        "RECEIVES":            "RECEIVED_BY",
        "RECEIVED_FROM":       "RECEIVED_BY",
        # --- Permission / Prohibition ---
        "PERMITS":             "PERMITTED_TO",
        "PERMITTED_FOR":       "PERMITTED_TO",
        "PERMITTED_TO_USE":    "PERMITTED_TO",
        "CAN_PERFORM":         "PERMITTED_TO",
        "PROHIBITS":           "PROHIBITED_FROM",
        "PROHIBITED_FOR":      "PROHIBITED_FROM",
        "PROHIBITED_IN":       "PROHIBITED_FROM",
        "PROHIBITED_BY":       "PROHIBITED_FROM",
        "PREVENTED_BY":        "PROHIBITED_FROM",
        "PROHIBITED_FROM_CONTAINING": "PROHIBITED_FROM",
        "MUST_NOT_PERFORM":    "PROHIBITED_FROM",
        "MUST_NOT_MISLEAD":    "PROHIBITED_FROM",
        "MUST_PROTECT":        "PROTECTED_BY",
        "PROTECTS":            "PROTECTED_BY",
        "EXEMPT":              "EXEMPT_FROM",
        "EXCLUDED_FROM":       "EXEMPT_FROM",
        "EXEMPTS":             "EXEMPT_FROM",
        "ENTITLES":            "ENTITLED_TO",
        "PRIORITY_FOR":        "ENTITLED_TO",
        "PRIORITY_OVER":       "ENTITLED_TO",
        "HAS_PRIORITY_ACCESS_TO": "ENTITLED_TO",
        # --- Requirement / Compliance ---
        "REQUIRES":            "REQUIRED_FOR",
        "REQUIRED":            "REQUIRED_FOR",
        "MUST_COMPLY_WITH":    "COMPLIES_WITH",
        "MUST_MEET":           "COMPLIES_WITH",
        "CONFORMS_TO":         "COMPLIES_WITH",
        "CONDITIONED_ON":      "COMPLIES_WITH",
        "ENSURES_COMPLIANCE_WITH": "COMPLIES_WITH",
        "COMPLIES":            "COMPLIES_WITH",
        "ENSURES":             "ENSURED_BY",
        "AFFECTED":            "AFFECTED_BY",
        "AFFECTS":             "AFFECTED_BY",
        "SUBJECTED_TO":        "AFFECTED_BY",
        # --- Definition / Classification ---
        "DEFINES":             "DEFINED_IN",
        "DEFINED_AS":          "DEFINED_IN",
        "DEFINED_BY":          "DEFINED_IN",
        "DEFINED_FOR":         "DEFINED_IN",
        "DEFINED_BY_ABSENCE_OF": "DEFINED_IN",
        "CLASSIFIED_IN":       "CLASSIFIED_AS",
        "CLASSIFIED":          "CLASSIFIED_AS",
        "MARKED_WITH":         "CLASSIFIED_AS",
        "NAMED_AFTER":         "CLASSIFIED_AS",
        "IS_LEGAL_REPRESENTATIVE_OF": "BELONGS_TO",
        "CONTAINED_IN":        "PART_OF",
        "CONTAINS":            "PART_OF",
        "INCLUDED_IN":         "PART_OF",
        "INCLUDES":            "PART_OF",
        "ATTACHED_TO":         "PART_OF",
        "WORKS_FOR":           "BELONGS_TO",
        "WORKS_IN":            "BELONGS_TO",
        "REPRESENTS":          "REPRESENTED_BY",
        # --- Financial ---
        "FUNDS":               "FUNDED_BY",
        "PAYS":                "PAID_TO",
        "PAID_FOR":            "PAID_TO",
        "DEDUCTED_FROM":       "PAID_TO",
        "COMPENSATED_BY":      "PAID_BY",
        "COLLECTS":            "COLLECTED_BY",
        "DEPOSITED_TO":        "PAID_TO",
        "PURCHASED_FROM":      "PAID_BY",
        # --- Financial Synonyms ---
        "FUNDED":              "FUNDED_BY",
        "ALLOCATED_BY":        "FUNDED_BY",
        "ALLOCATED_FOR":       "FUNDED_BY",
        "ALLOCATED_FROM":      "FUNDED_BY",
        "ALLOCATED_TO":        "FUNDED_BY",
        "ALLOCATED_VIA":       "FUNDED_BY",
        "ALLOCATES":           "FUNDED_BY",
        # --- Document synonyms ---
        "REPLACES":            "REPLACED_BY",
        "AMENDS":              "AMENDED_BY",
        "AMENDED":             "AMENDED_BY",
        "REPEALS":             "REPEALED_BY",
        "REMOVED_FROM":        "REPEALED_BY",
        "REFERENCES":          "REFERENCED_BY",
        "REFERS_TO":           "REFERENCED_BY",
        "REFERRED_BY":         "REFERENCED_BY",
        # --- General ---
        "ISSUED_FOR":          "APPLIES_TO",
        "DESIGNED_FOR":        "APPLIES_TO",
        "ORGANIZED_FOR":       "APPLIES_TO",
        "PERMITTED_BY":        "APPLIES_TO",
        "APPLIES":             "APPLIES_TO",
        "RELATED":             "RELATED_TO",
        "CONNECTED_TO":        "RELATED_TO",
        "SYNCHRONIZED_WITH":   "RELATED_TO",
        "IS_EQUAL_TO":         "RELATED_TO",
        "SUPPLEMENTED_BY":     "RELATED_TO",
        "COVERS":              "RELATED_TO",
        "COVERED_BY":          "RELATED_TO",
        "LEASED_BY":           "RELATED_TO",
        "ASSUMES":             "RELATED_TO",
        "SEPARATED_FROM":      "RELATED_TO",
        "STORED_AT":           "LOCATED_IN",
        # --- Other active → canonical ---
        "PRODUCES":            "CREATED_BY",
        "GENERATED_BY":        "CREATED_BY",
        "GENERATES":           "CREATED_BY",
        "REVOKES":             "REPEALED_BY",
        "REVOKED_BY":          "REPEALED_BY",
        "SUSPENDED_BY":        "REPEALED_BY",
        "TERMINATED_BY":       "REPEALED_BY",
        "UPDATED_BY":          "AMENDED_BY",
        "UPDATES":             "AMENDED_BY",
        "MODIFIED_BY":         "AMENDED_BY",
        "MODIFIED_VIA":        "AMENDED_BY",
        "CORRECTED_BY":        "AMENDED_BY",
        "EXTENDED_BY":         "AMENDED_BY",
        "SUPPORTED_BY":        "IMPLEMENTED_BY",
        "SUPPORTS":            "IMPLEMENTED_BY",
        "ASSISTED_BY":         "IMPLEMENTED_BY",
        "MAINTAINED_BY":       "IMPLEMENTED_BY",
        "MAINTAINS":           "IMPLEMENTED_BY",
        "SERVICED_BY":         "IMPLEMENTED_BY",
        "SERVICES":            "IMPLEMENTED_BY",
        "PROCESSED_BY":        "IMPLEMENTED_BY",
        "PROCESSED_IN":        "IMPLEMENTED_BY",
        "TEACHES":             "EDUCATED_BY",
        "TRAINING_PROVIDED_BY": "EDUCATED_BY",
        "TRIGGERS":            "TRIGGERED_BY",
        "USES":                "USED_BY",
        "PROVIDES":            "PROVIDED_BY",
        "PROVIDES_TO":         "PROVIDED_TO",
        "OPENS":               "OPENED_BY",
        "DEVELOPS":            "DEVELOPED_BY",
        "PREPARES":            "PREPARED_BY",
        "PREPARED_WITH":       "PREPARED_BY",
        "AWARDS":              "AWARDED_BY",
        "AWARDED_WITH":        "AWARDED_TO",
        "CALCULATED_FROM":     "CALCULATED_BY",
        "MONITORED_BY":        "MANAGED_BY",
        "INSPECTED_BY":        "MANAGED_BY",
        "ASSESSED_BY":         "MANAGED_BY",
        "EVALUATED_BY":        "MANAGED_BY",
        "VERIFIED_BY":         "MANAGED_BY",
        "REVIEWED_BY":         "MANAGED_BY",
        "REVIEWS":             "MANAGED_BY",
        "ANALYZED_BY":         "MANAGED_BY",
        "AUDITED_BY":          "MANAGED_BY",
        "INVESTIGATED_BY":     "MANAGED_BY",
        "PENALIZED_BY":        "AFFECTED_BY",
        "CONFISCATED_BY":      "AFFECTED_BY",
        "DETAINED_BY":         "AFFECTED_BY",
        "EXPROPRIATED_BY":     "AFFECTED_BY",
        "EXPROPRIATES":        "AFFECTED_BY",
        "PROPOSES":            "PROPOSED_BY",
        "ELECTS":              "ELECTED_BY",
    }
    if s in _ALIAS:
        s = _ALIAS[s]

    # 3. Exact match FIXED (sau alias)
    if s in FIXED_NODE_RELATIONS:
        return s

    # 4. Verb-root fuzzy match → canonical FIXED
    first_word = s.split('_')[0]
    for i in range(len(first_word), 3, -1):
        prefix = first_word[:i]
        if prefix in _VERB_ROOT_CANONICAL:
            return _VERB_ROOT_CANONICAL[prefix]

    # 5. Fallback nếu không khớp bất cứ quy tắc nào -> GIỮ NGUYÊN (bảo tồn)
    return s

_CROSS_VERB_MAPPING = {
    # Cập nhật / Sửa đổi -> AMENDED_BY
    "UPDATED_BY": "AMENDED_BY", "UPDATES": "AMENDED_BY", "UPDATED_EVERY": "AMENDED_BY",
    "MODIFIED_BY": "AMENDED_BY", "MODIFIED_VIA": "AMENDED_BY",
    "CORRECTED_BY": "AMENDED_BY", "EXTENDED_BY": "AMENDED_BY", "EXTENDS": "AMENDED_BY",

    # Giám sát / Kiểm tra -> REVIEWED_BY
    "MONITORED_BY": "REVIEWED_BY", "MONITORS": "REVIEWED_BY",
    "INSPECTED_BY": "REVIEWED_BY", "INSPECTS": "REVIEWED_BY",
    "INVESTIGATED_BY": "REVIEWED_BY", "INVESTIGATES": "REVIEWED_BY",
    "VERIFIED_BY": "REVIEWED_BY",

    # Đánh giá -> ASSESSED_BY
    "EVALUATED_BY": "ASSESSED_BY", "EVALUATES": "ASSESSED_BY", "TESTED_BY": "ASSESSED_BY",

    # Phân công -> ALLOCATED_TO / ALLOCATED_BY
    "ASSIGNED_TO": "ALLOCATED_TO", "ASSIGNED_BY": "ALLOCATED_BY",

    # Cung cấp / Phân phối -> PROVIDED_TO / PROVIDED_BY
    "DELIVERED_TO": "PROVIDED_TO", "DELIVERS_TO": "PROVIDED_TO",
    "TRANSMITTED_TO": "PROVIDED_TO", "TRANSMITS_DATA_TO": "PROVIDED_TO",
    "TRANSMITTED_BY": "PROVIDED_BY", "TRANSMITTED_VIA": "PROVIDED_BY",
    "PROVIDED_WITH": "PROVIDED_BY",

    # Tạo ra / Sản xuất -> CREATED_BY
    "PRODUCED_BY": "CREATED_BY", "PRODUCES": "CREATED_BY",
    "PRODUCED_IN": "CREATED_BY", "PRODUCED_WITH": "CREATED_BY",
    "GENERATES": "CREATED_BY", "CONSTRUCTED_BY": "CREATED_BY",
    
    # Yêu cầu -> REQUESTED_BY
    "REQUESTS_ACTION_FROM": "REQUESTED_BY", "REQUESTS_INFO_FROM": "REQUESTED_BY",
    "REQUESTED_FROM": "REQUESTED_BY",

    # Hỗ trợ -> SUPPORTED_BY
    "ASSISTED_BY": "SUPPORTED_BY", "ASSISTS": "SUPPORTED_BY",
    
    # Tuân thủ -> COMPLIES_WITH
    "CONFORMS_TO": "COMPLIES_WITH", "MUST_MEET": "COMPLIES_WITH",

    # Dọn rác đuôi dài -> RELATED_TO hoặc nhãn chuẩn hơn
    "IS_EQUAL_TO": "RELATED_TO", "NOT_EQUAL_TO": "RELATED_TO",
    "SELF_REFERENCE": "RELATED_TO", "LACKS": "RELATED_TO",
    "LACKS_SKILL": "RELATED_TO", "LACKS_KNOWLEDGE": "RELATED_TO",
    "MUST_NOT_MISLEAD": "RELATED_TO", "PROVES": "RELATED_TO",
    "PERFORMS_IF_DISAGREES": "IMPLEMENTED_BY",
    "USES_FOR": "USED_FOR", "USED_WITH": "USED_IN",
}


def _normalize_doc_relation(raw_rel: str) -> str:
    if not raw_rel:
        return "BASED_ON"
    s = raw_rel.strip().upper().replace(" ", "_")

    alias_map = {
        "REFERENCES": "BASED_ON",
        "APPLIES": "APPLIES_TO",
    }
    if s in alias_map:
        s = alias_map[s]

    if s in FIXED_DOC_RELATIONS or s in DYNAMIC_DOC_RELATIONS:
        return s

    if _RE_SCREAMING.match(s):
        DYNAMIC_DOC_RELATIONS.add(s)
        return s

    return "BASED_ON"


# ==========================================
# HELPERS: build prompt + parse response
# ==========================================

def build_unified_prompt(batch_info: List[Dict[str, str]]) -> str:
    """
    Tạo nội dung prompt unified từ danh sách {s_doc, context}.
    Trả về chuỗi prompt đầy đủ để gửi cho LLM.
    """
    ctx_text = ""
    for j, item in enumerate(batch_info):
        ctx_text += f"\n--- ĐOẠN {j+1} (VB: {item['s_doc']}) ---\n{item['context']}\n"
        
    allowed_entities = " | ".join(sorted(FIXED_ENTITY_TYPES | DYNAMIC_ENTITY_TYPES))
    # Chỉ gửi FIXED vào prompt — KHÔNG gửi DYNAMIC để tránh feedback loop
    allowed_relations = " | ".join(sorted(FIXED_NODE_RELATIONS))
    allowed_doc_relations = " | ".join(sorted(FIXED_DOC_RELATIONS))

    return LEGAL_UNIFIED_EXTRACTOR_PROMPT.format(
        contexts=ctx_text,
        allowed_entity_types=allowed_entities,
        allowed_node_relations=allowed_relations,
        allowed_doc_relations=allowed_doc_relations
    )


def parse_unified_response(resp_text: str) -> Dict[str, Any]:
    """
    Parse JSON response từ unified LLM call.
    Trả về dict: {doc_relations: [...], entities: {...}, node_relations: [...]}
    
    Post-processing:
    - entity_type được normalize về FIXED_ENTITY_TYPES (dedup sau normalize)
    - relationship được normalize về FIXED_NODE_RELATIONS
    Đảm bảo luôn trả về cấu trúc hợp lệ dù LLM fail.
    """
    empty = {
        "doc_relations": [],
        "entities": dict(_EMPTY_ENTITIES),
        "node_relations": [],
    }
    if not resp_text:
        return empty

    try:
        json_str = extract_json_from_text(resp_text)
        if not json_str:
            json_str = resp_text.strip()
        # Sửa trailing commas phổ biến
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
        data = json.loads(json_str)
    except Exception as e:
        print(f"[Unified Parser] JSON parse error: {e}")
        try:
            import os
            os.makedirs(".debug", exist_ok=True)
            with open(".debug/llm_failures.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- [Unified Parser Error: {e}] ---\n")
                f.write(f"Raw Response:\n{resp_text}\n")
                f.write("-" * 50 + "\n")
        except:
            pass
        return empty

    # --- doc_relations: chuẩn hoá edge_label ---
    doc_rels = data.get("doc_relations", [])
    for r in doc_rels:
        r["edge_label"] = _normalize_doc_relation(r.get("edge_label", ""))

    # --- entities: normalize type → FIXED_ENTITY_TYPES, merge duplicates ---
    raw_ents = data.get("entities", {})
    entities: Dict[str, List[str]] = {}
    if isinstance(raw_ents, dict):
        for k, val in raw_ents.items():
            norm_type = _normalize_entity_type(str(k))
            
            # LỌC BỎ NHÃN CẤU TRÚC (STRUCTURAL LABELS)
            # Không cho phép trích xuất tự do các nhãn này vì chúng được tạo bằng logic tĩnh.
            if norm_type in ["Document", "LegalArticle", "Article", "Clause", "Chunk"]:
                continue
                
            if isinstance(val, list):
                vals = [str(v).strip() for v in val if v and str(v).strip()]
            elif isinstance(val, str):
                vals = [val.strip()] if val.strip() else []
            else:
                continue
            if norm_type not in entities:
                entities[norm_type] = []
            # Dedup exact sau normalize
            existing_lower = {e.lower() for e in entities[norm_type]}
            for v in vals:
                norm_v = _normalize_entity_name(v, norm_type)
                if norm_v and norm_v.lower() not in existing_lower:
                    entities[norm_type].append(norm_v)
                    existing_lower.add(norm_v.lower())

    # Substring containment dedup + thu thập alias_map để redirect relations
    entity_alias_map: dict[str, str] = {}  # {removed_lower → canonical}
    for etype in entities:
        deduped, aliases = _dedup_entity_values(entities[etype])
        entities[etype] = deduped
        entity_alias_map.update(aliases)

    # --- node_relations: normalize relationship + source/target_type ---
    raw_node_rels = data.get("node_relations", [])
    node_rels = []
    for nr in raw_node_rels:
        nr["relationship"] = _normalize_relationship(nr.get("relationship", ""))
        nr["source_type"] = _normalize_entity_type(nr.get("source_type", ""))
        # target_type có thể là "Document" hoặc EntityType
        raw_tgt_type = str(nr.get("target_type", "")).strip()
        if raw_tgt_type.lower() == "document":
            nr["target_type"] = "Document"
        else:
            nr["target_type"] = _normalize_entity_type(raw_tgt_type)

        norm_src = _normalize_entity_name(nr.get("source_node", ""), nr["source_type"])
        norm_tgt = _normalize_entity_name(nr.get("target_node", ""), nr["target_type"])

        # BỘ LỌC: Bỏ qua toàn bộ quan hệ nếu 1 trong 2 đầu mút là rác (bị chặn)
        if not norm_src or not norm_tgt:
            continue

        # REDIRECT: Nếu entity bị dedup (VD: "Windows" → "Hệ điều hành Windows")
        # thì relation vẫn được giữ nhưng endpoint được đổi sang canonical
        norm_src = entity_alias_map.get(norm_src.lower(), norm_src)
        norm_tgt = entity_alias_map.get(norm_tgt.lower(), norm_tgt)

        nr["source_node"] = norm_src
        nr["target_node"] = norm_tgt
        node_rels.append(nr)

    return {
        "doc_relations": doc_rels,
        "entities": entities,
        "node_relations": node_rels,
    }
