"""
Centralized Configuration for Legal-RAG
========================================
Chứa toàn bộ các thiết lập tĩnh, danh sách hằng số và từ điển chuẩn hoá
cho toàn bộ pipeline Ingestion, LLM và Agent.
"""

# =====================================================================
# 1. LLM CONFIGURATION
# =====================================================================
_ICLLM_CONFIG = {
    "AppCode": "LEGAL_RAG",
    "FunctionCode": "standard_chat",
    "ModelLLM": "llama3.1-8b", 
    "UrlPrompt": "https://staging.pontusinc.com/api/chatbot/v1/prompt/list",
    "LLMName": "legal_rag_chat",
    "UrlLLMApi": "http://10.9.3.75:30031/api/llama3/8b", # Hoặc endpoint mới như 10.9.3.241:5564/api/Qas/v2
    "BaseDirLog": "logs/llm_logs",
    "BaseDirPostProcess": "logs/llm_logs/post_process",
    "BaseDirPrompt": "logs/llm_logs/prompt",
    "IsLog": True,
    "IsShowConsole": False, # Có thể để True nếu muốn in chi tiết ICLLM ra terminal
    "IsGetPromptOnline": False, # Đặt False để chạy prompt dưới ổ cứng, hãy đảm bảo thư mục BaseDirPrompt có template phù hợp
}

_JSON_ENFORCEMENT_PROMPT = (
    "\nQUAN TRỌNG: Bạn BẮT BUỘC phải trả lời bằng định dạng JSON hợp lệ. "
    "Không kèm theo bất kỳ văn bản giải thích nào ngoài khối JSON."
)


# =====================================================================
# 2. NEO4J / GRAPH DATABASE CONFIGURATION
# =====================================================================
_ENTITY_LABELS = [
    "Organization", "Subject", "LegalConcept", "Procedure", "Location",
    "Right_Obligation", "Sanction", "Timeframe", "Form", "Fee"
]

CHUNKING_RELATIONS = {
    "HAS_ENTITY", "HAS_ARTICLE", "HAS_TYPE", "HAS_SECTOR", 
    "PART_OF", "BELONGS_TO", "ISSUED_BY", "SIGNED_BY", "BASED_ON"
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

BLACKLIST_RELATIONS = {
    "IS", "HAS", "DEADLINE", "EFFECTIVE_FROM", "HAS_MINIMUM_SIZE",
    "PUBLISHED_ON", "STARTS_AT", "ENDS_AT", "IS_EQUAL_TO", "NOT_EQUAL_TO",
    "MUST_NOT_BE_HIGHER_THAN", "MUST_NOT_BE_LOWER_THAN",
    "OCCURS_AT", "OCCURS_EVERY", "EXPIRES_ON", "MEETS_EVERY",
    "UPDATED_EVERY", "EXECUTED_AT", "EXECUTED_ON",
}

FIXED_DOC_RELATIONS = {
    "BASED_ON", "AMENDS", "REPEALS", "REPLACES",
    "GUIDES", "APPLIES_TO", "ISSUED_WITH", "ASSIGNS", "CORRECTS",
    "AMENDED_BY", "REPEALED_BY", "REPLACED_BY", "GUIDED_BY",
    "REFERENCED_BY",
}

FIXED_ENTITY_TYPES = {
    "Organization", "Person", "Location", "Document", "LegalArticle",
    "Procedure", "Condition", "Fee", "Penalty", "Timeframe", "Role", "Concept",
}

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
}

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

# =====================================================================
# 3. DYNAMIC INGESTION STATE (In-Memory)
# =====================================================================
# Tập động: ghi nhận nhãn mới LLM tạo trong quá trình pipeline
# → các lần normalize sau sẽ khoanh ngắ luôn mà không force fallback
DYNAMIC_ENTITY_TYPES: set = set()
DYNAMIC_NODE_RELATIONS: set = set()
DYNAMIC_DOC_RELATIONS: set = set()

# Cấu trúc rỗng ban đầu (nếu cần fallback)
_EMPTY_ENTITIES: dict = {}

# =====================================================================
# 4. AGENT / UI LOCALIZATION CONFIG
# =====================================================================
_VI_TRANSLATION_MAP = {
    # Node Labels
    "Chunk": "Đoạn trích",
    "Clause": "Khoản",
    "Concept": "Khái niệm",
    "Condition": "Điều kiện",
    "Distance": "Khoảng cách",
    "Document": "Văn bản",
    "Equipment": "Thiết bị",
    "Event": "Sự kiện",
    "Fee": "Phí",
    "LegalArticle": "Điều luật",
    "LegalType": "Loại văn bản",
    "Location": "Địa điểm",
    "Network": "Mạng lưới",
    "Organization": "Tổ chức",
    "Penalty": "Hình phạt",
    "Person": "Cá nhân",
    "Procedure": "Thủ tục",
    "ProductGroup": "Nhóm sản phẩm",
    "Project": "Dự án",
    "Role": "Vai trò",
    "Sector": "Lĩnh vực",
    "Service": "Dịch vụ",
    "Software": "Phần mềm",
    "Standard": "Tiêu chuẩn",
    "Timeframe": "Thời gian",
    "Entity": "Thực thể",
    
    # Relationships
    "ACCEPTED_BY": "Được chấp nhận bởi",
    "ACKNOWLEDGED_BY": "Được ghi nhận bởi",
    "AFFECTED_BY": "Bị tác động bởi",
    "AMENDED_BY": "Bị sửa đổi bởi",
    "APPLIED_BY": "Được áp dụng bởi",
    "APPLIES_TO": "Áp dụng cho",
    "APPOINTED_BY": "Được bổ nhiệm bởi",
    "APPROVED_BY": "Được phê duyệt bởi",
    "ATTENDED_BY": "Được tham dự bởi",
    "AWARDED_BY": "Được trao bởi",
    "AWARDED_TO": "Trao cho",
    "BASED_ON": "Căn cứ vào",
    "BELONGS_TO": "Thuộc về",
    "CALCULATED_BY": "Được tính bởi",
    "CERTIFIED_BY": "Được chứng nhận bởi",
    "CHALLENGED_BY": "Bị phản đối bởi",
    "CLASSIFIED_AS": "Được phân loại là",
    "COLLECTED_BY": "Được thu thập bởi",
    "COMPLIES_WITH": "Tuân thủ",
    "CONFIRMED_BY": "Được xác nhận bởi",
    "CONSULTED_BY": "Được tham vấn bởi",
    "CONTRIBUTES_TO": "Đóng góp cho",
    "COOPERATES_WITH": "Hợp tác với",
    "COORDINATED_BY": "Được phối hợp bởi",
    "CREATED_BY": "Được tạo bởi",
    "DECIDED_BY": "Được quyết định bởi",
    "DEFINED_IN": "Được quy định tại",
    "DELEGATED_TO": "Được ủy quyền cho",
    "DESIGNATED_BY": "Được chỉ định bởi",
    "DESTROYED_BY": "Bị tiêu hủy bởi",
    "DETERMINED_BY": "Được xác định bởi",
    "DEVELOPED_BY": "Được phát triển bởi",
    "EDUCATED_BY": "Được đào tạo bởi",
    "ELECTED_BY": "Được bầu bởi",
    "EMPLOYED_BY": "Được thuê bởi",
    "ENFORCED_BY": "Được thi hành bởi",
    "ENSURED_BY": "Được đảm bảo bởi",
    "ENTITLED_TO": "Có quyền",
    "ESTABLISHED_BY": "Được thành lập bởi",
    "EXECUTED_BY": "Được thực thi bởi",
    "EXEMPT_FROM": "Được miễn trừ",
    "EXPORTED_TO": "Được xuất khẩu sang",
    "FACILITATED_BY": "Được tạo điều kiện bởi",
    "FORMATTED_WITH": "Được định dạng với",
    "FUNDED_BY": "Được tài trợ bởi",
    "GUIDED_BY": "Được hướng dẫn bởi",
    "HANDLED_BY": "Được xử lý bởi",
    "IMPLEMENTED_BY": "Được thực hiện bởi",
    "ISSUED_BY": "Được ban hành bởi",
    "MANAGED_BY": "Được quản lý bởi",
    "RELATED_TO": "Liên quan đến",
    "REPEALED_BY": "Bị bãi bỏ bởi",
    "REPLACED_BY": "Bị thay thế bởi",
    "REPORTED_TO": "Báo cáo cho",
    "REQUIRED_BY": "Được yêu cầu bởi",
    "SIGNED_BY": "Được ký bởi",
    "USED_BY": "Được sử dụng bởi",
    "HAS_ENTITY": "Có thực thể",
    "HAS_ARTICLE": "Có điều khoản",
    "HAS_COMPONENT": "Có thành phần",
    "HAS_TYPE": "Thuộc loại",
    "HAS_SECTOR": "Thuộc lĩnh vực",
    "HAS_PART": "Bao gồm phần",
    "PART_OF": "Là một phần của",
    "AMENDS": "Sửa đổi",
    "REPLACES": "Thay thế",
    "REPEALS": "Bãi bỏ",
    "GUIDES": "Hướng dẫn",
    "APPLIES": "Áp dụng",
    "ISSUED_WITH": "Ban hành kèm theo",
    "ASSIGNS": "Phân công",
    "CORRECTS": "Đính chính",
    "LOCATED_IN": "Nằm tại",
    "PAID_TO": "Trả cho",
    "REPRESENTED_BY": "Đại diện bởi",
    "PROTECTED_BY": "Bảo vệ bởi",
    "PROVIDED_BY": "Được cung cấp bởi",
    "PROVIDED_TO": "Cung cấp cho",
}
