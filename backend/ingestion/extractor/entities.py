"""
entities.py — Trích xuất Thực thể + Quan hệ Node từ văn bản pháp luật.

Exports:
  FIXED_ENTITY_TYPES       — danh sách 12 loại entity cố định
  FIXED_NODE_RELATIONS     — danh sách 16 loại node relation cố định
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
    "ISSUED_BY", "SIGNED_BY", "AFFECTED_BY", "REGULATED_BY",
    "APPLIED_BY", "MANAGED_BY", "ASSIGNED_BY",
    "REQUIRED_FOR", "DEFINED_IN", "LOCATED_IN", "CLASSIFIED_AS", "PART_OF",
    "REPLACED_BY", "RELATED_TO",
}

# Quan hệ LLM hay sinh nhầm — bị loại bỏ hoàn toàn, fallback sang RELATED_TO
BLACKLIST_RELATIONS = {
    "IS", "DEADLINE", "EFFECTIVE_FROM", "HAS_MINIMUM_SIZE",
    "PUBLISHED_ON", "STARTS_AT", "ENDS_AT",
}

FIXED_DOC_RELATIONS = {
    "BASED_ON", "AMENDS", "REPEALS", "REPLACES",
    "GUIDES", "APPLIES_TO", "ISSUED_WITH", "ASSIGNS", "CORRECTS",
    "AMENDED_BY", "REPEALED_BY", "REPLACED_BY", "GUIDED_BY"
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
    ƯU TIÊN: FIXED → DYNAMIC → chấp nhận nhãn mới hợp lệ (SCREAMING_SNAKE_CASE)
    Dự phòng cuối: 'RELATED_TO'.
    """
    if not raw_rel:
        return "RELATED_TO"
    s = raw_rel.strip().upper().replace(" ", "_")

    # 0. Alias Mapper (Gộp các quan hệ đồng nghĩa / sai format)
    # Lưu ý: chỉ alias các nhãn VIẼT chữ́ động sang bị động tương ưứng (không đảo ngược)
    alias_map = {
        "ISSUED":              "ISSUED_BY",
        "SIGNED":              "SIGNED_BY",
        "APPLIES":             "APPLIES_TO",
        "SUPERVISED_BY":       "MANAGED_BY",
        "ACCOUNTABLE_TO":      "MANAGED_BY",
        "RESPONSIBLE_FOR":     "MANAGED_BY",
        "IMPLEMENTED_BY":      "MANAGED_BY",
        "ASSIGNS":             "ASSIGNED_BY",
        "OPINION_SEEKED_FROM": "OPINION_SOUGHT_FROM",
        "SUBMITTED_TO_FOR_OPINION": "SUBMITTED_FOR_OPINION",
    }
    if s in alias_map:
        s = alias_map[s]

    # Blacklist: quan hệ nhiễu/property giả — fallback về RELATED_TO
    if s in BLACKLIST_RELATIONS:
        return "RELATED_TO"

    # 1. Exact match
    if s in FIXED_NODE_RELATIONS or s in DYNAMIC_NODE_RELATIONS:
        return s

    # 2. nhãn mới: chấp nhận nếu format SCREAMING_SNAKE_CASE hợp lệ
    if _RE_SCREAMING.match(s):
        DYNAMIC_NODE_RELATIONS.add(s)
        return s

    # 3. Fallback
    return "RELATED_TO"


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
    allowed_relations = " | ".join(sorted(FIXED_NODE_RELATIONS | DYNAMIC_NODE_RELATIONS))
    allowed_doc_relations = " | ".join(sorted(FIXED_DOC_RELATIONS | DYNAMIC_DOC_RELATIONS))
    
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
            # Dedup sau normalize
            existing = set(entities[norm_type])
            for v in vals:
                norm_v = _normalize_entity_name(v, norm_type)
                if norm_v and norm_v not in existing:
                    entities[norm_type].append(norm_v)
                    existing.add(norm_v)

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
            
        nr["source_node"] = norm_src
        nr["target_node"] = norm_tgt
        node_rels.append(nr)

    return {
        "doc_relations": doc_rels,
        "entities": entities,
        "node_relations": node_rels,
    }
