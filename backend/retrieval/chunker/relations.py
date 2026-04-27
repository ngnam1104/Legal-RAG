import re
import json
import time
import random
import pandas as pd
from typing import Dict, List
from backend.retrieval.chunker import metadata as md
from backend.llm.factory import get_client
from backend.utils.text_utils import strip_thinking_tags

llm_client = get_client()

# ==========================================
# BƯỚC HẬU XỬ LÝ: CHUẨN HÓA THỰC THỂ (ENTITY NORMALIZATION)
# ==========================================
REGEX_DOC_NUM = r"\b\d+[\/\-](?:20\d{2}|19\d{2})[\/\-][A-ZĐ][A-Z0-9Đa-zđ\-\/]*\b|\b\d+[\/\-][A-ZĐ][A-Z0-9Đa-zđ\-\/]*\b"

# THÊM: Hằng số cấu hình LLM
MAX_INPUT_LENGTH_CONFIG = 20000

OMNIBUS_LAWS_MAPPER = {
    "37 luật có liên quan đến quy hoạch": "35/2018/QH14",
    "luật chứng khoán, luật kế toán, luật kiểm toán độc lập": "56/2024/QH15",
    "hiến pháp nước cộng hòa xã hội chủ nghĩa việt nam": "Hiến pháp 2013",
    "bộ luật dân sự": "91/2015/QH13",
    "bộ luật hình sự": "100/2015/QH13",
    "bộ luật lao động": "45/2019/QH14",
    "bộ luật tố tụng hình sự": "101/2015/QH13",
    "bộ luật tố tụng dân sự": "92/2015/QH13",
    "luật bầu cử đại biểu quốc hội và đại biểu hội đồng nhân dân": "85/2015/QH13",
    "luật tổ chức chính quyền địa phương": "77/2015/QH13",
    "luật tổ chức chính phủ": "76/2015/QH13",
    "luật ban hành văn bản quy phạm pháp luật": "80/2015/QH13",
    "luật thi đua, khen thưởng": "06/2022/QH15",
    "luật ngân sách nhà nước": "83/2015/QH13",
    "luật quản lý thuế": "38/2019/QH14",
    "luật thuế giá trị gia tăng": "48/2024/QH15",
    "luật thuế thu nhập cá nhân": "04/2007/QH12",
    "luật thuế xuất khẩu, thuế nhập khẩu": "107/2016/QH13",
    "luật kế toán": "88/2015/QH13",
    "luật kiểm toán độc lập": "67/2011/QH12",
    "luật quản lý, sử dụng tài sản công": "15/2017/QH14",
    "luật dự trữ quốc gia": "22/2012/QH13",
    "luật hải quan": "54/2014/QH13",
    "luật đầu tư theo phương thức đối tác công tư": "64/2020/QH14",
    "luật đầu tư công": "39/2019/QH14",
    "luật đầu tư": "61/2020/QH14",
    "luật doanh nghiệp": "59/2020/QH14",
    "luật đấu thầu": "22/2023/QH15",
    "luật chứng khoán": "54/2019/QH14",
    "luật các tổ chức tín dụng": "32/2024/QH15",
    "luật đất đai": "31/2024/QH15",
    "luật nhà ở": "27/2023/QH15",
    "luật kinh doanh bất động sản": "29/2023/QH15",
    "luật bảo hiểm xã hội": "41/2024/QH15",
    "luật giáo dục": "43/2019/QH14",
    "luật khám bệnh, chữa bệnh": "15/2023/QH15",
    "luật xử lý vi phạm hành chính": "15/2012/QH13",
    "luật xuất bản": "19/2012/QH13",
    "luật báo chí": "103/2016/QH13",
    "luật an toàn, vệ sinh lao động": "84/2015/QH13",
    "luật dược": "105/2016/QH13",
    "luật phòng, chống bệnh truyền nhiễm": "03/2007/QH12",
    "luật bảo hiểm y tế": "25/2008/QH12",
    "luật hiến, lấy, ghép mô, bộ phận cơ thể người và hiến, lấy xác": "73/2006/QH11",
}

SORTED_MAPPER = dict(sorted(OMNIBUS_LAWS_MAPPER.items(), key=lambda item: len(item[0]), reverse=True))

_law_kws = "|".join([re.escape(k) for k in SORTED_MAPPER.keys()])
REGEX_LAW_NAME = rf"\b(?:{_law_kws})(?:\s+năm\s+\d{{4}})?|\b(?:Hiến pháp|Bộ luật|Luật)[\w\s\,\.\-\u00C0-\u1EF9]*?(?:\s+năm\s+\d{{4}})|\bHiến pháp(?:\s+năm\s+\d{{4}})?"
REGEX_DOC_NUM_STRICT = re.compile(
    r"\b(\d+[\-\/](?:20\d{2}|19\d{2})[\-\/][A-Z\u0110][A-Z0-9\u0110a-z\u0111\-\/]*)\b"
)

# ==========================================
# BỘ TỪ KHOÁ PHÂN VÙNG (Zone-aware Label Keywords)
# ==========================================
# Vùng CĂN CỨ (Preamble): chỉ BASED_ON — passive chain sẽ xử lý AMENDS/REPLACES/REPEALS
_LABEL_KEYWORDS_PREAMBLE = [
    ("GUIDES",     ["hướng dẫn thi hành", "quy định chi tiết", "hướng dẫn thực hiện"]),
    ("APPLIES",    ["áp dụng", "thực hiện theo", "theo quy định tại", "dẫn chiếu", "chiếu theo"]),
    ("ISSUED_WITH",["ban hành kèm theo", "ban hành kèm", "đính kèm", "kèm theo"]),
    # Mọi entity còn lại → BASED_ON (mặc định)
]
# Vùng ĐIỀU KHOẢN (Articles): full bộ chủ động
_LABEL_KEYWORDS_ARTICLE = [
    ("REPEALS",    ["bãi bỏ", "hết hiệu lực", "hủy bỏ", "đình chỉ", "chấm dứt hiệu lực",
                    "không còn hiệu lực", "bãi bỏ quy định tại", "ngưng hiệu lực",
                    "hết hiệu lực thi hành", "hủy bỏ một phần"]),
    ("REPLACES",   ["thay thế", "thay thế cho", "thay thế văn bản", "thay cho", "thay thế quy định tại"]),
    ("AMENDS",     ["sửa đổi", "bổ sung", "đính chính", "hiệu chỉnh", "sửa đổi tại",
                    "bổ sung tại", "điều chỉnh", "thay đổi một số", "cập nhật", "sửa đổi nội dung"]),
    ("GUIDES",     ["hướng dẫn thi hành", "quy định chi tiết", "hướng dẫn thực hiện",
                    "giải thích", "hướng dẫn", "chi tiết tại", "triển khai",
                    "về việc hướng dẫn", "phổ biến hướng dẫn"]),
    ("APPLIES",    ["áp dụng", "thực hiện theo", "áp dụng theo", "tuân thủ theo",
                    "đối chiếu theo", "dẫn chiếu", "chiếu theo", "theo nội dung tại", "theo quy định tại"]),
    ("ISSUED_WITH",["ban hành kèm theo", "ban hành kèm", "đính kèm", "kèm theo",
                    "tài liệu kèm theo", "văn bản kèm theo", "phụ lục kèm theo", "phụ lục"]),
    ("ASSIGNS",    ["giao cho", "giao trách nhiệm", "phân công", "giao nhiệm vụ",
                    "ủy quyền", "trách nhiệm của", "giao đơn vị", "ủy thác", "giao"]),
    ("CORRECTS",   ["đính chính lỗi", "sửa lỗi", "đính chính sai sót",
                    "khắc phục sai sót", "đính chính văn bản"]),
]

# Pattern bị động giữa 2 thực thể trong vùng Căn cứ:
# [A] đã được sửa đổi, bổ sung ... theo [B]
PASSIVE_BETWEEN = re.compile(
    r"(?:đã\s+)?được\s+"
    r"(?:sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|đính\s+chính|hết\s+hiệu\s+lực|hủy\s+bỏ|đình\s+chỉ|ngưng\s+hiệu\s+lực|hiệu\s+chỉnh|điều\s+chỉnh|cập\s+nhật)"
    r"(?:[^,\.]{0,100}?,\s*(?:sửa\s+đổi|bổ\s+sung|đính\s+chính|thay\s+thế|bãi\s+bỏ))?"
    r"[^\n\.]{0,150}?(?:theo|bởi|tại)\s+",
    re.IGNORECASE
)
# Regex nhận diện vùng Căn cứ (preamble)
_PREAMBLE_DETECT = re.compile(
    r'^\s*(?:căn\s+cứ|-\s*căn\s+cứ|theo\s+(?:quy\s+định|điều|khoản))',
    re.IGNORECASE
)


def detect_passive_chains(para: str, norm_s_doc: str, already_captured: set):
    """
    Phát hiện mẫu bị động trong từng dòng của vùng Căn cứ:
      Căn cứ [mã VB A / Luật A] đã được sửa đổi, bổ sung ... theo [mã VB B / Luật B]
    Quét từng dòng bắt đầu bằng "Căn cứ" để tránh bỏ sót khi block đầu
    bắt đầu bằng header (QUỐC HỘI, CHÍNH PHỦ...) thay vì "Căn cứ".
    Trả về:
      direct_rels: [(source→BASED_ON→A)]
      cross_rels:  [(B→[REL]→A)] với is_cross_doc=True
    """
    direct_rels, cross_rels = [], []

    # Gom toàn bộ xuống 1 dòng dài (bỏ xuống dòng) để không bị đứt gãy câu
    para_clean = para.replace("\n", " ")
    
    # Ở Việt Nam vùng Căn cứ phân tách với nhau bằng dấu chấm phẩy ;
    # Cắt theo dấu chấm phẩy để xét từng mệnh đề Căn cứ riêng biệt
    clauses = para_clean.split(";")

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # Thu thập tất cả thực thể (số hiệu VB hoặc tên luật) trong mệnh đề này
        entities = []  # (start, end, normalized_id)
        for m in REGEX_DOC_NUM_STRICT.finditer(clause):
            tgt = normalize_entity(m.group(1))
            if tgt and tgt != "UNKNOWN" and tgt != norm_s_doc and is_vd(tgt):
                entities.append((m.start(), m.end(), tgt))
        for m in re.finditer(REGEX_LAW_NAME, clause, re.IGNORECASE):
            raw = m.group(0).lower().strip()
            core_name = re.sub(r"\s+năm\s+\d{4}$", "", raw).strip()
            tgt = OMNIBUS_LAWS_MAPPER.get(core_name) or OMNIBUS_LAWS_MAPPER.get(raw)
            if tgt and tgt != norm_s_doc and tgt != "UNKNOWN":
                overlap = any(abs(e[0] - m.start()) < 5 for e in entities)
                if not overlap:
                    entities.append((m.start(), m.end(), tgt))
        entities.sort(key=lambda x: x[0])

        if len(entities) < 2:
            continue  # Cần ít nhất 2 entity trong cùng 1 mệnh đề

        for i in range(len(entities) - 1):
            start_A, end_A, tgt_A = entities[i]
            start_B, end_B, tgt_B = entities[i + 1]

            between = clause[end_A:start_B]
            if not PASSIVE_BETWEEN.search(between):
                continue

            # Context: lấy nguyên mệnh đề
            focused = clause.replace("\n", " ").strip()

            # Xác định loại quan hệ dựa trên từ khóa giữa A và B
            lower_between = between.lower()
            if any(w in lower_between for w in ["thay thế", "thay cho"]):
                edge_lbl = "REPLACES"
                rel_phrase = "thay thế (passive chain)"
            elif any(w in lower_between for w in ["bãi bỏ", "hết hiệu lực", "hủy bỏ", "đình chỉ",
                                                   "ngưng hiệu lực", "chấm dứt hiệu lực", "không còn hiệu lực"]):
                edge_lbl = "REPEALS"
                rel_phrase = "bãi bỏ (passive chain)"
            elif any(w in lower_between for w in ["đính chính", "sửa lỗi", "khắc phục sai sót"]):
                edge_lbl = "CORRECTS"
                rel_phrase = "đính chính (passive chain)"
            else:
                edge_lbl = "AMENDS"
                rel_phrase = "sửa đổi bổ sung (passive chain)"

            # direct: X → BASED_ON → A
            if (norm_s_doc, tgt_A) not in already_captured and (tgt_A, norm_s_doc) not in already_captured:
                direct_rels.append({
                    "source_doc": norm_s_doc,
                    "target_doc": tgt_A,
                    "relation_phrase": "căn cứ (passive chain)",
                    "edge_label": "BASED_ON",
                    "context": focused,
                    "target_article": "", "target_clause": "", "target_text": ""
                })
                already_captured.add((norm_s_doc, tgt_A))
                # print(f"\n[DEBUG PASSIVE] (Direct) {norm_s_doc} --BASED_ON--> {tgt_A}\n   ╰─ Line: '{focused[:150]}'")

            # cross: B → [REL] → A  (cross-doc, source_doc = tgt_B)
            if tgt_B != tgt_A and (tgt_B, tgt_A) not in already_captured:
                cross_rels.append({
                    "source_doc": tgt_B,
                    "target_doc": tgt_A,
                    "relation_phrase": rel_phrase,
                    "edge_label": edge_lbl,
                    "is_cross_doc": True,
                    "context": focused,
                    "target_article": "", "target_clause": "", "target_text": ""
                })
                already_captured.add((tgt_B, tgt_A))
                # print(f"\n[DEBUG PASSIVE] (Cross)  {tgt_B} --{edge_lbl}--> {tgt_A}\n   ╰─ Line: '{focused[:150]}'")

    return direct_rels, cross_rels

def normalize_entity(text):
    if not text or pd.isna(text):
        return "UNKNOWN"
    text = str(text).strip()
    text = re.sub(r"^(số|của|tại|theo|quy định tại|căn cứ)\s+", "", text, flags=re.IGNORECASE).strip()
    num_match = re.search(REGEX_DOC_NUM, text)
    if num_match:
        return num_match.group(0).upper()
    law_match = re.search(REGEX_LAW_NAME, text)
    if law_match:
        raw_law_name = law_match.group(0).strip()
        lower_law_name = raw_law_name.lower()
        for key_phrase, short_name in SORTED_MAPPER.items():
            if key_phrase in lower_law_name:
                return short_name
        return raw_law_name.title().replace("Năm", "năm")
    return text

def extract_references_via_llm(texts: List[str]) -> List[dict]:
    if not texts: return []
    prompt = f'''Trích xuất CĂN CỨ pháp lý từ:
{json.dumps(texts, ensure_ascii=False, indent=2)}
CHỈ TRẢ VỀ JSON: {{"references": [{{"basis_line": "...", "doc_type": "...", "doc_number": "...", "doc_year": "...", "doc_title": "..."}}]}}'''
    try:
        resp = llm_client.chat_completion([{"role": "user", "content": prompt}], temperature=0.1, response_format={"type": "json_object"})
        resp = strip_thinking_tags(resp) if resp else ""
        resp = resp.replace("```json", "").replace("```", "").strip()
        from backend.utils.text_utils import extract_json_from_text
        json_str = extract_json_from_text(resp)
        if json_str:
            json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
            return json.loads(json_str).get("references", [])
    except Exception as e:
        pass  # Lỗi parse JSON từ LLM — bỏ qua để không làm gián đoạn pipeline
    return []

def extract_exact_article(content: str, article_name: str) -> str:
    if not content or not article_name: return ""
    parts = article_name.strip().split()
    safe_name = r"\s+".join([re.escape(p) for p in parts])
    start_pattern = re.compile(r"(?im)^\s*" + safe_name + r"\b[\.\:\-]?\s*")
    start_match = start_pattern.search(content)
    if not start_match: return ""
    start_pos = start_match.start()
    remaining = content[start_match.end():]
    end_pattern = re.compile(r"(?im)^\s*(?:Điều\s+\d|Chương\s+[IVXLCDM0-9]|Phần\s+(?:thứ\s+)?[IVXLCDM0-9]|Mục\s+\d+|Phụ\s+lục)")
    end_match = end_pattern.search(remaining)
    if end_match: result = content[start_pos: start_match.end() + end_match.start()]
    else: result = content[start_pos:]
    return result.strip()

def extract_target_content_via_llm(full_text_target: str, target_article: str, target_clause: str = None) -> str:
    truncated_text = extract_exact_article(full_text_target, target_article)
    if not truncated_text: return ""
    if not target_clause: return truncated_text
    prompt = f'''Trích xuất NGUYÊN VĂN nội dung của "{target_clause} {target_article}" từ:\n{truncated_text}\nCHỈ trả về NỘI DUNG VĂN BẢN (Text trơn), TUYỆT ĐỐI không giải thích, không dùng markdown codeblock.'''
    try:
        resp = llm_client.chat_completion([{"role": "user", "content": prompt}], temperature=0.1)
        if resp.strip(): return resp.strip()
    except Exception as e: print(f"⚠️ LLM Error: {e}")
    return truncated_text

def is_vd(ds):
    """Đã vá lỗi bộ lọc để không chặn nhầm văn bản pháp luật hợp lệ"""
    ds = str(ds).strip().lower()
    if len(ds) < 4: return False 
    
    # Ưu tiên 1: Chứa số hiệu văn bản (vd: 15/2020/NĐ-CP) -> Chắc chắn là văn bản
    if re.search(r"\d+[\/\-][a-zđ]+", ds):
        return True
        
    # Ưu tiên 2: Chứa từ khóa định danh pháp luật
    legal_keywords = ["luật", "nghị định", "thông tư", "hiến pháp", "quyết định", "chỉ thị", "bộ luật"]
    if any(kw in ds for kw in legal_keywords):
        return True
        
    # Blocklist chỉ kích hoạt khi chuỗi CHỈ chứa tên tổ chức
    bl_orgs = ["bộ ", "sở ", "tổng công ty", "tập đoàn", "chính phủ", "doanh nghiệp", "ủy ban", "thanh tra"]
    return not any(ds.startswith(o) for o in bl_orgs)

def extract_ontology_relationships_batch(docs_list: List[Dict], global_doc_lookup: dict = None, skip_llm: bool = False) -> Dict[str, List[dict]]:
    if not docs_list: return {}
    
    # Uses the global strict REGEX_LAW_NAME to prevent garbage extraction from long text sentences
    REGEX_PATTERN = re.compile(f"({REGEX_DOC_NUM}|{REGEX_LAW_NAME})", re.IGNORECASE)
    
    # PHA 1: TRÍCH XUẤT QUAN HỆ
    # =========================================================================
    BATCH_SIZE = 8

    # Xóa sạch file debug ở mỗi lần chạy mới (ghi đè thay vì append)
    for _dbg_file in ("notebook/debug_/debug_important_contexts.txt", "notebook/debug_/debug_relations_extracted.txt"):
        import os
        os.makedirs("notebook/debug_", exist_ok=True)
        try:
            open(_dbg_file, "w", encoding="utf-8").close()
        except Exception:
            pass

    all_matched_info = [] # (source_doc, full_context)
    seen_contexts = set()
    
    for doc in docs_list:
        s_doc = doc.get("source_doc")
        cnt = doc.get("content")
        if not cnt or not s_doc: continue
        # Sửa đổi: Gom nhóm theo cấu trúc Điều/Chương để giữ nguyên vẹn Context điều khoản thay vì cắt theo từng dòng
        blocks = re.split(r"(?im)^\s*((?:Điều|Chương|Phần|Mục)\s+[0-9IVXLCDM]+[a-zA-ZĐđ]*[\.\:\-]*\s*)", cnt)
        paras = []
        if blocks:
            if len(blocks[0].strip()) > 30:
                paras.append(blocks[0].strip())
            for i in range(1, len(blocks), 2):
                title = blocks[i]
                body = blocks[i+1] if i+1 < len(blocks) else ""
                full_block = (title + body).strip()
                if len(full_block) > 30:
                    paras.append(full_block)
                    
        norm_s_doc = normalize_entity(s_doc)
        
        # Regex nhận diện đại từ nội-văn-bản (tham chiếu nội bộ không phải văn bản khác)
        SELF_REF_PATTERN = re.compile(
            r"(?:Luật|Nghị định|Thông tư|Quyết định|Bộ luật|Văn bản|Quy định|Điều|Khoản|Mục)\s+này",
            re.IGNORECASE
        )
        
        for idx, para in enumerate(paras):
            # Bước lọc 1: Xóa sạch các đại từ tự chỉ định để tránh match nhầm ("Luật này", "Điều này"...)
            filtered_para = SELF_REF_PATTERN.sub("", para)
            
            matches = list(re.finditer(REGEX_PATTERN, filtered_para))
            if not matches: continue
            
            has_target_entity = False
            for m in matches:
                raw_match = m.group(0).strip()
                norm_match = normalize_entity(raw_match)
                
                # Bước lọc 2: Entity phải khác source và không phải UNKNOWN
                if norm_match == norm_s_doc or norm_match == "UNKNOWN":
                    continue
                
                # Bước lọc 3: Entity phải chứa số hiệu VB (dấu /) HOẶC tên luật rõ ràng (>8 ký tự)
                has_doc_num = bool(re.search(r"\d+[\/\-][A-Z0-9a-z\u0110\u0111]+", raw_match))
                has_law_name = (
                    bool(re.search(r"(?:Hi\u1ebfn ph\u00e1p|B\u1ed9 lu\u1eadt|Lu\u1eadt)\s+\S", raw_match, re.IGNORECASE))
                    and len(raw_match) > 8
                )
                
                if has_doc_num or has_law_name:
                    has_target_entity = True
                    break
            
            # Nếu chỉ nhắc đến bản thân hoặc đại từ nội bộ thì BỎ QUA
            if not has_target_entity:
                continue
                
            # RÚT GỌN WINDOW: Nới kịch trần lên 2500 char để bọc trọn những Điều khoản bổ sung dài
            if len(para) > 2500:
                # Tìm vị trí thực thể đầu tiên
                first_match = matches[0]
                start_cut = max(0, first_match.start() - 500)
                end_cut = min(len(para), first_match.end() + 2000)
                full_context = "..." + para[start_cut:end_cut] + "..."
            else:
                full_context = para
                
            # Hash Deduplication: Chống trùng lặp gửi nhiều đoạn giống nhau
            if full_context in seen_contexts:
                continue
            seen_contexts.add(full_context)
            
            all_matched_info.append({"s_doc": s_doc, "context": full_context})

    results_map = {d.get("source_doc"): [] for d in docs_list if d.get("source_doc")}

    # =========================================================================
    # PASSIVE CHAIN DETECTION — chạy độc lập, TRƯỚC filter has_target_entity
    # Lý do: đoạn "Căn cứ Luật A được sửa đổi theo Luật B" có thể chứa tên luật
    # thuần túy (không có số hiệu dạng xx/xxxx) → bị bỏ qua bởi has_target_entity
    # =========================================================================
    if skip_llm:
        # Cần quét lại RAW paragraphs (trước khi lọc has_target_entity)
        for doc in docs_list:
            s_doc = doc.get("source_doc")
            cnt = doc.get("content")
            if not cnt or not s_doc:
                continue
            norm_s_doc_pc = normalize_entity(s_doc)
            # Tìm key trong results_map
            found_k_pc = None
            for k in results_map.keys():
                if normalize_entity(k) == norm_s_doc_pc:
                    found_k_pc = k
                    break
            tgt_key_pc = found_k_pc if found_k_pc else s_doc
            if tgt_key_pc not in results_map:
                results_map[tgt_key_pc] = []

            # Split raw paragraphs (giống logic trên)
            raw_blocks = re.split(r"(?im)^\s*((?:Điều|Chương|Phần|Mục)\s+[0-9IVXLCDM]+[a-zA-ZĐđ]*[\.:\-]*\s*)", cnt)
            raw_paras = []
            if raw_blocks:
                if len(raw_blocks[0].strip()) > 30:
                    raw_paras.append(raw_blocks[0].strip())
                for i in range(1, len(raw_blocks), 2):
                    body = raw_blocks[i+1] if i+1 < len(raw_blocks) else ""
                    fp = (raw_blocks[i] + body).strip()
                    if len(fp) > 30:
                        raw_paras.append(fp)

            already_pc = set()
            for rel in results_map[tgt_key_pc]:
                already_pc.add((rel["source_doc"], rel["target_doc"]))
                already_pc.add((rel["target_doc"], rel["source_doc"]))

            # Chỉ cần quét block đầu (trước Điều 1) — đó là phần Căn cứ
            # detect_passive_chains tự lọc từng dòng bắt đầu bằng "Căn cứ" bên trong
            if raw_paras:
                d_rels, c_rels = detect_passive_chains(raw_paras[0], norm_s_doc_pc, already_pc)
                for r in d_rels + c_rels:
                    results_map[tgt_key_pc].append(r)

    if not all_matched_info:
        return results_map

    if skip_llm:

        for item in all_matched_info:
            s_doc = item["s_doc"]
            para = item["context"]
            norm_s_doc = normalize_entity(s_doc)

            # Xác định vùng văn bản để chọn bộ keyword phù hợp
            is_preamble = bool(_PREAMBLE_DETECT.match(para))
            active_keywords = _LABEL_KEYWORDS_PREAMBLE if is_preamble else _LABEL_KEYWORDS_ARTICLE

            # Tìm owner key trong results_map
            found_k = None
            for k in results_map.keys():
                if normalize_entity(k) == norm_s_doc:
                    found_k = k
                    break
            target_list_key = found_k if found_k else s_doc
            if target_list_key not in results_map:
                results_map[target_list_key] = []

            # Tập hợp đã xử lý để tránh trùng
            already_captured = set()
            for rel in results_map[target_list_key]:
                already_captured.add((rel["source_doc"], rel["target_doc"]))
                already_captured.add((rel["target_doc"], rel["source_doc"]))

            # --- LƯỢT 1: Quét số hiệu chuẩn xác ---
            matches = list(re.finditer(REGEX_PATTERN, para))
            for m in matches:
                tgt = normalize_entity(m.group(0))
                if not tgt or tgt == norm_s_doc or tgt == "UNKNOWN" or not is_vd(tgt):
                    continue
                if (norm_s_doc, tgt) in already_captured or (tgt, norm_s_doc) in already_captured:
                    continue

                art_match = re.search(r"((?:Điều|Chương|Phần|Mục)\s+[0-9A-ZĐđ]+)", para, re.IGNORECASE)
                art = art_match.group(1) if art_match else ""
                cl_match = re.search(r"(Khoản\s+\d+)", para, re.IGNORECASE)
                cl = cl_match.group(1) if cl_match else ""

                # Window quét ngược về phía trước entity để tìm nhãn quan hệ
                win_start = max(0, m.start() - 200)
                win_end = m.start()
                window = para[win_start:win_end].lower()

                best_pos, edge_label = -1, "BASED_ON"
                for label, kws in active_keywords:
                    for kw in kws:
                        pos = window.rfind(kw)
                        if pos > best_pos:
                            best_pos, edge_label = pos, label

                ctx_start = max(0, m.start() - 200)
                ctx_end = min(len(para), m.end() + 200)
                focused_context = para[ctx_start:ctx_end].replace("\n", " ").strip()

                rel_obj = {
                    "source_doc": norm_s_doc,
                    "target_doc": tgt,
                    "relation_phrase": "tham chiếu (regex)",
                    "edge_label": edge_label,
                    "context": focused_context,
                    "target_article": art,
                    "target_clause": cl,
                    "target_text": ""
                }
                if art and global_doc_lookup:
                    key = md.normalize_doc_key(tgt)
                    if key in global_doc_lookup:
                        target_full_text = global_doc_lookup[key]
                        truncated_text = extract_exact_article(target_full_text, art)
                        rel_obj["target_text"] = truncated_text

                exists = any(
                    e["target_doc"] == tgt and e["target_article"] == art and e["edge_label"] == edge_label
                    for e in results_map[target_list_key]
                )
                if not exists:
                    results_map[target_list_key].append(rel_obj)
                    already_captured.add((norm_s_doc, tgt))
                    # print(f"\n[DEBUG REGEX] {norm_s_doc} --{edge_label}--> {tgt}\n   ╰─ Context: '{focused_context}'")

        total_rels = sum(len(rel_list) for rel_list in results_map.values())
        return results_map

    # Gom nhóm 30 đoạn văn vào 1 prompt
    prompts_to_send = []
    metadata_for_prompts = [] # Để biết prompt này thuộc về những đoạn văn nào
    
    total_paras = len(all_matched_info)
    for i in range(0, total_paras, BATCH_SIZE):
        batch = all_matched_info[i : i + BATCH_SIZE]
        
        # Build context text cho prompt
        ctx_text = ""
        for j, item in enumerate(batch):
            ctx_text += f"\n--- ĐOẠN {j+1} (VB: {item['s_doc']}) ---\n{item['context']}\n"
            
        prompt = f"""Trích xuất TẤT CẢ triplets quan hệ pháp luật từ {len(batch)} đoạn dưới đây.

BƯỚC 1 — QUÉT SỐ HIỆU (BẮT BUỘC TRƯỚC KHI LÀM):
Trước khi viết JSON, hãy nhẩm: "Đoạn này nhắc đến những số hiệu VB nào?" — kể cả số hiệu xuất hiện trong cụm "đã được sửa đổi theo Luật số X", "theo Luật số Y, Z".
→ Mỗi số hiệu tìm được PHẢI tạo ít nhất 1 quan hệ, dù chỉ là REFERENCES.

QUY TẮC:
- Trường "source": TUYỆT ĐỐI không dùng "Luật này", "Nghị định này" — phải ghi số hiệu đầy đủ lấy từ header đoạn văn (VD: 44/2019/QH14).
- Cứ mỗi Điều/Khoản bị sửa đổi → tách thành 1 object JSON riêng biệt (không gộp chung).
- Các VB được nhắc đến ngầm ("đã được sửa đổi theo Luật số X") → tạo quan hệ REFERENCES cho từng cái.

CÁC NHÃN QUAN HỆ:
- BASED_ON: "Căn cứ ...", "Trên cơ sở ...", VB được nhắc đến ngầm trong ngữ cảnh ("theo Luật số X", "đã được sửa đổi theo...")
- AMENDS: "Sửa đổi, bổ sung ..."
- REPEALS: "Bãi bỏ ...", "hết hiệu lực ..."
- REPLACES: "Thay thế ..."
- GUIDES: "Hướng dẫn thi hành ..."
- APPLIES: "Áp dụng ...", "thực hiện theo ..."
- ISSUED_WITH: "Ban hành kèm theo ..."
- ASSIGNS: "Giao trách nhiệm ...", "phân công ..."
- CORRECTS: "Đính chính ...", "sửa lỗi ..."

VÍ DỤ (sát thực tế luật VN có tham chiếu lồng nhau):
Đoạn (VB: 44/2019/QH14):
  "1. Sửa đổi khoản 8 Điều 8 của Luật GT đường bộ số 23/2008/QH12 đã được sửa đổi theo Luật số 35/2018/QH14.
   2. Sửa đổi khoản 8 Điều 8 của Luật GT đường thủy số 23/2004/QH11 đã được sửa đổi theo Luật số 48/2014/QH13 và Luật số 97/2015/QH13."
→ Phải trích ra 5 quan hệ:
  1. 44/2019/QH14 AMENDS 23/2008/QH12 (Khoản 8 Điều 8)
  2. 44/2019/QH14 BASED_ON 35/2018/QH14   ← nhắc ngầm trong ngữ cảnh
  3. 44/2019/QH14 AMENDS 23/2004/QH11 (Khoản 8 Điều 8)
  4. 44/2019/QH14 BASED_ON 48/2014/QH13   ← nhắc ngầm
  5. 44/2019/QH14 BASED_ON 97/2015/QH13   ← nhắc ngầm

NGỮ CẢNH:
{ctx_text}

ĐỊNH DẠNG BẮT BUỘC (Chỉ JSON thuần, không giải thích):
{{
  "relationships": [
    {{
      "source": "Số hiệu VB nguồn (lấy từ header đoạn)",
      "target": "Số hiệu hoặc Tên VB đích",
      "relation_phrase": "cụm từ gốc trong văn bản",
      "edge_label": "NHÃN_QUAN_HỆ",
      "target_article": "Điều X (nếu có)",
      "target_clause": "Khoản Y (nếu có)",
      "target_text_content": "nội dung tóm tắt (nếu có)"
    }}
  ]
}}"""

        prompts_to_send.append([{"role": "user", "content": prompt}])
        metadata_for_prompts.append(batch)

    total_prompts = len(prompts_to_send)
    print(f"      [Pha 1] 🚀 Tìm thấy {total_paras} đoạn văn quan trọng. Đang xử lý {total_prompts} mẻ ({BATCH_SIZE} đoạn/prompt)...")
    
    try:
        with open("notebook/debug_/debug_important_contexts.txt", "a", encoding="utf-8") as f:
            f.write(f"\n--- TÌM THẤY {total_paras} ĐOẠN QUAN TRỌNG ĐỂ GỬI LLM ---\n")
            for item in all_matched_info:
                f.write(f"[VĂN BẢN: {item['s_doc']}]\n{item['context']}\n{'-'*60}\n")
    except Exception: pass
    
    # Thực thi gọi LLM Batch
    responses = llm_client.batch_chat_completion(
        messages_list=prompts_to_send,
        temperature=0.1,
        max_tokens=4000,
        response_format={"type": "json_object"}
    )

    # 3. Phân tích kết quả
    for i, resp_text in enumerate(responses):
        if not resp_text: continue
        # 1. Ưu tiên bóc tách từ Markdown Code Block
        json_str = None
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        
        # 2. Sử dụng helper robust
        if not json_str:
            from backend.utils.text_utils import extract_json_from_text
            json_str = extract_json_from_text(resp_text)
                        
        if not json_str: continue
        
        try:
            # Fix trailing commas
            json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
            data = json.loads(json_str)
            rel_list = data.get("relationships", [])
            
            # Chuẩn hóa: gộp REFERENCES vào BASED_ON
            for r in rel_list:
                if (r.get("edge_label") or "").strip().upper() == "REFERENCES":
                    r["edge_label"] = "BASED_ON"
            
            if not rel_list:
                pass # print(f"      [DEBUG Mẻ {i+1}] LLM trả về 0 quan hệ...")
            else:
                pass # print(f"      [DEBUG Mẻ {i+1}] LLM trả về {len(rel_list)} quan hệ dạng RAW.")
            
            # Map ngược lại context dựa trên source_doc
            batch_meta = metadata_for_prompts[i]
            
            for r in rel_list:
                src = normalize_entity(r.get("source", ""))
                tgt = normalize_entity(r.get("target", ""))
                
                if not src or not tgt or src == tgt or not is_vd(src) or not is_vd(tgt): 
                    # print(f"      [FILTERED] Bỏ qua quan hệ: '{src}' -> '{tgt}'")
                    continue
                
                # SỬA LỖI LỚN: Tìm đúng Văn bản gốc (doc_owner) của đoạn văn này. 
                # Vì LLM trả về Triplet (A -> B), nên A hoặc B phải trùng với tên Văn bản sở hữu Context (s_doc).
                doc_owner = None
                exact_ctx = ""
                for m in batch_meta:
                    norm_owner = normalize_entity(m["s_doc"])
                    if norm_owner == src or norm_owner == tgt:
                        doc_owner = m["s_doc"]
                        exact_ctx = m["context"]
                        break
                        
                if not doc_owner: 
                    doc_owner = batch_meta[0]["s_doc"]
                    exact_ctx = batch_meta[0]["context"]

                art = (r.get("target_article") or "").strip()
                cl = (r.get("target_clause") or "").strip()
                snp = (r.get("target_text_content") or "").strip()
                
                rel_obj = {
                    "source_doc": src, "target_doc": tgt, "relation_phrase": (r.get("relation_phrase") or "").strip().lower(),
                    "edge_label": (r.get("edge_label") or "UNKNOWN").strip().upper(), "context": exact_ctx,
                    "target_article": art, "target_clause": cl, "target_text": snp
                }
                
                # Trích xuất trực tiếp bằng Regex, KHÔNG dùng Pha 2
                if art and global_doc_lookup:
                    key = md.normalize_doc_key(tgt)
                    if key in global_doc_lookup:
                        target_full_text = global_doc_lookup[key]
                        truncated_text = extract_exact_article(target_full_text, art)
                        
                        if truncated_text:
                            rel_obj["target_text"] = truncated_text

                # Append theo đúng DOC_OWNER thay vì nhét nhầm vào src
                if doc_owner not in results_map:
                    results_map[doc_owner] = []
                    
                exists = False
                for existing in results_map[doc_owner]:
                    if existing["source_doc"] == rel_obj["source_doc"] and existing["target_doc"] == rel_obj["target_doc"] and existing["target_article"] == rel_obj["target_article"] and existing["edge_label"] == rel_obj["edge_label"]:
                        exists = True
                        break
                        
                if not exists:
                    results_map[doc_owner].append(rel_obj)
                    
                    target_info = []
                    if rel_obj['target_clause']: target_info.append(rel_obj['target_clause'])
                    if rel_obj['target_article']: target_info.append(rel_obj['target_article'])
                    target_str = f" tại {' '.join(target_info)}" if target_info else ""
                    
                    print(f"      [DEBUG] ✓ Đã LƯU quan hệ: {src} --[{rel_obj['edge_label']}]--> {tgt}{target_str} (vào '{doc_owner}')")
                    
                    try:
                        with open("notebook/debug_/debug_relations_extracted.txt", "a", encoding="utf-8") as f:
                            f.write(f"[{doc_owner} LLM] {src} --[{rel_obj['edge_label']}]--> {tgt}{target_str}\n")
                            f.write(f"=== CONTEXT GỐC ===\n{rel_obj['context']}\n")
                            f.write(f"=== NỘI DUNG ĐÍCH (Target Text) ===\n{rel_obj.get('target_text', 'N/A')}\n")
                            f.write("-" * 50 + "\n")
                    except Exception: pass
                    
        except Exception as e: 
            print(f"      [Lỗi] Mẻ prompt {i+1} thất bại (Có thể do JSON bị cắt cụt hoặc sai định dạng): {e}")
            continue

    # =========================================================================
    # PHA QUÉT BÙ: Regex sweep qua tất cả context đã gửi LLM
    # Mục đích: Tóm gọn các số hiệu VB nhắc đến ngầm mà LLM bỏ sót
    # =========================================================================
    for item in all_matched_info:
        s_doc = item["s_doc"]
        ctx = item["context"]
        norm_s_doc = normalize_entity(s_doc)
        is_preamble = bool(_PREAMBLE_DETECT.match(ctx))
        active_keywords = _LABEL_KEYWORDS_PREAMBLE if is_preamble else _LABEL_KEYWORDS_ARTICLE

        # Tìm owner key trong results_map
        owner_key = s_doc
        for k in results_map.keys():
            if normalize_entity(k) == norm_s_doc:
                owner_key = k
                break
        if owner_key not in results_map:
            results_map[owner_key] = []

        # Thu thập tất cả VB đã có trong results_map[owner_key] để tránh trùng
        already_captured = set()
        for rel in results_map.get(owner_key, []):
            already_captured.add((rel["source_doc"], rel["target_doc"]))
            already_captured.add((rel["target_doc"], rel["source_doc"]))

        # --- LƯỢT 1: Quét số hiệu chuẩn xác ---
        for m in REGEX_DOC_NUM_STRICT.finditer(ctx):
            raw = m.group(1)
            tgt = normalize_entity(raw)

            if not tgt or tgt == norm_s_doc or tgt == "UNKNOWN":
                continue
            if not is_vd(tgt):
                continue
            if (norm_s_doc, tgt) in already_captured or (tgt, norm_s_doc) in already_captured:
                continue

            win_start = max(0, m.start() - 200)
            win_end = m.start()
            window = ctx[win_start:win_end].lower()

            best_pos, edge_label = -1, "BASED_ON"
            for label, kws in active_keywords:
                for kw in kws:
                    pos = window.rfind(kw)
                    if pos > best_pos:
                        best_pos, edge_label = pos, label

            rel_obj = {
                "source_doc": norm_s_doc,
                "target_doc": tgt,
                "relation_phrase": "tham chiếu (regex sweep)",
                "edge_label": edge_label,
                "context": ctx,
                "target_article": "",
                "target_clause": "",
                "target_text": ""
            }
            results_map[owner_key].append(rel_obj)
            already_captured.add((norm_s_doc, tgt))

        # --- LƯỢT 2: Quét Tên Luật (Sử dụng OMNIBUS_LAWS_MAPPER) ---
        for m in re.finditer(REGEX_LAW_NAME, ctx, re.IGNORECASE):
            raw = m.group(0).lower().strip()
            core_name = re.sub(r"\s+năm\s+\d{4}$", "", raw).strip()

            tgt = None
            if core_name in OMNIBUS_LAWS_MAPPER:
                tgt = OMNIBUS_LAWS_MAPPER[core_name]
            elif raw in OMNIBUS_LAWS_MAPPER:
                tgt = OMNIBUS_LAWS_MAPPER[raw]

            if not tgt or tgt == norm_s_doc or tgt == "UNKNOWN":
                continue
            if (norm_s_doc, tgt) in already_captured or (tgt, norm_s_doc) in already_captured:
                continue

            win_start = max(0, m.start() - 200)
            win_end = m.start()
            window = ctx[win_start:win_end].lower()

            best_pos, edge_label = -1, "BASED_ON"
            for label, kws in active_keywords:
                for kw in kws:
                    pos = window.rfind(kw)
                    if pos > best_pos:
                        best_pos, edge_label = pos, label

            rel_obj = {
                "source_doc": norm_s_doc,
                "target_doc": tgt,
                "relation_phrase": "tham chiếu tên luật (regex sweep)",
                "edge_label": edge_label,
                "context": ctx,
                "target_article": "",
                "target_clause": "",
                "target_text": ""
            }
            results_map[owner_key].append(rel_obj)
            already_captured.add((norm_s_doc, tgt))

    total_rels = sum(len(rel_list) for rel_list in results_map.values())
    return results_map


def extract_ontology_relationships(content: str, source_doc: str, global_doc_lookup: dict = None, skip_llm: bool = False) -> List[dict]:
    """Wrapper for single document processing — trả về flat list cho core.py."""
    res = extract_ontology_relationships_batch([{"source_doc": source_doc, "content": content}], global_doc_lookup, skip_llm=skip_llm)
    # Gom cả quan hệ thường lẫn cross_rels (is_cross_doc=True) vào cùng 1 list
    return res.get(source_doc, [])