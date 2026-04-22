import re
import json
import time
import random
import pandas as pd
from typing import Dict, List
from backend.retrieval.chunker import metadata as md
from backend.llm.client import InternalLLMClient

llm_client = InternalLLMClient()

# ==========================================
# BƯỚC HẬU XỬ LÝ: CHUẨN HÓA THỰC THỂ (ENTITY NORMALIZATION)
# ==========================================
REGEX_DOC_NUM = r"\b\d+[\/\-](?:20\d{2}|19\d{2})[\/\-][A-ZĐ][A-Z0-9Đa-zđ\-\/]*\b|\b\d+[\/\-][A-ZĐ][A-Z0-9Đa-zđ\-\/]*\b"
REGEX_LAW_NAME = r"(?i)(?:Hiến pháp|Bộ luật|Luật)\s+[\w\s\,\.\-]+(?:năm\s+\d{4})?"
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
}

SORTED_MAPPER = dict(sorted(OMNIBUS_LAWS_MAPPER.items(), key=lambda item: len(item[0]), reverse=True))

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
        resp = resp.replace("```json", "").replace("```", "").strip() if resp else ""
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if match: return json.loads(match.group(0)).get("references", [])
    except Exception as e: print(f"⚠️ LLM Error: {e}")
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
    end_pattern = re.compile(r"(?im)^\s*(?:Điều\s+\d|Chương\s+[IVXLCDM0-9]|Phần\s+(?:thứ\s+)?[IVXLCDM0-9])")
    end_match = end_pattern.search(remaining)
    if end_match: result = content[start_pos: start_match.end() + end_match.start()]
    else: result = content[start_pos:]
    return result.strip()

def extract_target_content_via_llm(full_text_target: str, target_article: str, target_clause: str = None) -> str:
    truncated_text = extract_exact_article(full_text_target, target_article)
    if not truncated_text: return ""
    if not target_clause: return truncated_text
    prompt = f'''Trích xuất NGUYÊN VĂN nội dung của "{target_clause} {target_article}" từ:\n{truncated_text}\nCHỈ trả về nội dung, không giải thích.'''
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

def extract_ontology_relationships_batch(docs_list: List[Dict], global_doc_lookup: dict = None) -> Dict[str, List[dict]]:
    if not docs_list: return {}
    
    REGEX_DOC_NUM = r"\b\d+[\/\-](?:20\d{2}|19\d{2})[\/\-][A-ZĐ][A-Z0-9Đa-zđ\-\/]*\b|\b\d+[\/\-][A-ZĐ][A-Z0-9Đa-zđ\-\/]*\b"
    REGEX_LAW_NAME = r"(?:Hiến pháp|Bộ luật|Luật)\s+[A-ZĐ][a-zđA-ZĐ\s]*(?:năm\s+\d{4})?|(?:Hiến pháp)(?:\s+năm\s+\d{4})?"
    REGEX_PATTERN = re.compile(f"({REGEX_DOC_NUM}|{REGEX_LAW_NAME})")
    
    # =========================================================================
    # PHA 1: TRÍCH XUẤT QUAN HỆ (Gom 30 đoạn văn/prompt để tối ưu hiệu suất)
    # =========================================================================
    BATCH_SIZE = 30
    all_matched_info = [] # (source_doc, full_context)
    
    for doc in docs_list:
        s_doc = doc.get("source_doc")
        cnt = doc.get("content")
        if not cnt or not s_doc: continue
        paras = [p.strip() for p in cnt.splitlines() if len(p.strip()) > 30]
        
        for idx, para in enumerate(paras):
            matches = list(re.finditer(REGEX_PATTERN, para))
            if not matches: continue
            window = paras[max(0, idx-1) : min(len(paras), idx+6)]
            full_context = "\n".join(window)
            all_matched_info.append({"s_doc": s_doc, "context": full_context})

    results_map = {d.get("source_doc"): [] for d in docs_list if d.get("source_doc")}
    if not all_matched_info:
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
            
        prompt = f"""Bạn là Chuyên gia Knowledge Graph Pháp luật.
NHIỆM VỤ: Trích xuất JSON triplets quan hệ giữa các văn bản pháp luật từ {len(batch)} đoạn văn sau.
CÁC NHÃN QUAN HỆ: BASED_ON, AMENDS, REPEALS, REPLACES, GUIDES, APPLIES, REFERENCES, ISSUED_WITH, ASSIGNS, CORRECTS.

NGỮ CẢNH:
{ctx_text}

YÊU CẦU ĐỊNH DẠNG:
- Trả về JSON chuẩn 100% theo mẫu:
{{"relationships": [{{"source": "Số hiệu VB nguồn", "target": "Số hiệu VB đích", "relation_phrase": "cụm từ thể hiện quan hệ", "edge_label": "NHÃN_QUAN_HỆ", "target_article": "Điều mấy", "target_clause": "Khoản mấy", "target_text_content": "nội dung trích dẫn ngắn"}}]}}
- Escape dấu ngoặc kép (") bên trong bằng dấu gạch chéo ngược (\\")."""

        prompts_to_send.append([{"role": "user", "content": prompt}])
        metadata_for_prompts.append(batch)

    total_prompts = len(prompts_to_send)
    print(f"      [Pha 1] 🚀 Tìm thấy {total_paras} đoạn văn quan trọng. Đang xử lý {total_prompts} mẻ ({BATCH_SIZE} đoạn/prompt)...")
    
    # Thực thi gọi LLM Batch
    responses = llm_client.batch_chat_completion(
        messages_list=prompts_to_send,
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    # 3. Phân tích kết quả Pha 1 và chuẩn bị Pha 2
    pending_phase2_tasks = []
    
    for i, resp_text in enumerate(responses):
        if not resp_text: continue
        clean_resp = resp_text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", clean_resp, re.DOTALL)
        if not match: continue
        
        try:
            json_str = re.sub(r',\s*([\]\}])', r'\1', match.group(0))
            data = json.loads(json_str)
            rel_list = data.get("relationships", [])
            
            # Map ngược lại context dựa trên source_doc
            batch_meta = metadata_for_prompts[i]
            
            for r in rel_list:
                src = normalize_entity(r.get("source", ""))
                tgt = normalize_entity(r.get("target", ""))
                if not src or not tgt or src == tgt or not is_vd(src) or not is_vd(tgt): continue
                
                # Tìm context gốc của đoạn văn chứa quan hệ này
                exact_ctx = ""
                for m in batch_meta:
                    if normalize_entity(m["s_doc"]) == src:
                        exact_ctx = m["context"]
                        break
                if not exact_ctx: exact_ctx = batch_meta[0]["context"]

                art = r.get("target_article", "").strip()
                cl = r.get("target_clause", "").strip()
                snp = r.get("target_text_content", "").strip()
                
                rel_obj = {
                    "source_doc": src, "target_doc": tgt, "relation_phrase": r.get("relation_phrase", "").strip().lower(),
                    "edge_label": r.get("edge_label", "UNKNOWN").strip().upper(), "context": exact_ctx,
                    "target_article": art, "target_clause": cl, "target_text": snp
                }
                
                # Check nếu có điều khoản -> Gom vào mảng để gọi Pha 2
                if art and global_doc_lookup:
                    key = md.normalize_doc_key(tgt)
                    if key in global_doc_lookup:
                        target_full_text = global_doc_lookup[key]
                        truncated_text = extract_exact_article(target_full_text, art)
                        
                        if truncated_text and cl:
                            pending_phase2_tasks.append({
                                "rel_obj_ref": rel_obj,
                                "article": art,
                                "clause": cl,
                                "context": truncated_text
                            })
                        elif truncated_text:
                            rel_obj["target_text"] = truncated_text

                if rel_obj not in results_map.get(src, []):
                    # Tìm đúng source_doc trong docs_list để map vào results_map
                    # Vì results_map keys là source_doc ban đầu
                    found_key = None
                    for k in results_map.keys():
                        if normalize_entity(k) == src:
                            found_key = k
                            break
                    if found_key: results_map[found_key].append(rel_obj)
                    elif src in results_map: results_map[src].append(rel_obj)
                    
        except Exception: continue

    # =========================================================================
    # PHA 2: TRÍCH XUẤT ĐIỀU/KHOẢN CỤ THỂ (Nếu có)
    # =========================================================================
    if pending_phase2_tasks:
        print(f"      [Pha 2] 🔍 Đang trích xuất chi tiết {len(pending_phase2_tasks)} điều/khoản bằng Batch...")
        phase2_messages = []
        
        for task in pending_phase2_tasks:
            prompt = f"Trích xuất NGUYÊN VĂN nội dung của '{task['clause']} {task['article']}' từ:\n{task['context']}\nCHỈ trả về nội dung, không giải thích."
            phase2_messages.append([{"role": "user", "content": prompt}])
            
        phase2_responses = []
        
        for i in range(0, len(phase2_messages), SUB_BATCH_SIZE):
            sub_messages = phase2_messages[i : i + SUB_BATCH_SIZE]
            try:
                sub_res = llm_client.batch_chat_completion(
                    messages_list=sub_messages,
                    temperature=0.1
                )
                phase2_responses.extend(sub_res)
            except Exception as e:
                print(f"      [Pha 2] ❌ Lỗi mẻ phụ: {e}")
                phase2_responses.extend([""] * len(sub_messages))
                
        # Ráp kết quả lại
        for i, resp_text in enumerate(phase2_responses):
            if resp_text and resp_text.strip():
                # Gán trực tiếp text mới vào object (dict) do Python truyền tham chiếu
                pending_phase2_tasks[i]["rel_obj_ref"]["target_text"] = resp_text.strip()

    total_rels = sum(len(rel_list) for rel_list in results_map.values())
    print(f"      [Hoàn tất] ✓ Trích xuất {total_rels} quan hệ từ {len(docs_list)} văn bản.")
    
    return results_map

def extract_ontology_relationships(content: str, source_doc: str, global_doc_lookup: dict = None) -> List[dict]:
    # Fallback function for single document processing
    res = extract_ontology_relationships_batch([{"source_doc": source_doc, "content": content}], global_doc_lookup)
    return res.get(source_doc, [])