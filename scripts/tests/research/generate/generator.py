import json
import time
import os
import sys
from typing import List, Dict, Any
from groq import Groq

# Cấu hình đường dẫn: 4 cấp lên tới Root (scripts/tests/research/generate/file.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESEARCH_DIR = os.path.join(ROOT_DIR, "scripts", "tests", "research")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if RESEARCH_DIR not in sys.path:
    sys.path.insert(0, RESEARCH_DIR)

from config import GROQ_API_KEY, QA_MODEL_NAME
from generate.prompts import MODE1_SEARCH_PROMPT, MODE2_QA_PROMPT, MODE3_CONFLICT_PROMPT

# Cấu hình Groq Client
client = Groq(api_key=GROQ_API_KEY)

def call_groq(prompt: str, metadata_list: List[Dict[str, Any]], max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Gọi Groq API với JSON Mode. Tham số đầu vào là một list các chunks (cluster).
    """
    # Xoá bớt các trường vector đồ sộ của tất cả các chunks và giới hạn token (vì Groq Free Tier TPM khá thấp)
    filtered_metadata = []
    for metadata in metadata_list:
        safe_metadata = {}
        # Chỉ giữ lại những trường trọng tâm nhất để LLM làm context
        for key in ["chunk_id", "document_number", "title", "reference_tag", "chunk_text"]:
            if key in metadata:
                safe_metadata[key] = metadata[key]
                
        # Truncate text nếu quá dài (khoảng 300 từ mỗi chunk, 3 chunk là ~1000 từ -> an toàn cho groq limit)
        if "chunk_text" in safe_metadata and isinstance(safe_metadata["chunk_text"], str):
            words = safe_metadata["chunk_text"].split()
            if len(words) > 300:
                safe_metadata["chunk_text"] = " ".join(words[:300]) + "..."
                
        filtered_metadata.append(safe_metadata)
    
    formatted_prompt = prompt.format(full_metadata=json.dumps(filtered_metadata, ensure_ascii=False, indent=2))
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là chuyên gia về AI và dữ liệu Pháp lý. Hãy luôn phản hồi bằng chuẩn định dạng JSON theo đúng schema được yêu cầu."
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                model=QA_MODEL_NAME,
                temperature=0.3,
                # Bật JSON mode
                response_format={"type": "json_object"},
            )
            
            response_text = chat_completion.choices[0].message.content
            data = json.loads(response_text)
            
            # Extract key "items" từ json object
            return data.get("items", [])
            
        except json.JSONDecodeError as e:
            print(f"  [!] Lỗi Parse JSON từ LLM (Thử lại {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"  [!] Lỗi gọi Groq API (Thử lại {attempt+1}/{max_retries}): {e}")
            time.sleep(5) 
            
    return []

def generate_mode_1_search(metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return call_groq(MODE1_SEARCH_PROMPT, metadata_list)

def generate_mode_2_qa(metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return call_groq(MODE2_QA_PROMPT, metadata_list)

def generate_mode_3_conflict(metadata_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return call_groq(MODE3_CONFLICT_PROMPT, metadata_list)
