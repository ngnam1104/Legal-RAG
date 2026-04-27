import re

def extract_thinking_and_answer(text: str) -> tuple[str, str]:
    """
    Bóc tách phần 'thinking' và phần 'answer' thành 2 chuỗi riêng biệt.
    Trả về: (thinking_content, answer_content)
    Nếu không tìm thấy thinking block, thinking_content sẽ là chuỗi rỗng.
    """
    if not text: return "", ""
    
    thinking_parts = []
    answer = text
    
    # 1. Tìm các block có cấu trúc đóng mở rõ ràng (<think>...</think>, [THINKING]...[/THINKING])
    tags = ['thinking', 'think', 'thought']
    for tag in tags:
        pattern = rf'<{tag}>(.*?)</{tag}>'
        matches = re.findall(pattern, answer, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            thinking_parts.extend(matches)
            answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)
            
        # Thẻ mở mồ côi (lấy toàn bộ đuôi nếu vỡ thẻ mở)
        orphans = re.findall(rf'<{tag}>(.*)', answer, flags=re.DOTALL | re.IGNORECASE)
        if orphans:
            thinking_parts.extend(orphans)
            answer = re.sub(rf'<{tag}>.*', '', answer, flags=re.DOTALL | re.IGNORECASE)

    # 2. Xóa thẻ đóng mồ côi nếu còn vương vãi
    for tag in tags:
        answer = re.sub(rf'</{tag}>', '', answer, flags=re.IGNORECASE)

    # 3. Chỉ dùng marker cực kỳ an toàn
    safe_markers = [
        r"(?im)^(?:Final )?Answer:\s*",
        r"(?im)^(?:Final )?Output:\s*"
    ]
    
    found_marker_pos = -1
    marker_end = -1
    
    for marker_pat in safe_markers:
        match = re.search(marker_pat, answer)
        if match and match.start() < 1000:
            if match.start() > found_marker_pos:
                found_marker_pos = match.start()
                marker_end = match.end()
    
    if found_marker_pos != -1:
        thinking_candidate = answer[:found_marker_pos].strip()
        thinking_parts.append(thinking_candidate)
        answer = answer[marker_end:].strip()
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()

    final_thinking = "\n\n".join([p.strip() for p in thinking_parts if p.strip()])
    return final_thinking, answer.strip()

def strip_thinking_tags(text: str) -> str:
    """Loại bỏ hoàn toàn các tag thinking và nội dung bên trong."""
    _, answer = extract_thinking_and_answer(text)
    return answer

def extract_json_from_text(text: str) -> str | None:
    """
    Tìm và trích xuất khối JSON đầu tiên hợp lệ từ text bằng cách đếm dấu ngoặc {}.
    Hỗ trợ nested JSON.
    """
    if not text: return None
    
    # Tìm tất cả vị trí của dấu {
    starts = [m.start() for m in re.finditer(r'\{', text)]
    
    for start_pos in starts:
        count = 0
        for i in range(start_pos, len(text)):
            if text[i] == '{':
                count += 1
            elif text[i] == '}':
                count -= 1
                if count == 0:
                    potential_json = text[start_pos:i+1]
                    # Kiểm tra xem có thực sự là JSON không (tránh { mồ côi)
                    try:
                        import json
                        json.loads(potential_json)
                        return potential_json
                    except:
                        break # Thử vị trí start tiếp theo
    return None
