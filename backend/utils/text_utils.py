import re

def extract_thinking_and_answer(text: str) -> tuple[str, str]:
    """
    Bóc tách phần 'thinking' và phần 'answer' thành 2 chuỗi riêng biệt.
    Do model mới không còn bị lộ thẻ thinking, hàm này đơn giản hóa trả về nguyên gốc.
    """
    if not text: return "", ""
    return "", text.strip()

def strip_thinking_tags(text: str) -> str:
    """Loại bỏ hoàn toàn các tag thinking và nội dung bên trong. Do model mới đã fix nên return nguyên gốc."""
    return text.strip() if text else ""

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
