import os
import sys

# Force UTF-8 on Windows
os.environ["PYTHONUTF8"] = "1"

# Mock doc and content
doc = {
    'id': 'test_123',
    'type': 'Nghị định',
    'number': '91/2015/NĐ-CP',
    'authority': 'Chính phủ',
    'title': 'Nghị định về đầu tư vốn nhà nước vào doanh nghiệp'
}
content = "Đây là nội dung thử nghiệm với Tiếng Việt. \n\nĐiều 1: Phạm vi điều chỉnh.\nNghị định này quy định về việc đầu tư vốn nhà nước vào doanh nghiệp."

# Add project root to path (scripts/tests/debug/file.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.crawl_legal_docs import save_as_pdf, OUTPUT_DIR

# Setup test dirs
os.makedirs(os.path.join(OUTPUT_DIR, "pdf"), exist_ok=True)

import traceback

try:
    print("Testing save_as_pdf...")
    save_as_pdf(doc, content)
    print("PDF generated successfully.")
except Exception as e:
    print(f"Error during PDF generation: {e}")
    traceback.print_exc()
