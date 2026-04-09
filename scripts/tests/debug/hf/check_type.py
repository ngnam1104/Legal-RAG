import os
import sys
from datasets import load_dataset

# Add project root to path (scripts/tests/debug/file.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

def check_type():
    print("Loading HF dataset stream...")
    ds = load_dataset('nhn309261/vietnamese-legal-documents', 'content', split='data', streaming=True)
    it = iter(ds)
    try:
        rec = next(it)
        print(f"ID: {rec['id']}, Type: {type(rec['id'])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_type()
