import os
import sys
from datasets import load_dataset

# Add project root to path (scripts/tests/debug/file.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
HF_REPO = "nhn309261/vietnamese-legal-documents"

try:
    print(f"Loading dataset {HF_REPO} (streaming=True)...")
    ds = load_dataset(HF_REPO, "content", split="data", streaming=True)
    print("Dataset loaded successfully.")
    
    print("Fetching first 5 records...")
    count = 0
    for record in ds:
        print(f"ID: {record.get('id')}")
        print(f"Keys: {list(record.keys())}")
        count += 1
        if count >= 5:
            break
except Exception as e:
    print(f"Error: {e}")
