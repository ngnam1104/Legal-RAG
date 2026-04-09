import os
import sys
from datasets import load_dataset_builder, get_dataset_config_names

# Add project root to path (scripts/tests/debug/file.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
HF_REPO = "nhn309261/vietnamese-legal-documents"

try:
    print(f"Checking configs for {HF_REPO}...")
    configs = get_dataset_config_names(HF_REPO)
    print(f"Configs: {configs}")
    
    # Check builder
    builder = load_dataset_builder(HF_REPO, "content")
    print(f"Features: {builder.info.features}")
    print(f"Splits: {builder.info.splits}")
    
except Exception as e:
    print(f"Error: {e}")
