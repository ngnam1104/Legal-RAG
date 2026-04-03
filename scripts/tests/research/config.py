import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env ở thư mục gốc
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
load_dotenv(dotenv_path=env_path)

# Groq LLM Config
GROQ_API_KEY = os.getenv("LLM_API_KEY")
QA_MODEL_NAME = "llama-3.3-70b-versatile"

# Qdrant Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6335")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_rag_docs_5000")

# Limit config cho 50-100 questions tổng cộng (~30-40 chunks)
# Mỗi chunk sẽ sinh ra ~10 record mảng (4 search + 3 qa + 3 conflict)
# Vậy 10 chunks -> ~100 questions output các loại. Lấy limit mặc định 15 để dư dả.
DEFAULT_CHUNK_LIMIT = 15

# Output paths
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "QA_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODE1_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_mode1_search.json")
MODE2_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_mode2_qa.json")
MODE3_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_mode3_conflict.json")

# Benchmark Report Paths
BENCHMARK_REPORT_JSON = os.path.join(OUTPUT_DIR, "benchmark_report.json")
BENCHMARK_REPORT_MD = os.path.join(OUTPUT_DIR, "benchmark_report.md")
