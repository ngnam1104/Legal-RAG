import os
import logging
from pydantic_settings import BaseSettings

# Silence redundant httpx logs globally
logging.getLogger("httpx").setLevel(logging.WARNING)

class Settings(BaseSettings):
    # LLM (Đổi sang Internal On-Premise)
    LLM_PROVIDER: str = "internal"
    LLM_API_KEY: str = ""  
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_ROUTING_MODEL: str = "llama3"
    LLM_CORE_MODEL: str = "llama3"
    
    # Retry strategy cho hệ thống mạng nội bộ
    LLM_RETRY_DELAY: int = 3  # Seconds
    LLM_MAX_RETRIES: int = 3
    
    # Micro-batching để bảo vệ LLM Memory
    LLM_MICRO_BATCH_SIZE: int = 4
    LLM_INTER_BATCH_SLEEP: float = 3.0

    # RAG Retrieval limits
    MAX_CONTEXT_CHARS: int = 50000  # Soft safety limit, no longer a hard cut-off
    MAX_RETRIEVAL_HITS: int = 20    # Default top-K chunks to include in context
    
    HF_HOME: str = "./.cache/huggingface"
    SENTENCE_TRANSFORMERS_HOME: str = "./.cache/sentence_transformers"

    LEGAL_DENSE_MODEL: str = "BAAI/bge-m3"

    # Vector DB / Queue
    QDRANT_URL: str = "http://localhost:6335"
    QDRANT_PATH: str = "" # Set this to use local path instead of URL
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "legal_hybrid_rag_docs"
    ENABLE_RERANK: bool = True
    ENABLE_GRADING: bool = False     # [Ablation Study] Bật/tắt node Grade
    ENABLE_REFLECTION: bool = False  # [Ablation Study] Bật/tắt node Reflect
    QDRANT_READ_ONLY: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Neo4j Graph DB (Bắt buộc dùng Docker Local)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "u7aGQYEWeFJD-jyeHB4ATtoAud73PptW35M1RzFlT-0"

    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        "case_sensitive": True,
        "extra": "ignore"
    }

settings = Settings()

# Đảm bảo đường dẫn cache là đường dẫn tuyệt đối để tránh tạo folder "D:" giả trong repo
settings.HF_HOME = os.path.abspath(settings.HF_HOME)
settings.SENTENCE_TRANSFORMERS_HOME = os.path.abspath(settings.SENTENCE_TRANSFORMERS_HOME)

# [QUAN TRỌNG] Phải Ghi đè biến môi trường hệ thống để thư viện HuggingFace thực sự lưu vào ổ D thay vì ổ C mặc định
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\huggingface_cache\sentence_transformers"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface_cache"

# end of config
