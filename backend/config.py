import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "groq"  # groq | gemini | ollama
    LLM_API_KEY: str = ""  
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_CHAT_MODEL: str = "llama-3.1-8b-instant"  # Deprecated in favor of ROUTING/CORE
    LLM_ROUTING_MODEL: str = "llama-3.1-8b-instant"
    LLM_CORE_MODEL: str = "llama-3.3-70b-versatile"
    
    # Retry strategy for FREE Tier (TPM/RPM limits)
    LLM_RETRY_DELAY: int = 20  # Seconds
    LLM_MAX_RETRIES: int = 5
    
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "models/gemini-1.5-flash"
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3"
    OLLAMA_API_KEY: str = ""

    # RAG Context limits
    MAX_CONTEXT_CHARS: int = 25000  # ~3500 tokens (Tối ưu cho 70B FREE Tier 12,000 TPM, cho phép 2 lần gọi/request)
    
    HF_HOME: str = "./.cache/huggingface"
    SENTENCE_TRANSFORMERS_HOME: str = "./.cache/sentence_transformers"

    LEGAL_DENSE_MODEL: str = "BAAI/bge-m3"

    # Vector DB / Queue
    QDRANT_URL: str = "http://localhost:6335"
    QDRANT_PATH: str = "" # Set this to use local path instead of URL
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "legal_rag_docs_5000"
    ENABLE_RERANK: bool = True
    QDRANT_READ_ONLY: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

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
