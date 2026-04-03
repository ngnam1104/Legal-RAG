import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "groq"  # groq | gemini | ollama
    LLM_API_KEY: str = ""  
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_CHAT_MODEL: str = "llama-3.1-8b-instant"
    
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "gemini-3-flash-preview"
    
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3"
    OLLAMA_API_KEY: str = ""

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
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }

settings = Settings()

# Đảm bảo đường dẫn cache là đường dẫn tuyệt đối để tránh tạo folder "D:" giả trong repo
settings.HF_HOME = os.path.abspath(settings.HF_HOME)
settings.SENTENCE_TRANSFORMERS_HOME = os.path.abspath(settings.SENTENCE_TRANSFORMERS_HOME)

