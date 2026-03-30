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

    LEGAL_DENSE_MODEL: str = "BAAI/bge-m3"

    # Vector DB / Queue
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "legal_rag_10000"
    QDRANT_READ_ONLY: bool = True
    REDIS_URL: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
