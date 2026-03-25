from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "groq"  # groq | gemini
    LLM_API_KEY: str = ""  
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_CHAT_MODEL: str = "llama-3.1-8b-instant"
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "gemini-3-flash-preview"
    LLM_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector DB / Queue
    QDRANT_URL: str = "http://qdrant:6333"
    QDRANT_COLLECTION: str = "legal_vn_200_docs"
    REDIS_URL: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
