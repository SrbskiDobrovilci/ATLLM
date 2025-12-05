import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # GigaChat API
    GIGACHAT_API_KEY: str = os.getenv("GIGACHAT_API_KEY")
    GIGACHAT_BASE_URL: str = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
    GIGACHAT_MODEL: str = os.getenv("GIGACHAT_MODEL", "GigaChat")
    GIGACHAT_SCOPE: str = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    
    # Embedding model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sberbank-ai/sbert_large_nlu_ru")
    
    # Vector Database (Qdrant)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_documents")
    
    # Redis for memory
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SIMILARITY_THRESHOLD: float = 0.7
    TOP_K_RESULTS: int = 5
    
    class Config:
        env_file = ".env"

settings = Settings()