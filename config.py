import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///gigachat.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # GigaChat Configuration
    GIGACHAT_CLIENT_ID = os.environ.get('GIGACHAT_CLIENT_ID')
    GIGACHAT_CLIENT_SECRET = os.environ.get('GIGACHAT_CLIENT_SECRET')
    GIGACHAT_SCOPE = os.environ.get('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')
    GIGACHAT_AUTH_URL = os.environ.get('GIGACHAT_AUTH_URL', 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth')
    GIGACHAT_API_URL = os.environ.get('GIGACHAT_API_URL', 'https://gigachat.devices.sberbank.ru/api/v1')
    
    # RAG Configuration
    RAG_ENABLED = os.environ.get('RAG_ENABLED', 'true').lower() == 'true'
    DOCUMENTS_PATH = os.environ.get('DOCUMENTS_PATH', 'documents')
    VECTOR_STORE_PATH = os.environ.get('VECTOR_STORE_PATH', 'vector_store')
    DATA_PATH = os.environ.get('DATA_PATH', 'data')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 200))
    MAX_RETRIEVAL_CHUNKS = int(os.environ.get('MAX_RETRIEVAL_CHUNKS', 3))
    
    # Application Configuration
    MAX_CONTEXT_CHATS = 5
    MAX_MESSAGES_PER_CHAT = 50
    MAX_CONVERSATION_TOKENS = 4000