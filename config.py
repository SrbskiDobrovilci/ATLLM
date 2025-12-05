import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///gigachat.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    GIGACHAT_CLIENT_ID = os.environ.get('GIGACHAT_CLIENT_ID')
    GIGACHAT_CLIENT_SECRET = os.environ.get('GIGACHAT_CLIENT_SECRET')
    GIGACHAT_SCOPE = os.environ.get('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')
    GIGACHAT_AUTH_URL = os.environ.get('GIGACHAT_AUTH_URL', 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth')
    GIGACHAT_API_URL = os.environ.get('GIGACHAT_API_URL', 'https://gigachat.devices.sberbank.ru/api/v1')
    MAX_CONTEXT_CHATS = 5  # Maximum number of saved conversations per user
    MAX_MESSAGES_PER_CHAT = 50  # Maximum messages per conversation