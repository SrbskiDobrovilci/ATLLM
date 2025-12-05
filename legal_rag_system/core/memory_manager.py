import redis
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from config import settings

class MemoryManager:
    """Менеджер памяти для хранения истории диалогов"""
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL)
            self.redis_client.ping()
            logger.info("Подключение к Redis успешно")
        except Exception as e:
            logger.warning(f"Не удалось подключиться к Redis: {str(e)}. Используем in-memory хранилище.")
            self.redis_client = None
            self.memory_store = {}
    
    def _get_key(self, session_id: str) -> str:
        """Генерация ключа для сессии"""
        return f"legal_chat:{session_id}"
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Добавление сообщения в историю"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        key = self._get_key(session_id)
        
        if self.redis_client:
            try:
                # Добавляем сообщение в список
                self.redis_client.rpush(key, json.dumps(message, ensure_ascii=False))
                # Устанавливаем TTL на 24 часа
                self.redis_client.expire(key, 86400)
            except Exception as e:
                logger.error(f"Ошибка при сохранении в Redis: {str(e)}")
        else:
            if session_id not in self.memory_store:
                self.memory_store[session_id] = []
            self.memory_store[session_id].append(message)
    
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение истории сообщений"""
        key = self._get_key(session_id)
        
        if self.redis_client:
            try:
                messages_json = self.redis_client.lrange(key, -limit, -1)
                messages = [json.loads(msg) for msg in messages_json]
                return messages
            except Exception as e:
                logger.error(f"Ошибка при чтении из Redis: {str(e)}")
                return []
        else:
            return self.memory_store.get(session_id, [])[-limit:]
    
    def clear_history(self, session_id: str):
        """Очистка истории сессии"""
        key = self._get_key(session_id)
        
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Ошибка при удалении из Redis: {str(e)}")
        else:
            if session_id in self.memory_store:
                del self.memory_store[session_id]
    
    def get_conversation_context(self, session_id: str, max_tokens: int = 1000) -> str:
        """Получение сжатого контекста разговора"""
        history = self.get_history(session_id, limit=20)
        
        if not history:
            return ""
        
        # Сжимаем историю, оставляя только самое важное
        context_parts = []
        token_count = 0
        
        for msg in reversed(history):
            role = msg["role"]
            content = msg["content"]
            # Примерная оценка токенов (1 токен ≈ 4 символа)
            content_tokens = len(content) // 4
            
            if token_count + content_tokens <= max_tokens:
                context_parts.insert(0, f"{role}: {content}")
                token_count += content_tokens
            else:
                break
        
        return "\n".join(context_parts)