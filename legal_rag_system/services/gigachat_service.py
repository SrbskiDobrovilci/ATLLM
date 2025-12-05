import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp
from loguru import logger

from config import settings

class GigaChatService:
    """Сервис для работы с GigaChat API"""
    
    def __init__(self):
        self.api_key = settings.GIGACHAT_API_KEY
        self.base_url = settings.GIGACHAT_BASE_URL
        self.model = settings.GIGACHAT_MODEL
        self.scope = settings.GIGACHAT_SCOPE
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        # Кэш токена
        self._access_token = None
        self._token_expires_at = 0
    
    async def _get_access_token(self) -> str:
        """Получение access token для GigaChat API"""
        # Проверяем, действителен ли текущий токен
        import time
        if self._access_token and time.time() < self._token_expires_at - 60:
            return self._access_token
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': '6f0b1291-c7f3-43c6-bb2e-9f3efb2dc98e'
        }
        
        data = {'scope': self.scope}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.auth_url,
                headers=headers,
                data=data,
                ssl=False
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self._access_token = result.get('access_token')
                    # Токен обычно действует 30 минут
                    self._token_expires_at = time.time() + 1800
                    logger.info("Токен GigaChat успешно получен")
                    return self._access_token
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка получения токена: {error_text}")
                    raise Exception(f"Ошибка аутентификации: {response.status}")
    
    def _create_legal_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Создание промпта для юридического анализа"""
        # Формируем контекст из найденных документов
        context_texts = []
        for i, doc in enumerate(context, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("document_type", "документ")
            if "article" in metadata:
                source += f", статья {metadata['article']}"
            
            context_texts.append(
                f"[Документ {i}] Источник: {source}\n"
                f"Текст: {doc['text'][:500]}...\n"
            )
        
        context_str = "\n".join(context_texts)
        
        prompt = f"""Ты — опытный российский юрист-аналитик. Проанализируй правовую ситуацию на основе предоставленных документов российского законодательства.

КОНТЕКСТ (нормы права и судебная практика):
{context_str}

ВОПРОС ПОЛЬЗОВАТЕЛЯ (описание кейса):
{query}

ИНСТРУКЦИИ:
1. Анализируй ТОЛЬКО на основе предоставленного контекста. Не используй свои внешние знания.
2. Если информации недостаточно для полного ответа, прямо укажи на это.
3. Ссылайся на конкретные документы из контекста (например: "Согласно документу 1...").
4. Не придумывай нормы права, которых нет в контексте.
5. Будь точным в формулировках.

Структура ответа:
1. ПРАВОВАЯ КВАЛИФИКАЦИЯ: Определи, к какой области права относится ситуация.
2. ПРИМЕНИМЫЕ НОРМЫ ПРАВА: Перечисли конкретные статьи и законы из контекста.
3. АНАЛИЗ: Примени нормы к фактам из вопроса.
4. РИСКИ И РЕКОМЕНДАЦИИ: Укажи возможные правовые риски и дай рекомендации.
5. ИСТОЧНИКИ: Ссылки на использованные документы.

ОТВЕТ:"""
        
        return prompt
    
    async def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Генерация ответа с использованием GigaChat"""
        token = await self._get_access_token()
        
        prompt = self._create_legal_prompt(query, context)
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.3,  # Низкая температура для более точных ответов
            'max_tokens': 2000
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        logger.info("Ответ от GigaChat успешно получен")
                        return content
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка GigaChat API: {error_text}")
                        return f"Ошибка при генерации ответа: {response.status}"
            except asyncio.TimeoutError:
                logger.error("Таймаут при запросе к GigaChat")
                return "Таймаут при запросе к модели. Попробуйте еще раз."
            except Exception as e:
                logger.error(f"Ошибка при запросе к GigaChat: {str(e)}")
                return f"Ошибка: {str(e)}"
    
    async def stream_response(self, query: str, context: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """Потоковая генерация ответа"""
        token = await self._get_access_token()
        
        prompt = self._create_legal_prompt(query, context)
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.3,
            'max_tokens': 2000,
            'stream': True
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f'{self.base_url}/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                if line_text.startswith('data: '):
                                    data = line_text[6:]
                                    if data != '[DONE]':
                                        try:
                                            chunk = json.loads(data)
                                            if 'choices' in chunk and chunk['choices']:
                                                delta = chunk['choices'][0].get('delta', {})
                                                if 'content' in delta:
                                                    yield delta['content']
                                        except json.JSONDecodeError:
                                            continue
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка GigaChat API: {error_text}")
                        yield f"Ошибка: {response.status}"
            except Exception as e:
                logger.error(f"Ошибка при потоковом запросе: {str(e)}")
                yield f"Ошибка: {str(e)}"