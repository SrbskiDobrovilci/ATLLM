from typing import List, Dict, Any
from loguru import logger

from services.gigachat_service import GigaChatService
from database.vector_store import VectorStore

class ResponseGenerator:
    """Генератор ответов для Legal RAG системы"""
    
    def __init__(self, vector_store: VectorStore, gigachat_service: GigaChatService = None):
        self.vector_store = vector_store
        self.gigachat_service = gigachat_service or GigaChatService()
    
    async def generate_legal_analysis(self, query: str, use_hybrid: bool = True) -> Dict[str, Any]:
        """Генерация юридического анализа на основе запроса"""
        logger.info(f"Генерация анализа для запроса: {query}")
        
        # Поиск релевантных документов
        if use_hybrid:
            context_docs = self.vector_store.hybrid_search(query)
        else:
            context_docs = self.vector_store.search(query)
        
        if not context_docs:
            return {
                "answer": "По вашему запросу не найдено релевантных правовых документов в базе.",
                "sources": [],
                "context": []
            }
        
        # Генерация ответа с использованием GigaChat
        answer = await self.gigachat_service.generate_response(query, context_docs)
        
        # Форматирование источников
        sources = []
        for i, doc in enumerate(context_docs, 1):
            metadata = doc.get("metadata", {})
            source_info = {
                "id": i,
                "score": doc["score"],
                "document_type": metadata.get("document_type", "Неизвестно"),
                "title": metadata.get("title", metadata.get("file_name", "Без названия")),
                "article": metadata.get("article"),
                "section": metadata.get("section")
            }
            sources.append(source_info)
        
        # Извлечение использованных статей из ответа
        used_articles = self._extract_used_articles(answer, context_docs)
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context_docs,
            "used_articles": used_articles,
            "query": query
        }
    
    def _extract_used_articles(self, answer: str, context_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Извлечение использованных статей из ответа"""
        articles = []
        
        # Ищем ссылки на статьи в ответе
        import re
        article_patterns = [
            r'Статья\s+(\d+(?:\.\d+)*)',
            r'ст\.\s*(\d+(?:\.\d+)*)',
            r'по\s+статье\s+(\d+)',
        ]
        
        found_articles = []
        for pattern in article_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            found_articles.extend(matches)
        
        # Ищем соответствующие статьи в контексте
        for article_num in set(found_articles):
            for doc in context_docs:
                metadata = doc.get("metadata", {})
                if metadata.get("article") == article_num:
                    articles.append({
                        "article": article_num,
                        "document_type": metadata.get("document_type"),
                        "title": metadata.get("title", ""),
                        "text_preview": doc["text"][:200] + "..."
                    })
                    break
        
        return articles