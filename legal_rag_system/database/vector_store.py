from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    ScrollRequest, SearchParams, SearchRequest
)
from loguru import logger

from config import settings
from utils.legal_chunker import LegalChunk
from core.embedder import SBertEmbedder

class VectorStore:
    """Клиент для работы с векторной базой данных Qdrant"""
    
    def __init__(self, embedder: SBertEmbedder = None):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60
        )
        self.embedder = embedder or SBertEmbedder()
        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Создание коллекции, если она не существует"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Создание коллекции: {self.collection_name}")
            
            # Получаем размерность эмбеддингов
            dim = self.embedder.get_embedding_dimension()
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Коллекция {self.collection_name} создана с размерностью {dim}")
        else:
            logger.info(f"Коллекция {self.collection_name} уже существует")
    
    def add_documents(self, chunks: List[LegalChunk]) -> List[str]:
        """Добавление документов в векторную базу данных"""
        if not chunks:
            return []
        
        logger.info(f"Добавление {len(chunks)} чанков в векторную БД")
        
        points = []
        texts = [chunk.text for chunk in chunks]
        
        # Создаем эмбеддинги
        embeddings = self.embedder.encode(texts)
        
        for i, chunk in enumerate(chunks):
            point = PointStruct(
                id=i,  # В реальной системе нужно использовать уникальные ID
                vector=embeddings[i].tolist(),
                payload={
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "section_type": chunk.section_type,
                    **chunk.metadata
                }
            )
            points.append(point)
        
        # Загружаем точки в Qdrant
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Успешно добавлено {len(points)} точек. Статус: {operation_info.status}")
        return [chunk.chunk_id for chunk in chunks]
    
    def search(self, query: str, top_k: int = None, filter_condition: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Поиск похожих документов по запросу"""
        top_k = top_k or settings.TOP_K_RESULTS
        
        # Создаем эмбеддинг для запроса
        query_embedding = self.embedder.encode_single(query)
        
        # Строим фильтр если нужно
        search_filter = None
        if filter_condition:
            field_conditions = []
            for field, value in filter_condition.items():
                field_conditions.append(FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                ))
            search_filter = Filter(must=field_conditions)
        
        # Выполняем поиск
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=search_filter,
            limit=top_k,
            search_params=SearchParams(
                hnsw_ef=128,
                exact=False
            )
        )
        
        # Форматируем результаты
        results = []
        for hit in search_result:
            if hit.score >= settings.SIMILARITY_THRESHOLD:
                result = {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
                }
                results.append(result)
        
        logger.info(f"Найдено {len(results)} релевантных документов для запроса")
        return results
    
    def hybrid_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Гибридный поиск: семантический + ключевой поиск по важным терминам"""
        # Извлекаем ключевые термины из запроса
        keywords = self._extract_keywords(query)
        
        all_results = []
        
        # Семантический поиск
        semantic_results = self.search(query, top_k=top_k * 2 if top_k else 10)
        all_results.extend(semantic_results)
        
        # Ключевой поиск по каждому термину
        for keyword in keywords:
            keyword_results = self.search(keyword, top_k=3)
            all_results.extend(keyword_results)
        
        # Удаляем дубликаты и сортируем по релевантности
        unique_results = {}
        for result in all_results:
            text = result["text"]
            if text not in unique_results or result["score"] > unique_results[text]["score"]:
                unique_results[text] = result
        
        # Сортируем по score и возвращаем топ-K
        sorted_results = sorted(
            unique_results.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )
        
        return sorted_results[:top_k] if top_k else sorted_results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Извлечение ключевых терминов из запроса"""
        # Ищем юридические термины, номера статей и законы
        patterns = [
            r'Статья\s+\d+',
            r'ст\.\s*\d+',
            r'ГК\s*РФ',
            r'УК\s*РФ',
            r'НК\s*РФ',
            r'ТК\s*РФ',
            r'Федеральный закон',
            r'№\s*\d+',
        ]
        
        keywords = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            keywords.extend(matches)
        
        # Добавляем слова длиннее 4 символов (предполагая, что это юридические термины)
        words = query.split()
        keywords.extend([word for word in words if len(word) > 4 and word.isalpha()])
        
        return list(set(keywords))
    
    def delete_collection(self):
        """Удаление коллекции"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Коллекция {self.collection_name} удалена")