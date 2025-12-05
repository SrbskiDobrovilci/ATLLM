import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import settings

class SBertEmbedder:
    """Класс для создания эмбеддингов с использованием sbert_large_nlu_ru"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели Sentence Transformers"""
        logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            # Тестируем модель
            test_embedding = self.model.encode(["тестовый текст"])
            logger.info(f"Модель успешно загружена. Размер эмбеддинга: {test_embedding.shape}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {str(e)}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Создание эмбеддингов для списка текстов"""
        if not self.model:
            self._load_model()
        
        logger.debug(f"Создание эмбеддингов для {len(texts)} текстов")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Создание эмбеддинга для одного текста"""
        return self.encode([text])[0]
    
    def get_embedding_dimension(self) -> int:
        """Получение размерности эмбеддингов"""
        if not self.model:
            self._load_model()
        # Получаем размерность из тестового эмбеддинга
        test_embedding = self.model.encode(["тест"])
        return test_embedding.shape[1]