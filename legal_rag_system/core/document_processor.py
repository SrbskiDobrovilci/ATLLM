import os
import hashlib
from typing import List, Dict, Any
from loguru import logger

from utils.legal_chunker import LegalChunker, LegalChunk
from utils.text_processor import TextProcessor

class DocumentProcessor:
    """Обработчик документов для Legal RAG системы"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunker = LegalChunker(chunk_size, chunk_overlap)
        self.text_processor = TextProcessor()
    
    def process_document(self, file_path: str) -> List[LegalChunk]:
        """Обработка документа из файла"""
        logger.info(f"Обработка документа: {file_path}")
        
        # Извлекаем текст
        if file_path.lower().endswith('.pdf'):
            text = self.text_processor.extract_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Нормализуем текст
        text = self.text_processor.normalize_legal_text(text)
        
        # Извлекаем метаданные
        metadata = self.text_processor.extract_metadata_from_text(text)
        metadata.update({
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "document_id": hashlib.md5(text.encode()).hexdigest()[:16]
        })
        
        # Чанкинг документа
        chunks = self.chunker.chunk_legal_document(text, metadata)
        
        logger.info(f"Документ разбит на {len(chunks)} чанков")
        return chunks
    
    def process_directory(self, directory_path: str) -> List[LegalChunk]:
        """Обработка всех документов в директории"""
        all_chunks = []
        
        supported_extensions = ['.pdf', '.txt', '.docx', '.doc']
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        chunks = self.process_document(file_path)
                        all_chunks.extend(chunks)
                        logger.info(f"Успешно обработан: {file}")
                    except Exception as e:
                        logger.error(f"Ошибка при обработке {file}: {str(e)}")
        
        return all_chunks