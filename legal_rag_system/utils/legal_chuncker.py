import re
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib

@dataclass
class LegalChunk:
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    section_type: str  # law, article, part, clause, decision_section

class LegalChunker:
    """Специализированный чанкер для юридических документов"""
    
    # Паттерны для российского законодательства
    LAW_PATTERNS = {
        'federal_law': r'Федеральный закон\s+(?:от\s+\d{1,2}\s+\w+\s+\d{4}\s+г\.\s+)?№?\s*\d+[-\w]*',
        'code': r'(?:Гражданский|Уголовный|Налоговый|Трудовой|Семейный|Арбитражный процессуальный|Гражданский процессуальный|Уголовно-процессуальный|Уголовно-исполнительный)\s+(?:кодекс|Кодекс)',
        'article': r'Статья\s+\d+(?:\.\d+)*[-\w]*\.',
        'part': r'Часть\s+\d+\.',
        'clause': r'пункт\s+\d+\.',
        'chapter': r'Глава\s+\d+\.',
    }
    
    # Паттерны для судебных решений
    COURT_DECISION_PATTERNS = {
        'fabula': r'^(?:УСТАНОВИЛ:|ФАБУЛА ДЕЛА:)',
        'circumstances': r'^(?:ОБСТОЯТЕЛЬСТВА ДЕЛА:|УСТАНОВЛЕННЫЕ ОБСТОЯТЕЛЬСТВА:)',
        'legal_basis': r'^(?:ПРАВОВЫЕ ОСНОВАНИЯ:|РУКОВОДСТВУЯСЬ:)',
        'decision': r'^(?:РЕШИЛ:|ПОСТАНОВИЛ:)',
    }
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_legal_document(self, text: str, metadata: Dict[str, Any]) -> List[LegalChunk]:
        """Основной метод для чанкинга юридических документов"""
        document_id = metadata.get("document_id", hashlib.md5(text.encode()).hexdigest())
        
        # Определяем тип документа
        if self._is_law_text(text):
            return self._chunk_law_text(text, document_id, metadata)
        elif self._is_court_decision(text):
            return self._chunk_court_decision(text, document_id, metadata)
        else:
            # Используем семантический чанкинг для остальных документов
            return self._semantic_chunking(text, document_id, metadata)
    
    def _is_law_text(self, text: str) -> bool:
        """Определяет, является ли текст законом/кодексом"""
        return any(re.search(pattern, text, re.IGNORECASE) 
                  for pattern in [self.LAW_PATTERNS['federal_law'], self.LAW_PATTERNS['code']])
    
    def _is_court_decision(self, text: str) -> bool:
        """Определяет, является ли текст судебным решением"""
        return any(re.search(pattern, text, re.IGNORECASE) 
                  for pattern in self.COURT_DECISION_PATTERNS.values())
    
    def _chunk_law_text(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[LegalChunk]:
        """Чанкинг законов и кодексов по структурным единицам"""
        chunks = []
        
        # Разбиваем на статьи
        articles = re.split(r'(Статья\s+\d+(?:\.\d+)*[-\w]*\.)', text)
        
        current_article = None
        for i, part in enumerate(articles):
            if i % 2 == 0 and i > 0:
                # Это текст статьи
                article_text = part.strip()
                if article_text and current_article:
                    # Разбиваем статью на части если она большая
                    if len(article_text) > self.chunk_size:
                        sub_chunks = self._split_by_semantic_units(article_text, "article_part")
                        for j, sub_text in enumerate(sub_chunks):
                            chunk_metadata = metadata.copy()
                            chunk_metadata.update({
                                "article": current_article,
                                "part_number": j + 1,
                                "section_type": "article_part"
                            })
                            
                            chunk = LegalChunk(
                                text=sub_text,
                                metadata=chunk_metadata,
                                chunk_id=f"{document_id}_art_{current_article}_part_{j+1}",
                                document_id=document_id,
                                section_type="article_part"
                            )
                            chunks.append(chunk)
                    else:
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "article": current_article,
                            "section_type": "article"
                        })
                        
                        chunk = LegalChunk(
                            text=article_text,
                            metadata=chunk_metadata,
                            chunk_id=f"{document_id}_art_{current_article}",
                            document_id=document_id,
                            section_type="article"
                        )
                        chunks.append(chunk)
            else:
                # Это заголовок статьи
                match = re.match(r'Статья\s+(\d+(?:\.\d+)*[-\w]*)\.', part.strip())
                if match:
                    current_article = match.group(1)
        
        return chunks
    
    def _chunk_court_decision(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[LegalChunk]:
        """Чанкинг судебных решений по секциям"""
        chunks = []
        
        # Ищем основные секции
        current_section = None
        current_text = []
        lines = text.split('\n')
        
        for line in lines:
            # Проверяем, является ли строка началом новой секции
            section_found = False
            for section_name, pattern in self.COURT_DECISION_PATTERNS.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Сохраняем предыдущую секцию
                    if current_section and current_text:
                        chunk_text = '\n'.join(current_text).strip()
                        if chunk_text:
                            chunk_metadata = metadata.copy()
                            chunk_metadata.update({
                                "section": current_section,
                                "section_type": "court_decision"
                            })
                            
                            chunk = LegalChunk(
                                text=chunk_text,
                                metadata=chunk_metadata,
                                chunk_id=f"{document_id}_{current_section}",
                                document_id=document_id,
                                section_type="court_decision"
                            )
                            chunks.append(chunk)
                    
                    # Начинаем новую секцию
                    current_section = section_name
                    current_text = [line]
                    section_found = True
                    break
            
            if not section_found:
                if current_section:
                    current_text.append(line)
        
        # Добавляем последнюю секцию
        if current_section and current_text:
            chunk_text = '\n'.join(current_text).strip()
            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "section": current_section,
                    "section_type": "court_decision"
                })
                
                chunk = LegalChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=f"{document_id}_{current_section}",
                    document_id=document_id,
                    section_type="court_decision"
                )
                chunks.append(chunk)
        
        return chunks
    
    def _semantic_chunking(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[LegalChunk]:
        """Семантический чанкинг для документов без четкой структуры"""
        chunks = []
        
        # Разбиваем по предложениям
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "section_type": "semantic_chunk"
                    })
                    
                    chunk = LegalChunk(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        chunk_id=f"{document_id}_chunk_{len(chunks)}",
                        document_id=document_id,
                        section_type="semantic_chunk"
                    )
                    chunks.append(chunk)
                
                # Начинаем новый чанк с перекрытием
                current_chunk = current_chunk[-self._get_overlap_sentences():] + [sentence]
                current_length = sum(len(s) for s in current_chunk)
        
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "section_type": "semantic_chunk"
            })
            
            chunk = LegalChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id,
                section_type="semantic_chunk"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_semantic_units(self, text: str, unit_type: str) -> List[str]:
        """Разбивает текст на семантические единицы"""
        if unit_type == "article_part":
            # Разбиваем статью на части и пункты
            parts = re.split(r'(Часть\s+\d+\.|пункт\s+\d+\.)', text)
            result = []
            current_part = []
            
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if current_part:
                        result.append(''.join(current_part).strip())
                        current_part = []
                else:
                    current_part.append(part)
                    if i + 1 < len(parts):
                        current_part.append(parts[i + 1])
            
            if current_part:
                result.append(''.join(current_part).strip())
            
            return result if result else [text]
        
        return [text]
    
    def _get_overlap_sentences(self) -> int:
        """Вычисляет количество предложений для перекрытия"""
        return max(1, int(self.chunk_overlap / 50))  # Предполагаем среднюю длину предложения 50 символов