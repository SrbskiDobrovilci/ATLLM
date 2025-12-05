import re
import string
from typing import List, Dict, Any
import PyPDF2
import fitz  # PyMuPDF

class TextProcessor:
    """Обработка текста из различных форматов"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Очистка текста от лишних пробелов и символов"""
        # Удаляем множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        # Удаляем специальные символы, но сохраняем пунктуацию
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}«»"-]', '', text)
        return text.strip()
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Извлечение текста из PDF файла"""
        text = ""
        
        try:
            # Сначала пробуем PyMuPDF для лучшей поддержки русского языка
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            # Если PyMuPDF не сработал, используем PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            except Exception as e2:
                raise Exception(f"Ошибка при чтении PDF: {str(e2)}")
        
        return TextProcessor.clean_text(text)
    
    @staticmethod
    def normalize_legal_text(text: str) -> str:
        """Нормализация юридического текста"""
        # Стандартизируем кавычки
        text = text.replace('``', '«').replace("''", '»')
        text = text.replace('"', '«')
        
        # Стандартизируем номера статей
        text = re.sub(r'ст\.\s*(\d+)', r'Статья \1', text, flags=re.IGNORECASE)
        text = re.sub(r'№\s*', '№', text)
        
        return text
    
    @staticmethod
    def extract_metadata_from_text(text: str) -> Dict[str, Any]:
        """Извлечение метаданных из текста документа"""
        metadata = {
            "document_type": "unknown",
            "law_number": None,
            "date": None,
            "title": None
        }
        
        # Пытаемся определить тип документа
        if re.search(r'Федеральный закон', text, re.IGNORECASE):
            metadata["document_type"] = "federal_law"
            # Ищем номер закона
            match = re.search(r'Федеральный закон\s+(?:от\s+[^№]*)?(№\s*\d+[-\w]*)', text, re.IGNORECASE)
            if match:
                metadata["law_number"] = match.group(1)
        elif re.search(r'кодекс', text, re.IGNORECASE):
            metadata["document_type"] = "code"
        elif re.search(r'РЕШЕНИЕ|ПОСТАНОВЛЕНИЕ|ОПРЕДЕЛЕНИЕ', text):
            metadata["document_type"] = "court_decision"
        
        # Ищем дату
        date_patterns = [
            r'от\s+(\d{1,2}\s+\w+\s+\d{4}\s+г\.)',
            r'(\d{1,2}[./]\d{1,2}[./]\d{4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["date"] = match.group(1)
                break
        
        # Извлекаем заголовок (первые 200 символов)
        lines = text.split('\n')
        for line in lines[:5]:
            if len(line.strip()) > 20:
                metadata["title"] = line.strip()[:200]
                break
        
        return metadata