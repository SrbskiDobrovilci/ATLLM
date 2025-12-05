import streamlit as st
import asyncio
import uuid
from typing import List, Dict, Any
import tempfile
import os
from datetime import datetime

from config import settings
from core.document_processor import DocumentProcessor
from core.embedder import SBertEmbedder
from database.vector_store import VectorStore
from core.generator import ResponseGenerator
from services.gigachat_service import GigaChatService
from core.memory_manager import MemoryManager

# Настройки страницы
st.set_page_config(
    page_title="Legal RAG Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация сессионных состояний
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "generator" not in st.session_state:
    st.session_state.generator = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

class LegalRAGApp:
    def __init__(self):
        self.initialize_services()
    
    def initialize_services(self):
        """Инициализация всех сервисов"""
        try:
            with st.spinner("Инициализация сервисов..."):
                # Инициализация эмбеддера
                embedder = SBertEmbedder()
                
                # Инициализация векторного хранилища
                vector_store = VectorStore(embedder)
                st.session_state.vector_store = vector_store
                
                # Инициализация GigaChat сервиса
                gigachat_service = GigaChatService()
                
                # Инициализация генератора
                generator = ResponseGenerator(vector_store, gigachat_service)
                st.session_state.generator = generator
                
                st.success("Сервисы успешно инициализированы!")
        except Exception as e:
            st.error(f"Ошибка инициализации: {str(e)}")
    
    def load_documents(self, uploaded_files):
        """Загрузка документов в систему"""
        if not uploaded_files:
            st.warning("Пожалуйста, загрузите документы")
            return
        
        processor = DocumentProcessor()
        
        with st.spinner("Обработка документов..."):
            all_chunks = []
            
            for uploaded_file in uploaded_files:
                # Сохраняем файл во временную директорию
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Обрабатываем документ
                    chunks = processor.process_document(tmp_path)
                    all_chunks.extend(chunks)
                    st.success(f"Обработан: {uploaded_file.name} ({len(chunks)} фрагментов)")
                except Exception as e:
                    st.error(f"Ошибка обработки {uploaded_file.name}: {str(e)}")
                finally:
                    # Удаляем временный файл
                    os.unlink(tmp_path)
            
            if all_chunks:
                # Добавляем в векторную базу данных
                vector_store = st.session_state.vector_store
                chunk_ids = vector_store.add_documents(all_chunks)
                st.session_state.documents_loaded = True
                st.success(f"✅ Загружено {len(chunk_ids)} фрагментов из {len(uploaded_files)} документов")
    
    def display_chat_message(self, role, content, metadata=None):
        """Отображение сообщения в чате"""
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)
                
                if metadata and "sources" in metadata:
                    with st.expander("📚 Источники", expanded=False):
                        for source in metadata["sources"]:
                            st.markdown(f"**{source['id']}. {source['title']}**")
                            if source.get("article"):
                                st.caption(f"Статья: {source['article']}")
                            st.caption(f"Релевантность: {source['score']:.3f}")
                            st.divider()
    
    async def process_query(self, query: str):
        """Обработка запроса пользователя"""
        if not st.session_state.documents_loaded:
            st.error("Пожалуйста, сначала загрузите юридические документы")
            return
        
        generator = st.session_state.generator
        
        # Сохраняем запрос пользователя в историю
        st.session_state.memory.add_message(
            st.session_state.session_id,
            "user",
            query
        )
        
        # Отображаем запрос пользователя
        self.display_chat_message("user", query)
        
        # Получаем контекст из истории
        conversation_context = st.session_state.memory.get_conversation_context(
            st.session_state.session_id
        )
        
        # Формируем полный запрос с историей
        full_query = query
        if conversation_context:
            full_query = f"Контекст предыдущего разговора:\n{conversation_context}\n\nТекущий вопрос: {query}"
        
        # Генерация ответа
        with st.spinner("Анализирую правовую ситуацию..."):
            try:
                # Получаем ответ
                response_data = await generator.generate_legal_analysis(full_query)
                answer = response_data["answer"]
                sources = response_data["sources"]
                
                # Сохраняем ответ ассистента в историю
                st.session_state.memory.add_message(
                    st.session_state.session_id,
                    "assistant",
                    answer,
                    {"sources": sources}
                )
                
                # Отображаем ответ
                self.display_chat_message("assistant", answer, {"sources": sources})
                
            except Exception as e:
                error_msg = f"Ошибка при обработке запроса: {str(e)}"
                st.error(error_msg)
    
    def run(self):
        """Запуск основного приложения"""
        # Заголовок
        st.title("⚖️ Legal RAG Analyzer")
        st.markdown("Система анализа юридических кейсов на основе российского законодательства")
        
        # Сайдбар
        with st.sidebar:
            st.header("📁 Загрузка документов")
            
            uploaded_files = st.file_uploader(
                "Загрузите PDF документы",
                type=['pdf'],
                accept_multiple_files=True
            )
            
            if st.button("Загрузить документы", type="primary"):
                self.load_documents(uploaded_files)
            
            st.divider()
            
            st.header("⚙️ Настройки")
            
            # Настройки поиска
            top_k = st.slider("Количество релевантных документов", 1, 10, 5)
            settings.TOP_K_RESULTS = top_k
            
            similarity_threshold = st.slider("Порог схожести", 0.1, 1.0, 0.7, 0.05)
            settings.SIMILARITY_THRESHOLD = similarity_threshold
            
            use_hybrid = st.checkbox("Использовать гибридный поиск", value=True)
            
            st.divider()
            
            # Кнопка очистки истории
            if st.button("🧹 Очистить историю диалога"):
                st.session_state.memory.clear_history(st.session_state.session_id)
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
            
            # Информация о системе
            st.caption(f"Сессия: {st.session_state.session_id[:8]}")
            st.caption(f"Модель эмбеддингов: {settings.EMBEDDING_MODEL}")
            st.caption(f"LLM: GigaChat")
        
        # Основная область
        tab1, tab2 = st.tabs(["💬 Чат-анализ", "📊 Информация о системе"])
        
        with tab1:
            # Отображение истории чата
            history = st.session_state.memory.get_history(st.session_state.session_id)
            for msg in history:
                self.display_chat_message(
                    msg["role"],
                    msg["content"],
                    msg.get("metadata")
                )
            
            # Поле ввода запроса
            query = st.chat_input("Опишите ваш юридический кейс...")
            
            if query:
                # Запускаем асинхронную обработку
                asyncio.run(self.process_query(query))
        
        with tab2:
            st.header("Информация о системе")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Архитектура")
                st.markdown("""
                - **Модульный RAG**: Каждый компонент независим и заменяем
                - **Гибридный поиск**: Семантический + ключевой поиск
                - **Специализированный чанкинг**: Для юридических документов
                - **Контекстная память**: Хранение истории диалога
                """)
            
            with col2:
                st.subheader("Технологии")
                st.markdown("""
                - **Модель эмбеддингов**: sberbank-ai/sbert_large_nlu_ru
                - **LLM**: GigaChat API
                - **Векторная БД**: Qdrant
                - **Фронтенд**: Streamlit
                - **Хранилище памяти**: Redis
                """)
            
            st.subheader("Как это работает")
            st.markdown("""
            1. **Загрузка документов**: PDF документы разбиваются на смысловые фрагменты
            2. **Векторизация**: Каждый фрагмент преобразуется в вектор с помощью SBERT
            3. **Поиск**: По запросу находятся наиболее релевантные фрагменты
            4. **Анализ**: GigaChat анализирует найденные документы и генерирует ответ
            5. **Ответ**: Структурированный анализ с ссылками на источники
            """)
            
            if st.session_state.documents_loaded:
                st.success("✅ Документы загружены в систему")
            else:
                st.warning("⚠️ Документы не загружены. Пожалуйста, загрузите PDF файлы.")

def main():
    app = LegalRAGApp()
    app.run()

if __name__ == "__main__":
    main()