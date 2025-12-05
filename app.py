import os
import uuid
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
import requests
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import hashlib
from dotenv import load_dotenv
import urllib3
from gigachat import GigaChat


urllib3.disable_warnings()
# Load environment variables
load_dotenv()

# Configuration
class Config:
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///gigachat.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # GigaChat Configuration
    GIGACHAT_CLIENT_ID = os.environ.get('GIGACHAT_CLIENT_ID')
    GIGACHAT_CLIENT_SECRET = os.environ.get('GIGACHAT_CLIENT_SECRET')
    GIGACHAT_SCOPE = os.environ.get('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')
    GIGACHAT_AUTH_URL = os.environ.get('GIGACHAT_AUTH_URL', 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth')
    GIGACHAT_API_URL = os.environ.get('GIGACHAT_API_URL', 'https://gigachat.devices.sberbank.ru/api/v1')
    GIGACHAT_API_AUTH_KEY = os.environ.get('GIGACHAT_API_AUTH_KEY')

    # RAG Configuration
    RAG_ENABLED = os.environ.get('RAG_ENABLED', 'true').lower() == 'true'
    DOCUMENTS_PATH = os.environ.get('DOCUMENTS_PATH', 'documents')
    VECTOR_STORE_PATH = os.environ.get('VECTOR_STORE_PATH', 'vector_store')
    DATA_PATH = os.environ.get('DATA_PATH', 'data')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 200))
    MAX_RETRIEVAL_CHUNKS = int(os.environ.get('MAX_RETRIEVAL_CHUNKS', 3))
    
    # Application Configuration
    MAX_CONTEXT_CHATS = 5
    MAX_MESSAGES_PER_CHAT = 50
    MAX_CONVERSATION_TOKENS = 4000

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)


# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    conversations = db.relationship('Conversation', backref='user', lazy=True, cascade='all, delete-orphan')

class Conversation(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default='New Conversation')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('Message', backref='conversation', lazy=True, cascade='all, delete-orphan')
    use_rag = db.Column(db.Boolean, default=True)  # Whether to use RAG for this conversation

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversation.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    tokens = db.Column(db.Integer, default=0)
    context_used = db.Column(db.Boolean, default=False)  # Whether RAG context was used
    context_sources = db.Column(db.Text)  # JSON string of context sources

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# RAG System Implementation
class LegalDocumentRAG:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG system for legal documents.
        """
        self.config = config
        self.documents_path = Path(config.get('documents_path', 'documents'))
        self.vector_store_path = Path(config.get('vector_store_path', 'vector_store'))
        self.data_path = Path(config.get('data_path', 'data'))
        
        # Create directories if they don't exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get('chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap', 200),
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            config.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        )
        
        # Initialize storage
        self.index = None
        self.document_chunks = []
        self.chunk_metadata = []
        
        # Load or create vector store
        self.load_or_create_vector_store()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file.
        """
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text += page.get_text()
                    text += f"\n--- Page {page_num + 1} ---\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        
        return text
    
    def process_documents(self) -> Tuple[List[str], List[Dict]]:
        """
        Process all PDF documents in the documents folder.
        """
        all_chunks = []
        all_metadata = []
        
        pdf_files = list(self.documents_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.documents_path}")
            return all_chunks, all_metadata
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            if not text.strip():
                print(f"No text extracted from {pdf_file.name}")
                continue
            
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "file_path": str(pdf_file),
                    "document_type": self.identify_document_type(pdf_file.name),
                    "processing_date": datetime.now().isoformat()
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            for chunk in chunks:
                # Clean chunk text
                cleaned_text = self.clean_legal_text(chunk.page_content)
                if cleaned_text.strip():
                    all_chunks.append(cleaned_text)
                    all_metadata.append({
                        **chunk.metadata,
                        "chunk_id": hashlib.md5(cleaned_text.encode()).hexdigest(),
                        "text_length": len(cleaned_text)
                    })
            
            print(f"  Extracted {len(chunks)} chunks from {pdf_file.name}")
        
        print(f"Total chunks extracted: {len(all_chunks)}")
        return all_chunks, all_metadata
    
    def identify_document_type(self, filename: str) -> str:
        """
        Identify type of legal document based on filename.
        """
        filename_lower = filename.lower()
        
        if "граждан" in filename_lower:
            return "Гражданский кодекс"
        elif "уголов" in filename_lower:
            return "Уголовный кодекс"
        elif "налог" in filename_lower:
            return "Налоговый кодекс"
        elif "трудов" in filename_lower:
            return "Трудовой кодекс"
        elif "семей" in filename_lower:
            return "Семейный кодекс"
        elif "арбитраж" in filename_lower:
            return "Арбитражный кодекс"
        elif "административ" in filename_lower:
            return "Кодекс об административных правонарушениях"
        elif "бюджет" in filename_lower:
            return "Бюджетный кодекс"
        elif "земель" in filename_lower:
            return "Земельный кодекс"
        elif "вод" in filename_lower:
            return "Водный кодекс"
        elif "лес" in filename_lower:
            return "Лесной кодекс"
        elif "конституция" in filename_lower or "конституц" in filename_lower:
            return "Конституция РФ"
        elif "федеральн" in filename_lower and "закон" in filename_lower:
            return "Федеральный закон"
        elif "постановление" in filename_lower:
            return "Постановление Правительства"
        elif "приказ" in filename_lower:
            return "Приказ министерства"
        else:
            return "Юридический документ"
    
    def clean_legal_text(self, text: str) -> str:
        """
        Clean legal text by removing excessive whitespace and normalizing.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = cleaned_text.replace('--- Page', '')
        
        return cleaned_text
    
    def create_vector_store(self, chunks: List[str], metadata: List[Dict]) -> None:
        """
        Create FAISS vector store from document chunks.
        """
        print("Creating embeddings...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Save document chunks and metadata
        self.document_chunks = chunks
        self.chunk_metadata = metadata
        
        print(f"Created index with {len(chunks)} chunks")
        
        # Save to disk
        self.save_vector_store()
    
    def save_vector_store(self) -> None:
        """Save vector store to disk."""
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, str(self.vector_store_path / "index.faiss"))
            
            # Save chunks and metadata
            with open(self.data_path / "chunks.pkl", "wb") as f:
                pickle.dump(self.document_chunks, f)
            
            with open(self.data_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.chunk_metadata, f)
            
            # Save configuration
            config_data = {
                "embedding_model": self.config.get('embedding_model'),
                "chunk_size": self.config.get('chunk_size'),
                "chunk_overlap": self.config.get('chunk_overlap'),
                "total_chunks": len(self.document_chunks),
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.data_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            print("Vector store saved successfully")
    
    def load_vector_store(self) -> bool:
        """
        Load vector store from disk.
        """
        index_path = self.vector_store_path / "index.faiss"
        chunks_path = self.data_path / "chunks.pkl"
        metadata_path = self.data_path / "metadata.pkl"
        
        if all(p.exists() for p in [index_path, chunks_path, metadata_path]):
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load chunks and metadata
                with open(chunks_path, "rb") as f:
                    self.document_chunks = pickle.load(f)
                
                with open(metadata_path, "rb") as f:
                    self.chunk_metadata = pickle.load(f)
                
                print(f"Loaded vector store with {len(self.document_chunks)} chunks")
                return True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return False
        else:
            print("Vector store files not found")
            return False
    
    def load_or_create_vector_store(self) -> None:
        """Load existing vector store or create new one."""
        if not self.load_vector_store():
            print("Creating new vector store...")
            chunks, metadata = self.process_documents()
            if chunks:
                self.create_vector_store(chunks, metadata)
            else:
                print("No documents processed. RAG system will not be available.")
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar document chunks to the query.
        """
        if self.index is None or not self.document_chunks:
            print("Vector store not initialized")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.document_chunks):
                result = {
                    "chunk": self.document_chunks[idx],
                    "metadata": self.chunk_metadata[idx],
                    "similarity_score": 1.0 / (1.0 + distance),
                    "rank": i + 1
                }
                results.append(result)
        
        return results
    
    def retrieve_relevant_context(self, query: str, max_chunks: int = 3) -> str:
        """
        Retrieve relevant context for a query.
        """
        similar_chunks = self.search_similar_chunks(query, k=max_chunks)
        
        if not similar_chunks:
            return ""
        
        context_parts = []
        
        for i, chunk_data in enumerate(similar_chunks):
            chunk = chunk_data["chunk"]
            metadata = chunk_data["metadata"]
            similarity = chunk_data["similarity_score"]
            
            # Format chunk with metadata
            source = metadata.get("source", "Unknown")
            doc_type = metadata.get("document_type", "legal_document")
            
            context_part = f"[Источник: {source} | Тип: {doc_type} | Релевантность: {similarity:.2f}]\n{chunk}"
            context_parts.append(context_part)
        
        # Join with separators
        context = "\n\n---\n\n".join(context_parts)
        
        return context
    
    def format_prompt_with_context(self, query: str, context: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """
        Format prompt with RAG context for GigaChat.
        """
        messages = []
        
        # Combine system prompt with context
        system_prompt = """Вы - опытный юридический ассистент, специализирующийся на законодательстве Российской Федерации. 
Ваша задача - предоставлять точные, актуальные и полезные ответы на основе российского законодательства.

ПРАВИЛА ОТВЕТА:
1. Всегда ссылайтесь на конкретные законы, кодексы или нормативные акты
2. Указывайте статьи, пункты и части, если они известны из контекста
3. Если информация основана на предоставленном контексте, укажите это
4. Будьте точны и избегайте двусмысленности
5. В сложных случаях рекомендуйте обратиться к профессиональному юристу
6. Сохраняйте профессиональный и формальный тон
7. Отвечайте на русском языке
8. Если информация не найдена в законодательстве, честно признайте это"""
        
        # Add context if available
        if context:
            system_prompt += f"\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ ЗАКОНОДАТЕЛЬСТВА РФ:\n{context}"
            system_prompt += "\n\nОТВЕТЬТЕ НА ВОПРОС ПОЛЬЗОВАТЕЛЯ НА ОСНОВАНИИ ВЫШЕПРИВЕДЕННОЙ ИНФОРМАЦИИ:"
        
        # System message must be first
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation history if available (excluding any system messages)
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                if msg["role"] != "system":
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def get_vector_store_info(self) -> Dict:
        """
        Get information about the vector store.
        """
        if not self.document_chunks:
            return {"status": "not_initialized"}
        
        # Count documents by type
        doc_types = {}
        for metadata in self.chunk_metadata:
            doc_type = metadata.get("document_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            "status": "initialized",
            "total_chunks": len(self.document_chunks),
            "document_types": doc_types,
            "embedding_model": self.config.get('embedding_model')
        }
    
    def update_documents(self) -> bool:
        """
        Update vector store with new documents.
        """
        print("Updating vector store with new documents...")
        
        # Process all documents (including new ones)
        chunks, metadata = self.process_documents()
        
        if chunks:
            # Create new vector store
            self.create_vector_store(chunks, metadata)
            return True
        
        return False

# Initialize RAG System
rag_config = {
    'documents_path': app.config['DOCUMENTS_PATH'],
    'vector_store_path': app.config['VECTOR_STORE_PATH'],
    'data_path': app.config['DATA_PATH'],
    'embedding_model': app.config['EMBEDDING_MODEL'],
    'chunk_size': app.config['CHUNK_SIZE'],
    'chunk_overlap': app.config['CHUNK_OVERLAP']
}

rag_system = None
if app.config['RAG_ENABLED']:
    rag_system = LegalDocumentRAG(rag_config)

# Enhanced GigaChat Client with RAG
class EnhancedGigaChatClient:
    def __init__(self, client_id, client_secret, scope, auth_url, auth_key, api_url, rag_system=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self.auth_url = auth_url
        self.auth_key = auth_key
        self.api_url = api_url
        self.access_token = None
        self.token_expires = None
        self.rag_system = rag_system
    
    def get_access_token(self):
        """Get access token for GigaChat API"""
        try:
            giga = GigaChat(
            credentials=self.auth_key,
            scope=self.scope,
            model="GigaChat",
            ca_bundle_file="russian_trusted_root_ca_pem.crt"
            )

            response = giga.get_token()
            
            if response.access_token != None:
                self.access_token = response.access_token
                return self.access_token
            else:
                print(f"Token request failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting access token: {e}")
            return None
    
    def send_message_with_rag(self, query, conversation_history=None, use_rag=True, max_tokens=2048, temperature=0.7):
        """Send message to GigaChat with RAG context"""
        if not self.access_token:
            if not self.get_access_token():
                return None
        
        # Retrieve context from RAG if enabled
        context = ""
        context_sources = []
        
        if use_rag and self.rag_system:
            context = self.rag_system.retrieve_relevant_context(
                query, 
                max_chunks=app.config['MAX_RETRIEVAL_CHUNKS']
            )
            
            if context:
                # Get sources for logging
                similar_chunks = self.rag_system.search_similar_chunks(query, k=app.config['MAX_RETRIEVAL_CHUNKS'])
                context_sources = [
                    {
                        "source": chunk["metadata"].get("source"),
                        "document_type": chunk["metadata"].get("document_type"),
                        "similarity": float(chunk["similarity_score"])
                    }
                    for chunk in similar_chunks
                ]
        
        # Build messages array properly
        messages = []
        
        # Create the main system prompt with context
        system_prompt = """Вы - опытный юридический ассистент, специализирующийся на законодательстве Российской Федерации. 
Ваша задача - предоставлять точные, актуальные и полезные ответы на основе российского законодательства.

ПРАВИЛА ОТВЕТА:
1. Всегда ссылайтесь на конкретные законы, кодексы или нормативные акты
2. Указывайте статьи, пункты и части, если они известны из контекста
3. Если информация основана на предоставленном контексте, укажите это
4. Будьте точны и избегайте двусмысленности
5. В сложных случаях рекомендуйте обратиться к профессиональному юристу
6. Сохраняйте профессиональный и формальный тон
7. Отвечайте на русском языке
8. Если информация не найдена в законодательстве, честно признайте это"""

        # Add context to system prompt if available
        if context:
            system_prompt += f"\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ ЗАКОНОДАТЕЛЬСТВА РФ:\n{context}"
            system_prompt += "\n\nОТВЕТЬТЕ НА ВОПРОС ПОЛЬЗОВАТЕЛЯ НА ОСНОВАНИИ ВЫШЕПРИВЕДЕННОЙ ИНФОРМАЦИИ:"
        
        # System message must be first if present
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation history if available
        if conversation_history:
            # Filter out any system messages from history to avoid duplicates
            for msg in conversation_history[-6:]:  # Last 6 messages for context
                if msg["role"] != "system":
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Debug: print messages structure
        print(f"Sending {len(messages)} messages to GigaChat")
        for i, msg in enumerate(messages):
            print(f"Message {i}: role={msg['role']}, content preview={msg['content'][:100]}...")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'GigaChat',
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                verify=False,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                result['context_used'] = bool(context)
                result['context_sources'] = context_sources
                return result
            elif response.status_code == 422:
                # Try alternative message format
                print("Received 422 error, trying alternative format...")
                return self.send_message_alternative_format(query, context, conversation_history, context_sources)
            elif response.status_code == 401:  # Token expired
                print("Token expired, refreshing...")
                self.get_access_token()
                headers['Authorization'] = f'Bearer {self.access_token}'
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    result['context_used'] = bool(context)
                    result['context_sources'] = context_sources
                    return result
            
            print(f"API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        except requests.exceptions.Timeout:
            print("GigaChat API request timeout")
            return None
        except Exception as e:
            print(f"Error sending message: {e}")
            return None
    
    def send_message_alternative_format(self, query, context, conversation_history, context_sources):
        """Alternative message format for GigaChat API"""
        messages = []
        
        # Build system message differently
        if context:
            # Include context in the system message
            system_content = f"""Вы - юридический ассистент. Отвечайте на основе предоставленной информации из законодательства РФ.

ИНФОРМАЦИЯ ИЗ ЗАКОНОДАТЕЛЬСТВА:
{context}

ОТВЕТЬТЕ НА ВОПРОС:"""
        else:
            system_content = """Вы - юридический ассистент. Отвечайте на вопросы о законодательстве РФ."""
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history[-6:]:
                if msg["role"] != "system":
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": query
        })
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'GigaChat',
            'messages': messages,
            'max_tokens': 2048,
            'temperature': 0.7,
            'stream': False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                verify=False,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                result['context_used'] = bool(context)
                result['context_sources'] = context_sources
                return result
            else:
                print(f"Alternative format also failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"Error in alternative format: {e}")
            return None
        
# Initialize enhanced GigaChat client
giga_client = EnhancedGigaChatClient(
    client_id=app.config['GIGACHAT_CLIENT_ID'],
    client_secret=app.config['GIGACHAT_CLIENT_SECRET'],
    scope=app.config['GIGACHAT_SCOPE'],
    auth_url=app.config['GIGACHAT_AUTH_URL'],
    api_url=app.config['GIGACHAT_API_URL'],
    auth_key= app.config['GIGACHAT_API_AUTH_KEY'],
    rag_system=rag_system
)

# Routes
@app.route('/')
@login_required
def index():
    conversations = Conversation.query.filter_by(user_id=current_user.id)\
        .order_by(Conversation.updated_at.desc())\
        .limit(app.config['MAX_CONTEXT_CHATS'])\
        .all()
    
    # Get RAG system info
    rag_info = None
    if rag_system:
        rag_info = rag_system.get_vector_store_info()
    
    return render_template('index.html', 
                         conversations=conversations,
                         rag_enabled=app.config['RAG_ENABLED'],
                         rag_info=None,
                         max_chats=app.config['MAX_CONTEXT_CHATS'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        
        return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')
        
        user = User(
            username=username,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    message = data.get('message')
    conversation_id = data.get('conversation_id')
    use_rag = data.get('use_rag', True)
    
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    
    # Get or create conversation
    if conversation_id:
        conversation = Conversation.query.filter_by(
            id=conversation_id, 
            user_id=current_user.id
        ).first()
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
    else:
        # Create new conversation
        conversation = Conversation(
            user_id=current_user.id,
            title=message[:100] + ('...' if len(message) > 100 else ''),
            use_rag=use_rag
        )
        db.session.add(conversation)
        db.session.commit()
    
    # Get conversation history
    previous_messages = Message.query.filter_by(conversation_id=conversation.id)\
        .order_by(Message.timestamp.desc())\
        .limit(10)\
        .all()
    
    previous_messages.reverse()
    
    # Format conversation history for API
    conversation_history = []
    for msg in previous_messages:
        conversation_history.append({
            'role': msg.role,
            'content': msg.content
        })
    
    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role='user',
        content=message
    )
    db.session.add(user_message)
    
    # Get response from GigaChat with RAG
    response_data = giga_client.send_message_with_rag(
        query=message,
        conversation_history=conversation_history,
        use_rag=use_rag and conversation.use_rag
    )
    
    if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
        assistant_message = response_data['choices'][0]['message']['content']
        
        # Save assistant response
        assistant_msg = Message(
            conversation_id=conversation.id,
            role='assistant',
            content=assistant_message,
            tokens=response_data.get('usage', {}).get('total_tokens', 0),
            context_used=response_data.get('context_used', False),
            context_sources=json.dumps(response_data.get('context_sources', []), ensure_ascii=False)
        )
        db.session.add(assistant_msg)
        
        # Update conversation
        conversation.updated_at = datetime.utcnow()
        
        # Update title if it's the first message
        if len(conversation.messages) <= 2:  # User message + assistant response
            conversation.title = message[:100] + ('...' if len(message) > 100 else '')
        
        db.session.commit()
        
        return jsonify({
            'response': assistant_message,
            'conversation_id': conversation.id,
            'message_id': assistant_msg.id,
            'context_used': assistant_msg.context_used,
            'context_sources': json.loads(assistant_msg.context_sources) if assistant_msg.context_sources else []
        })
    else:
        return jsonify({'error': 'Failed to get response from GigaChat'}), 500

@app.route('/api/conversations', methods=['GET'])
@login_required
def get_conversations():
    conversations = Conversation.query.filter_by(user_id=current_user.id)\
        .order_by(Conversation.updated_at.desc())\
        .limit(app.config['MAX_CONTEXT_CHATS'])\
        .all()
    
    conversations_list = []
    for conv in conversations:
        conversations_list.append({
            'id': conv.id,
            'title': conv.title,
            'created_at': conv.created_at.isoformat(),
            'updated_at': conv.updated_at.isoformat(),
            'message_count': len(conv.messages),
            'use_rag': conv.use_rag
        })
    
    return jsonify(conversations_list)

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
@login_required
def get_conversation(conversation_id):
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    messages = Message.query.filter_by(conversation_id=conversation.id)\
        .order_by(Message.timestamp.asc())\
        .all()
    
    messages_list = []
    for msg in messages:
        messages_list.append({
            'id': msg.id,
            'role': msg.role,
            'content': msg.content,
            'timestamp': msg.timestamp.isoformat(),
            'context_used': msg.context_used,
            'context_sources': json.loads(msg.context_sources) if msg.context_sources else []
        })
    
    return jsonify({
        'id': conversation.id,
        'title': conversation.title,
        'use_rag': conversation.use_rag,
        'messages': messages_list
    })

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    db.session.delete(conversation)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/conversations/<conversation_id>/title', methods=['PUT'])
@login_required
def update_conversation_title(conversation_id):
    data = request.json
    new_title = data.get('title', '').strip()
    
    if not new_title:
        return jsonify({'error': 'Title is required'}), 400
    
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    conversation.title = new_title[:200]
    db.session.commit()
    
    return jsonify({'success': True, 'title': conversation.title})

@app.route('/api/conversations/<conversation_id>/toggle_rag', methods=['POST'])
@login_required
def toggle_conversation_rag(conversation_id):
    conversation = Conversation.query.filter_by(
        id=conversation_id, 
        user_id=current_user.id
    ).first()
    
    if not conversation:
        return jsonify({'error': 'Conversation not found'}), 404
    
    conversation.use_rag = not conversation.use_rag
    db.session.commit()
    
    return jsonify({
        'success': True,
        'use_rag': conversation.use_rag
    })

@app.route('/api/rag/info', methods=['GET'])
@login_required
def get_rag_info():
    if not rag_system:
        return jsonify({'enabled': False})
    
    info = rag_system.get_vector_store_info()
    return jsonify({
        'enabled': True,
        **info
    })

@app.route('/api/rag/update', methods=['POST'])
@login_required
def update_rag_documents():
    if not rag_system:
        return jsonify({'error': 'RAG system not available'}), 400
    
    try:
        success = rag_system.update_documents()
        if success:
            return jsonify({'success': True, 'message': 'Vector store updated successfully'})
        else:
            return jsonify({'error': 'Failed to update vector store'}), 500
    except Exception as e:
        return jsonify({'error': f'Update error: {str(e)}'}), 500

@app.route('/api/rag/search', methods=['POST'])
@login_required
def search_rag():
    if not rag_system:
        return jsonify({'error': 'RAG system not available'}), 400
    
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        results = rag_system.search_similar_chunks(query, k=k)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'flask': 'running',
        'database': 'connected',
        'gigachat': 'configured' if app.config['GIGACHAT_CLIENT_ID'] else 'not_configured',
        'rag': 'enabled' if app.config['RAG_ENABLED'] and rag_system else 'disabled'
    }
    
    if rag_system:
        rag_info = rag_system.get_vector_store_info()
        status['rag_status'] = rag_info['status']
        status['rag_chunks'] = rag_info.get('total_chunks', 0)
    
    return jsonify(status)

@app.route('/admin/rag-status', methods=['GET'])
@login_required
def admin_rag_status():
    """Admin endpoint to check RAG status"""
    # Simple admin check (in production, use proper admin roles)
    if current_user.username != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    if not rag_system:
        return jsonify({'status': 'not_initialized'})
    
    info = rag_system.get_vector_store_info()
    
    # Get document statistics
    doc_stats = {}
    for metadata in rag_system.chunk_metadata:
        source = metadata.get('source', 'unknown')
        doc_stats[source] = doc_stats.get(source, 0) + 1
    
    return jsonify({
        'rag_info': info,
        'document_stats': doc_stats,
        'total_chunks': len(rag_system.document_chunks),
        'embedding_dimension': rag_system.index.d if rag_system.index else 0
    })

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# Initialize database
with app.app_context():
    db.drop_all()
    db.create_all()
    
    # Create admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password_hash=generate_password_hash('admin123')
        )
        db.session.add(admin)
        db.session.commit()
        print("Created default admin user: admin / admin123")

# Command line interface for setup
def setup_rag_command():
    """Command to setup RAG system"""
    if not app.config['RAG_ENABLED']:
        print("RAG is disabled in configuration")
        return
    
    print("=" * 60)
    print("Setting up RAG system...")
    print("=" * 60)
    
    # Check documents directory
    documents_path = Path(app.config['DOCUMENTS_PATH'])
    if not documents_path.exists():
        print(f"Creating documents directory: {documents_path}")
        documents_path.mkdir(parents=True)
        print(f"\nPlease add PDF documents to '{documents_path}' directory.")
        print("After adding files, restart the application.")
        return
    
    # Check for PDF files
    pdf_files = list(documents_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {documents_path}")
        print("Please add PDF documents to the 'documents' directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    global rag_system
    rag_system = LegalDocumentRAG(rag_config)
    
    # Check initialization
    info = rag_system.get_vector_store_info()
    if info['status'] == 'initialized':
        print(f"\n✓ RAG system initialized successfully!")
        print(f"  Total document chunks: {info['total_chunks']}")
        print("  Document types loaded:")
        for doc_type, count in info['document_types'].items():
            print(f"    - {doc_type}: {count} chunks")
    else:
        print("\n✗ Failed to initialize RAG system")

if __name__ == '__main__':
    import sys
    
    # Check for setup command
    if len(sys.argv) > 1 and sys.argv[1] == 'setup-rag':
        setup_rag_command()
        sys.exit(0)
    
    # Run the application
    print("=" * 60)
    print("Legal GigaChat Web Application")
    print("=" * 60)
    print(f"RAG Enabled: {app.config['RAG_ENABLED']}")
    if app.config['RAG_ENABLED'] and rag_system:
        info = rag_system.get_vector_store_info()
        print(f"RAG Status: {info['status']}")
        if info['status'] == 'initialized':
            print(f"Document Chunks: {info['total_chunks']}")
    print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)