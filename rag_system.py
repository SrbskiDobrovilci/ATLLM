import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import hashlib
from datetime import datetime
import json

class LegalDocumentRAG:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG system for legal documents.
        
        Args:
            config: Configuration dictionary
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
        
        # Initialize embedding model (multilingual for Russian legal texts)
        self.embedding_model = SentenceTransformer(
            config.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        )
        
        # Load or create vector store
        self.index = None
        self.document_chunks = []
        self.chunk_metadata = []
        self.load_or_create_vector_store()
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text += page.get_text()
                    # Add page separator
                    text += f"\n--- Page {page_num + 1} ---\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        
        return text
    
    def process_documents(self) -> Tuple[List[str], List[Dict]]:
        """
        Process all PDF documents in the documents folder.
        
        Returns:
            Tuple of (chunks, metadata)
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
        
        Args:
            filename: PDF filename
            
        Returns:
            Document type
        """
        filename_lower = filename.lower()
        
        if "граждан" in filename_lower:
            return "civil_code"
        elif "уголов" in filename_lower:
            return "criminal_code"
        elif "налог" in filename_lower:
            return "tax_code"
        elif "трудов" in filename_lower:
            return "labor_code"
        elif "семей" in filename_lower:
            return "family_code"
        elif "арбитраж" in filename_lower:
            return "arbitration_code"
        elif "административ" in filename_lower:
            return "administrative_code"
        elif "бюджет" in filename_lower:
            return "budget_code"
        elif "земель" in filename_lower:
            return "land_code"
        elif "вод" in filename_lower:
            return "water_code"
        elif "лес" in filename_lower:
            return "forest_code"
        else:
            return "other_legal_document"
    
    def clean_legal_text(self, text: str) -> str:
        """
        Clean legal text by removing excessive whitespace and normalizing.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Normalize multiple spaces
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        # Join with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove page number markers
        cleaned_text = cleaned_text.replace('--- Page', '')
        
        return cleaned_text
    
    def create_vector_store(self, chunks: List[str], metadata: List[Dict]) -> None:
        """
        Create FAISS vector store from document chunks.
        
        Args:
            chunks: List of text chunks
            metadata: List of metadata for each chunk
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
        
        Returns:
            True if loaded successfully, False otherwise
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
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar chunks with metadata
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
                    "similarity_score": 1.0 / (1.0 + distance),  # Convert distance to similarity
                    "rank": i + 1
                }
                results.append(result)
        
        return results
    
    def retrieve_relevant_context(self, query: str, max_chunks: int = 3) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query
            max_chunks: Maximum number of chunks to include
            
        Returns:
            Formatted context string
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
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous conversation messages
            
        Returns:
            Formatted messages for GigaChat API
        """
        messages = []
        
        # System prompt for legal assistant
        system_prompt = """Вы - опытный юридический ассистент, специализирующийся на законодательстве Российской Федерации. 
Ваша задача - предоставлять точные, актуальные и полезные ответы на основе российского законодательства.

ПРИ ОТВЕТЕ ИСПОЛЬЗУЙТЕ СЛЕДУЮЩИЕ ПРАВИЛА:
1. Всегда ссылайтесь на конкретные законы, кодексы или нормативные акты
2. Указывайте статьи, пункты и части, если они известны
3. Если информация основана на предоставленном контексте, укажите это
4. Будьте точны и избегайте двусмысленности
5. В сложных случаях рекоменуйте обратиться к профессиональному юристу
6. Сохраняйте профессиональный и формальный тон

Контекст законодательства РФ:"""
        
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add context if available
        if context:
            context_message = f"АКТУАЛЬНАЯ ИНФОРМАЦИЯ ИЗ ЗАКОНОДАТЕЛЬСТВА РФ:\n{context}\n\nНА ОСНОВАНИИ ВЫШЕПРИВЕДЕННОЙ ИНФОРМАЦИИ ОТВЕТЬТЕ НА ВОПРОС:"
            messages.append({
                "role": "system",
                "content": context_message
            })
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-6:]:  # Last 6 messages for context
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
        
        Returns:
            Dictionary with vector store information
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
        
        Returns:
            True if update successful
        """
        print("Updating vector store with new documents...")
        
        # Process all documents (including new ones)
        chunks, metadata = self.process_documents()
        
        if chunks:
            # Create new vector store
            self.create_vector_store(chunks, metadata)
            return True
        
        return False