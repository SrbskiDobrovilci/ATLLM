#!/usr/bin/env python3
"""
Setup script for the RAG system.
Run this script to initialize the vector store with PDF documents.
"""

import os
import sys
from pathlib import Path
from config import Config
from rag_system import LegalDocumentRAG

def setup_rag_system():
    """Initialize the RAG system with PDF documents."""
    
    # Check if documents directory exists
    documents_path = Path(Config.DOCUMENTS_PATH)
    if not documents_path.exists():
        print(f"Creating documents directory: {documents_path}")
        documents_path.mkdir(parents=True)
        
        print("\nPlease add PDF documents to the 'documents' directory.")
        print("Example legal documents you might want to add:")
        print("  - Гражданский кодекс РФ (Civil Code)")
        print("  - Уголовный кодекс РФ (Criminal Code)")
        print("  - Налоговый кодекс РФ (Tax Code)")
        print("  - Трудовой кодекс РФ (Labor Code)")
        print("  - Семейный кодекс РФ (Family Code)")
        print("\nAfter adding PDF files, run this script again.")
        return False
    
    # Check for PDF files
    pdf_files = list(documents_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {documents_path}")
        print("Please add PDF documents to the 'documents' directory.")
        return False
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Initialize RAG system
    print("\nInitializing RAG system...")
    
    rag_config = {
        'documents_path': Config.DOCUMENTS_PATH,
        'vector_store_path': Config.VECTOR_STORE_PATH,
        'data_path': Config.DATA_PATH,
        'embedding_model': Config.EMBEDDING_MODEL,
        'chunk_size': Config.CHUNK_SIZE,
        'chunk_overlap': Config.CHUNK_OVERLAP
    }
    
    rag_system = LegalDocumentRAG(rag_config)
    
    # Check if vector store was created successfully
    info = rag_system.get_vector_store_info()
    if info['status'] == 'initialized':
        print(f"\nRAG system initialized successfully!")
        print(f"Total document chunks: {info['total_chunks']}")
        print("Document types loaded:")
        for doc_type, count in info['document_types'].items():
            print(f"  - {doc_type}: {count} chunks")
        return True
    else:
        print("\nFailed to initialize RAG system.")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Legal GigaChat - RAG System Setup")
    print("=" * 60)
    
    success = setup_rag_system()
    
    if success:
        print("\n" + "=" * 60)
        print("Setup completed successfully!")
        print("\nYou can now run the application:")
        print("  python app.py")
        print("\nAccess the application at: http://localhost:5000")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Setup failed. Please check the messages above.")
        print("=" * 60)
        sys.exit(1)