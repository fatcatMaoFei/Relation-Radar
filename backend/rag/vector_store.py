from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb  # noqa: E402
import numpy as np  # noqa: E402


class VectorStore:
    """
    Vector store using ChromaDB for document storage and retrieval.
    
    Provides functionality to store document embeddings and perform
    similarity searches for RAG applications.
    """
    
    def __init__(self, collection_name: str = "relation_radar", persist_directory: str = "data/chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to store vectors
            persist_directory: Directory to persist the vector database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_documents(self, texts: List[str], ids: Optional[List[str]] = None, 
                     metadatas: Optional[List[dict]] = None) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text documents to add
            ids: Optional list of document IDs. If not provided, UUIDs will be generated
            metadatas: Optional list of metadata for each document
            
        Returns:
            List of document IDs that were added
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if len(texts) != len(ids):
            raise ValueError("Number of texts must match number of IDs")
        
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")
        
        # Add documents to collection
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        return ids
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str], 
                   metadatas: Optional[List[dict]] = None) -> None:
        """
        Add pre-computed vectors to the store.
        
        Args:
            vectors: Array of vectors to add
            ids: List of document IDs
            metadatas: Optional metadata for each vector
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        if metadatas and len(metadatas) != len(vectors):
            raise ValueError("Number of metadatas must match number of vectors")
        
        # Convert numpy array to list for ChromaDB
        embeddings_list = vectors.tolist()
        
        self.collection.add(
            embeddings=embeddings_list,
            ids=ids,
            metadatas=metadatas
        )
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
              where: Optional[dict] = None) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector to search for
            top_k: Number of top results to return
            where: Optional metadata filter conditions
            
        Returns:
            Tuple of (ids, distances, metadatas)
        """
        # Convert numpy array to list for ChromaDB
        query_list = query_vector.tolist()
        
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            where=where
        )
        
        # Extract results
        ids = results['ids'][0] if results['ids'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        return ids, distances, metadatas
    
    def search_by_text(self, query_text: str, top_k: int = 10, 
                      where: Optional[dict] = None) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search using text query (ChromaDB will handle embedding).
        
        Args:
            query_text: Text query to search for
            top_k: Number of top results to return
            where: Optional metadata filter conditions
            
        Returns:
            Tuple of (ids, distances, metadatas)
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where
        )
        
        # Extract results
        ids = results['ids'][0] if results['ids'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        return ids, distances, metadatas
    
    def get_by_id(self, doc_id: str) -> Optional[dict]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document data or None if not found
        """
        results = self.collection.get(ids=[doc_id])
        
        if not results['ids']:
            return None
        
        return {
            'id': results['ids'][0],
            'document': results['documents'][0] if results['documents'] else None,
            'metadata': results['metadatas'][0] if results['metadatas'] else None,
            'embedding': results['embeddings'][0] if results['embeddings'] else None
        }
    
    def delete_by_id(self, doc_id: str) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def count(self) -> int:
        """
        Get the number of documents in the store.
        
        Returns:
            Number of documents
        """
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        # Delete the collection and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


# Global instance for easy import and reuse
_vector_store = None


def get_vector_store() -> VectorStore:
    """
    Get a global singleton instance of the vector store.
    
    Returns:
        Global VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
