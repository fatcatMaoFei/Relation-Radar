from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Force HuggingFace offline mode - no network access needed
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
# Set HuggingFace mirror for China users (when online)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


class EmbeddingClient:
    """
    Embedding client using Sentence Transformers for text vectorization.
    
    Provides a unified interface for converting text to vectors,
    with support for both single texts and batch processing.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding client.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       Default is a lightweight multilingual model.
        """
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to avoid loading during import."""
        if self._model is None:
            try:
                # Try offline first
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                # Fallback to online if offline fails
                print(f"Loading model {self.model_name} from cache failed, trying online...")
                self._model = SentenceTransformer(self.model_name, local_files_only=False)
        return self._model
    
    def encode(self, text: str) -> np.ndarray:
        """
        Convert a single text to vector.
        
        Args:
            text: Input text to vectorize
            
        Returns:
            Vector representation as numpy array
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        vector = self.model.encode(text.strip(), convert_to_numpy=True)
        return vector
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to vectors in batch.
        
        Args:
            texts: List of texts to vectorize
            
        Returns:
            Array of vectors with shape (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts
        filtered_texts = [text.strip() for text in texts if text and text.strip()]
        if not filtered_texts:
            raise ValueError("No valid texts found after filtering empty ones")
        
        vectors = self.model.encode(filtered_texts, convert_to_numpy=True)
        return vectors
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension size of vectors produced by this model
        """
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        vec1 = self.encode(text1)
        vec2 = self.encode(text2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return float(dot_product / norms)


# Global instance for easy import and reuse
_embedding_client = None


def get_embedding_client() -> EmbeddingClient:
    """
    Get a global singleton instance of the embedding client.
    
    This ensures the heavy model is only loaded once throughout
    the application lifecycle.
    
    Returns:
        Global EmbeddingClient instance
    """
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
