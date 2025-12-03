from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Force HuggingFace offline mode by default to avoid
# unexpected network calls in environments without internet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Set HuggingFace mirror for China users (when online)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402


class _HashEmbeddingModel:
    """
    Lightweight, fully local embedding model used as a fallback.

    It generates deterministic vectors based on text hash, which is
    sufficient for tests and offline CI environments where downloading
    real models is not possible.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def encode(self, texts, convert_to_numpy: bool = True):
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        vectors = []
        for text in texts:
            if not text or not str(text).strip():
                raise ValueError("Text cannot be empty")

            # Deterministic random vector based on hash
            hash_val = hash(str(text).strip()) % (2**32)
            rng = np.random.default_rng(hash_val)
            vec = rng.standard_normal(self._dimension, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        arr = np.stack(vectors, axis=0)
        if single_input:
            return arr[0]
        return arr

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension


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
    def model(self):
        """
        Lazy load the underlying embedding model.

        Priority:
        1. If env RELATION_RADAR_EMBEDDINGS_MODE=mock → use hash-based model.
        2. Try loading real SentenceTransformer in offline mode.
        3. Try online mode (when allowed).
        4. Fallback to hash-based model if all else fails.
        """
        if self._model is not None:
            return self._model

        # Explicit mock mode for CI / constrained environments
        mode = os.getenv("RELATION_RADAR_EMBEDDINGS_MODE", "").lower()
        if mode == "mock":
            self._model = _HashEmbeddingModel()
            return self._model

        # Try real model (offline first)
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
            return self._model
        except Exception:
            pass

        try:
            # Fallback to online if offline cache is missing
            print(
                f"Loading model {self.model_name} from cache failed, "
                "trying to load with network access...",
            )
            self._model = SentenceTransformer(self.model_name, local_files_only=False)
            return self._model
        except Exception as exc:
            # Final fallback: deterministic local embedding model
            print(
                "Falling back to hash-based embedding model "
                f"due to error loading SentenceTransformer: {exc}",
            )
            self._model = _HashEmbeddingModel()
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
