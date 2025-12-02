#!/usr/bin/env python3
"""
Mock test script for PR-0.1-05: Embedding pipeline & vector store
This version uses mock embeddings to test the pipeline without requiring network access.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.vector_store import get_vector_store  # noqa: E402


class MockEmbeddingClient:
    """Mock embedding client for testing without network dependencies."""
    
    def __init__(self):
        self.dimension = 384  # Same as all-MiniLM-L6-v2
    
    def encode(self, text: str) -> np.ndarray:
        """Create a mock vector based on text hash."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Create a deterministic vector based on text content
        hash_val = hash(text.strip()) % (2**32)
        np.random.seed(hash_val)
        vector = np.random.randn(self.dimension).astype(np.float32)
        # Normalize to unit vector
        vector = vector / np.linalg.norm(vector)
        return vector
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Create mock vectors for multiple texts."""
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        vectors = []
        for text in texts:
            if text and text.strip():
                vectors.append(self.encode(text))
        
        return np.array(vectors)
    
    def get_dimension(self) -> int:
        return self.dimension
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        vec1 = self.encode(text1)
        vec2 = self.encode(text2)
        
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        return float(dot_product / norms)


def test_embedding_pipeline():
    """Test the complete embedding pipeline with mock embeddings."""
    print("🚀 Testing Embedding Pipeline (Mock Version)...")
    
    # Initialize clients
    embedding_client = MockEmbeddingClient()
    vector_store = get_vector_store()
    
    # Clear any existing data
    vector_store.clear()
    print(f"📊 Vector store initialized. Count: {vector_store.count()}")
    
    # Test data - sample relation events
    test_documents = [
        "猫今天心情很好，我们一起去吃了川菜，她很喜欢麻辣的味道",
        "阿B昨天健身很累，不过他说感觉很有成就感，下次还想一起去",
        "小张生日快到了，她平时喜欢安静的环境，不太喜欢太热闹的地方",
        "和猫聊天时发现她最近压力有点大，工作上的事情让她有些焦虑",
        "阿B推荐了一家新的健身房，设备很好，环境也不错"
    ]
    
    print("\n📝 Test Documents:")
    for i, doc in enumerate(test_documents, 1):
        print(f"  {i}. {doc}")
    
    # Test 1: Single text embedding
    print("\n🔧 Test 1: Single Text Embedding")
    test_text = test_documents[0]
    vector = embedding_client.encode(test_text)
    print(f"Text: {test_text}")
    print(f"Vector dimension: {len(vector)}")
    print(f"Vector (first 5): {vector[:5]}")
    print(f"Expected dimension: {embedding_client.get_dimension()}")
    
    # Test 2: Batch embedding
    print("\n🔧 Test 2: Batch Text Embedding")
    vectors = embedding_client.encode_batch(test_documents)
    print(f"Batch size: {len(test_documents)}")
    print(f"Vector matrix shape: {vectors.shape}")
    
    # Test 3: Add vectors to vector store
    print("\n🔧 Test 3: Adding Vectors to Vector Store")
    doc_ids = [f"doc_{i}" for i in range(len(test_documents))]
    metadatas = [
        {"person": "猫", "type": "聚餐", "emotion": "开心"},
        {"person": "阿B", "type": "健身", "emotion": "满足"},
        {"person": "小张", "type": "生日", "emotion": "中性"},
        {"person": "猫", "type": "聊天", "emotion": "焦虑"},
        {"person": "阿B", "type": "推荐", "emotion": "积极"}
    ]
    
    vector_store.add_vectors(
        vectors=vectors,
        ids=doc_ids,
        metadatas=metadatas
    )
    print(f"Added {len(doc_ids)} vectors")
    print(f"Vector store count: {vector_store.count()}")
    
    # Test 4: Vector similarity search
    print("\n🔧 Test 4: Vector Similarity Search")
    query_text = "猫的心情怎么样？"
    query_vector = embedding_client.encode(query_text)
    print(f"Query: {query_text}")
    
    ids, distances, metadatas = vector_store.search(
        query_vector=query_vector,
        top_k=3
    )
    
    print("Top 3 similar documents:")
    for i, (doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
        doc_idx = int(doc_id.split('_')[1])
        document = test_documents[doc_idx]
        print(f"  {i+1}. ID: {doc_id}")
        print(f"     Distance: {distance:.4f}")
        print(f"     Metadata: {metadata}")
        print(f"     Text: {document}")
        print()
    
    # Test 5: Different query
    print("🔧 Test 5: Health/Fitness Query")
    query_vector = embedding_client.encode("健身相关的内容")
    ids, distances, metadatas = vector_store.search(
        query_vector=query_vector,
        top_k=2
    )
    
    print("Query: 健身相关的内容")
    print("Top 2 similar documents:")
    for i, (doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
        doc_idx = int(doc_id.split('_')[1])
        document = test_documents[doc_idx]
        print(f"  {i+1}. ID: {doc_id}, Distance: {distance:.4f}")
        print(f"     Text: {document}")
        print()
    
    # Test 6: Text similarity calculation
    print("🔧 Test 6: Text Similarity")
    text1 = "猫心情很好"
    text2 = "猫今天很开心"
    text3 = "阿B在健身"
    
    sim_score = embedding_client.similarity(text1, text2)
    print(f"Similarity between '{text1}' and '{text2}': {sim_score:.4f}")
    
    sim_score = embedding_client.similarity(text1, text3)
    print(f"Similarity between '{text1}' and '{text3}': {sim_score:.4f}")
    
    # Test 7: Metadata filtering
    print("\n🔧 Test 7: Metadata Filtering")
    ids, distances, metadatas = vector_store.search(
        query_vector=embedding_client.encode("心情"),
        top_k=5,
        where={"person": "猫"}
    )
    
    print("Query: '心情' filtered by person='猫'")
    print(f"Found {len(ids)} documents:")
    for i, (doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
        doc_idx = int(doc_id.split('_')[1])
        document = test_documents[doc_idx]
        print(f"  {i+1}. {document}")
    
    # Test 8: Get by ID
    print("\n🔧 Test 8: Get Document by ID")
    doc_data = vector_store.get_by_id("doc_0")
    if doc_data:
        print(f"Retrieved document: {doc_data}")
    else:
        print("Document not found")
    
    print("\n✅ All tests completed successfully!")
    print(f"📊 Final vector store count: {vector_store.count()}")
    
    return True


if __name__ == "__main__":
    try:
        test_embedding_pipeline()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
