#!/usr/bin/env python3
"""
Test script for PR-0.1-05: Embedding pipeline & vector store
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.embeddings import get_embedding_client  # noqa: E402
from backend.rag.vector_store import get_vector_store  # noqa: E402


def test_embedding_pipeline():
    """Test the complete embedding pipeline."""
    print("🚀 Testing Embedding Pipeline...")
    
    # Initialize clients
    embedding_client = get_embedding_client()
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
    
    # Test 2: Batch embedding
    print("\n🔧 Test 2: Batch Text Embedding")
    vectors = embedding_client.encode_batch(test_documents)
    print(f"Batch size: {len(test_documents)}")
    print(f"Vector matrix shape: {vectors.shape}")
    
    # Test 3: Add documents to vector store
    print("\n🔧 Test 3: Adding Documents to Vector Store")
    doc_ids = [f"doc_{i}" for i in range(len(test_documents))]
    metadatas = [
        {"person": "猫", "type": "聚餐", "emotion": "开心"},
        {"person": "阿B", "type": "健身", "emotion": "满足"},
        {"person": "小张", "type": "生日", "emotion": "中性"},
        {"person": "猫", "type": "聊天", "emotion": "焦虑"},
        {"person": "阿B", "type": "推荐", "emotion": "积极"}
    ]
    
    added_ids = vector_store.add_documents(
        texts=test_documents,
        ids=doc_ids,
        metadatas=metadatas
    )
    print(f"Added {len(added_ids)} documents")
    print(f"Vector store count: {vector_store.count()}")
    
    # Test 4: Text similarity search
    print("\n🔧 Test 4: Similarity Search")
    query_text = "猫的心情怎么样？"
    print(f"Query: {query_text}")
    
    ids, distances, metadatas = vector_store.search_by_text(
        query_text=query_text,
        top_k=3
    )
    
    print("Top 3 similar documents:")
    for i, (doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
        doc_data = vector_store.get_by_id(doc_id)
        document = doc_data['document'] if doc_data else "Not found"
        print(f"  {i+1}. ID: {doc_id}")
        print(f"     Distance: {distance:.4f}")
        print(f"     Metadata: {metadata}")
        print(f"     Text: {document}")
        print()
    
    # Test 5: Vector similarity search
    print("🔧 Test 5: Vector-based Search")
    query_vector = embedding_client.encode("健身相关的内容")
    ids, distances, metadatas = vector_store.search(
        query_vector=query_vector,
        top_k=2
    )
    
    print("Query: 健身相关的内容")
    print("Top 2 similar documents:")
    for i, (doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
        doc_data = vector_store.get_by_id(doc_id)
        document = doc_data['document'] if doc_data else "Not found"
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
    ids, distances, metadatas = vector_store.search_by_text(
        query_text="心情",
        top_k=5,
        where={"person": "猫"}
    )
    
    print("Query: '心情' filtered by person='猫'")
    print(f"Found {len(ids)} documents:")
    for i, (doc_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
        doc_data = vector_store.get_by_id(doc_id)
        document = doc_data['document'] if doc_data else "Not found"
        print(f"  {i+1}. {document}")
    
    print("\n✅ All tests completed successfully!")
    print(f"📊 Final vector store count: {vector_store.count()}")
    
    return True


if __name__ == "__main__":
    try:
        test_embedding_pipeline()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
