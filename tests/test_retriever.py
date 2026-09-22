import pytest
import numpy as np
from app.services.vector_store import VectorStore

def test_vector_store_flat():
    vs = VectorStore(dimension=4)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    metadatas = [{"id": 1}, {"id": 2}]
    
    vs.add_vectors(vectors, metadatas, index_type="flat")
    
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results, distances = vs.search(query, k=1, index_type="flat")
    
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert distances[0] == 0.0
