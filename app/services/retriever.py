from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
from typing import List, Dict, Any

class RetrieverService:
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve_and_rerank(self, query: str, index_type: str = "flat", top_k: int = 10, threshold: float = 0.75, final_k: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.embedding_service.get_query_embedding(query)
        
        raw_results, distances = self.vector_store.search(query_vector, k=top_k, index_type=index_type)
        
        reranked_results = []
        for res, dist in zip(raw_results, distances):
            # Convert L2 distance to cosine similarity 
            # (Assuming embeddings are normalized to length 1)
            cosine_sim = 1.0 - (dist / 2.0)
            
            if cosine_sim >= threshold:
                res_copy = dict(res)
                res_copy["confidence"] = cosine_sim
                reranked_results.append(res_copy)
                
        # Sort by confidence descending
        reranked_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return reranked_results[:final_k]
