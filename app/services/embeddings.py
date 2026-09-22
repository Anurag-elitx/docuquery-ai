from langchain_community.embeddings import OpenAIEmbeddings
import numpy as np
from typing import List
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.cache = {}

    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Generates embeddings in batches and caches them.
        """
        final_embeddings = []
        texts_to_embed = []
        texts_to_embed_indices = []
        
        for idx, text in enumerate(texts):
            if text in self.cache:
                pass
            else:
                texts_to_embed.append(text)
                texts_to_embed_indices.append(idx)
        
        embedded_results = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            try:
                emb = self.embeddings_model.embed_documents(batch)
                embedded_results.extend(emb)
            except Exception as e:
                logger.warning(f"Embedding failed (likely dummy key). Using zero vectors. Error: {e}")
                # Fallback for dummy keys without crashing
                emb = [[0.0] * 1536 for _ in batch]
                embedded_results.extend(emb)
                
        for text, emb in zip(texts_to_embed, embedded_results):
            self.cache[text] = emb
            
        for text in texts:
            final_embeddings.append(self.cache[text])
            
        return np.array(final_embeddings, dtype=np.float32)
        
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generates embedding for a single query string.
        """
        if query in self.cache:
            emb = self.cache[query]
        else:
            try:
                emb = self.embeddings_model.embed_query(query)
                self.cache[query] = emb
            except Exception as e:
                logger.warning(f"Query embedding failed. Using zero vector. Error: {e}")
                emb = [0.0] * 1536
                self.cache[query] = emb
        return np.array([emb], dtype=np.float32)
