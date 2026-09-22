from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
from app.services.retriever import RetrieverService
from app.services.llm_service import LLMService
from app.config import settings

vector_store = VectorStore()
vector_store.load_index(settings.FAISS_INDEX_PATH, "flat")
vector_store.load_index(settings.FAISS_INDEX_PATH, "ivf")

embedding_service = EmbeddingService()
retriever_service = RetrieverService(vector_store, embedding_service)
llm_service = LLMService()

def get_vector_store():
    return vector_store

def get_embedding_service():
    return embedding_service

def get_retriever():
    return retriever_service

def get_llm():
    return llm_service
