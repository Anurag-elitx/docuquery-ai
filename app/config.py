import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "DocuQuery AI"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-dummy-openai-key")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "dummy-gemini-key")
    # For portfolio completeness, default to sqlite so it runs instantly without docker if needed
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./docuquery.db") 
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    
    class Config:
        env_file = ".env"

settings = Settings()
