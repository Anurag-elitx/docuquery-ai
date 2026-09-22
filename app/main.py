from fastapi import FastAPI
from app.config import settings
from app.routes import documents, query

app = FastAPI(title=settings.PROJECT_NAME, description="Document Question Answering System using RAG")

app.include_router(documents.router)
app.include_router(query.router)

@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}
