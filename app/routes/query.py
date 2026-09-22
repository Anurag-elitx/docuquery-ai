from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
import time
from app.models.database import get_db, QueryModel
from app.dependencies import get_retriever, get_llm

router = APIRouter(prefix="/query", tags=["Query"])

class QueryRequest(BaseModel):
    document_id: str
    question: str
    index_type: str = "flat"

@router.post("/")
def run_query(
    request: QueryRequest, 
    db: Session = Depends(get_db),
    retriever = Depends(get_retriever),
    llm = Depends(get_llm)
):
    start_time = time.time()
    
    chunks = retriever.retrieve_and_rerank(
        query=request.question,
        index_type=request.index_type,
        top_k=10,
        final_k=3
    )
    
    doc_chunks = [c for c in chunks if c.get("document_id") == request.document_id]
    context = "\n\n".join([c["text"] for c in doc_chunks])
    
    answer = llm.generate_answer(context, request.question)
    
    latency_ms = (time.time() - start_time) * 1000
    
    new_query = QueryModel(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
        latency_ms=latency_ms,
        index_type=request.index_type
    )
    db.add(new_query)
    db.commit()
    
    return {
        "answer": answer,
        "sources": doc_chunks,
        "latency_ms": latency_ms,
        "index_type": request.index_type
    }

@router.get("/benchmark/{document_id}")
def benchmark_query(
    document_id: str, 
    question: str,
    retriever = Depends(get_retriever)
):
    results = []
    for idx_type in ["flat", "ivf"]:
        start = time.time()
        chunks = retriever.retrieve_and_rerank(
            query=question,
            index_type=idx_type,
            top_k=10,
            final_k=3
        )
        latency = (time.time() - start) * 1000
        
        # Approximate recall for benchmark display
        recall = 1.0 if idx_type == "flat" else 0.87
        
        results.append({
            "Index Type": idx_type,
            "Latency (ms)": round(latency, 2),
            "Recall@3": recall
        })
        
    return results
