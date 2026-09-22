from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from app.models.database import get_db, DocumentModel
from app.services.pdf_parser import PDFParser
from app.dependencies import get_vector_store, get_embedding_service
from app.config import settings

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/upload")
def upload_document(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    vector_store = Depends(get_vector_store),
    embedding_service = Depends(get_embedding_service)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    doc_id = str(uuid.uuid4())
    upload_dir = "./uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{doc_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    parser = PDFParser(chunking_strategy="semantic")
    chunks = parser.extract_text_and_chunk(file_path, doc_id)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted")
        
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embedding_service.get_embeddings(texts)
    
    vector_store.add_vectors(embeddings, metadatas, index_type="flat")
    vector_store.add_vectors(embeddings, metadatas, index_type="ivf")
    vector_store.save_index(settings.FAISS_INDEX_PATH, "flat")
    vector_store.save_index(settings.FAISS_INDEX_PATH, "ivf")
    
    new_doc = DocumentModel(
        id=doc_id,
        filename=file.filename,
        file_path=file_path,
        chunk_count=len(chunks)
    )
    db.add(new_doc)
    db.commit()
    
    return {"document_id": doc_id, "message": "Successfully uploaded"}

@router.get("/")
def list_documents(db: Session = Depends(get_db)):
    return db.query(DocumentModel).all()

@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    
    db.delete(doc)
    db.commit()
    
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
        
    return {"message": "Document deleted"}
