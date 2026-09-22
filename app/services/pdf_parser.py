import fitz  # PyMuPDF
from typing import List, Dict, Any
from app.utils.chunking import fixed_size_chunking, semantic_chunking

class PDFParser:
    def __init__(self, chunking_strategy: str = "fixed"):
        self.chunking_strategy = chunking_strategy

    def extract_text_and_chunk(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF, cleans it, chunks it, and returns a list of chunks with metadata.
        """
        doc = fitz.open(file_path)
        chunks_with_metadata = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            # Clean extracted text
            text = self._clean_text(text)
            if not text:
                continue
                
            if self.chunking_strategy == "semantic":
                page_chunks = semantic_chunking(text)
            else:
                page_chunks = fixed_size_chunking(text)
                
            for idx, chunk in enumerate(page_chunks):
                chunks_with_metadata.append({
                    "text": chunk,
                    "metadata": {
                        "document_id": document_id,
                        "page_number": page_num + 1,
                        "chunk_index": idx
                    }
                })
                
        return chunks_with_metadata

    def _clean_text(self, text: str) -> str:
        # Basic cleanup: normalize whitespaces, remove excessive spaces
        cleaned = " ".join(text.split())
        return cleaned
