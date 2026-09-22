def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of roughly `chunk_size` words, 
    with `overlap` words overlapping between consecutive chunks.
    """
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
        # Prevent infinite loop if chunk_size <= overlap
        if chunk_size <= overlap:
            break
    return chunks

def semantic_chunking(text: str) -> list[str]:
    """
    Splits text on paragraph boundaries (double newlines).
    """
    paragraphs = text.split("\n\n")
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks
