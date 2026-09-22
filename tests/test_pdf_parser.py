import pytest
from app.utils.chunking import fixed_size_chunking, semantic_chunking

def test_fixed_size_chunking():
    text = "word " * 100
    chunks = fixed_size_chunking(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert len(chunks[0].split()) == 50

def test_semantic_chunking():
    text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
    chunks = semantic_chunking(text)
    assert len(chunks) == 3
    assert chunks[0] == "Paragraph 1"
