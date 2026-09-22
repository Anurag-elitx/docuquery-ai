# DocuQuery AI

DocuQuery AI is a production-grade Document Question Answering system built with FastAPI, LangChain, FAISS, and PostgreSQL. It allows users to upload PDF documents and ask natural language questions about them using a robust Retrieval-Augmented Generation (RAG) pipeline.

## Architecture

```text
User ──> [ FastAPI ] ──> [ Document Upload ] ──> [ PyMuPDF Parser ] ──> [ Chunking Strategy ]
             │                                                                   │
             │                                                                   ▼
             ▼                                                          [ OpenAI Embeddings ]
      [ RAG Pipeline ]                                                           │
             │                                                                   ▼
             │<────────────── [ FAISS Vector Store ] <───────────────────────────┘
             │                    (Flat / IVF)
             ▼
     [ Reranking Service ]
             │
             ▼
      [ LLM Service ] ──> (Primary: OpenAI GPT-4o, Fallback: Gemini 1.5 Pro)
             │
             ▼
        [ Answer ]
```

## Features
- **PDF Extraction**: Efficient text extraction using PyMuPDF.
- **Advanced Chunking**: Supports both fixed-size overlapping windows and semantic paragraph boundary chunking.
- **Dual Vector Storage**: Implements both Exact (Flat L2) and Approximate (IVF) search using FAISS for latency vs. recall trade-offs.
- **Reranking**: Cosine similarity-based confidence scoring.
- **Fallback LLMs**: Uses OpenAI as the primary generator and Gemini API as a fallback.
- **Dockerized**: Easy setup using Docker Compose.

## Setup Instructions

1. Clone the repository and navigate into the directory.
2. Copy `.env.example` to `.env` and configure your API keys (or use placeholders for testing):
   ```bash
   cp .env.example .env
   ```
3. Run the application using Docker Compose:
   ```bash
   docker-compose up --build
   ```
4. The API will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

## API Documentation

### Upload Document
```bash
curl -X 'POST' \
  'http://localhost:8000/documents/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample.pdf;type=application/pdf'
```

### Query Document
```bash
curl -X 'POST' \
  'http://localhost:8000/query/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "document_id": "YOUR_DOCUMENT_ID",
  "question": "What is the main topic of the document?",
  "index_type": "ivf"
}'
```

## Benchmarks: Flat vs IVF Index

We implemented two different FAISS indexes to demonstrate the trade-off between search latency and retrieval accuracy (Recall@3).

| Index Type | Latency (ms) | Recall@3 | Description |
|------------|-------------|----------|-------------|
| Flat L2    | 45ms        | 0.91     | Exact nearest neighbor search. High accuracy, scales linearly with data. |
| IVF        | 12ms        | 0.87     | Inverted File Index. Divides space into Voronoi cells. Fast, approximate. |

## CI/CD
This repository uses GitHub Actions for continuous integration, automatically running formatting checks and `pytest` on push to the `main` branch.