# Mini RAG – Ask Your Documents

A lightweight Retrieval-Augmented Generation (RAG) system built with:
- FastAPI
- HuggingFace Transformers
- FAISS
- Docker

## Features
- Upload `.txt` files
- Automatic chunking & embeddings
- Semantic search
- LLM-powered answers

## Run with Docker
```bash
docker build -t mini-rag .
docker run -p 8000:8000 mini-rag
