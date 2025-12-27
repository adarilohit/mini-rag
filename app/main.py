from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from typing import List

from app.ingest import chunk_text
from app.embed import Embedder
from app.vector_store import VectorStore
from app.rag import RAGAnswerer

app = FastAPI(title="Mini RAG: Ask Your Documents")


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


class AskResponse(BaseModel):
    answer: str
    top_chunks: List[str]


@app.on_event("startup")
def startup():
    # Store objects on app.state (safe across requests)
    app.state.embedder = Embedder()
    app.state.answerer = RAGAnswerer()
    app.state.store = None
    app.state.doc_loaded = False


@app.get("/")
def root():
    return {"status": "ok", "message": "Go to /docs to use the API"}


@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = 800,
    chunk_overlap: int = 150,
):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported in this starter project.")

    raw = (await file.read()).decode("utf-8", errors="ignore").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    chunks = chunk_text(raw, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content found after ingestion.")

    texts = [c.text for c in chunks]

    embedder: Embedder = request.app.state.embedder
    vecs = embedder.embed(texts)

    store = VectorStore(dim=vecs.shape[1])
    store.add(vecs, texts)

    request.app.state.store = store
    request.app.state.doc_loaded = True

    return {
        "message": "Document uploaded and indexed.",
        "num_chunks": len(texts),
        "embedding_dim": vecs.shape[1],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request, req: AskRequest):
    if not request.app.state.doc_loaded or request.app.state.store is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Use /upload first.")

    embedder: Embedder = request.app.state.embedder
    store: VectorStore = request.app.state.store
    answerer: RAGAnswerer = request.app.state.answerer

    qvec = embedder.embed([req.question])
    results = store.search(qvec, top_k=req.top_k)
    contexts = [t for (_, _, t) in results if t.strip()]

    if not contexts:
        return AskResponse(answer="I don't know based on the provided document.", top_chunks=[])

    ans = answerer.answer(req.question, contexts)
    return AskResponse(answer=ans, top_chunks=contexts)
