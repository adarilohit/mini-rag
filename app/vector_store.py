from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import faiss

class VectorStore:
    """
    Minimal in-memory FAISS store:
    - embeddings in FAISS
    - chunk text kept in a dict by internal integer id
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim if embeddings normalized
        self.id_to_text: Dict[int, str] = {}
        self.next_id = 0

    def add(self, embeddings: np.ndarray, texts: List[str]) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be shape (n, {self.dim})")
        n = embeddings.shape[0]
        ids = list(range(self.next_id, self.next_id + n))
        self.index.add(embeddings)
        for i, t in zip(ids, texts):
            self.id_to_text[i] = t
        self.next_id += n

    def search(self, query_emb: np.ndarray, top_k: int = 4) -> List[Tuple[int, float, str]]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        scores, idxs = self.index.search(query_emb.astype("float32"), top_k)
        results: List[Tuple[int, float, str]] = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            results.append((i, float(s), self.id_to_text.get(i, "")))
        return results
