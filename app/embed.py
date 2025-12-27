from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Small + strong starter embedding model
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = _EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype("float32")
