"""Persistent vector store backed by FAISS, with a parallel metadata store
for chunk text/metadata (FAISS itself only stores raw vectors)."""
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from rag.config import settings
from rag.schema import Chunk
from rag.vectordb.indexes import build_index


class VectorStore:
    def __init__(self, dimension: int = None, index_type: str = "flat",
                 index_path: str = None, metadata_path: str = None):
        self.dimension = dimension or settings.vector_dim
        self.index_type = index_type
        self.index_path = Path(index_path or settings.index_path)
        self.metadata_path = Path(metadata_path or settings.metadata_path)

        self.index = build_index(self.dimension, index_type)
        self.chunks: List[Chunk] = []  # position i corresponds to vector row i

    def add(self, chunks: List[Chunk]) -> None:
        vectors = np.array([c.embedding for c in chunks], dtype="float32")
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[Chunk, float]]:
        vector = np.array([query_vector], dtype="float32")
        scores, indices = self.index.search(vector, top_k)
        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"No saved index found at {self.index_path} / {self.metadata_path}"
            )
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "rb") as f:
            self.chunks = pickle.load(f)

    @classmethod
    def load_from_disk(cls, index_path: str = None, metadata_path: str = None) -> "VectorStore":
        store = cls(index_path=index_path, metadata_path=metadata_path)
        store.load()
        return store

    def __len__(self) -> int:
        return len(self.chunks)
