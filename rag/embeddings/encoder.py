"""High-level encoder that batches chunks through an embedding provider and
attaches the resulting vectors back onto each Chunk."""
from typing import List

from rag.embeddings.providers import EmbeddingProvider, get_provider
from rag.schema import Chunk


class Encoder:
    def __init__(self, provider: EmbeddingProvider = None, batch_size: int = 64):
        self.provider = provider or get_provider()
        self.batch_size = batch_size

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            vectors.extend(self.provider.encode(batch))
        return vectors

    def encode_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [c.text for c in chunks]
        vectors = self.encode_texts(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        return chunks

    def encode_query(self, query: str) -> List[float]:
        return self.provider.encode([query])[0]

    @property
    def dimension(self) -> int:
        return self.provider.dimension
