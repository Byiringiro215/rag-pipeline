"""Dense (embedding-based) retrieval over the vector store."""
from typing import List

from rag.embeddings.encoder import Encoder
from rag.schema import RetrievedChunk
from rag.vectordb.vector_store import VectorStore


class DenseRetriever:
    def __init__(self, vector_store: VectorStore, encoder: Encoder):
        self.vector_store = vector_store
        self.encoder = encoder

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        query_vector = self.encoder.encode_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return [RetrievedChunk(chunk=chunk, score=score, source="dense") for chunk, score in results]
