"""Hybrid retrieval combining dense and sparse results via Reciprocal Rank
Fusion (RRF). RRF is used instead of a naive weighted-score blend because
dense (cosine ~0-1) and sparse (BM25, unbounded) scores live on completely
different scales — RRF sidesteps that by fusing on rank position instead."""
from typing import Dict, List

from rag.retrieval.dense import DenseRetriever
from rag.retrieval.sparse import SparseRetriever
from rag.schema import RetrievedChunk


class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever,
                 rrf_k: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k_dense: int = 10, top_k_sparse: int = 10,
                 top_k_final: int = 10) -> List[RetrievedChunk]:
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k_dense)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k_sparse)

        fused_scores: Dict[str, float] = {}
        chunk_lookup = {}

        for rank, result in enumerate(dense_results):
            key = result.chunk.text
            chunk_lookup[key] = result.chunk
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, result in enumerate(sparse_results):
            key = result.chunk.text
            chunk_lookup[key] = result.chunk
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_final]
        return [
            RetrievedChunk(chunk=chunk_lookup[key], score=score, source="hybrid")
            for key, score in ranked
        ]
