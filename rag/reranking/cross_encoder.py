"""Cross-encoder reranking: re-scores a small candidate set with a model
that looks at the query and chunk jointly, which is more accurate — but
slower — than the bi-encoder used for initial retrieval. Run this only on
the top candidates from dense/sparse/hybrid retrieval, never the full corpus."""
from typing import List

from rag.config import settings
from rag.schema import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = None):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model = CrossEncoder(model_name or settings.reranker_model)

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self.model.predict(pairs)

        reranked = [
            RetrievedChunk(chunk=c.chunk, score=float(score), source="reranked")
            for c, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]
