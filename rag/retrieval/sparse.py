"""Sparse (keyword-based) retrieval using BM25. Good for exact terms, IDs,
and proper nouns that dense embeddings sometimes blur together."""
import re
from typing import List

from rank_bm25 import BM25Okapi

from rag.schema import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseRetriever:
    """BM25 retriever built once over the full chunk corpus. Rebuild
    (re-instantiate) whenever the corpus changes."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        if not self.bm25:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=chunk, score=float(score), source="sparse")
            for chunk, score in ranked if score > 0
        ]
