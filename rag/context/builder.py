"""Build the final context string sent to the LLM from retrieved chunks:
deduplicates overlapping text and respects a hard token budget so the
prompt never overflows the model's context window."""
from typing import List

import tiktoken

from rag.config import settings
from rag.schema import RetrievedChunk


class ContextBuilder:
    def __init__(self, max_tokens: int = None, encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens or settings.max_context_tokens
        self.encoding = tiktoken.get_encoding(encoding_name)

    def _dedupe(self, results: List[RetrievedChunk]) -> List[RetrievedChunk]:
        seen = set()
        unique = []
        for r in results:
            key = r.chunk.text.strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        return unique

    def build(self, results: List[RetrievedChunk]) -> str:
        results = self._dedupe(results)
        parts = []
        used_tokens = 0

        for i, result in enumerate(results):
            source = result.chunk.metadata.get("source", "unknown")
            page = result.chunk.metadata.get("page")
            label = f"[{i + 1}] source: {source}" + (f", page {page}" if page else "")
            block = f"{label}\n{result.chunk.text.strip()}"

            token_count = len(self.encoding.encode(block))
            if used_tokens + token_count > self.max_tokens:
                break

            parts.append(block)
            used_tokens += token_count

        return "\n\n---\n\n".join(parts)
