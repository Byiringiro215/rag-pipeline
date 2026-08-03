"""Recursively split text on a hierarchy of separators until every chunk
fits within chunk_size. Falls back to the next, finer-grained separator
whenever a piece is still too large (similar to LangChain's
RecursiveCharacterTextSplitter, implemented here with no extra dependency)."""
from typing import List, Optional

from rag.schema import Chunk, Document

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120,
                 separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep, *rest_seps = separators
        parts = text.split(sep) if sep else list(text)

        chunks: List[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_text(part, rest_seps))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            overlapped.append(prev_tail + chunks[i])
        return overlapped

    def split(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            raw_chunks = self._split_text(doc.content, self.separators)
            raw_chunks = self._add_overlap(raw_chunks)
            for idx, text in enumerate(raw_chunks):
                if not text.strip():
                    continue
                meta = dict(doc.metadata)
                meta["chunk_index"] = idx
                chunks.append(Chunk(text=text.strip(), metadata=meta))
        return chunks
