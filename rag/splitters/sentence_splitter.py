"""Split documents into chunks aligned to sentence boundaries, grouping
sentences together until a target character budget is reached. Useful when
you want chunks that never cut a sentence in half."""
import re
from typing import List

from rag.schema import Chunk, Document

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


class SentenceSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap_sentences: int = 1):
        self.chunk_size = chunk_size
        self.chunk_overlap_sentences = chunk_overlap_sentences

    @staticmethod
    def _sentences(text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    def split(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            sentences = self._sentences(doc.content)
            current: List[str] = []
            current_len = 0
            chunk_idx = 0

            def flush():
                nonlocal current, chunk_idx
                if not current:
                    return
                meta = dict(doc.metadata)
                meta["chunk_index"] = chunk_idx
                chunks.append(Chunk(text=" ".join(current), metadata=meta))
                chunk_idx += 1

            for sentence in sentences:
                if current_len + len(sentence) > self.chunk_size and current:
                    flush()
                    current = current[-self.chunk_overlap_sentences:] if self.chunk_overlap_sentences else []
                    current_len = sum(len(s) for s in current)
                current.append(sentence)
                current_len += len(sentence)
            flush()
        return chunks
