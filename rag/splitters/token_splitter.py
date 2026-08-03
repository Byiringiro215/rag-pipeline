"""Split documents based on exact token counts using tiktoken. Useful when
you need chunks to precisely respect an LLM's context window rather than an
approximate character count."""
from typing import List

import tiktoken

from rag.schema import Chunk, Document


class TokenSplitter:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50,
                 encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def split(self, documents: List[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc in documents:
            tokens = self.encoding.encode(doc.content)
            step = max(self.chunk_size - self.chunk_overlap, 1)
            chunk_idx = 0
            for start in range(0, len(tokens), step):
                token_slice = tokens[start:start + self.chunk_size]
                if not token_slice:
                    continue
                text = self.encoding.decode(token_slice)
                meta = dict(doc.metadata)
                meta["chunk_index"] = chunk_idx
                meta["token_count"] = len(token_slice)
                chunks.append(Chunk(text=text, metadata=meta))
                chunk_idx += 1
                if start + self.chunk_size >= len(tokens):
                    break
        return chunks
