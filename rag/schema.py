"""Shared data schemas used across the RAG pipeline."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A raw loaded document, before splitting."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A split piece of a Document, ready for embedding and indexing."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class RetrievedChunk:
    """A chunk returned from retrieval, carrying a relevance score."""
    chunk: Chunk
    score: float
    source: str = "dense"  # dense | sparse | hybrid | reranked
