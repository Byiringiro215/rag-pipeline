"""Embedding provider implementations. Each provider exposes `.encode(texts)`
returning a list of float vectors, so the rest of the pipeline never has to
know which backend is in use."""
from abc import ABC, abstractmethod
from typing import List

from rag.config import settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerProvider(EmbeddingProvider):
    """Local, free embeddings via sentence-transformers. Good default for
    development, or whenever you don't want to depend on an external API."""

    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via the OpenAI API (e.g. text-embedding-3-small)."""

    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = None, api_key: str = None):
        from openai import OpenAI  # lazy import

        self.model_name = model_name or settings.openai_embedding_model
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)

    def encode(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._DIMENSIONS.get(self.model_name, 1536)


def get_provider(name: str = None) -> EmbeddingProvider:
    name = (name or settings.embedding_provider).lower()
    if name == "openai":
        return OpenAIEmbeddingProvider()
    if name in ("sentence_transformers", "st", "local"):
        return SentenceTransformerProvider()
    raise ValueError(f"Unknown embedding provider: {name}")
