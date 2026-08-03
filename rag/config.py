"""Central configuration for the RAG pipeline.

All values can be overridden via environment variables (see .env.example)
without touching any code.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # --- Embeddings ---
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")  # or "openai"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Splitting ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))

    # --- Vector store ---
    vector_dim: int = int(os.getenv("VECTOR_DIM", 384))  # 384 = MiniLM default; 1536 for OpenAI small
    index_path: str = os.getenv("INDEX_PATH", "storage/faiss.index")
    metadata_path: str = os.getenv("METADATA_PATH", "storage/metadata.pkl")

    # --- Retrieval ---
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", 10))
    top_k_sparse: int = int(os.getenv("TOP_K_SPARSE", 10))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", 5))

    # --- Reranking ---
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"

    # --- Context assembly ---
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", 3000))

    # --- LLM generation ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.2))


settings = Settings()
