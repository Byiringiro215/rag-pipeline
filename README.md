# rag-pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub issues](https://img.shields.io/github/issues/Byiringiro215/rag-pipeline)](https://github.com/Byiringiro215/rag-pipeline/issues)
[![GitHub stars](https://img.shields.io/github/stars/Byiringiro215/rag-pipeline)](https://github.com/Byiringiro215/rag-pipeline/stargazers)

A modular Retrieval-Augmented Generation pipeline: load documents, split
them, embed and index them, retrieve relevant chunks (dense, sparse, or
hybrid), rerank, assemble context, and generate grounded answers.

**Contributions welcome** — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
PR guidelines, and good first ideas (loaders, vector backends, tests, docs).

## Structure

```
rag/
├── loaders/          # pdf_loader, word_loader, html_loader -> Document
├── splitters/        # recursive_splitter, sentence_splitter, token_splitter -> Chunk
├── embeddings/        # encoder.py (batching) + providers.py (sentence-transformers / OpenAI)
├── vectordb/          # vector_store.py (FAISS + metadata persistence), indexes.py (flat/ivf/hnsw)
├── retrieval/          # dense.py, sparse.py (BM25), hybrid.py (Reciprocal Rank Fusion)
├── reranking/          # cross_encoder.py
├── context/            # builder.py (dedupe + token-budget-aware context assembly)
├── prompts/            # template.py
├── llm/                # openai.py (chat completion wrapper)
├── pipeline/           # rag.py (RAGPipeline — ties everything together)
├── schema.py           # Document / Chunk / RetrievedChunk dataclasses (shared)
└── config.py           # Settings, overridable via environment variables
main.py                 # CLI: ingest / query / chat
```

`schema.py` and `config.py` aren't in the folder tree you sketched, but every
module needs a shared data shape and a shared settings source, so they live
at the top of the `rag/` package.

## Install

```bash
pip install -r requirements.txt --break-system-packages
cp .env.example .env   # then edit values, especially OPENAI_API_KEY
```

By default, embeddings run locally via `sentence-transformers`
(`all-MiniLM-L6-v2`, 384-dim, no API key needed). Only text generation
(`OpenAILLM`) requires `OPENAI_API_KEY`. Set `EMBEDDING_PROVIDER=openai` in
`.env` to use OpenAI embeddings instead (remember to also set
`VECTOR_DIM=1536`).

## Usage

```bash
# Ingest documents (repeatable; adds to the existing index if one exists)
python main.py ingest docs/handbook.pdf docs/policy.docx docs/faq.html

# Ask a single question
python main.py query "What is the refund policy?"

# Interactive chat loop
python main.py chat

# Choose retrieval strategy / disable reranking
python main.py --mode dense --no-rerank query "..."
```

Index files are written to `storage/faiss.index` and `storage/metadata.pkl`.

## How a query flows

1. **Retrieve** — `dense` (FAISS cosine similarity), `sparse` (BM25 keyword
   match), or `hybrid` (both, fused by Reciprocal Rank Fusion — robust to
   the fact dense and sparse scores live on different scales).
2. **Rerank** — a cross-encoder re-scores the top candidates against the
   query jointly (slower, more accurate than the initial bi-encoder pass).
   Toggle with `USE_RERANKER` / `--no-rerank`.
3. **Build context** — dedupes chunks and packs them into a token budget
   (`MAX_CONTEXT_TOKENS`), tagging each with `[n] source: ..., page ...` so
   the model can cite sources.
4. **Prompt + generate** — `PromptTemplate` builds system/user messages;
   `OpenAILLM` calls the chat completions API.

## Extending

- **New loader**: add a class with a `.load(path) -> List[Document]` method
  under `loaders/`, then register its extension in `_LOADER_BY_EXT` in
  `pipeline/rag.py`.
- **New splitter**: any class with `.split(documents) -> List[Chunk]` is a
  drop-in replacement for `RecursiveSplitter` in `RAGPipeline.__init__`.
- **New embedding provider**: implement `EmbeddingProvider` in
  `embeddings/providers.py` and add it to `get_provider()`.
- **New vector backend**: swap `VectorStore` for another implementation
  with the same `add` / `search` / `save` / `load` interface (e.g. Qdrant,
  Pinecone, pgvector) without touching the retrieval layer.

## Contributing

1. Fork the repo and create a feature branch.
2. Keep PRs focused; match existing interfaces under `rag/`.
3. Open a PR using the template — details in [CONTRIBUTING.md](CONTRIBUTING.md).

Bug reports and feature ideas: use the GitHub issue templates.

## License

MIT — see [LICENSE](LICENSE).
