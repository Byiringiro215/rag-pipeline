# Contributing to rag-pipeline

Thanks for helping improve this project. Small, focused PRs are welcome.

## Ways to contribute

- Bug fixes and clearer error messages
- New document loaders (CSV, Markdown, Notion exports, etc.)
- Alternative vector backends (Qdrant, Pinecone, pgvector)
- Tests, docs, and usage examples
- Performance improvements (batching, caching, index tuning)

## Setup

```bash
git clone https://github.com/Byiringiro215/rag-pipeline.git
cd rag-pipeline
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY if you need generation
```

## Workflow

1. Open an issue describing the change (or claim an existing one).
2. Fork the repo and create a branch: `git checkout -b feature/short-name`
3. Keep changes scoped to one concern.
4. Follow existing module patterns under `rag/` (loaders, splitters, retrieval, etc.).
5. Do not commit `.env`, API keys, or files under `storage/` except `.gitkeep`.
6. Open a pull request against `main` with:
   - what changed and why
   - how you tested it (`python main.py ingest ...`, `query`, etc.)

## Code style

- Prefer small, readable modules over large frameworks.
- Match existing naming and interfaces (`Document`, `Chunk`, `add` / `search` / `save` / `load`).
- Avoid unrelated refactors in the same PR.

## Good first issues

Look for issues labeled `good first issue` or `help wanted`. If none exist yet, open one proposing a small improvement before coding a large feature.
