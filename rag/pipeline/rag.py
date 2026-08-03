"""End-to-end RAG pipeline: ingestion (load -> split -> embed -> index) and
querying (retrieve -> rerank -> build context -> prompt -> generate)."""
from typing import List, Optional

from rag.config import settings
from rag.context.builder import ContextBuilder
from rag.embeddings.encoder import Encoder
from rag.llm.openai import OpenAILLM
from rag.loaders.html_loader import HTMLLoader
from rag.loaders.pdf_loader import PDFLoader
from rag.loaders.word_loader import WordLoader
from rag.prompts.template import PromptTemplate
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.sparse import SparseRetriever
from rag.schema import Chunk, Document, RetrievedChunk
from rag.splitters.recursive_splitter import RecursiveSplitter
from rag.vectordb.vector_store import VectorStore

_LOADER_BY_EXT = {
    ".pdf": PDFLoader(),
    ".docx": WordLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
}


class RAGPipeline:
    def __init__(self, use_reranker: Optional[bool] = None, retrieval_mode: str = "hybrid"):
        self.encoder = Encoder()
        self.vector_store = VectorStore(dimension=self.encoder.dimension)
        self.splitter = RecursiveSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        self.context_builder = ContextBuilder()
        self.prompt_template = PromptTemplate()
        self.llm = OpenAILLM()

        self.retrieval_mode = retrieval_mode
        self.use_reranker = settings.use_reranker if use_reranker is None else use_reranker
        self.reranker = CrossEncoderReranker() if self.use_reranker else None

        self._sparse_retriever: Optional[SparseRetriever] = None
        self._dense_retriever: Optional[DenseRetriever] = None
        self._hybrid_retriever: Optional[HybridRetriever] = None

    # ---------- Ingestion ----------

    def _load(self, file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1].lower()
        loader = _LOADER_BY_EXT.get(ext)
        if not loader:
            raise ValueError(f"No loader registered for extension: {ext}")
        return loader.load(file_path)

    def ingest(self, file_paths: List[str]) -> int:
        """Load, split, embed, and index a batch of files. Returns the
        number of chunks added."""
        all_chunks: List[Chunk] = []
        for path in file_paths:
            documents = self._load(path)
            chunks = self.splitter.split(documents)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        self.encoder.encode_chunks(all_chunks)
        self.vector_store.add(all_chunks)
        self._rebuild_retrievers()
        return len(all_chunks)

    def _rebuild_retrievers(self) -> None:
        self._dense_retriever = DenseRetriever(self.vector_store, self.encoder)
        self._sparse_retriever = SparseRetriever(self.vector_store.chunks)
        self._hybrid_retriever = HybridRetriever(self._dense_retriever, self._sparse_retriever)

    def save(self) -> None:
        self.vector_store.save()

    def load(self) -> None:
        self.vector_store.load()
        self._rebuild_retrievers()

    # ---------- Querying ----------

    def _retrieve(self, question: str) -> List[RetrievedChunk]:
        if not self._dense_retriever:
            self._rebuild_retrievers()

        if self.retrieval_mode == "dense":
            return self._dense_retriever.retrieve(question, top_k=settings.top_k_dense)
        if self.retrieval_mode == "sparse":
            return self._sparse_retriever.retrieve(question, top_k=settings.top_k_sparse)
        return self._hybrid_retriever.retrieve(
            question,
            top_k_dense=settings.top_k_dense,
            top_k_sparse=settings.top_k_sparse,
            top_k_final=max(settings.top_k_final, settings.top_k_dense),
        )

    def query(self, question: str) -> str:
        candidates = self._retrieve(question)

        if self.reranker and candidates:
            candidates = self.reranker.rerank(question, candidates, top_k=settings.top_k_final)
        else:
            candidates = candidates[: settings.top_k_final]

        context = self.context_builder.build(candidates)
        messages = self.prompt_template.build_messages(question=question, context=context)
        return self.llm.generate(messages)
