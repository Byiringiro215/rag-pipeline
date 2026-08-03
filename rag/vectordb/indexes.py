"""Factory for building FAISS indexes of different types."""
import faiss


def build_index(dimension: int, index_type: str = "flat"):
    """Create a FAISS index.

    index_type:
        - "flat": exact search, best for small/medium corpora
        - "ivf":  IVF-Flat, faster approximate search for larger corpora
        - "hnsw": HNSW graph index, strong recall/speed tradeoff

    All index types use inner-product similarity; embeddings should be
    L2-normalized beforehand (SentenceTransformerProvider already does this)
    so that inner product is equivalent to cosine similarity.
    """
    index_type = index_type.lower()

    if index_type == "flat":
        return faiss.IndexFlatIP(dimension)

    if index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dimension)
        n_lists = 100
        return faiss.IndexIVFFlat(quantizer, dimension, n_lists, faiss.METRIC_INNER_PRODUCT)

    if index_type == "hnsw":
        return faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)

    raise ValueError(f"Unknown index_type: {index_type}")
