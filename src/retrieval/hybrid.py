"""Hybrid retrieval using dense search, BM25, and RRF fusion."""

from collections import defaultdict

from langchain_core.documents import Document

from src.retrieval import dense, sparse


RRF_K = 60  # Reciprocal Rank Fusion smoothing factor

DENSE_CANDIDATES = 20
DENSE_WEIGHT = 1.0

SPARSE_CANDIDATES = 10
SPARSE_WEIGHT = 0.2  # BM25 weight for sparse search - controls sparse-only chunk boost


def rrf_score(rank: int, weight: float) -> float:
    """Compute the weighted Reciprocal Rank Fusion contribution for one result."""
    return weight / (RRF_K + rank)


def fuse_ranked_documents(
    ranked_document_sets: tuple[tuple[list[Document], float], ...],
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    documents_by_id: dict[str, Document] = {}

    # BM25 currently acts as a secondary exact-term boost.
    for documents, weight in ranked_document_sets:
        # RRF rewards documents that appear high in any ranking.
        # If the same chunk appears in dense and sparse results, both contributions are added.
        for rank, document in enumerate(documents, start=1):
            chunk_id = document.metadata["chunk_id"]
            scores[chunk_id] += rrf_score(rank, weight)
            documents_by_id.setdefault(chunk_id, document)

    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [documents_by_id[chunk_id] for chunk_id in ranked_ids]


def retrieve(query: str, k: int = 4) -> list[Document]:
    dense_documents = dense.retrieve(query, k=DENSE_CANDIDATES)
    sparse_documents = sparse.retrieve(query, k=SPARSE_CANDIDATES)

    fused_documents = fuse_ranked_documents(
        (
            (dense_documents, DENSE_WEIGHT),
            (sparse_documents, SPARSE_WEIGHT),
        )
    )
    return fused_documents[:k]
