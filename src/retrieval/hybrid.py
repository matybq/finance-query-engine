"""Hybrid retrieval using dense search, BM25, and RRF fusion."""

from collections import defaultdict

from langchain_core.documents import Document

from src.retrieval import dense, sparse


RRF_K = 60
CANDIDATES = 20
SPARSE_CANDIDATES = 10
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 0.2


def retrieve(query: str, k: int = 4) -> list[Document]:
    dense_documents = dense.retrieve(query, k=CANDIDATES)
    sparse_documents = sparse.retrieve(query, k=SPARSE_CANDIDATES)

    scores: dict[str, float] = defaultdict(float)
    documents_by_id: dict[str, Document] = {}
    # BM25 is for exact-term rescue, not an equal vote on broad semantic queries;
    # these weights are provisional pending RAGAS evals in Fase 5.
    for documents, weight in (
        (dense_documents, DENSE_WEIGHT),
        (sparse_documents, SPARSE_WEIGHT),
    ):
        for rank, document in enumerate(documents, start=1):
            chunk_id = document.metadata["chunk_id"]
            scores[chunk_id] += weight / (RRF_K + rank)
            documents_by_id.setdefault(chunk_id, document)

    # Equal fused scores keep dict insertion order, dense-first, by design.
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [documents_by_id[chunk_id] for chunk_id in ranked_ids[:k]]
