import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid import RRF_K, fuse_ranked_documents, rrf_score


def make_document(chunk_id: str, text: str | None = None) -> Document:
    return Document(
        page_content=text or f"content for {chunk_id}",
        metadata={
            "chunk_id": chunk_id,
            "source": "source.txt",
            "section": "section",
        },
    )


def test_rrf_score_applies_rank_and_weight() -> None:
    assert rrf_score(rank=2, weight=0.5) == pytest.approx(0.5 / (RRF_K + 2))


def test_fuse_ranked_documents_combines_scores_across_rankings() -> None:
    doc_a = make_document("a")
    doc_b_dense = make_document("b", "dense copy")
    doc_b_sparse = make_document("b", "sparse copy")

    result = fuse_ranked_documents(
        (
            ([doc_a, doc_b_dense], 1.0),
            ([doc_b_sparse], 1.0),
        )
    )

    assert [doc.metadata["chunk_id"] for doc in result] == ["b", "a"]
    assert result[0].page_content == "dense copy"


def test_fuse_ranked_documents_respects_weights() -> None:
    dense_top = make_document("dense")
    sparse_top = make_document("sparse")

    result = fuse_ranked_documents(
        (
            ([dense_top], 1.0),
            ([sparse_top], 0.2),
        )
    )

    assert [doc.metadata["chunk_id"] for doc in result] == ["dense", "sparse"]
