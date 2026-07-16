"""BM25 retrieval over the saved chunk corpus."""

import json
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import get_settings

# Keep financial/reporting terms like "10-K", "12.5", and "year-over-year"
# as single tokens instead of splitting them on punctuation.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Normalize text into BM25 tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def top_positive_score_indices(scores, k: int) -> list[int]:
    """Return the top-k indices with a meaningful BM25 match."""
    return sorted(
        (index for index, score in enumerate(scores) if score > 0),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]


# This cache is process-lifetime; services that re-ingest must call
# load_sparse_index.cache_clear() so BM25 does not serve stale chunks.
@lru_cache
def load_sparse_index() -> tuple[BM25Okapi, tuple[dict[str, str], ...]]:
    settings = get_settings()
    chunks_path = settings.data_processed_dir / "chunks.jsonl"

    # Load the same persisted chunks used by dense retrieval so both retrievers
    # can be fused later by shared chunk_id.
    records = []
    with chunks_path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    # BM25 stores only tokenized text internally, so records are kept alongside
    # the index to reconstruct LangChain Documents after ranking.
    tokenized_corpus = [tokenize(record["text"]) for record in records]
    return BM25Okapi(tokenized_corpus), tuple(records)


def retrieve(query: str, k: int = 4) -> list[Document]:
    bm25, records = load_sparse_index()

    # Score every chunk against the query tokens, then keep only positive BM25
    # matches so sparse retrieval does not add unrelated fallback documents.
    scores = bm25.get_scores(tokenize(query))
    top_indices = top_positive_score_indices(scores, k)

    documents = []
    for index in top_indices:
        record = records[index]
        # Convert the ranked JSONL record back into the Document shape expected
        # by the hybrid retriever and answer generation code.
        documents.append(
            Document(
                page_content=record["text"],
                metadata={
                    "chunk_id": record["chunk_id"],
                    "source": record["source"],
                    "section": record["section"],
                },
            )
        )
    return documents
