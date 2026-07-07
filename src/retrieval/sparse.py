"""BM25 retrieval over the saved chunk corpus."""

import json
import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import get_settings


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# This cache is process-lifetime; services that re-ingest must call
# load_sparse_index.cache_clear() so BM25 does not serve stale chunks.
@lru_cache
def load_sparse_index() -> tuple[BM25Okapi, tuple[dict[str, str], ...]]:
    settings = get_settings()
    chunks_path = settings.data_processed_dir / "chunks.jsonl"

    records = []
    with chunks_path.open(encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    tokenized_corpus = [tokenize(record["text"]) for record in records]
    return BM25Okapi(tokenized_corpus), tuple(records)


def retrieve(query: str, k: int = 4) -> list[Document]:
    bm25, records = load_sparse_index()
    scores = bm25.get_scores(tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]

    documents = []
    for index in top_indices:
        record = records[index]
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
