"""Compare dense-only retrieval with hybrid retrieval."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.retrieval import dense, hybrid


QUERIES = [
    "How does Airbnb make money?",
    "property damage protection $3 million",
    "Reserve Now, Pay Later",
    "risks related to hosts",
    "adjusted EBITDA",
]


def preview(text: str) -> str:
    return " ".join(text.split())[:90]


def describe(doc) -> str:
    return f"{doc.metadata['chunk_id']} | {doc.metadata['section']} | {preview(doc.page_content)}"


def main() -> None:
    for query in QUERIES:
        dense_docs = dense.retrieve(query, k=4)
        hybrid_docs = hybrid.retrieve(query, k=4)
        dense_ids = {doc.metadata["chunk_id"] for doc in dense_docs}

        print(f"\nQuery: {query}")
        print("Dense top-4")
        for index, doc in enumerate(dense_docs, start=1):
            print(f"  {index}. {describe(doc)}")

        print("Hybrid top-4")
        for index, doc in enumerate(hybrid_docs, start=1):
            marker = " *" if doc.metadata["chunk_id"] not in dense_ids else ""
            print(f"  {index}. {describe(doc)}{marker}")


if __name__ == "__main__":
    main()
