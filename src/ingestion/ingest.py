"""Build the local Chroma index from plain-text filing sections."""
# ver
import json
from pathlib import Path
from shutil import rmtree

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import COLLECTION_NAME, get_settings


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

SECTION_NAMES = {
    "item1_business.txt": "Item 1 – Business",
    "item1a_risk_factors.txt": "Item 1A – Risk Factors",
    "item7_mda.txt": "Item 7 – MD&A",
}


def load_documents(raw_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(raw_dir.iterdir()):
        if path.suffix != ".txt":
            continue

        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={
                    "source": path.name,
                    "section": SECTION_NAMES.get(path.name, path.stem),
                },
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def assign_chunk_ids(chunks: list[Document]) -> list[str]:
    source_counts: dict[str, int] = {}
    chunk_ids = []

    for chunk in chunks:
        source = chunk.metadata["source"]
        index = source_counts.get(source, 0)
        source_counts[source] = index + 1

        chunk_id = f"{source}:{index}"
        chunk.metadata["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)

    return chunk_ids


def write_chunks_jsonl(chunks: list[Document], path: Path) -> None:
    tmp_path = path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "chunk_id": chunk.metadata["chunk_id"],
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "section": chunk.metadata["section"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def count_chunks_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for _ in file)


def build_index() -> tuple[int, int, str, Path, Path]:
    settings = get_settings()
    documents = load_documents(settings.data_raw_dir)
    chunks = split_documents(documents)
    chunk_ids = assign_chunk_ids(chunks)
    persist_dir = settings.chroma_persist_dir
    assert persist_dir is not None
    tmp_persist_dir = persist_dir.with_name(f"{persist_dir.name}.tmp")
    chunks_path = settings.data_processed_dir / "chunks.jsonl"

    if tmp_persist_dir.exists():
        rmtree(tmp_persist_dir)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(tmp_persist_dir),
    )
    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    write_chunks_jsonl(chunks, chunks_path)
    if persist_dir.exists():
        rmtree(persist_dir)
    tmp_persist_dir.rename(persist_dir)

    chunks_jsonl_count = count_chunks_jsonl(chunks_path)
    if chunks_jsonl_count != len(chunks):
        raise RuntimeError(
            "Sparse corpus consistency check failed: "
            f"chunks.jsonl has {chunks_jsonl_count} records, expected {len(chunks)}."
        )

    return len(documents), len(chunks), COLLECTION_NAME, persist_dir, chunks_path


def main() -> None:
    docs_loaded, chunks_created, collection_name, persist_path, chunks_path = build_index()
    print(f"Documents loaded: {docs_loaded}")
    print(f"Chunks created: {chunks_created}")
    print(f"Collection: {collection_name}")
    print(f"Persist path: {persist_path}")
    print(f"Sparse corpus: {chunks_path}")


if __name__ == "__main__":
    main()
