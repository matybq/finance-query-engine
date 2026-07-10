"""Dense retrieval over the persisted Chroma collection."""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import COLLECTION_NAME, get_settings


# This cache is process-lifetime; services that re-ingest in-process must call
# load_vector_store.cache_clear() so dense retrieval does not serve stale chunks.
@lru_cache
def load_vector_store() -> Chroma:
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )


def retrieve(query: str, k: int = 4) -> list[Document]:
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)
