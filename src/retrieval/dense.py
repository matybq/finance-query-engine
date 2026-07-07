"""Dense retrieval over the persisted Chroma collection."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import COLLECTION_NAME, get_settings


def retrieve(query: str, k: int = 4) -> list[Document]:
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )
    return vector_store.similarity_search(query, k=k)
