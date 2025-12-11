import logging
import os
from typing import List
from time import sleep

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from splitter import split_documents
from dotenv import load_dotenv
from langchain_core.documents import Document

EMBED_DELAY = 0.02  # reduce CPU usage during embedding


class EmbeddingProxy:
    """
    Pequeño wrapper para reducir uso de CPU al generar embeddings.
    """
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    """
    Crea una base vectorial Chroma a partir de Document() o strings.
    Compatible con tu RAG moderno.
    """

    if not texts:
        logging.warning("Se intentó crear una base vectorial con textos vacíos.")

    # Selección automática de embeddings OpenAI
    if not embeddings:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Falta OPENAI_API_KEY para crear embeddings.")

        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )

    proxy_embeddings = EmbeddingProxy(embeddings)

    db = Chroma(
        collection_name=collection_name,
        embedding_function=proxy_embeddings,
        persist_directory=os.path.join("store/", collection_name)
    )

    # Normalizar: convertir strings a Document()
    docs = []
    for t in texts:
        if isinstance(t, str):
            docs.append(Document(page_content=t))
        else:
            docs.append(t)

    db.add_documents(docs)
    return db


def find_similar(vs, query: str):
    """
    Búsqueda simple para debugging o inspección del vector store.
    """
    return vs.similarity_search(query)


def main():
    load_dotenv()
    print("El módulo vector_store está listo. Se usa automáticamente en el chatbot RAG.")


if __name__ == "__main__":
    main()
