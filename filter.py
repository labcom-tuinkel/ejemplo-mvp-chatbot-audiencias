from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    LongContextReorder
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from vector_store import create_vector_db
from splitter import split_documents


def create_retriever(texts):
    """
    Retriever híbrido moderno compatible con LangChain.
    Combina:
    - BM25
    - Embeddings densos
    - Embeddings esparsos
    - Filtro de redundancia
    - Reordenamiento contextual
    """

    # Embeddings densos y esparsos
    dense_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sparse_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        encode_kwargs={'normalize_embeddings': False}
    )

    dense_vs = create_vector_db(texts, collection_name="dense", embeddings=dense_embeddings)
    sparse_vs = create_vector_db(texts, collection_name="sparse", embeddings=sparse_embeddings)

    dense_retriever = dense_vs.as_retriever(search_kwargs={"k": 4})
    sparse_retriever = sparse_vs.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_texts(
        [t.page_content for t in texts]
    )

    redundant_filter = EmbeddingsRedundantFilter(embeddings=sparse_embeddings)
    reordering = LongContextReorder()

    class ModernHybridRetriever(BaseRetriever):

        def _get_relevant_documents(self, query, *, run_manager=None):
            """
            Método OBLIGATORIO en LangChain moderno.
            """

            # Recuperación híbrida
            docs = (
                dense_retriever.get_relevant_documents(query)
                + sparse_retriever.get_relevant_documents(query)
                + bm25_retriever.get_relevant_documents(query)
            )

            # Eliminar duplicados manteniendo orden
            seen = set()
            unique_docs = []
            for d in docs:
                if d.page_content not in seen:
                    unique_docs.append(d)
                    seen.add(d.page_content)

            # Filtro de documentos redundantes
            unique_docs = redundant_filter.transform_documents(unique_docs)

            # Reordenamiento para mejorar la coherencia contextual
            unique_docs = reordering.transform_documents(unique_docs)

            return unique_docs

    return ModernHybridRetriever()
