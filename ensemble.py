from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from splitter import split_documents
from vector_store import create_vector_db


def safe_content(d):
    """Acceso seguro al contenido, compatible con LangChain 0.2+ y Pydantic."""
    if isinstance(d, Document):
        return d.page_content
    
    try:
        return d.dict().get("page_content", "")
    except Exception:
        return str(d)


def ensemble_retriever_from_docs(docs, embeddings=None):

    # 1. Split de documentos
    texts = split_documents(docs)

    # 2. Crea vector store retriever moderno
    vector_store = create_vector_db(texts, embeddings)
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 3. BM25 moderno (usa invoke, no get_relevant_documents)
    bm25_retriever = BM25Retriever.from_texts(
        [t.page_content for t in texts]
    )

    class HybridRetriever(BaseRetriever):

        def _get_relevant_documents(self, query, *, run_manager=None):

            # LA API CORRECTA EN LANGCHAIN 0.2+
            docs_sem = semantic_retriever.invoke(query)
            docs_bm25 = bm25_retriever.invoke(query)

            # Fusionar eliminando duplicados
            seen = set()
            merged = []

            for d in docs_sem + docs_bm25:
                content = safe_content(d)
                if content not in seen:
                    seen.add(content)
                    # Convertir a Document siempre
                    merged.append(Document(page_content=content))

            return merged

    return HybridRetriever()
