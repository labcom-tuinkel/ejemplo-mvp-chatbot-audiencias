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


# ============================================================
# LISTA DE DOCUMENTOS "CORE" QUE SIEMPRE DEBEN APARECER
# ============================================================

CORE_KEYWORDS = {
    "PERFILES": ["perfil a", "perfil b", "perfil c", "perfil d", "perfiles comportamentales"],
    "NORMAS": ["principios rectores", "política y reglas de comunicación", "tono general"],
    "PROBLEMAS": ["problemas cognitivos", "ecoansiedad", "baja autoeficacia", "polarización"],
    "SEGMENTACION": ["adolescencia", "juventud adulta", "adultez media", "adultez madura", "senior"],
    "INSIGHTS": ["autoeficacia", "inercia", "dragones", "normas sociales", "distancia psicológica"]
}


def is_core_doc(doc: Document):
    """Determina si un documento pertenece a los esenciales."""
    text = doc.page_content.lower()
    return any(keyword in text for keywords in CORE_KEYWORDS.values() for keyword in keywords)


# ============================================================
# RETRIEVER MEJORADO
# ============================================================

def create_retriever(texts):
    """
    Retriever híbrido mejorado:
    - Recupera documentos por similitud híbrida
    - Añade SIEMPRE documentos core
    - Filtra redundancia
    - Reordena para coherencia contextual
    """

    # === Embeddings densos y esparsos ===
    dense_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    sparse_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-large-en",
        encode_kwargs={'normalize_embeddings': False}
    )

    dense_vs = create_vector_db(texts, collection_name="dense", embeddings=dense_embeddings)
    sparse_vs = create_vector_db(texts, collection_name="sparse", embeddings=sparse_embeddings)

    dense_retriever = dense_vs.as_retriever(search_kwargs={"k": 3})
    sparse_retriever = sparse_vs.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    redundant_filter = EmbeddingsRedundantFilter(embeddings=sparse_embeddings)
    reordering = LongContextReorder()

    # === SELECCIÓN PREVIA: documentos core ===
    core_docs = [d for d in texts if is_core_doc(d)]
    core_docs = core_docs[:8]  # evita meter demasiado contexto

    # ============================================================
    # Modern Retriever con priorización
    # ============================================================

    class ModernHybridRetriever(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):

            # 1 — Recuperación híbrida clásica
            docs = (
                dense_retriever.invoke(query)
                + sparse_retriever.invoke(query)
                + bm25_retriever.invoke(query)
            )


            # 2 — Añadir documentos core SIEMPRE
            docs = core_docs + docs

            # 3 — Eliminar duplicados preservando orden
            seen = set()
            unique_docs = []
            for d in docs:
                if d.page_content not in seen:
                    unique_docs.append(d)
                    seen.add(d.page_content)

            # 4 — Filtrar redundancias
            unique_docs = redundant_filter.transform_documents(unique_docs)

            # 5 — Reorganizar para coherencia
            unique_docs = reordering.transform_documents(unique_docs)

            return unique_docs

    return ModernHybridRetriever()
