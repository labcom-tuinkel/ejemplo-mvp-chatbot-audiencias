from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(docs):
    """
    Divide documentos en chunks compatibles con LangChain moderno, 
    preservando metadatos cuando existen.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )

    processed_docs = []

    for doc in docs:
        if isinstance(doc, Document):
            # Preserva metadatos al dividir
            chunks = text_splitter.split_documents([doc])
            processed_docs.extend(chunks)
        else:
            # Caso donde doc es string
            chunks = text_splitter.create_documents([doc])
            processed_docs.extend(chunks)

    print(f"Split into {len(processed_docs)} chunks")
    return processed_docs
