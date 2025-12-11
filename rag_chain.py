import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from splitter import split_documents
from vector_store import create_vector_db

from basic_chain import basic_chain, get_model


# === PROMPT RAG PERSONALIZADO EN ESPAÑOL ===

rag_template_es = """
Eres un asistente experto en comunicación y generación de mensajes adaptados a públicos específicos.
Utiliza exclusivamente la información del contexto proporcionado, que describe al público objetivo:
perfil, necesidades, motivaciones, lenguaje recomendado, estilo comunicativo, insights y cualquier dato relevante.

Tu función es:
- responder preguntas sobre ese público,
- generar mensajes redactados específicamente para él,
- proponer ideas de comunicación,
- ajustar el tono y estilo según lo descrito,
- crear textos persuasivos, informativos o emocionales,
- reescribir mensajes adaptándolos al público objetivo.

SIEMPRE responde en español.

-----------------------
CONTEXTO:
{context}
-----------------------

PREGUNTA O TAREA:
{question}

RESPUESTA:
"""

rag_prompt_es = ChatPromptTemplate.from_template(rag_template_es)


# === UTILS ===

def format_docs(docs):
    """Convierte una lista de Document() en texto listo para el prompt."""
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    """Extrae la pregunta del usuario en múltiples formatos posibles."""
    if not input:
        return None

    if isinstance(input, str):
        return input

    if isinstance(input, dict) and "question" in input:
        return input["question"]

    if isinstance(input, BaseMessage):
        return input.content

    raise ValueError("Entrada no válida para RAG: se esperaba string, mensaje o {'question': texto}.")


def make_rag_chain(model, retriever, rag_prompt=None):
    """Crea una cadena RAG moderna usando LangChain."""

    if not rag_prompt:
        rag_prompt = rag_prompt_es

    rag_chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | model
    )

    return rag_chain


# === TEST (solo si ejecutas este archivo directamente) ===

def main():
    load_dotenv()

    model = get_model("ChatGPT")

    # Carga de documentos para ejemplo
    from remote_loader import get_wiki_docs
    docs = get_wiki_docs(query="Bertrand Russell", load_max_docs=5)
    texts = split_documents(docs)
    vs = create_vector_db(texts)

    retriever = vs.as_retriever()

    output_parser = StrOutputParser()
    rag_chain = make_rag_chain(model, retriever) | output_parser

    questions = [
        "¿Cuáles fueron los aportes filosóficos más importantes de Russell?",
        "¿Cuál fue su primer libro publicado?",
        "¿Por qué fue importante 'An Essay on the Foundations of Geometry'?"
    ]

    for q in questions:
        print("\n--- PREGUNTA:", q)
        print("* RAG:\n", rag_chain.invoke(q))


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
