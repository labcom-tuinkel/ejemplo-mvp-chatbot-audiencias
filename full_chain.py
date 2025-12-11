import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from basic_chain import get_model
from ensemble import ensemble_retriever_from_docs
from local_loader import load_txt_files
from memory import create_memory_chain
from rag_chain import make_rag_chain


def create_full_chain(retriever, openai_api_key=None, chat_memory=ChatMessageHistory()):
    model = get_model("ChatGPT", openai_api_key=openai_api_key)

    system_prompt = """
    Eres un asistente experto en comunicación y generación de mensajes adaptados a públicos específicos.
    Utiliza la información del contexto (proveniente del RAG) y el historial del usuario para:

    - responder preguntas sobre el público objetivo,
    - generar mensajes personalizados según las características del público,
    - recomendar tonos, estilos y enfoques comunicativos,
    - redactar textos persuasivos, informativos o emocionales,
    - reescribir mensajes adaptados a ese público,
    - mantener coherencia durante la conversación.

    Si no puedes responder con la información disponible, di que no tienes datos suficientes.

    Contexto relevante: {context}

    Pregunta o tarea del usuario:
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    rag_chain = make_rag_chain(model, retriever, rag_prompt=prompt)
    chain = create_memory_chain(model, rag_chain, chat_memory)
    return chain


def ask_question(chain, query):
    return chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": "foo"}}
    )


def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    retriever = ensemble_retriever_from_docs(docs)
    chain = create_full_chain(retriever)

    queries = [
        "Generate a grocery list for my family meal plan for the next week (7 days). Prefer local, in-season ingredients.",
        "Create a list of estimated calorie counts and grams of carbohydrates for each meal."
    ]

    for query in queries:
        response = ask_question(chain, query)
        console.print(Markdown(response.content))


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
