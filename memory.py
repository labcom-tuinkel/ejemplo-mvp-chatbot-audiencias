import os
from typing import List, Iterable, Any

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory

from basic_chain import get_model
from rag_chain import make_rag_chain


def create_memory_chain(llm, base_chain, chat_memory):
    contextualize_q_system_prompt = """
Dado el historial de la conversación y la última pregunta del usuario,
reformula la pregunta para que pueda entenderse de manera independiente,
sin necesitar el historial previo.

NO respondas la pregunta.
Si no es necesario reformular, deja la pregunta tal como está.
"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    runnable = contextualize_q_prompt | llm | base_chain

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        return chat_memory

    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


class SimpleTextRetriever(BaseRetriever):
    docs: List[Document]
    """Documentos base."""

    @classmethod
    def from_texts(cls, texts: Iterable[str], **kwargs: Any):
        docs = [Document(page_content=t) for t in texts]
        return cls(docs=docs, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self.docs


def main():
    load_dotenv()
    model = get_model("ChatGPT")
    chat_memory = ChatMessageHistory()

    system_prompt = """
Eres un asistente diseñado para generar información y mensajes basados en el contexto proporcionado.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    text_path = "examples/grocery.md"
    text = open(text_path, "r").read()
    retriever = SimpleTextRetriever.from_texts([text])
    rag_chain = make_rag_chain(model, retriever, rag_prompt=None)

    chain = create_memory_chain(model, rag_chain, chat_memory) | StrOutputParser()

    queries = [
        "¿Qué información contiene este texto?",
        "¿Puedes resumirlo?",
    ]

    for query in queries:
        print(f"\nPregunta: {query}")
        response = chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "foo"}}
        )
        print(f"Respuesta: {response}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
