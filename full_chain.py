import os
import re
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

from basic_chain import get_model
from rag_chain import make_rag_chain
from local_loader import load_txt_files


# --------------------------------------------------------
# DETECTORES DE INTENCIÓN Y EXTRACTORES DE SUJETO / PERFIL
# --------------------------------------------------------

def extract_subject_description(text):
    """
    Detecta si el usuario describe a una persona real para análisis.
    Ejemplos:
    - "soy un hombre de 29 años..."
    - "una mujer de 40 años colaboradora..."
    """

    patterns = [
        r"soy\s+[^\n]+",
        r"persona\s+[^\n]+",
        r"(hombre|mujer)[^\n]+años",
        r"\b\d{2}\s*años\b",
        r"socio",
        r"colaborador",
        r"voluntari[oa]",
    ]

    for p in patterns:
        match = re.search(p, text.lower())
        if match:
            return text  # almacenamos toda la descripción
    return None


def user_is_asking_for_profile(text):
    """
    Detecta si el usuario quiere identificar un perfil.
    """
    keywords = [
        "qué perfil", "que perfil", "que comportamiento", "qué comportamiento",
        "a qué perfil corresponde", "que sería", "qué sería"
    ]
    return any(k in text.lower() for k in keywords)


def user_is_asking_for_message(text):
    """
    Detecta si el usuario quiere generar un mensaje.
    """
    keywords = [
        "haz un mensaje", "genera un mensaje", "mensaje para", "email",
        "correo", "escribe un mensaje", "invitar", "convocar", "campaña"
    ]
    return any(k in text.lower() for k in keywords)


def user_is_comparing_profiles(text):
    """
    Detecta comparaciones.
    """
    keywords = ["compara", "diferencia", "qué diferencia", "cuál es mejor"]
    return any(k in text.lower() for k in keywords)


def user_is_preparing_campaign(text):
    """
    Detecta consultas de campaña.
    """
    keywords = ["campaña", "segmentación", "target", "público objetivo"]
    return any(k in text.lower() for k in keywords)


# --------------------------------------------------------
# CADENA PRINCIPAL
# --------------------------------------------------------

def create_full_chain(retriever, openai_api_key=None, chat_memory=None):

    model = get_model("ChatGPT", openai_api_key=openai_api_key)

    if chat_memory is None:
        chat_memory = ChatMessageHistory()

    rag_chain = make_rag_chain(model, retriever)

    # ---- CONTENEDOR DE ESTADO ----
    class MemoryWrappedChain:
        def __init__(self, rag_chain, chat_memory):
            self.rag_chain = rag_chain
            self.chat_memory = chat_memory

            self.current_subject = ""          # Descripción textual del sujeto
            self.current_behavioral_profile = ""  # Perfil comportamental asignado
            self.current_goal = ""             # Objetivo comunicacional actual (mensaje/campaña/etc.)

        def update_state_from_query(self, query):

            # A) Detectamos si describe un sujeto nuevo
            subject = extract_subject_description(query)
            if subject:
                self.current_subject = subject
                self.current_behavioral_profile = ""  # obligamos recalcular si lo piden

            # B) Identificamos tipo de consulta
            if user_is_asking_for_profile(query):
                self.current_goal = "identificar_perfil"

            elif user_is_asking_for_message(query):
                self.current_goal = "generar_mensaje"

            elif user_is_preparing_campaign(query):
                self.current_goal = "asesoramiento_campaña"

            elif user_is_comparing_profiles(query):
                self.current_goal = "comparar_perfiles"

        def invoke(self, user_query):

            # 1. Actualizar estado semántico
            self.update_state_from_query(user_query)

            # 2. Guardar mensaje usuario
            self.chat_memory.add_user_message(user_query)

            # 3. Historial compacto
            history_text = "\n".join(
                f"{m.type.upper()}: {m.content}"
                for m in self.chat_memory.messages[-6:]
            )

            # 4. Ejecutar RAG con estado completo del usuario
            response = self.rag_chain.invoke({
                "question": user_query,
                "chat_history": history_text,
                "current_subject": self.current_subject,
                "current_behavioral_profile": self.current_behavioral_profile,
                "current_goal": self.current_goal,
            })

            # 5. Guardar respuesta
            self.chat_memory.add_ai_message(response.content)

            # 6. Actualizar perfil comportamental si el asistente lo identifica
            if "PERFIL_ACTUAL:" in response.content:
                extracted = re.search(r"PERFIL_ACTUAL:\s*(.*)", response.content)
                if extracted:
                    self.current_behavioral_profile = extracted.group(1).strip()

            return response

    return MemoryWrappedChain(rag_chain, chat_memory)


# -----------------
# ASK_QUESTION WRAPPER
# -----------------

def ask_question(chain, query):
    return chain.invoke(query)


# -----------------
# LOCAL TEST
# -----------------

def main():
    load_dotenv()

    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()

    docs = load_txt_files()
    from ensemble import ensemble_retriever_from_docs
    retriever = ensemble_retriever_from_docs(docs)

    chain = create_full_chain(retriever)

    queries = [
        "Soy un hombre de 29 años socio, padre, vivo en Madrid y colaboro cada año.",
        "¿Qué perfil sería yo?",
        "Haz un mensaje para él invitándole a un evento de recaudación",
        "¿Para qué perfil es ese mensaje?",
        "Compárame este perfil con un adulto mayor activo",
        "Quiero una campaña para este perfil",
    ]

    for q in queries:
        response = ask_question(chain, q)
        console.print(Markdown(response.content))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
