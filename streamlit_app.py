import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings

from local_loader import load_txt_files
from filter import create_retriever   # 🔥 Nuevo retriever híbrido con documentos core
from rag_chain import make_rag_chain  # 🔥 Nuevo RAG maestro (contexto estructurado + reglas duras)
from basic_chain import get_model      # 🔥 Modelo base que respeta identidad y normas


# --------------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------------------------

st.set_page_config(page_title="Chat RAG en Español – Públicos Objetivo")
st.title("Chat RAG en Español – Públicos Objetivo")


# --------------------------------------------------------------
# INTERFAZ DE CHAT
# --------------------------------------------------------------

def show_ui(chain, prompt_to_user="¿En qué puedo ayudarte? Puedes pedirme que genere mensajes adaptados a tu público objetivo."):

    # Inicializa historial si no existe
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Capturar mensaje del usuario
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generar respuesta solo si el último mensaje no es del asistente
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generando respuesta..."):
                response = chain.invoke(prompt)

                # extraer texto del objeto devuelto
                text = response.content if hasattr(response, "content") else str(response)

                st.markdown(text)
                st.session_state.messages.append({"role": "assistant", "content": text})



# --------------------------------------------------------------
# RETRIEVER – Con vectorización y documentos CORE
# --------------------------------------------------------------

@st.cache_resource
def get_retriever(openai_api_key=None):
    docs = load_txt_files()  # Carga los Document()
    
    # Creamos embeddings SOLO para vectorización dentro de create_vector_db
    # Este embeddings no se usa directamente aquí.
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model="text-embedding-3-small"
    )

    # Usamos tu retriever híbrido mejorado
    retriever = create_retriever(docs)
    return retriever


# --------------------------------------------------------------
# CONSTRUCCIÓN DE LA CADENA COMPLETA (RAG + memoria)
# --------------------------------------------------------------

def get_chain(openai_api_key=None):

    retriever = get_retriever(openai_api_key=openai_api_key)

    model = get_model(openai_api_key=openai_api_key)

    # Chat memory para conservar el hilo conversacional
    memory = StreamlitChatMessageHistory(key="langchain_messages")

    # Tu cadena final: RAG maestro → modelo
    chain = make_rag_chain(model, retriever)

    return chain


# --------------------------------------------------------------
# GESTIÓN DE SECRETS Y API KEYS
# --------------------------------------------------------------

def get_secret_or_input(secret_key, secret_name, info_link=None):

    safe_secrets = getattr(st, "secrets", {})

    if secret_key in safe_secrets:
        st.write(f"Found {secret_name} en secrets.toml")
        secret_value = safe_secrets[secret_key]

    else:
        st.write(f"Introduce tu {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")

        if secret_value:
            st.session_state[secret_key] = secret_value

        if info_link:
            st.markdown(f"[Obtener {secret_name}]({info_link})")

    return secret_value


# --------------------------------------------------------------
# EJECUCIÓN PRINCIPAL
# --------------------------------------------------------------

def run():
    ready = True

    openai_api_key = st.session_state.get("OPENAI_API_KEY")

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input(
                "OPENAI_API_KEY",
                "OpenAI API key",
                info_link="https://platform.openai.com/account/api-keys"
            )

    if not openai_api_key:
        st.warning("Falta OPENAI_API_KEY")
        ready = False

    if ready:
        chain = get_chain(openai_api_key=openai_api_key)
        st.subheader("Haz preguntas o pide que genere mensajes adaptados a tu público objetivo.")
        show_ui(chain, "¿Qué mensaje quieres generar o qué deseas saber sobre tu público?")
    else:
        st.stop()


run()
