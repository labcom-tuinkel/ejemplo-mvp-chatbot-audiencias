import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from splitter import split_documents
from vector_store import create_vector_db
from basic_chain import get_model


# ============================================================
# SAFE GET – evita errores cuando el input es un string
# ============================================================

def safe_get(x, key, default=""):
    if isinstance(x, dict):
        return x.get(key, default)
    return default


# ============================================================
# CONTEXT STRUCTURING
# ============================================================

def structure_context(docs):
    sections = {
        "PERFILES_COMPORTAMENTALES": [],
        "NORMAS_COMUNICACION": [],
        "PROBLEMAS_AUDIENCIA": [],
        "SEGMENTACION_EDADES": [],
        "INSIGHTS_PSICOLOGICOS": [],
        "OTROS": []
    }

    for doc in docs:
        text = doc.page_content.lower()

        if "perfil a" in text or "perfil b" in text or "perfil c" in text:
            sections["PERFILES_COMPORTAMENTALES"].append(doc.page_content)

        elif "principios rectores" in text or "tono y estilo" in text:
            sections["NORMAS_COMUNICACION"].append(doc.page_content)

        elif "problemas cognitivos" in text or "problemas emocionales" in text:
            sections["PROBLEMAS_AUDIENCIA"].append(doc.page_content)

        elif "adolescencia" in text or "juventud adulta" in text or "adultez" in text:
            sections["SEGMENTACION_EDADES"].append(doc.page_content)

        elif "autoeficacia" in text or "dragones de la inactividad" in text:
            sections["INSIGHTS_PSICOLOGICOS"].append(doc.page_content)

        else:
            sections["OTROS"].append(doc.page_content)

    final_context = ""
    for name, content in sections.items():
        if content:
            final_context += f"\n\n===== {name} =====\n" + "\n".join(content)

    return final_context.strip()


# ============================================================
# PROMPT MAESTRO (resumido aquí)
# ============================================================

rag_template_es = """
Eres el Estratega de Comunicación, Audiencias y Comportamiento de la ONG *Cambia el Clima*.
Te usa principalmente una persona que diseña campañas de marketing, segmentación, mensajes
y comparativas entre perfiles comportamentales.

COMPORTAMIENTO GENERAL:
- NO apliques análisis avanzados (perfiles, segmentación, barreras, insights) a menos que
  el usuario lo solicite de forma directa o sea evidente por el objetivo actual.
- Si el usuario solo saluda o hace una pregunta genérica, responde de forma breve y clara.
- Si no queda claro a qué público, perfil o campaña se refiere, pide aclaración.
- NO inventes información: usa solo lo que aparezca en el contexto recuperado.

OBJETIVOS TÍPICOS QUE PUEDEN PREGUNTARTE:
- Identificar el perfil comportamental de una persona o segmento.
- Refinar o comparar perfiles.
- Preparar campañas de marketing (objetivo, tono, canales, mensajes clave).
- Generar mensajes (emails, posts, llamadas, copy para landings, etc.) adaptados a un perfil.
- Adaptar un mensaje existente a otro perfil o segmento.
- Analizar barreras y palancas motivacionales para un público concreto.
- Recomendar segmentación, tono y enfoque según los documentos.

USO DEL OBJETIVO ACTUAL ({current_goal}):
- "identificar_perfil": céntrate en deducir el perfil comportamental y justificarlo.
- "generar_mensaje": asume que existe un sujeto o perfil relevante y genera mensajes aplicando
  perfiles + normas + barreras + palancas.
- "asesoramiento_campaña": ayuda a diseñar estructura de campaña, mensajes clave, segmentos,
  posibles A/B tests, etc.
- "comparar_perfiles": señala diferencias prácticas entre perfiles para comunicación.

SUJETO Y PERFIL ACTIVOS EN LA CONVERSACIÓN:
- SUJETO ACTIVO: descripción actual de la persona o segmento sobre el que se trabaja.
- PERFIL ACTIVO: perfil comportamental ya identificado (si existe).
- Si el usuario dice "él", "ella", "para esa persona", "ese perfil", "lo anterior", por defecto
  se refiere al SUJETO ACTIVO y/o al PERFIL ACTIVO, salvo que especifique otra cosa.

CUANDO ACTIVES EL ANÁLISIS AVANZADO:
Debes justificar SIEMPRE usando el CONTEXTO RAG.
- Si identificas un perfil: vincúlalo con su definición y rasgos en los documentos.
- Si mencionas barreras: enlázalas con los problemas y bloqueos descritos en el contexto.
- Si usas insights psicológicos: refiérete a los conceptos (ej. autoeficacia, distancia
  psicológica, normas sociales, dragones de la inactividad, etc.).
- Si usas segmentación por edad: justifícala con la tabla/clasificación del documento.
- NO inventes elementos que no estén en los textos.

FORMATO ESPECIAL PARA PERFIL:
- Cuando identifiques o confirmes un perfil comportamental para el SUJETO ACTIVO, al final
  de la respuesta incluye SIEMPRE una línea en mayúsculas con el formato exacto:
  PERFIL_ACTUAL: <nombre o código del perfil>
  Por ejemplo:
  PERFIL_ACTUAL: Perfil B – Eco-consumidor pragmático

ESTRUCTURA RECOMENDADA (cuando sea relevante):
1. Indica brevemente qué partes del CONTEXTO estás usando (perfiles, segmentación, problemas,
   normas, insights).
2. Si procede, identifica el perfil comportamental y justifícalo con el contexto.
3. Si procede, indica la segmentación por edad relevante.
4. Señala barreras cognitivas/emocionales/prácticas relevantes para comunicación.
5. Menciona los insights psicológicos clave aplicables.
6. Aplica las normas de comunicación para construir mensajes o recomendaciones.
7. Genera el output pedido (mensaje, comparativa, idea de campaña, etc.).
8. Justifica brevemente por qué ese output es adecuado para ese perfil/objetivo.

RECUERDA:
- Si no tienes suficiente información del contexto para justificar un perfil, dilo explícitamente
  y pide más datos.
- Si la consulta es ambigua, pide aclaración antes de asumir un perfil o público.
- SIEMPRE responde en español.

-----------------------
CONTEXTO RAG:
{context}
-----------------------
SUJETO ACTIVO (si existe):
{current_subject}
-----------------------
PERFIL COMPORTAMENTAL ACTIVO (si existe):
{current_behavioral_profile}
-----------------------
OBJETIVO ACTUAL DE LA CONSULTA:
{current_goal}
-----------------------
HISTORIAL RECIENTE:
{chat_history}
-----------------------
PREGUNTA DEL USUARIO:
{question}

RESPUESTA:
"""

rag_prompt_es = ChatPromptTemplate.from_template(rag_template_es)


# ============================================================
# UTILIDADES
# ============================================================

def format_docs(docs):
    if not docs:
        return ""
    return structure_context(docs)


def get_question(input_):
    if isinstance(input_, str):
        return input_
    if isinstance(input_, dict) and "question" in input_:
        return input_["question"]
    if isinstance(input_, BaseMessage):
        return input_.content
    return ""


# ============================================================
# CADENA RAG FINAL (con SAFE_GET)
# ============================================================

def make_rag_chain(model, retriever, rag_prompt=None):
    if rag_prompt is None:
        rag_prompt = rag_prompt_es

    rag_chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnableLambda(lambda x: get_question(x)),
            "chat_history": RunnableLambda(lambda x: safe_get(x, "chat_history")),
            "current_subject": RunnableLambda(lambda x: safe_get(x, "current_subject")),
            "current_behavioral_profile": RunnableLambda(lambda x: safe_get(x, "current_behavioral_profile")),
            "current_goal": RunnableLambda(lambda x: safe_get(x, "current_goal")),
        }
        | rag_prompt
        | model
    )
    return rag_chain


# ============================================================
# TEST
# ============================================================

def main():
    load_dotenv()
    model = get_model("ChatGPT")

    from remote_loader import get_wiki_docs
    docs = get_wiki_docs(query="Cambio climático", load_max_docs=5)
    texts = split_documents(docs)
    vs = create_vector_db(texts)
    retriever = vs.as_retriever()

    chain = make_rag_chain(model, retriever)
    output = chain.invoke({
        "question": "Genera un mensaje para eco-consumidores",
        "chat_history": "",
        "current_subject": "hombre de 29 años socio",
        "current_behavioral_profile": "",
        "current_goal": "generar_mensaje"
    })

    print(output)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

