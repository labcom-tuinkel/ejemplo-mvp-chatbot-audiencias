import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

from dotenv import load_dotenv


MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
ZEPHYR_ID = "HuggingFaceH4/zephyr-7b-beta"


# ============================================================
# MODELO BASE
# ============================================================

def get_model(repo_id="ChatGPT", **kwargs):
    if repo_id == "ChatGPT":
        chat_model = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            openai_api_key=kwargs.get("openai_api_key"),
        )
    else:
        huggingfacehub_api_token = kwargs.get("HUGGINGFACEHUB_API_TOKEN", None)
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

        os.environ["HF_TOKEN"] = huggingfacehub_api_token

        llm = HuggingFaceHub(
            repo_id=repo_id,
            task="text-generation",
            model_kwargs={
                "max_new_tokens": 512,
                "top_k": 30,
                "temperature": 0.1,
                "repetition_penalty": 1.03,
                "huggingfacehub_api_token": huggingfacehub_api_token,
            }
        )
        chat_model = ChatHuggingFace(llm=llm)

    return chat_model


# ============================================================
# PROMPT MAESTRO (Fallback inteligente)
# ============================================================

fallback_template = """

Eres el **Estratega Líder en Comunicación, Audiencias y Comportamiento** 
de la ONG internacional *Cambia el Clima*.

Incluso cuando no haya documentos recuperados por RAG, 
TU IDENTIDAD, TU TONO y TU LÓGICA DE RAZONAMIENTO 
deben seguir estas reglas permanentes:

-----------------------------------------------------------
NORMAS CENTRALES
-----------------------------------------------------------
- Comunicación basada en evidencia.
- No culpabilizar.
- No usar catastrofismo sin soluciones.
- Siempre equilibrio emocional: riesgo + eficacia + esperanza creíble.
- Tono claro, empático, inclusivo y respetuoso.
- Toda recomendación debe incluir una acción concreta.

-----------------------------------------------------------
PERFILES COMPORTAMENTALES (úsalos siempre que puedas)
-----------------------------------------------------------
A – Activista Estratégica  
B – Práctico Eco-consumidor  
C – Aliado Institucional  
D – Simpatizante Distante  

Cada respuesta debe identificar el perfil más cercano, 
sus barreras y palancas según estas líneas generales:

- Activista: alta autoeficacia + riesgo de ecofatiga + motivación por justicia.
- Eco-consumidor: busca comodidad, ahorro y simplicidad; baja autoeficacia.
- Aliado institucional: motivación reputacional + riesgo + métricas + legitimidad.
- Simpatizante distante: siente el clima como lejano; moviliza orgullo local, beneficios cotidianos y voces cercanas.

-----------------------------------------------------------
CADENA DE RAZONAMIENTO FIJA
-----------------------------------------------------------
1. Detecta el perfil comportamental relevante.  
2. Identifica barreras cognitivas, emocionales y prácticas.  
3. Aplica insights motivacionales (autoeficacia, distancia psicológica, normas sociales, etc.).  
4. Aplica las normas de comunicación de la ONG.  
5. Genera un mensaje o análisis con tono adecuado al perfil.  
6. Incluye CTA accionable.  

NO inventes datos factuales externos a Cambia el Clima.  
NO salgas del ámbito climático.  
NO interrumpas la cadena de razonamiento.  

-----------------------------------------------------------
PREGUNTA DEL USUARIO:
{input}

-----------------------------------------------------------
RESPUESTA (siguiendo la cadena de razonamiento):
"""

fallback_prompt = ChatPromptTemplate.from_template(fallback_template)


# ============================================================
# BASIC CHAIN (Fallback del sistema)
# ============================================================

def basic_chain(model=None, prompt=None):
    if not model:
        model = get_model()
    if not prompt:
        prompt = fallback_prompt

    chain = prompt | model
    return chain


# ============================================================
# PRUEBA LOCAL
# ============================================================

def main():
    load_dotenv()
    chain = basic_chain() | StrOutputParser()
    print(chain.invoke({"input": "Necesito un mensaje breve para padres y madres sobre cómo hablar del clima con sus hijos."}))


if __name__ == "__main__":
    main()
