import requests
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_core.documents import Document
from local_loader import get_document_text
import wikipedia


# Carpeta donde se guardarán archivos descargados
CONTENT_DIR = os.path.dirname(__file__)


def load_web_page(page_url):
    """Carga contenido de una página web."""
    loader = WebBaseLoader(page_url)
    return loader.load()


def load_online_pdf(pdf_url):
    """Carga PDF remoto."""
    loader = OnlinePDFLoader(pdf_url)
    return loader.load()


def filename_from_url(url):
    return url.split("/")[-1]


def download_file(url, filename=None):
    """Descarga un archivo al directorio local."""
    response = requests.get(url)
    filename = filename or filename_from_url(url)
    full_path = os.path.join(CONTENT_DIR, filename)

    with open(full_path, "wb") as f:
        f.write(response.content)

    print(f"Archivo descargado en {full_path}")
    return full_path


def get_wiki_docs(query, load_max_docs=2):
    """
    Carga artículos de Wikipedia usando la librería oficial.
    Devuelve Document() compatibles con LangChain moderno.
    """
    wikipedia.set_lang("es")  # opcional: selecciona idioma

    pages = wikipedia.search(query, results=load_max_docs)
    docs = []

    for title in pages:
        try:
            page = wikipedia.page(title)
            docs.append(
                Document(
                    page_content=page.content,
                    metadata={"title": title, "url": page.url}
                )
            )
        except:
            pass

    return docs


def main():
    print("Módulo remote_loader listo. No se ejecuta directamente.")


if __name__ == "__main__":
    main()
