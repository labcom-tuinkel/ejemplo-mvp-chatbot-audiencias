import os
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader


# ==========================================================
# LISTAR ARCHIVOS .TXT
# ==========================================================

def list_txt_files(data_dir="./data"):
    """Devuelve todos los archivos TXT dentro de /data."""
    paths = Path(data_dir).glob("**/*.txt")
    for path in paths:
        yield str(path)


# ==========================================================
# LOADER SEGURO PARA ARCHIVOS DE TEXTO
# ==========================================================

def safe_load_text(path):
    """
    Intenta cargar un archivo TXT probando varias codificaciones.
    Evita el error de RuntimeError de TextLoader en Windows.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()

            return [Document(page_content=text, metadata={"source": path})]

        except Exception:
            continue

    raise RuntimeError(f"No se pudo cargar el archivo con ninguna codificación: {path}")


# ==========================================================
# CARGADOR PRINCIPAL DE ARCHIVOS TXT
# ==========================================================

def load_txt_files(data_dir="./data"):
    """
    Carga todos los TXT ignorando errores de codificación.
    """
    docs = []
    paths = list_txt_files(data_dir)

    for path in paths:
        print(f"Cargando archivo: {path}")

        try:
            docs.extend(safe_load_text(path))
        except Exception as e:
            print(f"⚠️  ERROR cargando {path}: {e}")
            print("❌ Archivo omitido.\n")
            continue

    return docs


# ==========================================================
# CARGAR CSV
# ==========================================================

def load_csv_files(data_dir="./data"):
    docs = []
    paths = Path(data_dir).glob("**/*.csv")
    for path in paths:
        loader = CSVLoader(file_path=str(path))
        docs.extend(loader.load())
    return docs


# ==========================================================
# PROCESAR PDF O TXT SUBIDO
# ==========================================================

def get_document_text(uploaded_file, title=None):
    """
    Convierte PDF o TXT en Document() compatible con el pipeline RAG moderno.
    """
    docs = []
    fname = uploaded_file.name

    if not title:
        title = os.path.basename(fname)

    # PDF
    if fname.lower().endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)

        for num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            docs.append(
                Document(
                    page_content=page_text,
                    metadata={"title": title, "page": num + 1},
                )
            )

    # Texto plano
    else:
        try:
            text = uploaded_file.read().decode("utf-8")
        except:
            text = uploaded_file.read().decode("latin-1")

        docs.append(Document(page_content=text, metadata={"title": title}))

    return docs
