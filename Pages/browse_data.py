import streamlit as st

from local_loader import list_txt_files

st.title("Explorar archivos de datos")

paths = list(list_txt_files())

file_path = st.selectbox("Selecciona un archivo para visualizar", paths, index=None)

if file_path:
    st.subheader(f"Contenido de: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            st.markdown(content)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
