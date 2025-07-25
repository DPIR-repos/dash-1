import streamlit as st

# Configuración de la página (puedes mover esto aquí o mantenerlo en cada página)

import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

st.set_page_config(
    page_title="Observatorio de Precios Estatales",
    page_icon="\U0001F441",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definición de las páginas
pages = {
    "Herramientas": [st.Page("observatorio.py", title="Observatorio de Precios"), st.Page("busquedas_personalizadas.py", title="Busquedas Personalizadas") ],
    "Documentación": [st.Page("documentacion.py", title="Documentación Técnica"), st.Page('about_source.py', title='Estructura de ficheros')],
}

# Crear la navegación
navigation = st.navigation(pages, position="top", )
navigation.run()