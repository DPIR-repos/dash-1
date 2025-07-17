import streamlit as st

# Configuración de la página (puedes mover esto aquí o mantenerlo en cada página)
st.set_page_config(
    page_title="Observatorio de Precios Estatales",
    page_icon="\U0001F441",
    layout="wide",
    initial_sidebar_state="expanded",
    server.enableXsrfProtection==False,
    server.enableWebsocketCompression==False,
    server.enableCORS==False,
)

import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
# Definición de las páginas
pages = {
    "Observatorio": [st.Page("observatorio.py", title="Observatorio de Precios")],
    "Documentación": [st.Page("documentacion.py", title="Documentación Técnica"), st.Page('about_source.py', title='Estructura de ficheros')],
}

# Crear la navegación
navigation = st.navigation(pages, position="top", )
navigation.run()