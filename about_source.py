import streamlit as st
from pathlib import Path
import os
import sys

def resource_path(file_path, local_folder_name=None):
    """
    Busca un archivo en múltiples ubicaciones, incluyendo rutas con '_internal'.
    Compatible con PyInstaller en modos OneFile y OneDirectory.
    
    Args:
        file_path (str/Path): Ruta original del archivo
        local_folder_name (str, opcional): Nombre del directorio base donde buscar.
                                          Si None, se determina automáticamente.
    
    Returns:
        Path: Ruta del archivo encontrado o la original si no se encontró
    """
    # Convertir a Path si es necesario
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # 1. Intentar con la ruta original primero
    if path_obj.exists():
        return path_obj
    
    # 2. Determinar el directorio base según el modo de empaquetado
    if getattr(sys, 'frozen', False):
        # Aplicación empaquetada
        if '_MEIPASS' in os.environ:
            # Modo OneFile - los recursos están en _MEIPASS
            base_dir = Path(os.environ['_MEIPASS'])
        else:
            # Modo OneDirectory - usar directorio del ejecutable
            base_dir = Path(sys.executable).parent
    else:
        # Modo desarrollo - usar directorio del script
        base_dir = Path(__file__).parent
    
    # 3. Si se especificó un local_folder_name, usarlo como referencia
    if local_folder_name:
        parts = list(path_obj.parts)
        try:
            base_index = parts.index(local_folder_name)
            base_parts = parts[:base_index+1]
            remaining_parts = parts[base_index+1:]
            new_parts = base_parts + ['_internal'] + remaining_parts
            new_path = Path(*new_parts)
            if new_path.exists():
                return new_path
        except ValueError:
            pass  # Continuar con la lógica normal si no se encuentra el folder
    
    # 4. Intentar rutas alternativas (versión corregida)
    possible_paths = []
    
    # Ruta directa desde el directorio base
    possible_paths.append(base_dir / path_obj.name)
    
    # Ruta manteniendo estructura pero desde base_dir
    if not path_obj.is_absolute():
        possible_paths.append(base_dir / path_obj)
    else:
        possible_paths.append(path_obj)
    
    # Ruta con _internal (versión simplificada para evitar errores de sintaxis)
    if not path_obj.is_absolute():
        possible_paths.append(base_dir / '_internal' / path_obj)
    else:
        try:
            base_index = path_obj.parts.index(base_dir.name)
            new_parts = (path_obj.parts[:base_index+1] + ('_internal',) + 
                        path_obj.parts[base_index+1:])
            possible_paths.append(Path(*new_parts))
        except ValueError:
            pass
    
    # Ruta temporal de OneFile (si aplica)
    if getattr(sys, 'frozen', False) and '_MEIPASS' in os.environ:
        possible_paths.append(base_dir / '_internal' / path_obj.name)
    
    # Filtrar paths válidos y verificar existencia
    for test_path in possible_paths:
        try:
            if test_path.exists():
                return test_path
        except (TypeError, AttributeError):
            continue
    
    # 5. Como último recurso, intentar con el directorio de trabajo actual
    test_path = Path.cwd() / path_obj if not path_obj.is_absolute() else path_obj
    if test_path.exists():
        return test_path
    
    # Si no se encontró en ninguna ruta, devolver la original
    return path_obj

def show():
    # Logo del INE
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://www.ine.gob.gt/ine/wp-content/uploads/2017/09/cropped-INE.png" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    current_folder = Path(os.getcwd())
    soruce_folder = "source_data"
    st.sidebar.markdown(" ")
    st.sidebar.image(resource_path(current_folder/soruce_folder/"DPIR_logo_2.png"))
    
    # Tabla de contenido en el sidebar
    st.sidebar.markdown("## Tabla de Contenido")
    st.sidebar.markdown("""
    - [Acerca del Observatorio de Precios](#acerca-del-observatorio-de-precios)
    - [Funcionalidades principales](#funcionalidades-principales)
    - [Metodología](#metodología)
    - [Detalles técnicos](#detalles-técnicos-de-los-gráficos)
    - [Cómo usar la aplicación](#cómo-usar-la-aplicación)
    """)
    
    st.title("📚 Documentación Técnica")
    
    st.markdown("""
    <a id="acerca-del-observatorio-de-precios"></a>
    ## Acerca del Observatorio de Precios
    
    Esta aplicación permite analizar los precios de insumos adjudicados en Guatecompras,
    con ajustes por inflación y visualizaciones interactivas.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <a id="funcionalidades-principales"></a>
    ### Funcionalidades principales:
    - Análisis de precios por insumo y variedad
    - Corrección por inflación (nacional/regional)
    - Visualizaciones geográficas por departamento
    - Series temporales de evolución de precios
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <a id="metodología"></a>
    ### Metodología:
    Los precios se calculan usando la media geométrica ponderada por cantidad ofertada,
    con intervalos de confianza ajustables.
    """, unsafe_allow_html=True)
    
    with st.expander("📊 Detalles técnicos de los gráficos"):
        st.markdown("""
        <a id="detalles-técnicos-de-los-gráficos"></a>
        - **Histogramas**: Muestran distribución de precios con KDE (Kernel Density Estimation). El número de bins para el histograma se determina
                           siguiendo la regla de Sturges, cuya fórmula matematica es $$N_{bins}=1+log_{2}(M)$$ donde $$M$$ es el número total de observaciones. 
                           En este caso el número total de observaciones es igual a la suma del número total de unidades ofertadas.  
        - **Mapas**: Coropleticos por departamento con precios promedio
        - **Series temporales**: Evolución mensual con bandas de desviación estándar
        """, unsafe_allow_html=True)
    
    with st.expander("🛠️ Cómo usar la aplicación"):
        st.markdown("""
        <a id="cómo-usar-la-aplicación"></a>
        1. Seleccione el año de interés en el sidebar
        2. Elija el código de insumo (puede buscar por código o descripción)
        3. Filtre por variedad si es necesario
        4. Active corrección por inflación si lo requiere
        5. Explore los gráficos y tablas generadas
        """, unsafe_allow_html=True)
        
    col1Logo, col2Logo = st.columns([9, 1])

    with col2Logo:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.image(resource_path(current_folder / soruce_folder / "DPIR_logo_2.png"), width=120)

# Llamada a la función principal
show()