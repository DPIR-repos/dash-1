import streamlit as st
from pathlib import Path
import os
import pandas as pd

def resource_path(file_path, local_folder_name=None):
    """
    Función para manejar rutas de archivos en diferentes entornos (desarrollo/empaquetado)
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    if path_obj.exists():
        return path_obj
    
    base_dir = Path(__file__).parent
    possible_paths = [
        base_dir / path_obj.name,
        base_dir / path_obj,
        base_dir / '_internal' / path_obj,
        Path.cwd() / path_obj
    ]
    
    for test_path in possible_paths:
        try:
            if test_path.exists():
                return test_path
        except (TypeError, AttributeError):
            continue
    
    return path_obj

def show():
    # Configuración de la página
    st.set_page_config(
        page_title="Datos de Compras Públicas Guatemala", 
        layout="wide",
        page_icon="📊"
    )
    
    # Logo del INE (copiado exactamente del ejemplo)
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://www.ine.gob.gt/ine/wp-content/uploads/2017/09/cropped-INE.png" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Logo de la aplicación (copiado exactamente del ejemplo)
    current_folder = Path(os.getcwd())
    source_folder = "source_data"
    st.sidebar.image(resource_path(current_folder/source_folder/"DPIR_logo_2.png"))
    
    # Título principal (copiado del ejemplo con ajuste de texto)
    st.title("📊 Estructura de la base de datos de compras públicas")
    st.markdown("""
    Documentación técnica del archivo CSV que contiene los registros de compras públicas adjudicadas en Guatemala.
    """)
    
    # Descripción detallada del archivo CSV (estructura copiada del ejemplo)
    st.markdown("""
    ## Archivo principal: compras_publicas_2020.csv
    
    **Descripción**: 
    Este dataset contiene todos los registros históricos de compras públicas adjudicadas a través del sistema Guatecompras,
    con información detallada sobre productos, proveedores, entidades compradoras y montos.
    """)
    
    # Estructura de columnas (formato de tabla copiado exactamente del ejemplo)
    st.markdown("""
    ### Estructura de columnas:
    
    | Nombre Columna | Tipo de Dato | Descripción | Ejemplo |
    |---------------|--------------|-------------|---------|
    | Filename | Texto | Nombre del archivo origen | "10572740_2020_01_webpage_source.txt" |
    | NOG | Texto | Número de operación gubernamental | "10572740" |
    | Descripcion | Texto | Descripción detallada del producto | "Set de placa de compresión dinámica" |
    | Grupo | Texto | Código de grupo | "2.0" |
    | Subgrupo | Texto | Código de subgrupo | "29.0" |
    | Renglon | Texto | Código de renglón | "295.0" |
    | Codigo Insumo | Texto | Código único del insumo | "65370.0" |
    | Insumo Match | Texto | Descripción normalizada | "Set de placa de compresión dinámica" |
    | Score | Decimal | Puntaje de coincidencia | 0.9069 |
    | Producto | Texto | Nombre del producto | "Set de placa de compresión dinámica" |
    | Marca | Texto | Marca del producto | "Asco Medical" |
    | Unidad de Medida | Texto | Unidad de medida | "Unidad - 1 Unidad" |
    | Cantidad Ofertada | Decimal | Cantidad adjudicada | 10.0 |
    | Precio unitario | Decimal | Precio por unidad | 2950.0 |
    | Monto ofertado | Decimal | Monto total | 29500.0 |
    | Caracteristicas | Texto | Especificaciones técnicas | "Incluye: Placa de minifragmentos..." |
    | Modalidad | Texto | Modalidad de compra | "Compra Directa con Oferta Electrónica" |
    """)
    
    # Sección de usos y aplicaciones (estructura copiada del ejemplo)
    st.markdown("""
    ## Aplicaciones principales
    
    ### 1. Análisis de distribución de precios
    - Histogramas de frecuencia de precios por insumo
    - Comparación de rangos de precios entre categorías
    - Identificación de valores atípicos
    
    ### 2. Evolución temporal
    - Tendencia de precios mensuales/anuales
    - Comparación entre períodos
    - Efectos de inflación en precios
    
    ### 3. Análisis geográfico
    - Mapeo de compras por departamento/municipio
    - Identificación de zonas con mayores transacciones
    - Diferencias regionales
    
    ### 4. Benchmarking de proveedores
    - Comparación de precios entre proveedores
    - Análisis de concentración de mercado
    - Detección de posibles patrones anómalos
    """)
    
    # Sección técnica (copiada exactamente del ejemplo con ajuste de contenido)
    with st.expander("📝 Notas técnicas y metadatos"):
        st.markdown("""
        - **Codificación de archivo**: UTF-8
        - **Separador de campos**: Coma (,)
        - **Formato fechas**: Campos separados (día, mes, año)
        - **Precisión decimal**: 2 dígitos para montos
        - **Periodicidad de actualización**: Mensual
        - **Cobertura temporal**: Enero 2020
        - **Cobertura geográfica**: Todos los departamentos de Guatemala
        """)
    
    # Ejemplo de datos (estructura copiada exactamente del ejemplo)
    with st.expander("🖥️ Ejemplo de registros (estructura)"):
        sample_data = {
            "Filename": ["10572740_2020_01_webpage_source.txt", "11600802_2020_01_webpage_source.txt"],
            "Descripcion": ["Set de placa de compresión dinámica", "ADQUISICIÓN DE ESPECIES"],
            "Producto": ["Set de placa de compresión", "Tamarindo"],
            "Marca": ["Asco Medical", "DM"],
            "Unidad de Medida": ["Unidad - 1 Unidad", "Bolsa - 1 Libra(lb)"],
            "Precio unitario": [2950.0, 14.0],
            "Cantidad Ofertada": [10.0, 60.0],
            "Monto ofertado": [29500.0, 840.0],
            "Comprador": ["INSTITUTO GUATEMALTECO DE SEGURIDAD SOCIAL", "MINISTERIO DE LA DEFENSA NACIONAL"]
        }
        st.dataframe(pd.DataFrame(sample_data))
    
    # Pie de página con logo (copiado exactamente del ejemplo)
    col1, col2 = st.columns([9, 1])
    with col2:
        st.markdown("")
        st.image(resource_path(current_folder / source_folder / "DPIR_logo_2.png"), width=120)

# Llamada a la función principal (copiada exactamente del ejemplo)
show()