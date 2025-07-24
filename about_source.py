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
    st.set_page_config(page_title="Estructura de la base de datos", layout="wide")
    
    # Logo del INE
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://www.ine.gob.gt/ine/wp-content/uploads/2017/09/cropped-INE.png" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Logo de la aplicación
    current_folder = Path(os.getcwd())
    source_folder = "source_data"
    st.sidebar.image(resource_path(current_folder/source_folder/"DPIR_logo_2.png"))
    
    # Título principal
    st.title("📊 Estructura de la base de datos de precios")
    st.markdown("""
    Documentación técnica del archivo CSV que contiene los precios históricos de insumos adjudicados en Guatecompras.
    """)
    
    # Descripción detallada del archivo CSV
    st.markdown("""
    ## Archivo principal: precios_insumos.csv
    
    **Descripción**: 
    Este dataset contiene todos los registros históricos de precios de insumos adjudicados a través del sistema Guatecompras,
    con información detallada sobre productos, ubicación, fechas y montos.
    """)
    
    # Estructura de columnas
    st.markdown("""
    ### Estructura de columnas:
    
    | Nombre Columna | Tipo de Dato | Descripción | Ejemplo |
    |---------------|--------------|-------------|---------|
    | CODIGO_CATEGORIA | Texto | Código de categoría del insumo | 210101 |
    | DESCRIPCION_CATEGORIA | Texto | Descripción de la categoría | "MEDICAMENTOS" |
    | CODIGO_INSUMO | Texto | Código único del insumo | 2101010010 |
    | DESCRIPCION_INSUMO | Texto | Nombre descriptivo completo del insumo | "PARACETAMOL TABLETA 500MG" |
    | VARIEDAD | Texto | Subtipo o variante del insumo | "CAJA X 100 TABLETAS" |
    | UNIDAD_MEDIDA | Texto | Unidad de medida del insumo | "CAJA", "KG", "LT" |
    | PRECIO_UNITARIO | Decimal | Precio por unidad de medida | 125.50 |
    | CANTIDAD_ADJUDICADA | Decimal | Cantidad total adjudicada | 50.0 |
    | MONTO_TOTAL | Decimal | Precio total (PRECIO_UNITARIO * CANTIDAD_ADJUDICADA) | 6275.00 |
    | FECHA_ADJUDICACION | Fecha | Fecha de adjudicación (YYYY-MM-DD) | 2022-05-15 |
    | DEPARTAMENTO | Texto | Departamento donde se adjudicó | "GUATEMALA" |
    | MUNICIPIO | Texto | Municipio donde se adjudicó | "MIXCO" |
    | PROVEEDOR | Texto | Nombre del proveedor adjudicado | "FARMACIA XYZ, S.A." |
    | CODIGO_PROCESO | Texto | Código único del proceso de compra | "CG-001-2022" |
    | ANIO | Entero | Año de adjudicación (derivado de fecha) | 2022 |
    | MES | Entero | Mes de adjudicación (derivado de fecha) | 5 |
    """)
    
    # Sección de usos y aplicaciones
    st.markdown("""
    ## Aplicaciones principales
    
    ### 1. Análisis de distribución de precios
    - Histogramas de frecuencia de precios por insumo
    - Comparación de rangos de precios entre variedades
    - Identificación de valores atípicos
    
    ### 2. Evolución temporal
    - Tendencia de precios mensuales/anuales
    - Comparación pre-pandemia/post-pandemia
    - Efectos de inflación en precios
    
    ### 3. Análisis geográfico
    - Mapeo de precios por departamento/municipio
    - Identificación de zonas con mayores precios
    - Diferencias urbano/rural
    
    ### 4. Benchmarking de proveedores
    - Comparación de precios entre proveedores
    - Análisis de concentración de mercado
    - Detección de posibles colusiones
    """)
    
    # Sección técnica
    with st.expander("📝 Notas técnicas y metadatos"):
        st.markdown("""
        - **Codificación de archivo**: UTF-8
        - **Separador de campos**: Coma (,)
        - **Formato fechas**: ISO 8601 (YYYY-MM-DD)
        - **Precisión decimal**: 2 dígitos para montos
        - **Periodicidad de actualización**: Mensual
        - **Cobertura temporal**: Desde enero 2015 hasta actualidad
        - **Cobertura geográfica**: Todos los departamentos de Guatemala
        """)
    
    # Ejemplo de datos
    with st.expander("🖥️ Ejemplo de registros (estructura)"):
        sample_data = {
            "CODIGO_CATEGORIA": ["210101", "210101"],
            "DESCRIPCION_CATEGORIA": ["MEDICAMENTOS", "MEDICAMENTOS"],
            "CODIGO_INSUMO": ["2101010010", "2101010015"],
            "DESCRIPCION_INSUMO": ["PARACETAMOL TAB 500MG", "IBUPROFENO TAB 400MG"],
            "VARIEDAD": ["CAJA X 100 TAB", "CAJA X 50 TAB"],
            "UNIDAD_MEDIDA": ["CAJA", "CAJA"],
            "PRECIO_UNITARIO": [125.50, 89.75],
            "CANTIDAD_ADJUDICADA": [50, 30],
            "MONTO_TOTAL": [6275.00, 2692.50],
            "FECHA_ADJUDICACION": ["2022-05-15", "2022-06-20"],
            "DEPARTAMENTO": ["GUATEMALA", "QUETZALTENANGO"],
            "MUNICIPIO": ["MIXCO", "QUETZALTENANGO"],
            "PROVEEDOR": ["FARMACIA XYZ, S.A.", "DISTRIBUIDORA ABC"],
            "CODIGO_PROCESO": ["CG-001-2022", "CG-045-2022"],
            "ANIO": [2022, 2022],
            "MES": [5, 6]
        }
        st.dataframe(pd.DataFrame(sample_data))
    
    # Pie de página con logo
    col1, col2 = st.columns([9, 1])
    with col2:
        st.markdown("")
        st.image(resource_path(current_folder / source_folder / "DPIR_logo_2.png"), width=120)

# Llamada a la función principal
show()