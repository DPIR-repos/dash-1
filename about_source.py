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
    
    # Logo institucional
    st.sidebar.markdown(
        """
        <div style="text-align: center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/Flag_of_Guatemala.svg" width="150">
            <h3>Portal de Datos de Compras Públicas</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Logo de la aplicación
    current_folder = Path(os.getcwd())
    source_folder = "source_data"
    st.sidebar.image(resource_path(current_folder/source_folder/"guatecompras_logo.png"), use_column_width=True)
    
    # Título principal
    st.title("📊 Estructura de la base de datos de compras públicas")
    st.markdown("""
    Documentación técnica del archivo CSV que contiene los registros de compras públicas adjudicadas en Guatemala.
    """)
    
    # Descripción detallada del archivo CSV
    st.markdown("""
    ## Archivo principal: compras_publicas_2020.csv
    
    **Descripción**: 
    Este dataset contiene registros de compras públicas adjudicadas en enero de 2020, con información detallada sobre productos, 
    proveedores, compradores, montos y características técnicas de los insumos adquiridos.
    
    **Origen de los datos**: 
    Sistema de compras públicas de Guatemala (Guatecompras)
    
    **Cobertura temporal**: 
    Enero 2020
    """)
    
    # Estructura de columnas
    st.markdown("""
    ### Estructura de columnas:
    
    | Nombre Columna | Tipo de Dato | Descripción | Ejemplo |
    |---------------|--------------|-------------|---------|
    | Filename | Texto | Nombre del archivo origen | "10572740_2020_01_webpage_source.txt" |
    | NOG | Texto | Número de operación gubernamental | "10572740" |
    | Descripcion | Texto | Descripción detallada del producto | "Set de placa de compresión dinámica (DCP) de minifragmentos para fémur de niños" |
    | Grupo | Texto | Código de grupo | "2.0" |
    | Subgrupo | Texto | Código de subgrupo | "29.0" |
    | Renglon | Texto | Código de renglón | "295.0" |
    | Codigo Insumo | Texto | Código único del insumo | "65370.0" |
    | Insumo Match | Texto | Descripción normalizada del insumo | "Set de placa de compresión dinámica (dcp) de minifragmentos para fémur de niños" |
    | Score | Decimal | Puntaje de coincidencia en el matching | 0.9069 |
    | Producto | Texto | Nombre del producto adjudicado | "Set de placa de compresión dinámica (dcp) de minifragmentos para fémur de niños" |
    | Marca | Texto | Marca del producto | "Asco Medical" |
    | Unidad de Medida | Texto | Unidad de medida del producto | "Unidad - 1 Unidad" |
    | Cantidad Ofertada | Decimal | Cantidad adjudicada | 10.0 |
    | Precio unitario | Decimal | Precio por unidad | 2950.0 |
    | Monto ofertado | Decimal | Monto total (precio unitario * cantidad) | 29500.0 |
    | Caracteristicas | Texto | Especificaciones técnicas del producto | "Incluye: Placa de minifragmentos dcp con tornillos..." |
    | Modalidad | Texto | Modalidad de compra | "Compra Directa con Oferta Electrónica (Art. 43 LCE Inciso b)" |
    | Adjudicado | Texto | Indicador de adjudicación | "1" |
    | Dia Publicacion | Texto | Día de publicación | "23" |
    | Mes Publicacion | Texto | Mes de publicación | "1" |
    | Anio Publicacion | Texto | Año de publicación | "2020" |
    | Hora Publicacion | Texto | Hora de publicación | "12:36:57" |
    | Dia Adjudicacion | Texto | Día de adjudicación | "18" |
    | Mes Adjudicacion | Texto | Mes de adjudicación | "2" |
    | Anio Adjudicacion | Texto | Año de adjudicación | "2020" |
    | Hora Adjudicacion | Texto | Hora de adjudicación | "16:02:50" |
    | NIT Oferente | Texto | NIT del proveedor | "96598689" |
    | Oferente | Texto | Nombre del proveedor | "IMPLANTES ORTOPEDICOS Y SUMINISTROS MEDICOS PROMEDIC, SOCIEDAD ANONIMA" |
    | Direccion Oferente | Texto | Dirección del proveedor | "23 CALLE CONDADO EL NARANJO 14-50 404 Z.4 EDIFICIO CRECE II" |
    | Localidad Oferente | Texto | Localidad del proveedor | "MIXCO" |
    | Region Oferente | Texto | Región del proveedor | "GUATEMALA" |
    | Pais Oferente | Texto | País del proveedor | "GUATEMALA" |
    | NIT Comprador | Texto | NIT de la entidad compradora | "2342855" |
    | Comprador | Texto | Nombre de la entidad compradora | "INSTITUTO GUATEMALTECO DE SEGURIDAD SOCIAL -IGSS-" |
    | Direccion Comprador | Texto | Dirección de la entidad compradora | "7a. Avenida 22-72, Centro cívico, Zona 1" |
    | Localidad Comprador | Texto | Localidad de la entidad compradora | "GUATEMALA" |
    | Region Comprador | Texto | Región de la entidad compradora | "GUATEMALA" |
    | Pais Comprador | Texto | País de la entidad compradora | "GUATEMALA" |
    """)
    
    # Sección de categorías principales
    st.markdown("""
    ## Categorías principales de productos
    
    ### 1. Suministros médicos y equipos (Grupo 2, Subgrupo 29)
    - Implantes ortopédicos
    - Instrumentos quirúrgicos
    - Prótesis
    - Equipos médicos
    
    ### 2. Alimentos y víveres (Grupo 2, Subgrupo 21)
    - Carnes
    - Granos básicos
    - Frutas y verduras
    - Productos enlatados
    
    ### 3. Materiales de construcción (Grupo 2, Subgrupo 28)
    - Madera
    - Láminas
    - Clavos y herrajes
    
    ### 4. Tecnología y equipos (Grupo 3, Subgrupo 32)
    - Equipos de cómputo
    - Sistemas de seguridad
    - Equipos eléctricos
    """)
    
    # Sección de usos y aplicaciones
    st.markdown("""
    ## Aplicaciones principales
    
    ### 1. Análisis de precios de referencia
    - Establecer rangos de precios para productos similares
    - Identificar valores atípicos en adjudicaciones
    - Comparar precios entre proveedores
    
    ### 2. Evaluación de proveedores
    - Frecuencia de adjudicaciones por proveedor
    - Análisis de concentración de mercado
    - Evaluación de cumplimiento en entregas
    
    ### 3. Eficiencia en procesos de compra
    - Tiempos entre publicación y adjudicación
    - Comparación entre modalidades de compra
    - Análisis de competencia en licitaciones
    
    ### 4. Planeación estratégica
    - Patrones de compra por institución
    - Estacionalidad en adquisiciones
    - Optimización de presupuestos
    """)
    
    # Sección técnica
    with st.expander("📝 Notas técnicas y metadatos"):
        st.markdown("""
        - **Codificación de archivo**: UTF-8
        - **Separador de campos**: Coma (,)
        - **Formato fechas**: Campos separados (día, mes, año)
        - **Precisión decimal**: 4 dígitos para scores, 2 dígitos para montos
        - **Periodicidad de actualización**: Mensual
        - **Cobertura geográfica**: Todo el territorio nacional
        - **Instituciones principales**: IGSS, Ministerio de Defensa, Municipalidades
        """)
    
    # Ejemplo de datos
    with st.expander("🖥️ Ejemplo de registros (estructura)"):
        sample_data = {
            "Filename": ["10572740_2020_01_webpage_source.txt", "11600802_2020_01_webpage_source.txt"],
            "Descripcion": ["Set de placa de compresión dinámica para fémur", "ADQUISICIÓN DE ESPECIES PARA LA UNIDAD DE ADMINISTRACIÓN MARÍTIMA"],
            "Producto": ["Set de placa de compresión dinámica", "Tamarindo"],
            "Marca": ["Asco Medical", "DM"],
            "Unidad de Medida": ["Unidad - 1 Unidad", "Bolsa - 1 Libra(lb)"],
            "Cantidad Ofertada": [10.0, 60.0],
            "Precio unitario": [2950.0, 14.0],
            "Monto ofertado": [29500.0, 840.0],
            "Modalidad": ["Compra Directa con Oferta Electrónica", "Compra Directa con Oferta Electrónica"],
            "Comprador": ["INSTITUTO GUATEMALTECO DE SEGURIDAD SOCIAL", "MINISTERIO DE LA DEFENSA NACIONAL"],
            "Oferente": ["IMPLANTES ORTOPEDICOS Y SUMINISTROS MEDICOS", "PADILLA,CRUZ,,LESVIA,HAYDEE"]
        }
        st.dataframe(pd.DataFrame(sample_data))
    
    # Sección de análisis exploratorio
    with st.expander("🔍 Hallazgos iniciales"):
        st.markdown("""
        - **Rango de precios**: Desde Q1.78 (bolsas médicas) hasta Q410,000 (UPS modular)
        - **Proveedores frecuentes**: 
          - Asco Medical (suministros médicos)
          - Promedic (implantes ortopédicos)
          - Ortopedia de Guatemala
        - **Instituciones compradoras principales**:
          - IGSS (Instituto Guatemalteco de Seguridad Social)
          - Ministerio de Defensa Nacional
          - Municipalidades
        - **Modalidades más usadas**:
          - Compra Directa con Oferta Electrónica (Art. 43 LCE Inciso b)
          - Cotización (Art. 38 LCE)
          - Licitación Pública (Art. 17 LCE)
        """)
    
    # Pie de página
    st.markdown("---")
    col1, col2 = st.columns([9, 1])
    with col1:
        st.caption("© 2023 - Sistema de Análisis de Compras Públicas de Guatemala")
    with col2:
        st.image(resource_path(current_folder / source_folder / "guatecompras_logo.png"), width=80)

# Llamada a la función principal
show()