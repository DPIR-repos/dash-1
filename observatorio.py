# Dashboard
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import os
import numpy as np
import plotly.express as px
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from scipy.stats import gmean  # Para la media geométrica
from unidecode import unidecode
import geopandas as gpd
from shapely.geometry import Point
import re
import streamlit.components.v1 as components
from streamlit.components.v1 import html 
from sklearn.linear_model import LinearRegression
import scipy.stats as sps
import sys
from PIL import Image
from plotly.subplots import make_subplots
from datetime import datetime

INEBlueColors=[px.colors.qualitative.G10[0], px.colors.qualitative.D3[0], px.colors.qualitative.G10[0],
               px.colors.qualitative.T10[0], px.colors.qualitative.T10[0], px.colors.qualitative.Set3[4]  ]
INEGreenColors=[px.colors.qualitative.Vivid[5]]
INEOrangeColors=[px.colors.qualitative.D3[1]]


def fix_price_inflacion_mensual(dfInflacion, precio_inicial, anio_inicio, mes_inicio, anio_fin, mes_fin, Inflacion_Choice=None, region=None):
    """Ajusta el precio con respecto a la inflacion anual

    Args:
        dfInflacion: DataFrame con información de inflación (debe contener columnas 'Anio', 'Mes', 'IPC' y/o 'IPC_RX')
        precio_inicial: Precio inicial a ajustar
        anio_inicio: Año inicial
        mes_inicio: Mes inicial (1-12)
        anio_fin: Año final
        mes_fin: Mes final (1-12)
        Inflacion_Choice: Tipo de corrección ('Republica', 'Regional' o None)
        region: Número de región (solo necesario si Inflacion_Choice es 'Regional')

    Returns:
        float: Precio ajustado por inflación
    """
    meses = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }
    
    # Diccionario de regiones normalizado (sin espacios, minúsculas, sin acentos)
    regiones_gt = {
        'guatemala':1, 'altaverapaz':2, 'bajaverapaz':2, 
        'chiquimula':3, 'elprogreso':3, 'izabal':3, 'zacapa':3, 
        'jalapa':4, 'jutiapa':4, 'santarosa':4,
        'chimaltenango':5, 'escuintla':5, 'sacatepequez':5, 
        'retalhuleu':6, 'sanmarcos':6, 'solola':6, 'suchitepequez':6, 
        'totonicapan':6, 'quetzaltenango':6,
        'huehuetenango':7, 'quiche':7, 'peten':8
    }
    
    # Si no se especifica corrección por inflación, devolver el precio original
    if Inflacion_Choice is None:
        return precio_inicial
    
    # Normalizar el tipo de inflación
    tipo_inflacion = unidecode(Inflacion_Choice.lower()) if Inflacion_Choice else None
    
    if tipo_inflacion == 'republica':
        IPC_init = dfInflacion.loc[
            (dfInflacion['Anio'] == anio_inicio) & 
            (dfInflacion['Mes'] == meses[mes_inicio]),
            'IPC'
        ].iloc[0]

        IPC_fin = dfInflacion.loc[
            (dfInflacion['Anio'] == anio_fin) & 
            (dfInflacion['Mes'] == meses[mes_fin]),
            'IPC'
        ].iloc[0]

        precio_final = precio_inicial * (IPC_fin/IPC_init)
        
    elif tipo_inflacion == 'regional':
        if region is None:
            raise ValueError("Se requiere el parámetro 'region' para corrección regional")
            
        IPC_init = dfInflacion.loc[
            (dfInflacion['Anio'] == anio_inicio) & 
            (dfInflacion['Mes'] == meses[mes_inicio]),
            'IPC_R'+str(regiones_gt[unidecode(region.lower())])
        ].iloc[0]

        IPC_fin = dfInflacion.loc[
            (dfInflacion['Anio'] == anio_fin) & 
            (dfInflacion['Mes'] == meses[mes_fin]),
            'IPC_R'+str(regiones_gt[unidecode(region.lower())])
        ].iloc[0]
        
        precio_final = precio_inicial * (IPC_fin/IPC_init)
        
    else:
        # Si el tipo de inflación no es reconocido, devolver el precio original
        precio_final = precio_inicial

    return precio_final


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

def top10_insumos_adjudicaciones_unidades(dfY):
    # Sumar adjudicaciones (columna 'Adjudicado') por 'Codigo Insumo'
    top_adjudicaciones = (
        dfY.groupby('Codigo Insumo')['Adjudicado']
        .sum()
        .reset_index()
        .sort_values(by='Adjudicado', ascending=False)
        .head(10)
    )
    
    # Sumar unidades ofertadas por 'Codigo Insumo'
    top_unidades = (
        dfY.groupby('Codigo Insumo')['Cantidad Ofertada']
        .sum()
        .reset_index()
        .sort_values(by='Cantidad Ofertada', ascending=False)
        .head(10)
    )
    
    return top_adjudicaciones, top_unidades


#====================
#   PLOT FUNCIONTS
#====================


def abc_analysis(
    df,
    grupo_por="Unidad de Medida",
    inflacion=False,
    dfInflacion=None,
    anio_fin=None,
    mes_fin=None,
    Inflacion_Choice=None
):
    """
    Realiza un análisis ABC (Pareto) sobre datos de ofertas, generando visualizaciones y resultados detallados.
    
    El análisis ABC clasifica los elementos en tres categorías:
    - A: Elementos que representan aproximadamente el 80% del valor total (20% superior)
    - B: Elementos que representan el 15% siguiente del valor total
    - C: Elementos que representan el 5% restante del valor total

    Args:
        df (pd.DataFrame): DataFrame con los datos de ofertas que debe contener:
            - 'Cantidad Ofertada': Cantidad de unidades ofertadas
            - 'Precio unitario': Precio por unidad
            - Columna especificada en 'grupo_por': Categoría para agrupación
            
        grupo_por (str, optional): Nombre de la columna para agrupar los elementos. Default: "Unidad de Medida"
        
        inflacion (bool, optional): Si es True, aplica corrección por inflación. Default: False
        
        dfInflacion (pd.DataFrame, optional): DataFrame con datos de inflación. Requerido si inflacion=True.
        
        anio_fin (int, optional): Año de referencia para corrección. Requerido si inflacion=True.
        
        mes_fin (int, optional): Mes de referencia para corrección. Requerido si inflacion=True.
        
        Inflacion_Choice (str, optional): Tipo de corrección ('Nacional', 'Regional', etc.). Requerido si inflacion=True.

    Returns:
        tuple: Contiene cuatro elementos:
            - fig_bar (plotly.graph_objects.Figure): Gráfico de barras de la distribución ABC
            - fig_pie (plotly.graph_objects.Figure): Gráfico circular del valor por categoría ABC
            - df_abc (pd.DataFrame): DataFrame con la clasificación ABC detallada
            - resumen (pd.DataFrame): Resumen estadístico por categoría ABC

    Example:
        >>> # Análisis básico sin corrección por inflación
        >>> fig_bar, fig_pie, df_abc, resumen = abc_analysis(df_ofertas)
        >>>
        >>> # Análisis con corrección por inflación
        >>> fig_bar_corr, fig_pie_corr, df_abc_corr, resumen_corr = abc_analysis(
        ...     df_ofertas,
        ...     inflacion=True,
        ...     dfInflacion=df_inflacion,
        ...     anio_fin=2023,
        ...     mes_fin=12,
        ...     Inflacion_Choice='Nacional'
        ... )

    Notes:
        - Los colores utilizados son:
          * Azul (#1f77b4) para categoría A
          * Celeste (#17becf) para categoría B
          * Rojo (#d62728) para categoría C
        - Para casos con menos de 3 elementos, se aplica una clasificación especial:
          * 1 elemento: A
          * 2 elementos: A, B
          * 3 elementos: A, B, C
    """
    
    # =========================================================================
    # 1. Preparación inicial de datos
    # =========================================================================
    df = df.copy()  # Evitar modificar el DataFrame original
    
    # Conversión de tipos numéricos
    df["Cantidad Ofertada"] = pd.to_numeric(df["Cantidad Ofertada"], errors="coerce")
    df["Precio unitario"] = pd.to_numeric(df["Precio unitario"], errors="coerce")

    # Validación de columna de agrupación
    if grupo_por not in df.columns:
        raise ValueError(f"La columna '{grupo_por}' no existe en el DataFrame.")

    # Limpieza de datos faltantes
    df = df.dropna(subset=["Cantidad Ofertada", "Precio unitario", grupo_por])

    # =========================================================================
    # 2. Ajuste por inflación (si se especifica)
    # =========================================================================
    if all([inflacion, dfInflacion is not None, anio_fin is not None, 
            mes_fin is not None, Inflacion_Choice is not None]):
        df["Precio Corregido"] = df.apply(
            lambda row: fix_price_inflacion_mensual(
                dfInflacion,
                row["Precio unitario"],
                row["Anio Publicacion"],
                row["Mes Publicacion"],
                anio_fin,
                mes_fin,
                Inflacion_Choice,
                row["Region Oferente"] if unidecode(Inflacion_Choice.lower()) == 'regional' else None
            ) if pd.notna(row["Precio unitario"]) else np.nan,
            axis=1
        )
        df["Valor Total"] = df["Cantidad Ofertada"] * df["Precio Corregido"]
    else:
        df["Valor Total"] = df["Cantidad Ofertada"] * df["Precio unitario"]

    # =========================================================================
    # 3. Agrupación y cálculo de valores totales
    # =========================================================================
    df_abc = df.groupby(grupo_por, as_index=False).agg({"Valor Total": "sum"})
    df_abc = df_abc.sort_values(by="Valor Total", ascending=False).reset_index(drop=True)

    # Cálculo de porcentajes
    df_abc["Porcentaje"] = df_abc["Valor Total"] / df_abc["Valor Total"].sum()
    df_abc["Porcentaje Acumulado"] = df_abc["Porcentaje"].cumsum()

    # =========================================================================
    # 4. Clasificación ABC
    # =========================================================================
    def clasificar_pct(df_pct):
        """Clasifica elementos en categorías ABC basado en el porcentaje acumulado."""
        n = len(df_pct)
        clasificaciones = []
        
        # Casos especiales para pocos elementos
        if n == 1:
            clasificaciones = ["A"]
        elif n == 2:
            clasificaciones = ["A", "B"]
        elif n == 3:
            clasificaciones = ["A", "B", "C"]
        else:
            # Clasificación estándar (80/15/5)
            for pct_acum in df_pct["Porcentaje Acumulado"]:
                if pct_acum <= 0.8:
                    clasificaciones.append("A")
                elif pct_acum <= 0.95:
                    clasificaciones.append("B")
                else:
                    clasificaciones.append("C")
        return clasificaciones

    df_abc["Clasificación"] = clasificar_pct(df_abc)

    # =========================================================================
    # 5. Resumen estadístico por categoría
    # =========================================================================
    categorias = ['A', 'B', 'C']
    resumen = (
        df_abc.groupby('Clasificación')
        .agg(**{
            'Elementos': (grupo_por, 'count'),
            'Valor Total': ('Valor Total', 'sum')
        })
        .reindex(categorias)  # Asegurar todas las categorías
        .fillna(0)           # Rellenar con 0 si alguna categoría falta
        .reset_index()
    )

    # Cálculo de porcentajes en el resumen
    total_valor = resumen['Valor Total'].sum()
    total_elementos = resumen['Elementos'].sum()
    
    resumen['% del Valor'] = resumen['Valor Total'].apply(
        lambda x: (x / total_valor * 100) if total_valor > 0 else 0
    )
    resumen['% del Total'] = resumen['Elementos'].apply(
        lambda x: (x / total_elementos * 100) if total_elementos > 0 else 0
    )

    # =========================================================================
    # 6. Visualizaciones
    # =========================================================================
    # Paleta de colores para las categorías
    color_discrete_map = {
        "A": "#1f77b4",  # Azul
        "B": "#17becf",  # Celeste
        "C": "#d62728"   # Rojo
    }

    # 6.1 Gráfico de barras de la distribución ABC
    fig_bar = px.bar(
        df_abc,
        x=grupo_por,
        y="Valor Total",
        color="Clasificación",
        title=f"<b>Distribución ABC por {grupo_por}</b>",
        labels={"Valor Total": "Valor ofertado (Q)"},
        hover_data=["Porcentaje", "Porcentaje Acumulado"],
        color_discrete_map=color_discrete_map
    )
    
    # Personalización del gráfico de barras
    fig_bar.update_layout(
        xaxis_title=None,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )

    # 6.2 Gráfico circular del valor por categoría
    fig_pie = px.pie(
        resumen[resumen['Valor Total'] > 0],  # Mostrar solo categorías con valor
        values="Valor Total",
        names="Clasificación",
        title="<b>Distribución del Valor por Clasificación ABC</b>",
        hole=0.4,  # Donut chart
        color='Clasificación',
        color_discrete_map=color_discrete_map,
        category_orders={'Clasificación': categorias}
    )

    # Personalización del gráfico circular
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        showlegend=True,
        hovertemplate="<b>%{label}</b><br>Valor: %{value:,.2f}<br>Porcentaje: %{percent}"
    )

    fig_pie.update_layout(
        legend=dict(
            title_text='<b>Clasificación</b>',
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig_bar, fig_pie, df_abc, resumen

def plot_map_abc_dep(df_abc, dfGeoDATA):
    """
    Genera un mapa coroplético interactivo mostrando la clasificación ABC por departamento.
    
    Esta función visualiza la clasificación ABC (A, B, C) de departamentos en un mapa interactivo,
    permitiendo identificar patrones geográficos en la categorización. Los departamentos sin
    clasificación se muestran en color distintivo.

    Args:
        df_abc (pd.DataFrame): DataFrame con los datos de clasificación ABC que debe contener:
            - Columna 0: Nombre del departamento
            - Última columna: Clasificación ABC ('A', 'B' o 'C')
            
        dfGeoDATA (gpd.GeoDataFrame): GeoDataFrame con las geometrías de los departamentos que debe contener:
            - 'NAME_1': Nombre del departamento
            - 'geometry': Geometrías para el mapeo

    Returns:
        plotly.graph_objects.Figure: Mapa coroplético interactivo con la clasificación ABC por departamento

    Example:
        >>> # Cargar datos de clasificación y geometrías
        >>> df_clasificacion = pd.read_csv('abc_departamentos.csv')
        >>> geo_departamentos = gpd.read_file('departamentos.geojson')
        >>>
        >>> # Generar y mostrar el mapa
        >>> mapa = plot_map_abc_dep(df_clasificacion, geo_departamentos)
        >>> mapa.show()

    Notes:
        - Paleta de colores utilizada:
          * Azul (#1f77b4) para categoría A (alto valor/prioridad)
          * Celeste (#17becf) para categoría B (valor medio)
          * Rojo (#d62728) para categoría C (bajo valor)
          * Gris claro (#fbfbfb) para no clasificados
        - El mapa se ajusta automáticamente al área geográfica de los departamentos
        - Los tooltips muestran el nombre del departamento al pasar el cursor
    """
    
    # =========================================================================
    # 1. Preparación de datos: normalización de nombres para unión
    # =========================================================================
    # Crear copias para evitar modificar los DataFrames originales
    dfGeoDATA = dfGeoDATA.copy()
    df_abc = df_abc.copy()
    
    # Normalizar nombres de departamentos (sin espacios, minúsculas, sin acentos)
    dfGeoDATA['muni_key'] = dfGeoDATA['NAME_1'].str.replace(' ', '').str.lower().apply(unidecode)
    
    # Identificar columnas relevantes (primera: nombre departamento, última: clasificación)
    nombre_col_departamento = df_abc.columns[0]
    nombre_col_clasificacion = df_abc.columns[-1]
    
    # Normalizar nombres en los datos ABC para hacer match con las geometrías
    df_abc['muni_key'] = df_abc[nombre_col_departamento].str.replace(' ', '').str.lower().apply(unidecode)

    # =========================================================================
    # 2. Unión de datos geográficos con clasificación ABC
    # =========================================================================
    # Realizar merge manteniendo todos los departamentos (left join)
    gdf_merge = dfGeoDATA.merge(
        df_abc[['muni_key', nombre_col_clasificacion]], 
        on='muni_key', 
        how='left'
    )

    # Asignar "Sin Clasificar" a departamentos sin datos
    gdf_merge[nombre_col_clasificacion] = gdf_merge[nombre_col_clasificacion].fillna("Sin Clasificar")

    # =========================================================================
    # 3. Configuración de estilo visual
    # =========================================================================
    # Paleta de colores fija para las categorías ABC
    color_discrete_map = {
        "A": "#1f77b4",           # Azul fuerte para categoría A
        "B": "#17becf",           # Celeste para categoría B
        "C": "#d62728",           # Rojo para categoría C
        "Sin Clasificar": "#fbfbfb"  # Gris claro para no clasificados
    }

    # Orden definido para consistencia en la leyenda
    category_order = ["A", "B", "C", "Sin Clasificar"]

    # =========================================================================
    # 4. Creación del mapa coroplético
    # =========================================================================
    fig_map = px.choropleth(
        gdf_merge,
        geojson=gdf_merge.geometry,
        locations=gdf_merge.index,
        color=nombre_col_clasificacion,
        hover_name="NAME_1",  # Mostrar nombre original en tooltips
        category_orders={nombre_col_clasificacion: category_order},
        color_discrete_map=color_discrete_map,
        title="<b>Clasificación ABC por Departamento</b>",
        labels={nombre_col_clasificacion: "Clasificación ABC"}
    )

    # =========================================================================
    # 5. Configuración del mapa base
    # =========================================================================
    fig_map.update_geos(
        fitbounds="locations",  # Ajustar zoom a la extensión de los departamentos
        visible=False,         # Ocultar fondo del mapa base
        projection_scale=5,    # Nivel de zoom inicial
        #center={              # Centrar en el punto medio de los departamentos
        #    "lat": gdf_merge.geometry.centroid.y.mean(),
        #    "lon": gdf_merge.geometry.centroid.x.mean()
        #},
        showcountries=False,   # Ocultar fronteras de países
        showcoastlines=False,  # Ocultar líneas costeras
        showland=False         # Ocultar relleno de tierra
    )

    # =========================================================================
    # 6. Configuración del layout y estilo
    # =========================================================================
    fig_map.update_layout(
        title_x=0.0,  # Alinear título a la izquierda
        margin={"r": 10, "t": 60, "l": 10, "b": 10},  # Márgenes ajustados
        height=700,   # Altura fija del mapa
        width=900,    # Ancho fijo del mapa
        paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        plot_bgcolor='rgba(0,0,0,0)',   # Área de gráfico transparente
        legend_title_text="<b>Clasificación</b>",  # Título de leyenda en negrita
        legend=dict(
            title_font_size=12,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.7)'  # Fondo semitransparente para legibilidad
        )
    )

    return fig_map

def plot_map_abc_muni(df_abc, dfGeoDATA):
    """
    Genera un mapa coroplético interactivo mostrando la clasificación ABC por municipio.
    
    Esta función toma datos de clasificación ABC y geometrías de municipios para crear
    un mapa visual que muestra:
    - Municipios clasificados como A, B o C
    - Municipios sin clasificar
    - Incluye tooltips interactivos con el nombre del municipio
    
    Args:
        df_abc (pd.DataFrame): DataFrame con la clasificación ABC que debe contener:
            - Primera columna: Nombre del municipio
            - Última columna: Clasificación ABC (valores 'A', 'B' o 'C')
            
        dfGeoDATA (gpd.GeoDataFrame): GeoDataFrame con geometrías de municipios que debe contener:
            - 'NAME_2': Nombre del municipio
            - 'geometry': Geometrías para el mapeo
    
    Returns:
        plotly.graph_objects.Figure: Mapa coroplético interactivo con la clasificación ABC
    
    Example:
        >>> # Cargar datos de clasificación ABC y geometrías
        >>> df_abc = pd.read_csv('clasificacion_abc.csv')
        >>> df_geodata = gpd.read_file('municipios.geojson')
        >>>
        >>> # Generar y mostrar el mapa
        >>> fig = plot_map_abc_muni(df_abc, df_geodata)
        >>> fig.show()
    
    Notes:
        - Los municipios sin clasificación se muestran en gris claro
        - El mapa se centra automáticamente en los municipios disponibles
        - La paleta de colores usa:
          * Azul (#1f77b4) para categoría A
          * Celeste (#17becf) para categoría B
          * Rojo (#d62728) para categoría C
          * Gris claro (#fbfbfb) para no clasificados
    """
    
    # =========================================================================
    # 1. Preparación de datos: normalización de nombres para unión
    # =========================================================================
    # Crear copias para evitar modificar los DataFrames originales
    dfGeoDATA = dfGeoDATA.copy()
    df_abc = df_abc.copy()
    
    # Normalizar nombres de municipios (sin espacios, minúsculas, sin acentos)
    dfGeoDATA['muni_key'] = dfGeoDATA['NAME_2'].str.replace(' ', '').str.lower().apply(unidecode)
    
    # Identificar columnas relevantes en df_abc (primera columna: municipio, última: clasificación)
    nombre_col_departamento = df_abc.columns[0]
    nombre_col_clasificacion = df_abc.columns[-1]
    
    # Normalizar nombres en df_abc para hacer match con dfGeoDATA
    df_abc['muni_key'] = df_abc[nombre_col_departamento].str.replace(' ', '').str.lower().apply(unidecode)

    # =========================================================================
    # 2. Unión de datos geográficos con clasificación ABC
    # =========================================================================
    # Hacer merge manteniendo todos los municipios (left join)
    gdf_merge = dfGeoDATA.merge(
        df_abc[['muni_key', nombre_col_clasificacion]], 
        on='muni_key', 
        how='left'
    )

    # Rellenar municipios sin clasificación
    gdf_merge[nombre_col_clasificacion] = gdf_merge[nombre_col_clasificacion].fillna("Sin Clasificar")

    # =========================================================================
    # 3. Configuración de visualización
    # =========================================================================
    # Paleta de colores fija para las categorías
    color_discrete_map = {
        "A": "#1f77b4",           # Azul fuerte para categoría A
        "B": "#17becf",           # Celeste para categoría B
        "C": "#d62728",           # Rojo para categoría C
        "Sin Clasificar": "#fbfbfb"  # Gris claro para no clasificados
    }

    # Orden de categorías para consistencia en la leyenda
    category_order = ["A", "B", "C", "Sin Clasificar"]

    # =========================================================================
    # 4. Creación del mapa coroplético
    # =========================================================================
    fig_map = px.choropleth(
        gdf_merge,
        geojson=gdf_merge.geometry,
        locations=gdf_merge.index,
        color=nombre_col_clasificacion,
        hover_name="NAME_2",  # Mostrar nombre original en tooltips
        category_orders={nombre_col_clasificacion: category_order},
        color_discrete_map=color_discrete_map,
        title="<b>Clasificación ABC por Municipio</b>",
        labels={nombre_col_clasificacion: "Clasificación ABC"}
    )

    # =========================================================================
    # 5. Configuración del mapa base
    # =========================================================================
    fig_map.update_geos(
        fitbounds="locations",  # Ajustar zoom a los municipios disponibles
        visible=False,          # Ocultar fondo del mapa
        projection_scale=5,     # Nivel de zoom inicial
        #center={               # Centrar en el punto medio de los municipios
        #    "lat": gdf_merge.geometry.centroid.y.mean(),
        #    "lon": gdf_merge.geometry.centroid.x.mean()
        #},
        showcountries=False,    # Ocultar fronteras de países
        showcoastlines=False,   # Ocultar líneas costeras
        showland=False          # Ocultar relleno de tierra
    )

    # =========================================================================
    # 6. Configuración del layout
    # =========================================================================
    fig_map.update_layout(
        title_x=0.0,  # Alinear título a la izquierda
        margin={"r": 10, "t": 60, "l": 10, "b": 10},  # Márgenes ajustados
        height=700,   # Altura fija
        width=900,    # Ancho fijo
        paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        plot_bgcolor='rgba(0,0,0,0)',   # Área de gráfico transparente
        legend_title_text="<b>Clasificación</b>",  # Título de leyenda
        legend=dict(
            title_font_size=12,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.7)'  # Fondo semitransparente para leyenda
        )
    )

    return fig_map

def plot_map_departamentos(df_ventas, dfGeoDATA, Inflacion=False, dfInflacion=None, anio_fin=None, mes_fin=None, Inflacion_Choice=None):
    """
    Genera un mapa coroplético interactivo de precios por departamento y una tabla de resultados.
    
    Esta función procesa datos de ventas y geográficos para visualizar:
    - Precios promedio (geométricos) por departamento
    - Desviaciones estándar de precios
    - Opcionalmente, precios corregidos por inflación
    - Incluye un panel lateral con información detallada
    
    Args:
        df_ventas (pd.DataFrame): DataFrame con datos de ventas que debe contener:
            - 'Region Oferente': Nombre del departamento
            - 'Precio unitario': Precio original
            - 'Cantidad Ofertada': Cantidad de unidades
            - 'Anio Publicacion': Año de publicación
            - 'Mes Publicacion': Mes de publicación
            
        dfGeoDATA (gpd.GeoDataFrame): GeoDataFrame con geometrías de departamentos que debe contener:
            - 'NAME_1': Nombre del departamento
            - 'geometry': Geometrías para el mapeo
            
        Inflacion (bool, optional): Si es True, aplica corrección por inflación. Default: False
        
        dfInflacion (pd.DataFrame, optional): DataFrame con datos de inflación. Requerido si Inflacion=True.
            Debe contener columnas para años, meses, regiones y tasas de inflación.
            
        anio_fin (int, optional): Año de referencia para corrección. Requerido si Inflacion=True.
        
        mes_fin (int, optional): Mes de referencia para corrección. Requerido si Inflacion=True.
        
        Inflacion_Choice (str, optional): Método de corrección ('Mensual', 'Acumulada', etc.). 
            Requerido si Inflacion=True.
    
    Returns:
        tuple: Contiene dos elementos:
            - fig_map (plotly.graph_objects.Figure): Mapa coroplético interactivo
            - df_resultados (pd.DataFrame): Tabla con precios y desviaciones por departamento
    
    Example:
        >>> # Sin corrección por inflación
        >>> mapa, resultados = plot_map_departamentos(df_ventas, df_geodata)
        >>>
        >>> # Con corrección por inflación
        >>> mapa_corr, resultados_corr = plot_map_departamentos(
        ...     df_ventas, 
        ...     df_geodata,
        ...     Inflacion=True,
        ...     dfInflacion=df_inflacion,
        ...     anio_fin=2023,
        ...     mes_fin=12,
        ...     Inflacion_Choice='Mensual'
        ... )
    
    Notes:
        - Los precios promedios se calculan usando la media geométrica ponderada por cantidad
        - Las desviaciones estándar consideran la expansión por cantidad ofertada
        - El mapa muestra tooltips con información detallada al pasar el cursor
        - El panel lateral muestra comparación entre precios originales y corregidos (si aplica)
    """
    
    # =========================================================================
    # 1. Normalización de nombres de departamentos
    # =========================================================================
    # Crear claves normalizadas (sin espacios, minúsculas, sin acentos) para unir datos
    dfGeoDATA['muni_key'] = dfGeoDATA['NAME_1'].str.replace(' ', '').str.lower().apply(unidecode).copy()
    df_ventas['muni_key'] = df_ventas['Region Oferente'].str.replace(' ', '').str.lower().apply(unidecode).copy()

    # =========================================================================
    # 2. Corrección por inflación (si se especifica)
    # =========================================================================
    if all([Inflacion is True, dfInflacion is not None, anio_fin is not None, mes_fin is not None, Inflacion_Choice is not None]):
        # Aplicar corrección a cada registro usando la función fix_price_inflacion_mensual
        df_ventas['Precio_corregido'] = df_ventas.apply(
            lambda row: fix_price_inflacion_mensual(
                dfInflacion=dfInflacion,
                precio_inicial=row['Precio unitario'],
                anio_inicio=row['Anio Publicacion'],
                mes_inicio=row['Mes Publicacion'],
                anio_fin=anio_fin,
                mes_fin=mes_fin,
                Inflacion_Choice=Inflacion_Choice,
                region=row['Region Oferente']
            ),
            axis=1
        )
    else:
        # Si no se aplica corrección, usar precios originales
        df_ventas['Precio_corregido'] = df_ventas['Precio unitario'].copy()

    # =========================================================================
    # 3. Funciones auxiliares para cálculo de estadísticas expandidas
    # =========================================================================
    def expanded_geometric_mean(group, price_col):
        """Calcula la media geométrica expandiendo los datos según cantidad ofertada."""
        try:
            expanded_prices = group.loc[group.index.repeat(group['Cantidad Ofertada'])][price_col]
            return gmean(expanded_prices)
        except:
            return np.nan
    
    def expanded_stdv(group, price_col):
        """Calcula la desviación estándar expandiendo los datos según cantidad ofertada."""
        try:
            expanded_prices = group.loc[group.index.repeat(group['Cantidad Ofertada'])][price_col]
            return expanded_prices.std()
        except:
            return np.nan

    # =========================================================================
    # 4. Cálculo de estadísticas por departamento
    # =========================================================================
    # Precios originales
    ventas_muni = df_ventas.groupby('muni_key', group_keys=False).apply(lambda x: expanded_geometric_mean(x, 'Precio unitario')).reset_index()
    stdv_ventas_muni = df_ventas.groupby('muni_key', group_keys=False).apply(lambda x: expanded_stdv(x, 'Precio unitario')).reset_index()
    ventas_muni.columns = ['muni_key', 'Precio_geo_mean']
    stdv_ventas_muni.columns = ['muni_key', 'Stdv_Precio']

    # Precios corregidos (si aplica)
    ventas_muni_corr = df_ventas.groupby('muni_key', group_keys=False).apply(lambda x: expanded_geometric_mean(x, 'Precio_corregido')).reset_index()
    stdv_ventas_muni_corr = df_ventas.groupby('muni_key', group_keys=False).apply(lambda x: expanded_stdv(x, 'Precio_corregido')).reset_index()
    ventas_muni_corr.columns = ['muni_key', 'Precio_corregido_mean']
    stdv_ventas_muni_corr.columns = ['muni_key', 'Stdv_Precio_corr']

    # =========================================================================
    # 5. Combinación de datos geográficos y estadísticas
    # =========================================================================
    gdf_merge = dfGeoDATA.merge(ventas_muni, on='muni_key', how='left')
    gdf_merge = gdf_merge.merge(stdv_ventas_muni, on='muni_key', how='left')
    gdf_merge = gdf_merge.merge(ventas_muni_corr, on='muni_key', how='left')
    gdf_merge = gdf_merge.merge(stdv_ventas_muni_corr, on='muni_key', how='left')

    # Rellenar valores NaN con 0 para visualización
    gdf_merge['precio_promedio'] = gdf_merge['Precio_geo_mean'].fillna(0)
    gdf_merge['precio_corregido'] = gdf_merge['Precio_corregido_mean'].fillna(0)
    gdf_merge['desviacion_estandar'] = gdf_merge['Stdv_Precio'].fillna(0)
    gdf_merge['desviacion_estandar_corr'] = gdf_merge['Stdv_Precio_corr'].fillna(0)

    # =========================================================================
    # 6. Preparación de tabla de resultados
    # =========================================================================
    df_resultados = gdf_merge[['NAME_1', 'precio_promedio', 'desviacion_estandar', 'precio_corregido', 'desviacion_estandar_corr']].copy()
    df_resultados.columns = ['Departamento', 'Precio Promedio (Q)', 'Desv. Estándar', 'Precio Corregido (Q)', 'Desv. Estándar Corr.']
    df_resultados = df_resultados[df_resultados['Precio Promedio (Q)'] > 0].copy()  # Filtrar departamentos con datos
    df_resultados = df_resultados.sort_values('Precio Promedio (Q)', ascending=False).reset_index(drop=True)

    # =========================================================================
    # 7. Creación del mapa coroplético
    # =========================================================================
    fig_map = px.choropleth(
        gdf_merge,
        geojson=gdf_merge.geometry,
        locations=gdf_merge.index,
        color='precio_promedio',
        hover_name="NAME_1",
        hover_data={
            'precio_promedio': ':.2f',
            'desviacion_estandar': ':.2f',
            'precio_corregido': ':.2f',
            'desviacion_estandar_corr': ':.2f'
        },
        color_continuous_scale="Blues",
        labels={
            'precio_promedio': 'Precio Promedio (Q)',
            'precio_corregido': 'Precio Corregido (Q)',
            'desviacion_estandar_corr': 'Desv. Estándar Corregida'
        }
    )

    # Configuración del mapa
    fig_map.update_geos(
        fitbounds="locations",
        visible=False,
        projection_scale=5,
        #center={"lat": gdf_merge.geometry.centroid.y.mean(), 
        #        "lon": gdf_merge.geometry.centroid.x.mean()},
        bgcolor='rgba(0,0,0,0)'
    )

    # Título dinámico según parámetros
    title = "Precio Promedio por Departamento"
    if Inflacion_Choice:
        title += f" (Corregido a {mes_fin}/{anio_fin} - {Inflacion_Choice})"

    fig_map.update_layout(
        title=title,
        title_x=0.0,
        margin={"r": 10, "t": 60, "l": 10, "b": 10},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        width=900,
        coloraxis_colorbar={
            'title': 'Q',
            'len': 0.5,
            'yanchor': 'middle',
            'y': 0.5
        }
    )

    # =========================================================================
    # 8. Panel lateral con información detallada
    # =========================================================================
    df_final = df_resultados.sort_values('Precio Promedio (Q)', ascending=False)
    
    if Inflacion:
        # Formato con precios originales y corregidos
        table_text = "<br>".join(
            f"<b>{row['Departamento']}</b>:<br>"
            f"• <u>Original</u>: Q{row['Precio Promedio (Q)']:.2f} ± {row['Desv. Estándar']:.2f}<br>"
            f"• <u>Corregido</u>: Q{row['Precio Corregido (Q)']:.2f} ± {row['Desv. Estándar Corr.']:.2f}"
            for _, row in df_final.iterrows()
        )
    else:
        # Formato solo con precios originales
        table_text = "<br>".join(
            f"<b>{row['Departamento']}</b>:<br>"
            f"•  Q{row['Precio Promedio (Q)']:.2f} ± {row['Desv. Estándar']:.2f}"
            for _, row in df_final.iterrows()
        )

    fig_map.add_annotation(
        text=table_text,
        align="left",
        showarrow=False,
        xref="paper", yref="paper",
        x=1.05, y=1,
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=12),
        width=160  # Aumentamos el ancho para acomodar más información
    )

    return fig_map, df_resultados

def plot_map_unidades_departamentos(df_ventas, dfGeoDATA):
    # Normalizar claves de departamento
    dfGeoDATA['muni_key'] = dfGeoDATA['NAME_1'].str.replace(' ', '').str.lower().apply(unidecode)
    df_ventas['muni_key'] = df_ventas['Region Oferente'].str.replace(' ', '').str.lower().apply(unidecode)

    # Agrupación y suma de unidades vendidas por departamento
    unidades_muni = df_ventas.groupby('muni_key')['Cantidad Ofertada'].sum().reset_index()
    unidades_muni.columns = ['muni_key', 'Unidades_Vendidas']

    # Combinar con datos geográficos
    gdf_merge = dfGeoDATA.merge(unidades_muni, on='muni_key', how='left')
    
    # Rellenar valores NaN con 0
    gdf_merge['Unidades_Vendidas'] = gdf_merge['Unidades_Vendidas'].fillna(0)

    # Crear DataFrame para resultados
    df_resultados = gdf_merge[['NAME_1', 'Unidades_Vendidas']].copy()
    df_resultados.columns = ['Departamento', 'Unidades Vendidas']
    df_resultados = df_resultados[df_resultados['Unidades Vendidas'] > 0]
    df_resultados = df_resultados.sort_values('Unidades Vendidas', ascending=False).reset_index(drop=True)

    # Mapa coroplético
    fig_map = px.choropleth(
        gdf_merge,
        geojson=gdf_merge.geometry,
        locations=gdf_merge.index,
        color='Unidades_Vendidas',
        hover_name="NAME_1",
        hover_data={'Unidades_Vendidas': ':,.0f'},
        color_continuous_scale="Blues",
        labels={'Unidades_Vendidas': 'Unidades Vendidas'}
    )

    # Configuración del mapa
    fig_map.update_geos(
        fitbounds="locations",
        visible=False,
        projection_scale=5,
        center={"lat": gdf_merge.geometry.centroid.y.mean(), 
                "lon": gdf_merge.geometry.centroid.x.mean()},
        bgcolor='rgba(0,0,0,0)'
    )

    fig_map.update_layout(
        title="Unidades Vendidas por Departamento",
        title_x=0.0,
        margin={"r": 10, "t": 60, "l": 10, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700,
        width=900,
        coloraxis_colorbar={
            'title': 'Unidades',
            'len': 0.3,
            'yanchor': 'middle',
            'y': 0.5
        }
    )

    # Panel lateral con unidades vendidas
    table_text = "<br>".join(
        f"<b>{row['Departamento']}</b>:<br> • {row['Unidades Vendidas']:,.0f} unidades"
        for _, row in df_resultados.iterrows()
    )

    fig_map.add_annotation(
        text=table_text,
        align="left",
        showarrow=False,
        xref="paper", yref="paper",
        x=1.05, y=0.8,
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        font=dict(size=12),
        width=160
    )

    return fig_map, df_resultados

def plot_prices_monts(df_ventas_va, Inflacion=False, dfInflacion=None, anio_fin=None, mes_fin=None, Inflacion_Choice=None):
    # Hacer una copia explícita del DataFrame para evitar warnings
    df = df_ventas_va.copy()
    
    # Diccionario de meses
    meses_dic = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }

    meses_long = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 
                  5:'Mayo', 6:'Junio', 7:'Julio', 8:'Agosto', 
                  9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}

    meses_dic_inverso = {v: k for k, v in meses_dic.items()}

    # Verificar si hay múltiples años
    years = df['Anio Publicacion'].unique()
    multi_year = len(years) > 1
    
    # Crear columna combinada de Mes-Año si hay múltiples años
    if multi_year:
        df.loc[:, 'Mes_Año'] = df.apply(
            lambda x: f"{meses_dic[x['Mes Publicacion']]}-{x['Anio Publicacion']}", 
            axis=1
        )
        grupos = df.groupby(['Mes Publicacion', 'Anio Publicacion', 'Mes_Año'], group_keys=False)
    else:
        grupos = df.groupby('Mes Publicacion', group_keys=False)
    
    # Calcular estadísticas para cada grupo
    precios_promedio = []
    precios_promedio_ajustados = []
    stdv_precios_prom = []
    stdv_precios_ajustados = []
    etiquetas = []
    
    for nombre, grupo in grupos:
        if multi_year:
            mes_num, year, etiqueta = nombre
        else:
            mes_num = nombre
            etiqueta = f"{meses_dic[mes_num]}-{years[0]}"
            year = years[0]
        
        try:
            # Expandir precios según cantidad ofertada (método más eficiente)
            precios_expandidos = []
            cantidades = grupo['Cantidad Ofertada'].astype(int).values
            precios = grupo['Precio unitario'].values
            
            for precio, cantidad in zip(precios, cantidades):
                precios_expandidos.extend([precio] * cantidad)
            
            precios_expandidos = np.array(precios_expandidos)
            
            if Inflacion and dfInflacion is not None and anio_fin is not None and mes_fin is not None and Inflacion_Choice is not None:
                if unidecode(Inflacion_Choice.lower())=='republica':
                    # Aplicar corrección por inflación a cada precio expandido (vectorizado)
                    anios = grupo['Anio Publicacion'].values
                    meses = grupo['Mes Publicacion'].values
                    
                    precios_ajustados = []
                    for precio, anio, mes, cantidad in zip(precios, anios, meses, cantidades):
                        precio_ajustado = fix_price_inflacion_mensual(
                            dfInflacion,
                            precio,
                            anio,
                            mes,
                            anio_fin,
                            mes_fin, Inflacion_Choice
                        )
                        precios_ajustados.extend([precio_ajustado] * cantidad)
                    
                    precios_ajustados = np.array(precios_ajustados)
                    
                    # Calcular estadísticas de precios ajustados
                    precio_prom_ajustado = gmean(precios_ajustados)
                    std_dev_ajustado = np.std(precios_ajustados)
                    
                    precios_promedio_ajustados.append(float(precio_prom_ajustado))
                    stdv_precios_ajustados.append(float(std_dev_ajustado))
                # INFLACION REGIONAL    
                if unidecode(Inflacion_Choice.lower())=='regional':
                    # Aplicar corrección por inflación a cada precio expandido (vectorizado)
                    anios = grupo['Anio Publicacion'].values
                    meses = grupo['Mes Publicacion'].values
                    depts = grupo['Region Oferente'].values #en esta fila estan los nombres de los departamentos
                    
                    precios_ajustados = []
                    for precio, anio, mes, cantidad, depa in zip(precios, anios, meses, cantidades, depts):
                        precio_ajustado = fix_price_inflacion_mensual(
                            dfInflacion,
                            precio,
                            anio,
                            mes,
                            anio_fin,
                            mes_fin, Inflacion_Choice, depa
                        )
                        precios_ajustados.extend([precio_ajustado] * cantidad)
                    
                    precios_ajustados = np.array(precios_ajustados)
                    
                    # Calcular estadísticas de precios ajustados
                    precio_prom_ajustado = gmean(precios_ajustados)
                    std_dev_ajustado = np.std(precios_ajustados)
                    
                    precios_promedio_ajustados.append(float(precio_prom_ajustado))
                    stdv_precios_ajustados.append(float(std_dev_ajustado))
            
            # Calcular estadísticas de precios originales
            precio_prom = gmean(precios_expandidos)
            std_dev = np.std(precios_expandidos)
            
            precios_promedio.append(float(precio_prom))
            stdv_precios_prom.append(float(std_dev))
            etiquetas.append(etiqueta)
            
        except Exception as e:
            print(f"Error procesando grupo {nombre}: {str(e)}")
            continue
    
    # Ordenar por fecha cronológica
    if multi_year:
        if Inflacion and precios_promedio_ajustados:
            datos_ordenados = sorted(
                zip(etiquetas, precios_promedio, stdv_precios_prom, precios_promedio_ajustados, stdv_precios_ajustados),
                key=lambda x: (
                    int(x[0].split('-')[1]),
                    meses_dic_inverso[x[0].split('-')[0]]
                )
            )
            etiquetas = [x[0] for x in datos_ordenados]
            precios_promedio = [x[1] for x in datos_ordenados]
            stdv_precios_prom = [x[2] for x in datos_ordenados]
            precios_promedio_ajustados = [x[3] for x in datos_ordenados]
            stdv_precios_ajustados = [x[4] for x in datos_ordenados]
        else:
            datos_ordenados = sorted(
                zip(etiquetas, precios_promedio, stdv_precios_prom),
                key=lambda x: (
                    int(x[0].split('-')[1]),
                    meses_dic_inverso[x[0].split('-')[0]]
                )
            )
            etiquetas = [x[0] for x in datos_ordenados]
            precios_promedio = [x[1] for x in datos_ordenados]
            stdv_precios_prom = [x[2] for x in datos_ordenados]
    
    # Crear el gráfico
    fig = go.Figure()
    
    if Inflacion and dfInflacion is not None and anio_fin is not None and mes_fin is not None and precios_promedio_ajustados:
        fig.add_trace(go.Scatter(
            x=etiquetas,
            y=precios_promedio_ajustados,
            mode='lines+markers',
            name='Media geométrica (ajustada)',
            line=dict(color=INEOrangeColors[0], width=2),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=stdv_precios_ajustados,
                visible=True,
                color=INEOrangeColors[0],
                thickness=2,
                width=3
            )
        ))
        
        fig.add_trace(go.Scatter(
            x=etiquetas,
            y=precios_promedio,
            mode='lines+markers',
            name='Media geométrica (sin ajuste)',
            line=dict(color=INEBlueColors[0], width=2),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=stdv_precios_prom,
                visible=True,
                color=INEBlueColors[0],
                thickness=2,
                width=3
            )
        ))
        
        #fig.add_trace(go.Scatter(
        #    x=etiquetas,
        #    y=precios_promedio,
        #    mode='lines',
        #    name='Media geométrica (original)',
        #    line=dict(color=INEOrangeColors[0], width=3, dash='dot')
        #))
    else:
        fig.add_trace(go.Scatter(
            x=etiquetas,
            y=precios_promedio,
            mode='lines+markers',
            name='Media geométrica',
            line=dict(color=INEBlueColors[0], width=2),
            marker=dict(size=10),
            error_y=dict(
                type='data',
                array=stdv_precios_prom,
                visible=True,
                color='gray',
                thickness=3,
                width=3
            )
        ))
    
    # Personalizar el layout
    title_suffix = ''
    if mes_fin is not None and multi_year:
        title_suffix = f" ({meses_long[mes_fin]}-{anio_fin})"
    elif not multi_year:
        title_suffix = f" (Año {years[0]})"
    
    fig.update_layout(
        title=f'Evolución de Precios Mensuales{" con corrección por inflación" if Inflacion_Choice else "" } {Inflacion_Choice} {title_suffix}',
        xaxis_title='Período',
        yaxis_title='Precio Promedio [Q]',
        showlegend=True,
        template='plotly_white',
        xaxis={'type': 'category'}
    )
    
    return fig


def plot_precio_vs_unidades_inflacion(df, Inflacion=False, dfInflacion=None, anio_fin=None, mes_fin=None, Inflacion_Choice=None):
    """
    Genera un gráfico de dispersión de Precio Unitario vs Unidades Vendidas con ajuste lineal,
    opcionalmente ajustando precios por inflación y devuelve dos DataFrames con los datos y estadísticas.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos de productos
    Inflacion : bool, opcional
        Si True, ajusta los precios por inflación
    dfInflacion : pandas.DataFrame, opcional
        DataFrame con datos de inflación
    anio_fin : int, opcional
        Año de referencia para el ajuste
    mes_fin : int, opcional
        Mes de referencia para el ajuste
    Inflacion_Choice : str, opcional
        Tipo de ajuste de inflación
        
    Retorna:
    --------
    tuple (fig, df_datos, df_ajuste) or None
        fig : plotly.graph_objects.Figure or None
        df_datos : pandas.DataFrame or None
        df_ajuste : pandas.DataFrame or None
    """
    
    # Verificar datos de entrada
    if df is None or df.empty:
        st.error("Error: No se han proporcionado datos para analizar.")
        return None, None, None
    
    try:
        # Crear copia del DataFrame
        df_clean = df[['Precio unitario', 'Cantidad Ofertada', 'Anio Publicacion', 'Mes Publicacion']].copy()
        
        # Procesamiento de inflación si está activado
        if Inflacion and dfInflacion is not None and anio_fin is not None and mes_fin is not None and Inflacion_Choice is not None:
            if unidecode(Inflacion_Choice.lower()) == 'regional':
                df_clean = df.copy()
                df_clean['Region Oferente'] = df.get('Region Oferente', None)
                df_clean = df_clean.dropna(subset=['Region Oferente'])
            
            precios_ajustados = []
            for idx, row in df_clean.iterrows():
                try:
                    if unidecode(Inflacion_Choice.lower()) == 'regional':
                        precio_aj = fix_price_inflacion_mensual(
                            dfInflacion,
                            row['Precio unitario'],
                            row['Anio Publicacion'],
                            row['Mes Publicacion'],
                            anio_fin,
                            mes_fin,
                            Inflacion_Choice,
                            row['Region Oferente']
                        )
                    else:
                        precio_aj = fix_price_inflacion_mensual(
                            dfInflacion,
                            row['Precio unitario'],
                            row['Anio Publicacion'],
                            row['Mes Publicacion'],
                            anio_fin,
                            mes_fin,
                            Inflacion_Choice
                        )
                    precios_ajustados.append(precio_aj)
                except Exception as e:
                    st.warning(f"Advertencia: No se pudo ajustar precio para fila {idx} - {str(e)}")
                    precios_ajustados.append(np.nan)
            
            df_clean['Precio Ajustado'] = precios_ajustados
            df_clean = df_clean.dropna(subset=['Precio Ajustado', 'Cantidad Ofertada'])
            X = df_clean['Precio Ajustado'].values.reshape(-1, 1)
            x_label = 'Precio Unitario Ajustado'
            columna_precio = 'Precio Ajustado'
        else:
            df_clean = df_clean.dropna(subset=['Precio unitario', 'Cantidad Ofertada'])
            X = df_clean['Precio unitario'].values.reshape(-1, 1)
            x_label = 'Precio Unitario'
            columna_precio = 'Precio unitario'

        y = df_clean['Cantidad Ofertada'].values

        # Verificar suficientes datos
        if len(X) < 2:
            st.error("Error: No hay suficientes datos para realizar el análisis (mínimo 2 puntos requeridos).")
            return None, None, None

        # Ajustar modelo
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calcular estadísticas
        corr_coef = np.corrcoef(X.flatten(), y)[0, 1]
        n = len(X)
        p = 2  # número de parámetros
        
        # Error estándar residual
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        sigma = np.sqrt(RSS / (n - p))
        
        # Matriz de diseño
        X_design = np.column_stack([np.ones(n), X.flatten()])
        
        # Matriz de covarianza
        try:
            cov_matrix = np.linalg.inv(X_design.T @ X_design) * (sigma**2)
        except np.linalg.LinAlgError:
            st.error("Error: Problema con los datos - no se puede calcular la matriz de covarianza")
            return None, None, None
        
        # Errores estándar
        se_intercept = np.sqrt(cov_matrix[0, 0])
        se_slope = np.sqrt(cov_matrix[1, 1])
        
        # Valor t para intervalo de confianza
        # Por esta (valor t para 68% de confianza):
        t_val = sps.t.ppf(1 - (1 - 0.68)/2, df=n - p) # al 68% de confianza a 1sigma
        
        # Calcular franja de error
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_main = model.intercept_ + model.coef_[0] * x_vals
        y_upper = (model.intercept_ + t_val * se_intercept) + (model.coef_[0] + t_val * se_slope) * x_vals
        y_lower = (model.intercept_ - t_val * se_intercept) + (model.coef_[0] - t_val * se_slope) * x_vals

        # Crear figura
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Datos'))
        fig.add_trace(go.Scatter(x=x_vals, y=y_main, mode='lines', name='Tendencia'))
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_vals, x_vals[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Franja de error a 1σ'
        ))
        
        fig.update_layout(
            title=f"Relación Precio vs Unidades Vendidas (r={corr_coef:.2f})",
            xaxis_title=x_label,
            yaxis_title="Unidades Vendidas",
            template='plotly_white'
        )

        # Crear DataFrames de retorno
        df_datos = df_clean[[columna_precio, 'Cantidad Ofertada']].copy()
        df_datos.columns = ['Precio', 'Unidades_Vendidas']
        df_datos['Tipo'] = 'Original'
        
        df_modelo = pd.DataFrame({
            'Precio': x_vals,
            'Unidades_Vendidas': y_main,
            'Tipo': 'Modelo'
        })
        
        df_datos_completo = pd.concat([df_datos, df_modelo], ignore_index=True)

        df_ajuste = pd.DataFrame({
            'Parametro': ['Pendiente', 'Intercepto', 'Coef_Correlacion', 'R_cuadrado', 'N_Puntos', 
                         'Error_Pendiente', 'Error_Intercepto', 'Valor_t'],
            'Valor': [model.coef_[0], model.intercept_, corr_coef, model.score(X, y), n,
                     se_slope, se_intercept, t_val],
            'Descripcion': [
                'Pendiente de la recta de regresión',
                'Punto donde la recta corta el eje Y',
                'Coeficiente de correlación de Pearson',
                'Coeficiente de determinación (R²)',
                'Número de puntos utilizados en el ajuste',
                'Error estándar de la pendiente',
                'Error estándar del intercepto',
                'Valor t de Student para intervalo 68%'
            ]
        })

        return fig, df_datos_completo, df_ajuste

    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return None, None, None

def plot_rfm_norm(df, id_col, df_codigos_desc=None):
    try:
        pastel_colors = px.colors.qualitative.Pastel

        if df.empty:
            raise ValueError("El DataFrame de entrada está vacío")

        df = df.copy()
        df['Fecha'] = pd.to_datetime(dict(
            year=df['Anio Publicacion'],
            month=df['Mes Publicacion'],
            day=df['Dia Publicacion']
        ), errors='coerce')

        df = df.dropna(subset=['Fecha'])
        if df.empty:
            raise ValueError("No hay fechas válidas después de la limpieza")

        fecha_max = df['Fecha'].max()

        df_rfm = df.groupby(id_col).agg(
            Recencia=('Fecha', lambda x: max((fecha_max - x.max()).days, 0) if not x.empty else np.nan),
            Frecuencia=('Fecha', 'size'),
            Valor_Monetario=('Precio unitario', lambda x: (x * df.loc[x.index, 'Cantidad Ofertada']).sum() if not x.empty else np.nan)
        ).reset_index()

        df_rfm = df_rfm.dropna(subset=['Recencia', 'Frecuencia', 'Valor_Monetario'])
        if df_rfm.empty:
            raise ValueError("No hay datos válidos después del cálculo RFM")

        if id_col == 'Codigo Insumo' and df_codigos_desc is not None:
            descripcion_map = dict(zip(df_codigos_desc['Codigo Insumo'], df_codigos_desc['Insumo Match']))
            df_rfm['Descripcion'] = df_rfm['Codigo Insumo'].map(descripcion_map).fillna("Sin descripción")
            label_col = df_rfm['Codigo Insumo'] + " - " + df_rfm['Descripcion']
        else:
            label_col = df_rfm[id_col]

        df_rfm['Label'] = label_col

        # 🚨 CASO DE SOLO UN ELEMENTO
        if df_rfm.shape[0] == 1:
            st.warning("⚠️ Para realizar el análisis RFM se requieren al menos 2 elementos. Se muestra el único disponible con valores normalizados en 100.")
            df_rfm['Recencia_norm'] = 100.0
            df_rfm['Frecuencia_norm'] = 100.0
            df_rfm['Valor_norm'] = 100.0
            df_rfm['RFM_Score'] = 100.0

            fig = px.bar(
                df_rfm,
                x='RFM_Score',
                y='Label',
                orientation='h',
                title='Único elemento con RFM Score = 100',
                color_discrete_sequence=[INEBlueColors[0]],
                text='RFM_Score'
            ).update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside'
            ).update_layout(
                yaxis=dict(autorange='reversed', title=None),
                margin=dict(l=20)
            )

            return fig, fig, fig, fig, df_rfm, ["0.3", "0.3", "0.4"]

        # 🔁 CASO NORMAL CON MÚLTIPLES ELEMENTOS
        def safe_min_max_norm(col):
            min_val, max_val = col.min(), col.max()
            if np.isclose(min_val, max_val):
                return np.zeros(len(col))
            return (col - min_val) / (max_val - min_val)

        df_rfm['Recencia_norm'] = (1 - safe_min_max_norm(df_rfm['Recencia'])) * 100
        df_rfm['Frecuencia_norm'] = safe_min_max_norm(df_rfm['Frecuencia']) * 100
        df_rfm['Valor_norm'] = safe_min_max_norm(df_rfm['Valor_Monetario']) * 100

        # Buscar mejores pesos
        combinations = []
        for R in np.arange(0.1, 0.6, 0.1):
            for F in np.arange(0.1, 0.6, 0.1):
                for M in np.arange(0.1, 0.6, 0.1):
                    if np.isclose(R + F + M, 1.0, rtol=1e-5):
                        combinations.append((R, F, M))

        results = []
        for R, F, M in combinations:
            df_rfm['RFM_Score'] = (R * df_rfm['Recencia_norm'] +
                                   F * df_rfm['Frecuencia_norm'] +
                                   M * df_rfm['Valor_norm'])
            mean_score = df_rfm['RFM_Score'].mean()
            std_score = df_rfm['RFM_Score'].std()
            cv_score = std_score / mean_score if not np.isclose(mean_score, 0) else np.inf
            results.append((R, F, M, mean_score, std_score, cv_score))

        results_df = pd.DataFrame(results, columns=['R', 'F', 'M', 'Mean', 'Std', 'CV']).dropna(subset=['CV'])
        best_combination = results_df.loc[results_df['CV'].idxmin()]
        R_best, F_best, M_best = best_combination['R'], best_combination['F'], best_combination['M']

        total_weight = R_best + F_best + M_best
        df_rfm['RFM_Score'] = (R_best * df_rfm['Recencia_norm'] +
                               F_best * df_rfm['Frecuencia_norm'] +
                               M_best * df_rfm['Valor_norm']) / total_weight

        def safe_nlargest(df, col, n=10):
            return df.nlargest(n, col) if not df.empty else pd.DataFrame()

        top_recencia = safe_nlargest(df_rfm, 'Recencia_norm').sort_values('Recencia_norm')
        top_frecuencia = safe_nlargest(df_rfm, 'Frecuencia_norm').sort_values('Frecuencia_norm')
        top_valor = safe_nlargest(df_rfm, 'Valor_norm').sort_values('Valor_norm')
        top_rfm = safe_nlargest(df_rfm, 'RFM_Score').sort_values('RFM_Score')

        def safe_plot_rfm_bar(df_top, metric, title, color):
            if df_top.empty:
                return px.bar(title=title + " (No hay datos disponibles)")
            return px.bar(
                df_top,
                x=metric,
                y='Label',
                orientation='h',
                title=title,
                color_discrete_sequence=[color],
                text=metric
            ).update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside'
            ).update_layout(
                yaxis=dict(autorange='reversed', title=None),
                margin=dict(l=20)
            )

        fig_rec = safe_plot_rfm_bar(top_recencia, 'Recencia_norm', 
                                    'Top 10 por Recencia (más reciente)', pastel_colors[0])
        fig_freq = safe_plot_rfm_bar(top_frecuencia, 'Frecuencia_norm', 
                                     'Top 10 por Frecuencia', pastel_colors[1])
        fig_val = safe_plot_rfm_bar(top_valor, 'Valor_norm', 
                                    'Top 10 por Valor Monetario', pastel_colors[2])
        fig_best = safe_plot_rfm_bar(top_rfm, 'RFM_Score',
                                     f'Top 10 por Puntaje RFM (Pesos: R={R_best:.1f}, F={F_best:.1f}, M={M_best:.1f})',
                                     INEBlueColors[0])

        return fig_rec, fig_freq, fig_val, fig_best, top_rfm, [f"{R_best:.1f}", f"{F_best:.1f}", f"{M_best:.1f}"]

    except Exception as e:
        st.error(f"Error en el análisis RFM: {str(e)}")
        empty_fig = px.bar(title="Error en la generación del gráfico")
        return empty_fig, empty_fig, empty_fig, empty_fig, pd.DataFrame(), ["1/3", "1/3", "1/3"]


def plot_unidades_monts(df_ventas_va):
    # Hacer una copia explícita del DataFrame para evitar warnings
    df = df_ventas_va.copy()
    
    # Diccionario de meses
    meses_dic = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }

    meses_long = {1:'Enero', 2:'Febrero', 3:'Marzo', 4:'Abril', 
                  5:'Mayo', 6:'Junio', 7:'Julio', 8:'Agosto', 
                  9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}

    meses_dic_inverso = {v: k for k, v in meses_dic.items()}

    # Verificar si hay múltiples años
    years = df['Anio Publicacion'].unique()
    multi_year = len(years) > 1
    
    # Crear columna combinada de Mes-Año si hay múltiples años
    if multi_year:
        df.loc[:, 'Mes_Año'] = df.apply(
            lambda x: f"{meses_dic[x['Mes Publicacion']]}-{x['Anio Publicacion']}", 
            axis=1
        )
        grupos = df.groupby(['Mes Publicacion', 'Anio Publicacion', 'Mes_Año'])
    else:
        grupos = df.groupby('Mes Publicacion')
    
    # Calcular unidades vendidas por mes
    unidades_vendidas = []
    etiquetas = []
    
    for nombre, grupo in grupos:
        if multi_year:
            mes_num, year, etiqueta = nombre
        else:
            mes_num = nombre
            etiqueta = f"{meses_dic[mes_num]}-{years[0]}"
            year = years[0]
        
        try:
            # Sumar todas las unidades vendidas en el mes
            total_unidades = grupo['Cantidad Ofertada'].sum()
            
            unidades_vendidas.append(int(total_unidades))
            etiquetas.append(etiqueta)
            
        except Exception as e:
            print(f"Error procesando grupo {nombre}: {str(e)}")
            continue
    
    # Ordenar por fecha cronológica
    if multi_year:
        datos_ordenados = sorted(
            zip(etiquetas, unidades_vendidas),
            key=lambda x: (
                int(x[0].split('-')[1]),
                meses_dic_inverso[x[0].split('-')[0]]
            )
        )
        etiquetas = [x[0] for x in datos_ordenados]
        unidades_vendidas = [x[1] for x in datos_ordenados]
    
    # Crear el gráfico
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=etiquetas,
        y=unidades_vendidas,
        name='Unidades vendidas',
        marker_color=INEBlueColors[0],
        text=unidades_vendidas,
        textposition='auto'
    ))
    
    # Personalizar el layout
    title_suffix = ''
    if not multi_year:
        title_suffix = f" (Año {years[0]})"
    
    fig.update_layout(
        title=f'Unidades Ofertadas por Mes{title_suffix}',
        xaxis_title='Período',
        yaxis_title='Unidades Vendidas',
        showlegend=False,
        template='plotly_white',
        xaxis={'type': 'category'}
    )
    
    return fig


def plot_precio_ponderado_mensual(df_ventas_va):
    # Copia para evitar SettingWithCopyWarning
    df = df_ventas_va.copy()

    # Diccionario de meses
    meses_dic = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }
    meses_dic_inverso = {v: k for k, v in meses_dic.items()}

    # Verificar si hay múltiples años
    years = df['Anio Publicacion'].unique()
    multi_year = len(years) > 1

    # Crear columna combinada de Mes-Año
    df['Mes_Año'] = df.apply(
        lambda x: f"{meses_dic[x['Mes Publicacion']]}-{x['Anio Publicacion']}", axis=1
    )

    # Agrupar por mes y año
    grupos = df.groupby(['Mes Publicacion', 'Anio Publicacion', 'Mes_Año'])

    etiquetas = []
    precios_ponderados = []

    for nombre, grupo in grupos:
        mes_num, anio, etiqueta = nombre
        try:
            total_valor = (grupo['Precio unitario'] * grupo['Cantidad Ofertada']).sum()
            total_cantidad = grupo['Cantidad Ofertada'].sum()

            if total_cantidad > 0:
                precio_ponderado = total_valor / total_cantidad
                etiquetas.append(etiqueta)
                precios_ponderados.append(precio_ponderado)
        except Exception as e:
            print(f"Error en grupo {nombre}: {str(e)}")
            continue

    # Ordenar por fecha cronológica
    datos_ordenados = sorted(
        zip(etiquetas, precios_ponderados),
        key=lambda x: (
            int(x[0].split('-')[1]),  # Año
            meses_dic_inverso[x[0].split('-')[0]]  # Mes
        )
    )
    etiquetas = [x[0] for x in datos_ordenados]
    precios_ponderados = [x[1] for x in datos_ordenados]

    # Crear gráfico Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=etiquetas,
        y=precios_ponderados,
        mode='lines+markers',
        name='Precio ponderado',
        line=dict(color='darkorange', width=2),
        marker=dict(size=6)
    ))

    title_suffix = ''
    if not multi_year:
        title_suffix = f" (Año {years[0]})"

    fig.update_layout(
        title=f'Precio Unitario Ponderado por Mes{title_suffix}',
        xaxis_title='Período',
        yaxis_title='Precio Ponderado',
        showlegend=False,
        template='plotly_white',
        xaxis={'type': 'category'}
    )

    return fig


def plot_Hvariety(dfVariety, variedad, CL=95, Inflacion=False, dfInflacion=None, anio_fin=None, mes_fin=None, Inflacion_Choice=None):
    """Genera histograma por codigo de insumo y varidad de un año especifico usando plotly"""
    # Verificar si hay suficientes datos
    dfVariety["Precio unitario"] = pd.to_numeric(dfVariety["Precio unitario"], errors="coerce")
    
    # Crear copia del dataframe para no modificar el original
    dfPlot = dfVariety.copy()
    
    # Aplicar corrección por inflación si se solicita
    if Inflacion and dfInflacion is not None and anio_fin is not None and mes_fin is not None and Inflacion_Choice is not None:
        # Aplicar corrección individual para cada registro según su fecha de publicación
        if unidecode(Inflacion_Choice.lower()) == 'republica':
            dfPlot["Precio unitario corregido"] = dfPlot.apply(
                lambda row: fix_price_inflacion_mensual(
                    dfInflacion, 
                    row["Precio unitario"], 
                    row["Anio Publicacion"], 
                    row["Mes Publicacion"], 
                    anio_fin, 
                    mes_fin, 
                    Inflacion_Choice
                ) if pd.notna(row["Precio unitario"]) else np.nan,
                axis=1
            )
        if unidecode(Inflacion_Choice.lower()) == 'regional':
            dfPlot["Precio unitario corregido"] = dfPlot.apply(
                lambda row: fix_price_inflacion_mensual(
                    dfInflacion, 
                    row["Precio unitario"], 
                    row["Anio Publicacion"], 
                    row["Mes Publicacion"], 
                    anio_fin, 
                    mes_fin, 
                    Inflacion_Choice,
                    row['Region Oferente']
                ) if pd.notna(row["Precio unitario"]) else np.nan,
                axis=1
            )
    
    # Verificar que haya precios no nulos
    data = dfPlot["Precio unitario"].dropna()    
    
    # Número de bins usando la regla de Sturges (ajustado para suma)
    Nbins = int(1 + np.log2(dfPlot['Cantidad Ofertada'].sum()))
    
    # Numero de precios diferentes
    ndif_prices=data.unique()
    
    # --- Cálculo de KDE para precios originales ---
    x_max = np.nan
    if len(ndif_prices) >= 3:
        kde = gaussian_kde(data)
        x_kde = np.linspace(data.min(), data.max(), 500)
        y_kde = kde(x_kde)
        sum_total = dfPlot["Cantidad Ofertada"].sum()
        range_x = data.max() - data.min()
        y_kde_scaled = y_kde * sum_total * (range_x / Nbins)
        idx_max = np.argmax(y_kde_scaled)
        x_max = x_kde[idx_max]
    
    # --- Cálculo de KDE para precios corregidos (si aplica) ---
    x_max_corregido = np.nan
    if Inflacion and dfInflacion is not None and anio_fin is not None and mes_fin is not None:
        data_corregido = dfPlot["Precio unitario corregido"].dropna()
        if len(data_corregido) >= 3:
            kde_corregido = gaussian_kde(data_corregido)
            x_kde_corregido = np.linspace(data_corregido.min(), data_corregido.max(), 500)
            y_kde_corregido = kde_corregido(x_kde_corregido)
            sum_total_corregido = dfPlot["Cantidad Ofertada"].sum()
            range_x_corregido = data_corregido.max() - data_corregido.min()
            y_kde_scaled_corregido = y_kde_corregido * sum_total_corregido * (range_x_corregido / Nbins)
            idx_max_corregido = np.argmax(y_kde_scaled_corregido)
            x_max_corregido = x_kde_corregido[idx_max_corregido]
    
    # --- Cálculo de estadísticas en formato de 3 columnas ---
    stats = []
    metricas = [
        "Media geométrica", 
        "Media aritmética", 
        "Desviación estándar",
        "Precio estimado KDE",
        f"Precio mínimo {CL:.0f}% a CL",
        f"Precio máximo {CL:.0f}% a CL"
    ]
    
    # Calcular valores originales
    for metrica in metricas:
        stat_row = {"Métrica": metrica}
        
        # Calcular valor original
        if "geométrica" in metrica:
            try:
                stat_row["Valor [Q]"] = gmean(data)
            except:
                stat_row["Valor [Q]"] = np.nan
        elif "aritmética" in metrica:
            stat_row["Valor [Q]"] = data.mean()
        elif "estándar" in metrica:
            stat_row["Valor [Q]"] = data.std()
        elif "KDE" in metrica:
            stat_row["Valor [Q]"] = x_max
        elif "mínimo" in metrica:
            stat_row["Valor [Q]"] = np.percentile(data, 50-(CL/2))
        elif "máximo" in metrica:
            stat_row["Valor [Q]"] = np.percentile(data, 50+(CL/2))
        
        stats.append(stat_row)
    
    # Si hay corrección por inflación, calcular valores ajustados
    if Inflacion and dfInflacion is not None and anio_fin is not None and mes_fin is not None:
        data_corregido = dfPlot["Precio unitario corregido"].dropna()
        
        for i, metrica in enumerate(metricas):
            if "geométrica" in metrica:
                try:
                    stats[i]["Valor+inflacion [Q]"] = gmean(data_corregido)
                except:
                    stats[i]["Valor+inflacion [Q]"] = np.nan
            elif "aritmética" in metrica:
                stats[i]["Valor+inflacion [Q]"] = data_corregido.mean()
            elif "estándar" in metrica:
                stats[i]["Valor+inflacion [Q]"] = data_corregido.std()
            elif "KDE" in metrica:
                stats[i]["Valor+inflacion [Q]"] = x_max_corregido
            elif "mínimo" in metrica:
                stats[i]["Valor+inflacion [Q]"] = np.percentile(data_corregido, 50-(CL/2))
            elif "máximo" in metrica:
                stats[i]["Valor+inflacion [Q]"] = np.percentile(data_corregido, 50+(CL/2))
    
    # Convertir a DataFrame
    df_stats = pd.DataFrame(stats)
    
    # --- Creación de gráficos ---
    # Calcular cuartiles para precios originales
    CLevel = CL
    QLow = 50-(CLevel/2)
    QHig = (CLevel/2)+50
    q_low = np.percentile(data, QLow)
    q_high = np.percentile(data, QHig)

    # Crear histograma original
    fig_original = px.histogram(
        dfPlot,
        x="Precio unitario",
        y="Cantidad Ofertada",
        color_discrete_sequence=[INEBlueColors[3]],
        histfunc='sum',
        nbins=Nbins,
        title=f'Histograma de precios (sin ajuste por inflación)'
    )

    if len(ndif_prices) >= 3:
        fig_original.add_trace(
            go.Scatter(
                x=x_kde,
                y=y_kde_scaled,
                mode='lines',
                line=dict(color='red', width=2),
                hovertemplate='Precio: %{x:.2f}<br>KDE: %{y:.2f}<extra></extra>',
                showlegend=True,
                name='KDE'
            )
        )

    # Resaltar cuartiles
    fig_original.add_vrect(
        x0=q_low, x1=q_high,
        fillcolor="lightgray", opacity=0.2,
        annotation_text=f"{CLevel}% de los datos", annotation_position="top left"
    )

    # Línea de media
    fig_original.add_vline(
        x=data.mean(),
        line_dash="dash",
        line_color=INEOrangeColors[0],
        annotation_text=f"Media: {data.mean():.2f}"
    )

    # Ajustes finales
    fig_original.update_layout(
        xaxis_title=f"Precio Unitario Ofertado [Q]",
        yaxis_title="Frecuencia",
        bargap=0.01,
        showlegend=True
    )
    
    # Si no hay inflación o faltan datos para el cálculo, retornar solo el gráfico original
    if not Inflacion or dfInflacion is None or anio_fin is None or mes_fin is None:
        return fig_original, df_stats
    
    # Crear histograma corregido por inflación
    data_corregido = dfPlot["Precio unitario corregido"].dropna()
    q_low_corregido = np.percentile(data_corregido, QLow)
    q_high_corregido = np.percentile(data_corregido, QHig)

    fig_corregido = px.histogram(
        dfPlot,
        x="Precio unitario corregido",
        y="Cantidad Ofertada",
        color_discrete_sequence=[INEOrangeColors[0]],
        histfunc='sum',
        nbins=Nbins,
        title=f'Histograma de precios (ajustado por inflación {Inflacion_Choice})'
    )

    if len(data_corregido) >= 3:
        fig_corregido.add_trace(
            go.Scatter(
                x=x_kde_corregido,
                y=y_kde_scaled_corregido,
                mode='lines',
                line=dict(color='red', width=2),
                hovertemplate='Precio: %{x:.2f}<br>KDE: %{y:.2f}<extra></extra>',
                showlegend=True,
                name='KDE'
            )
        )

    # Resaltar cuartiles
    fig_corregido.add_vrect(
        x0=q_low_corregido, x1=q_high_corregido,
        fillcolor="lightgray", opacity=0.2,
        annotation_text=f"{CLevel}% de los datos", annotation_position="top left"
    )

    # Línea de media
    fig_corregido.add_vline(
        x=data_corregido.mean(),
        line_dash="dash",
        line_color=INEBlueColors[3],
        annotation_text=f"Media: {data_corregido.mean():.2f}"
    )

    # Ajustes finales
    fig_corregido.update_layout(
        xaxis_title=f"Precio Unitario Ofertado [Q]",
        yaxis_title="Frecuencia",
        bargap=0.01,
        showlegend=True
    )
    
    return [fig_original, fig_corregido], df_stats

def obtener_orden_variedades(df_filtrado):
    """Devuelve las variedades ordenadas por cantidad ofertada (de mayor a menor)"""
    df_orden = df_filtrado.groupby('Unidad de Medida')['Cantidad Ofertada'].sum().reset_index()
    df_orden = df_orden.sort_values('Cantidad Ofertada', ascending=False)
    return df_orden['Unidad de Medida'].tolist()

####PieChart

def plot_variedades_pie(df_filtrado, orden_variedades):
    """
    Crea un gráfico de pastel con la distribución de unidades ofertadas por variedad.
    
    Args:
        df_filtrado (pd.DataFrame): DataFrame filtrado por código de insumo.
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de pastel interactivo.
    """
    # Agrupar por variedad y sumar las unidades ofertadas
    df_variedades = (
        df_filtrado.groupby('Unidad de Medida', as_index=False)
        .agg({'Cantidad Ofertada': 'sum'})
        .set_index('Unidad de Medida')
        .loc[orden_variedades]
        .reset_index()
    )
    
    # Ordenar de mayor a menor (opcional, para mejor visualización)
    #df_variedades = df_variedades.sort_values('Cantidad Ofertada', ascending=False)
    
    # Crear el pie chart
    fig = px.pie(
        df_variedades,
        names='Unidad de Medida',
        values='Cantidad Ofertada',
        category_orders={'Unidad de Medida': orden_variedades},  # Orden consistente
        title='<b>Distribución de unidades ofertadas por variedad</b>',
        color='Unidad de Medida',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.3,  # Agujero en el centro (opcional, quitar si no se desea)
        labels={'Unidad de Medida': 'Variedad', 'Cantidad Ofertada': 'Unidades'}
    )
    
    # Personalizar diseño
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=0),
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        showlegend=True,
        legend=dict(
            x=-0.2,              # Posición horizontal (negativo = izquierda)
            y=1,               # Posición vertical (>1 = arriba)
            xanchor='left',       # Anclaje al borde izquierdo
            yanchor='top',       # Anclaje al borde superior
            bgcolor='rgba(0,0,0,0)',  # Fondo semitransparente
            bordercolor='#CCC',
            borderwidth=1
        )
    )
    
    # Formatear tooltips y etiquetas
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Porcentaje: %{percent:.1%}<br>Unidades: %{value:,.0f}',
        textinfo='percent',  # Muestra porcentaje y etiqueta
        textposition='inside',     # Texto dentro de las porciones
        insidetextorientation='radial'  # Orientación del texto
    )
    
    return fig

def plot_adjudicaciones_por_variedad(df_filtrado,orden_variedades):
    """
    Crea un gráfico de barras horizontales con el número de adjudicaciones por variedad.
    
    Args:
        df_filtrado (pd.DataFrame): DataFrame filtrado por código de insumo.
        
    Returns:
        plotly.graph_objects.Figure: Gráfico de barras interactivo.
    """
    # Agrupar por variedad y sumar las adjudicaciones (1=adjudicado, 0=no adjudicado)
    df_adjudicaciones = (
        df_filtrado.groupby('Unidad de Medida', as_index=False)
        .agg({'Adjudicado': 'sum'})
        .set_index('Unidad de Medida')
        .loc[orden_variedades[::-1]]  # Invertir orden para barras horizontales
        .reset_index()
    )
    
    # Ordenar de mayor a menor adjudicaciones
    #df_adjudicaciones = df_adjudicaciones.sort_values('Adjudicado', ascending=True)  # Ascendente para barras horizontales
    
    # Crear el gráfico de barras horizontales
    fig = px.bar(
        df_adjudicaciones,
        y='Unidad de Medida',
        x='Adjudicado',
        orientation='h',  # Barras horizontales
        title='<b>Número de adjudicaciones por variedad</b>',
        labels={'Unidad de Medida': 'Variedad', 'Adjudicado': 'Adjudicaciones'},
        category_orders={'Unidad de Medida': orden_variedades},  # Orden consistente
        color='Unidad de Medida',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text_auto=True  # Muestra los valores en las barras
    )
    
    # Personalizar diseño
    fig.update_layout(
        yaxis_title=None,
        xaxis_title='Número de adjudicaciones',
        showlegend=False,  # Ocultar leyenda (las etiquetas ya están en el eje Y)
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=400  # Altura fija para mejor visualización
    )
    
    # Formatear tooltips y texto de las barras
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Adjudicaciones: %{x}',
        texttemplate='%{x}',  # Texto que se muestra en las barras
        textposition='outside'
    )
    
    return fig

def plot_NOGs_por_variedad(df_filtrado, orden_variedades):
    """
    Crea un gráfico de barras con el número de NOGs únicos por variedad.
    
    Esta función toma un DataFrame filtrado y genera una visualización del conteo de
    Números de Orden de Grano (NOGs) distintos para cada variedad, ordenados según
    la lista proporcionada.
    
    Args:
        df_filtrado (pd.DataFrame): DataFrame filtrado que contiene los datos de NOGs.
            Debe contener las columnas 'Unidad de Medida' y 'NOG'.
        orden_variedades (list): Lista con el orden específico en que deben mostrarse
            las variedades en el gráfico.
            
    Returns:
        plotly.graph_objects.Figure: Objeto de figura de Plotly con el gráfico de
        barras horizontales interactivo.
        
    Example:
        >>> orden = ['Variedad A', 'Variedad B', 'Variedad C']
        >>> fig = plot_NOGs_por_variedad(df_filtrado, orden)
        >>> fig.show()
    """
    # Contar NOGs únicos por variedad (usando nunique para contar valores distintos)
    df_nogs = (
        df_filtrado.groupby('Unidad de Medida', as_index=False)
        .agg({'NOG': 'nunique'})  # Contar NOGs distintos
        .set_index('Unidad de Medida')
        .loc[orden_variedades]  # Mantener el orden especificado
        .reset_index()
    )
    
    # Crear el gráfico de barras HORIZONTALES
    fig = px.bar(
        df_nogs,
        y='Unidad de Medida',  # Eje Y para variedades (barras horizontales)
        x='NOG',               # Eje X para conteo
        orientation='h',        # Barras horizontales
        title='<b>Número de NOGs distintos por variedad</b>',
        labels={'Unidad de Medida': 'Variedad', 'NOG': 'Número de NOGs'},
        category_orders={'Unidad de Medida': orden_variedades},
        color='Unidad de Medida',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text_auto=True
    )
    
    # Personalizar diseño del gráfico horizontal
    fig.update_layout(
        yaxis_title=None,        # Eliminar título del eje Y (las etiquetas son claras)
        xaxis_title='Número de NOGs distintos',
        showlegend=False,        # Ocultar leyenda ya que las barras están etiquetadas
        plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,              # Altura fija
        width=800                # Ancho fijo para mejor visualización
    )
    
    # Formatear tooltips y texto de las barras
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>NOGs distintos: %{x}',
        texttemplate='%{x}',
        textposition='outside'
    )
    
    # Ajustar márgenes y espaciado para etiquetas largas
    fig.update_yaxes(
        automargin=True,         # Ajustar automáticamente el margen para etiquetas
        title_standoff=15        # Espacio entre ejes y título
    )
    
    return fig

def plot_tiempo_adjudicacion(df_filtrado, orden_variedades=None):
    """
    Calcula el tiempo de adjudicación (días entre publicación y adjudicación) y genera un gráfico de líneas por variedad (Unidad de Medida).
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de compras.
        orden_variedades (list): Lista con el orden deseado de variedades (Unidad de Medida).
        
    Returns:
        tuple: (fig, df_tiempo), donde fig es el gráfico de Plotly y df_tiempo es el DataFrame con los datos calculados.
    """
    # Convertir fechas a datetime
    df=df_filtrado.copy()
    df['Fecha_Publicacion'] = pd.to_datetime(
        df['Anio Publicacion'].astype(str) + '-' + 
        df['Mes Publicacion'].astype(str) + '-' + 
        df['Dia Publicacion'].astype(str),
        errors='coerce'
    )
    df['Fecha_Adjudicacion'] = pd.to_datetime(
        df['Anio Adjudicacion'].astype(str) + '-' + 
        df['Mes Adjudicacion'].astype(str) + '-' + 
        df['Dia Adjudicacion'].astype(str),
        errors='coerce'
    )

    # Calcular días de adjudicación
    df['Dias_Adjudicacion'] = (df['Fecha_Adjudicacion'] - df['Fecha_Publicacion']).dt.days

    # Filtrar fechas válidas
    df = df[df['Dias_Adjudicacion'].notna() & (df['Dias_Adjudicacion'] >= 0)].copy()

    # Agrupar por mes y variedad
    df['Mes_Anio'] = df['Fecha_Publicacion'].dt.to_period('M').astype(str)
    df_tiempo = df.groupby(['Mes_Anio', 'Unidad de Medida'])['Dias_Adjudicacion'].mean().reset_index()

    # Crear gráfico
    fig = px.line(
        df_tiempo,
        x='Mes_Anio',
        y='Dias_Adjudicacion',
        color='Unidad de Medida',
        title='Tiempo Promedio de Adjudicación por Variedad',
        labels={
            'Mes_Anio': 'Mes y Año',
            'Dias_Adjudicacion': 'Días Promedio de Adjudicación',
            'Unidad de Medida': 'Variedad'
        },
        category_orders={'Unidad de Medida': orden_variedades} if orden_variedades else None,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_layout(
        xaxis_title='Mes y Año',
        yaxis_title='Días Promedio de Adjudicación',
        hovermode='x unified',
        xaxis=dict(tickangle=45),
        legend_title='Variedad',
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig, df_tiempo


#======================
#   BLOQUE ANALISIS ABC
#=======================


def render_abc_block(
    df_filtrado,
    titulo: str,
    state_prefix: str,
    grupo_principal: str,
    grupo_secundario: str = None,
    map_func_principal=None,
    map_func_secundario=None,
    df_geo_principal=None,
    df_geo_secundario=None,
    show_map=True
):
    if show_map:
        col1bot, col2bot, col3bot, col4bot = st.columns([0.85, 0.05, 0.05, 0.05])
    else:
        col1bot, col2bot, col3bot = st.columns([0.90, 0.05, 0.05])
        col4bot = None

    # Inicializar claves
    for key in ["plots", "table", "map"] if show_map else ["plots", "table"]:
        k = f"show_{state_prefix}_{key}"
        if k not in st.session_state:
            st.session_state[k] = False

    # Botones de acción
    with col2bot:
        if st.button("📊", key=f"btn_plot_{state_prefix}", help="Mostrar gráficos"):
            st.session_state[f"show_{state_prefix}_plots"] = True
            st.session_state[f"show_{state_prefix}_table"] = False
            if show_map:
                st.session_state[f"show_{state_prefix}_map"] = False

    with col3bot:
        if st.button("🖽", key=f"btn_table_{state_prefix}", help="Mostrar tabla de datos"):
            st.session_state[f"show_{state_prefix}_table"] = True
            st.session_state[f"show_{state_prefix}_plots"] = False
            if show_map:
                st.session_state[f"show_{state_prefix}_map"] = False

    if show_map:
        with col4bot:
            if st.button("🗺️", key=f"btn_map_{state_prefix}", help="Mostrar mapa"):
                st.session_state[f"show_{state_prefix}_map"] = True
                st.session_state[f"show_{state_prefix}_plots"] = False
                st.session_state[f"show_{state_prefix}_table"] = False

    # Resultados principales
    if st.session_state.get(f"show_{state_prefix}_plots", True):
        col1ABC, col2ABC = st.columns([0.5, 0.5])
        abc_results = abc_analysis(df_filtrado, grupo_por=grupo_principal)
        with col1ABC:
            st.plotly_chart(abc_results[0], use_container_width=True, key=f"{state_prefix}_bar_main")
        with col2ABC:
            st.plotly_chart(abc_results[1], use_container_width=True, key=f"{state_prefix}_pie_main")

        if grupo_secundario:
            abc_results2 = abc_analysis(df_filtrado, grupo_por=grupo_secundario)
            with col1ABC:
                st.plotly_chart(abc_results2[0], use_container_width=True, key=f"{state_prefix}_bar_sec")
            with col2ABC:
                st.plotly_chart(abc_results2[1], use_container_width=True, key=f"{state_prefix}_pie_sec")

    if st.session_state[f"show_{state_prefix}_table"]:
        abc_results = abc_analysis(df_filtrado, grupo_por=grupo_principal)
        st.dataframe(abc_results[2], hide_index=True, key=f"{state_prefix}_data_main")

        if grupo_secundario:
            abc_results2 = abc_analysis(df_filtrado, grupo_por=grupo_secundario)
            st.dataframe(abc_results2[2], hide_index=True, key=f"{state_prefix}_data_sec")

        st.markdown(
            f"""
            Este análisis sigue el principio de Pareto: aproximadamente el 20% de los productos
            representan el 80% del valor. Es útil para enfocar estrategias de monitoreo,
            control de precios y eficiencia en las compras públicas.  
            <a href="/documentacion#analisis-abc" target="_self" style="text-decoration: none; color: #1f77b4; font-size: 0.8em;">[Ver documentación]</a>
            """,
            unsafe_allow_html=True,
        )

    if show_map and st.session_state[f"show_{state_prefix}_map"]:
        abc_results = abc_analysis(df_filtrado, grupo_por=grupo_principal)
        col1ABC, col2ABC = st.columns([0.5, 0.5])
        if map_func_principal:
            map1 = map_func_principal(abc_results[2], df_geo_principal)
            with col1ABC:
                st.plotly_chart(map1, use_container_width=True, key=f"{state_prefix}_map_main")
        if grupo_secundario and map_func_secundario:
            abc_results2 = abc_analysis(df_filtrado, grupo_por=grupo_secundario)
            map2 = map_func_secundario(abc_results2[2], df_geo_secundario)
            with col2ABC:
                st.plotly_chart(map2, use_container_width=True, key=f"{state_prefix}_map_sec")



#alt.themes.enable("dark")
#=================================
#   FUNCIONES PARA CARGAR DATA
#=================================
@st.cache_data
@st.cache_resource

def load_data_year(year):
    """Carga y concatena todos los archivos CSV para un año específico"""
    # Estructura de directorios
    year_dir = Path("source_data") / "data_base_guatecompras" / f"{year}"
    
    # Obtener ruta válida
    dir_path = resource_path(year_dir)
    
    try:
        # Buscar todos los archivos CSV en el directorio del año
        csv_files = list(dir_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No se encontraron archivos CSV en {dir_path}")
        
        # Leer y concatenar todos los archivos
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(
                csv_file,
                sep=',',
                encoding='utf-8-sig',
                quoting=1,
                low_memory=False
            )
            dfs.append(df)
        
        # Concatenar todos los DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Filtrar precios unitarios cero y eliminar duplicados
        filtered_df = combined_df[combined_df['Precio unitario'] != 0].drop_duplicates()
        
        return filtered_df
    
    except Exception as e:
        available_files = "\n".join(str(p) for p in dir_path.glob("*")) if dir_path.exists() else "Directorio no encontrado"
        raise FileNotFoundError(
            f"No se pudo cargar datos para {year}.\n"
            f"Directorio buscado: {dir_path}\n"
            f"Archivos disponibles:\n{available_files}\n"
            f"Error original: {str(e)}"
        )

def load_GEOdata(value):
    """
    Carga datos geoespaciales para departamentos o municipios de Guatemala.
    
    Args:
        value (str): "departamento" o "municipio" para especificar qué datos cargar
    
    Returns:
        GeoDataFrame: Los datos geoespaciales solicitados
    
    Raises:
        ValueError: Si el valor no es "departamento" o "municipio"
        FileNotFoundError: Si no se encuentran los archivos GEOJSON
    """
    # Mapeo de tipos de datos a archivos
    geo_files = {
        "departamento": "gadm41_GTM_1.json",
        "municipio": "gadm41_GTM_2.json"
    }
    
    # Validar input
    if value not in geo_files:
        raise ValueError(f"Valor '{value}' no válido. Debe ser 'departamento' o 'municipio'")
    
    # Construir ruta relativa
    relative_path = Path("source_data") / geo_files[value]
    
    # Obtener ruta válida usando resource_path
    file_path = resource_path(relative_path)
    
    try:
        # Leer el archivo geoespacial
        gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        available_files = "\n".join(str(p) for p in file_path.parent.glob("*.json"))
        raise FileNotFoundError(
            f"No se pudo cargar datos geoespaciales para {value}.\n"
            f"Archivo buscado: {file_path}\n"
            f"Archivos disponibles:\n{available_files}\n"
            f"Error original: {str(e)}"
        )


def load_clasificador():
    """
    Carga el archivo CSV con el catálogo de insumos y sus descripciones.
    
    Returns:
        DataFrame: Contenido del catálogo de insumos
    
    Raises:
        FileNotFoundError: Si no se encuentra el archivo del catálogo
    """
    # Construir ruta relativa
    relative_path = Path("source_data") / "catalogo_insumos_full_description.csv"
    
    # Obtener ruta válida usando resource_path
    file_path = resource_path(relative_path)
    
    try:
        # Leer el archivo CSV con parámetros óptimos
        return pd.read_csv(
            file_path,
            encoding='utf-8-sig',
            low_memory=False
        )
    except Exception as e:
        available_files = "\n".join(str(p) for p in file_path.parent.glob("*.csv"))
        raise FileNotFoundError(
            f"No se pudo cargar el catálogo de insumos.\n"
            f"Archivo buscado: {file_path}\n"
            f"Archivos disponibles en el directorio:\n{available_files}\n"
            f"Error original: {str(e)}"
        )
    

def load_inflacion(inflacion_choice):
    """
    Carga los datos de inflación desde archivos Excel según la selección.
    
    Args:
        inflacion_choice (str): 'republica' o 'regional' para seleccionar el tipo de datos
    
    Returns:
        pd.DataFrame: DataFrame con los datos de inflación procesados
    
    Raises:
        ValueError: Si inflacion_choice no es válido
        FileNotFoundError: Si no se encuentra el archivo requerido
        Exception: Para otros errores durante la carga de datos
    """
    # Configuración de archivos según tipo
    config = {
        'republica': {
            'filename': "ipc_emprepu_es25.xlsx",
            'cols': [0, 1, 2],
            'names': ['Anio', 'Mes', 'IPC']
        },
        'regional': {
            'filename': "ipc_empreg_es25.xlsx",
            'cols': [0, 1, 2, 10, 18, 26, 34, 42, 50, 58],
            'names': ['Anio', 'Mes', 'IPC_R1', 'IPC_R2', 'IPC_R3', 
                     'IPC_R4', 'IPC_R5', 'IPC_R6', 'IPC_R7', 'IPC_R8']
        }
    }
    
    # Validar input
    choice = inflacion_choice.lower()
    if choice not in config:
        raise ValueError(f"Opción '{inflacion_choice}' no válida. Debe ser 'republica' o 'regional'")
    
    # Obtener configuración
    conf = config[choice]
    
    # Construir ruta y cargar archivo
    relative_path = Path("source_data") / conf['filename']
    file_path = resource_path(relative_path)
    
    try:
        # Cargar archivo Excel
        df = pd.read_excel(
            file_path,
            engine='openpyxl',
            header=0  # Asegurar que usa la primera fila como encabezados
        )
        
        # Seleccionar y renombrar columnas
        df_final = df.iloc[:, conf['cols']].copy()
        df_final.columns = conf['names']
        
        # Convertir columnas numéricas
        ipc_cols = [col for col in df_final.columns if col.startswith('IPC')]
        df_final[ipc_cols] = df_final[ipc_cols].apply(pd.to_numeric, errors='coerce')
        
        return df_final
    
    except FileNotFoundError:
        available_files = "\n".join(str(p) for p in file_path.parent.glob("*.xlsx"))
        raise FileNotFoundError(
            f"No se encontró el archivo de inflación {conf['filename']}.\n"
            f"Ruta buscada: {file_path}\n"
            f"Archivos disponibles:\n{available_files}"
        )
    except Exception as e:
        raise Exception(
            f"Error al procesar el archivo de inflación {conf['filename']}:\n"
            f"Error original: {str(e)}"
        )

#===============================
# Page configuration
#===============================

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Configuración para evitar el error del event loop

# Opcional: Deshabilitar el watchdog explícitamente
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"


###---SIDE-BAR
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
soruce_folder="source_data"
st.sidebar.markdown(" ")
image_path = resource_path(current_folder/ soruce_folder/ "DPIR_logo_2.png")
image=Image.open(image_path)
st.sidebar.image(image)

#st.sidebar.markdown("## DIRECCION DE PRECIOS E ÍNDICES DE REFERENCIA")

st.sidebar.header("Filtros")

years_disp = [2020, 2021, 2022, 2023, 2024]


# Mostrar multiselect
year = st.sidebar.multiselect(
    "Seleccione el año:",
    options=['Todos'] + years_disp,
    default=None,
    placeholder="Escriba o seleccione...",
    key="selected_years_widget"
)


st.markdown("## OBSERVATORIO DE PRECIOS ESTATALES")
st.markdown("---")

#if year:
#    st.markdown(f"Ha seleccionado el/los año(s) {year}")
#year = st.sidebar.selectbox(
#    "Seleccione el año:",
#    options=[2020, 2021, 2022, 2023, 2024],
#    index=3
#)

# Cargando la data


#Meese par ser tomados en la inflacion
meses={1:'Enero', 2:'Febrero', 3:'Marzo',4:'Abril', 5:'Mayo', 6:'Junio', 7:'Julio', 8:'Agosto', 9:'Septiembre', 10:'Octubre', 11:'Noviembre', 12:'Diciembre'}
meses_short={1:'Ene', 2:'Feb', 3:'Mar',4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
meses_short_inverse={'Ene':1, 'Feb':2, 'Mar':3, 'Abr':4, 'May':5, 'Jun':6, 'Jul':7, 'Ago':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dic':12}
#Meese par ser tomados en la inflacion
meses_dicReverse={'Enero':1, 'Febrero':2, 'Marzo':3, 'Abril':4, 'Mayo':5, 'Junio':6, 'Julio':7, 'Agosto':8, 'Septiembre':9, 'Octubre':10, 'Noviembre':11, 'Diciembre':12}

if len(year)>=1:
    dfTemp=[] #lista de dataframes por año
    if any(str(t).lower() == 'todos' for t in year):
        for k in years_disp:
            dfTemp.append(load_data_year(k))
    else:
        for k in year:
            dfTemp.append(load_data_year(k))
            
    dfT = pd.concat(dfTemp, axis=0).reset_index(drop=True)
    dfY=dfT.dropna()
    dfY["Codigo Insumo"] = dfY["Codigo Insumo"].astype(int)
    codigos_insumo =sorted( dfY["Codigo Insumo"].unique().tolist())
    dfG_dep=load_GEOdata('departamento')
    dfG_muni=load_GEOdata('municipio')
    #clasificador de insumos
    dfCI=load_clasificador()
    # --- Crear un mapeo código -> descripción ---
    # Fusionamos los códigos únicos de dfY con las descripciones de dfCI
    df_codigos_desc = pd.DataFrame({'Codigo Insumo': codigos_insumo}).merge(
        dfY[['Codigo Insumo', 'Insumo Match']].drop_duplicates(),
        left_on='Codigo Insumo',
        right_on='Codigo Insumo',
        how='left'
    )
    codigo_a_descripcion = dict(zip(df_codigos_desc['Codigo Insumo'], df_codigos_desc['Insumo Match']))

    # --- Selectbox con búsqueda combinada ---
    insumoCode = st.sidebar.selectbox(
        "🔍 Buscar por código o descripción:",
        options=codigos_insumo,
        format_func=lambda x: f"{x} - {codigo_a_descripcion.get(x, 'Sin descripción')}",
        index=None,
        placeholder="Escriba (código o nombre)..."
    )


    # Filtramos el DataFrame basado en los códigos seleccionados
    if insumoCode:
#======================================
#   Crear las pestañas
#====================================
        tab1, tab2, tab3, tab4 = st.tabs([
            "ℹ Inf. General", 
            "💰 Análisis Precios", 
            "🔄 Análisis RFM",
            "Análisis 🅰️🅱️©️"
        ])
                
        df_filtrado = dfY[dfY["Codigo Insumo"].isin([insumoCode])] #filtrado por codigo de insumo    
        current_group=dfCI[dfCI["Grupo"]==int(df_filtrado['Grupo'].iloc[0])]
        # Obtenemos las variedades solo para los códigos seleccionados
        #orden de las variedades
        orden_variedades = obtener_orden_variedades(df_filtrado)
        variedades_insumo = orden_variedades
        #dividir la infor en columnas
#==================================
#   Section 1: INFORMACION GENERAL
#===================================    
        with tab1:        
            st.markdown("##### Información general")
            col1Info, col2halfInfo, col2Info = st.columns([0.525, 0.05, 0.425])
            with col1Info:
                # Crear lista de información
                info_data = [
                f"Base de datos: {', '.join(map(str, years_disp)) if any(isinstance(y, str) and y.lower() == 'todos' for y in year) else (', '.join(map(str, year)) if isinstance(year, list) else year)}",    
                f"Grupo: {int(df_filtrado['Grupo'].iloc[0])} - {current_group['Nombre Grupo'].iloc[0]}",
                f"Subgrupo: {int(df_filtrado['Subgrupo'].iloc[0])} - {current_group['Nombre Subgrupo'].iloc[0]}",
                f"Renglón: {int(df_filtrado['Renglon'].iloc[0])} - {current_group['Concepto Renglon'].iloc[0]}",
                f"Código de insumo: {', '.join(map(str, insumoCode)) if isinstance(insumoCode, list) else insumoCode} - {df_filtrado['Insumo Match'].iloc[0]}",
                f"Caracteristicas: {df_filtrado['Caracteristicas'].iloc[0]} ",
                f"Variedades disponibles: {len(variedades_insumo)}"
                            ]
                # Convertir a DataFrame
                df_info = pd.DataFrame(info_data, columns=[""])
                # Mostrar sin índices ni encabezados
                st.dataframe(
                        df_info,
                        column_config={"__": st.column_config.Column(width="wide")},  # Ocupa todo el ancho
                        hide_index=True,
                        use_container_width=True
                    )
                col1InfoIn, col2InfoIn = st.columns([0.95, 0.05])
                with col1InfoIn:
                    # Llamar a la función
                    fig_adjudicaciones = plot_adjudicaciones_por_variedad(df_filtrado,orden_variedades)
                    # Mostrar el gráfico
                    st.plotly_chart(fig_adjudicaciones,  use_container_width=True)
                   
            
#====================================
#   SECTION 1.1: more information    
#====================================
            col1InfoIn2, col2InfoIn2 = st.columns([0.5, 0.5])
            with col1InfoIn2:
                fig, stats = plot_tiempo_adjudicacion(df_filtrado, orden_variedades)

                st.plotly_chart(fig, use_container_width=True)
                
            with col2InfoIn2:
                fig_NOGs=plot_NOGs_por_variedad(df_filtrado,orden_variedades)
                
                    # Mostrar el gráfico
                st.plotly_chart(fig_NOGs,  use_container_width=True)
            ####dataframe con la base de datos 
            with st.expander(f"**Base de datos para el código: {insumoCode}**"):
                change_columns={'Localidad Oferente':'Municipio Oferente','Region Oferente': 'Departamento Oferente', 
                        'Localidad Comprador':'Municipio Comprador','Region Comprador': 'Departamento Comprador'}
                df_filtrado=df_filtrado.raname(columns=change_columns)
                st.dataframe(df_filtrado,hide_index=True)    
                
            with col2Info:
                # Llamar a la función
                fig_pie = plot_variedades_pie(df_filtrado,orden_variedades)
                # Mostrar el gráfico
                st.plotly_chart(fig_pie, height=100)
            
            variedad_select = st.sidebar.multiselect(
                "🔍 Buscar o seleccionar la variedad:",
                options=['Todas']+variedades_insumo,
                default=None,
                placeholder="Escriba o seleccione..."
            )
        #correccion por inflacion
        if len(variedad_select)>=1:
            inflacion_choice=st.sidebar.selectbox(
                    "Correcion por inflacion:",
                    options=[None,'Regional', 'República'],
                    index=None,
                    placeholder="Escriba o seleccione"
                )
            #cargando la inflacion
            if inflacion_choice is not None:
                inflacion_year=load_inflacion(unidecode(inflacion_choice).lower())
        
        # Aplicamos el filtro de variedad si se seleccionó alguna
        if 'Todas' in variedad_select:
            variedad=variedades_insumo
        else:
            variedad=variedad_select
        
        with tab2:
            
            if variedad:
                for idx, v in enumerate(variedad):
                    df_v = df_filtrado[df_filtrado['Unidad de Medida'] == v] #filtrado por codigo de insumo y por variedad
                    if inflacion_choice is not None:
                        #----CORRECCION-POR INFLACION-----
                        anio_max=df_v['Anio Publicacion'].max() #Obtengo el maximo año que hay en la data
                        df_v_anio=df_v[df_v['Anio Publicacion']==anio_max] #filtro con respecto al año max
                        month_max=df_v_anio['Mes Publicacion'].max() #Obtengo el mes maximo que hay en la variedad
                        #el año maximo es el año minimo el cual se debe corregir por inflacion siempre que este año sea menor al actual
                        #el mes maximo-1 es el mes minimo al que se debe corregir la inflacion
                        #obteniendo el año y mes maximos de la inflacion que se tiene en el dataframe inflacion_year
                        #Obtener el último año y mes disponible en inflacion_year
                        inflaY_available = int(inflacion_year['Anio'].max()) # Último año disponible
                        month_availableName = inflacion_year.loc[inflacion_year['Anio']==inflaY_available, 'Mes'].iloc[-1] #el ultimo mes disponible en la lista
                        month_available=meses_dicReverse[month_availableName] #numero del mes disponible en la lista

                        # Generar todas las opciones desde month_max-anio_max hasta month_available-inflaY_available
                        opciones_inflacion = []
                        current_year = anio_max
                        current_month = month_max

                        while current_year <= inflaY_available:
                            # Para el año actual, empezamos desde el mes actual (solo en el primer año)
                            start_month = current_month if current_year == anio_max else 1
                            
                            # Para el último año, terminamos en month_available
                            end_month = month_available if current_year == inflaY_available else 12
                            
                            for month in range(start_month, end_month + 1):
                                opciones_inflacion.append(f"{meses_short[month]}-{current_year}")
                            
                            current_year += 1
                            
                        #seleccionador de inflacion
                        st.sidebar.markdown("---")
                        st.sidebar.markdown(f"**{v}**")
                        inflacion_month = st.sidebar.selectbox(
                                            "Seleccione la fecha:",
                                            options=[None]+opciones_inflacion,
                                            index=0,
                                            key=f"SelectboxID_{v}_{idx}"
                            )
                    else:
                        inflacion_month=None
                    
                    #st.markdown("<hr>", unsafe_allow_html=True)  # Usando HTML
                    #df_v = df_filtrado[df_filtrado['Unidad de Medida'] == v]
                    #fig, stats = plot_Hvariety(df_v, v)
    #===================================
    #   SECTION 2: ANALISIS DE PRECIOS
    #===================================
            
                    with st.expander(f" 📈 **Análisis de precios - {v}**"):
                        #st.markdown(f"##### 📈 Análisis de precios - {v}")
                        col1, col2 = st.columns([0.75, 0.25])
                        
                        with col2:
                            st.markdown("#### 📊 Estadísticas")
                            # Slider para CL en la columna derecha
                            CLevel = st.slider(
                                    f"Seleccione el nivel de confianza (CL)",
                                    min_value=50,
                                    max_value=99,
                                    value=95,  # Valor por defecto
                                    key=f"cl_slider_{v}_{idx}"  # Clave única por variedad
                                )
                            
                        with col1:
                            if inflacion_month==None:
                                fig, stats = plot_Hvariety(df_v, v,CLevel)
                                if fig:
                                    st.plotly_chart(fig, 
                                                        use_container_width=True,
                                                        key=f"histogram_{v}_{idx}"
                                                        )
                            else:
                                col1Histo, col2Histo = st.columns([0.5, 0.5]) #creo dos columnas para mostrar los dos plots
                                anio_fin = int(inflacion_month.split("-")[1])
                                mes_fin = meses_short_inverse[inflacion_month.split("-")[0]]
                                fig, stats = plot_Hvariety(df_v, v,CLevel, True, inflacion_year, anio_fin, mes_fin,inflacion_choice)
                                if fig:
                                    with col1Histo:
                                        st.plotly_chart(fig[0], 
                                                        use_container_width=True,
                                                        key=f"histogram_{v}_{idx}"
                                                        )
                                    with col2Histo:
                                        st.plotly_chart(fig[1], 
                                                        use_container_width=True,
                                                        key=f"histogram_corregido_{v}_{idx}"
                                                        )
                        with col2:
                            if inflacion_month==None:
                                st.dataframe(stats.style.format({"Valor [Q]": "{:.2f}"}, na_rep="--"),use_container_width=True, key=f"stats_{v}_{idx}",  hide_index=True) 
                                #stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Valor [Q]', 'Valor+Inflacion [Q]'])
                                #tabla con la informacion estadistica
                                # Tabla de estadísticas con tamaño de letra adaptable
                                #styled_df = stats_df.style.format({"Valor [Q]": "{:.2f}"})
                                #st.dataframe(styled_df, use_container_width=True, key=f"stats_{v}_{idx}")  # O st.table(styled_df)
                            else:
                                styled_df = stats.style.format({
                                        "Valor [Q]": "{:.2f}",
                                        "Valor+inflacion [Q]": "{:.2f}"
                                    }, na_rep="--")
                
                                st.dataframe(styled_df, use_container_width=True, key=f"stats_{v}_{idx}",  hide_index=True )
                        
        #====================================================
        # Section 3: Evolucion temporal  y por departamento
        #=====================================================
                        col1Sec3, col2Sec3 = st.columns([0.55, 0.45])
                        
                        
                        with col1Sec3:
                            if inflacion_month==None:
                                figEvP=plot_prices_monts(df_v)
                                if figEvP:
                                    st.plotly_chart(figEvP, use_container_width=True, key=f"evolucion_{v}_{idx}")
                            else:
                                figEvP=plot_prices_monts(df_v,True, inflacion_year, anio_fin, mes_fin,inflacion_choice)
                                if figEvP:
                                    st.plotly_chart(figEvP, use_container_width=True, key=f"evolucion_{v}_{idx}")
                        
                        with col2Sec3:
                            if inflacion_month is None:
                                geo_fig, geo_df = plot_map_departamentos(df_v,dfG_dep)
                                geo_fig.update_layout(
                                    autosize=True,
                                    height=None,  # Elimina altura fija
                                    width=None    # Elimina ancho fijo
                                )
                                if geo_fig:
                                    st.plotly_chart(geo_fig, use_container_width=True, key=f"mapa_{v}_{idx}")
                            else:
                                #anio_fin = int(inflacion_month.split("-")[1])
                                #mes_fin = meses_short_inverse[inflacion_month.split("-")[0]]
                                geo_fig, geo_df = plot_map_departamentos(df_v,dfG_dep,True, inflacion_year,anio_fin,mes_fin,inflacion_choice)
                                geo_fig.update_layout(
                                    autosize=True,
                                    height=None,  # Elimina altura fija
                                    width=None    # Elimina ancho fijo
                                )
                                if geo_fig:
                                    st.plotly_chart(geo_fig, use_container_width=True, key=f"mapa_{v}_{idx}")
        #====================================================
        # Section 4: Unidades ofertadas por mes y por departamento
        #=====================================================
                        col1Sec4, col2Sec4 = st.columns([0.50, 0.50])
                        with col1Sec4:
                            fig_uni=plot_unidades_monts(df_v)
                            if fig_uni:
                                st.plotly_chart(fig_uni,use_container_width=True, key=f"uni_{v}_{idx}" )
                        with col2Sec4:
                            if inflacion_month is None:
                                corr_fig1, corr_data1, corr_regresion1 =plot_precio_vs_unidades_inflacion(df_v)
                                if corr_fig1:
                                    st.plotly_chart(corr_fig1, use_container_width=True, key=f"mapa2_{v}_{idx}" )
                            else:
                                corr_fig2, corr_data2, corr_regresion2 =plot_precio_vs_unidades_inflacion(df_v,True,inflacion_year,anio_fin,mes_fin,inflacion_choice)
                                if  corr_fig2:
                                    st.plotly_chart(corr_fig2, use_container_width=True, key=f"corr_infla_{v}_{idx}" )
            else:
                st.markdown("Seleccione una variedad para iniciar con el análisis de precios...")
#=============================
#   ANALISIS RFM
#==============================                                
        with tab3:
            st.markdown("## Análisis RFM", help="""El **análisis RFM** es una técnica poderosa para evaluar el compromiso y valor de cualquier entidad o interacción, 
                        ya sean clientes, instituciones o tipos de productos. Se basa en tres dimensiones clave: **Recencia (R)**, que mide cuán recientemente ocurrió la interacción;
                        **Frecuencia (F)**, que evalúa con qué asiduidad se repite; **Valor Monetario (M)**, que cuantifica el valor económico.""" )

            with st.expander("## **Análisis por variedad**", expanded=False):
                col1bot, col2bot, col3bot = st.columns([0.90,0.05,0.05])
                
                # Botón para gráficos
                with col2bot:
                    if st.button("📊", key="toggle_variedad_plot", help="""Mostrar gráficos """):
                        st.session_state.show_variedad_plots = not st.session_state.get("show_variedad_plots", False)
                        st.session_state.show_variedad_table = False  # Asegurar que la tabla se oculte
                
                # Botón para tablas
                with col3bot:
                    if st.button("🖽", key="toggle_variedad_table", help="""Mostrar tabla de datos """):
                        st.session_state.show_variedad_table = not st.session_state.get("show_variedad_table", False)
                        st.session_state.show_variedad_plots = False  # Asegurar que los gráficos se oculten
                
                # Mostrar gráficos si está activo
                if st.session_state.get("show_variedad_plots", True):
                    col1rf_v, col2rfm_v, col3rfm_v = st.columns([1/3, 1/3, 1/3])
                    col1rfm_best_v, col2rfm_best_v = st.columns([0.9, 0.1])
                    
                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Unidad de Medida')
                    
                    with col1rf_v:
                        st.plotly_chart(fig_rfm_variedad[0], use_container_width=True, key="rfm_v_1")
                    with col2rfm_v:
                        st.plotly_chart(fig_rfm_variedad[1], use_container_width=True, key="rfm_v_2")
                    with col3rfm_v:
                        st.plotly_chart(fig_rfm_variedad[2], use_container_width=True, key="rfm_v_3")
                    with col1rfm_best_v:
                        st.plotly_chart(fig_rfm_variedad[3], use_container_width=True, key="rfm_v_4")
                
                # Mostrar tablas si está activo
                if st.session_state.get("show_variedad_table", False):

                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Unidad de Medida')
                    wR, wF, wM= fig_rfm_variedad[5]
                    df_rfm_v=fig_rfm_variedad[4]
                    df_rfm_v=df_rfm_v.rename(columns={'Unidad de Medida': 'Variedad', 'RFM_Score': '*RFM_Score'})
                    st.markdown(f"**Resumen de los resultados del analísis RFM**")
                    st.dataframe(df_rfm_v, hide_index=True, use_container_width=True, key="rfm_vd_4")
                    st.markdown(
                            f"""
                            *RFM_Score tomando los pesos R={wR}, F={wF}, M={wM}  
                            <a href="/documentacion#analisis-rfm" target="_self" style="text-decoration: none; color: #1f77b4; font-size: 0.8em;">[Ver documentación]</a>
                            """,
                            unsafe_allow_html=True
                        )

            with st.expander("## **Análisis por comprador**", expanded=False):
                col1bot, col2bot, col3bot = st.columns([0.90,0.05,0.05])
                
                # Botón para gráficos
                with col2bot:
                    if st.button("📊", key="toggle_variedad_plot_c", help="""Mostrar gráficos """):
                        st.session_state.show_variedad_plotsc = not st.session_state.get("show_variedad_plotsc", False)
                        st.session_state.show_variedad_tablec = False  # Asegurar que la tabla se oculte
                
                # Botón para tablas
                with col3bot:
                    if st.button("🖽", key="toggle_variedad_tablec", help="""Mostrar tabla de datos """):
                        st.session_state.show_variedad_tablec = not st.session_state.get("show_variedad_tablec", False)
                        st.session_state.show_variedad_plotsc = False  # Asegurar que los gráficos se oculten
                
                # Mostrar gráficos si está activo
                if st.session_state.get("show_variedad_plotsc", True):
                    col1rf_v, col2rfm_v, col3rfm_v = st.columns([1/3, 1/3, 1/3])
                    col1rfm_best_v, col2rfm_best_v = st.columns([0.9, 0.1])
                    
                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Comprador')
                    
                    with col1rf_v:
                        st.plotly_chart(fig_rfm_variedad[0], use_container_width=True, key="rfm_vc_1")
                    with col2rfm_v:
                        st.plotly_chart(fig_rfm_variedad[1], use_container_width=True, key="rfm_vc_2")
                    with col3rfm_v:
                        st.plotly_chart(fig_rfm_variedad[2], use_container_width=True, key="rfm_vc_3")
                    with col1rfm_best_v:
                        st.plotly_chart(fig_rfm_variedad[3], use_container_width=True, key="rfm_vc_4")
                
                # Mostrar tablas si está activo
                if st.session_state.get("show_variedad_tablec", False):

                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Comprador')
                    wR, wF, wM= fig_rfm_variedad[5]
                    df_rfm_v=fig_rfm_variedad[4]
                    df_rfm_v=df_rfm_v.rename(columns={'RFM_Score': '*RFM_Score'})
                    st.markdown(f"**Resumen de los resultados del analísis RFM**")
                    st.dataframe(df_rfm_v, hide_index=True, use_container_width=True, key="rfm_vdc_4")
                    st.markdown(
                            f"""
                            *RFM_Score tomando los pesos R={wR}, F={wF}, M={wM}  
                            <a href="/documentacion#analisis-rfm" target="_self" style="text-decoration: none; color: #1f77b4; font-size: 0.8em;">[Ver documentación]</a>
                            """,
                            unsafe_allow_html=True
                        )

            with st.expander("## **Análisis por oferente**", expanded=False):
                col1bot, col2bot, col3bot = st.columns([0.90,0.05,0.05])
                
                # Botón para gráficos
                with col2bot:
                    if st.button("📊", key="toggle_variedad_plot_o", help="""Mostrar gráficos """):
                        st.session_state.show_variedad_plotso = not st.session_state.get("show_variedad_plotso", False)
                        st.session_state.show_variedad_tableo = False  # Asegurar que la tabla se oculte
                
                # Botón para tablas
                with col3bot:
                    if st.button("🖽", key="toggle_variedad_tableo", help="""Mostrar tabla de datos """):
                        st.session_state.show_variedad_tableo = not st.session_state.get("show_variedad_tableo", False)
                        st.session_state.show_variedad_plotso = False  # Asegurar que los gráficos se oculten
                
                # Mostrar gráficos si está activo
                if st.session_state.get("show_variedad_plotso", True):
                    col1rf_v, col2rfm_v, col3rfm_v = st.columns([1/3, 1/3, 1/3])
                    col1rfm_best_v, col2rfm_best_v = st.columns([0.9, 0.1])
                    
                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Oferente')
                    
                    with col1rf_v:
                        st.plotly_chart(fig_rfm_variedad[0], use_container_width=True, key="rfm_vo_1")
                    with col2rfm_v:
                        st.plotly_chart(fig_rfm_variedad[1], use_container_width=True, key="rfm_vo_2")
                    with col3rfm_v:
                        st.plotly_chart(fig_rfm_variedad[2], use_container_width=True, key="rfm_vo_3")
                    with col1rfm_best_v:
                        st.plotly_chart(fig_rfm_variedad[3], use_container_width=True, key="rfm_vo_4")
                
                # Mostrar tablas si está activo
                if st.session_state.get("show_variedad_tableo", False):

                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Oferente')
                    wR, wF, wM= fig_rfm_variedad[5]
                    df_rfm_v=fig_rfm_variedad[4]
                    df_rfm_v=df_rfm_v.rename(columns={'RFM_Score': '*RFM_Score'})
                    st.markdown(f"**Resumen de los resultados del analísis RFM**")
                    st.dataframe(df_rfm_v, hide_index=True, use_container_width=True, key="rfm_vdo_4")
                    st.markdown(
                            f"""
                            *RFM_Score tomando los pesos R={wR}, F={wF}, M={wM}  
                            <a href="/documentacion#analisis-rfm" target="_self" style="text-decoration: none; color: #1f77b4; font-size: 0.8em;">[Ver documentación]</a>
                            """,
                            unsafe_allow_html=True
                        )


            with st.expander("## **Análisis por modalidad de compra**", expanded=False):
                col1bot, col2bot, col3bot = st.columns([0.90,0.05,0.05])
                
                # Botón para gráficos
                with col2bot:
                    if st.button("📊", key="toggle_variedad_plot_mc", help="""Mostrar gráficos """):
                        st.session_state.show_variedad_plotsdo = not st.session_state.get("show_variedad_plotsdo", False)
                        st.session_state.show_variedad_tabledo = False  # Asegurar que la tabla se oculte
                
                # Botón para tablas
                with col3bot:
                    if st.button("🖽", key="toggle_variedad_tabledomc", help="""Mostrar tabla de datos """):
                        st.session_state.show_variedad_tabledo = not st.session_state.get("show_variedad_tabledo", False)
                        st.session_state.show_variedad_plotsdo = False  # Asegurar que los gráficos se oculten
                
                # Mostrar gráficos si está activo
                if st.session_state.get("show_variedad_plotsdo", True):
                    col1rf_v, col2rfm_v, col3rfm_v = st.columns([1/3, 1/3, 1/3])
                    col1rfm_best_v, col2rfm_best_v = st.columns([0.9, 0.1])
                    
                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Modalidad')
                    
                    with col1rf_v:
                        st.plotly_chart(fig_rfm_variedad[0], use_container_width=True, key="rfm_vdomc_1")
                    with col2rfm_v:
                        st.plotly_chart(fig_rfm_variedad[1], use_container_width=True, key="rfm_vdomc_2")
                    with col3rfm_v:
                        st.plotly_chart(fig_rfm_variedad[2], use_container_width=True, key="rfm_vdmc_3")
                    with col1rfm_best_v:
                        st.plotly_chart(fig_rfm_variedad[3], use_container_width=True, key="rfm_vdmc_4")
                
                # Mostrar tablas si está activo
                if st.session_state.get("show_variedad_tabledo", False):

                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Modalidad')
                    wR, wF, wM= fig_rfm_variedad[5]
                    df_rfm_v=fig_rfm_variedad[4]
                    df_rfm_v=df_rfm_v.rename(columns={'Region Oferente': 'Departamento Oferente','RFM_Score': '*RFM_Score'})
                    st.markdown(f"**Resumen de los resultados del analísis RFM**")
                    st.dataframe(df_rfm_v, hide_index=True, use_container_width=True, key="rfm_vddo_4")
                    st.markdown(
                            f"""
                            *RFM_Score tomando los pesos R={wR}, F={wF}, M={wM}  
                            <a href="/documentacion#analisis-rfm" target="_self" style="text-decoration: none; color: #1f77b4; font-size: 0.8em;">[Ver documentación]</a>
                            """,
                            unsafe_allow_html=True
                        )

            with st.expander("## **Análisis por departamento-oferente**", expanded=False):
                col1bot, col2bot, col3bot = st.columns([0.90,0.05,0.05])
                
                # Botón para gráficos
                with col2bot:
                    if st.button("📊", key="toggle_variedad_plot_do", help="""Mostrar gráficos """):
                        st.session_state.show_variedad_plotsdo = not st.session_state.get("show_variedad_plotsdo", False)
                        st.session_state.show_variedad_tabledo = False  # Asegurar que la tabla se oculte
                
                # Botón para tablas
                with col3bot:
                    if st.button("🖽", key="toggle_variedad_tabledo", help="""Mostrar tabla de datos """):
                        st.session_state.show_variedad_tabledo = not st.session_state.get("show_variedad_tabledo", False)
                        st.session_state.show_variedad_plotsdo = False  # Asegurar que los gráficos se oculten
                
                # Mostrar gráficos si está activo
                if st.session_state.get("show_variedad_plotsdo", True):
                    col1rf_v, col2rfm_v, col3rfm_v = st.columns([1/3, 1/3, 1/3])
                    col1rfm_best_v, col2rfm_best_v = st.columns([0.9, 0.1])
                    
                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Region Oferente')
                    
                    with col1rf_v:
                        st.plotly_chart(fig_rfm_variedad[0], use_container_width=True, key="rfm_vdo_1")
                    with col2rfm_v:
                        st.plotly_chart(fig_rfm_variedad[1], use_container_width=True, key="rfm_vdo_2")
                    with col3rfm_v:
                        st.plotly_chart(fig_rfm_variedad[2], use_container_width=True, key="rfm_vdo_3")
                    with col1rfm_best_v:
                        st.plotly_chart(fig_rfm_variedad[3], use_container_width=True, key="rfm_vdo_4")
                
                # Mostrar tablas si está activo
                if st.session_state.get("show_variedad_tabledo", False):

                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Region Oferente')
                    wR, wF, wM= fig_rfm_variedad[5]
                    df_rfm_v=fig_rfm_variedad[4]
                    df_rfm_v=df_rfm_v.rename(columns={'Region Oferente': 'Departamento Oferente','RFM_Score': '*RFM_Score'})
                    st.markdown(f"**Resumen de los resultados del analísis RFM**")
                    st.dataframe(df_rfm_v, hide_index=True, use_container_width=True, key="rfm_vddo_4")
                    st.markdown(
                            f"""
                            *RFM_Score tomando los pesos R={wR}, F={wF}, M={wM}  
                            <a href="/documentacion#analisis-rfm" target="_self" style="text-decoration: none; color: #1f77b4; font-size: 0.8em;">[Ver documentación]</a>
                            """,
                            unsafe_allow_html=True
                        )

            with st.expander("## **Análisis por departamento/municipio-comprador**", expanded=False):
                col1bot, col2bot, col3bot = st.columns([0.90,0.05,0.05])
                
                # Botón para gráficos
                with col2bot:
                    if st.button("📊", key="toggle_variedad_plot_do_co", help="""Mostrar gráficos """):
                        st.session_state.show_variedad_plotsdo = not st.session_state.get("show_variedad_plotsdo", False)
                        st.session_state.show_variedad_tabledo = False  # Asegurar que la tabla se oculte
                
                # Botón para tablas
                with col3bot:
                    if st.button("🖽", key="toggle_variedad_tabledo_co", help="""Mostrar tabla de datos """):
                        st.session_state.show_variedad_tabledo = not st.session_state.get("show_variedad_tabledo", False)
                        st.session_state.show_variedad_plotsdo = False  # Asegurar que los gráficos se oculten
                
                # Mostrar gráficos si está activo
                if st.session_state.get("show_variedad_plotsdo", True):
                    col1rf_v, col2rfm_v, col3rfm_v = st.columns([1/3, 1/3, 1/3])
                    col1rfm_best_v, col2rfm_best_v = st.columns([0.9, 0.1])
                    
                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Region Comprador')
                    
                    with col1rf_v:
                        st.plotly_chart(fig_rfm_variedad[0], use_container_width=True, key="rfm_vdocom_1")
                    with col2rfm_v:
                        st.plotly_chart(fig_rfm_variedad[1], use_container_width=True, key="rfm_vdocom_2")
                    with col3rfm_v:
                        st.plotly_chart(fig_rfm_variedad[2], use_container_width=True, key="rfm_vdocom_3")
                    with col1rfm_best_v:
                        st.plotly_chart(fig_rfm_variedad[3], use_container_width=True, key="rfm_vdocom_4")
                
                # Mostrar tablas si está activo
                if st.session_state.get("show_variedad_tabledo", False):

                    fig_rfm_variedad = plot_rfm_norm(df_filtrado, 'Region Comprador')
                    wR, wF, wM= fig_rfm_variedad[5]
                    df_rfm_v=fig_rfm_variedad[4]
                    df_rfm_v=df_rfm_v.rename(columns={'Region Comprador': 'Departamento Comprador','RFM_Score': '*RFM_Score'})
                    st.markdown(f"**Resumen de los resultados del analísis RFM**")
                    st.dataframe(df_rfm_v, hide_index=True, use_container_width=True, key="rfm_vddo_com_4")
                    st.markdown(
                            f"""
                            *RFM_Score tomando los pesos R={wR}, F={wF}, M={wM}  
                            <a href="/documentacion#analisis-rfm" target="_self" style="text-decoration: none; color: #1f77b4; font-size: 0.8em;">[Ver documentación]</a>
                            """,
                            unsafe_allow_html=True
                        )

#==============================
#   ANALISIS ABC
#==============================
        with tab4:
            st.markdown("## Análisis ABC", help=""" El **análisis ABC** es una técnica de clasificación que permite priorizar productos o insumos 
                        en función de su impacto económico total. Agrupa los elementos en tres categorías: **Clase A**: productos críticos que representan la mayor parte del valor ofertado,
                        aunque sean pocos en número. **Clase B**: productos intermedios que contribuyen moderadamente al valor. **Clase C**: numerosos productos con bajo impacto individual. 
                        Este análisis sigue el principio de Pareto: aproximadamente el 20% de los productos representan el 80% del valor. Es útil para enfocar estrategias de monitoreo,
                        control de precios y eficiencia en las compras públicas.""")
            
            with st.expander("**Análisis por variedad**", expanded=False):
                render_abc_block(df_filtrado, "Variedad", "variedad", grupo_principal="Unidad de Medida", show_map=False)

            with st.expander("**Análisis por comprador**", expanded=False):
                render_abc_block(df_filtrado, "Comprador", "comprador", grupo_principal="Comprador", show_map=False)

            with st.expander("**Análisis por oferente**", expanded=False):
                render_abc_block(df_filtrado, "Oferente", "oferente", grupo_principal="Oferente", show_map=False)

            with st.expander("**Análisis por modalidad de compra**", expanded=False):
                render_abc_block(df_filtrado, "Modalidad", "modalidad", grupo_principal="Modalidad", show_map=False)

            with st.expander("**Análisis por departamento/municipio-oferente**", expanded=False):
                render_abc_block(
                    df_filtrado,
                    "Depto/Muni Oferente",
                    "depOf",
                    grupo_principal="Region Oferente",
                    grupo_secundario="Localidad Oferente",
                    map_func_principal=plot_map_abc_dep,
                    map_func_secundario=plot_map_abc_muni,
                    df_geo_principal=dfG_dep,
                    df_geo_secundario=dfG_muni,
                    show_map=True
                )

            with st.expander("**Análisis por departamento/municipio-comprador**", expanded=False):
                render_abc_block(
                    df_filtrado,
                    "Depto/Muni Comprador",
                    "depCo",
                    grupo_principal="Region Comprador",
                    grupo_secundario="Localidad Comprador",
                    map_func_principal=plot_map_abc_dep,
                    map_func_secundario=plot_map_abc_muni,
                    df_geo_principal=dfG_dep,
                    df_geo_secundario=dfG_muni,
                    show_map=True
                )
                     
            # --- Selector de formato de descarga ---
        st.sidebar.markdown("---")  # Línea separadora
        formato_descarga = st.sidebar.radio(
            "📤 Descargar info. por código:",
            options=["CSV", "TXT", "Excel (XLSX)"],
            index=0,  # Opción predeterminada (CSV)
            horizontal=True  # Diseño en línea
        )

        # --- Botón de descarga dinámico ---
        if df_filtrado:
            change_columns={'Localidad Oferente':'Municipio Oferente','Region Oferente': 'Departamento Oferente', 
                            'Localidad Comprador':'Municipio Comprador','Region Comprador': 'Departamento Comprador'}
            df_filtrado=df_filtrado.raname(columns=change_columns)
        if formato_descarga == "CSV":
            file_extension = "csv"
            mime_type = "text/csv"
            data = df_filtrado.to_csv(index=False).encode('utf-8-sig')
        elif formato_descarga == "TXT":
            file_extension = "txt"
            mime_type = "text/plain"
            data = df_filtrado.to_csv(index=False, sep='\t').encode('utf-8-sig')  # TXT con tabulador
        else:  # Excel (XLSX)
            file_extension = "xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_filtrado.to_excel(writer, index=False)
            data = output.getvalue()

        # Nombre del archivo con año y formato
        filename = f"datos_filtrados_{year}.{file_extension}"

        st.sidebar.download_button(
            label=f"⬇️ Descargar como {formato_descarga}",
            data=data,
            file_name=filename,
            mime=mime_type,
            help=f"Descarga los datos filtrados en formato {formato_descarga}"
        )            
    #st.write(df_filtrado)
    # Mostramos el DataFrame filtrado
    #st.write(df_filtrado)
    # =============================================
    # Botones flotantes de Scroll (JavaScript)
    # =============================================
    else:
        # Si no se seleccionó ningún código, mostramos todo
        st.markdown(f"La base de datos de este año cuenta con **{len(codigos_insumo)} códigos de insumo**.")
        st.markdown("Para obtener el precio estimado de un insumo, **seleccione** en la barra derecha el **código de insumo y variedad de interes**.")            


else:
# Mensaje de bienvenida
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 30px;">
    <p>  <strong> Bienvenido al Sistema de Monitoreo de Precios Estatales </strong>  </p>

    Esta herramienta, desarrollada por la **Dirección de Precios e Índices de Referencia (DPIR)**, 
    está diseñada para el análisis estratégico de precios en contrataciones públicas. Los datos 
    que se visualizan aquí han sido extraídos directamente desde la API y el portal web de Guatecompras **unicamente para aquellos eventos con estatus de ADJUDICADO**.  La asignación del 
    código de insumo se ha realizado mediante el modelo all-MiniLM-L6-v2-similarity-es. 
    **Este sistema debe considerarse un complemento de apoyo para el analista de precios, 
    facilitando la identificación de tendencias y comparaciones en el mercado estatal.** 
    
    </div>

    ---

    ### 🔍 ¿Qué puede hacer esta plataforma?

    1. **Analizar** precios unitarios históricos de miles de insumos
    2. **Comparar** adjudicaciones entre instituciones y regiones
    3. **Detectar** patrones y anomalías en compras gubernamentales
    4. **Optimizar** presupuestos con inteligencia de mercado

    ---

    ### 🛠️ ¿Cómo utilizarlo?

    1. **Seleccione el año** de interés en el panel izquierdo
    2. **Busque un insumo** por código o descripción
    3. **Explore las variedades** disponibles
    4. **Active filtros** avanzados según sus necesidades
    5. **Descargue reportes** en múltiples formatos

    ---

    ### 📊 Principales funcionalidades

    | Módulo | Descripción | Beneficio |
    |--------|-------------|-----------|
    | **Análisis ABC** | Clasificación de insumos por impacto económico | Priorización eficiente |
    | **Evolución temporal** | Series históricas con ajuste por inflación | Tendencias a largo plazo |
    | **Mapas interactivos** | Comparación geográfica de precios | Identificación de mercados regionales |
    | **RFM avanzado** | Evaluación multidimensional de proveedores | Mejor selección de contratistas |

    ---

    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 30px;">
        <h3 style="color: #2c3e50; margin-top: 0;">📌 ¿Primera vez usando el sistema?</h3>
        <p>Revise nuestra <a href="/documentacion" target="_self">guía técnica</a> o contacte al equipo DPIR:</p>
        <p>📧 <a href="mailto:dpir@ine.gob.gt">dpir@ine.gob.gt</a> | 📞 (+502) 1234-5678</p>
    </div>

    <div style="text-align: center; margin-top: 40px; font-size: 0.9em; color: #7f8c8d;">
        <p>Sistema desarrollado por la <strong>Dirección de Precios e Índices de Referencia</strong></p>
        <p>Instituto Nacional de Estadística - Guatemala</p>
    </div>
    """, unsafe_allow_html=True)

# Crear dos columnas vacías: una ancha y una angosta
current_folder = Path(os.getcwd())
soruce_folder = "source_data"

col1Logo, col2Logo = st.columns([9, 1])  # col1 es más angosta

with col2Logo:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    image_path = resource_path(current_folder/ soruce_folder/ "DPIR_logo_2.png")
    image=Image.open(image_path)
    st.image(image)