import streamlit as st
from pathlib import Path
import os
import sys
#by elser lopez

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
    - [Información General](#información-general)
    - [Análisis de Precios](#analisis-precios)
    - [Análisis RFM](#analisis-rfm)
    - [Análisis ABC](#analisis-abc)
    - [Cómo usar la aplicación](#cómo-usar-la-aplicación)
    """)
    
    st.title("📚 Documentación Técnica")
    
    st.markdown("""
    <a id="acerca-del-observatorio-de-precios"></a>
    #### Acerca del Observatorio de Precios
    
    El Observatorio de Precios Estatales es una plataforma analítica avanzada desarrollada por la Dirección de Precios e Índices de Referencia (DPIR) para monitorear, 
    analizar y optimizar los procesos de compras gubernamentales en Guatemala. Funciona como un sistema de inteligencia de mercado especializado que transforma datos 
    crudos de Guatecompras en información estratégica para la toma de decisiones.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <a id="funcionalidades-principales"></a>
    #### Funcionalidades principales:
    - Análisis de precios por insumo y variedad
    - Corrección por inflación (nacional/regional)
    - Visualizaciones geográficas por departamento
    - Series temporales de evolución de precios
    """, unsafe_allow_html=True)
        

    with st.expander("ℹ **Información General**", expanded=False):
        st.markdown(r"""
        <a id="información-general"></a>            
        ## 📋 Sección de Información General

        Esta sección proporciona un panorama completo del insumo seleccionado, mostrando metadatos clave, distribución por variedad y métricas fundamentales de adjudicación.

        ---

        ### 🔍 Componentes Principales

        1. **Panel de Metadatos**  
        - Muestra información estructural del insumo (Grupo > Subgrupo > Renglón)
        - Incluye características técnicas y descripción completa
        - Lista todas las variedades disponibles con conteo

        2. **Gráfico Circular de Variedades**  
        - Visualización interactiva con porcentajes y valores absolutos
        - Ordenamiento automático por volumen (mayor a menor)
        - Efecto "donut" para mejor legibilidad

        3. **Adjudicaciones por Variedad**  
        - Gráfico de barras horizontales con conteo de adjudicaciones
        - Colorización por variedad (escala cualitativa)
        - Texto de valores y tooltips detallados

        4. **Tiempo de Adjudicación**  
        - Serie temporal con promedio de días por mes-año
        - Líneas diferenciadas por variedad
        - Escala temporal adaptativa (meses o trimestres según rango)

        5. **NOGs por Variedad**  
        - Gráfico de barras verticales con conteo de números de oferta únicos
        - Rotación de etiquetas para mejor visualización
        - Ancho de barras proporcional al volumen

        ---

        ### 📊 Métricas Clave Calculadas

        Para cada variedad del insumo:

        $$
        \text{Unidades Ofertadas} = \sum_{\text{registros}} \text{Cantidad Ofertada}_i
        $$

        $$
        \text{Adjudicaciones} = \sum_{\text{registros}} \mathbb{I}(\text{Adjudicado}_i = 1)
        $$

        $$
        \text{Tiempo Adjudicación} = \frac{\sum (\text{Fecha Adjudicación}_i - \text{Fecha Publicación}_i)}{\text{N° registros}}
        $$

        $$
        \text{NOGs Únicos} = \text{Cardinalidad}(\{\text{NOG}_1, \text{NOG}_2, ..., \text{NOG}_n\})
        $$

        ---

        ### 🛠️ Procesamiento de Datos

        ```python
        # Filtrado inicial
        df_filtrado = dfY[dfY["Codigo Insumo"].isin([insumoCode])]
        
        # Ordenamiento de variedades
        orden_variedades = (df_filtrado.groupby('Unidad de Medida')['Cantidad Ofertada']
                            .sum()
                            .sort_values(ascending=False)
                            .index.tolist())
        ```

        ---

        ### 📈 Interpretación de Resultados

        - **Variedad Dominante**: Mayor área en gráfico circular + barras más largas
        - **Eficiencia de Mercado**: ↓ Tiempo adjudicación + ↑ NOGs
        - **Concentración**: >70% unidades/adjudicaciones en pocas variedades

        ---

        ### ⚠️ Limitaciones

        1. Tiempos de adjudicación pueden incluir outliers
        2. Correlación NOGs-variedad ≠ causalidad
        3. Inconsistencias en datos históricos

        🔍 *Sugerencia:* Combine con análisis ABC para priorización estratégica.
        """, unsafe_allow_html=True)


    with st.expander("💰 **Análisis de Precios**"):
        st.markdown(r"""
        <a id="analisis-precios"></a>

        ## 💰 Análisis de Precios Unitarios

        Esta sección documenta la metodología para analizar precios unitarios ofertados por insumo y variedad en Guatecompras, incluyendo su distribución, evolución mensual, comparación regional y ajuste por inflación.

        ---

        ### 📊 Histogramas de Precios

        Se genera un **histograma ponderado por cantidad ofertada**, que refleja la frecuencia de precios observados. El número de intervalos (bins) se calcula con la **regla de Sturges**:

        $$
        N_{\text{bins}} = 1 + \log_2(N)
        $$

        donde \(N\) es la **suma total de unidades ofertadas**.

        ---

        ### 🌫️ Estimación de Densidad (KDE)

        Se incluye una curva de **estimación de densidad por kernel (KDE)**, la cual suaviza la distribución del histograma y facilita identificar la **moda de la distribución** (precio más común).

        - **Núcleo utilizado:** gaussiano (distribución normal).
        - **Aplicación:** permite ver tendencias, agrupaciones, y asimetrías que podrían no ser evidentes en el histograma.

        > 💡 *Insight:* si el KDE tiene múltiples picos, podría haber varios submercados o tipos de proveedor con estrategias de precios distintas.

        ---

        ### 📐 Métricas Estadísticas Clave

        Se calculan tanto para precios originales como ajustados por inflación:

        #### 1. **Media Geométrica**:
        $$
        \overline{x}_g = \left( \prod_{i=1}^{n} x_i \right)^{1/n}
        $$
        - **Uso:** Ideal para comparar precios cuando hay alta dispersión o presencia de valores extremos.
        - **Ventaja:** Reduce el efecto de precios atípicos.

        #### 2. **Media Aritmética**:
        $$
        \overline{x}_a = \frac{1}{n} \sum_{i=1}^{n} x_i
        $$
        - **Uso:** Buena para comparaciones simples cuando la distribución es simétrica.
        - **Riesgo:** Puede distorsionarse por precios extremos.

        #### 3. **Desviación Estándar**:
        $$
        \sigma = \sqrt{ \frac{1}{n - 1} \sum (x_i - \overline{x}_a)^2 }
        $$
        - **Uso:** Mide la dispersión del precio ofertado.
        - **Interpretación:** Alta desviación indica mercado heterogéneo; baja indica consenso de precios.

        #### 4. **Intervalos de Confianza (percentiles):**
        $$
        P_{\min} = P_{50 - CL/2}, \quad P_{\max} = P_{50 + CL/2}
        $$
        - **Uso:** Delimita el rango central de precios donde se concentra la mayoría de ofertas.
        - **Insight:** Si el rango es amplio, puede indicar incertidumbre o falta de estandarización.

        #### 5. **Moda KDE (estimada):**
        - Se identifica el valor de precio donde la curva KDE alcanza su máximo.
        - **Interpretación:** Precio más comúnmente ofertado.

        ---

        ### 📈 Series Temporales

        Se muestra la evolución de precios por mes (o mes-año si hay más de un año):

        - Media geométrica mensual.
        - Desviación estándar como banda de error.
        - Puede incluir precios corregidos por inflación.

        > 💡 *Insight:* Permite detectar estacionalidades, aumentos sistemáticos, o efectos de política pública.

        ---

        ### 🌍 Mapas por Departamento

        Se visualiza la media geométrica de precios ofertados por departamento mediante un mapa coroplético:

        - **Color:** Nivel de precio promedio.
        - **Tooltip:** Incluye también desviación estándar y versión ajustada por inflación.
        - **Corrección por inflación:** Opcional a mes/año de referencia.

        > 💡 *Insight:* Departamentos con precios consistentemente más altos podrían requerir revisión de condiciones de mercado, logística o competencia.

        ---

        ### 📉 Relación Precio - Unidades

        Se calcula una **regresión lineal** entre precio unitario y unidades ofertadas:

        - Se muestra la línea de tendencia con su intervalo de confianza del 95%.
        - Se reporta el coeficiente de correlación \( r \).

        > 💡 *Insight:* Un coeficiente negativo fuerte sugiere economía de escala (a mayor volumen, menor precio). Un coeficiente cercano a cero indica independencia.

        ---

        ### 💵 Corrección por Inflación

        Todos los precios pueden ser ajustados para eliminar el efecto de inflación, comparando en términos reales.

        #### Fórmulas:

        - **Corrección Nacional:**
        $$
        P_{\text{ajustado}}(t) = P_{original}(t_{0}) \dfrac{ IPC_{nacional}(t) }{ IPC_{nacional}(t_{0}) } 
        $$

        - **Corrección Regional:**
        $$
        P_{\text{ajustado}}(t) = P_{original}(t_{0}) \dfrac{ IPC_{regional}(t) }{ IPC_{regional}(t_{0}) } 
        $$

        donde $$ t $$ es la fecha a la que se requiere ajustar el precio dado en la fecha $$t_0$$.
        
        > 💡 *Insight:* El ajuste permite comparar precios de años diferentes como si fueran del mismo periodo económico.

        ---

        """, unsafe_allow_html=True)

    with st.expander("🔄 **Análisis RFM**"):
        st.markdown(r"""
        <a id="analisis-rfm"></a>

        ## 🔄 Análisis RFM: Recencia, Frecuencia y Valor Monetario

        El **análisis RFM** es una técnica poderosa para evaluar el compromiso y valor de cualquier entidad o interacción,
                    ya sean clientes, instituciones o tipos de productos. Se basa en tres dimensiones clave:
                    
        - **Recencia (R)**, que mide cuán recientemente ocurrió la interacción;
        - **Frecuencia (F)**, que evalúa con qué asiduidad se repite;
        - **Valor Monetario (M)**, que cuantifica el valor económico.
        
        Esta técnica es ampliamente utilizada en marketing y análisis de clientes, pero en este caso ha sido adaptada para analizar insumos o proveedores del sistema de compras públicas.

        ---

        ### 📐 Cálculo de Métricas RFM

        Dado un conjunto de datos con información de compras, se construyen las siguientes métricas:

        **1. Recencia (R)**  
        Cantidad de días desde la compra más reciente hasta la fecha máxima del conjunto de datos:

        $$
        R = \text{días}(\text{fecha}_{\text{máx}} - \text{última fecha de compra})
        $$

        **2. Frecuencia (F)**  
        Número de veces que se ha registrado una compra:

        $$
        F = \text{número de registros de compra}
        $$

        **3. Valor Monetario (M)**  
        Monto total ofertado por un proveedor o insumo:

        $$
        M = \sum_{i=1}^{n} \left( \text{Precio Unitario}_i \times \text{Cantidad Ofertada}_i \right)
        $$

        ---

        ### 📏 Normalización Min-Max

        Para escalar todas las métricas en el mismo rango (0 a 100):

        - **Recencia (invertida, porque menor es mejor):**

        $$
        R_{\text{norm}} = (1 - \frac{R - R_{\min}}{R_{\max} - R_{\min}}) \times 100
        $$

        - **Frecuencia y Valor:**

        $$
        F_{\text{norm}} = \frac{F - F_{\min}}{F_{\max} - F_{\min}} \times 100 \\
        M_{\text{norm}} = \frac{M - M_{\min}}{M_{\max} - M_{\min}} \times 100
        $$

        ---

        ### 🧮 Puntaje RFM

        Una vez normalizadas las métricas, se calcula un puntaje RFM combinado usando pesos:

        $$
        \text{RFM Score} = R_w \cdot R_{\text{norm}} + F_w \cdot F_{\text{norm}} + M_w \cdot M_{\text{norm}}
        $$

        Con la condición de que:

        $$
        R_w + F_w + M_w = 1
        $$

        ---

        ### 🧠 Selección Óptima de Pesos

        Para encontrar la mejor combinación de pesos se prueban múltiples combinaciones y se calcula el **coeficiente de variación (CV)** del puntaje RFM:

        $$
        CV = \frac{\sigma_{\text{RFM}}}{\mu_{\text{RFM}}}
        $$

        Se selecciona la combinación que **minimiza el CV**, lo cual permite maximizar la diferenciación entre observaciones.

        ---

        ### 📈 Visualización

        Se generan gráficos con los **Top 10** elementos según:

        - Recencia normalizada (más recientes)
        - Frecuencia (más compras)
        - Valor monetario ofertado
        - Puntaje RFM total (con los pesos óptimos)

        Estos gráficos permiten identificar insumos o proveedores más activos y valiosos en el sistema de compras públicas.

                """, unsafe_allow_html=True)

    with st.expander(" **Análisis ABC**"):
        st.markdown(r"""
    <a id="analisis-abc"></a>

    # 📊 Análisis ABC

    El **análisis ABC** es una técnica de categorización basada en el **principio de Pareto (80/20)**, aplicada para identificar los elementos que concentran la mayor parte del valor económico en un conjunto de datos. En el contexto de **compras públicas**, permite clasificar productos, oferentes, compradores o unidades según su impacto económico en las adjudicaciones.

    ---

    ## 🧮 Fundamento Matemático

    Para cada grupo \( i \) por ejemplo: 'Variedad', 'Oferente', etc., se calcula el **valor total adjudicado**:

    $$
    V_i = \sum_{j \in i} \left( \text{Precio Unitario}_j \times \text{Cantidad Ofertada}_j \right)
    $$

    Luego, se calcula:

    - **Porcentaje individual** de participación:

    $$
    p_i = \frac{V_i}{\sum_k V_k}
    $$

    - **Porcentaje acumulado** ordenando de mayor a menor:

    $$
    P_i = \sum_{k=1}^{i} p_k
    $$

    ---

    ## 🧮 Clasificación ABC

    Con base en el porcentaje acumulado \( P_i \), cada grupo se clasifica como:

    $$
    \text{Clase}_i =
    \begin{cases}
    \text{A} & \text{si } P_i \leq 0.80 \\
    \text{B} & \text{si } 0.80 < P_i \leq 0.95 \\
    \text{C} & \text{si } P_i > 0.95
    \end{cases}
    $$

    Esta clasificación ayuda a focalizar la atención sobre los elementos más relevantes económicamente.

    ---

    ## 📊 Visualización

    ### 📊 Gráfico de Barras

    Muestra el **valor adjudicado** de cada grupo, ordenado de mayor a menor, y **coloreado por clase ABC**. Permite visualizar rápidamente cuáles son los elementos dominantes y su categoría.

    - Eje X: Valor total adjudicado.
    - Eje Y: Nombre del grupo.
    - Colores: Representan las clases A, B o C (paleta pastel).

    ### 🥧 Gráfico de Pastel

    Representa la **proporción total del valor adjudicado** que corresponde a cada **clase ABC**.

    - Utiliza una paleta **Pastel** de colores.
    - Muestra porcentajes y totales.
    - Permite identificar visualmente el **peso relativo de cada clase**.

    ---

    ## 🗺️ Visualización Geográfica (opcional)

    Si se activa la opción **"Mostrar mapa"**, se generan mapas interactivos utilizando datos geoespaciales:

    ### 🗺️ Mapa por Departamento

    - Representa cada departamento de Guatemala.
    - El color indica el **valor promedio adjudicado**, ajustado o no por inflación.
    - Permite identificar **regiones con alta actividad económica** en compras estatales.

    ### 🗺️ Mapa por Municipio

    - Más detallado: visualiza el desglose por municipio.
    - Útil para análisis territoriales o regionales.
    - Puede ser filtrado por variedad, oferente o comprador.

    Ambos mapas usan **Plotly** y permiten zoom, tooltips y navegación interactiva.

    ---

    ## 🧾 Tablas de Resumen

    Al final del bloque se incluyen una o más **tablas interactivas** con los siguientes detalles:

    - Nombre del grupo (por ejemplo: unidad, comprador, oferente).
    - Valor adjudicado total.
    - Porcentaje y porcentaje acumulado.
    - Clasificación ABC.
    - Otras métricas auxiliares (si aplica).

    Las tablas permiten ordenar y explorar los datos detalladamente, con valores **ajustados por inflación si se selecciona la opción**.

    ---

    ## 💵 Corrección por Inflación (opcional)

    Cuando el análisis cubre múltiples años, se puede aplicar una **corrección por inflación** para expresar todos los valores monetarios en términos reales:

    $$
    P_{\text{ajustado}} = P_{\text{original}} \times \frac{IPC_{\text{fin}}}{IPC_{\text{inicio}}}
    $$

    Esto asegura una comparación justa del **valor adjudicado a lo largo del tiempo**, tomando como referencia el poder adquisitivo de un periodo determinado.

    ---

    ## 🧠 Interpretación Estratégica

    - **Clase A**: Elementos críticos que representan la mayoría del valor. Requieren análisis y control detallado.
    - **Clase B**: Elementos intermedios que pueden optimizarse con estrategias de eficiencia.
    - **Clase C**: Elementos de bajo impacto económico. Se pueden revisar en bloque, consolidar o estandarizar.

    ---

    """, unsafe_allow_html=True)



        col1Logo, col2Logo = st.columns([9, 1])

        with col2Logo:
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.image(resource_path(current_folder / soruce_folder / "DPIR_logo_2.png"), width=120)

    with st.expander("🛠️ Cómo usar la aplicación"):
        st.markdown("""
        <a id="cómo-usar-la-aplicación"></a>
        1. Seleccione el año de interés en el sidebar
        2. Elija el código de insumo (puede buscar por código o descripción)
        3. Filtre por variedad si es necesario
        4. Active corrección por inflación si lo requiere
        5. Explore los gráficos y tablas generadas
        """, unsafe_allow_html=True)
        

# Llamada a la función principal
show()