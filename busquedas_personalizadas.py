# Dashboard
import streamlit as st
import pandas as pd
from pathlib import Path
import os
import sys
from unidecode import unidecode

def resource_path(file_path, local_folder_name=None):
    """Busca un archivo en múltiples ubicaciones"""
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    if path_obj.exists():
        return path_obj
    
    if getattr(sys, 'frozen', False):
        if '_MEIPASS' in os.environ:
            base_dir = Path(os.environ['_MEIPASS'])
        else:
            base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent
    
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
            pass
    
    possible_paths = []
    possible_paths.append(base_dir / path_obj.name)
    if not path_obj.is_absolute():
        possible_paths.append(base_dir / path_obj)
    else:
        possible_paths.append(path_obj)
    
    if not path_obj.is_absolute():
        possible_paths.append(base_dir / '_internal' / path_obj)
    else:
        try:
            new_parts = (path_obj.parts[:base_index+1] + ('_internal',) + path_obj.parts[base_index+1:])
            possible_paths.append(Path(*new_parts))
        except ValueError:
            pass
    
    if getattr(sys, 'frozen', False) and '_MEIPASS' in os.environ:
        possible_paths.append(base_dir / '_internal' / path_obj.name)
    
    for test_path in possible_paths:
        try:
            if test_path.exists():
                return test_path
        except (TypeError, AttributeError):
            continue
    
    test_path = Path.cwd() / path_obj if not path_obj.is_absolute() else path_obj
    if test_path.exists():
        return test_path
    
    return path_obj

@st.cache_data
def load_data_year(year):
    """Carga datos para un año específico"""
    year_dir = Path("source_data") / "data_base_guatecompras" / f"{year}"
    dir_path = resource_path(year_dir)
    
    try:
        csv_files = list(dir_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No se encontraron archivos CSV en {dir_path}")
        
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, sep=',', encoding='utf-8-sig', quoting=1, low_memory=False)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
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

@st.cache_data
def load_clasificador():
    """Carga el catálogo de insumos"""
    relative_path = Path("source_data") / "catalogo_insumos_full_description.csv"
    file_path = resource_path(relative_path)
    
    try:
        return pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
    except Exception as e:
        available_files = "\n".join(str(p) for p in file_path.parent.glob("*.csv"))
        raise FileNotFoundError(
            f"No se pudo cargar el catálogo de insumos.\n"
            f"Archivo buscado: {file_path}\n"
            f"Archivos disponibles:\n{available_files}\n"
            f"Error original: {str(e)}"
        )

# Configuración de la página
st.set_page_config(page_title="Búsquedas Personalizadas", layout="wide")

# Logo del INE
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.ine.gob.gt/ine/wp-content/uploads/2017/09/cropped-INE.png" width="150">
    </div>
    """,
    unsafe_allow_html=True
)

# Título principal
st.markdown("# BÚSQUEDAS PERSONALIZADAS")
st.markdown("---")

# Años disponibles
years_disp = [2020, 2021, 2022, 2023, 2024]

# Filtro de años
year = st.sidebar.multiselect(
    "Seleccione el año:",
    options=['Todos'] + years_disp,
    default=None,
    placeholder="Escriba o seleccione..."
)

# Cargar datos si se seleccionó algún año
if year:
    dfTemp = []
    if any(str(t).lower() == 'todos' for t in year):
        for k in years_disp:
            dfTemp.append(load_data_year(k))
    else:
        for k in year:
            dfTemp.append(load_data_year(k))
    
    dfT = pd.concat(dfTemp, axis=0).reset_index(drop=True)
    dfY = dfT.dropna()
    dfY["Codigo Insumo"] = dfY["Codigo Insumo"].astype(int)
    
    # Cargar clasificador
    dfCI = load_clasificador()
    
    # Obtener renglones únicos
    renglones = sorted(dfY["Renglon"].unique().tolist())
    
    # Selector de renglón
    renglon_select = st.sidebar.selectbox(
        "🔍 Seleccione el renglón:",
        options=renglones,
        index=None,
        placeholder="Escriba o seleccione..."
    )
    
    if renglon_select:
        # Filtrar por renglón
        df_filtrado = dfY[dfY["Renglon"] == renglon_select].copy()
        
        # Obtener información del renglón del clasificador
        renglon_info = dfCI[dfCI["Renglon"] == renglon_select].iloc[0]
        
        # Mostrar información del renglón
        st.markdown(f"### Renglón {renglon_select}: {renglon_info['Concepto Renglon']}")
        st.markdown(f"**Subgrupo:** {renglon_info['Nombre Subgrupo']}")
        st.markdown(f"**Grupo:** {renglon_info['Nombre Grupo']}")
        
        # Mostrar dataframe con los productos del renglón
        st.markdown("### Productos en este renglón")
        
        # Renombrar columnas para mejor visualización
        df_display = df_filtrado.rename(columns={
            'Codigo Insumo': 'Código',
            'Insumo Match': 'Descripción',
            'Unidad de Medida': 'Variedad',
            'Precio unitario': 'Precio (Q)',
            'Cantidad Ofertada': 'Cantidad',
            'Oferente': 'Proveedor',
            'Comprador': 'Entidad Compradora',
            'Region Oferente': 'Departamento Proveedor',
            'Region Comprador': 'Departamento Comprador'
        })
        
        # Columnas a mostrar
        columnas = ['Código', 'Descripción', 'Variedad', 'Precio (Q)', 'Cantidad', 
                   'Proveedor', 'Entidad Compradora', 'Departamento Proveedor', 'Departamento Comprador']
        
        st.dataframe(
            df_display[columnas],
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        # Opción para descargar los datos
        csv = df_display[columnas].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos en CSV",
            data=csv,
            file_name=f"productos_renglon_{renglon_select}.csv",
            mime="text/csv"
        )