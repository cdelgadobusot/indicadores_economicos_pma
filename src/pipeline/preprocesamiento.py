"""
preprocesamiento.py  —  MÓDULO 1 (parte B): PREPROCESAMIENTO Y TRANSFORMACIÓN
=============================================================================
Toma los datos crudos en formato largo que produce `ingesta.ingestar_todo()` y
los convierte en datasets limpios y listos para el análisis, el Machine Learning
y el dashboard.

Etapas implementadas:
    1. Limpieza      -> tipos correctos, eliminación de duplicados, orden.
    2. Pivoteo       -> de formato largo (tidy) a formato ancho (una columna por
                        indicador), que es el más cómodo para modelar.
    3. Manejo de nulos -> interpolación temporal + relleno hacia adelante/atrás.
    4. Ingeniería de características (feature engineering) -> variaciones anuales,
       medias móviles y rezagos (lags) que enriquecen el dataset para los modelos.

Cada etapa está en su propia función para que el notebook pueda explicarlas una
por una. La función `preprocesar_todo()` las encadena y guarda los resultados en
`data/processed/`.
"""
from __future__ import annotations

import pandas as pd

from .. import config


# ===========================================================================
# 1. LIMPIEZA
# ===========================================================================
def limpiar(largo: pd.DataFrame) -> pd.DataFrame:
    """Limpia el DataFrame largo crudo.

    - Convierte 'anio' a entero y 'valor' a numérico (los no convertibles -> NaN).
    - Elimina filas sin año.
    - Elimina duplicados de (anio, codigo) conservando el último.
    - Ordena por indicador y año.

    Returns
    -------
    pd.DataFrame con columnas ['anio', 'codigo', 'valor', 'fuente'] ya limpias.
    """
    df = largo.copy()  # copiamos para no mutar el DataFrame original recibido

    # pd.to_numeric con errors="coerce": convierte a número y, si un valor NO se
    # puede convertir (texto raro, vacío), lo vuelve NaN en vez de lanzar error.
    # astype("Int64") es el entero "nullable" de pandas (admite NaN).
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    # Una observación sin año no sirve para una serie de tiempo: la quitamos.
    df = df.dropna(subset=["anio"])
    df["anio"] = df["anio"].astype(int)  # ya sin NaN, pasamos a entero normal

    # Si por algún motivo llegara un (anio, codigo) repetido, conservamos el
    # último valor observado (la fuente más reciente). keep="last".
    df = df.drop_duplicates(subset=["anio", "codigo"], keep="last")

    # Ordenar por (indicador, año) deja cada serie en orden cronológico, lo cual es
    # IMPRESCINDIBLE para que rezagos y medias móviles se calculen bien después.
    df = df.sort_values(["codigo", "anio"]).reset_index(drop=True)
    return df


# ===========================================================================
# 2. PIVOTEO A FORMATO ANCHO
# ===========================================================================
def a_formato_ancho(largo: pd.DataFrame) -> pd.DataFrame:
    """Convierte de formato largo a ancho (pivot).

    Resultado: una fila por año y una columna por indicador. Las columnas se
    ordenan según el catálogo de `config` para mantener consistencia.
    """
    # pivot_table "gira" la tabla:
    #   index="anio"     -> cada año será UNA fila.
    #   columns="codigo" -> cada valor único de la columna 'codigo'
    #                       (pib_crecimiento, inflacion, ...) se vuelve UNA columna.
    #   values="valor"   -> el contenido de cada celda es el número de 'valor'.
    ancho = largo.pivot_table(index="anio", columns="codigo", values="valor")

    # Reordena columnas en el orden del catálogo (solo las presentes).
    columnas_ordenadas = [c for c in config.CODIGOS_INDICADORES if c in ancho.columns]
    ancho = ancho[columnas_ordenadas]

    # 'anio' deja de ser índice para volverse una columna normal.
    ancho = ancho.reset_index().sort_values("anio").reset_index(drop=True)
    ancho.columns.name = None
    return ancho


# ===========================================================================
# 3. MANEJO DE VALORES NULOS
# ===========================================================================
def manejar_nulos(ancho: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputa valores faltantes en el DataFrame ancho.

    Estrategia (apropiada para series de tiempo anuales):
        1. Interpolación lineal a lo largo del tiempo (rellena huecos internos).
        2. Relleno hacia adelante y hacia atrás (cubre extremos: primeros/últimos
           años sin dato).

    Returns
    -------
    (df_imputado, reporte)
        df_imputado : DataFrame ancho sin nulos.
        reporte     : DataFrame con el conteo de nulos por indicador ANTES de
                      imputar (útil para el análisis de calidad de datos).
    """
    df = ancho.copy()
    columnas_indicadores = [c for c in df.columns if c != "anio"]

    # Reporte de nulos antes de imputar.
    nulos_antes = df[columnas_indicadores].isna().sum()
    reporte = (
        nulos_antes.rename("nulos")
        .reset_index()
        .rename(columns={"index": "codigo"})
    )
    reporte["nombre"] = reporte["codigo"].map(config.nombre_indicador)
    reporte["pct_nulos"] = (reporte["nulos"] / len(df) * 100).round(1)
    reporte = reporte[["codigo", "nombre", "nulos", "pct_nulos"]]

    # Interpolación temporal + relleno de extremos. Orden cronológico primero.
    df = df.sort_values("anio").reset_index(drop=True)
    df[columnas_indicadores] = (
        df[columnas_indicadores]
        # interpolate(method="linear"): rellena un hueco INTERNO trazando una recta
        # entre el valor anterior y el siguiente (estimación; ver DOC §14.8).
        .interpolate(method="linear", limit_direction="both")
        # ffill (forward fill): copia el último valor conocido HACIA ADELANTE,
        # para cubrir huecos al FINAL de la serie (último año sin dato).
        .ffill()
        # bfill (backward fill): copia el siguiente valor conocido HACIA ATRÁS,
        # para cubrir huecos al INICIO de la serie. (Ver DOC §14.9.)
        .bfill()
    )
    return df, reporte


# ===========================================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# ===========================================================================
def crear_features(ancho: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas a partir de los indicadores base.

    Para cada indicador numérico se generan:
        * <ind>_var      : variación porcentual interanual (year-over-year).
        * <ind>_mm3      : media móvil de 3 años (suaviza la tendencia).
        * <ind>_lag1     : valor del año anterior (rezago de 1 período), útil como
                           predictor en los modelos de regresión.

    Estas características capturan dinámica temporal (aceleración, tendencia,
    memoria) que un modelo no vería usando solo el nivel del indicador.
    """
    df = ancho.sort_values("anio").reset_index(drop=True).copy()
    columnas_indicadores = [c for c in df.columns if c != "anio"]

    nuevas: dict[str, pd.Series] = {}
    for col in columnas_indicadores:
        # pct_change(): variación porcentual respecto a la fila anterior (año
        # anterior). Mide aceleración/desaceleración. ×100 para dejarlo en %.
        # fill_method=None evita un relleno implícito que pandas marca obsoleto.
        nuevas[f"{col}_var"] = df[col].pct_change(fill_method=None) * 100
        # rolling(window=3).mean(): media móvil de 3 años (promedio del año actual
        # y los 2 previos). Suaviza el ruido y deja ver la TENDENCIA.
        # min_periods=1 calcula igual aunque aún no haya 3 años (primeros años).
        nuevas[f"{col}_mm3"] = df[col].rolling(window=3, min_periods=1).mean()
        # shift(1): rezago (lag) de 1 año = el valor del año anterior. Es la
        # "memoria" de la serie y un predictor clave para los modelos.
        nuevas[f"{col}_lag1"] = df[col].shift(1)

    df_features = pd.concat([df, pd.DataFrame(nuevas, index=df.index)], axis=1)

    # La primera fila tendrá NaN en variaciones y lags: se rellenan con 0 (var) y
    # con el propio valor (lag), una convención simple y razonable.
    columnas_var = [c for c in df_features.columns if c.endswith("_var")]
    df_features[columnas_var] = df_features[columnas_var].fillna(0.0)
    for col in columnas_indicadores:
        df_features[f"{col}_lag1"] = df_features[f"{col}_lag1"].fillna(df_features[col])

    return df_features


# ===========================================================================
# 5. RESUMEN DE CALIDAD (para el EDA del notebook)
# ===========================================================================
def resumen_calidad(largo: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un resumen de cobertura por indicador.

    Para cada indicador: número de observaciones, años mínimo y máximo, y la
    fuente. Sirve para documentar la calidad/cobertura de los datos.
    """
    resumen = (
        largo.dropna(subset=["valor"])
        .groupby("codigo")
        .agg(
            observaciones=("valor", "size"),
            anio_min=("anio", "min"),
            anio_max=("anio", "max"),
            fuente=("fuente", "first"),
        )
        .reset_index()
    )
    resumen["nombre"] = resumen["codigo"].map(config.nombre_indicador)
    resumen["unidad"] = resumen["codigo"].map(config.unidad_indicador)
    columnas = ["codigo", "nombre", "fuente", "unidad",
                "observaciones", "anio_min", "anio_max"]
    return resumen[columnas]


# ===========================================================================
# ORQUESTADOR DE PREPROCESAMIENTO
# ===========================================================================
def preprocesar_todo(largo_crudo: pd.DataFrame, guardar: bool = True) -> dict:
    """Ejecuta todo el preprocesamiento y devuelve los datasets resultantes.

    Parameters
    ----------
    largo_crudo : DataFrame largo proveniente de la ingesta.
    guardar     : si True, escribe los CSV en `data/processed/`.

    Returns
    -------
    dict con llaves:
        'largo'    -> DataFrame largo limpio.
        'ancho'    -> DataFrame ancho imputado (una col por indicador).
        'features' -> DataFrame ancho + variables derivadas.
        'reporte_nulos' -> DataFrame con nulos por indicador.
        'calidad'  -> DataFrame de cobertura por indicador.
    """
    print("Preprocesando datos...")
    largo = limpiar(largo_crudo)
    calidad = resumen_calidad(largo)

    ancho = a_formato_ancho(largo)
    ancho_imputado, reporte_nulos = manejar_nulos(ancho)
    features = crear_features(ancho_imputado)

    if guardar:
        largo.to_csv(config.DATASET_LARGO, index=False)
        ancho_imputado.to_csv(config.DATASET_ANCHO, index=False)
        features.to_csv(config.DATASET_FEATURES, index=False)
        print(f"  [ok] Datasets guardados en {config.PROCESSED_DIR}")

    print(
        f"Preprocesamiento completo: {ancho_imputado.shape[0]} años x "
        f"{ancho_imputado.shape[1] - 1} indicadores; "
        f"{features.shape[1] - 1} columnas con features."
    )
    return {
        "largo": largo,
        "ancho": ancho_imputado,
        "features": features,
        "reporte_nulos": reporte_nulos,
        "calidad": calidad,
    }


if __name__ == "__main__":
    from .ingesta import ingestar_todo

    crudo = ingestar_todo()
    resultado = preprocesar_todo(crudo)
    print(resultado["ancho"].tail().to_string(index=False))
