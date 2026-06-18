"""
modelos.py  —  MÓDULO 3: MACHINE LEARNING
==========================================
Aplica técnicas de aprendizaje automático sobre los indicadores económicos de
Panamá. Cubre dos de las técnicas pedidas en el curso:

  A) REGRESIÓN / PRONÓSTICO de series de tiempo
     Predice el valor futuro de al menos 2 indicadores (PIB per cápita e
     inflación, por defecto) usando regresión con características de rezago
     (modelo autorregresivo con tendencia). Incluye evaluación con partición
     temporal y métricas (MAE, RMSE, R²).

  B) CLUSTERING (KMeans)
     Agrupa los años en "regímenes económicos" similares a partir de varios
     indicadores estandarizados, e identifica, por ejemplo, los años de crisis
     frente a los de expansión.

El módulo está pensado para ser llamado tanto desde el notebook como desde el
dashboard. Todas las funciones devuelven estructuras (DataFrames y dicts) fáciles
de graficar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config


# ===========================================================================
# A) REGRESIÓN / PRONÓSTICO
# ===========================================================================
def _crear_modelo(nombre: str):
    """Fábrica de modelos de regresión.

    'linear' -> regresión lineal estandarizada (extrapola con tendencia; ideal
                para pronosticar años futuros).
    'rf'     -> Random Forest (capta relaciones no lineales pero NO extrapola
                fuera del rango visto; se incluye para comparar en la evaluación).
    """
    if nombre == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=42)
    # Por defecto, regresión lineal dentro de un Pipeline con estandarización.
    return Pipeline([
        ("escala", StandardScaler()),
        ("reg", LinearRegression()),
    ])


def _construir_supervisado(
    features: pd.DataFrame, objetivo: str, n_lags: int = 2
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Construye la matriz supervisada (X, y) para un indicador objetivo.

    Predictores:
        * 'anio'             -> captura la tendencia temporal.
        * '<objetivo>_lagk'  -> valores de los k años anteriores (autocorrelación).

    Se descartan las primeras `n_lags` filas, que no tienen historia suficiente.

    Returns
    -------
    (X, y, predictores)
    """
    df = features[["anio", objetivo]].sort_values("anio").reset_index(drop=True).copy()
    predictores = ["anio"]
    for k in range(1, n_lags + 1):
        col = f"{objetivo}_lag{k}"
        df[col] = df[objetivo].shift(k)
        predictores.append(col)

    df = df.dropna().reset_index(drop=True)
    X = df[predictores]
    y = df[objetivo]
    return X, y, predictores


def evaluar_modelo(
    features: pd.DataFrame,
    objetivo: str,
    n_test: int = 5,
    modelo: str = "linear",
    n_lags: int = 2,
) -> dict:
    """Evalúa el modelo con una partición temporal (hold-out de los últimos años).

    En series de tiempo NO se debe usar una partición aleatoria: se entrena con
    el pasado y se prueba con el futuro. Aquí se reservan los últimos `n_test`
    años como conjunto de prueba.

    Returns
    -------
    dict con:
        'objetivo', 'modelo', 'metricas' {MAE, RMSE, R2},
        'anios_test', 'y_test', 'y_pred'  (para graficar).
    """
    X, y, _ = _construir_supervisado(features, objetivo, n_lags)
    anios = X["anio"].to_numpy()

    # Partición temporal: las primeras filas para entrenar, las últimas para test.
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

    est = _crear_modelo(modelo)
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)

    metricas = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(root_mean_squared_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }
    return {
        "objetivo": objetivo,
        "nombre": config.nombre_indicador(objetivo),
        "modelo": modelo,
        "metricas": metricas,
        "anios_test": anios[-n_test:],
        "y_test": y_test.to_numpy(),
        "y_pred": np.asarray(y_pred),
    }


def pronosticar(
    features: pd.DataFrame,
    objetivo: str,
    n_futuro: int | None = None,
    modelo: str = "linear",
    n_lags: int = 2,
) -> pd.DataFrame:
    """Pronostica los próximos `n_futuro` años de un indicador.

    Reentrena el modelo con TODOS los datos disponibles y luego genera el
    pronóstico de forma **recursiva**: la predicción de un año se usa como rezago
    para predecir el siguiente.

    Returns
    -------
    pd.DataFrame con columnas ['anio', objetivo, 'tipo'] donde 'tipo' es
    'histórico' o 'pronóstico'.
    """
    if n_futuro is None:
        n_futuro = config.ANIOS_PRONOSTICO

    X, y, predictores = _construir_supervisado(features, objetivo, n_lags)
    est = _crear_modelo(modelo)
    est.fit(X, y)

    # Serie histórica completa (con los valores reales, incluidos los primeros
    # años que no entraron al supervisado).
    hist = features[["anio", objetivo]].sort_values("anio").reset_index(drop=True)
    valores = hist[objetivo].tolist()
    anios = hist["anio"].tolist()

    filas_pronostico = []
    for paso in range(1, n_futuro + 1):
        anio_nuevo = anios[-1] + 1
        # Construye el vector de predictores con los últimos valores conocidos.
        fila = {"anio": anio_nuevo}
        for k in range(1, n_lags + 1):
            fila[f"{objetivo}_lag{k}"] = valores[-k]
        x_nuevo = pd.DataFrame([fila])[predictores]
        pred = float(est.predict(x_nuevo)[0])

        valores.append(pred)
        anios.append(anio_nuevo)
        filas_pronostico.append({"anio": anio_nuevo, objetivo: pred, "tipo": "pronóstico"})

    historico = hist.assign(tipo="histórico")
    pronostico = pd.DataFrame(filas_pronostico)
    return pd.concat([historico, pronostico], ignore_index=True)


def entrenar_indicadores(
    features: pd.DataFrame,
    objetivos: list[str] | None = None,
    n_test: int = 5,
    n_futuro: int | None = None,
    modelo: str = "linear",
) -> dict:
    """Evalúa y pronostica una lista de indicadores objetivo (≥ 2 por requisito).

    Returns
    -------
    dict { codigo_objetivo: {'evaluacion': ..., 'pronostico': DataFrame} }
    """
    if objetivos is None:
        objetivos = ["pib_per_capita", "inflacion"]

    resultados = {}
    for obj in objetivos:
        evaluacion = evaluar_modelo(features, obj, n_test=n_test, modelo=modelo)
        df_pron = pronosticar(features, obj, n_futuro=n_futuro, modelo=modelo)
        resultados[obj] = {"evaluacion": evaluacion, "pronostico": df_pron}
        m = evaluacion["metricas"]
        print(
            f"  [{config.nombre_indicador(obj)}] "
            f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}"
        )
    return resultados


# ===========================================================================
# B) CLUSTERING DE REGÍMENES ECONÓMICOS (KMeans)
# ===========================================================================
def clustering_regimenes(
    ancho: pd.DataFrame,
    variables: list[str] | None = None,
    n_clusters: int = 3,
) -> dict:
    """Agrupa los años en regímenes económicos mediante KMeans.

    Estandariza las variables seleccionadas (para que todas pesen igual sin
    importar su escala) y aplica KMeans. Devuelve los años etiquetados con su
    clúster, el perfil promedio de cada clúster y el coeficiente de silueta
    (medida de qué tan bien separados están los grupos).

    Returns
    -------
    dict con:
        'asignaciones' -> DataFrame ['anio', <variables>, 'cluster', 'etiqueta'].
        'perfiles'     -> DataFrame con la media de cada variable por clúster.
        'silueta'      -> float (entre -1 y 1; mayor es mejor).
        'variables'    -> lista de variables usadas.
    """
    if variables is None:
        variables = ["pib_crecimiento", "inflacion", "desempleo", "imae"]
    variables = [v for v in variables if v in ancho.columns]

    df = ancho[["anio"] + variables].dropna().reset_index(drop=True).copy()

    X = StandardScaler().fit_transform(df[variables])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    etiquetas = km.fit_predict(X)
    df["cluster"] = etiquetas

    # Coeficiente de silueta (requiere al menos 2 clústeres con datos).
    try:
        silueta = float(silhouette_score(X, etiquetas))
    except ValueError:
        silueta = float("nan")

    # Perfil promedio de cada clúster (en las unidades originales).
    perfiles = df.groupby("cluster")[variables].mean().round(2).reset_index()

    # Etiqueta interpretable según el crecimiento medio del clúster.
    orden = perfiles.sort_values("pib_crecimiento")["cluster"].tolist()
    nombres_regimen = _nombres_regimen(len(orden))
    mapa_etiqueta = {cl: nombres_regimen[i] for i, cl in enumerate(orden)}
    df["etiqueta"] = df["cluster"].map(mapa_etiqueta)
    perfiles["etiqueta"] = perfiles["cluster"].map(mapa_etiqueta)

    print(
        f"  Clustering KMeans: {n_clusters} regímenes, "
        f"coeficiente de silueta = {silueta:.3f}"
    )
    return {
        "asignaciones": df,
        "perfiles": perfiles,
        "silueta": silueta,
        "variables": variables,
    }


def _nombres_regimen(n: int) -> list[str]:
    """Devuelve etiquetas legibles ordenadas de menor a mayor crecimiento."""
    if n <= 2:
        return ["Contracción / crisis", "Expansión"][:n]
    if n == 3:
        return ["Contracción / crisis", "Crecimiento moderado", "Expansión fuerte"]
    return [f"Régimen {i + 1}" for i in range(n)]


if __name__ == "__main__":
    from ..pipeline.ingesta import ingestar_todo
    from ..pipeline.preprocesamiento import preprocesar_todo

    datos = preprocesar_todo(ingestar_todo())
    print("\n== Pronósticos ==")
    res = entrenar_indicadores(datos["features"])
    for obj, r in res.items():
        print(r["pronostico"].tail(4).to_string(index=False))
    print("\n== Clustering ==")
    cl = clustering_regimenes(datos["ancho"])
    print(cl["perfiles"].to_string(index=False))
