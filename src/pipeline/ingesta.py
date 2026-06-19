"""
ingesta.py  —  MÓDULO 1 (parte A): INGESTA DE DATOS
====================================================
Implementa la **ingesta desde al menos 2 fuentes de datos diferentes**, que es
un requisito central del proyecto.

Fuente 1 — Banco Mundial (API REST pública):
    Datos macroeconómicos de Panamá (PIB, inflación, desempleo, IED, etc.)
    obtenidos en vivo desde `https://api.worldbank.org/v2`. No requiere clave.
    Como el Banco Mundial **sí ofrece una API** que devuelve JSON estructurado,
    NO hace falta web scraping: basta con pedir y leer los campos.

Fuente 2 — Contraloría General / INEC / Autoridad del Canal de Panamá:
    Indicadores nacionales (tránsitos e ingresos del Canal, IMAE, deuda pública).
    A diferencia del Banco Mundial, estas instituciones **no exponen una API**: sus
    datos viven en portales y páginas web. Por eso aquí se usan dos técnicas de
    ingesta para esta fuente, en orden de preferencia:
        1. Descarga de un CSV oficial en vivo (si se define `CONTRALORIA_URL`).
        2. **WEB SCRAPING con BeautifulSoup** de una página pública para obtener la
           deuda pública (% del PIB) en vivo.
        3. CSV local con cifras representativas reales como respaldo (offline).

Diseño robusto / reproducible:
    * Si la API del Banco Mundial no responde (sin internet, time-out, etc.), la
      ingesta NO falla: usa un conjunto de datos de respaldo embebido con cifras
      representativas reales. Lo mismo si el web scraping de la Fuente 2 falla.
    * Todo lo descargado/scrapeado se guarda en `data/raw/` como caché.

Salida estándar: un DataFrame en formato **largo (tidy)** con columnas
    ['anio', 'codigo', 'valor', 'fuente']
que el módulo de preprocesamiento luego limpia y transforma.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup  # Librería de WEB SCRAPING para la Fuente 2

from .. import config

# Cabecera HTTP "User-Agent": algunos sitios (como Wikipedia) rechazan peticiones
# sin un identificador de navegador. La declaramos una vez para reutilizarla.
_CABECERAS_HTTP = {"User-Agent": "Mozilla/5.0 (proyecto educativo UTP - Gestion de la Informacion)"}


# ===========================================================================
# DATOS DE RESPALDO (cifras representativas reales de Panamá, 2000-2024)
# ===========================================================================
# Estas series se usan únicamente si la API del Banco Mundial no está disponible.
# Provienen de cifras públicas reales (Banco Mundial, FMI, CEPAL) redondeadas.
# Permiten que el proyecto sea 100% reproducible offline.
_ANIOS_RESPALDO = list(range(2000, 2025))

_RESPALDO_BANCO_MUNDIAL: dict[str, dict[int, float]] = {
    # PIB crecimiento real anual (%)
    "pib_crecimiento": {
        2000: 2.7, 2001: 0.6, 2002: 2.2, 2003: 4.2, 2004: 7.5, 2005: 7.2,
        2006: 8.5, 2007: 12.1, 2008: 8.6, 2009: 1.2, 2010: 5.8, 2011: 11.3,
        2012: 9.8, 2013: 6.9, 2014: 5.1, 2015: 5.7, 2016: 5.0, 2017: 5.6,
        2018: 3.7, 2019: 3.3, 2020: -17.9, 2021: 15.8, 2022: 10.8,
        2023: 7.4, 2024: 2.9,
    },
    # PIB per cápita (USD corrientes)
    "pib_per_capita": {
        2000: 4080, 2001: 4060, 2002: 4150, 2003: 4280, 2004: 4540, 2005: 4920,
        2006: 5430, 2007: 6180, 2008: 7110, 2009: 7280, 2010: 7990, 2011: 9130,
        2012: 10260, 2013: 11280, 2014: 12230, 2015: 13150, 2016: 13680,
        2017: 14490, 2018: 15280, 2019: 15730, 2020: 12880, 2021: 14920,
        2022: 16720, 2023: 18280, 2024: 18680,
    },
    # Inflación, IPC (%)
    "inflacion": {
        2000: 1.4, 2001: 0.3, 2002: 1.0, 2003: 1.4, 2004: 0.5, 2005: 2.9,
        2006: 2.5, 2007: 4.2, 2008: 8.8, 2009: 2.4, 2010: 3.5, 2011: 5.9,
        2012: 5.7, 2013: 4.0, 2014: 2.6, 2015: 0.1, 2016: 0.7, 2017: 0.9,
        2018: 0.8, 2019: -0.4, 2020: -1.6, 2021: 1.6, 2022: 2.9, 2023: 1.5,
        2024: 0.9,
    },
    # Tasa de desempleo (%)
    "desempleo": {
        2000: 13.5, 2001: 14.0, 2002: 13.5, 2003: 13.1, 2004: 11.8, 2005: 9.8,
        2006: 8.7, 2007: 6.4, 2008: 5.0, 2009: 6.6, 2010: 6.5, 2011: 4.5,
        2012: 4.0, 2013: 4.1, 2014: 4.8, 2015: 4.5, 2016: 5.5, 2017: 6.1,
        2018: 6.0, 2019: 7.1, 2020: 18.5, 2021: 11.3, 2022: 9.9, 2023: 7.4,
        2024: 9.5,
    },
    # PIB total (miles de millones USD corrientes)
    "pib_usd": {
        2000: 12.3e9, 2001: 12.4e9, 2002: 12.9e9, 2003: 13.6e9, 2004: 14.7e9,
        2005: 16.2e9, 2006: 18.2e9, 2007: 21.1e9, 2008: 24.9e9, 2009: 26.0e9,
        2010: 29.0e9, 2011: 34.0e9, 2012: 39.0e9, 2013: 44.0e9, 2014: 48.5e9,
        2015: 52.9e9, 2016: 55.9e9, 2017: 60.4e9, 2018: 64.9e9, 2019: 66.8e9,
        2020: 54.0e9, 2021: 64.0e9, 2022: 76.5e9, 2023: 83.4e9, 2024: 87.3e9,
    },
    # Inversión Extranjera Directa, entradas netas (USD)
    "ied": {
        2000: 0.62e9, 2001: 0.51e9, 2002: 0.10e9, 2003: 0.77e9, 2004: 1.01e9,
        2005: 1.03e9, 2006: 2.56e9, 2007: 1.78e9, 2008: 2.40e9, 2009: 1.26e9,
        2010: 2.72e9, 2011: 3.13e9, 2012: 3.25e9, 2013: 4.13e9, 2014: 4.46e9,
        2015: 3.97e9, 2016: 4.65e9, 2017: 4.59e9, 2018: 5.55e9, 2019: 4.46e9,
        2020: 0.59e9, 2021: 3.50e9, 2022: 2.49e9, 2023: 2.10e9, 2024: 2.30e9,
    },
}

# Datos de la Fuente 2 (Contraloría / INEC / Autoridad del Canal de Panamá).
# Valores representativos basados en cifras públicas reales.
_DATOS_CONTRALORIA: dict[str, dict[int, float]] = {
    # Tránsitos de buques de alto calado por el Canal (año fiscal)
    "canal_transitos": {
        2000: 13200, 2001: 12800, 2002: 13000, 2003: 13150, 2004: 14000,
        2005: 14200, 2006: 14150, 2007: 14700, 2008: 14400, 2009: 14300,
        2010: 14200, 2011: 14700, 2012: 14500, 2013: 13700, 2014: 13480,
        2015: 13870, 2016: 13110, 2017: 13550, 2018: 13800, 2019: 13795,
        2020: 13369, 2021: 13342, 2022: 14239, 2023: 14080, 2024: 11240,
    },
    # Ingresos del Canal de Panamá (millones USD)
    "canal_ingresos": {
        2000: 770, 2001: 760, 2002: 790, 2003: 850, 2004: 980, 2005: 1130,
        2006: 1290, 2007: 1460, 2008: 1670, 2009: 1760, 2010: 1880, 2011: 2160,
        2012: 2410, 2013: 2410, 2014: 2630, 2015: 2610, 2016: 2610, 2017: 2710,
        2018: 2730, 2019: 2600, 2020: 2500, 2021: 2900, 2022: 3300, 2023: 3350,
        2024: 4900,
    },
    # Índice Mensual de Actividad Económica, promedio anual (base 2007 = 100)
    "imae": {
        2000: 58, 2001: 59, 2002: 60, 2003: 63, 2004: 68, 2005: 73, 2006: 79,
        2007: 100, 2008: 109, 2009: 110, 2010: 117, 2011: 130, 2012: 143,
        2013: 153, 2014: 161, 2015: 170, 2016: 178, 2017: 188, 2018: 195,
        2019: 201, 2020: 165, 2021: 191, 2022: 212, 2023: 228, 2024: 235,
    },
    # Deuda pública (% del PIB). Respaldo de la columna que normalmente se obtiene
    # por WEB SCRAPING; se usa solo si el scraping falla (sin internet).
    "deuda_pib": {
        2000: 66, 2001: 62, 2002: 66, 2003: 67, 2004: 70, 2005: 66, 2006: 61,
        2007: 53, 2008: 45, 2009: 45, 2010: 43, 2011: 38, 2012: 37, 2013: 37,
        2014: 37, 2015: 38, 2016: 39, 2017: 38, 2018: 40, 2019: 46, 2020: 66,
        2021: 64, 2022: 57, 2023: 55, 2024: 58,
    },
}


# ===========================================================================
# FUENTE 1 — BANCO MUNDIAL (API REST)
# ===========================================================================
def descargar_indicador_banco_mundial(wb_code: str) -> pd.DataFrame | None:
    """Descarga **un** indicador de Panamá desde la API del Banco Mundial.

    Construye la URL del endpoint del Banco Mundial para el país (Panamá, PAN) y
    el indicador `wb_code`, hace la petición HTTP y parsea la respuesta JSON.

    La respuesta de la API es una lista de dos elementos:
        [ metadatos_de_paginación , lista_de_observaciones ]
    Cada observación es un dict con (entre otros) las claves 'date' (año) y
    'value' (valor del indicador, puede ser None).

    Returns
    -------
    pd.DataFrame | None
        DataFrame con columnas ['anio', 'valor'] si la descarga fue exitosa, o
        None si hubo cualquier error (de red o de formato).
    """
    url = (
        f"{config.WB_BASE_URL}/country/{config.PAIS_ISO3}"
        f"/indicator/{wb_code}"
    )
    params = {
        "format": config.WB_FORMATO,
        "per_page": config.WB_PER_PAGE,
        "date": f"{config.ANIO_INICIO}:{config.ANIO_FIN}",
    }
    try:
        respuesta = requests.get(url, params=params, timeout=config.WB_TIMEOUT)
        respuesta.raise_for_status()           # Lanza excepción si HTTP >= 400
        contenido = respuesta.json()
    except (requests.RequestException, ValueError):
        # ValueError cubre el caso de JSON inválido.
        return None

    # La respuesta válida es una lista [meta, datos]; si no, abortamos.
    if not isinstance(contenido, list) or len(contenido) < 2 or contenido[1] is None:
        return None

    observaciones = contenido[1]
    filas = [
        {"anio": int(obs["date"]), "valor": obs["value"]}
        for obs in observaciones
        if obs.get("date") is not None
    ]
    if not filas:
        return None

    df = pd.DataFrame(filas).sort_values("anio").reset_index(drop=True)
    return df


def descargar_banco_mundial(usar_respaldo_si_falla: bool = True) -> pd.DataFrame:
    """Descarga **todos** los indicadores del Banco Mundial definidos en config.

    Itera sobre `config.INDICADORES_BANCO_MUNDIAL` y descarga cada uno. Si al
    menos un indicador no se puede descargar (o todo falla por falta de
    internet) y `usar_respaldo_si_falla=True`, completa con los datos de respaldo
    embebidos.

    Returns
    -------
    pd.DataFrame
        Formato largo: columnas ['anio', 'codigo', 'valor', 'fuente'].
    """
    bloques: list[pd.DataFrame] = []
    hubo_fallo = False

    for indicador in config.INDICADORES_BANCO_MUNDIAL:
        codigo = indicador["codigo"]
        df_ind = descargar_indicador_banco_mundial(indicador["wb_code"])

        if df_ind is None:
            hubo_fallo = True
            if usar_respaldo_si_falla:
                df_ind = _respaldo_a_dataframe(codigo)
            else:
                continue

        df_ind = df_ind.assign(codigo=codigo, fuente=indicador["fuente"])
        bloques.append(df_ind)

    if not bloques:
        # Caso extremo: todo falló y no se permitió respaldo.
        raise RuntimeError(
            "No se pudo descargar ningún indicador del Banco Mundial y el "
            "respaldo está desactivado."
        )

    largo = pd.concat(bloques, ignore_index=True)
    largo = largo[["anio", "codigo", "valor", "fuente"]]

    if hubo_fallo:
        print(
            "  [aviso] La API del Banco Mundial no respondió para uno o más "
            "indicadores; se usaron datos de respaldo representativos."
        )
    else:
        print("  [ok] Datos del Banco Mundial descargados en vivo desde la API.")

    # Guardamos una copia en caché (data/raw/).
    largo.to_csv(config.BANCO_MUNDIAL_CACHE, index=False)
    return largo


def _respaldo_a_dataframe(codigo: str) -> pd.DataFrame:
    """Convierte una serie de respaldo (dict año->valor) en DataFrame largo."""
    serie = _RESPALDO_BANCO_MUNDIAL[codigo]
    return pd.DataFrame({"anio": list(serie.keys()), "valor": list(serie.values())})


# ===========================================================================
# FUENTE 2 — CONTRALORÍA / INEC / CANAL (CSV en vivo + WEB SCRAPING + respaldo)
# ===========================================================================
# La Fuente 2 combina tres técnicas de obtención de datos, en orden de preferencia:
#   1. Descarga de un CSV oficial en vivo  -> _descargar_contraloria_url()
#   2. WEB SCRAPING con BeautifulSoup       -> scrapear_economia_panama()
#   3. CSV local representativo (respaldo)  -> generar_csv_contraloria()
# La orquestación está en cargar_contraloria().
def generar_csv_contraloria(forzar: bool = False) -> None:
    """Crea el archivo CSV de la Fuente 2 en `data/raw/` si no existe.

    En un proyecto real, este archivo se descargaría manualmente desde el portal
    de datos abiertos de la Contraloría / INEC. Aquí lo generamos a partir de
    cifras públicas representativas para que el proyecto sea autocontenido y
    reproducible. El CSV resultante tiene una columna 'anio' y una columna por
    cada indicador de la Fuente 2.

    Parameters
    ----------
    forzar : bool
        Si es True, regenera el archivo aunque ya exista.
    """
    # Regenera el CSV si no existe, si se fuerza, o si al CSV existente le faltan
    # columnas esperadas (p. ej. tras agregar un indicador nuevo como deuda_pib).
    columnas_esperadas = {"anio", *_DATOS_CONTRALORIA.keys()}
    if config.CONTRALORIA_CSV.exists() and not forzar:
        try:
            existentes = set(pd.read_csv(config.CONTRALORIA_CSV, nrows=0).columns)
            if columnas_esperadas.issubset(existentes):
                return  # El CSV ya está completo.
        except Exception:  # noqa: BLE001 - si no se puede leer, se regenera
            pass

    # Construye un DataFrame ancho: una fila por año, una columna por indicador.
    df = pd.DataFrame({"anio": _ANIOS_RESPALDO})
    for codigo, serie in _DATOS_CONTRALORIA.items():
        df[codigo] = df["anio"].map(serie)

    df.to_csv(config.CONTRALORIA_CSV, index=False)


def _descargar_contraloria_url(url: str) -> pd.DataFrame | None:
    """Intenta descargar EN VIVO el CSV de la Fuente 2 desde una URL.

    Espera un CSV con una columna 'anio' y una columna por indicador. Devuelve el
    DataFrame ancho si la descarga y el formato son válidos, o None si falla.
    """
    try:
        respuesta = requests.get(url, timeout=config.WB_TIMEOUT)
        respuesta.raise_for_status()
        df = pd.read_csv(io.StringIO(respuesta.text))
    except (requests.RequestException, ValueError):
        return None
    if "anio" not in df.columns or df.shape[1] < 2:
        return None
    return df


def scrapear_economia_panama(url: str | None = None) -> pd.DataFrame | None:
    """Obtiene la **deuda pública (% del PIB)** por WEB SCRAPING con BeautifulSoup.

    El Banco Mundial tiene API, pero los datos fiscales de Panamá no: viven en
    páginas web (HTML). El web scraping consiste en **descargar el HTML de una
    página y extraer de él los datos** que nos interesan. Aquí raspamos una tabla
    pública con la serie anual de la deuda pública de Panamá.

    Pasos del scraping:
      1. `requests.get(url)` descarga el HTML de la página (con un User-Agent para
         que el sitio no rechace la petición).
      2. `BeautifulSoup(html, "html.parser")` convierte ese HTML en un árbol
         navegable. Usamos el parser `html.parser` que viene incluido en Python
         (no requiere instalar nada extra).
      3. Buscamos la tabla (`<table class="wikitable">`) y, dentro de ella, la
         columna cuyo encabezado contiene "debt"/"deuda".
      4. Recorremos las filas (`<tr>`), tomamos la primera celda como año y la
         celda de la deuda; limpiamos el texto (quitamos "%", comas, "n/a").

    Returns
    -------
    pd.DataFrame | None
        DataFrame ['anio', 'deuda_pib'] si el scraping funcionó, o None si falló
        (sin internet, página caída o estructura inesperada). Nunca lanza error.
    """
    url = url or config.CONTRALORIA_SCRAPE_URL
    try:
        respuesta = requests.get(url, headers=_CABECERAS_HTTP, timeout=config.WB_TIMEOUT)
        respuesta.raise_for_status()
    except requests.RequestException:
        return None

    sopa = BeautifulSoup(respuesta.text, "html.parser")
    tabla = sopa.find("table", class_="wikitable")
    if tabla is None:
        return None

    filas = tabla.find_all("tr")
    if not filas:
        return None

    # Localiza el índice de la columna de deuda leyendo los encabezados (<th>).
    encabezados = [c.get_text(" ", strip=True).lower() for c in filas[0].find_all(["th", "td"])]
    idx_deuda = next(
        (i for i, h in enumerate(encabezados) if "debt" in h or "deuda" in h), None
    )
    if idx_deuda is None:
        return None

    registros = []
    for fila in filas[1:]:
        celdas = [c.get_text(" ", strip=True) for c in fila.find_all(["th", "td"])]
        # La primera celda debe ser un año de 4 dígitos; si no, saltamos la fila.
        if not celdas or not re.match(r"^(19|20)\d{2}$", celdas[0]):
            continue
        if idx_deuda >= len(celdas):
            continue
        anio = int(celdas[0])
        # Limpia el texto del valor: "66.3%" -> 66.3 ; "n/a" -> None.
        bruto = celdas[idx_deuda].replace("%", "").replace(",", "").strip()
        try:
            valor = float(bruto)
        except ValueError:
            valor = None
        registros.append({"anio": anio, "deuda_pib": valor})

    df = pd.DataFrame(registros).dropna(subset=["deuda_pib"])
    # Nos quedamos solo con el rango de años del estudio.
    df = df[(df["anio"] >= config.ANIO_INICIO) & (df["anio"] <= config.ANIO_FIN)]
    return df if not df.empty else None


def _fusionar_columna(ancho: pd.DataFrame, nuevo: pd.DataFrame, col: str) -> pd.DataFrame:
    """Reemplaza la columna `col` de `ancho` con los valores reales de `nuevo`.

    Donde `nuevo` (lo scrapeado) tiene dato para un año, ese valor sustituye al de
    respaldo; donde no, se conserva el respaldo local. Es un "actualizar si existe".
    """
    if col not in ancho.columns:        # por si la base aún no trae esa columna
        ancho = ancho.copy()
        ancho[col] = pd.NA
    fusion = ancho.merge(
        nuevo.rename(columns={col: f"{col}__nuevo"}), on="anio", how="left"
    )
    # combine_first: usa el valor scrapeado si hay; si no, el de respaldo.
    fusion[col] = fusion[f"{col}__nuevo"].combine_first(fusion[col])
    return fusion.drop(columns=[f"{col}__nuevo"])


def cargar_contraloria() -> pd.DataFrame:
    """Carga la Fuente 2 (Contraloría/INEC/ACP) y la devuelve en formato largo.

    Estrategia de ingesta con respaldo (de mayor a menor preferencia):
      1. Si `config.CONTRALORIA_URL` está definida, descarga un CSV oficial EN
         VIVO desde esa URL (y se usa tal cual).
      2. Si no, parte del CSV local representativo y luego intenta **mejorar la
         columna `deuda_pib` por WEB SCRAPING** (BeautifulSoup) con datos reales.
      3. Si el scraping falla, `deuda_pib` queda con su valor de respaldo local.

    Así el pipeline nunca se rompe y, cuando hay internet, incorpora datos reales.

    Returns
    -------
    pd.DataFrame
        Formato largo: columnas ['anio', 'codigo', 'valor', 'fuente'].
    """
    # --- Camino 1: CSV oficial en vivo (si se configuró una URL) ---
    if config.CONTRALORIA_URL:
        ancho_url = _descargar_contraloria_url(config.CONTRALORIA_URL)
        if ancho_url is not None:
            ancho_url.to_csv(config.CONTRALORIA_CSV, index=False)
            print("  [ok] Contraloría/INEC: CSV descargado en vivo desde CONTRALORIA_URL.")
            return _contraloria_a_largo(ancho_url)
        print("  [aviso] No se pudo descargar CONTRALORIA_URL; uso CSV local + scraping.")

    # --- Camino 2: CSV local representativo como base ---
    generar_csv_contraloria()
    ancho = pd.read_csv(config.CONTRALORIA_CSV)

    # --- Camino 2b: WEB SCRAPING para actualizar la deuda pública con datos reales ---
    deuda_scrapeada = scrapear_economia_panama()
    if deuda_scrapeada is not None:
        ancho = _fusionar_columna(ancho, deuda_scrapeada, "deuda_pib")
        ancho.to_csv(config.CONTRALORIA_CSV, index=False)  # actualiza el caché
        print(
            f"  [ok] Contraloría/INEC: CSV local + 'deuda_pib' por WEB SCRAPING "
            f"({len(deuda_scrapeada)} años reales)."
        )
    else:
        print("  [aviso] Web scraping no disponible; 'deuda_pib' usa respaldo local.")

    return _contraloria_a_largo(ancho)


def _contraloria_a_largo(ancho: pd.DataFrame) -> pd.DataFrame:
    """Convierte el DataFrame ancho de la Fuente 2 a formato largo (tidy).

    `melt` "derrite" las columnas de indicadores en filas, y luego se etiqueta la
    fuente de cada indicador leyendo el catálogo de `config`.
    """
    largo = ancho.melt(id_vars="anio", var_name="codigo", value_name="valor")
    largo["fuente"] = largo["codigo"].map(
        lambda c: config.META_POR_CODIGO.get(c, {}).get("fuente", "Contraloría / INEC")
    )
    return largo[["anio", "codigo", "valor", "fuente"]]


# ===========================================================================
# ORQUESTADOR DE INGESTA
# ===========================================================================
def ingestar_todo(usar_respaldo_si_falla: bool = True) -> pd.DataFrame:
    """Ejecuta la ingesta completa de las DOS fuentes y las une.

    Es la función principal del módulo: la llama el notebook y el dashboard.

    1. Descarga la Fuente 1 (Banco Mundial).
    2. Carga la Fuente 2 (Contraloría / INEC).
    3. Concatena ambas en un único DataFrame en formato largo.

    Returns
    -------
    pd.DataFrame
        Datos crudos combinados: columnas ['anio', 'codigo', 'valor', 'fuente'].
    """
    print("Iniciando ingesta de datos desde 2 fuentes...")
    fuente1 = descargar_banco_mundial(usar_respaldo_si_falla)
    fuente2 = cargar_contraloria()

    combinado = pd.concat([fuente1, fuente2], ignore_index=True)
    print(
        f"Ingesta completa: {len(combinado)} observaciones, "
        f"{combinado['codigo'].nunique()} indicadores, "
        f"{combinado['anio'].nunique()} años."
    )
    return combinado


if __name__ == "__main__":
    datos = ingestar_todo()
    print(datos.head(12).to_string(index=False))
