"""
config.py
=========
Configuración central del proyecto **Dashboard de Indicadores Económicos de
Panamá con IA**.

Aquí se definen, en un único lugar, todos los parámetros que el resto del
proyecto (pipeline, modelos de Machine Learning, chatbot RAG y dashboard)
necesita compartir:

* Rutas de carpetas (raw / processed / data).
* El catálogo de indicadores económicos con su código, nombre legible,
  fuente, unidad y código de la API del Banco Mundial.
* Parámetros de la API del Banco Mundial.
* El modelo de Claude usado por el chatbot.

Centralizar esta información evita el "número mágico" disperso por el código y
hace que el proyecto sea fácil de mantener: si cambia un código de indicador,
se cambia en un solo sitio.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. RUTAS DEL PROYECTO
# ---------------------------------------------------------------------------
# BASE_DIR apunta a la raíz del proyecto (la carpeta que contiene src/, data/,
# dashboard/, etc.). Se calcula de forma relativa a este archivo para que el
# código funcione sin importar desde dónde se ejecute.
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# Archivos que produce / consume el pipeline.
CONTRALORIA_CSV: Path = RAW_DIR / "contraloria_panama.csv"      # Fuente 2 (CSV)
BANCO_MUNDIAL_CACHE: Path = RAW_DIR / "banco_mundial_panama.csv"  # cache Fuente 1
DATASET_LARGO: Path = PROCESSED_DIR / "indicadores_largo.csv"    # formato tidy
DATASET_ANCHO: Path = PROCESSED_DIR / "indicadores_ancho.csv"    # formato wide
DATASET_FEATURES: Path = PROCESSED_DIR / "indicadores_features.csv"  # con features

# Asegura que las carpetas existan al importar el módulo (idempotente).
for _carpeta in (RAW_DIR, PROCESSED_DIR):
    _carpeta.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. PARÁMETROS DE PAÍS Y RANGO TEMPORAL
# ---------------------------------------------------------------------------
PAIS_NOMBRE: str = "Panamá"
PAIS_ISO3: str = "PAN"        # Código ISO-3166 alfa-3 usado por el Banco Mundial

# Año inicial del estudio. Se usa 2000 porque la cobertura de datos del Banco
# Mundial para Panamá es completa y consistente desde entonces, y 25 años dan
# suficiente historia para detectar tendencias y entrenar los modelos.
ANIO_INICIO: int = 2000

# Año final = AÑO ACTUAL (dinámico). Antes estaba fijo en 2024; ahora se calcula
# con la fecha del sistema para pedir SIEMPRE los datos más recientes posibles.
# Importante: el año final EFECTIVO de los datos depende de la disponibilidad en
# la fuente. El Banco Mundial publica los indicadores con 1-2 años de rezago, así
# que el último año con dato real suele ser el año actual menos 1 o 2.
ANIO_FIN: int = datetime.now().year

ANIOS_PRONOSTICO: int = 3      # Cuántos años hacia el futuro predicen los modelos


# ---------------------------------------------------------------------------
# 3. CATÁLOGO DE INDICADORES
# ---------------------------------------------------------------------------
# Cada indicador se describe con un diccionario. El "codigo" es el identificador
# interno (corto, en español, sin espacios) que se usa como nombre de columna en
# los DataFrames. "wb_code" es el código oficial del Banco Mundial usado en la
# API. Los indicadores cuyo "wb_code" es None provienen de la Fuente 2
# (Contraloría / INEC / Autoridad del Canal de Panamá).
#
# Estructura de cada entrada:
#   codigo   -> str   : nombre de columna interno
#   nombre   -> str   : nombre legible para gráficas y reportes
#   fuente   -> str   : origen de los datos
#   unidad   -> str   : unidad de medida
#   wb_code  -> str|None : código de la API del Banco Mundial (None si es Fuente 2)
#   descripcion -> str : explicación usada por el chatbot RAG
INDICADORES: list[dict] = [
    # ---- FUENTE 1: Banco Mundial (API REST) ----------------------------
    {
        "codigo": "pib_crecimiento",
        "nombre": "PIB - Crecimiento anual",
        "fuente": "Banco Mundial",
        "unidad": "%",
        "wb_code": "NY.GDP.MKTP.KD.ZG",
        "descripcion": (
            "Tasa de crecimiento porcentual anual del Producto Interno Bruto "
            "real de Panamá. Mide qué tan rápido se expande o contrae la "
            "economía. Panamá tuvo una caída histórica de -17.9% en 2020 por "
            "la pandemia de COVID-19 y un fuerte rebote en 2021."
        ),
    },
    {
        "codigo": "pib_per_capita",
        "nombre": "PIB per cápita",
        "fuente": "Banco Mundial",
        "unidad": "USD",
        "wb_code": "NY.GDP.PCAP.CD",
        "descripcion": (
            "Producto Interno Bruto dividido entre la población total, en "
            "dólares estadounidenses corrientes. Es una medida aproximada del "
            "ingreso o nivel de vida promedio por habitante."
        ),
    },
    {
        "codigo": "inflacion",
        "nombre": "Inflación (IPC)",
        "fuente": "Banco Mundial",
        "unidad": "%",
        "wb_code": "FP.CPI.TOTL.ZG",
        "descripcion": (
            "Variación porcentual anual del Índice de Precios al Consumidor. "
            "Mide cuánto suben (o bajan) en promedio los precios. Panamá usa el "
            "dólar, por lo que su inflación suele ser baja; en 2020 hubo "
            "deflación de -1.6%."
        ),
    },
    {
        "codigo": "desempleo",
        "nombre": "Tasa de desempleo",
        "fuente": "Banco Mundial",
        "unidad": "%",
        "wb_code": "SL.UEM.TOTL.ZS",
        "descripcion": (
            "Porcentaje de la fuerza laboral que está sin empleo pero buscando "
            "trabajo (estimación modelada OIT). Subió de 7.1% en 2019 a cerca "
            "de 18% en 2020 por la pandemia."
        ),
    },
    {
        "codigo": "pib_usd",
        "nombre": "PIB total",
        "fuente": "Banco Mundial",
        "unidad": "USD",
        "wb_code": "NY.GDP.MKTP.CD",
        "descripcion": (
            "Producto Interno Bruto total de Panamá en dólares corrientes. Es "
            "el valor de todos los bienes y servicios finales producidos en un "
            "año."
        ),
    },
    {
        "codigo": "ied",
        "nombre": "Inversión Extranjera Directa",
        "fuente": "Banco Mundial",
        "unidad": "USD",
        "wb_code": "BX.KLT.DINV.CD.WD",
        "descripcion": (
            "Entradas netas de Inversión Extranjera Directa (IED) en dólares. "
            "Refleja la confianza de empresas extranjeras para invertir en "
            "Panamá."
        ),
    },
    # ---- FUENTE 2: Contraloría / INEC / Autoridad del Canal -------------
    {
        "codigo": "canal_transitos",
        "nombre": "Tránsitos por el Canal de Panamá",
        "fuente": "Contraloría / ACP",
        "unidad": "buques/año",
        "wb_code": None,
        "descripcion": (
            "Cantidad de buques de alto calado que transitaron el Canal de "
            "Panamá en el año fiscal. Es un termómetro del comercio mundial y "
            "una fuente clave de ingresos para el Estado panameño. La sequía de "
            "2023-2024 redujo los tránsitos."
        ),
    },
    {
        "codigo": "canal_ingresos",
        "nombre": "Ingresos del Canal de Panamá",
        "fuente": "Contraloría / ACP",
        "unidad": "millones USD",
        "wb_code": None,
        "descripcion": (
            "Ingresos por peajes y servicios del Canal de Panamá, en millones "
            "de dólares. El Canal aporta cada año al Tesoro Nacional."
        ),
    },
    {
        "codigo": "imae",
        "nombre": "Índice Mensual de Actividad Económica (promedio)",
        "fuente": "Contraloría / INEC",
        "unidad": "índice 2007=100",
        "wb_code": None,
        "descripcion": (
            "Promedio anual del Índice Mensual de Actividad Económica (IMAE) "
            "que publica la Contraloría. Es un indicador adelantado del ritmo "
            "de la economía panameña, con base 2007 = 100."
        ),
    },
]

# Vistas auxiliares derivadas del catálogo, para acceso rápido en el resto del
# código.
CODIGOS_INDICADORES: list[str] = [ind["codigo"] for ind in INDICADORES]
INDICADORES_BANCO_MUNDIAL: list[dict] = [i for i in INDICADORES if i["wb_code"]]
INDICADORES_CONTRALORIA: list[dict] = [i for i in INDICADORES if not i["wb_code"]]
# Diccionario código -> metadatos, útil para buscar nombre/unidad de un indicador.
META_POR_CODIGO: dict[str, dict] = {ind["codigo"]: ind for ind in INDICADORES}


def nombre_indicador(codigo: str) -> str:
    """Devuelve el nombre legible de un indicador a partir de su código."""
    return META_POR_CODIGO.get(codigo, {}).get("nombre", codigo)


def unidad_indicador(codigo: str) -> str:
    """Devuelve la unidad de medida de un indicador a partir de su código."""
    return META_POR_CODIGO.get(codigo, {}).get("unidad", "")


# ---------------------------------------------------------------------------
# 4. PARÁMETROS DE LA API DEL BANCO MUNDIAL
# ---------------------------------------------------------------------------
# La API pública del Banco Mundial no requiere autenticación. La URL tiene la
# forma:
#   https://api.worldbank.org/v2/country/{ISO3}/indicator/{COD}?format=json&...
WB_BASE_URL: str = "https://api.worldbank.org/v2"
WB_FORMATO: str = "json"
WB_PER_PAGE: int = 500        # Suficiente para traer todos los años en una página
WB_TIMEOUT: int = 20          # Segundos antes de abortar una petición HTTP

# URL para descargar la Fuente 2 (Contraloría/INEC/ACP) EN TIEMPO REAL. Si está
# definida (por variable de entorno CONTRALORIA_URL), el pipeline intenta bajar un
# CSV en vivo desde ahí; si está vacía o la descarga falla, usa el CSV local
# representativo como respaldo. El CSV remoto debe tener una columna 'anio' y una
# columna por indicador (canal_transitos, canal_ingresos, imae).
CONTRALORIA_URL: str = os.environ.get("CONTRALORIA_URL", "")


# ---------------------------------------------------------------------------
# 5. PARÁMETROS DEL CHATBOT RAG (Claude / Anthropic)
# ---------------------------------------------------------------------------
# Modelo de Claude usado para la generación de respuestas. Se usa el modelo
# Opus 4.8 más reciente por defecto. El chatbot funciona también sin clave de
# API (modo extractivo de respaldo), por lo que el proyecto es 100% reproducible
# sin costo.
MODELO_CLAUDE: str = "claude-opus-4-8"
MAX_TOKENS_RESPUESTA: int = 1024
RAG_TOP_K: int = 5            # Número de fragmentos recuperados por consulta

# ---------------------------------------------------------------------------
# 6. PROVEEDOR DE GENERACIÓN DEL CHATBOT (Ollama gratis / Claude / extractivo)
# ---------------------------------------------------------------------------
# El chatbot puede generar la respuesta con distintos motores. Para tener un RAG
# "gratis y de calidad" se usa OLLAMA: un modelo de lenguaje que corre LOCALMENTE
# en tu computadora, sin clave de API, sin costo y sin límites.
#
# PROVEEDOR_LLM controla qué motor se usa (se puede fijar por variable de entorno
# CHATBOT_PROVEEDOR):
#   "auto"       -> usa Ollama si está corriendo (gratis); si no, Claude (si hay
#                   ANTHROPIC_API_KEY); si no, el modo extractivo. (Recomendado.)
#   "ollama"     -> fuerza Ollama (gratis, local).
#   "claude"     -> fuerza Claude (requiere ANTHROPIC_API_KEY).
#   "extractivo" -> sin modelo de lenguaje (siempre funciona).
PROVEEDOR_LLM: str = os.environ.get("CHATBOT_PROVEEDOR", "auto")

# Parámetros de Ollama (servidor local de modelos de lenguaje gratuitos).
# Modelos recomendados (ligeros y buenos en español): "llama3.2", "qwen2.5:3b".
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODELO: str = os.environ.get("OLLAMA_MODELO", "llama3.2")
OLLAMA_TIMEOUT: int = 120     # Segundos máximos por respuesta del modelo local


if __name__ == "__main__":
    # Pequeña prueba manual: imprime el catálogo al ejecutar `python src/config.py`
    print(f"Proyecto: Indicadores económicos de {PAIS_NOMBRE} ({PAIS_ISO3})")
    print(f"Rango temporal: {ANIO_INICIO}-{ANIO_FIN}")
    print(f"Total de indicadores: {len(INDICADORES)}")
    print(f"  - Banco Mundial (API): {len(INDICADORES_BANCO_MUNDIAL)}")
    print(f"  - Contraloría/INEC (CSV): {len(INDICADORES_CONTRALORIA)}")
    print(f"Modelo de IA para el chatbot: {MODELO_CLAUDE}")
