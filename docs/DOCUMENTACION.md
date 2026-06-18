# 📘 Documentación Detallada del Proyecto
## Dashboard de Indicadores Económicos de Panamá con IA

**Universidad Tecnológica de Panamá** · Facultad de Ingeniería de Sistemas Computacionales
**Gestión de la Información** — I Semestre 2026 · **Proyecto Integrador — Segundo Parcial**

---

Este documento explica **en detalle y línea por línea** todo lo implementado en el
notebook [`notebooks/proyecto_indicadores_economicos.ipynb`](../notebooks/proyecto_indicadores_economicos.ipynb)
y en los módulos de Python que lo soportan.

---

## Índice

1. [Visión general y arquitectura](#1-visión-general-y-arquitectura)
2. [Mapa del repositorio](#2-mapa-del-repositorio)
3. [Cómo ejecutar el proyecto](#3-cómo-ejecutar-el-proyecto)
4. [El módulo de configuración (`config.py`)](#4-el-módulo-de-configuración-configpy)
5. [Módulo 1 — Pipeline de datos](#5-módulo-1--pipeline-de-datos)
   - 5.1 [Ingesta (`ingesta.py`)](#51-ingesta-ingestapy)
   - 5.2 [Preprocesamiento (`preprocesamiento.py`)](#52-preprocesamiento-preprocesamientopy)
6. [Módulo 2 — Análisis exploratorio y visualización](#6-módulo-2--análisis-exploratorio-y-visualización)
7. [Módulo 3 — Machine Learning (`modelos.py`)](#7-módulo-3--machine-learning-modelospy)
8. [Módulo 4 — Dashboard interactivo (`app.py`)](#8-módulo-4--dashboard-interactivo-apppy)
9. [Módulo 5 — Chatbot con RAG (`chatbot.py`)](#9-módulo-5--chatbot-con-rag-chatbotpy)
10. [Recorrido celda por celda del notebook](#10-recorrido-celda-por-celda-del-notebook)
11. [Decisiones de diseño y justificación](#11-decisiones-de-diseño-y-justificación)
12. [Mapeo con la rúbrica de evaluación](#12-mapeo-con-la-rúbrica-de-evaluación)
13. [Glosario de términos](#13-glosario-de-términos)

---

## 1. Visión general y arquitectura

El proyecto es un **sistema de gestión de información** que toma datos económicos
públicos de Panamá, los procesa mediante un *pipeline*, les aplica *Machine
Learning*, y los expone en un *dashboard* y a través de un *chatbot* con IA.

La arquitectura sigue el flujo clásico de un proyecto de datos:

```
   FUENTES               PIPELINE                 INTELIGENCIA            PRESENTACIÓN
┌────────────┐      ┌──────────────────┐      ┌──────────────────┐    ┌──────────────┐
│ Banco      │─API─▶│ Ingesta          │      │ ML: regresión    │    │ Notebook     │
│ Mundial    │      │   ↓              │─────▶│  (pronóstico)    │───▶│ (.ipynb)     │
├────────────┤      │ Limpieza         │      │ ML: clustering   │    ├──────────────┤
│ Contraloría│─CSV─▶│   ↓              │      │  (KMeans)        │    │ Dashboard    │
│ INEC / ACP │      │ Transformación   │      ├──────────────────┤    │ (Streamlit)  │
└────────────┘      │   ↓              │      │ Chatbot RAG      │    ├──────────────┤
                    │ Features         │      │ (TF-IDF + Claude)│───▶│ Chatbot web  │
                    └──────────────────┘      └──────────────────┘    └──────────────┘
```

**Principio de diseño clave:** toda la lógica vive en el paquete `src/` (módulos
reutilizables). El **notebook** y el **dashboard** son dos "clientes" que
consumen esos mismos módulos. Así no se duplica código y el comportamiento es
idéntico en ambos.

**Lenguaje:** Python. **Librerías principales:** `pandas`, `numpy`,
`scikit-learn`, `matplotlib`, `plotly`, `streamlit`, `requests`, `anthropic`.

---

## 2. Mapa del repositorio

```
parcial2/
├── README.md                     # Presentación e instrucciones rápidas
├── requirements.txt              # Dependencias de Python
├── .gitignore                    # Archivos que Git debe ignorar
├── .streamlit/
│   └── config.toml               # Tema visual del dashboard
│
├── notebooks/
│   └── proyecto_indicadores_economicos.ipynb   # ⭐ Notebook principal
│
├── src/                          # Código fuente (paquete reutilizable)
│   ├── config.py                 # Catálogo de indicadores y parámetros
│   ├── pipeline/
│   │   ├── ingesta.py            # Módulo 1A: ingesta de 2 fuentes
│   │   └── preprocesamiento.py   # Módulo 1B: limpieza y transformación
│   ├── ml/
│   │   └── modelos.py            # Módulo 3: regresión + clustering
│   └── rag/
│       └── chatbot.py            # Módulo 5: chatbot con RAG
│
├── dashboard/
│   └── app.py                    # Módulo 4: dashboard Streamlit
│
├── data/
│   ├── raw/                      # Datos crudos (descargas y CSV de la Contraloría)
│   └── processed/                # Datasets limpios que produce el pipeline
│
└── docs/
    └── DOCUMENTACION.md          # ESTE documento
```

---

## 3. Cómo ejecutar el proyecto

### 3.1 Instalación

```bash
# 1) (Opcional pero recomendado) crear un entorno virtual
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

# 2) Instalar las dependencias
pip install -r requirements.txt
```

### 3.2 Ejecutar el notebook

```bash
jupyter notebook notebooks/proyecto_indicadores_economicos.ipynb
# (o abrirlo en VS Code y ejecutar todas las celdas)
```

### 3.3 Ejecutar el dashboard

```bash
streamlit run dashboard/app.py
# Se abre en el navegador, normalmente en http://localhost:8501
```

### 3.4 (Opcional) Activar el chatbot con Claude

El chatbot funciona **sin clave** (modo extractivo). Para que use el modelo de IA
Claude y redacte respuestas más naturales:

```bash
export ANTHROPIC_API_KEY="tu-clave-de-anthropic"   # Windows: set ANTHROPIC_API_KEY=...
```

> **Reproducibilidad sin internet.** Si la API del Banco Mundial no está
> disponible, el pipeline usa automáticamente datos de respaldo representativos,
> por lo que el notebook y el dashboard **siempre se ejecutan**.

---

## 4. El módulo de configuración (`config.py`)

Este módulo es la "fuente única de verdad" del proyecto. Centraliza todo lo que
los demás módulos necesitan compartir, evitando valores dispersos ("números
mágicos").

### 4.1 Rutas del proyecto

```python
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR  = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
```

- `Path(__file__)` es la ruta de `config.py`. `.resolve().parent.parent` sube dos
  niveles (de `src/config.py` a la raíz del proyecto). Así las rutas funcionan sin
  importar **desde dónde** se ejecute el código.
- Al importar el módulo se crean las carpetas `raw/` y `processed/` si no existen
  (operación *idempotente*: no falla si ya existen).

### 4.2 Catálogo de indicadores (`INDICADORES`)

Es una **lista de diccionarios**; cada uno describe un indicador con:

| Campo | Significado | Ejemplo |
|-------|-------------|---------|
| `codigo` | Nombre interno (columna en los DataFrames) | `"pib_crecimiento"` |
| `nombre` | Nombre legible para gráficas | `"PIB - Crecimiento anual"` |
| `fuente` | De dónde provienen los datos | `"Banco Mundial"` |
| `unidad` | Unidad de medida | `"%"` |
| `wb_code` | Código de la API del Banco Mundial (`None` si es Fuente 2) | `"NY.GDP.MKTP.KD.ZG"` |
| `descripcion` | Texto que usa el chatbot RAG | *"Tasa de crecimiento…"* |

Se definen **9 indicadores**:

| Código | Indicador | Fuente | Unidad | Código Banco Mundial |
|--------|-----------|--------|--------|----------------------|
| `pib_crecimiento` | PIB - Crecimiento anual | Banco Mundial | % | `NY.GDP.MKTP.KD.ZG` |
| `pib_per_capita` | PIB per cápita | Banco Mundial | USD | `NY.GDP.PCAP.CD` |
| `inflacion` | Inflación (IPC) | Banco Mundial | % | `FP.CPI.TOTL.ZG` |
| `desempleo` | Tasa de desempleo | Banco Mundial | % | `SL.UEM.TOTL.ZS` |
| `pib_usd` | PIB total | Banco Mundial | USD | `NY.GDP.MKTP.CD` |
| `ied` | Inversión Extranjera Directa | Banco Mundial | USD | `BX.KLT.DINV.CD.WD` |
| `canal_transitos` | Tránsitos por el Canal | Contraloría/ACP | buques/año | *(CSV)* |
| `canal_ingresos` | Ingresos del Canal | Contraloría/ACP | millones USD | *(CSV)* |
| `imae` | Índice Mensual de Actividad Económica | Contraloría/INEC | índice 2007=100 | *(CSV)* |

Se derivan vistas auxiliares: `CODIGOS_INDICADORES`, `INDICADORES_BANCO_MUNDIAL`,
`INDICADORES_CONTRALORIA` y `META_POR_CODIGO` (diccionario `código → metadatos`),
más dos funciones de ayuda: `nombre_indicador(codigo)` y `unidad_indicador(codigo)`.

### 4.3 Parámetros de la API y del chatbot

```python
WB_BASE_URL = "https://api.worldbank.org/v2"   # API del Banco Mundial
PAIS_ISO3   = "PAN"                             # Código ISO de Panamá
ANIO_INICIO, ANIO_FIN = 2000, 2024
ANIOS_PRONOSTICO = 3                            # Años a predecir
MODELO_CLAUDE = "claude-opus-4-8"              # Modelo de IA del chatbot
RAG_TOP_K = 5                                   # Fragmentos recuperados por consulta
```

---

## 5. Módulo 1 — Pipeline de datos

El pipeline cumple el requisito de **ingesta de al menos 2 fuentes de datos
diferentes** y de **preprocesar y transformar** los datos.

### 5.1 Ingesta (`ingesta.py`)

#### Fuente 1 — Banco Mundial (API REST)

**`descargar_indicador_banco_mundial(wb_code)`** descarga **un** indicador:

1. Construye la URL:
   `https://api.worldbank.org/v2/country/PAN/indicator/{wb_code}`
   con parámetros `format=json`, `per_page=500`, `date=2000:2024`.
2. Hace la petición con `requests.get(...)` y `raise_for_status()` (lanza
   excepción si el código HTTP es ≥ 400).
3. La API devuelve una **lista de 2 elementos**: `[metadatos, observaciones]`.
   Cada observación es un dict con `date` (año) y `value` (valor, puede ser `None`).
4. Construye un `DataFrame` con columnas `['anio', 'valor']`, ordenado por año.
5. **Manejo de errores:** si hay cualquier problema de red o de formato
   (`requests.RequestException` o `ValueError` de JSON inválido), devuelve `None`
   en lugar de fallar.

**`descargar_banco_mundial(usar_respaldo_si_falla=True)`** itera sobre los 6
indicadores del Banco Mundial. Si alguno devuelve `None`, usa los **datos de
respaldo** embebidos (`_RESPALDO_BANCO_MUNDIAL`), que contienen cifras reales
representativas de Panamá 2000–2024. Devuelve un DataFrame en **formato largo**:
`['anio', 'codigo', 'valor', 'fuente']`. Guarda una copia en
`data/raw/banco_mundial_panama.csv`.

> **¿Por qué datos de respaldo?** Garantiza que el proyecto sea **100 %
> reproducible** aunque no haya internet o la API esté caída. Cuando sí hay
> conexión, los datos se descargan **en vivo** (mensaje `[ok] … en vivo`).

#### Fuente 2 — Contraloría / INEC / Canal (archivo CSV)

- **`generar_csv_contraloria()`** crea `data/raw/contraloria_panama.csv` a partir
  de `_DATOS_CONTRALORIA` (cifras representativas reales de tránsitos e ingresos
  del Canal e IMAE). Simula la descarga manual desde el portal de la Contraloría.
- **`cargar_contraloria()`** lee ese CSV (formato **ancho**: una columna por
  indicador) y lo convierte a **formato largo** con `melt`, etiquetando la fuente.

> **Honestidad metodológica.** El CSV de la Fuente 2 contiene valores
> **representativos basados en cifras públicas reales** de la Contraloría/ACP, no
> una descarga byte a byte del portal. Esto se documenta explícitamente para no
> presentar datos sintéticos como oficiales. La Fuente 1 (Banco Mundial) sí es
> una descarga real en vivo cuando hay internet.

#### Orquestador

**`ingestar_todo()`** llama a las dos fuentes, concatena los resultados con
`pd.concat` y devuelve el DataFrame combinado en formato largo. Es la función que
usan el notebook y el dashboard.

**Resultado típico:** 225 observaciones = 9 indicadores × 25 años.

### 5.2 Preprocesamiento (`preprocesamiento.py`)

Toma el DataFrame largo crudo y lo transforma en datasets listos para analizar.

#### Etapa 1 — Limpieza (`limpiar`)

- `pd.to_numeric(..., errors="coerce")` convierte `anio` a entero y `valor` a
  numérico; lo no convertible se vuelve `NaN`.
- Elimina filas sin año y **duplicados** de `(anio, codigo)` conservando el último.
- Ordena por `(codigo, anio)`.

#### Etapa 2 — Pivoteo a formato ancho (`a_formato_ancho`)

`pivot_table(index="anio", columns="codigo", values="valor")` transforma de
**largo** (una fila por observación) a **ancho** (una fila por año, una columna por
indicador). Las columnas se reordenan según el catálogo de `config`.

| largo (tidy) | → | ancho (wide) |
|---|---|---|
| 2020, pib_crecimiento, -17.9 | | anio · pib_crecimiento · inflacion · … |
| 2020, inflacion, -1.6 | | 2020 · -17.9 · -1.6 · … |

#### Etapa 3 — Manejo de nulos (`manejar_nulos`)

Estrategia apropiada para **series de tiempo anuales**:

1. **Interpolación lineal** en el tiempo (`interpolate(method="linear")`): rellena
   huecos internos estimando un valor entre los vecinos.
2. **Relleno hacia adelante y atrás** (`.ffill().bfill()`): cubre los extremos
   (primeros/últimos años sin dato).

Devuelve también un **reporte de nulos** (cuántos faltaban por indicador antes de
imputar), útil para documentar la calidad de los datos.

#### Etapa 4 — Ingeniería de características (`crear_features`)

Para **cada** indicador genera tres variables derivadas:

| Sufijo | Qué es | Cálculo | Para qué sirve |
|--------|--------|---------|----------------|
| `_var` | Variación interanual (%) | `pct_change()*100` | Mide aceleración/desaceleración |
| `_mm3` | Media móvil de 3 años | `rolling(3).mean()` | Suaviza la tendencia |
| `_lag1` | Valor del año anterior | `shift(1)` | "Memoria" de la serie (autocorrelación) |

> **Detalle técnico (pandas 3.x):** `pct_change(fill_method=None)` se usa
> explícitamente porque el relleno implícito quedó obsoleto. La primera fila no
> tiene año anterior, así que `_var` se rellena con 0 y `_lag1` con el propio valor.

#### Resumen de calidad (`resumen_calidad`)

Devuelve, por indicador: número de observaciones, año mínimo, año máximo y fuente.
Sirve para documentar la cobertura de los datos.

#### Orquestador (`preprocesar_todo`)

Encadena las 4 etapas y devuelve un **diccionario** con 5 datasets:

| Llave | Contenido |
|-------|-----------|
| `largo` | DataFrame largo limpio |
| `ancho` | Una columna por indicador, sin nulos (25 años × 9 indicadores) |
| `features` | `ancho` + las 27 variables derivadas (37 columnas: `anio` + 9 indicadores + 27 derivadas) |
| `reporte_nulos` | Nulos por indicador antes de imputar |
| `calidad` | Cobertura por indicador |

Guarda los CSV en `data/processed/`.

---

## 6. Módulo 2 — Análisis exploratorio y visualización

El EDA (*Exploratory Data Analysis*) vive en el notebook (Sección 4) usando
`matplotlib`. Se generan:

1. **Estadística descriptiva** con `df_ancho[...].describe()` (conteo, media,
   desviación, mínimo, cuartiles, máximo).
2. **Línea del crecimiento del PIB** con el desplome de 2020 resaltado
   (`ax.scatter` + `ax.annotate` para marcar el punto COVID-19).
3. **Inflación (barras) y desempleo (línea)** en un gráfico de **doble eje Y**
   (`ax.twinx()`), porque tienen escalas distintas.
4. **Canal de Panamá**: tránsitos vs. ingresos, también con doble eje.
5. **Matriz de correlación** (`df.corr()`) dibujada como mapa de calor con
   `imshow` y el valor numérico anotado en cada celda.

> **¿Por qué `matplotlib` en el notebook y `plotly` en el dashboard?** En el
> notebook interesa que las imágenes queden **embebidas** y se vean al abrir el
> archivo; `matplotlib` produce PNG estáticos ideales para eso. En el dashboard
> interesa la **interactividad** (zoom, hover), y para eso `plotly` es mejor.

---

## 7. Módulo 3 — Machine Learning (`modelos.py`)

Se implementan **dos técnicas** de ML, cubriendo el requisito de "al menos 1
técnica" con margen.

### 7.1 A) Regresión / pronóstico de series de tiempo

**Objetivo:** predecir el valor futuro de **2 indicadores** (PIB per cápita e
inflación).

#### `_crear_modelo(nombre)` — fábrica de modelos

- `"linear"` → `Pipeline([StandardScaler, LinearRegression])`. La
  estandarización pone todas las variables en la misma escala; la regresión lineal
  **sí extrapola** con tendencia, ideal para predecir años futuros.
- `"rf"` → `RandomForestRegressor`. Capta no linealidades pero **no extrapola**
  fuera del rango visto; se incluye solo para comparar.

#### `_construir_supervisado(features, objetivo, n_lags=2)`

Convierte la serie de tiempo en un problema **supervisado**. Predictores:
- `anio` → captura la **tendencia** temporal.
- `<objetivo>_lag1`, `<objetivo>_lag2` → los **2 valores anteriores**
  (autocorrelación; modelo autorregresivo AR).

Descarta las primeras filas que no tienen historia suficiente. Esto equivale a un
modelo **AR(2) con término de tendencia**.

#### `evaluar_modelo(...)` — validación honesta

- **Partición temporal** (no aleatoria): los primeros años entrenan, los **últimos
  `n_test=5`** se reservan para probar. *En series de tiempo nunca se debe usar el
  futuro para entrenar.*
- Entrena, predice el periodo de prueba y calcula tres métricas:

| Métrica | Significado | Mejor cuando |
|---------|-------------|--------------|
| **MAE** (Error Absoluto Medio) | Promedio de los errores en valor absoluto | Más bajo |
| **RMSE** (Raíz del Error Cuadrático Medio) | Penaliza más los errores grandes | Más bajo |
| **R²** (Coeficiente de determinación) | % de la varianza explicada (1 = perfecto, 0 = no mejora la media, negativo = peor que la media) | Más alto |

#### `pronosticar(...)` — predicción del futuro

Reentrena con **todos** los datos y pronostica de forma **recursiva**: la
predicción de un año se reutiliza como rezago para predecir el siguiente. Devuelve
un DataFrame con columna `tipo` ∈ {`histórico`, `pronóstico`}.

#### `entrenar_indicadores(...)`

Aplica evaluación + pronóstico a la lista de objetivos (por defecto
`["pib_per_capita", "inflacion"]`) y devuelve un dict con todo.

> **Interpretación de resultados.** El PIB per cápita, con tendencia marcada, se
> predice razonablemente. La **inflación** es ruidosa y depende de choques
> externos, por lo que su R² puede ser bajo o **negativo**; se reporta con
> honestidad. Con solo 25 puntos anuales, esto es esperable y se documenta como
> una limitación, no se "maquilla".

### 7.2 B) Clustering de regímenes económicos

**`clustering_regimenes(ancho, variables, n_clusters=3)`**:

1. Selecciona variables (`pib_crecimiento`, `inflacion`, `desempleo`, `imae`).
2. **Estandariza** con `StandardScaler` (para que todas pesen igual sin importar
   su escala — el crecimiento va en %, el IMAE en cientos).
3. Aplica **KMeans** (`n_clusters=3`, `random_state=42` para reproducibilidad,
   `n_init=10`). KMeans agrupa los años en 3 grupos de comportamiento similar.
4. Calcula el **coeficiente de silueta** (`silhouette_score`): mide qué tan bien
   separados quedan los grupos, de −1 (mal) a +1 (excelente).
5. **Etiqueta interpretable:** ordena los clústeres por crecimiento medio y les
   pone nombre: *Contracción / crisis*, *Crecimiento moderado*, *Expansión fuerte*.

Devuelve: `asignaciones` (cada año con su clúster y etiqueta), `perfiles`
(promedio de cada variable por clúster), `silueta` y `variables`.

---

## 8. Módulo 4 — Dashboard interactivo (`app.py`)

Aplicación web hecha con **Streamlit** que reutiliza todos los módulos de `src/`.

### 8.1 Estructura

- **Configuración de página** (`st.set_page_config`): título, ícono 🇵🇦, ancho.
- **Carga con caché:** `@st.cache_data` envuelve el pipeline, los modelos y el
  clustering para que **no se recalculen** en cada interacción del usuario.
  `@st.cache_resource` guarda el chatbot (que mantiene un modelo TF-IDF en memoria).
  El prefijo `_` en los argumentos (`_features`, `_ancho`) le dice a Streamlit que
  no intente "hashear" esos DataFrames.
- **KPIs** (`st.metric`): muestran el último valor de 4 indicadores clave con su
  **variación** (delta) respecto al año anterior.

### 8.2 Las 4 pestañas (`st.tabs`)

| Pestaña | Contenido | Componentes |
|---------|-----------|-------------|
| **📈 Tendencias** | Líneas de los indicadores seleccionados | `st.multiselect`, `st.slider` (rango de años), checkbox de normalización, `plotly.express.line` |
| **🔮 Predicciones** | Histórico + pronóstico del indicador elegido + métricas | `st.selectbox`, `st.metric`, `plotly.graph_objects.Scatter` |
| **🧩 Análisis** | Clustering: dispersión coloreada por régimen + perfiles + línea de tiempo | `plotly.express.scatter`, `st.dataframe` |
| **💬 Chatbot RAG** | Caja de preguntas + botones de ejemplo + respuesta + fuentes | `st.text_input`, `st.button`, `st.expander` |

### 8.3 Detalles de UX

- Botones de **preguntas de ejemplo** que rellenan la caja del chatbot vía
  `st.session_state`.
- **Indicador del modo** del chatbot (Claude vs. extractivo) según haya o no clave.
- **Barra lateral** (`st.sidebar`) con la descripción del proyecto y las fuentes.
- Tema visual definido en `.streamlit/config.toml` (azul de la bandera de Panamá).

> **Nota sobre la API de Streamlit.** Se usa `width="stretch"` (la API moderna) en
> lugar del antiguo `use_container_width=True`, que quedó obsoleto.

---

## 9. Módulo 5 — Chatbot con RAG (`chatbot.py`)

### 9.1 ¿Qué es RAG?

**RAG = Retrieval-Augmented Generation** (Generación Aumentada por Recuperación).
En lugar de pedirle al modelo que conteste de memoria (lo que puede llevar a
**alucinaciones**, es decir, inventar cifras), el sistema:

1. **Recupera** los fragmentos de datos más relevantes para la pregunta, desde una
   base de conocimiento hecha con **nuestros** datos.
2. **Genera** la respuesta con un modelo de lenguaje, pasándole esos fragmentos
   como contexto y pidiéndole que responda **solo** con base en ellos.

Resultado: respuestas **ancladas (grounded)** en datos reales y verificables.

### 9.2 Construcción de la base de conocimiento (`construir_base_conocimiento`)

Genera ~165 **documentos** de texto corto a partir de los datasets procesados:

| Tipo | Ejemplo |
|------|---------|
| `descripcion` | *"PIB per cápita (USD), fuente: Banco Mundial. Producto interno bruto…"* |
| `valor` | *"En 2020, PIB - Crecimiento anual de Panamá fue -17.9% (%)."* |
| `resumen` | *"Resumen de Tasa de desempleo (2000-2024): pasó de … máximo … mínimo …"* |
| `pronostico` | *"Pronóstico del modelo para PIB per cápita: 2025: …; 2026: …"* |
| `contexto` | *"En 2020 la pandemia de COVID-19 provocó la mayor recesión…"* |

La función `_formato_valor` da formato legible según el tipo (%, USD en millones/
miles de millones, buques, etc.).

### 9.3 Recuperación (retrieval)

```python
self.vectorizador = TfidfVectorizer(
    stop_words=STOPWORDS_ES, strip_accents="unicode",
    lowercase=True, ngram_range=(1, 2),
)
self.matriz = self.vectorizador.fit_transform(self.textos)
```

- **TF-IDF** (*Term Frequency – Inverse Document Frequency*) convierte cada
  documento en un **vector numérico** que pondera las palabras: las frecuentes en
  un documento pero raras en general pesan más.
- `strip_accents="unicode"` normaliza las tildes; `ngram_range=(1, 2)` incluye
  **bigramas** (pares de palabras) para captar frases; `STOPWORDS_ES` es una lista
  de palabras vacías en español (artículos, preposiciones).
- **`recuperar(pregunta, k)`** vectoriza la pregunta con el mismo TF-IDF, calcula
  la **similitud de coseno** (`cosine_similarity`) contra todos los documentos y
  devuelve los `k=5` más parecidos (con su puntaje).

### 9.4 Generación (generation)

**`responder(pregunta)`** orquesta:

1. Recupera el contexto.
2. Si hay cliente de Claude → **`_generar_con_claude`**: arma un *prompt* con un
   **system prompt** (instruye a responder solo con el contexto, en español, sin
   inventar) y la pregunta + contexto, y llama a:
   ```python
   self.cliente_claude.messages.create(
       model="claude-opus-4-8", max_tokens=1024,
       system=sistema, messages=[{"role": "user", "content": prompt}],
   )
   ```
   Extrae el texto de los bloques de contenido de la respuesta.
3. Si **no** hay clave o la llamada falla → **`_generar_extractivo`**: arma la
   respuesta listando los fragmentos recuperados. Garantiza que el chatbot
   **funcione siempre**.

Devuelve `{'respuesta', 'fuentes', 'modo'}` donde `modo` ∈ {`claude`, `extractivo`}.

### 9.5 Detección de la clave

`_crear_cliente_claude()` revisa la variable de entorno `ANTHROPIC_API_KEY`. Si
existe y el SDK `anthropic` está instalado, crea el cliente; si no, devuelve
`None` y el chatbot opera en modo extractivo.

> **Sobre el modelo de IA.** Se usa `claude-opus-4-8` (el modelo Claude Opus 4.8
> más reciente) por defecto, vía el SDK oficial de Anthropic. Es configurable en
> `config.MODELO_CLAUDE`. El proyecto no requiere costo para funcionar gracias al
> modo extractivo de respaldo.

---

## 10. Recorrido celda por celda del notebook

El notebook tiene **40 celdas** (21 de texto/Markdown + 19 de código). A
continuación, qué hace cada bloque.

### Sección 0–1 · Portada y problemática *(Markdown)*

Presentan el proyecto, la tabla de mapeo con los 5 módulos del curso, la
problemática (datos dispersos entre INEC, Contraloría, ACP y Banco Mundial) y el
objetivo. Incluyen la nota sobre el origen de los datos.

### Sección 2 · Configuración del entorno

**Celda de código 1 (importaciones y rutas).**
- Importa `sys`, `Path`, `numpy`, `pandas`, `matplotlib`.
- `%matplotlib inline` hace que las gráficas se rendericen dentro del notebook.
- Configura `rcParams` (tamaño de figura, rejilla) y opciones de visualización de
  pandas.
- Define `encontrar_raiz()`, que sube por el árbol de carpetas hasta hallar
  `src/config.py`. Así el notebook **funciona aunque se ejecute desde `notebooks/`**
  o desde la raíz. Añade esa raíz a `sys.path`.
- Importa los módulos del proyecto: `config`, `ingestar_todo`, `preprocesar_todo`,
  `entrenar_indicadores`, `clustering_regimenes`, `ChatbotRAG`.
- **Salida esperada:** la ruta de la raíz y el número de indicadores del catálogo.

### Sección 3 · Pipeline de datos

**Celda 2 — Ingesta.** Llama `ingestar_todo()`. Imprime mensajes de progreso de
cada fuente, la forma del DataFrame (225 filas × 4 columnas) y muestra
`head(8)`. **Salida:** primeras filas del formato largo `['anio','codigo','valor','fuente']`.

**Celda 3 — Observaciones por fuente.** Agrupa por `fuente` y cuenta observaciones
e indicadores. **Salida:** tabla mostrando que el Banco Mundial aporta 6
indicadores y la Contraloría/INEC/ACP aporta 3.

**Celda 4 — Preprocesamiento.** Llama `preprocesar_todo()`, desempaqueta los 5
datasets (`largo`, `ancho`, `features`, `reporte_nulos`, `calidad`). Imprime las
formas y muestra `df_ancho.tail(6)` (los años recientes con todos los indicadores
en columnas). **Salida:** 25 años × 9 indicadores; 37 columnas con features.

**Celda 5 — Calidad y nulos.** Muestra la tabla de `calidad` (cobertura por
indicador) y el `reporte_nulos`. **Salida:** dos tablas que documentan la
completitud de los datos.

**Celda 6 — Features.** Muestra columnas derivadas del PIB per cápita
(`_var`, `_mm3`, `_lag1`). **Salida:** tabla que ilustra la ingeniería de
características.

### Sección 4 · Análisis exploratorio (EDA)

**Celda 7 — Estadística descriptiva.** `describe()` de 6 indicadores macro.
**Salida:** tabla con media, desviación, mínimo, cuartiles, máximo.

**Celda 8 — Gráfica del PIB.** Línea del crecimiento del PIB con el punto de 2020
(−17.9 %) resaltado en rojo y anotado. **Salida:** imagen PNG embebida.

**Celda 9 — Inflación y desempleo.** Barras (inflación) + línea (desempleo) en
doble eje Y. **Salida:** imagen.

**Celda 10 — Canal de Panamá.** Tránsitos vs. ingresos en doble eje. **Salida:**
imagen.

**Celda 11 — Matriz de correlación.** Mapa de calor `RdBu_r` con los coeficientes
anotados. **Salida:** imagen que revela, por ejemplo, la correlación negativa
entre crecimiento y desempleo.

### Sección 5 · Machine Learning

**Celda 12 — Entrenamiento y métricas.** Llama `entrenar_indicadores(...)` para
PIB per cápita e inflación; arma una tabla con MAE, RMSE y R². **Salida:** tabla
de métricas + mensajes de progreso.

**Celda 13 — Gráficas de pronóstico.** Dos subgráficas (histórico azul +
pronóstico rojo punteado) para los 2 indicadores. **Salida:** imagen.

**Celda 14 — Valores pronosticados.** Imprime los valores numéricos de los
próximos 3 años para cada indicador. **Salida:** texto con las predicciones.

**Celda 15 — Clustering.** Llama `clustering_regimenes(...)`, imprime el
coeficiente de silueta y muestra los `perfiles` por régimen. **Salida:** tabla con
el perfil promedio de cada uno de los 3 regímenes.

**Celda 16 — Gráfica de regímenes.** Dispersión crecimiento vs. desempleo,
coloreada por régimen, con el año anotado en cada punto. **Salida:** imagen donde
2020 cae claramente en "Contracción / crisis".

### Sección 6 · Chatbot RAG

**Celda 17 — Construcción del chatbot.** Crea `ChatbotRAG(df_ancho, resultados_ml)`,
imprime el número de documentos indexados (~165) y si usa Claude. **Salida:**
texto informativo.

**Celda 18 — Prueba de recuperación.** Llama `chatbot.recuperar(...)` para una
pregunta y muestra los fragmentos con su puntaje de similitud. **Salida:** lista
de fragmentos relevantes ordenados.

**Celda 19 — Respuestas completas.** Recorre 4 preguntas de ejemplo y muestra la
respuesta y el modo de cada una. **Salida:** respuestas del chatbot ancladas en
los datos.

### Sección 7–9 · Dashboard, conclusiones y referencias *(Markdown)*

Explican cómo ejecutar el dashboard (`streamlit run dashboard/app.py`), resumen
los hallazgos y listan las fuentes y librerías.

---

## 11. Decisiones de diseño y justificación

| Decisión | Justificación |
|----------|---------------|
| **Código en `src/`, no todo en el notebook** | Reutilización: el notebook y el dashboard usan los mismos módulos. Es la práctica profesional y evita divergencias. |
| **Datos de respaldo embebidos** | Reproducibilidad total: el proyecto se ejecuta sin internet. |
| **2 mecanismos de ingesta (API + CSV)** | Cumple con creces el requisito de "≥ 2 fuentes diferentes" y demuestra dos formas distintas de obtener datos. |
| **Formato largo y ancho** | El largo (tidy) es ideal para combinar fuentes; el ancho es ideal para modelar y graficar. |
| **Partición temporal en ML** | En series de tiempo, usar partición aleatoria filtraría información del futuro; la temporal es metodológicamente correcta. |
| **Regresión lineal para pronosticar** | Extrapola con tendencia; Random Forest no puede predecir más allá del rango visto. |
| **TF-IDF en lugar de embeddings neuronales** | Funciona offline, sin claves ni costo, y es suficiente para el tamaño del corpus. |
| **Modo extractivo de respaldo en el chatbot** | El chatbot funciona siempre, con o sin clave de API. |
| **`claude-opus-4-8`** | Modelo Claude más reciente y capaz para la generación del chatbot. |

---

## 12. Mapeo con la rúbrica de evaluación

| Componente (peso) | Dónde se cumple | Evidencia |
|-------------------|-----------------|-----------|
| **Pipeline de datos (30 %)** | `src/pipeline/` + Notebook §3 | Ingesta de 2 fuentes (API Banco Mundial + CSV Contraloría/INEC), limpieza, pivoteo, manejo de nulos, feature engineering, datasets guardados en `data/processed/`. |
| **Análisis ML (25 %)** | `src/ml/modelos.py` + Notebook §5 | Regresión/pronóstico de **2 indicadores** con evaluación (MAE/RMSE/R²) + **clustering** KMeans con coeficiente de silueta. |
| **Visualización/Dashboard (25 %)** | `dashboard/app.py` + Notebook §4 | Dashboard interactivo Streamlit con 4 pestañas, KPIs y gráficas; EDA con 4 gráficas + matriz de correlación en el notebook. |
| **Documentación (20 %)** | Este `DOCUMENTACION.md` + `README.md` + comentarios | Documentación exhaustiva del pipeline, los modelos, el chatbot y cada celda del notebook. |
| **Chatbot con RAG** *(requisito específico)* | `src/rag/chatbot.py` + Notebook §6 | Recuperación TF-IDF + similitud de coseno + generación con Claude (con respaldo extractivo). |

---

## 13. Glosario de términos

- **Pipeline de datos:** secuencia automatizada de pasos que toma datos crudos y
  los deja listos para analizar.
- **Formato largo / ancho (tidy / wide):** dos formas de organizar una tabla. Largo
  = una fila por observación; ancho = una fila por sujeto (aquí, por año).
- **Imputación:** rellenar valores faltantes con una estimación.
- **Feature engineering:** crear nuevas variables (características) a partir de las
  existentes para mejorar los modelos.
- **Rezago (lag):** valor de una variable en un periodo anterior.
- **Regresión:** modelo que predice un número (aquí, el valor futuro de un indicador).
- **Partición temporal:** dividir los datos en entrenamiento (pasado) y prueba
  (futuro) respetando el orden del tiempo.
- **MAE / RMSE / R²:** métricas de error y bondad de ajuste de un modelo.
- **Clustering:** agrupar elementos parecidos sin etiquetas previas (no supervisado).
- **KMeans:** algoritmo de clustering que forma `k` grupos minimizando la distancia
  al centro de cada grupo.
- **Coeficiente de silueta:** medida (−1 a 1) de qué tan bien separados están los
  grupos de un clustering.
- **RAG (Retrieval-Augmented Generation):** técnica que combina recuperación de
  información con generación de texto por un modelo de lenguaje.
- **TF-IDF:** forma de representar texto como vectores ponderando la importancia de
  cada palabra.
- **Similitud de coseno:** medida de qué tan parecidos son dos vectores (1 = iguales).
- **Alucinación (LLM):** cuando un modelo de lenguaje inventa información falsa.
- **KPI:** *Key Performance Indicator*, indicador clave que se muestra destacado.

---

*Documentación del Proyecto Integrador — Gestión de la Información, Universidad
Tecnológica de Panamá, I Semestre 2026.*
