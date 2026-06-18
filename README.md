# 🇵🇦 Dashboard de Indicadores Económicos de Panamá con IA

> Proyecto Integrador — **Segundo Parcial** · Gestión de la Información
> Universidad Tecnológica de Panamá · Facultad de Ingeniería de Sistemas Computacionales · I Semestre 2026

Sistema de gestión de información que **ingiere, procesa, modela y presenta** los
principales indicadores económicos de Panamá, e incluye un **chatbot con IA (RAG)**
que responde preguntas sobre los datos.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Claude](https://img.shields.io/badge/Chatbot-Claude%20(RAG)-7b4ae2)

---

## 🎯 Problemática

Los indicadores económicos de Panamá los publican distintas instituciones (INEC /
Contraloría, Autoridad del Canal de Panamá, Banco Mundial) en portales y formatos
diferentes. Reunirlos, entender sus tendencias, anticipar su evolución y
consultarlos rápidamente es difícil. Este proyecto resuelve eso con un pipeline de
datos, modelos predictivos, un dashboard y un chatbot.

## ✨ Características

- 🔄 **Pipeline de datos** con ingesta de **2 fuentes diferentes**
  (API del Banco Mundial + CSV de la Contraloría/INEC/ACP).
- 🧹 **Preprocesamiento**: limpieza, manejo de nulos, pivoteo y *feature engineering*.
- 🤖 **Machine Learning**: pronóstico (regresión) de **2 indicadores** + **clustering**
  de regímenes económicos (KMeans).
- 📊 **Dashboard interactivo** (Streamlit) con tendencias, predicciones y análisis.
- 💬 **Chatbot con RAG** conectado a los datos (recuperación TF-IDF + generación con
  Claude `claude-opus-4-8`, con modo extractivo de respaldo sin clave).

## 🗺️ Arquitectura

```
 FUENTES            PIPELINE              INTELIGENCIA          PRESENTACIÓN
Banco Mundial ─API─▶ ingesta → limpieza  ─▶ ML: regresión   ─▶ Notebook (.ipynb)
Contraloría   ─CSV─▶ → transformación      ML: clustering      Dashboard (Streamlit)
INEC / ACP          → features             Chatbot RAG         Chatbot web
```

Toda la lógica vive en `src/`; el **notebook** y el **dashboard** la reutilizan.

## 📁 Estructura del repositorio

```
parcial2/
├── notebooks/proyecto_indicadores_economicos.ipynb   # ⭐ Notebook principal
├── src/
│   ├── config.py                 # Catálogo de indicadores y parámetros
│   ├── pipeline/{ingesta,preprocesamiento}.py   # Módulo 1
│   ├── ml/modelos.py             # Módulo 3 (regresión + clustering)
│   └── rag/chatbot.py            # Módulo 5 (RAG)
├── dashboard/app.py              # Módulo 4 (Streamlit)
├── data/{raw,processed}/         # Datos crudos y procesados
├── docs/DOCUMENTACION.md         # 📘 Documentación detallada
├── requirements.txt
└── README.md
```

## 🚀 Instalación y uso

> 📄 Guía exacta paso a paso: [`docs/GUIA_RAPIDA.md`](docs/GUIA_RAPIDA.md)

**Forma fácil (con scripts):**

```bash
bash scripts/instalar.sh          # 1.ª vez: crea el entorno e instala todo
bash scripts/preparar_ollama.sh   # 1.ª vez: prepara el chatbot gratis (Ollama)
bash scripts/dashboard.sh         # abre el dashboard en http://localhost:8501
# Otros:  bash scripts/notebook.sh   ·   bash scripts/chatbot.sh
```

**Forma manual:**

```bash
# 1) Clonar e instalar
git clone <url-del-repositorio>
cd parcial2
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Ejecutar el notebook
jupyter notebook notebooks/proyecto_indicadores_economicos.ipynb

# 3) Ejecutar el dashboard
streamlit run dashboard/app.py        # abre http://localhost:8501
```

### 💬 Chatbot con IA generativa (gratis)

El chatbot funciona sin instalar nada (modo extractivo). Para respuestas redactadas
por una IA **gratis y local**, usa **Ollama**:

```bash
# 1) Instala Ollama: https://ollama.com/download
# 2) Descarga un modelo (una vez):
ollama pull llama3.2
# 3) Corre el dashboard: el chatbot detecta Ollama automáticamente 🟢
```

Alternativa en la nube (de pago) con Claude:

```bash
export ANTHROPIC_API_KEY="tu-clave-de-anthropic"   # Windows: set ANTHROPIC_API_KEY=...
```

> **Reproducible sin internet:** si la API del Banco Mundial no responde, el
> pipeline usa datos de respaldo representativos y todo sigue funcionando.

## 📊 Fuentes de datos

| Fuente | Mecanismo | Indicadores |
|--------|-----------|-------------|
| **Banco Mundial** ([data.worldbank.org](https://data.worldbank.org)) | API REST | PIB, crecimiento, inflación, desempleo, IED, PIB per cápita |
| **Contraloría / INEC / ACP** ([contraloria.gob.pa](https://www.contraloria.gob.pa/inec/)) | Archivo CSV | Tránsitos e ingresos del Canal de Panamá, IMAE |

> Los datos del Banco Mundial se descargan en vivo. La fuente Contraloría/INEC/ACP
> se entrega como CSV con cifras representativas basadas en datos públicos reales
> (2000–2024), documentado en [`docs/DOCUMENTACION.md`](docs/DOCUMENTACION.md).

## 🤖 Modelos de Machine Learning

- **Pronóstico (regresión):** modelo autorregresivo con tendencia para **PIB per
  cápita** e **inflación**, evaluado con partición temporal (MAE, RMSE, R²).
- **Clustering (KMeans):** agrupa los años en regímenes económicos
  (*crisis · crecimiento moderado · expansión*), medido con el coeficiente de silueta.

## 🧮 Tecnologías

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `plotly` ·
`Streamlit` · `requests` · `Anthropic Claude API`

## 📋 Mapeo con la rúbrica

| Componente | Peso | Ubicación |
|------------|------|-----------|
| Pipeline de datos | 30 % | `src/pipeline/` · Notebook §3 |
| Análisis ML | 25 % | `src/ml/modelos.py` · Notebook §5 |
| Visualización / Dashboard | 25 % | `dashboard/app.py` · Notebook §4 |
| Documentación | 20 % | `docs/DOCUMENTACION.md` · `README.md` |

## 📖 Documentación

La explicación **detallada de cada componente y de cada celda del notebook** está
en 👉 [`docs/DOCUMENTACION.md`](docs/DOCUMENTACION.md).

## 👤 Autor
Carlos Delgado
Proyecto del curso **Gestión de la Información**, Universidad Tecnológica de Panamá,
I Semestre 2026.

---

*Trabajo académico. Las cifras de la fuente Contraloría/INEC/ACP son representativas
y basadas en datos públicos reales; verifica siempre con las fuentes oficiales para
usos no académicos.*
