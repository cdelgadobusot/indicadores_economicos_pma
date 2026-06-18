"""
app.py  —  MÓDULO 4: DASHBOARD INTERACTIVO (Streamlit)
======================================================
Dashboard web interactivo que integra todo el proyecto:

  • Indicadores clave (KPIs) con su variación más reciente.
  • Pestaña de TENDENCIAS: gráficas de la evolución de los indicadores.
  • Pestaña de PREDICCIONES: pronósticos del modelo de Machine Learning.
  • Pestaña de ANÁLISIS: clustering de regímenes económicos.
  • Pestaña de CHATBOT: asistente con RAG que responde preguntas sobre los datos.

Cómo ejecutar:
    streamlit run dashboard/app.py

El dashboard reutiliza los módulos del paquete `src`: ingesta, preprocesamiento,
modelos de ML y chatbot. Usa el sistema de caché de Streamlit para no recalcular
el pipeline en cada interacción.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite importar el paquete `src` cuando se ejecuta con `streamlit run`.
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.ml.modelos import clustering_regimenes, entrenar_indicadores  # noqa: E402
from src.pipeline.ingesta import ingestar_todo  # noqa: E402
from src.pipeline.preprocesamiento import preprocesar_todo  # noqa: E402
from src.rag.chatbot import ChatbotRAG  # noqa: E402

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Indicadores Económicos de Panamá con IA",
    page_icon="🇵🇦",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carga de datos y modelos (con caché para que sea rápido e interactivo)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Ejecutando el pipeline de datos...")
def cargar_datos() -> dict:
    """Ejecuta el pipeline completo (ingesta + preprocesamiento) una sola vez."""
    crudo = ingestar_todo()
    return preprocesar_todo(crudo, guardar=True)


@st.cache_data(show_spinner="Entrenando modelos de pronóstico...")
def cargar_modelos(_features: pd.DataFrame) -> dict:
    """Entrena los modelos predictivos. El guion bajo evita hashear el DataFrame."""
    return entrenar_indicadores(_features, objetivos=["pib_per_capita", "inflacion"])


@st.cache_data(show_spinner="Calculando clustering...")
def cargar_clustering(_ancho: pd.DataFrame) -> dict:
    return clustering_regimenes(_ancho)


@st.cache_resource(show_spinner="Inicializando el chatbot RAG...")
def cargar_chatbot(_ancho: pd.DataFrame, _resultados_ml: dict) -> ChatbotRAG:
    return ChatbotRAG(_ancho, _resultados_ml)


datos = cargar_datos()
ancho = datos["ancho"]
features = datos["features"]
resultados_ml = cargar_modelos(features)
clustering = cargar_clustering(ancho)
chatbot = cargar_chatbot(ancho, resultados_ml)

INDICADORES_DISPONIBLES = [c for c in config.CODIGOS_INDICADORES if c in ancho.columns]


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("🇵🇦 Dashboard de Indicadores Económicos de Panamá con IA")
st.caption(
    "Pipeline de datos (Banco Mundial + Contraloría/INEC) · Modelos predictivos · "
    "Chatbot con RAG · Proyecto Integrador — Gestión de la Información, UTP"
)


def kpi_indicador(columna, codigo: str) -> None:
    """Muestra una métrica (valor último año + variación respecto al anterior)."""
    serie = ancho[["anio", codigo]].dropna()
    if len(serie) < 2:
        return
    ultimo = serie.iloc[-1]
    previo = serie.iloc[-2]
    valor = ultimo[codigo]
    delta = valor - previo[codigo]
    unidad = config.unidad_indicador(codigo)

    if "USD" in unidad and abs(valor) >= 1e9:
        texto_valor = f"{valor / 1e9:,.1f} B USD"
        texto_delta = f"{delta / 1e9:,.1f} B"
    elif "USD" in unidad and abs(valor) >= 1e6:
        texto_valor = f"{valor / 1e6:,.0f} M USD"
        texto_delta = f"{delta / 1e6:,.0f} M"
    elif unidad == "%":
        texto_valor = f"{valor:.1f}%"
        texto_delta = f"{delta:+.1f} pp"
    else:
        texto_valor = f"{valor:,.0f}"
        texto_delta = f"{delta:+,.0f}"

    columna.metric(
        label=f"{config.nombre_indicador(codigo)} ({int(ultimo['anio'])})",
        value=texto_valor,
        delta=texto_delta,
    )


st.subheader("Indicadores clave (último año disponible)")
cols = st.columns(4)
for col, codigo in zip(cols, ["pib_crecimiento", "inflacion", "desempleo", "canal_ingresos"]):
    kpi_indicador(col, codigo)


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
tab_tend, tab_pred, tab_clust, tab_chat = st.tabs(
    ["📈 Tendencias", "🔮 Predicciones", "🧩 Análisis (Clustering)", "💬 Chatbot RAG"]
)

# ===== TENDENCIAS ==========================================================
with tab_tend:
    st.markdown("### Evolución histórica de los indicadores")
    col_izq, col_der = st.columns([1, 3])

    with col_izq:
        seleccion = st.multiselect(
            "Indicadores a mostrar",
            options=INDICADORES_DISPONIBLES,
            default=["pib_crecimiento", "inflacion", "desempleo"],
            format_func=config.nombre_indicador,
        )
        rango = st.slider(
            "Rango de años",
            int(ancho["anio"].min()),
            int(ancho["anio"].max()),
            (int(ancho["anio"].min()), int(ancho["anio"].max())),
        )
        normalizar = st.checkbox(
            "Normalizar (base 100 en el primer año)",
            value=False,
            help="Útil para comparar indicadores con escalas muy distintas.",
        )

    with col_der:
        if not seleccion:
            st.info("Selecciona al menos un indicador en el panel de la izquierda.")
        else:
            df_plot = ancho[(ancho["anio"] >= rango[0]) & (ancho["anio"] <= rango[1])].copy()
            largo_plot = df_plot.melt(
                id_vars="anio", value_vars=seleccion,
                var_name="codigo", value_name="valor",
            )
            if normalizar:
                base = largo_plot.sort_values("anio").groupby("codigo")["valor"].transform("first")
                largo_plot["valor"] = largo_plot["valor"] / base * 100
            largo_plot["Indicador"] = largo_plot["codigo"].map(config.nombre_indicador)

            fig = px.line(
                largo_plot, x="anio", y="valor", color="Indicador", markers=True,
                labels={"anio": "Año", "valor": "Valor" + (" (base 100)" if normalizar else "")},
            )
            fig.update_layout(height=460, legend_title_text="", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

    with st.expander("Ver tabla de datos"):
        st.dataframe(
            ancho.rename(columns={c: config.nombre_indicador(c) for c in INDICADORES_DISPONIBLES}),
            width="stretch",
        )

# ===== PREDICCIONES ========================================================
with tab_pred:
    st.markdown("### Pronóstico con modelos de Machine Learning")
    st.caption(
        "Regresión con características de rezago (modelo autorregresivo con "
        "tendencia). Evaluación con partición temporal (los últimos años como prueba)."
    )

    objetivo = st.selectbox(
        "Indicador a pronosticar",
        options=list(resultados_ml.keys()),
        format_func=config.nombre_indicador,
    )
    res = resultados_ml[objetivo]
    df_pron = res["pronostico"]
    metr = res["evaluacion"]["metricas"]

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (error medio absoluto)", f"{metr['MAE']:.2f}")
    c2.metric("RMSE (raíz del error cuadrático)", f"{metr['RMSE']:.2f}")
    c3.metric("R² (bondad de ajuste)", f"{metr['R2']:.3f}")

    hist = df_pron[df_pron["tipo"] == "histórico"]
    futu = df_pron[df_pron["tipo"] == "pronóstico"]
    # Unimos el último histórico al pronóstico para que la línea sea continua.
    futu_linea = pd.concat([hist.tail(1), futu], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["anio"], y=hist[objetivo], mode="lines+markers",
        name="Histórico", line=dict(color="#1f77b4"),
    ))
    fig.add_trace(go.Scatter(
        x=futu_linea["anio"], y=futu_linea[objetivo], mode="lines+markers",
        name="Pronóstico", line=dict(color="#d62728", dash="dash"),
    ))
    fig.update_layout(
        height=440, hovermode="x unified",
        xaxis_title="Año",
        yaxis_title=f"{config.nombre_indicador(objetivo)} ({config.unidad_indicador(objetivo)})",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Valores pronosticados:**")
    tabla = futu[["anio", objetivo]].copy()
    tabla.columns = ["Año", config.nombre_indicador(objetivo)]
    st.dataframe(tabla.style.format({config.nombre_indicador(objetivo): "{:,.2f}"}),
                 width="stretch", hide_index=True)

# ===== CLUSTERING ==========================================================
with tab_clust:
    st.markdown("### Regímenes económicos (clustering KMeans)")
    st.caption(
        "Los años se agrupan según su comportamiento conjunto en crecimiento, "
        "inflación, desempleo e IMAE. El coeficiente de silueta mide qué tan "
        f"bien separados están los grupos: **{clustering['silueta']:.3f}**."
    )

    asign = clustering["asignaciones"]
    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig = px.scatter(
            asign, x="pib_crecimiento", y="desempleo",
            color="etiqueta", text="anio", size="imae", size_max=22,
            labels={
                "pib_crecimiento": "Crecimiento del PIB (%)",
                "desempleo": "Desempleo (%)",
                "etiqueta": "Régimen",
            },
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=460, legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown("**Perfil promedio por régimen**")
        perfiles = clustering["perfiles"].copy()
        perfiles = perfiles.rename(columns={c: config.nombre_indicador(c)
                                            for c in clustering["variables"]})
        st.dataframe(perfiles.drop(columns="cluster"), width="stretch", hide_index=True)

    st.markdown("**Línea de tiempo de regímenes**")
    asign_tl = asign.assign(fila="Régimen económico")
    fig_tl = px.scatter(
        asign_tl, x="anio", y="fila", color="etiqueta",
        labels={"anio": "Año", "fila": ""},
    )
    fig_tl.update_traces(marker=dict(size=16))
    fig_tl.update_layout(height=200, yaxis=dict(showticklabels=False), legend_title_text="")
    st.plotly_chart(fig_tl, width="stretch")

# ===== CHATBOT RAG =========================================================
with tab_chat:
    st.markdown("### Chatbot con RAG sobre los indicadores")
    modo = "Claude (IA generativa)" if chatbot.usa_claude() else "extractivo (sin clave de API)"
    st.caption(
        f"El chatbot recupera datos relevantes y genera la respuesta. Modo actual: "
        f"**{modo}**. Para activar Claude, define la variable de entorno "
        f"`ANTHROPIC_API_KEY`."
    )

    ejemplos = [
        "¿Cómo le fue al PIB de Panamá en 2020?",
        "¿Qué pasó con el desempleo durante la pandemia?",
        "¿Cuál es el pronóstico de la inflación?",
        "¿Cuánto aporta el Canal de Panamá?",
    ]
    st.markdown("**Preguntas de ejemplo:**")
    cols_ej = st.columns(len(ejemplos))
    if "pregunta_chat" not in st.session_state:
        st.session_state.pregunta_chat = ""
    for col, ej in zip(cols_ej, ejemplos):
        if col.button(ej, width="stretch"):
            st.session_state.pregunta_chat = ej

    pregunta = st.text_input(
        "Escribe tu pregunta sobre la economía de Panamá:",
        value=st.session_state.pregunta_chat,
        placeholder="Ej.: ¿Cómo evolucionó la inflación en los últimos años?",
    )

    if pregunta:
        with st.spinner("Buscando en los datos y generando la respuesta..."):
            resultado = chatbot.responder(pregunta)
        st.markdown("#### Respuesta")
        st.write(resultado["respuesta"])
        st.caption(f"Generado en modo: {resultado['modo']}")

        with st.expander("📚 Fuentes recuperadas (RAG)"):
            for fuente in resultado["fuentes"]:
                st.markdown(f"- *(relevancia {fuente['score']:.2f})* {fuente['texto']}")


# ---------------------------------------------------------------------------
# Pie de página / barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Acerca del proyecto")
    st.write(
        "Sistema de gestión de información que integra las técnicas del curso: "
        "pipeline de datos, preprocesamiento, Machine Learning, visualización y "
        "un chatbot con RAG."
    )
    st.markdown("**Fuentes de datos**")
    st.markdown(
        "- 🌐 API del Banco Mundial (macroeconomía)\n"
        "- 📄 Contraloría / INEC / Canal de Panamá (CSV)"
    )
    st.markdown("**Indicadores cargados**")
    st.write(f"{len(INDICADORES_DISPONIBLES)} indicadores · "
             f"{int(ancho['anio'].min())}–{int(ancho['anio'].max())}")
    st.divider()
    st.caption("Universidad Tecnológica de Panamá · Gestión de la Información · 2026")
