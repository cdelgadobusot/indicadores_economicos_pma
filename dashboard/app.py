"""
app.py  —  MÓDULO 4: DASHBOARD INTERACTIVO (Streamlit)
======================================================
Dashboard web interactivo que integra todo el proyecto con un diseño profesional
de tema oscuro (azul marino), tipografías personalizadas y animaciones.

  - Indicadores clave (KPIs) con su variación más reciente.
  - Pestaña TENDENCIAS: evolución histórica de los indicadores.
  - Pestaña PREDICCIONES: pronósticos del modelo de Machine Learning.
  - Pestaña ANÁLISIS: clustering de regímenes económicos.
  - Pestaña ASISTENTE: chatbot con RAG que responde sobre los datos.

Cómo ejecutar:
    streamlit run dashboard/app.py

El dashboard reutiliza los módulos del paquete `src` y usa el sistema de caché de
Streamlit para no recalcular el pipeline en cada interacción.
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
# Paleta de colores del tema (azul marino profesional)
# ---------------------------------------------------------------------------
BG = "#071a2f"          # Fondo principal (azul marino profundo)
BG_CARD = "#0c2340"     # Tarjetas / superficies
BG_HOVER = "#10355c"    # Hover
BORDE = "#1c3e63"       # Bordes sutiles
TEXTO = "#e8f0f8"       # Texto principal
MUTED = "#93a9c4"       # Texto secundario
ACENTO = "#38bdf8"      # Acento primario (azul cielo)
ACENTO2 = "#f5b841"     # Acento secundario (ámbar)
VERDE = "#34d399"       # Variación positiva
ROJO = "#f87171"        # Variación negativa
# Secuencia de colores para las gráficas.
PALETA = ["#38bdf8", "#f5b841", "#34d399", "#a78bfa", "#fb923c", "#22d3ee", "#f472b6", "#4ade80"]

st.set_page_config(
    page_title="Indicadores Económicos de Panamá",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Estilos (CSS): tipografías, colores, animaciones
# ---------------------------------------------------------------------------
def inyectar_estilos() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        /* Fondo general con un degradado sutil */
        [data-testid="stApp"] {{
            background:
                radial-gradient(1200px 600px at 12% -8%, #0e2c4d 0%, rgba(14,44,77,0) 55%),
                radial-gradient(1000px 500px at 100% 0%, #0a2742 0%, rgba(10,39,66,0) 50%),
                {BG};
            color: {TEXTO};
            font-family: 'IBM Plex Sans', sans-serif;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Tipografías */
        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            color: {TEXTO}; letter-spacing: .3px;
        }}
        p, span, label, li {{ font-family: 'IBM Plex Sans', sans-serif; }}

        /* Animaciones */
        @keyframes subir {{
            from {{ opacity: 0; transform: translateY(14px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Encabezado tipo "hero" */
        .hero {{
            animation: subir .6s ease both;
            border: 1px solid {BORDE};
            background: linear-gradient(135deg, rgba(56,189,248,.10), rgba(245,184,65,.06)), {BG_CARD};
            border-radius: 18px; padding: 26px 30px; margin-bottom: 8px;
        }}
        .hero h1 {{ margin: 0; font-size: 2.05rem; font-weight: 700; }}
        .hero .sub {{ color: {MUTED}; margin-top: 6px; font-size: .98rem; }}
        .hero .barra {{
            height: 4px; width: 120px; margin-top: 16px; border-radius: 4px;
            background: linear-gradient(90deg, {ACENTO}, {ACENTO2});
            animation: subir .9s ease both;
        }}
        .chip {{
            display:inline-block; margin-right:8px; margin-top:12px; padding:4px 12px;
            border:1px solid {BORDE}; border-radius:999px; font-size:.78rem; color:{MUTED};
            background: rgba(56,189,248,.06);
        }}

        /* Tarjetas KPI */
        .kpi {{
            border: 1px solid {BORDE}; background: {BG_CARD};
            border-radius: 16px; padding: 18px 20px; height: 100%;
            animation: subir .6s ease both;
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        }}
        .kpi:hover {{
            transform: translateY(-5px);
            border-color: {ACENTO};
            box-shadow: 0 10px 30px rgba(0,0,0,.35), 0 0 0 1px rgba(56,189,248,.25);
        }}
        .kpi .etq {{ color: {MUTED}; font-size: .8rem; text-transform: uppercase; letter-spacing: .6px; }}
        .kpi .val {{
            font-family: 'IBM Plex Mono', monospace; font-weight: 600;
            font-size: 1.9rem; color: {TEXTO}; margin-top: 6px; line-height: 1.1;
        }}
        .kpi .delta {{ font-family: 'IBM Plex Mono', monospace; font-size: .9rem; margin-top: 6px; }}
        .kpi .up {{ color: {VERDE}; }}
        .kpi .down {{ color: {ROJO}; }}

        /* Títulos de sección */
        .sec {{
            font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600;
            margin: 6px 0 2px 0; padding-left: 12px; border-left: 4px solid {ACENTO};
        }}
        .sec-sub {{ color: {MUTED}; font-size: .9rem; margin: 0 0 8px 2px; }}

        /* Pestañas */
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDE}; }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent; color: {MUTED}; border-radius: 10px 10px 0 0;
            padding: 10px 18px; font-family: 'Space Grotesk', sans-serif; font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            color: {TEXTO} !important; background: {BG_CARD} !important;
            border-bottom: 2px solid {ACENTO} !important;
        }}

        /* Botones */
        .stButton button {{
            background: rgba(56,189,248,.08); color: {TEXTO};
            border: 1px solid {BORDE}; border-radius: 10px; font-weight: 500;
            transition: all .18s ease;
        }}
        .stButton button:hover {{
            border-color: {ACENTO}; background: rgba(56,189,248,.18);
            transform: translateY(-2px);
        }}

        /* Barra lateral */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0a2138, #08182a);
            border-right: 1px solid {BORDE};
        }}

        /* Entradas y selectores */
        [data-baseweb="select"] > div, .stTextInput input {{
            background-color: {BG_CARD} !important; border-color: {BORDE} !important;
        }}

        /* Caja de respuesta del asistente */
        .respuesta {{
            border: 1px solid {BORDE}; border-left: 4px solid {ACENTO};
            background: {BG_CARD}; border-radius: 12px; padding: 16px 18px;
            animation: subir .5s ease both; color: {TEXTO}; line-height: 1.55;
        }}
        .modo-badge {{
            display:inline-block; padding:3px 10px; border-radius:999px; font-size:.74rem;
            font-family:'IBM Plex Mono',monospace; border:1px solid {BORDE}; color:{MUTED};
        }}

        /* Barra de scroll */
        ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
        ::-webkit-scrollbar-thumb {{ background: {BORDE}; border-radius: 6px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {ACENTO}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inyectar_estilos()


def estilizar_fig(fig: go.Figure, alto: int = 440) -> go.Figure:
    """Aplica el tema oscuro del dashboard a una figura de Plotly."""
    fig.update_layout(
        template="plotly_dark",
        height=alto,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXTO, size=13),
        colorway=PALETA,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.18, x=0),
        hoverlabel=dict(font_family="IBM Plex Mono, monospace", bgcolor=BG_CARD),
    )
    fig.update_xaxes(gridcolor=BORDE, zerolinecolor=BORDE)
    fig.update_yaxes(gridcolor=BORDE, zerolinecolor=BORDE)
    return fig


def titulo_seccion(titulo: str, subtitulo: str = "") -> None:
    html = f'<div class="sec">{titulo}</div>'
    if subtitulo:
        html += f'<div class="sec-sub">{subtitulo}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de datos y modelos (con caché para que sea rápido e interactivo)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Ejecutando el pipeline de datos...")
def cargar_datos() -> dict:
    crudo = ingestar_todo()
    return preprocesar_todo(crudo, guardar=True)


@st.cache_data(show_spinner="Entrenando modelos de pronóstico...")
def cargar_modelos(_features: pd.DataFrame) -> dict:
    return entrenar_indicadores(_features, objetivos=["pib_per_capita", "inflacion"])


@st.cache_data(show_spinner="Calculando clustering...")
def cargar_clustering(_ancho: pd.DataFrame) -> dict:
    return clustering_regimenes(_ancho)


@st.cache_resource(show_spinner="Inicializando el asistente...")
def cargar_chatbot(_ancho: pd.DataFrame, _resultados_ml: dict) -> ChatbotRAG:
    return ChatbotRAG(_ancho, _resultados_ml)


datos = cargar_datos()
ancho = datos["ancho"]
features = datos["features"]
resultados_ml = cargar_modelos(features)
clustering = cargar_clustering(ancho)
chatbot = cargar_chatbot(ancho, resultados_ml)

INDICADORES_DISPONIBLES = [c for c in config.CODIGOS_INDICADORES if c in ancho.columns]
ANIO_MIN, ANIO_MAX = int(ancho["anio"].min()), int(ancho["anio"].max())


# ---------------------------------------------------------------------------
# Encabezado (hero)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>Indicadores Económicos de Panamá <span style="color:{ACENTO}">con IA</span></h1>
      <div class="sub">Pipeline de datos (Banco Mundial + Contraloría/INEC) &nbsp;|&nbsp;
      Modelos predictivos &nbsp;|&nbsp; Asistente con RAG</div>
      <div class="barra"></div>
      <span class="chip">{len(INDICADORES_DISPONIBLES)} indicadores</span>
      <span class="chip">{ANIO_MIN}&ndash;{ANIO_MAX}</span>
      <span class="chip">Proyecto Integrador &middot; UTP</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def tarjeta_kpi(codigo: str) -> str:
    """Devuelve el HTML de una tarjeta KPI con valor y variación."""
    serie = ancho[["anio", codigo]].dropna()
    if len(serie) < 2:
        return ""
    ultimo, previo = serie.iloc[-1], serie.iloc[-2]
    valor, delta = ultimo[codigo], ultimo[codigo] - previo[codigo]
    unidad = config.unidad_indicador(codigo)

    if "USD" in unidad and abs(valor) >= 1e9:
        v_txt, d_txt = f"{valor/1e9:,.1f}B", f"{delta/1e9:+,.1f}B"
    elif "USD" in unidad and abs(valor) >= 1e6:
        v_txt, d_txt = f"{valor/1e6:,.0f}M", f"{delta/1e6:+,.0f}M"
    elif unidad == "%":
        v_txt, d_txt = f"{valor:.1f}%", f"{delta:+.1f} pp"
    else:
        v_txt, d_txt = f"{valor:,.0f}", f"{delta:+,.0f}"

    clase = "up" if delta >= 0 else "down"
    flecha = "&#9650;" if delta >= 0 else "&#9660;"   # triángulos (no emoji)
    return (
        f'<div class="kpi"><div class="etq">{config.nombre_indicador(codigo)} '
        f'({int(ultimo["anio"])})</div>'
        f'<div class="val">{v_txt}</div>'
        f'<div class="delta {clase}">{flecha} {d_txt}</div></div>'
    )


cols = st.columns(4)
for col, codigo in zip(cols, ["pib_crecimiento", "inflacion", "desempleo", "canal_ingresos"]):
    with col:
        st.markdown(tarjeta_kpi(codigo), unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
tab_tend, tab_pred, tab_clust, tab_chat = st.tabs(
    ["Tendencias", "Predicciones", "Análisis", "Asistente"]
)

# ===== TENDENCIAS ==========================================================
with tab_tend:
    titulo_seccion("Evolución histórica de los indicadores",
                   "Selecciona indicadores y el rango de años a visualizar.")
    col_izq, col_der = st.columns([1, 3])

    with col_izq:
        seleccion = st.multiselect(
            "Indicadores",
            options=INDICADORES_DISPONIBLES,
            default=["pib_crecimiento", "inflacion", "desempleo"],
            format_func=config.nombre_indicador,
        )
        rango = st.slider("Rango de años", ANIO_MIN, ANIO_MAX, (ANIO_MIN, ANIO_MAX))
        normalizar = st.checkbox(
            "Normalizar (base 100)", value=False,
            help="Útil para comparar indicadores con escalas muy distintas.",
        )

    with col_der:
        if not seleccion:
            st.info("Selecciona al menos un indicador en el panel de la izquierda.")
        else:
            df_plot = ancho[(ancho["anio"] >= rango[0]) & (ancho["anio"] <= rango[1])].copy()
            largo_plot = df_plot.melt(
                id_vars="anio", value_vars=seleccion, var_name="codigo", value_name="valor",
            )
            if normalizar:
                base = largo_plot.sort_values("anio").groupby("codigo")["valor"].transform("first")
                largo_plot["valor"] = largo_plot["valor"] / base * 100
            largo_plot["Indicador"] = largo_plot["codigo"].map(config.nombre_indicador)

            fig = px.line(
                largo_plot, x="anio", y="valor", color="Indicador", markers=True,
                labels={"anio": "Año", "valor": "Valor" + (" (base 100)" if normalizar else "")},
            )
            fig.update_traces(line=dict(width=2.5))
            st.plotly_chart(estilizar_fig(fig), width="stretch")

    with st.expander("Ver tabla de datos"):
        st.dataframe(
            ancho.rename(columns={c: config.nombre_indicador(c) for c in INDICADORES_DISPONIBLES}),
            width="stretch",
        )

# ===== PREDICCIONES ========================================================
with tab_pred:
    titulo_seccion(
        "Pronóstico con modelos de Machine Learning",
        "Regresión con características de rezago; evaluación con partición temporal.",
    )
    objetivo = st.selectbox(
        "Indicador a pronosticar", options=list(resultados_ml.keys()),
        format_func=config.nombre_indicador,
    )
    res = resultados_ml[objetivo]
    df_pron, metr = res["pronostico"], res["evaluacion"]["metricas"]

    c1, c2, c3 = st.columns(3)
    for col, etq, val in zip(
        (c1, c2, c3),
        ("MAE · error medio absoluto", "RMSE · raíz del error cuadrático", "R² · bondad de ajuste"),
        (f"{metr['MAE']:.2f}", f"{metr['RMSE']:.2f}", f"{metr['R2']:.3f}"),
    ):
        col.markdown(
            f'<div class="kpi"><div class="etq">{etq}</div><div class="val">{val}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    hist = df_pron[df_pron["tipo"] == "histórico"]
    futu = df_pron[df_pron["tipo"] == "pronóstico"]
    futu_linea = pd.concat([hist.tail(1), futu], ignore_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["anio"], y=hist[objetivo], mode="lines+markers",
        name="Histórico", line=dict(color=ACENTO, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=futu_linea["anio"], y=futu_linea[objetivo], mode="lines+markers",
        name="Pronóstico", line=dict(color=ACENTO2, width=2.5, dash="dash"),
    ))
    fig.update_layout(xaxis_title="Año",
                      yaxis_title=f"{config.nombre_indicador(objetivo)} ({config.unidad_indicador(objetivo)})")
    st.plotly_chart(estilizar_fig(fig), width="stretch")

    tabla = futu[["anio", objetivo]].copy()
    tabla.columns = ["Año", config.nombre_indicador(objetivo)]
    st.dataframe(tabla.style.format({config.nombre_indicador(objetivo): "{:,.2f}"}),
                 width="stretch", hide_index=True)

# ===== ANÁLISIS / CLUSTERING ===============================================
with tab_clust:
    titulo_seccion(
        "Regímenes económicos (clustering KMeans)",
        f"Los años se agrupan por crecimiento, inflación, desempleo e IMAE. "
        f"Coeficiente de silueta: {clustering['silueta']:.3f}.",
    )
    asign = clustering["asignaciones"]
    mapa_color = {
        "Contracción / crisis": ROJO,
        "Crecimiento moderado": ACENTO2,
        "Expansión fuerte": VERDE,
    }

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig = px.scatter(
            asign, x="pib_crecimiento", y="desempleo", color="etiqueta",
            text="anio", size="imae", size_max=24, color_discrete_map=mapa_color,
            labels={"pib_crecimiento": "Crecimiento del PIB (%)", "desempleo": "Desempleo (%)",
                    "etiqueta": "Régimen"},
        )
        fig.update_traces(textposition="top center", textfont=dict(size=9, color=MUTED))
        st.plotly_chart(estilizar_fig(fig, 460), width="stretch")

    with col_b:
        st.markdown('<div class="sec-sub">Perfil promedio por régimen</div>', unsafe_allow_html=True)
        perfiles = clustering["perfiles"].rename(
            columns={c: config.nombre_indicador(c) for c in clustering["variables"]})
        st.dataframe(perfiles.drop(columns="cluster"), width="stretch", hide_index=True)

    st.markdown('<div class="sec-sub">Línea de tiempo de regímenes</div>', unsafe_allow_html=True)
    asign_tl = asign.assign(fila="Régimen")
    fig_tl = px.scatter(asign_tl, x="anio", y="fila", color="etiqueta",
                        color_discrete_map=mapa_color, labels={"anio": "Año", "fila": ""})
    fig_tl.update_traces(marker=dict(size=16))
    fig_tl.update_layout(yaxis=dict(showticklabels=False))
    st.plotly_chart(estilizar_fig(fig_tl, 200), width="stretch")

# ===== ASISTENTE / CHATBOT RAG =============================================
with tab_chat:
    titulo_seccion("Asistente con RAG sobre los indicadores",
                   "Recupera datos relevantes y genera la respuesta anclada en ellos.")
    nombres_modo = {
        "ollama": "Ollama &middot; IA local (gratis)",
        "claude": "Claude &middot; IA en la nube",
        "extractivo": "Extractivo &middot; sin modelo de lenguaje",
    }
    modo_actual = nombres_modo.get(chatbot.proveedor_activo(), chatbot.proveedor_activo())
    st.markdown(f'<span class="modo-badge">Motor: {modo_actual}</span>', unsafe_allow_html=True)
    st.write("")

    ejemplos = [
        "¿Cómo le fue al PIB de Panamá en 2020?",
        "¿Qué pasó con el desempleo durante la pandemia?",
        "¿Cuál es el pronóstico de la inflación?",
        "¿Cuánto aporta el Canal de Panamá?",
    ]
    if "pregunta_chat" not in st.session_state:
        st.session_state.pregunta_chat = ""
    cols_ej = st.columns(len(ejemplos))
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
        st.markdown(f'<div class="respuesta">{resultado["respuesta"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="margin-top:8px"><span class="modo-badge">Generado en modo: '
            f'{resultado["modo"]}</span></div>', unsafe_allow_html=True,
        )
        with st.expander("Fuentes recuperadas (RAG)"):
            for fuente in resultado["fuentes"]:
                st.markdown(f"- *(relevancia {fuente['score']:.2f})* {fuente['texto']}")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Acerca del proyecto")
    st.write(
        "Sistema de gestión de información que integra las técnicas del curso: "
        "pipeline de datos, preprocesamiento, Machine Learning, visualización y "
        "un asistente con RAG."
    )
    st.markdown("**Fuentes de datos**")
    st.markdown(
        "- API del Banco Mundial (macroeconomía)\n"
        "- Contraloría / INEC / Canal de Panamá"
    )
    st.markdown("**Cobertura**")
    st.write(f"{len(INDICADORES_DISPONIBLES)} indicadores · {ANIO_MIN}-{ANIO_MAX}")
    st.divider()
    st.caption("Universidad Tecnológica de Panamá · Gestión de la Información · 2026")
