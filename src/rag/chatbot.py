"""
chatbot.py  —  MÓDULO 5: CHATBOT CON RAG (Retrieval-Augmented Generation)
=========================================================================
Implementa un asistente conversacional que responde preguntas en lenguaje
natural sobre los indicadores económicos de Panamá, conectado a los datos del
proyecto.

¿Qué es RAG?
    RAG = "Generación Aumentada por Recuperación". En vez de pedirle al modelo de
    lenguaje que conteste de memoria (lo que puede llevar a inventar cifras), el
    sistema:
        1. RECUPERA (retrieval) los fragmentos de datos más relevantes para la
           pregunta, desde una base de conocimiento construida con NUESTROS datos.
        2. GENERA (generation) la respuesta con un modelo de lenguaje (Claude),
           pasándole esos fragmentos como contexto y pidiéndole que responda
           SOLO con base en ellos.
    Así las respuestas quedan "ancladas" (grounded) en datos reales y verificables.

Componentes:
    * Base de conocimiento: textos cortos generados a partir de los datasets
      procesados (valores por año, descripciones de indicadores, tendencias,
      pronósticos del modelo de ML, eventos económicos).
    * Recuperación: vectorización TF-IDF + similitud de coseno (scikit-learn).
      No requiere servicios externos ni claves; funciona totalmente offline.
    * Generación: Claude (modelo `claude-opus-4-8`) vía el SDK oficial de
      Anthropic. Si no hay clave de API (`ANTHROPIC_API_KEY`), el chatbot usa un
      modo EXTRACTIVO de respaldo que arma la respuesta con los fragmentos
      recuperados, de modo que el proyecto funciona siempre.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .. import config

# Lista mínima de palabras vacías en español para mejorar la recuperación TF-IDF.
STOPWORDS_ES = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "más",
    "pero", "sus", "le", "ya", "o", "este", "si", "porque", "esta", "entre",
    "cuando", "muy", "sin", "sobre", "tambien", "también", "me", "hasta", "hay",
    "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les",
    "ni", "contra", "fue", "cual", "cuanto", "cuánto", "cuanta", "es", "son",
    "tiene", "del", "año", "años", "panama", "panamá", "indicador", "indicadores",
    # Formas sin tilde (coinciden con strip_accents="unicode" del vectorizador).
    "ano", "anos",
]


# ===========================================================================
# CONSTRUCCIÓN DE LA BASE DE CONOCIMIENTO
# ===========================================================================
def construir_base_conocimiento(
    ancho: pd.DataFrame,
    resultados_ml: dict | None = None,
) -> list[dict]:
    """Genera la lista de documentos de texto sobre la cual se hace la búsqueda.

    Cada documento es un dict {'id', 'texto', 'tipo'}. Los tipos son:
      - 'descripcion' : qué mide cada indicador.
      - 'valor'       : el valor de un indicador en un año concreto.
      - 'resumen'     : estadísticas y tendencia de un indicador.
      - 'pronostico'  : la predicción del modelo de ML (si se provee).
      - 'contexto'    : hechos generales (fuentes, eventos económicos).

    Parameters
    ----------
    ancho : DataFrame ancho de indicadores (una col por indicador).
    resultados_ml : salida de `entrenar_indicadores` (opcional).
    """
    documentos: list[dict] = []
    columnas = [c for c in ancho.columns if c != "anio"]
    ancho = ancho.sort_values("anio").reset_index(drop=True)

    # 1) Descripción de cada indicador.
    for codigo in columnas:
        meta = config.META_POR_CODIGO.get(codigo, {})
        documentos.append({
            "id": f"desc::{codigo}",
            "tipo": "descripcion",
            "texto": (
                f"{meta.get('nombre', codigo)} ({meta.get('unidad', '')}), "
                f"fuente: {meta.get('fuente', '')}. {meta.get('descripcion', '')}"
            ),
        })

    # 2) Valor de cada indicador por año (solo años recientes para no saturar:
    #    desde 2010 en adelante, además del primero disponible).
    anios_clave = sorted(set(ancho["anio"]) & set(range(2010, config.ANIO_FIN + 1)))
    if not ancho.empty:
        anios_clave = sorted(set(anios_clave) | {int(ancho["anio"].min())})
    for _, fila in ancho[ancho["anio"].isin(anios_clave)].iterrows():
        anio = int(fila["anio"])
        for codigo in columnas:
            valor = fila[codigo]
            if pd.isna(valor):
                continue
            meta = config.META_POR_CODIGO.get(codigo, {})
            documentos.append({
                "id": f"valor::{codigo}::{anio}",
                "tipo": "valor",
                "texto": (
                    f"En {anio}, {meta.get('nombre', codigo)} de Panamá fue "
                    f"{_formato_valor(valor, codigo)} ({meta.get('unidad', '')})."
                ),
            })

    # 3) Resumen estadístico y tendencia de cada indicador.
    for codigo in columnas:
        serie = ancho[["anio", codigo]].dropna()
        if serie.empty:
            continue
        meta = config.META_POR_CODIGO.get(codigo, {})
        v_ini = serie.iloc[0][codigo]
        v_fin = serie.iloc[-1][codigo]
        a_ini, a_fin = int(serie.iloc[0]["anio"]), int(serie.iloc[-1]["anio"])
        idx_max = serie[codigo].idxmax()
        idx_min = serie[codigo].idxmin()
        tendencia = "subió" if v_fin > v_ini else "bajó"
        documentos.append({
            "id": f"resumen::{codigo}",
            "tipo": "resumen",
            "texto": (
                f"Resumen de {meta.get('nombre', codigo)} en Panamá ({a_ini}-{a_fin}): "
                f"pasó de {_formato_valor(v_ini, codigo)} en {a_ini} a "
                f"{_formato_valor(v_fin, codigo)} en {a_fin} (en general {tendencia}). "
                f"El valor máximo fue {_formato_valor(serie.loc[idx_max, codigo], codigo)} "
                f"en {int(serie.loc[idx_max, 'anio'])} y el mínimo "
                f"{_formato_valor(serie.loc[idx_min, codigo], codigo)} en "
                f"{int(serie.loc[idx_min, 'anio'])}."
            ),
        })

    # 4) Pronósticos del modelo de ML.
    if resultados_ml:
        for codigo, res in resultados_ml.items():
            df_pron = res["pronostico"]
            futuros = df_pron[df_pron["tipo"] == "pronóstico"]
            meta = config.META_POR_CODIGO.get(codigo, {})
            partes = [
                f"{int(r['anio'])}: {_formato_valor(r[codigo], codigo)}"
                for _, r in futuros.iterrows()
            ]
            m = res["evaluacion"]["metricas"]
            documentos.append({
                "id": f"pronostico::{codigo}",
                "tipo": "pronostico",
                "texto": (
                    f"Pronóstico del modelo para {meta.get('nombre', codigo)} de "
                    f"Panamá: {'; '.join(partes)}. "
                    f"(Calidad del modelo en prueba: R2={m['R2']:.2f}, RMSE={m['RMSE']:.2f}.)"
                ),
            })

    # 5) Contexto general y eventos económicos.
    documentos.extend([
        {
            "id": "contexto::fuentes",
            "tipo": "contexto",
            "texto": (
                "Los datos provienen de dos fuentes: la API del Banco Mundial "
                "(indicadores macro como PIB, inflación, desempleo e inversión "
                "extranjera) y archivos de la Contraloría General / INEC y la "
                "Autoridad del Canal de Panamá (tránsitos e ingresos del Canal e "
                "Índice Mensual de Actividad Económica)."
            ),
        },
        {
            "id": "contexto::covid",
            "tipo": "contexto",
            "texto": (
                "En 2020 la pandemia de COVID-19 provocó la mayor recesión de la "
                "historia reciente de Panamá: el PIB cayó cerca de 17.9%, el "
                "desempleo subió a alrededor de 18% y hubo deflación. La economía "
                "se recuperó con fuerza en 2021."
            ),
        },
        {
            "id": "contexto::mina_cobre",
            "tipo": "contexto",
            "texto": (
                "A finales de 2023 cerró la mina de cobre (Cobre Panamá), lo que "
                "desaceleró el crecimiento económico en 2024. Además, una sequía "
                "redujo los tránsitos por el Canal de Panamá en 2023-2024."
            ),
        },
    ])
    return documentos


def _formato_valor(valor: float, codigo: str) -> str:
    """Da formato legible a un valor según el tipo de indicador."""
    if pd.isna(valor):
        return "sin dato"
    unidad = config.unidad_indicador(codigo)
    if "USD" in unidad and abs(valor) >= 1e6:
        return f"{valor / 1e9:,.1f} mil millones USD" if abs(valor) >= 1e9 else f"{valor / 1e6:,.1f} millones USD"
    if unidad == "%":
        return f"{valor:.1f}%"
    if "buques" in unidad:
        return f"{valor:,.0f} buques"
    if "USD" in unidad:
        return f"{valor:,.0f} USD"
    return f"{valor:,.1f}"


# ===========================================================================
# CHATBOT RAG
# ===========================================================================
class ChatbotRAG:
    """Asistente RAG sobre los indicadores económicos de Panamá."""

    def __init__(
        self,
        ancho: pd.DataFrame | None = None,
        resultados_ml: dict | None = None,
    ) -> None:
        """Construye la base de conocimiento e indexa con TF-IDF.

        Si no se pasa `ancho`, intenta leer el dataset procesado del disco.
        """
        if ancho is None:
            ancho = pd.read_csv(config.DATASET_ANCHO)

        self.documentos = construir_base_conocimiento(ancho, resultados_ml)
        self.textos = [d["texto"] for d in self.documentos]

        # Vectorizador TF-IDF: convierte cada documento en un vector numérico que
        # pondera las palabras según su frecuencia/rareza. strip_accents normaliza
        # tildes; ngram_range incluye bigramas para captar frases.
        self.vectorizador = TfidfVectorizer(
            stop_words=STOPWORDS_ES,
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
        )
        self.matriz = self.vectorizador.fit_transform(self.textos)

        self.cliente_claude = self._crear_cliente_claude()

    # ---- Recuperación --------------------------------------------------
    def recuperar(self, pregunta: str, k: int | None = None) -> list[dict]:
        """Devuelve los `k` documentos más similares a la pregunta.

        Vectoriza la pregunta con el mismo TF-IDF y calcula la similitud de
        coseno contra todos los documentos. Devuelve los de mayor similitud.
        """
        if k is None:
            k = config.RAG_TOP_K
        vector_pregunta = self.vectorizador.transform([pregunta])
        similitudes = cosine_similarity(vector_pregunta, self.matriz)[0]
        indices = np.argsort(similitudes)[::-1][:k]
        return [
            {**self.documentos[i], "score": float(similitudes[i])}
            for i in indices
            if similitudes[i] > 0
        ]

    # ---- Generación ----------------------------------------------------
    def responder(self, pregunta: str, k: int | None = None) -> dict:
        """Responde una pregunta usando RAG.

        Returns
        -------
        dict con:
            'respuesta' -> texto de la respuesta.
            'fuentes'   -> lista de documentos recuperados (para citar).
            'modo'      -> 'claude' o 'extractivo'.
        """
        contexto = self.recuperar(pregunta, k)
        if not contexto:
            return {
                "respuesta": (
                    "No encontré información relacionada con esa pregunta en los "
                    "datos disponibles. Intenta preguntar sobre PIB, inflación, "
                    "desempleo, el Canal de Panamá u otro indicador."
                ),
                "fuentes": [],
                "modo": "extractivo",
            }

        if self.cliente_claude is not None:
            try:
                respuesta = self._generar_con_claude(pregunta, contexto)
                return {"respuesta": respuesta, "fuentes": contexto, "modo": "claude"}
            except Exception as exc:  # noqa: BLE001 - cae al modo de respaldo
                print(f"  [aviso] Falló la llamada a Claude ({exc}); uso modo extractivo.")

        respuesta = self._generar_extractivo(pregunta, contexto)
        return {"respuesta": respuesta, "fuentes": contexto, "modo": "extractivo"}

    def _generar_con_claude(self, pregunta: str, contexto: list[dict]) -> str:
        """Llama a Claude con el contexto recuperado (RAG real)."""
        bloques = "\n".join(f"- {d['texto']}" for d in contexto)
        sistema = (
            "Eres un asistente experto en economía panameña. Responde de forma "
            "clara, breve y en español. Usa ÚNICAMENTE la información del CONTEXTO "
            "que se te entrega; si el contexto no contiene la respuesta, dilo "
            "honestamente. No inventes cifras. Cita los años y valores relevantes."
        )
        prompt = (
            f"CONTEXTO (datos verificados de indicadores económicos de Panamá):\n"
            f"{bloques}\n\n"
            f"PREGUNTA DEL USUARIO: {pregunta}\n\n"
            f"Responde con base en el contexto anterior."
        )
        respuesta = self.cliente_claude.messages.create(
            model=config.MODELO_CLAUDE,
            max_tokens=config.MAX_TOKENS_RESPUESTA,
            system=sistema,
            messages=[{"role": "user", "content": prompt}],
        )
        # La respuesta es una lista de bloques de contenido; tomamos el texto.
        partes = [b.text for b in respuesta.content if getattr(b, "type", "") == "text"]
        return "\n".join(partes).strip()

    def _generar_extractivo(self, pregunta: str, contexto: list[dict]) -> str:
        """Modo de respaldo: arma la respuesta con los fragmentos recuperados.

        No usa ningún modelo de lenguaje; resume los documentos más relevantes.
        Garantiza que el chatbot funcione sin clave de API.
        """
        lineas = [f"• {d['texto']}" for d in contexto]
        return (
            "Según los datos disponibles sobre Panamá, esto es lo más relevante "
            f"para tu pregunta «{pregunta}»:\n" + "\n".join(lineas)
        )

    # ---- Utilidades ----------------------------------------------------
    @staticmethod
    def _crear_cliente_claude():
        """Crea el cliente de Anthropic si hay clave; si no, devuelve None."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            import anthropic

            return anthropic.Anthropic()
        except Exception:  # noqa: BLE001
            return None

    def usa_claude(self) -> bool:
        """Indica si el chatbot puede usar Claude (hay clave y SDK)."""
        return self.cliente_claude is not None


if __name__ == "__main__":
    bot = ChatbotRAG()
    print(f"Base de conocimiento: {len(bot.documentos)} documentos.")
    print(f"¿Usa Claude?: {bot.usa_claude()}")
    for pregunta in [
        "¿Cómo le fue al PIB de Panamá en 2020?",
        "¿Qué pasó con el desempleo durante la pandemia?",
        "¿Cuántos buques cruzan el Canal de Panamá?",
    ]:
        print("\nP:", pregunta)
        r = bot.responder(pregunta)
        print(f"[modo: {r['modo']}]")
        print(r["respuesta"])
