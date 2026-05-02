import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from collections import Counter
from pysentimiento import create_analyzer
import google.generativeai as genai

nltk.download("stopwords", quiet=True)

st.set_page_config(
    page_title="Sistema Inteligente de Desempeño Docente",
    layout="wide"
)

st.title("📊 Sistema Inteligente de Desempeño Docente")
st.caption("Análisis cuantitativo y cualitativo de evaluación docente con NLP, modelo entrenado y recomendaciones de mejora.")

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_percepcion_docente.pkl")

@st.cache_resource
def cargar_robertuito():
    return create_analyzer(task="sentiment", lang="es")

modelo = cargar_modelo()
analyzer = cargar_robertuito()

comentarios_invalidos = {
    "ninguno", "ninguna", "ningun comentario", "ningún comentario",
    "sin comentarios", "sin comentario", "no aplica", "n/a", "na",
    "ok", "todo bien", "sin novedad", ".", "..", "...", "-", "--",
    "no hay comentarios", "ningun", "ningún", "ningun comentario", "ningún comentario"
}

palabras_excluir = set(stopwords.words("spanish") + [
    "docente", "profesor", "profesora", "estudiante", "estudiantes",
    "clase", "clases", "materia", "curso", "tema", "temas",
    "excelente", "bueno", "buena", "muy", "bien", "gracias",
    "ninguno", "ninguna", "ok", "maestro", "maestra", "profe",
    "jaime", "johanna", "johana", "marcelo", "santiago", "pablo", "lorena", "fabián", "fabian",
    "siempre", "sido", "ser", "hacer", "forma", "parte"
])


def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower().strip()
    texto = re.sub(r"[^a-záéíóúñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def es_valido(texto):
    texto = limpiar_texto(texto)
    if texto in comentarios_invalidos:
        return False
    if len(texto) < 6:
        return False
    if len(texto.split()) < 2:
        return False
    return True


def analizar_sentimiento_robertuito(texto):
    try:
        pred = analyzer.predict(texto)
        mapa = {"POS": "Positivo", "NEU": "Neutro", "NEG": "Negativo"}
        return mapa[pred.output], pred.probas[pred.output]
    except Exception:
        return "Neutro", 0.0


def leer_csv_robusto(archivo):
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            archivo.seek(0)
            return pd.read_csv(archivo, encoding=enc)
        except Exception:
            continue
    archivo.seek(0)
    return pd.read_csv(archivo)


archivo = st.sidebar.file_uploader("Sube la evaluación del docente", type=["csv"])
api_key = st.sidebar.text_input("Gemini API Key opcional", type="password")

if archivo is None:
    st.info("Sube un archivo CSV de evaluación docente para iniciar el análisis.")
    st.stop()

df_raw = leer_csv_robusto(archivo)
df_raw.columns = df_raw.columns.str.strip()

st.subheader("1. Vista inicial del archivo")
st.write("Este bloque permite verificar que el archivo fue cargado correctamente y que contiene las columnas esperadas del reporte institucional.")
st.dataframe(df_raw.head(), use_container_width=True)

COL_TIPO = "Q Type"
COL_RESPUESTA = "Answer"
COL_RESPUESTA_LIKERT = "Answer Match"
COL_RESPUESTAS = "# Responses"

if COL_TIPO not in df_raw.columns or COL_RESPUESTA not in df_raw.columns:
    st.error("El archivo no tiene las columnas esperadas: Q Type y Answer.")
    st.stop()

# =========================
# 2. ANÁLISIS LIKERT
# =========================
st.subheader("2. Evaluación cuantitativa (Likert)")
st.write(
    "Este bloque calcula la valoración cuantitativa del docente a partir de preguntas cerradas respondidas "
    "por estudiantes en escala Likert. Permite obtener una calificación global sobre 5 y detectar dimensiones "
    "fuertes o sensibles del desempeño."
)

mapa_likert = {
    "Totalmente de acuerdo": 5,
    "De acuerdo": 4,
    "Neutral": 3,
    "Ni de acuerdo ni en desacuerdo": 3,
    "En desacuerdo": 2,
    "Totalmente en desacuerdo": 1,
}

# En los archivos de evaluación, las preguntas Likert suelen venir como LIK.
df_qt = df_raw[df_raw[COL_TIPO].astype(str).str.upper().str.strip() == "LIK"].copy()
nota_global = np.nan

if not df_qt.empty and COL_RESPUESTA_LIKERT in df_qt.columns:
    df_qt["valor_likert"] = df_qt[COL_RESPUESTA_LIKERT].astype(str).str.strip().map(mapa_likert)
    df_qt = df_qt.dropna(subset=["valor_likert"])

    if not df_qt.empty:
        if COL_RESPUESTAS in df_qt.columns:
            df_qt[COL_RESPUESTAS] = pd.to_numeric(df_qt[COL_RESPUESTAS], errors="coerce").fillna(0)
            total_respuestas = df_qt[COL_RESPUESTAS].sum()
            if total_respuestas > 0:
                nota_global = (df_qt["valor_likert"] * df_qt[COL_RESPUESTAS]).sum() / total_respuestas
            else:
                nota_global = df_qt["valor_likert"].mean()
        else:
            nota_global = df_qt["valor_likert"].mean()

        c1, c2 = st.columns(2)
        c1.metric("Nota global docente", f"{nota_global:.2f} / 5")
        c2.metric("Respuestas Likert válidas", len(df_qt))

        st.info(
            "La nota global resume la percepción cuantitativa sobre el desempeño docente. "
            "Este indicador complementa el análisis cualitativo: ayuda a comparar lo que los estudiantes puntúan "
            "con lo que expresan en los comentarios abiertos."
        )

        # Gráfico por pregunta o dimensión
        dimension_col = None
        for posible in ["Question", "Question Text", "Pregunta", "Dimension", "dimension", "dimensión"]:
            if posible in df_qt.columns:
                dimension_col = posible
                break

        if dimension_col:
            if COL_RESPUESTAS in df_qt.columns and df_qt[COL_RESPUESTAS].sum() > 0:
                df_dim = (
                    df_qt.groupby(dimension_col)
                    .apply(lambda x: (x["valor_likert"] * x[COL_RESPUESTAS]).sum() / x[COL_RESPUESTAS].sum())
                    .reset_index(name="puntaje_promedio")
                    .sort_values("puntaje_promedio", ascending=True)
                )
            else:
                df_dim = (
                    df_qt.groupby(dimension_col)["valor_likert"]
                    .mean()
                    .reset_index(name="puntaje_promedio")
                    .sort_values("puntaje_promedio", ascending=True)
                )

            st.write("### Indicadores por dimensión/pregunta")
            fig_likert, ax_likert = plt.subplots(figsize=(9, max(4, len(df_dim) * 0.35)))
            sns.barplot(data=df_dim, x="puntaje_promedio", y=dimension_col, ax=ax_likert)
            ax_likert.set_xlim(0, 5)
            ax_likert.set_xlabel("Puntaje promedio sobre 5")
            ax_likert.set_ylabel("Dimensión / pregunta")
            ax_likert.set_title("Desempeño cuantitativo por dimensión")
            st.pyplot(fig_likert)

            peor_dim = df_dim.iloc[0]
            mejor_dim = df_dim.iloc[-1]
            st.info(
                f"Este gráfico permite identificar fortalezas y oportunidades de mejora. "
                f"La dimensión con mejor valoración es **{mejor_dim[dimension_col]}** ({mejor_dim['puntaje_promedio']:.2f}/5). "
                f"La dimensión con menor valoración es **{peor_dim[dimension_col]}** ({peor_dim['puntaje_promedio']:.2f}/5), "
                "por lo que puede considerarse como un foco de seguimiento."
            )

        if nota_global >= 4.5:
            interp_likert = "El docente presenta una valoración cuantitativa sobresaliente y consistente."
        elif nota_global >= 4.0:
            interp_likert = "El docente presenta una valoración favorable con oportunidades puntuales de mejora."
        elif nota_global >= 3.5:
            interp_likert = "El docente presenta una valoración intermedia; se recomienda seguimiento preventivo."
        else:
            interp_likert = "El docente presenta señales cuantitativas que requieren intervención académica."

        st.success(f"Interpretación cuantitativa: {interp_likert}")
    else:
        st.warning("Se encontraron filas Likert, pero las respuestas no coincidieron con el mapa de escala esperado.")
else:
    st.warning("No se encontraron respuestas Likert válidas en el archivo.")

# =========================
# 3. LIMPIEZA DE COMENTARIOS
# =========================
df_re = df_raw[df_raw[COL_TIPO].astype(str).str.upper().str.strip() == "RE"].copy()
df_re["comentario_limpio"] = df_re[COL_RESPUESTA].apply(limpiar_texto)
df_re["comentario_valido"] = df_re["comentario_limpio"].apply(es_valido)

comentarios_antes = len(df_re)
df_re = df_re[df_re["comentario_valido"]].copy()
comentarios_validos = len(df_re)
comentarios_eliminados = comentarios_antes - comentarios_validos

st.subheader("3. Limpieza de comentarios")
c1, c2, c3 = st.columns(3)
c1.metric("Comentarios abiertos", comentarios_antes)
c2.metric("Comentarios válidos", comentarios_validos)
c3.metric("Eliminados", comentarios_eliminados)

st.info(
    "Se eliminaron comentarios que no aportan valor analítico, como 'ninguno', 'N/A', 'ok', puntos o respuestas demasiado cortas. "
    "Esto evita que el modelo confunda ausencia de opinión con una percepción real del estudiante."
)

if comentarios_validos == 0:
    st.warning("No hay comentarios válidos para analizar.")
    st.stop()

# =========================
# 4. MODELO ENTRENADO
# =========================
st.subheader("4. Análisis de percepción con modelo entrenado")
df_re["percepcion_modelo"] = modelo.predict(df_re["comentario_limpio"])

conteo_modelo = df_re["percepcion_modelo"].value_counts()
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df_re, x="percepcion_modelo", order=["Positivo", "Neutro", "Negativo"], ax=ax)
ax.set_title("Distribución de percepción estudiantil")
ax.set_xlabel("Percepción")
ax.set_ylabel("Cantidad de comentarios")
st.pyplot(fig)

predominante = conteo_modelo.idxmax()
porcentaje_predominante = conteo_modelo.max() / conteo_modelo.sum() * 100
st.info(
    f"La percepción predominante es **{predominante}** con **{porcentaje_predominante:.1f}%** de los comentarios válidos. "
    "Este resultado muestra cómo están reaccionando los estudiantes frente al desempeño docente a partir del lenguaje utilizado."
)

# =========================
# 5. ROBERTUITO
# =========================
st.subheader("5. Análisis de sentimiento con RoBERTuito")
resultados_robertuito = df_re["comentario_limpio"].apply(analizar_sentimiento_robertuito)
df_re["sentimiento_robertuito"] = [r[0] for r in resultados_robertuito]
df_re["confianza_robertuito"] = [r[1] for r in resultados_robertuito]

fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.countplot(data=df_re, x="sentimiento_robertuito", order=["Positivo", "Neutro", "Negativo"], ax=ax2)
ax2.set_title("Distribución de sentimiento con RoBERTuito")
ax2.set_xlabel("Sentimiento")
ax2.set_ylabel("Cantidad")
st.pyplot(fig2)

st.info(
    "RoBERTuito funciona como una segunda capa de validación semántica. Permite contrastar la percepción clasificada por el modelo entrenado "
    "con un modelo NLP preentrenado en español. Si ambos coinciden, la señal interpretativa es más fuerte."
)

# =========================
# 6. PALABRAS FRECUENTES
# =========================
st.subheader("6. Palabras relevantes más frecuentes")
texto_total = " ".join(df_re["comentario_limpio"].tolist())
palabras = texto_total.split()
palabras_filtradas = [p for p in palabras if p not in palabras_excluir and len(p) > 3]

df_palabras = pd.DataFrame(Counter(palabras_filtradas).most_common(20), columns=["Palabra", "Frecuencia"])

if not df_palabras.empty:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_palabras, x="Frecuencia", y="Palabra", ax=ax3)
    ax3.set_title("Top palabras relevantes")
    st.pyplot(fig3)
    top_terms = ", ".join(df_palabras.head(5)["Palabra"].tolist())
    st.info(
        f"Las palabras más repetidas después de eliminar términos genéricos son: **{top_terms}**. "
        "Este bloque ayuda a identificar temas recurrentes, como metodología, retroalimentación, tareas, claridad, práctica, participación o tiempo."
    )
else:
    st.warning("No se encontraron palabras relevantes suficientes después de la limpieza.")

# =========================
# 7. COMENTARIOS ALERTA
# =========================
st.subheader("7. Comentarios que requieren revisión")
df_alerta = df_re[(df_re["percepcion_modelo"] == "Negativo") | (df_re["sentimiento_robertuito"] == "Negativo")].copy()

if not df_alerta.empty:
    st.dataframe(df_alerta[[COL_RESPUESTA, "percepcion_modelo", "sentimiento_robertuito", "confianza_robertuito"]].head(10), use_container_width=True)
    st.info(
        "Estos comentarios requieren revisión porque contienen señales de inconformidad, dificultad de comprensión, problemas de comunicación o necesidades de acompañamiento. "
        "No constituyen una decisión automática, sino una alerta para revisión académica."
    )
else:
    st.success("No se detectaron comentarios negativos prioritarios según las dos capas de análisis.")

# =========================
# 8. DIAGNÓSTICO
# =========================
st.subheader("8. Diagnóstico interpretativo")
porc_neg = (df_re["percepcion_modelo"].eq("Negativo").mean() * 100)
porc_neu = (df_re["percepcion_modelo"].eq("Neutro").mean() * 100)
porc_pos = (df_re["percepcion_modelo"].eq("Positivo").mean() * 100)

if porc_neg >= 30:
    nivel_alerta = "Alto"
    interpretacion = "Se recomienda revisión académica prioritaria y acompañamiento docente."
elif porc_neg >= 15 or porc_neu >= 40:
    nivel_alerta = "Medio"
    interpretacion = "Se recomienda acompañamiento pedagógico y seguimiento en la siguiente evaluación."
else:
    nivel_alerta = "Bajo"
    interpretacion = "Se recomienda seguimiento regular y mantenimiento de buenas prácticas."

nota_texto = "No disponible" if pd.isna(nota_global) else f"{nota_global:.2f}/5"
st.markdown(f"""
**Resumen del análisis**

- Nota cuantitativa Likert: **{nota_texto}**
- Percepción positiva: **{porc_pos:.1f}%**
- Percepción neutra: **{porc_neu:.1f}%**
- Percepción negativa: **{porc_neg:.1f}%**
- Nivel de alerta académica: **{nivel_alerta}**

**Interpretación:**  
{interpretacion}
""")

# =========================
# 9. GEMINI
# =========================
st.subheader("9. Recomendaciones con Gemini")

if api_key:
    try:
        genai.configure(api_key=api_key)

        modelos_disponibles = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-pro-latest",
        ]

        prompt = f"""
Actúa como experto en evaluación docente y analítica educativa.

Resultados del análisis:
- Nota cuantitativa Likert: {nota_texto}
- Comentarios válidos: {comentarios_validos}
- Comentarios eliminados: {comentarios_eliminados}
- Percepción positiva: {porc_pos:.1f}%
- Percepción neutra: {porc_neu:.1f}%
- Percepción negativa: {porc_neg:.1f}%
- Nivel de alerta: {nivel_alerta}
- Palabras frecuentes: {df_palabras.head(10).to_dict(orient="records") if not df_palabras.empty else []}
- Comentarios críticos: {df_alerta[COL_RESPUESTA].head(5).tolist() if not df_alerta.empty else []}

Genera:
1. Resumen ejecutivo.
2. Qué están reflejando los estudiantes.
3. Fortalezas del docente.
4. Aspectos específicos que debe mejorar.
5. 10 recomendaciones accionables.
6. Cierre ético indicando que esto apoya decisiones, pero no reemplaza revisión humana.
Usa lenguaje profesional, orientado a mejora continua y evita tono punitivo.
"""

        respuesta_texto = None
        modelo_activo = None
        ultimo_error = None

        for nombre_modelo in modelos_disponibles:
            try:
                model_ia = genai.GenerativeModel(nombre_modelo)
                response = model_ia.generate_content(prompt)
                respuesta_texto = response.text
                modelo_activo = nombre_modelo
                break
            except Exception as e:
                ultimo_error = e
                continue

        if respuesta_texto:
            st.success(f"Recomendaciones generadas con: {modelo_activo}")
            st.write(respuesta_texto)
        else:
            st.error(f"No se pudo generar recomendaciones con ningún modelo Gemini disponible. Último error: {ultimo_error}")

    except Exception as e:
        st.error(f"Error general con Gemini: {e}")
else:
    st.warning("Agrega tu Gemini API Key en la barra lateral si deseas generar recomendaciones automáticas.")

# =========================
# 10. DESCARGA
# =========================
st.subheader("10. Descargar resultados")
st.download_button(
    "Descargar resultados CSV",
    data=df_re.to_csv(index=False, encoding="utf-8-sig"),
    file_name="resultados_streamlit_docente.csv",
    mime="text/csv",
)
