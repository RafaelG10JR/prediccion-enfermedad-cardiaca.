"""Prototipo académico: predicción de enfermedad cardíaca (Minería de Datos, tercer corte)."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
CAT_COLS = ["chest_pain_type", "gender", "residence_type", "smoking_status"]
TARGET = "heart_disease"
BP_MIN, BP_MAX = 50, 250  # misma regla de calidad usada en el notebook

INTEGRANTES = ["Rafael Gustavo Sierra Arrieta"] 

st.set_page_config(page_title="Predicción de enfermedad cardíaca", page_icon="🫀", layout="wide")


@st.cache_resource
def cargar_pipeline():
    return joblib.load(BASE / "pipeline_modelo.joblib")


@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE / "patient_dataset.csv")


@st.cache_data
def cargar_metricas():
    return json.loads((BASE / "metrics.json").read_text())


@st.cache_data
def cargar_muestra():
    return pd.read_csv(BASE / "muestra_prueba.csv", dtype={c: str for c in CAT_COLS})


pipe = cargar_pipeline()
df = cargar_datos()
metricas = cargar_metricas()


def rango(col):
    """Rango observado en el dataset, para limitar los campos del formulario."""
    return float(df[col].min()), float(df[col].max())


def validar(d):
    """Valida tipos, rangos y categorías. Devuelve una lista de errores."""
    errores = []
    for col in ["age", "cholesterol", "max_heart_rate", "bmi", "diabetes_pedigree"]:
        lo, hi = rango(col)
        v = d.get(col)
        if not isinstance(v, (int, float, np.integer, np.floating)) or not (lo <= v <= hi):
            errores.append(f"{col}: debe ser un número entre {lo:g} y {hi:g}.")
    bp = d.get("blood_pressure")
    if not isinstance(bp, (int, float)) or not (BP_MIN <= bp <= BP_MAX):
        errores.append(f"blood_pressure: debe estar entre {BP_MIN} y {BP_MAX} mm Hg.")
    for col in ["plasma_glucose", "skin_thickness", "insulin"]:
        v = d.get(col)
        if v is not None and not np.isnan(v):
            lo, hi = rango(col)
            if not (lo <= v <= hi):
                errores.append(f"{col}: debe estar entre {lo:g} y {hi:g} (o marcarse como no disponible).")
    if d.get("exercise_angina") not in (0, 1):
        errores.append("exercise_angina: debe ser 0 o 1.")
    if d.get("hypertension") not in (0, 1):
        errores.append("hypertension: debe ser 0 o 1.")
    if d.get("chest_pain_type") not in ("1", "2", "3", "4"):
        errores.append("chest_pain_type: categoría no válida.")
    if d.get("gender") not in ("0.0", "1.0", "Desconocido"):
        errores.append("gender: categoría no válida.")
    if d.get("residence_type") not in ("Urban", "Rural", "Desconocido"):
        errores.append("residence_type: categoría no válida.")
    if d.get("smoking_status") not in ("Smoker", "Non-Smoker", "Unknown"):
        errores.append("smoking_status: categoría no válida.")
    return errores


st.title("🫀 Predicción de enfermedad cardíaca")
st.warning("**Prototipo académico.** No sustituye el diagnóstico ni la decisión de un profesional de la salud.")

tab_proy, tab_eda, tab_pred, tab_met = st.tabs(
    ["Proyecto", "Análisis exploratorio", "Predicción", "Métricas y limitaciones"])

# ------------------------------------------------------------------ Proyecto
with tab_proy:
    st.header("Presentación del proyecto")
    st.markdown(
        "**Objetivo:** estimar, a partir de variables clínicas y demográficas, si un paciente "
        "presenta enfermedad cardíaca (`heart_disease`: 0 = no, 1 = sí) mediante un modelo de "
        "clasificación binaria.")
    st.markdown("**Fuente de datos:** archivo `patient_dataset.csv` "
                f"({len(df):,} registros, {df.shape[1]} columnas). *(Indica aquí el origen del dataset.)*")
    st.markdown("**Modelo final:** " + metricas["modelo"])
    st.markdown("**Integrantes:**")
    for n in INTEGRANTES:
        st.markdown(f"- {n}")
    st.caption("Curso de Minería de Datos — Tercer corte.")

# ----------------------------------------------------------------------- EDA
with tab_eda:
    st.header("Visualizaciones del EDA")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribución de la variable objetivo")
        conteo = df[TARGET].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(["Sin enfermedad (0)", "Con enfermedad (1)"], conteo.values, color="steelblue")
        ax.set_ylabel("Número de registros")
        st.pyplot(fig)

    with c2:
        st.subheader("Distribución de una variable numérica")
        num_cols = [c for c in df.select_dtypes("number").columns if c != TARGET]
        var = st.selectbox("Variable", num_cols, index=0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for clase, color in [(0, "steelblue"), (1, "tomato")]:
            ax.hist(df.loc[df[TARGET] == clase, var].dropna(), bins=25, alpha=0.55,
                    color=color, label=f"heart_disease = {clase}")
        ax.set_xlabel(var); ax.set_ylabel("Registros"); ax.legend()
        st.pyplot(fig)

    st.subheader("Valores faltantes por variable")
    faltantes = (df.isna().mean() * 100).round(2)
    faltantes = faltantes[faltantes > 0].sort_values()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(faltantes.index, faltantes.values, color="steelblue")
    ax.set_xlabel("% de valores faltantes")
    st.pyplot(fig)

# ---------------------------------------------------------------- Predicción
with tab_pred:
    st.header("Predicción para un paciente")

    with st.form("formulario"):
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Edad (años)", *map(int, rango("age")), value=50, step=1)
            gender = st.selectbox("Género (código del dataset)", ["0.0", "1.0", "Desconocido"],
                                  format_func=lambda x: {"0.0": "0", "1.0": "1"}.get(x, x))
            chest = st.selectbox("Tipo de dolor torácico", ["1", "2", "3", "4"],
                                 format_func=lambda x: f"Tipo {x}")
            bp = st.number_input("Presión arterial (mm Hg)", BP_MIN, BP_MAX, value=130, step=1)
            chol = st.number_input("Colesterol (mg/dL)", *map(int, rango("cholesterol")), value=200, step=1)
        with b:
            hr = st.number_input("Frecuencia cardíaca máxima (lpm)", *map(int, rango("max_heart_rate")),
                                 value=140, step=1)
            angina = st.selectbox("Angina por ejercicio", [0, 1], format_func=lambda x: "Sí" if x else "No")
            hyper = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x else "No")
            bmi_lo, bmi_hi = rango("bmi")
            bmi = st.number_input("IMC (kg/m²)", bmi_lo, bmi_hi, value=float(np.clip(25.0, bmi_lo, bmi_hi)),
                                  step=0.1)
            dpf_lo, dpf_hi = rango("diabetes_pedigree")
            dpf = st.number_input("Función de pedigrí de diabetes", dpf_lo, dpf_hi,
                                  value=float(np.clip(1.0, dpf_lo, dpf_hi)), step=0.01)
        with c:
            def opcional(label, col, defecto):
                lo, hi = rango(col)
                no_disp = st.checkbox(f"{label}: no disponible (se ignora el valor)", key=f"nd_{col}")
                v = st.number_input(label, lo, hi, value=float(np.clip(defecto, lo, hi)), step=1.0)
                return np.nan if no_disp else v

            glu = opcional("Glucosa en plasma (mg/dL)", "plasma_glucose", 120)
            skin = opcional("Grosor del pliegue cutáneo (mm)", "skin_thickness", 50)
            ins = opcional("Insulina (µU/mL)", "insulin", 130)
            res = st.selectbox("Tipo de residencia", ["Urban", "Rural", "Desconocido"])
            smk = st.selectbox("Estado de fumador", ["Smoker", "Non-Smoker", "Unknown"])
        enviar = st.form_submit_button("Predecir")

    if enviar:
        datos = {"age": age, "gender": gender, "chest_pain_type": chest, "blood_pressure": bp,
                 "cholesterol": chol, "max_heart_rate": hr, "exercise_angina": angina,
                 "plasma_glucose": glu, "skin_thickness": skin, "insulin": ins, "bmi": bmi,
                 "diabetes_pedigree": dpf, "hypertension": hyper,
                 "residence_type": res, "smoking_status": smk}
        errores = validar(datos)
        if errores:
            st.error("Entrada no válida:")
            for e in errores:
                st.write("• " + e)
        else:
            X_nuevo = pd.DataFrame([datos])[list(pipe.feature_names_in_)]
            pred = int(pipe.predict(X_nuevo)[0])
            st.subheader("Resultado")
            if pred == 1:
                st.error("Predicción: **posible enfermedad cardíaca (1)**")
            else:
                st.success("Predicción: **sin enfermedad cardíaca (0)**")
            if hasattr(pipe, "predict_proba"):
                p = pipe.predict_proba(X_nuevo)[0]
                st.write(f"Probabilidad de enfermedad: **{p[1]:.1%}**  |  de no enfermedad: **{p[0]:.1%}**")
                st.progress(float(p[1]))
            st.caption("Resultado de un modelo experimental; ver limitaciones en la pestaña de métricas.")

    st.divider()
    st.subheader("Verificación con un registro conocido del conjunto de prueba")
    muestra = cargar_muestra()
    idx = st.number_input("Fila de la muestra de prueba", 0, len(muestra) - 1, value=0, step=1)
    fila = muestra.iloc[[int(idx)]]
    X_fila = fila[list(pipe.feature_names_in_)]
    pred_app = int(pipe.predict(X_fila)[0])
    proba_app = float(pipe.predict_proba(X_fila)[0, 1])
    st.dataframe(fila.T.astype(str).rename(columns={fila.index[0]: "valor"}))
    col1, col2 = st.columns(2)
    col1.metric("Predicción en la app", pred_app, f"prob. = {proba_app:.4f}")
    col2.metric("Predicción en el notebook", int(fila["pred_notebook"].iloc[0]),
                f"prob. = {float(fila['proba_notebook'].iloc[0]):.4f}")
    if pred_app == int(fila["pred_notebook"].iloc[0]) and abs(proba_app - float(fila["proba_notebook"].iloc[0])) < 1e-4:
        st.success("✅ La app coincide con el notebook para este registro.")
    else:
        st.error("❌ La app NO coincide con el notebook. Revisar versiones y preprocesamiento.")
    st.caption(f"Etiqueta real del registro: {int(fila[TARGET].iloc[0])}")

# ------------------------------------------------------------------ Métricas
with tab_met:
    st.header("Métricas finales")
    t = metricas["test"]
    cols = st.columns(5)
    for col, (k, nombre) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"),
                                       ("recall", "Recall"), ("f1", "F1-score"), ("auc", "AUC-ROC")]):
        col.metric(nombre, f"{t[k]:.3f}")

    st.subheader("Validación cruzada (5 pliegues, solo entrenamiento)")
    cv = pd.DataFrame({k: {"media": v[0], "desviación": v[1]} for k, v in metricas["cv"].items()}).T.round(3)
    st.dataframe(cv)

    st.subheader("Matriz de confusión (conjunto de prueba)")
    cm = np.array(metricas["matriz_confusion"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred. 0", "Pred. 1"]); ax.set_yticklabels(["Real 0", "Real 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    st.pyplot(fig)

    st.subheader("Limitaciones")
    st.markdown(
        "- El rendimiento del modelo es cercano al azar (AUC ≈ 0,5): **no tiene capacidad predictiva útil**.\n"
        "- El conjunto de datos puede ser sintético o carecer de relación entre las variables y el diagnóstico.\n"
        "- Valores de presión arterial fuera de 50–250 mm Hg se trataron como faltantes durante el entrenamiento.\n"
        "- No fue validado clínicamente ni con datos externos.\n"
        "- Es un prototipo académico: no debe usarse para diagnosticar ni para tomar decisiones médicas.")
