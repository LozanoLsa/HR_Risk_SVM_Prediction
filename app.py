"""
HR Risk SVM Prediction — Streamlit Dashboard
LozanoLsa | Interactive decision-support tool
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HR Risk SVM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_PATH = "hr_risk_svm_data.csv"

NUM_FEATURES = [
    "training_hours_annual", "punctuality_ratio", "productivity_index",
    "scrap_associated_pct", "engagement_score", "years_experience",
    "area_rotation_rate",
]
CAT_FEATURES = ["area", "shift", "contract_type"]

AREAS     = ["Production", "Maintenance", "Logistics", "Quality", "Administration"]
SHIFTS    = ["Morning", "Evening", "Night"]
CONTRACTS = ["Permanent", "Temporary", "Outsourcing"]

PALETTE = {0: "#4C9BE8", 1: "#E8574C"}

# ─── Data loading ────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ─── Model training ──────────────────────────────────────────────────────────

@st.cache_resource
def train_model(use_gridsearch: bool = False):
    df = load_data()
    X = df.drop(columns=["high_risk"])
    y = df["high_risk"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    if use_gridsearch:
        param_grid = {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1, 10, 50],
            "clf__gamma": ["scale", 0.1, 0.01],
        }
        pipe = Pipeline(steps=[
            ("pre", preprocess),
            ("clf", SVC(probability=True, random_state=42)),
        ])
        grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        best_model = Pipeline(steps=[
            ("pre", preprocess),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ])
        best_model.fit(X_train, y_train)
        best_params = {"clf__kernel": "rbf", "clf__C": 10, "clf__gamma": "scale"}

    # Linear SVM for interpretability
    linear_model = Pipeline(steps=[
        ("pre", ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUM_FEATURES),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES),
            ]
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ])
    linear_model.fit(X_train, y_train)

    return best_model, linear_model, X_train, X_test, y_train, y_test, best_params


def get_feature_importance(linear_model):
    ohe = linear_model.named_steps["pre"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(CAT_FEATURES)
    feature_names = NUM_FEATURES + list(cat_names)
    coefs = linear_model.named_steps["clf"].coef_.ravel()
    importance = pd.DataFrame({"feature": feature_names, "coef": coefs})
    importance = importance.reindex(importance["coef"].abs().sort_values(ascending=False).index)
    return importance


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred),    4),
        "F1 Score":  round(f1_score(y_test, y_pred),        4),
    }, y_pred


# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://img.icons8.com/fluency/96/machine-learning.png",
    width=70,
)
st.sidebar.title("HR Risk SVM")
st.sidebar.markdown("Herramienta de apoyo para análisis de riesgo laboral")
st.sidebar.divider()

use_gs = st.sidebar.toggle(
    "GridSearchCV (más lento, óptimo)",
    value=False,
    help="Activa búsqueda exhaustiva de hiperparámetros. Puede tardar ~30 segundos.",
)

st.sidebar.divider()
st.sidebar.caption(
    "Este dashboard es una herramienta de **apoyo diagnóstico**, "
    "no un sistema de decisión automático.\n\n"
    "LozanoLsa · 2026"
)

# ─── Load data & model ───────────────────────────────────────────────────────

df = load_data()

with st.spinner("Entrenando modelo SVM…"):
    best_model, linear_model, X_train, X_test, y_train, y_test, best_params = train_model(use_gs)

metrics, y_pred = compute_metrics(best_model, X_test, y_test)
importance_df = get_feature_importance(linear_model)

# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🧠 Employee Risk Prediction — SVM Dashboard")
st.markdown(
    "Análisis sistémico de riesgo laboral usando Support Vector Machines. "
    "El riesgo emerge de **combinaciones** de factores, no de variables individuales."
)
st.divider()

# ─── KPI Strip ───────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
risk_rate = df["high_risk"].mean()
k1.metric("Total Empleados", f"{len(df):,}")
k2.metric("Alto Riesgo", f"{df['high_risk'].sum():,}", delta=f"{risk_rate:.1%} del total", delta_color="inverse")
k3.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
k4.metric("F1 Score",  f"{metrics['F1 Score']:.1%}")
k5.metric("Recall",    f"{metrics['Recall']:.1%}")

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploración de Datos",
    "🤖 Evaluación del Modelo",
    "🔮 Simulador de Escenarios",
    "📈 Importancia de Variables",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Exploración de Datos")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("**Distribución de Riesgo**")
        risk_counts = df["high_risk"].value_counts().rename({0: "Sin Riesgo", 1: "Alto Riesgo"})
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color_discrete_sequence=["#4C9BE8", "#E8574C"],
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("**Riesgo por Área, Turno y Contrato**")
        cat_sel = st.selectbox("Variable categórica", CAT_FEATURES, label_visibility="collapsed")
        risk_by_cat = df.groupby(cat_sel)["high_risk"].mean().sort_values(ascending=True) * 100
        fig_bar = px.bar(
            x=risk_by_cat.values,
            y=risk_by_cat.index,
            orientation="h",
            labels={"x": "% Alto Riesgo", "y": ""},
            color=risk_by_cat.values,
            color_continuous_scale=["#4C9BE8", "#E8574C"],
            range_color=[0, 60],
        )
        fig_bar.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.markdown("**Distribuciones de Variables Numéricas por Nivel de Riesgo**")

    num_sel = st.selectbox("Variable numérica", NUM_FEATURES)

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            df, x=num_sel, color="high_risk",
            color_discrete_map={0: "#4C9BE8", 1: "#E8574C"},
            barmode="overlay",
            opacity=0.7,
            nbins=30,
            labels={"high_risk": "Riesgo"},
        )
        fig_hist.update_layout(
            title=f"Distribución de {num_sel}",
            margin=dict(t=35, b=10, l=10, r=10),
            height=320,
            legend_title="Alto Riesgo",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(
            df, x="high_risk", y=num_sel,
            color="high_risk",
            color_discrete_map={0: "#4C9BE8", 1: "#E8574C"},
            labels={"high_risk": "Alto Riesgo"},
            points="outliers",
        )
        fig_box.update_layout(
            title=f"Boxplot de {num_sel}",
            margin=dict(t=35, b=10, l=10, r=10),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.markdown("**Mapa de Correlación (variables numéricas)**")
    corr = df[NUM_FEATURES + ["high_risk"]].corr()
    fig_corr = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_corr.update_layout(margin=dict(t=20, b=10, l=10, r=10), height=380)
    st.plotly_chart(fig_corr, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Evaluación del Modelo SVM")

    if use_gs:
        st.success(f"✅ GridSearchCV — Mejores parámetros: `{best_params}`")
    else:
        st.info("ℹ️ Modelo con parámetros fijos (RBF, C=10, gamma=scale). Activa GridSearchCV en el sidebar para optimización completa.")

    # Metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]
    for col, (k, v), c in zip([mc1, mc2, mc3, mc4], metrics.items(), colors):
        col.metric(k, f"{v:.1%}")

    st.divider()

    col_cm, col_rep = st.columns([1, 1])

    with col_cm:
        st.markdown("**Matriz de Confusión (test set)**")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicho", y="Real", color="Cantidad"),
            x=["Sin Riesgo", "Alto Riesgo"],
            y=["Sin Riesgo", "Alto Riesgo"],
            color_continuous_scale=["#EBF5FB", "#1A5276"],
            text_auto=True,
            aspect="equal",
        )
        fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_rep:
        st.markdown("**Reporte de Clasificación**")
        report = classification_report(y_test, y_pred, target_names=["Sin Riesgo", "Alto Riesgo"], output_dict=True)
        report_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore").round(3)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("**Distribución del conjunto de prueba**")
        test_dist = pd.Series(y_test).value_counts().rename({0: "Sin Riesgo", 1: "Alto Riesgo"})
        st.dataframe(test_dist.rename("Cantidad"), use_container_width=True)

    st.divider()
    st.markdown("**Probabilidad de riesgo predicha (test set)**")
    proba = best_model.predict_proba(X_test)[:, 1]
    fig_proba = px.histogram(
        x=proba,
        color=y_test.values.astype(str),
        color_discrete_map={"0": "#4C9BE8", "1": "#E8574C"},
        barmode="overlay",
        opacity=0.7,
        nbins=30,
        labels={"x": "Probabilidad de Alto Riesgo", "color": "Clase Real"},
    )
    fig_proba.update_layout(
        height=320,
        margin=dict(t=20, b=10, l=10, r=10),
        xaxis_title="P(Alto Riesgo)",
        legend_title="Clase Real",
    )
    st.plotly_chart(fig_proba, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIO SIMULATOR
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🔮 Simulador de Escenarios")
    st.markdown(
        "Ajusta los parámetros del empleado para explorar cómo cambia la predicción de riesgo. "
        "Útil para preguntas del tipo: *¿Qué pasa si aumentamos las horas de entrenamiento?*"
    )

    st.divider()

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        st.markdown("**Factores de Formación y Desempeño**")
        training_hours = st.slider(
            "Horas de entrenamiento anuales", 0, 60, 30,
            help="Rango en el dataset: 0.35 – 59.14"
        )
        punctuality = st.slider(
            "Ratio de puntualidad", 0.80, 1.00, 0.95, step=0.01,
            format="%.2f"
        )
        productivity = st.slider(
            "Índice de productividad", 70, 130, 100,
            help="Rango: 69 – 128"
        )
        engagement = st.select_slider(
            "Engagement score", options=[1, 2, 3, 4, 5], value=3,
            help="1 = muy bajo, 5 = muy alto"
        )

    with sim_col2:
        st.markdown("**Factores Operacionales y Experiencia**")
        scrap = st.slider(
            "Scrap / Retrabajo (%)", 0.0, 12.0, 5.0, step=0.1,
            format="%.1f%%"
        )
        experience = st.slider(
            "Años de experiencia", 0, 30, 10
        )
        rotation = st.slider(
            "Tasa de rotación del área", 0.00, 0.30, 0.10, step=0.01,
            format="%.2f"
        )

    with sim_col3:
        st.markdown("**Factores Estructurales**")
        area     = st.selectbox("Área funcional", AREAS)
        shift    = st.selectbox("Turno", SHIFTS)
        contract = st.selectbox("Tipo de contrato", CONTRACTS)

    st.divider()

    profile = {
        "training_hours_annual": training_hours,
        "punctuality_ratio":     punctuality,
        "productivity_index":    float(productivity),
        "scrap_associated_pct":  scrap,
        "engagement_score":      float(engagement),
        "years_experience":      float(experience),
        "area_rotation_rate":    rotation,
        "area":                  area,
        "shift":                 shift,
        "contract_type":         contract,
    }

    X_sim = pd.DataFrame([profile])
    pred_class = int(best_model.predict(X_sim)[0])
    pred_prob  = float(best_model.predict_proba(X_sim)[0][1])

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if pred_class == 1:
            st.error(f"### ⚠️ ALTO RIESGO\nProbabilidad: **{pred_prob:.1%}**")
        else:
            st.success(f"### ✅ SIN RIESGO\nProbabilidad de riesgo: **{pred_prob:.1%}**")

        st.markdown(f"**Confianza del modelo:** `{max(pred_prob, 1 - pred_prob):.1%}`")

    with res_col2:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": risk_rate * 100, "suffix": "% (promedio dataset)"},
            title={"text": "Probabilidad de Alto Riesgo"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#E8574C" if pred_class == 1 else "#4C9BE8"},
                "steps": [
                    {"range": [0,  40], "color": "#D5EFDF"},
                    {"range": [40, 65], "color": "#FCF3CF"},
                    {"range": [65, 100], "color": "#FADBD8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8,
                    "value": risk_rate * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Comparison table
    st.divider()
    st.markdown("**Perfil actual vs. referencias**")
    ref_low = {
        "training_hours_annual": 50, "punctuality_ratio": 0.99,
        "productivity_index": 110, "scrap_associated_pct": 2,
        "engagement_score": 5, "years_experience": 10,
        "area_rotation_rate": 0.05, "area": "Quality",
        "shift": "Morning", "contract_type": "Permanent",
    }
    ref_high = {
        "training_hours_annual": 5, "punctuality_ratio": 0.80,
        "productivity_index": 80, "scrap_associated_pct": 10,
        "engagement_score": 1, "years_experience": 1,
        "area_rotation_rate": 0.28, "area": "Production",
        "shift": "Night", "contract_type": "Outsourcing",
    }
    prob_low  = float(best_model.predict_proba(pd.DataFrame([ref_low]))[0][1])
    prob_high = float(best_model.predict_proba(pd.DataFrame([ref_high]))[0][1])

    comp_df = pd.DataFrame({
        "": ["Referencia Bajo Riesgo", "Perfil Actual", "Referencia Alto Riesgo"],
        "P(Alto Riesgo)": [f"{prob_low:.1%}", f"{pred_prob:.1%}", f"{prob_high:.1%}"],
        "Predicción": [
            "✅ Sin Riesgo",
            "⚠️ Alto Riesgo" if pred_class == 1 else "✅ Sin Riesgo",
            "⚠️ Alto Riesgo",
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Importancia de Variables — Linear SVM")
    st.markdown(
        "Los coeficientes del SVM lineal indican qué variables empujan la predicción "
        "hacia **alto riesgo** (positivos) o **sin riesgo** (negativos). "
        "Esto es un diagnóstico direccional, no causalidad."
    )

    n_features = st.slider("Número de variables a mostrar", 5, len(importance_df), 15)
    top_n = importance_df.head(n_features).copy()

    fig_imp = px.bar(
        top_n,
        x="coef",
        y="feature",
        orientation="h",
        color="coef",
        color_continuous_scale=["#4C9BE8", "#ECEFF1", "#E8574C"],
        color_continuous_midpoint=0,
        labels={"coef": "Coeficiente SVM", "feature": "Variable"},
    )
    fig_imp.add_vline(x=0, line_color="black", line_width=1)
    fig_imp.update_layout(
        height=max(350, n_features * 28),
        margin=dict(t=20, b=20, l=20, r=20),
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("**🔴 Top factores de riesgo** (coef. positivos)")
        top_risk = importance_df[importance_df["coef"] > 0].head(8)
        st.dataframe(
            top_risk[["feature", "coef"]].rename(columns={"feature": "Variable", "coef": "Coeficiente"}),
            use_container_width=True,
            hide_index=True,
        )

    with col_neg:
        st.markdown("**🔵 Top factores protectores** (coef. negativos)")
        top_safe = importance_df[importance_df["coef"] < 0].tail(8)
        st.dataframe(
            top_safe[["feature", "coef"]].rename(columns={"feature": "Variable", "coef": "Coeficiente"}),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("**Dataset completo**")
    with st.expander("Ver datos crudos"):
        col_filter = st.selectbox("Filtrar por riesgo", ["Todos", "Alto Riesgo (1)", "Sin Riesgo (0)"])
        df_show = df.copy()
        if col_filter == "Alto Riesgo (1)":
            df_show = df_show[df_show["high_risk"] == 1]
        elif col_filter == "Sin Riesgo (0)":
            df_show = df_show[df_show["high_risk"] == 0]
        st.dataframe(df_show, use_container_width=True)
        st.caption(f"{len(df_show):,} registros mostrados")
