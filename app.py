"""
app.py — HR Employee Risk Analytics Dashboard
LozanoLsa · Project 04 · Support Vector Machine · 2026
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SVM · HR Risk Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --blue:     #3b82f6;
    --blue2:    #60a5fa;
    --teal:     #2dd4bf;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--muted) !important; letter-spacing: 0.05em; }
[data-testid="stSidebar"] label { font-family: var(--fm) !important; font-size: 0.7rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--blue) !important; border: 2px solid var(--blue2) !important; box-shadow: 0 0 8px rgba(59,130,246,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--blue2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--blue) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--blue) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--blue2) !important; line-height: 1.1 !important; }
[data-testid="stMetricDelta"] > div { font-family: var(--fm) !important; font-size: 0.68rem !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--blue2) !important; background: rgba(59,130,246,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--blue) !important; border-bottom: 2px solid var(--blue) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; letter-spacing: 0.04em !important; border: none !important; }

[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; margin-bottom: 6px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; }
[data-testid="stExpander"] p { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--muted) !important; line-height: 1.7 !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }

h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; letter-spacing: -0.01em !important; }
h2 { font-size: 1.1rem !important; font-weight: 700 !important; }
h3 { font-size: 0.9rem !important; font-weight: 600 !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--blue); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); color: var(--blue2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH    = "hr_risk_svm_data.csv"
RANDOM_STATE = 42
NUM_COLS = ["training_hours_annual", "punctuality_rate", "productivity_index",
            "scrap_associated_pct", "engagement_score", "experience_yrs", "area_rotation_rate"]
CAT_COLS = ["department", "shift", "contract_type"]
TARGET   = "high_risk"

FEAT_LABELS = {
    "training_hours_annual":     "Training Hours / Year",
    "punctuality_rate":          "Punctuality Rate",
    "productivity_index":        "Productivity Index",
    "scrap_associated_pct":      "Scrap Associated (%)",
    "engagement_score":          "Engagement Score",
    "experience_yrs":            "Experience (yrs)",
    "area_rotation_rate":        "Area Rotation Rate",
    "department_Logistics":      "Dept: Logistics",
    "department_Maintenance":    "Dept: Maintenance",
    "department_Production":     "Dept: Production",
    "department_Quality":        "Dept: Quality",
    "shift_Afternoon":           "Shift: Afternoon",
    "shift_Night":               "Shift: Night",
    "contract_type_Outsourcing": "Contract: Outsourcing",
    "contract_type_Temporary":   "Contract: Temporary",
}

METRIC_EXPL = {
    "Accuracy":  "Out of every 100 employees, the model classifies this many correctly.",
    "Precision": "When the model flags someone as high-risk, how often they actually are.",
    "Recall":    "Out of all truly high-risk employees, how many the model catches.",
    "F1 Score":  "Balances precision and recall — the right metric with class imbalance.",
    "AUC-ROC":   "How well the model ranks high-risk above low-risk employees across all thresholds.",
}

ACTION_MAP = {
    "engagement_score":     "Schedule engagement conversation with direct manager — explore career path and workload",
    "scrap_associated_pct": "Review workstation quality conditions — may signal skill gaps, equipment, or ergonomic issues",
    "punctuality_rate":     "Discuss attendance patterns — check for commute, personal, or workload factors",
    "training_hours_annual":"Build a training plan for the next quarter — skills investment reduces risk",
    "productivity_index":   "Review work assignments and support resources — low productivity may indicate misalignment",
    "area_rotation_rate":   "Evaluate team stability and leadership in this department",
    "contract_type":        "Review contract conditions — temporary and outsourcing workers carry structural risk",
    "shift":                "Evaluate night-shift conditions — rotation, rest patterns, and support available",
}

# ─── PLOTLY COLORS ────────────────────────────────────────────────────────────
C_BLUE   = "#3b82f6"
C_BLUE2  = "#60a5fa"
C_TEAL   = "#2dd4bf"
C_DANGER = "#f87171"
C_WARN   = "#fbbf24"
C_MUTED  = "#4e6a8a"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color=C_MUTED, size=10),
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zeroline=False),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zeroline=False),
    margin=dict(l=4, r=4, t=40, b=4),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

def plot(fig, h=300):
    fig.update_layout(height=h, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv("https://raw.githubusercontent.com/LozanoLsa/HR_Risk_SVM_Prediction/main/hr_risk_svm_data.csv")

@st.cache_resource
def train_model(df):
    X, y = df.drop(TARGET, axis=1), df[TARGET]
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_COLS)
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                           random_state=RANDOM_STATE, stratify=y)
    pipe = Pipeline([("pre", prep), ("clf", SVC(probability=True, random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe,
                        {"clf__kernel": ["linear", "rbf"],
                         "clf__C": [0.1, 1, 10, 50],
                         "clf__gamma": ["scale", 0.1, 0.01]},
                        scoring="f1", cv=5, n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    yp, ypr = best.predict(Xte), best.predict_proba(Xte)[:, 1]
    metrics = {
        "Accuracy":  accuracy_score(yte, yp),
        "Precision": precision_score(yte, yp),
        "Recall":    recall_score(yte, yp),
        "F1 Score":  f1_score(yte, yp),
        "AUC-ROC":   roc_auc_score(yte, ypr),
    }
    lin = Pipeline([("pre", prep),
                    ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=5000))])
    lin.fit(Xtr, ytr)
    ohe     = lin.named_steps["pre"].named_transformers_["cat"]
    all_names = NUM_COLS + list(ohe.get_feature_names_out(CAT_COLS))
    coef_df = pd.DataFrame({"Feature": all_names,
                             "Coefficient": lin.named_steps["clf"].coef_.ravel()})
    return best, grid.best_params_, Xtr, Xte, ytr, yte, yp, ypr, metrics, coef_df

df = load_data()
best_svm, best_params, X_train, X_test, y_train, y_test, y_pred, y_prob, metrics, coef_df = train_model(df)
risk_rate = df[TARGET].mean()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="lsa-project-tag">LozanoLsa · Project 04</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
                color:#fff;margin-bottom:6px;">HR Risk<br>Predictor</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                color:#4e6a8a;line-height:1.7;margin-bottom:12px;">
        SVM trained on 1,247 employee records<br>7 numeric · 3 categorical features
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="lsa-section">// Behavioral</div>', unsafe_allow_html=True)
    engagement  = st.slider("Engagement Score (1–5)", 1, 5, 3)
    punctuality = st.slider("Punctuality Rate", 0.70, 1.00, 0.95, 0.01)

    st.markdown('<div class="lsa-section">// Operational</div>', unsafe_allow_html=True)
    productivity = st.slider("Productivity Index", 60.0, 130.0, 100.0, 1.0)
    scrap_pct    = st.slider("Scrap Associated (%)", 0.0, 20.0, 5.0, 0.5)
    training_hrs = st.slider("Training Hours / Year", 0.0, 80.0, 30.0, 1.0)
    exp_yrs      = st.slider("Experience (years)", 0, 30, 5)

    st.markdown('<div class="lsa-section">// Structural</div>', unsafe_allow_html=True)
    rotation   = st.slider("Area Rotation Rate", 0.00, 0.40, 0.10, 0.01)
    department = st.selectbox("Department", ["Production", "Quality", "Logistics", "Maintenance", "Administration"])
    shift      = st.selectbox("Shift", ["Morning", "Afternoon", "Night"])
    contract_t = st.selectbox("Contract Type", ["Permanent", "Temporary", "Outsourcing"])

    st.divider()
    st.caption(f"kernel={best_params.get('clf__kernel')} · C={best_params.get('clf__C')} · gamma={best_params.get('clf__gamma')}")
    st.caption("F1-optimized GridSearchCV · Where f(x) meets Kaizen · 2026")

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict_s(eng, punc, prod, scr, trn, exp, rot, dep, sh, ct):
    row = pd.DataFrame([{
        "training_hours_annual": trn, "punctuality_rate": punc,
        "productivity_index": prod, "scrap_associated_pct": scr,
        "engagement_score": eng, "experience_yrs": exp,
        "area_rotation_rate": rot, "department": dep,
        "shift": sh, "contract_type": ct
    }])
    p = best_svm.predict_proba(row)[0, 1]
    return p, int(p >= 0.5)

pred_prob, pred_class = predict_s(
    engagement, punctuality, productivity, scrap_pct,
    training_hrs, exp_yrs, rotation, department, shift, contract_t
)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #04 · Support Vector Machine · HR Analytics</div>
    <div class="lsa-title">Employee Risk Is Not Inevitable</div>
    <div class="lsa-tagline">A score is not a verdict. It is a signal — the start of a conversation, not the end of one.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">SVM · RBF / LINEAR</span>
        <span class="lsa-chip">10 FEATURES</span>
        <span class="lsa-chip">{metrics['Accuracy']:.1%} ACCURACY</span>
        <span class="lsa-chip">AUC {metrics['AUC-ROC']:.3f}</span>
        <span class="lsa-chip">F1-OPTIMIZED</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "SIMULATOR", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset overview</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Employees", f"{len(df):,}")
    k2.metric("High Risk",       f"{df[TARGET].sum():,}")
    k3.metric("Low Risk",        f"{(df[TARGET]==0).sum():,}")
    k4.metric("Risk Rate",       f"{risk_rate:.1%}")
    st.divider()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="lsa-section">// Risk class distribution</div>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=["Low Risk", "High Risk"],
            values=[(df[TARGET]==0).sum(), df[TARGET].sum()],
            marker_colors=[C_BLUE, C_DANGER],
            hole=0.52, textinfo="percent+label",
            textfont=dict(family="JetBrains Mono", size=11),
        ))
        plot(fig_pie, 270)

    with c2:
        st.markdown('<div class="lsa-section">// Risk rate by category</div>', unsafe_allow_html=True)
        cs = st.selectbox("Category:", CAT_COLS)
        r  = df.groupby(cs)[TARGET].mean().reset_index().sort_values(TARGET)
        fig_cat = go.Figure(go.Bar(
            x=r[TARGET], y=r[cs], orientation="h",
            marker_color=[C_DANGER if v > risk_rate else C_BLUE for v in r[TARGET]],
            text=[f"{v:.1%}" for v in r[TARGET]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fig_cat.update_layout(xaxis_tickformat=".0%")
        plot(fig_cat, 270)

    st.divider()
    st.markdown('<div class="lsa-section">// Feature distributions</div>', unsafe_allow_html=True)
    ns = st.selectbox("Numeric feature:", NUM_COLS,
                      format_func=lambda x: FEAT_LABELS.get(x, x))
    c3, c4 = st.columns(2)
    with c3:
        fig_hist = go.Figure()
        for cls, color, name in [(0, C_BLUE, "Low Risk"), (1, C_DANGER, "High Risk")]:
            fig_hist.add_trace(go.Histogram(
                x=df[df[TARGET]==cls][ns], name=name,
                marker_color=color, opacity=0.65, nbinsx=30,
            ))
        fig_hist.update_layout(barmode="overlay",
                               xaxis_title=FEAT_LABELS.get(ns, ns),
                               yaxis_title="Count")
        plot(fig_hist, 295)

    with c4:
        fig_box = px.box(
            df, x=df[TARGET].map({0: "Low Risk", 1: "High Risk"}), y=ns,
            color=df[TARGET].map({0: "Low Risk", 1: "High Risk"}),
            color_discrete_map={"Low Risk": C_BLUE, "High Risk": C_DANGER},
        )
        fig_box.update_layout(showlegend=False,
                              xaxis_title="",
                              yaxis_title=FEAT_LABELS.get(ns, ns))
        plot(fig_box, 295)

    st.divider()
    st.markdown('<div class="lsa-section">// Correlation with high risk</div>', unsafe_allow_html=True)
    df_enc2 = pd.get_dummies(df, drop_first=True)
    ct3     = df_enc2.corr()[[TARGET]].sort_values(TARGET, ascending=False)
    fig_corr = px.imshow(ct3.T, color_continuous_scale="RdBu_r",
                         zmin=-0.4, zmax=0.4, text_auto=".3f", aspect="auto")
    plot(fig_corr, 240)

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// Model metrics</div>', unsafe_allow_html=True)
    m_cols = st.columns(5)
    for i, (name, val) in enumerate(metrics.items()):
        m_cols[i].metric(name, f"{val:.3f}")
    st.divider()

    cm_arr = confusion_matrix(y_test, y_pred)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Confusion matrix</div>', unsafe_allow_html=True)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=["Pred: Low Risk", "Pred: High Risk"],
            y=["Actual: Low Risk", "Actual: High Risk"],
            colorscale=[[0, "#0a1525"], [1, C_BLUE]],
            text=cm_arr, texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono", size=18, color="#fff"),
            showscale=False,
        ))
        plot(fig_cm, 300)
        st.caption("Rows = actual class · Columns = predicted class · Diagonal = correct")

    with c2:
        st.markdown('<div class="lsa-section">// Predicted probability by true class</div>', unsafe_allow_html=True)
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Histogram(
            x=y_prob[y_test==0], name="Actual: Low Risk",
            marker_color=C_BLUE, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_trace(go.Histogram(
            x=y_prob[y_test==1], name="Actual: High Risk",
            marker_color=C_DANGER, opacity=0.65, nbinsx=30,
        ))
        fig_prob.add_vline(x=0.5, line_dash="dash", line_color="#fff", line_width=1,
                           annotation_text="threshold 0.5",
                           annotation_font=dict(family="JetBrains Mono", size=9, color=C_MUTED))
        fig_prob.update_layout(barmode="overlay",
                               xaxis_title="P(High Risk)",
                               yaxis_title="Count")
        plot(fig_prob, 300)

    st.divider()
    st.markdown('<div class="lsa-section">// ROC curve</div>', unsafe_allow_html=True)
    fpr_r, tpr_r, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_r, y=tpr_r, mode="lines",
        name=f"SVM (AUC = {metrics['AUC-ROC']:.3f})",
        line=dict(color=C_BLUE, width=2),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
        line=dict(color=C_MUTED, width=1, dash="dash"),
    ))
    fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    plot(fig_roc, 340)
    st.caption(f"SVM · kernel={best_params.get('clf__kernel')} · C={best_params.get('clf__C')} · "
               f"gamma={best_params.get('clf__gamma')} · 75/25 stratified · F1-optimized GridSearchCV")

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    for name, expl in METRIC_EXPL.items():
        with st.expander(f"{name}  —  {metrics[name]:.3f}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    left, right = st.columns([1, 2])
    with left:
        st.markdown('<div class="lsa-section">// High-risk probability</div>', unsafe_allow_html=True)
        gauge_color = C_DANGER if pred_class == 1 else C_TEAL
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            number={"suffix": "%", "font": {"size": 38, "family": "JetBrains Mono", "color": "#fff"}},
            title={"text": "P(High Risk)", "font": {"size": 11, "family": "JetBrains Mono", "color": C_MUTED}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickfont": {"size": 9, "family": "JetBrains Mono"},
                         "tickcolor": "#1e2d45"},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "#0e1420", "bordercolor": "#1e2d45",
                "steps": [
                    {"range": [0,  35], "color": "rgba(45,212,191,0.08)"},
                    {"range": [35, 65], "color": "rgba(251,191,36,0.08)"},
                    {"range": [65, 100], "color": "rgba(248,113,113,0.10)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fg.update_layout(height=290, paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fg, use_container_width=True)
        if pred_class == 1:
            st.error("HIGH RISK — SCHEDULE HR CONVERSATION")
        else:
            st.success("LOW RISK — STANDARD FOLLOW-UP")
        st.caption(f"Dataset avg: {risk_rate:.1%}  ·  This profile: {pred_prob:.1%}  ·  Δ {pred_prob - risk_rate:+.1%}")

    with right:
        st.markdown('<div class="lsa-section">// Linear SVM contribution by feature</div>', unsafe_allow_html=True)
        prep_obj  = best_svm.named_steps["pre"]
        row_df2   = pd.DataFrame([{
            "training_hours_annual": training_hrs, "punctuality_rate": punctuality,
            "productivity_index": productivity, "scrap_associated_pct": scrap_pct,
            "engagement_score": engagement, "experience_yrs": exp_yrs,
            "area_rotation_rate": rotation, "department": department,
            "shift": shift, "contract_type": contract_t
        }])
        row_t2    = prep_obj.transform(row_df2)
        ohe_obj   = prep_obj.named_transformers_["cat"]
        all_names2 = NUM_COLS + list(ohe_obj.get_feature_names_out(CAT_COLS))
        coef_vals2 = coef_df.set_index("Feature")["Coefficient"]
        contrib2   = coef_vals2 * pd.Series(row_t2[0], index=all_names2)
        top5       = contrib2.abs().nlargest(5).index
        c5         = contrib2[top5].reset_index()
        c5.columns = ["Feature", "Contribution"]
        c5["Label"] = c5["Feature"].map(lambda x: FEAT_LABELS.get(x, x.replace("_", " ").title()))
        fc2 = go.Figure(go.Bar(
            x=c5["Contribution"], y=c5["Label"], orientation="h",
            marker_color=[C_DANGER if v > 0 else C_TEAL for v in c5["Contribution"]],
            text=[f"{v:+.3f}" for v in c5["Contribution"]],
            textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
            textposition="outside",
        ))
        fc2.update_layout(xaxis_title="Linear SVM contribution (coef × scaled input)")
        plot(fc2, 270)
        if pred_prob >= 0.65:
            st.error("Priority: HIGH · 1:1 with HR + manager within 7 days")
        elif pred_prob >= 0.35:
            st.warning("Priority: MEDIUM · Follow-up review within 30 days")
        else:
            st.success("Priority: LOW · Quarterly check-in")

    st.divider()
    st.markdown('<div class="lsa-section">// Scenario comparison</div>', unsafe_allow_html=True)
    bp  = predict_s(5, 0.99, 110.0, 2.0, 50, 10, 0.05, "Quality",     "Morning", "Permanent")[0]
    wp  = predict_s(1, 0.78,  80.0, 10.0,  5,  1, 0.30, "Production",  "Night",   "Outsourcing")[0]
    cdf = pd.DataFrame([
        {"Scenario": "Best case — low-risk profile",  "P(High Risk)": f"{bp:.1%}",
         "Status": "Low Risk" if bp < 0.5 else "High Risk",
         "Δ vs current": f"{bp - pred_prob:+.1%}"},
        {"Scenario": "Your current profile",          "P(High Risk)": f"{pred_prob:.1%}",
         "Status": "Low Risk" if pred_class == 0 else "High Risk",
         "Δ vs current": "—"},
        {"Scenario": "Worst case — high-risk profile","P(High Risk)": f"{wp:.1%}",
         "Status": "Low Risk" if wp < 0.5 else "High Risk",
         "Δ vs current": f"{wp - pred_prob:+.1%}"},
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Linear SVM coefficients — risk drivers</div>', unsafe_allow_html=True)
    cs3 = coef_df.sort_values("Coefficient", ascending=True).copy()
    cs3["Label"] = cs3["Feature"].map(lambda x: FEAT_LABELS.get(x, x.replace("_", " ").title()))
    fig_coef = go.Figure(go.Bar(
        x=cs3["Coefficient"], y=cs3["Label"], orientation="h",
        marker_color=[C_TEAL if c < 0 else C_DANGER for c in cs3["Coefficient"]],
        text=[f"{c:+.3f}" for c in cs3["Coefficient"]],
        textfont=dict(family="JetBrains Mono", size=9, color="#c8d8f0"),
        textposition="outside",
    ))
    fig_coef.add_vline(x=0, line_color="#fff", line_width=1)
    fig_coef.update_layout(xaxis_title="Linear SVM coefficient (scaled features)")
    plot(fig_coef, 480)
    st.caption("Red = increases high-risk probability · Teal = reduces risk · Coefficients from LinearSVC trained on same pipeline.")

    st.divider()
    st.markdown('<div class="lsa-section">// Risk driver weight by factor category</div>', unsafe_allow_html=True)

    def cat_fn(n):
        if n in ["training_hours_annual", "punctuality_rate", "productivity_index",
                 "engagement_score", "experience_yrs"]:
            return "Behavioral / Performance"
        if n == "scrap_associated_pct":
            return "Quality"
        return "Structural / Context"

    cd3       = coef_df.copy()
    cd3["Cat"] = cd3["Feature"].apply(cat_fn)
    cd3["Abs"] = cd3["Coefficient"].abs()
    ci2       = cd3.groupby("Cat")["Abs"].sum().reset_index().sort_values("Abs", ascending=False)
    fig_cat2  = go.Figure(go.Bar(
        x=ci2["Cat"], y=ci2["Abs"],
        marker_color=[C_DANGER, C_BLUE, C_WARN][:len(ci2)],
        text=[f"{v:.3f}" for v in ci2["Abs"]],
        textfont=dict(family="JetBrains Mono", size=10, color="#c8d8f0"),
        textposition="outside",
    ))
    fig_cat2.update_layout(xaxis_title="", yaxis_title="Sum of |Coefficients|")
    plot(fig_cat2, 300)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="lsa-section">// Top risk factors (pushing risk up)</div>', unsafe_allow_html=True)
        risk_df = coef_df[coef_df["Coefficient"] > 0].sort_values("Coefficient", ascending=False).copy()
        risk_df["Label"] = risk_df["Feature"].map(lambda x: FEAT_LABELS.get(x, x.replace("_"," ").title()))
        st.dataframe(risk_df[["Label", "Coefficient"]].rename(columns={"Label": "Feature"}).round(4),
                     hide_index=True, use_container_width=True)
    with c2:
        st.markdown('<div class="lsa-section">// Protective factors (reducing risk)</div>', unsafe_allow_html=True)
        prot_df = coef_df[coef_df["Coefficient"] < 0].sort_values("Coefficient").copy()
        prot_df["Label"] = prot_df["Feature"].map(lambda x: FEAT_LABELS.get(x, x.replace("_"," ").title()))
        st.dataframe(prot_df[["Label", "Coefficient"]].rename(columns={"Label": "Feature"}).round(4),
                     hide_index=True, use_container_width=True)

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    if pred_prob >= 0.65:
        pl, hz, ac = "HIGH",   "7 days",         "1:1 with HR + direct manager — explore engagement, workload, and development plan"
    elif pred_prob >= 0.35:
        pl, hz, ac = "MEDIUM", "30 days",         "Follow-up review — check in on key factors and set short-term goals"
    else:
        pl, hz, ac = "LOW",    "Quarterly review", "Standard check-in — no urgent action required"

    badge_color = {"HIGH": C_DANGER, "MEDIUM": C_WARN, "LOW": C_TEAL}[pl]
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {badge_color};
                padding:1.2rem 1.4rem;border-radius:2px;margin-bottom:1rem;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px;">// Action plan</div>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;">
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Priority</div>
                <div style="font-family:var(--fh);font-size:1.3rem;font-weight:800;color:{badge_color};">{pl}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Est. Risk Prob</div>
                <div style="font-family:var(--fm);font-size:1.3rem;font-weight:600;color:#fff;">{pred_prob:.1%}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Horizon</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">{hz}</div>
            </div>
            <div>
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.12em;">Owner</div>
                <div style="font-family:var(--fm);font-size:0.85rem;color:var(--text);">HR Business Partner + Direct Manager</div>
            </div>
        </div>
        <div style="margin-top:12px;font-family:var(--fm);font-size:0.72rem;color:var(--text);">
            <span style="color:var(--muted);">Action → </span>{ac}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="lsa-section">// Active risk factors</div>', unsafe_allow_html=True)
    rfs = []
    if engagement  <= 2:                        rfs.append("engagement_score")
    if scrap_pct   >  7:                        rfs.append("scrap_associated_pct")
    if punctuality <  0.85:                     rfs.append("punctuality_rate")
    if training_hrs < 15:                       rfs.append("training_hours_annual")
    if productivity < 85:                       rfs.append("productivity_index")
    if rotation    >  0.20:                     rfs.append("area_rotation_rate")
    if contract_t in ["Temporary","Outsourcing"]: rfs.append("contract_type")
    if shift == "Night":                        rfs.append("shift")

    if rfs:
        for f in rfs[:4]:
            if f in ACTION_MAP:
                label = FEAT_LABELS.get(f, f.replace("_", " ").title())
                with st.expander(f"▲  {label}  —  active risk factor"):
                    st.write(ACTION_MAP[f])
    else:
        st.success("No elevated risk factors detected in the current profile.")

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid #4e6a8a;
                padding:0.9rem 1.2rem;border-radius:2px;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Important note</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--muted);line-height:1.7;">
            This tool supports HR decisions — it does not replace managerial judgment or formal HR processes.
            A high score flags an employee for a conversation, not a disciplinary action.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · HR Risk Predictor · Project 04 · v2.0
</div>
""", unsafe_allow_html=True)
