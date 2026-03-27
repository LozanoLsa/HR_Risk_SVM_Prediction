"""
app.py — HR Employee Risk Analytics Dashboard
LozanoLsa · Operational Excellence · ML Portfolio · 2026
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
    f1_score, roc_auc_score, confusion_matrix, classification_report)

st.set_page_config(page_title="HR Risk Predictor", page_icon="👥",
                   layout="wide", initial_sidebar_state="expanded")

DATA_PATH = "hr_risk_svm_data.csv"
RANDOM_STATE = 42
NUM_COLS = ["training_hours_annual","punctuality_rate","productivity_index",
            "scrap_associated_pct","engagement_score","experience_yrs","area_rotation_rate"]
CAT_COLS = ["department","shift","contract_type"]
TARGET   = "high_risk"

st.markdown("""<style>
    .block-container{padding-top:1.5rem;}
    [data-testid="metric-container"]{background:#1E2130;border-radius:8px;
        padding:12px 16px;border-left:3px solid #4C9BE8;}
</style>""", unsafe_allow_html=True)

METRIC_EXPL = {
    "Accuracy":  "Out of every 100 employees, the model classifies this many correctly.",
    "Precision": "When the model flags someone as high-risk, how often they actually are.",
    "Recall":    "Out of all truly high-risk employees, how many the model catches.",
    "F1 Score":  "Balances precision and recall — right metric with class imbalance.",
    "AUC-ROC":   "How well the model ranks high-risk above low-risk employees.",
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

@st.cache_data
def load_data():
    try: return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return pd.read_csv("https://raw.githubusercontent.com/LozanoLsa/04-SVM-HR-Risk/main/hr_risk_svm_data.csv")

@st.cache_resource
def train_model(df):
    X, y = df.drop(TARGET,axis=1), df[TARGET]
    prep = ColumnTransformer([("num",StandardScaler(),NUM_COLS),
                               ("cat",OneHotEncoder(drop="first",sparse_output=False),CAT_COLS)])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=RANDOM_STATE,stratify=y)
    pipe = Pipeline([("pre",prep),("clf",SVC(probability=True,random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe,{"clf__kernel":["linear","rbf"],"clf__C":[0.1,1,10,50],
                               "clf__gamma":["scale",0.1,0.01]},scoring="f1",cv=5,n_jobs=-1)
    grid.fit(Xtr,ytr)
    best = grid.best_estimator_
    yp,ypr = best.predict(Xte), best.predict_proba(Xte)[:,1]
    metrics = {"Accuracy":accuracy_score(yte,yp),"Precision":precision_score(yte,yp),
               "Recall":recall_score(yte,yp),"F1 Score":f1_score(yte,yp),
               "AUC-ROC":roc_auc_score(yte,ypr)}
    lin = Pipeline([("pre",prep),("clf",LinearSVC(C=1.0,random_state=RANDOM_STATE,max_iter=5000))])
    lin.fit(Xtr,ytr)
    ohe = lin.named_steps["pre"].named_transformers_["cat"]
    all_names = NUM_COLS + list(ohe.get_feature_names_out(CAT_COLS))
    coef_df = pd.DataFrame({"Feature":all_names,"Coefficient":lin.named_steps["clf"].coef_.ravel()})
    return best, grid.best_params_, Xtr, Xte, ytr, yte, yp, ypr, metrics, coef_df

df = load_data()
best_svm,best_params,X_train,X_test,y_train,y_test,y_pred,y_prob,metrics,coef_df = train_model(df)
risk_rate = df[TARGET].mean()

with st.sidebar:
    st.markdown("## 👥 HR Risk Predictor")
    st.markdown("SVM model trained on 1,000 employee records. Estimates high-risk probability from behavioral, operational, and structural indicators.")
    st.divider()
    st.markdown("### Employee Profile")
    st.markdown("**Behavioral**")
    engagement   = st.slider("Engagement Score (1-5)",1,5,3)
    punctuality  = st.slider("Punctuality Rate",0.70,1.00,0.95,0.01)
    st.markdown("**Operational**")
    productivity = st.slider("Productivity Index",60.0,130.0,100.0,1.0)
    scrap_pct    = st.slider("Scrap Associated (%)",0.0,20.0,5.0,0.5)
    training_hrs = st.slider("Training Hours / Year",0.0,80.0,30.0,1.0)
    exp_yrs      = st.slider("Experience (years)",0,30,5)
    st.markdown("**Structural**")
    rotation     = st.slider("Area Rotation Rate",0.00,0.40,0.10,0.01)
    department   = st.selectbox("Department",["Production","Quality","Logistics","Maintenance","Administration"])
    shift        = st.selectbox("Shift",["Morning","Afternoon","Night"])
    contract_t   = st.selectbox("Contract Type",["Permanent","Temporary","Outsourcing"])
    st.divider()
    st.caption(f"Best: kernel={best_params.get('clf__kernel')}, C={best_params.get('clf__C')}, gamma={best_params.get('clf__gamma')}")
    st.caption("LozanoLsa · Operational Excellence · ML Portfolio · 2026")

def predict_s(eng,punc,prod,scr,trn,exp,rot,dep,sh,ct):
    row = pd.DataFrame([{"training_hours_annual":trn,"punctuality_rate":punc,
                          "productivity_index":prod,"scrap_associated_pct":scr,
                          "engagement_score":eng,"experience_yrs":exp,"area_rotation_rate":rot,
                          "department":dep,"shift":sh,"contract_type":ct}])
    p = best_svm.predict_proba(row)[0,1]
    return p, int(p>=0.5)

pred_prob,pred_class = predict_s(engagement,punctuality,productivity,scrap_pct,
                                  training_hrs,exp_yrs,rotation,department,shift,contract_t)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Data Explorer","📈 Model Performance",
    "🎯 Scenario Simulator","🔍 Risk Drivers","📋 Action Plan"])

with tab1:
    st.subheader("Dataset Overview")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Employees",f"{len(df):,}"); k2.metric("High Risk",f"{df[TARGET].sum():,}")
    k3.metric("Low Risk",f"{(df[TARGET]==0).sum():,}"); k4.metric("Risk Rate",f"{risk_rate:.1%}")
    st.divider()
    c1,c2 = st.columns([1,2])
    with c1:
        fp2 = go.Figure(go.Pie(labels=["Low Risk","High Risk"],
                               values=[(df[TARGET]==0).sum(),df[TARGET].sum()],
                               marker_colors=["#4C9BE8","#E8574C"],hole=0.45,textinfo="percent+label"))
        fp2.update_layout(title="Risk Class Distribution",showlegend=False,height=280,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fp2,use_container_width=True)
    with c2:
        cs = st.selectbox("Risk rate by:",CAT_COLS)
        r = df.groupby(cs)[TARGET].mean().reset_index().sort_values(TARGET)
        fb3 = px.bar(r,x=TARGET,y=cs,orientation="h",color=TARGET,
                     color_continuous_scale=["#4C9BE8","#E8574C"],
                     labels={TARGET:"Risk Rate",cs:""},title=f"Risk Rate by {cs.replace('_',' ').title()}")
        fb3.update_xaxes(tickformat=".0%")
        fb3.update_layout(coloraxis_showscale=False,height=270,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fb3,use_container_width=True)
    st.divider()
    ns = st.selectbox("Numeric feature:",NUM_COLS)
    c3,c4 = st.columns(2)
    with c3:
        fh = px.histogram(df,x=ns,color=TARGET,color_discrete_map={0:"#4C9BE8",1:"#E8574C"},
                          barmode="overlay",opacity=0.7,title=f"Distribution: {ns.replace('_',' ').title()}")
        fh.update_layout(height=295,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fh,use_container_width=True)
    with c4:
        fbx = px.box(df,x=df[TARGET].map({0:"Low Risk",1:"High Risk"}),y=ns,
                     color=df[TARGET].map({0:"Low Risk",1:"High Risk"}),
                     color_discrete_map={"Low Risk":"#4C9BE8","High Risk":"#E8574C"},
                     labels={"x":"","y":ns.replace("_"," ").title()},
                     title=f"By Class: {ns.replace('_',' ').title()}")
        fbx.update_layout(showlegend=False,height=295,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fbx,use_container_width=True)
    df_enc2 = pd.get_dummies(df,drop_first=True)
    ct3 = df_enc2.corr()[[TARGET]].sort_values(TARGET,ascending=False)
    fcr = px.imshow(ct3.T,color_continuous_scale="RdBu_r",zmin=-0.4,zmax=0.4,
                    text_auto=".3f",title="Correlation with High Risk",aspect="auto")
    fcr.update_layout(height=250,paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fcr,use_container_width=True)

with tab2:
    st.info(f"Tested on {len(X_test)} employees. Correctly classified {metrics['Accuracy']:.0%}. "
            "With 30.8% risk rate, F1 and Recall are the operationally meaningful metrics.")
    mc = st.columns(5)
    for col,(n,v) in zip(mc,metrics.items()):
        col.metric(n,f"{v:.1%}"); col.caption(METRIC_EXPL[n])
    st.divider()
    c1,c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test,y_pred)
        fcm = px.imshow(cm,text_auto=True,x=["Pred: Low Risk","Pred: High Risk"],
                        y=["True: Low Risk","True: High Risk"],
                        color_continuous_scale="Blues",title="Confusion Matrix")
        fcm.update_layout(height=360,coloraxis_showscale=False,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fcm,use_container_width=True)
        st.caption("Rows = actual · Columns = predicted · Diagonal = correct")
    with c2:
        fpr3 = go.Figure()
        fpr3.add_trace(go.Histogram(x=y_prob[y_test==0],name="Actual: Low Risk",
                                    marker_color="#4C9BE8",opacity=0.7,nbinsx=30))
        fpr3.add_trace(go.Histogram(x=y_prob[y_test==1],name="Actual: High Risk",
                                    marker_color="#E8574C",opacity=0.7,nbinsx=30))
        fpr3.add_vline(x=0.5,line_dash="dash",line_color="white")
        fpr3.update_layout(barmode="overlay",title="Predicted Probability by True Risk Class",
                           xaxis_title="P(High Risk)",height=360,
                           paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fpr3,use_container_width=True)
    rep = pd.DataFrame(classification_report(y_test,y_pred,
                       target_names=["Low Risk","High Risk"],output_dict=True)).T.round(3)
    st.dataframe(rep.style.background_gradient(cmap="Blues",subset=["precision","recall","f1-score"]),
                 use_container_width=True)
    st.caption(f"SVM Pipeline | kernel={best_params.get('clf__kernel')}, C={best_params.get('clf__C')}, "
               f"gamma={best_params.get('clf__gamma')} | 75/25 stratified | F1-optimized")

with tab3:
    st.subheader("Employee Risk Probability Simulator")
    l2,r2 = st.columns([1,2])
    with l2:
        fg2 = go.Figure(go.Indicator(
            mode="gauge+number",value=pred_prob*100,
            number={"suffix":"%","font":{"size":42}},
            title={"text":"Estimated High-Risk Probability","font":{"size":14}},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#E8574C" if pred_class==1 else "#4C9BE8"},
                   "steps":[{"range":[0,30],"color":"#1a2e1a"},{"range":[30,60],"color":"#2e2a1a"},
                             {"range":[60,100],"color":"#2e1a1a"}],
                   "threshold":{"line":{"color":"white","width":3},"thickness":0.75,"value":50}}))
        fg2.update_layout(height=310,paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=60,b=20,l=20,r=20))
        st.plotly_chart(fg2,use_container_width=True)
        if pred_class==1: st.error("**HIGH RISK — SCHEDULE HR CONVERSATION**")
        else: st.success("**LOW RISK — STANDARD FOLLOW-UP**")
        st.caption(f"Dataset avg: {risk_rate:.1%}  ·  This profile: {pred_prob:.1%}  ·  Δ {pred_prob-risk_rate:+.1%}")
    with r2:
        prep_obj = best_svm.named_steps["pre"]
        row_df2 = pd.DataFrame([{"training_hours_annual":training_hrs,"punctuality_rate":punctuality,
                                  "productivity_index":productivity,"scrap_associated_pct":scrap_pct,
                                  "engagement_score":engagement,"experience_yrs":exp_yrs,
                                  "area_rotation_rate":rotation,"department":department,
                                  "shift":shift,"contract_type":contract_t}])
        row_t2 = prep_obj.transform(row_df2)
        ohe_obj = prep_obj.named_transformers_["cat"]
        all_names2 = NUM_COLS + list(ohe_obj.get_feature_names_out(CAT_COLS))
        coef_vals2 = coef_df.set_index("Feature")["Coefficient"]
        contrib2 = coef_vals2 * pd.Series(row_t2[0],index=all_names2)
        top5 = contrib2.abs().nlargest(5).index
        c5 = contrib2[top5].reset_index(); c5.columns = ["Feature","Contribution"]
        fc2 = go.Figure(go.Bar(x=c5["Contribution"],
                               y=c5["Feature"].str.replace("_"," ").str.title(),
                               orientation="h",
                               marker_color=["#E8574C" if v>0 else "#4C9BE8" for v in c5["Contribution"]]))
        fc2.update_layout(title="What's Driving This Score (Top 5)",xaxis_title="Linear SVM contribution",
                          height=290,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=50,b=40,l=10,r=10))
        st.plotly_chart(fc2,use_container_width=True)
        if pred_prob>=0.65: st.error("**Priority: HIGH** · 1:1 with HR + manager within 7 days")
        elif pred_prob>=0.35: st.warning("**Priority: MEDIUM** · Follow-up review within 30 days")
        else: st.success("**Priority: LOW** · Quarterly check-in")
    st.divider()
    st.markdown("### Scenario Comparison")
    bp = predict_s(5,0.99,110.0,2.0,50,10,0.05,"Quality","Morning","Permanent")[0]
    wp = predict_s(1,0.78,80.0,10.0,5,1,0.30,"Production","Night","Outsourcing")[0]
    cmp2 = pd.DataFrame([
        {"Scenario":"Best case (low-risk profile)","P(High Risk)":f"{bp:.1%}",
         "Class":"Low Risk" if bp<0.5 else "High Risk","Delta vs current":f"{bp-pred_prob:+.1%}"},
        {"Scenario":"Current profile","P(High Risk)":f"{pred_prob:.1%}",
         "Class":"Low Risk" if pred_class==0 else "High Risk","Delta vs current":"—"},
        {"Scenario":"Worst case (high-risk profile)","P(High Risk)":f"{wp:.1%}",
         "Class":"Low Risk" if wp<0.5 else "High Risk","Delta vs current":f"{wp-pred_prob:+.1%}"},
    ])
    st.dataframe(cmp2,use_container_width=True,hide_index=True)

with tab4:
    st.subheader("What Variables Drive High-Risk Classification?")
    st.caption("Coefficients from a Linear SVM trained on the same data. Directional influence — not causation.")
    cs3 = coef_df.sort_values("Coefficient",ascending=True)
    fig_c2 = go.Figure(go.Bar(x=cs3["Coefficient"],
                              y=cs3["Feature"].str.replace("_"," ").str.title(),
                              orientation="h",
                              marker_color=["#4C9BE8" if c<0 else "#E8574C" for c in cs3["Coefficient"]],
                              text=cs3["Coefficient"].round(3),textposition="outside"))
    fig_c2.add_vline(x=0,line_color="white",line_width=1.5)
    fig_c2.update_layout(title="Feature Coefficients — Risk UP (Red) vs Protective (Blue)",
                         xaxis_title="Linear SVM Coefficient",height=500,
                         paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_c2,use_container_width=True)
    st.divider()
    def cat_fn(n):
        if n in ["training_hours_annual","punctuality_rate","productivity_index","engagement_score","experience_yrs"]: return "Behavioral / Performance"
        if n=="scrap_associated_pct": return "Quality"
        return "Structural / Context"
    cd3 = coef_df.copy(); cd3["Cat"] = cd3["Feature"].apply(cat_fn); cd3["Abs"] = cd3["Coefficient"].abs()
    ci2 = cd3.groupby("Cat")["Abs"].sum().reset_index().sort_values("Abs",ascending=False)
    fig_ci2 = px.bar(ci2,x="Cat",y="Abs",color="Cat",
                     color_discrete_sequence=["#E8574C","#4C9BE8","#F0A500"],
                     title="Risk Driver Weight by Factor Category",
                     labels={"Abs":"Sum of |Coefficients|","Cat":""})
    fig_ci2.update_layout(showlegend=False,height=310,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ci2,use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**🔴 Top Risk Factors**")
        st.dataframe(coef_df[coef_df["Coefficient"]>0].sort_values("Coefficient",ascending=False)[["Feature","Coefficient"]].round(4),
                     hide_index=True,use_container_width=True)
    with c2:
        st.markdown("**🔵 Top Protective Factors**")
        st.dataframe(coef_df[coef_df["Coefficient"]<0].sort_values("Coefficient")[["Feature","Coefficient"]].round(4),
                     hide_index=True,use_container_width=True)

with tab5:
    st.subheader("Operational Action Plan")
    if pred_prob>=0.65: pl,hz,ac = "🔴 HIGH","7 days","1:1 with HR + direct manager — explore engagement, workload, and development"
    elif pred_prob>=0.35: pl,hz,ac = "🟡 MEDIUM","30 days","Follow-up review — check in on key factors and set short-term goals"
    else: pl,hz,ac = "🟢 LOW","Quarterly review","Standard check-in — no urgent action required"
    st.markdown(f"""
| Field | Value |
|---|---|
| **Priority** | {pl} |
| **Estimated risk probability** | {pred_prob:.1%} |
| **Suggested action** | {ac} |
| **Recommended horizon** | {hz} |
| **Suggested owner** | HR Business Partner + Direct Manager |
""")
    st.divider()
    st.markdown("### Key Levers for This Employee Profile")
    rfs = []
    if engagement<=2: rfs.append("engagement_score")
    if scrap_pct>7: rfs.append("scrap_associated_pct")
    if punctuality<0.85: rfs.append("punctuality_rate")
    if training_hrs<15: rfs.append("training_hours_annual")
    if productivity<85: rfs.append("productivity_index")
    if rotation>0.20: rfs.append("area_rotation_rate")
    if contract_t in ["Temporary","Outsourcing"]: rfs.append("contract_type")
    if shift=="Night": rfs.append("shift")
    if rfs:
        for f in rfs[:4]:
            if f in ACTION_MAP:
                with st.expander(f"▲ {f.replace('_',' ').title()} — active risk factor"):
                    st.write(ACTION_MAP[f])
    else:
        st.success("No elevated risk factors detected in the current profile.")
    st.divider()
    st.caption("_This tool supports HR decisions — it does not replace managerial judgment or formal HR processes. A high score flags an employee for a conversation, not a disciplinary action._")
