# HR Employee Risk Analytics — Support Vector Machine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/04-SVM-HR-Risk/blob/main/04_SVM_HR_Risk_Analytics.ipynb)

> *"Disengagement doesn't happen overnight — it accumulates through measurable signals for weeks before any formal event. The question is whether HR is reading them in time."*

---

## 🎯 Business Problem

Employee attrition, disengagement, and performance decline rarely appear without warning. By the time HR identifies a problem — at a resignation, a disciplinary event, or a poor performance review — the intervention window has already closed. This project builds a **behavioral early-warning risk score** that classifies employees as high-risk or not, based on a combination of behavioral indicators, operational performance metrics, and structural context variables. The goal is not prediction for its own sake — it's enabling targeted HR conversations *before* the cost becomes visible.

---

## 📊 Dataset

- **1,000 employee records** from a simulated manufacturing HR system
- **Target:** `high_risk` (binary) — 1 if the employee shows patterns consistent with disengagement or performance decline risk
- **Class balance:** 30.8% high-risk (moderately imbalanced)
- **Source:** Simulated operational HR data; features reflect realistic manufacturing workforce conditions

| Layer | Features |
|-------|----------|
| Behavioral | `punctuality_rate`, `engagement_score` |
| Operational | `productivity_index`, `scrap_associated_pct`, `training_hours_annual`, `experience_yrs` |
| Structural | `area_rotation_rate`, `department`, `shift`, `contract_type` |

**Key EDA findings:**
- Temporary contracts show 40.8% risk rate vs 25.7% for permanent — the largest structural gap
- Night shift workers show 35.1% risk vs 27.0% afternoon
- Administration and Production departments carry above-average risk (38.9% and 37.4%)
- Engagement score is the single strongest individual predictor

---

## 🤖 Model

**Algorithm:** Support Vector Machine (RBF kernel) — `sklearn.svm.SVC`

SVM was chosen because employee risk doesn't follow a linear rule. Risk emerges from *combinations* of conditions — an employee can have low punctuality without being high-risk if engagement and productivity are strong. The RBF kernel maps data into higher dimensions where these non-linear patterns become separable.

A **LinearSVC companion** is trained alongside the RBF model to provide directional coefficient interpretability — the same transparency that logistic regression offers, without sacrificing the predictive power of the non-linear boundary.

**Preprocessing:**
- `StandardScaler` on all numeric features (SVM is sensitive to scale)
- `OneHotEncoder` (drop_first=True) on categorical variables
- All preprocessing inside `sklearn.Pipeline` with `ColumnTransformer`

**Tuning:** `GridSearchCV` over kernel × C × gamma, scoring on F1 (correct metric for imbalanced data)
- **Best params:** `kernel=rbf`, `C=50`, `gamma=0.01`
- **Best CV F1:** 0.499

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Accuracy | 68.4% |
| ROC-AUC | 0.659 |
| Precision (High Risk) | 47.6% |
| Recall (High Risk) | 26.0% |
| F1 (High Risk) | 0.336 |

**Confusion matrix (250 test employees):**

| | Pred: Low Risk | Pred: High Risk |
|---|---|---|
| **Actual: Low Risk** | 151 (TN) | 22 (FP) |
| **Actual: High Risk** | 57 (FN) | 20 (TP) |

**Honest interpretation:** This is a moderately performing model, and that's expected — HR behavioral data is inherently noisier than sensor data. AUC > 0.5 confirms real signal exists. The model is most useful as a **risk ranking tool** (probability score) rather than a binary classifier at the 0.5 threshold. Lowering the threshold to 0.3 significantly improves recall at the cost of precision — a worthwhile trade in HR contexts where false negatives (missed interventions) are more costly than false positives.

---

## 🔍 Top Risk Drivers (LinearSVC Coefficients)

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| `contract_type_Permanent` | −0.269 | 🔵 Protective |
| `engagement_score` | −0.234 | 🔵 Protective |
| `shift_Night` | +0.268 | 🔴 Risk factor |
| `department_Quality` | −0.330 | 🔵 Protective |
| `experience_yrs` | −0.197 | 🔵 Protective |
| `training_hours_annual` | −0.181 | 🔵 Protective |
| `scrap_associated_pct` | +0.162 | 🔴 Risk factor |

**Key insight:** Night shift is the strongest structural risk driver. Engagement score is the most actionable lever — it's the variable HR can directly influence through management interventions.

---

## 🗂️ Repository Structure

```
04-SVM-HR-Risk/
├── 04_SVM_HR_Risk_Analytics.ipynb   # Full educational notebook
├── hr_risk_svm_data.csv             # Dataset (1,000 employee records)
├── HR_Risk_Analytics_SVM.pptx       # Presentation deck
└── README.md
```

---

## 🚀 How to Run

**Option 1 — Google Colab (recommended):**
Click the badge at the top of this README. No installation required.

**Option 2 — Local:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook 04_SVM_HR_Risk_Analytics.ipynb
```

The notebook loads data automatically — if the CSV is not in the same directory, it falls back to the GitHub raw URL.

---

## 💡 Key Learnings

1. **SVM excels at non-linear risk patterns** — employee risk is a combination problem, not a threshold problem. RBF kernel captures interaction effects that logistic regression misses.
2. **Moderate AUC is honest in HR analytics** — human behavior is noisier than physical systems. A model claiming 95% accuracy on HR data should raise skepticism.
3. **Use probability scores, not binary verdicts** — the `predict_proba()` output is more operationally useful than `predict()`. Score employees on a continuum and let HR decide the threshold.
4. **A linear companion is worth training** — LinearSVC coefficients translate model behavior into language HR managers actually understand. Interpretability is not optional in people decisions.
5. **Structural context matters as much as behavior** — night shift, contract type, and department explain a significant portion of risk variance. Fixing individual behavior without addressing structural conditions is incomplete.

---

## ⚠️ Important Caveat

This score is **not a verdict**. A high-risk probability flags an employee for a *conversation* — not a disciplinary action, not a termination decision. The model identifies patterns; context and human judgment determine the response. Used correctly, it shifts HR from reactive to proactive. Used incorrectly, it becomes a surveillance tool. The distinction matters.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Engineer · Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa)

*Not magic. Just probabilities. | Where f(x) meets Kaizen*
