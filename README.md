# HR Employee Risk Analytics — Support Vector Machine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LozanoLsa/04-SVM-HR-Risk/blob/main/04_SVM_HR_Risk_Analytics.ipynb)

> *"An employee doesn't disengage overnight. The signals are in the punctuality records, the engagement surveys, the scrap reports, the training logs. They've been there for months. The exit interview is already too late."*

---

## 🎯 Business Problem

HR analytics is still largely reactive in most manufacturing organizations. Turnover gets measured after people leave. Disengagement gets noticed after performance drops. The cost of replacement — conservatively 50–200% of annual salary per employee — gets absorbed as an operational given rather than addressed as a preventable risk.

This project builds an **early-warning risk score** for HR: a model that identifies employees showing behavioral and operational patterns consistent with high risk, before any formal event occurs. Not to surveil — to enable the right conversation at the right time.

**Important caveat upfront:** a high score flags an employee for a *conversation*, not a verdict. The model identifies patterns; context and human judgment determine the response. That distinction matters in people decisions more than anywhere else.

---

## 📊 Dataset

- **1,000 employee records** from a simulated manufacturing HR system
- **Target:** `high_risk` (binary) — 1 = patterns consistent with disengagement or performance decline risk
- **Class balance:** 30.8% high-risk
- **Source:** Simulated workforce data reflecting realistic manufacturing HR feature distributions

**Three layers of features:**

| Layer | Features | What it captures |
|-------|----------|-----------------|
| Behavioral | `punctuality_rate`, `engagement_score` | Direct disengagement signals |
| Operational | `productivity_index`, `scrap_associated_pct`, `training_hours_annual`, `experience_yrs` | What the employee produces and receives |
| Structural | `area_rotation_rate`, `department`, `shift`, `contract_type` | Organizational context |

**Risk rates by structural context (EDA):**

| Variable | Highest Risk Group | Rate | Lowest Risk Group | Rate |
|----------|-------------------|------|-------------------|------|
| Contract | Temporary | 40.8% | Permanent | 25.7% |
| Department | Administration | 38.9% | Quality | 21.7% |
| Shift | Night | 35.1% | Afternoon | 27.0% |

---

## 🤖 Model

**Algorithm:** Support Vector Machine (RBF kernel) — `sklearn.svm.SVC`

Employee risk doesn't follow a linear rule. A worker can have low punctuality without being high-risk if their engagement and productivity are strong. Risk emerges from *combinations* — which is exactly what the RBF kernel is designed to capture. It maps the feature space into higher dimensions where these non-linear patterns become separable with a maximum-margin boundary.

A **LinearSVC companion** is trained alongside the main model to provide coefficient-level interpretability — the same directional transparency as logistic regression, without sacrificing predictive power for stakeholder communication.

**Preprocessing:** StandardScaler on numerics, OneHotEncoder (drop_first) on categoricals, all inside a Pipeline.  
**Tuning:** GridSearchCV scoring on F1 (correct for imbalanced data) — best: `kernel=rbf`, `C=50`, `gamma=0.01`.

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

**Why the numbers look modest — and why that's honest:** HR behavioral data is noisier than sensor data. Human behavior is influenced by factors no dataset captures: personal circumstances, team dynamics, manager relationships. AUC 0.659 > 0.5 confirms real signal. The model's practical value is in **risk ranking** — sorting employees by probability score — not in binary classification at a fixed threshold.

---

## 🔍 Risk Drivers (LinearSVC Coefficients)

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| `contract_type_Permanent` | −0.269 | 🔵 Protective |
| `engagement_score` | −0.234 | 🔵 Most actionable lever |
| `shift_Night` | +0.268 | 🔴 Strongest structural risk |
| `department_Quality` | −0.330 | 🔵 Protective context |
| `experience_yrs` | −0.197 | 🔵 Protective |
| `training_hours_annual` | −0.181 | 🔵 Investment = protection |
| `scrap_associated_pct` | +0.162 | 🔴 Quality stress signal |

Engagement score is the highest-leverage variable HR can directly influence. Night shift is the highest structural risk factor — and it can't be solved with an engagement survey. Both insights require different interventions.

---

## 🗂️ Repository Structure

```
04-SVM-HR-Risk/
├── 04_SVM_HR_Risk_Analytics.ipynb  # Notebook (no outputs)
├── hr_risk_svm_data.csv            # Sample dataset (250 rows)
├── README.md
└── requirements.txt
```

> 📦 **Full Project Pack** — complete dataset (1,000 rows), notebook with full outputs, presentation deck (PPTX + PDF), and `app.py` employee risk simulator available on [Gumroad](https://lozanolsa.gumroad.com).

---

## 🚀 How to Run

**Option 1 — Google Colab:** Click the badge above.

**Option 2 — Local:**
```bash
pip install -r requirements.txt
jupyter notebook 04_SVM_HR_Risk_Analytics.ipynb
```

---

## 💡 Key Learnings

1. **Non-linear risk needs a non-linear model** — a single low punctuality score doesn't make someone high-risk. The combination does. RBF SVM finds that boundary where logistic regression can't.
2. **Train an interpretable companion** — RBF SVM predicts. LinearSVC explains. Same data, same preprocessing, two purposes. This pattern scales to any black-box model.
3. **Structural factors are as important as behavioral ones** — night shift increases risk independently of how engaged the employee is. Fixing engagement without fixing structure is incomplete.
4. **Moderate AUC in HR is not failure** — it's calibration to reality. A model claiming 95% accuracy on workforce data should raise skepticism, not admiration.
5. **The threshold is a business decision, not a model decision** — lowering from 0.5 to 0.3 trades precision for recall. In HR, catching more true positives (at the cost of more unnecessary conversations) is almost always the right trade.

---

## 👤 Author

**Luis Lozano** | Operational Excellence Manager · Master Black Belt · Machine Learning  
GitHub: [LozanoLsa](https://github.com/LozanoLsa) · Gumroad: [lozanolsa.gumroad.com](https://lozanolsa.gumroad.com)

*Turning Operations into Predictive Systems — Clone it. Fork it. Improve it.*
