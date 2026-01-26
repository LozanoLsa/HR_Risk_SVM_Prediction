# 🧔 Employee Risk Prediction using Support Vector Machines (SVM)

## Project Overview
This project is a **practical machine learning exercise inspired by real-world Human Resources and Operations analytics scenarios**.

Let’s be clear from the start 🙂  
The goal is **not** to label people or automate HR decisions.  
The goal is to **understand**, **visualize**, and **reason** about how employee risk emerges when **behavioral, operational, and structural conditions** start drifting.

A **Support Vector Machine (SVM)** classifier is used to explore **non-linear risk patterns** that are rarely captured by simple rules or linear models.

No black magic.  
Just patterns across people, processes, and context.

---

## Problem Statement
Employee risk — disengagement, low performance, or potential attrition — almost never comes from a single cause.

In real organizations, risk tends to emerge from combinations such as:
- Low engagement combined with poor training
- Operational stress reflected in scrap or rework
- Structural conditions like high rotation areas or unstable contracts
- Shift conditions that amplify fatigue or disengagement

Individually, these signals may look manageable.  
Together, they often signal **systemic risk**.

This project treats employee risk not as an isolated personal issue, but as a **pattern that emerges in specific regions of the organizational space**.

If you’ve worked in operations or HR analytics, this probably sounds familiar.

---

## Objective 🎯
The main objectives of this project are to:

- Build a **binary classification model** to identify employees at higher risk.
- Use **SVM** to capture **non-linear interactions** between HR-related variables.
- Analyze the relative impact of **personal, operational, and structural factors**.
- Emphasize **interpretability** using SHAP and decision boundary visualization.

This project is intentionally designed as a **learning and reasoning exercise**, which is why the dataset is limited to **786 observations** frontline people not indirect areas included.

---

## Dataset Description 📊
The dataset represents a **real and organizationally realistic HR environment**.

Each row corresponds to an employee snapshot under specific working conditions.

### Features (X)

#### Personal Factors
- **training_hours_annual**: Annual training hours
- **punctuality_ratio**: Attendance and punctuality ratio
- **productivity_index**: Performance or productivity index
- **engagement_score**: Engagement survey score (1–5)
- **years_experience**: Years of experience in the organization

#### Operational Factors
- **scrap_associated_pct**: Percentage of scrap or rework associated with the employee’s activities

#### Structural Factors
- **area**: Functional area (Production, Quality, Logistics, etc.)
- **shift**: Work shift (Morning, Evening, Night)
- **contract_type**: Contract type (Permanent, Temporary, Outsourcing)
- **area_rotation_rate**: Historical rotation rate of the employee’s area

### Target Variable (Y)
- **high_risk**
  - **0 = No risk**
  - **1 = High risk**

---

## Data Origin (Real-World HR Perspective)
In real organizations, **this dataset does not come from a single system**.

Each variable typically originates from a different HR, operational, or management source:

### Personal Metrics
- **Training hours**
  → Learning Management Systems (LMS), HR training records.
- **Punctuality**
  → Time & attendance systems, badge readers, payroll data.
- **Productivity**
  → Performance evaluations, KPIs, output tracking, supervisor assessments.
- **Engagement**
  → Employee surveys, pulse surveys, engagement assessments.
- **Experience**
  → HR master data, employee profiles.

### Operational Metrics
- **Scrap / rework association**
  → Quality systems, MES data, defect attribution, production reports.

### Structural Metrics
- **Area & shift**
  → Organizational structure, shift planning systems.
- **Contract type**
  → HR contracts, payroll classification.
- **Area rotation rate**
  → HR analytics, historical attrition data, workforce planning reports.

> In practice, building this dataset requires **integrating HR, Operations, Quality, and Management data**.  
> The model only works when the **organizational context** is understood alongside the numbers 🙂.

---

## Modeling Approach 🧠
A **Support Vector Machine (SVM)** classifier was selected because:

- It handles **non-linear decision boundaries** effectively.
- It performs well when risk emerges from **interacting variables**, not single thresholds.
- It allows the exploration of **risk regions** rather than deterministic rules.

Key steps include:
- Feature scaling and categorical encoding
- Hyperparameter tuning using GridSearchCV
- Evaluation using standard classification metrics
- Model interpretation using:
  - Linear SVM coefficients
  - SHAP summary plots
- Visualization of decision boundaries in 2D and 3D

---

## Why SVM fits this problem well ✅
- Employee risk is **non-linear and multi-dimensional**.
- Structural conditions can override individual performance.
- SVM helps answer a realistic organizational question:
  > “Under what combinations of conditions does risk start to appear?”

This makes it well suited for **HR risk screening and diagnostic analysis**, not automated decisions.

---

## Key Results 📈
Metrics are reported, but they are **not the main message**.

What matters most is that the model helps:
- Identify **high-risk organizational patterns**
- Separate **personal vs structural drivers**
- Visualize how engagement, training, and context interact

SHAP analysis confirms that:
- Low engagement and limited training strongly increase risk.
- Structural factors such as contract type and area rotation can outweigh individual effort.
- Risk is systemic, not personal.

---

## Simulation & Scenarios
A simple **employee risk simulator** is included.

It allows you to:
- Define hypothetical employee profiles
- Estimate relative risk probability
- Explore “what if?” scenarios for HR decision support

This turns the model into a **conversation tool**, not a verdict engine.

---

## Project Outputs 📂
This repository contains:
- A synthetic HR dataset ('.csv')
- A Jupyter Notebook with full analysis, modeling, and interpretation
- Visual reports and summary materials for non-technical audiences

---

## Next Steps 🚀
If this project were extended further, possible directions include:
- Time-based engagement and performance trends
- Early-warning indicators for attrition
- Comparison with tree-based or ensemble models
- Integration into workforce planning dashboards

But that’s a different project.

---

—
Not labels. Not judgments.  
**Just patterns in people, processes, and systems.**  
LozanoLsa  
Regards from MX
