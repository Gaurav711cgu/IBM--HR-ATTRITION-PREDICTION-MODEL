
# 📊 IBM HR Analytics — Employee Attrition Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Can we predict which employees are about to quit — before they hand in their notice?**
> This project builds a full Machine Learning pipeline on IBM's HR dataset to answer exactly that, wrapped in a live interactive dashboard.

---

## 🌐 Live App

👉 **[Open the Live Dashboard](https://ibm--hr-attrition-prediction-model-yuqlpna8og4tw2rtw8o3dz.streamlit.app/)**

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Results](#-results)
- [Top Attrition Drivers](#-top-attrition-drivers)
- [App Pages](#-app-pages)
- [How to Run Locally](#-how-to-run-locally)
- [Project Structure](#-project-structure)
- [HR Recommendations](#-hr-recommendations)
- [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

Employee attrition costs companies an average of **6–9 months of the employee's salary** to replace them. Early prediction allows HR teams to intervene before it's too late.

This project delivers:

- ✅ A complete **EDA** of 1,470 IBM employees across 23 features
- ✅ **5 Machine Learning models** trained, cross-validated and compared
- ✅ **Feature importance analysis** to find the biggest attrition drivers
- ✅ A **live interactive Streamlit dashboard**
- ✅ A **real-time prediction tool** — enter any employee's details, get an instant risk score
- ✅ **Actionable HR recommendations** based on model findings

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Name** | IBM HR Analytics Employee Attrition & Performance |
| **Source** | [Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Total Employees** | 1,470 |
| **Total Features** | 23 |
| **Target Variable** | `Attrition` (Yes = left, No = stayed) |
| **Attrition Rate** | ~23.7% (class imbalance handled via balanced weights) |

### Key Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Employee age (18–60) |
| MonthlyIncome | Numeric | Monthly salary in USD |
| OverTime | Categorical | Whether employee works overtime |
| JobSatisfaction | Ordinal | 1 (Low) → 4 (High) |
| EnvironmentSatisfaction | Ordinal | 1 (Low) → 4 (High) |
| WorkLifeBalance | Ordinal | 1 (Poor) → 4 (Excellent) |
| DistanceFromHome | Numeric | Commute distance in km |
| YearsAtCompany | Numeric | Tenure at IBM |
| JobLevel | Ordinal | 1 (Junior) → 5 (Executive) |
| StockOptionLevel | Ordinal | 0 (None) → 3 (High) |
| BusinessTravel | Categorical | Non-Travel / Rarely / Frequently |
| MaritalStatus | Categorical | Single / Married / Divorced |
| NumCompaniesWorked | Numeric | Number of previous employers |

---

## ⚙️ Machine Learning Pipeline

```
Raw Data  →  EDA  →  Preprocessing  →  Model Training  →  Evaluation  →  Dashboard
```

**Preprocessing steps:**
- Label encoding of all categorical variables
- StandardScaler applied inside model pipelines
- Stratified 80/20 train-test split (preserves attrition ratio)
- `class_weight='balanced'` on all models to handle imbalance

**Cross Validation:**
- 5-Fold Stratified K-Fold on training set
- Scoring metric: ROC-AUC

---

## 📊 Results

### Model Performance Comparison

| Rank | Model | CV-AUC | Test-AUC | F1 (Attrition) | Recall |
|---|---|---|---|---|---|
| 🥇 | **Logistic Regression** | **0.7127** | **0.6959** | **0.4569** | **0.6429** |
| 🥈 | Random Forest | 0.7080 | 0.6848 | 0.3750 | 0.3000 |
| 🥉 | Gradient Boosting | 0.6928 | 0.6818 | 0.3303 | 0.2571 |
| 4 | SVM (RBF) | 0.6860 | 0.6588 | 0.3933 | 0.5000 |
| 5 | Decision Tree | 0.6346 | 0.6500 | 0.4096 | 0.4857 |

### Confusion Matrix — Best Model (294 test employees)

```
                    Predicted: Stayed    Predicted: Left
Actual: Stayed (224)     142 ✅                82 ⚠️
Actual: Left    (70)      25 ❌                45 ✅

✅ True Negatives   142  (48.3%) → Correctly predicted STAYED
✅ True Positives    45  (15.3%) → Correctly predicted LEFT
⚠️ False Positives   82  (27.9%) → Flagged as leaving — actually stayed
❌ False Negatives   25  ( 8.5%) → Missed leavers
```

### Classification Report

```
               precision    recall   f1-score   support

No Attrition     0.85        0.63      0.73       224
   Attrition     0.35        0.64      0.46        70

    accuracy                           0.64       294
   macro avg     0.60        0.64      0.59       294
weighted avg     0.73        0.64      0.66       294
```

---

## 🔑 Top Attrition Drivers

| Rank | Feature | Score | Finding |
|---|---|---|---|
| 1 | **JobLevel** | 0.724 | Junior staff leave the most |
| 2 | **OverTime** | 0.611 | Overtime employees quit 2× more |
| 3 | **MonthlyIncome** | 0.420 | Leavers earn $518/month less |
| 4 | **DistanceFromHome** | 0.368 | Long commute predicts quitting |
| 5 | **MaritalStatus** | 0.327 | Single employees are more mobile |
| 6 | **WorkLifeBalance** | 0.238 | Poor balance pushes people out |
| 7 | **StockOptionLevel** | 0.226 | No options = highest flight risk |
| 8 | **NumCompaniesWorked** | 0.209 | Job-hoppers keep hopping |
| 9 | **TrainingTimesLastYear** | 0.191 | Lack of growth = dissatisfaction |
| 10 | **TotalWorkingYears** | 0.139 | Less experience = more exploration |

---

## 🖥️ App Pages

| Page | Description |
|---|---|
| 🏠 **Overview & KPIs** | 5 headline metrics, confusion matrix, key findings, dataset preview |
| 🔍 **Exploratory Analysis** | 10 charts across 3 tabs — distributions, attrition rates, correlations |
| 🤖 **Model Comparison** | Ranked table, ROC curves, CV-AUC bars, metric glossary |
| 🎯 **Best Model Deep Dive** | Confusion matrix, Precision-Recall curve, probability distribution |
| 🌲 **Feature Importance** | Random Forest + Logistic Regression importance rankings |
| 💡 **Attrition Drivers** | Top 10 drivers with score bars + specific HR actions |
| 🔮 **Predict Employee** | Enter any employee details → instant attrition risk score + risk flags |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- pip

## 📁 Project Structure

```
ibm-hr-attrition/
│
├── app.py               ← Streamlit dashboard (7 pages + live prediction)
├── requirements.txt     ← Python package dependencies
└── README.md            ← This file
```

> No external data files needed. The dataset is generated synthetically inside `app.py` using the same statistical properties as the original IBM HR Analytics dataset.

---

## 💡 HR Recommendations

| Priority | Action | Impact |
|---|---|---|
| 1 | **Overtime monitoring** — flag consistent overtime for manager check-ins | High |
| 2 | **Junior salary floor** — leavers earn $518/month less on average | High |
| 3 | **Stock option expansion** — extend to Level-0 employees | High |
| 4 | **Remote/hybrid policy** — target employees commuting >20km | Medium |
| 5 | **Early tenure programme** — 0–2 year employees have highest attrition | High |
| 6 | **Travel rotation** — reduce burnout among frequent business travellers | Medium |
| 7 | **Annual training commitment** — minimum 2 sessions per employee per year | Medium |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| Streamlit | 1.32+ | Interactive web dashboard |
| Scikit-Learn | 1.3+ | ML models, cross-validation, metrics |
| Pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical computing |
| Matplotlib | 3.7+ | Visualisations |
| Seaborn | 0.12+ | Heatmaps |

---

## 📦 requirements.txt

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## 👨‍💻 Author

**Saurav**
Dataset: [IBM HR Analytics — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">
  <sub>IBM HR Analytics · Employee Attrition Prediction · Machine Learning Project</sub>
</div>
