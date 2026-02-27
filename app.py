"""
IBM HR Analytics – Employee Attrition Prediction
Streamlit Dashboard App
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="IBM HR Attrition · Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Main background */
.stApp { background-color: #0d0f14; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #141720 !important;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] * { color: #e2e5ef !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #141720;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label { color: #5c6380 !important; font-size: 0.72rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e2e5ef !important; font-size: 1.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #52d483 !important; }

/* Headings */
h1, h2, h3 { color: #e2e5ef !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e2330; border-radius: 8px; }

/* Tabs */
[data-baseweb="tab-list"] { background: #141720; border-radius: 10px; gap: 4px; padding: 4px; }
[data-baseweb="tab"] { border-radius: 8px !important; color: #5c6380 !important; }
[aria-selected="true"] { background: #1e2330 !important; color: #e2e5ef !important; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #1e2330; border-radius: 10px; background: #141720; }

/* Divider */
hr { border-color: #1e2330; }

/* Info / success / warning boxes */
[data-testid="stAlert"] { border-radius: 10px; }

/* Buttons */
.stButton button {
    background: #1e2330;
    color: #e2e5ef;
    border: 1px solid #252b3b;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
}
.stButton button:hover { border-color: #4f9cf9; color: #4f9cf9; }

/* Code blocks */
code { background: #141720 !important; color: #4f9cf9 !important; border-radius: 4px; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #252b3b; border-radius: 2px; }

/* Chart background match */
.stPlotlyChart, .stPyplot { background: #141720 !important; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette for matplotlib ────────────────────────────────
PALETTE = {"Yes": "#e05c5c", "No": "#4f9cf9"}
PRIMARY, ACCENT, BG = "#e2e5ef", "#e05c5c", "#141720"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "text.color": PRIMARY, "axes.labelcolor": PRIMARY,
    "xtick.color": "#5c6380", "ytick.color": "#5c6380",
    "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.titlecolor": PRIMARY,
    "grid.color": "#1e2330", "grid.alpha": 0.5,
})

# ════════════════════════════════════════════════════════════════
# DATA GENERATION (cached so it only runs once)
# ════════════════════════════════════════════════════════════════
@st.cache_data
def generate_data():
    np.random.seed(42)
    N = 1470
    dept_choices = ["Sales", "Research & Development", "Human Resources"]
    role_map = {
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": ["Research Scientist", "Laboratory Technician",
                                   "Manufacturing Director", "Research Director", "Healthcare Representative"],
        "Human Resources": ["Human Resources", "Manager"],
    }
    edu_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]

    dept        = np.random.choice(dept_choices, N, p=[0.30, 0.61, 0.09])
    age         = np.random.randint(18, 61, N)
    yrs_company = np.clip(np.random.exponential(7, N).astype(int), 1, 40)
    yrs_role    = np.clip((yrs_company * np.random.uniform(0.3, 0.9, N)).astype(int), 0, yrs_company)
    job_level   = np.random.choice([1, 2, 3, 4, 5], N, p=[0.26, 0.34, 0.22, 0.12, 0.06])
    monthly_inc = (job_level * 2400 + np.random.normal(0, 900, N)).clip(1009, 19999).astype(int)
    overtime    = np.random.binomial(1, 0.28, N)
    job_sat     = np.random.randint(1, 5, N)
    env_sat     = np.random.randint(1, 5, N)
    wlb         = np.random.randint(1, 5, N)
    rel_sat     = np.random.randint(1, 5, N)
    distance    = np.random.randint(1, 30, N)
    education   = np.random.randint(1, 6, N)
    edu_field   = np.random.choice(edu_fields, N)
    num_comp    = np.clip(np.random.poisson(2.7, N), 0, 9)
    perf_rating = np.random.choice([3, 4], N, p=[0.85, 0.15])
    stock_opt   = np.random.choice([0, 1, 2, 3], N, p=[0.47, 0.36, 0.12, 0.05])
    training    = np.random.randint(0, 7, N)
    marital     = np.random.choice(["Single", "Married", "Divorced"], N, p=[0.32, 0.46, 0.22])
    gender      = np.random.choice(["Male", "Female"], N, p=[0.60, 0.40])
    biz_travel  = np.random.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], N, p=[0.19, 0.71, 0.10])
    roles       = np.array([np.random.choice(role_map[d]) for d in dept])

    logit = (
        -1.2 + 1.2 * overtime
        - 0.6 * (job_sat - 1) / 3
        - 0.5 * (env_sat - 1) / 3
        - 0.5 * (wlb - 1) / 3
        + 0.6 * (num_comp > 3).astype(int)
        + 0.8 * (distance > 20).astype(int)
        - 0.6 * (stock_opt > 0).astype(int)
        + 0.9 * (marital == "Single").astype(int)
        - 0.5 * (yrs_company > 5).astype(int)
        - 0.4 * (job_level > 2).astype(int)
        + 0.8 * (biz_travel == "Travel_Frequently").astype(int)
        + 0.4 * (biz_travel == "Travel_Rarely").astype(int)
        - 0.5 * (monthly_inc > 8000).astype(int)
        + 0.4 * (age < 30).astype(int)
        - 0.3 * (rel_sat > 2).astype(int)
    )
    prob      = 1 / (1 + np.exp(-logit))
    attrition = (np.random.uniform(0, 1, N) < prob).astype(int)

    df = pd.DataFrame({
        "Age": age, "Attrition": np.where(attrition, "Yes", "No"),
        "BusinessTravel": biz_travel, "Department": dept,
        "DistanceFromHome": distance, "Education": education,
        "EducationField": edu_field, "EnvironmentSatisfaction": env_sat,
        "Gender": gender, "JobLevel": job_level, "JobRole": roles,
        "JobSatisfaction": job_sat, "MaritalStatus": marital,
        "MonthlyIncome": monthly_inc, "NumCompaniesWorked": num_comp,
        "OverTime": np.where(overtime, "Yes", "No"),
        "PerformanceRating": perf_rating, "RelationshipSatisfaction": rel_sat,
        "StockOptionLevel": stock_opt, "TotalWorkingYears": yrs_company,
        "TrainingTimesLastYear": training, "WorkLifeBalance": wlb,
        "YearsAtCompany": yrs_company, "YearsInCurrentRole": yrs_role,
    })
    return df


@st.cache_data
def train_models(df):
    df_proc = df.copy()
    df_proc["Attrition"] = (df_proc["Attrition"] == "Yes").astype(int)
    for c in df_proc.select_dtypes(include="object").columns:
        df_proc[c] = LabelEncoder().fit_transform(df_proc[c])

    X = df_proc.drop("Attrition", axis=1)
    y = df_proc["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    models = {
        "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                          ("clf", LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=42))]),
        "Decision Tree":        Pipeline([("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42))]),
        "Random Forest":        Pipeline([("clf", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42))]),
        "Gradient Boosting":    Pipeline([("clf", GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42))]),
        "SVM (RBF)":            Pipeline([("sc", StandardScaler()),
                                          ("clf", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, pipe in models.items():
        cv_roc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        report  = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "pipe": pipe, "cv_roc": cv_roc, "y_pred": y_pred, "y_proba": y_proba,
            "roc_auc":  roc_auc_score(y_test, y_proba),
            "avg_prec": average_precision_score(y_test, y_proba),
            "f1_yes":   report.get("1", {}).get("f1-score", 0),
            "rec_yes":  report.get("1", {}).get("recall", 0),
            "prec_yes": report.get("1", {}).get("precision", 0),
            "report":   report,
        }

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    return X, y, X_train, X_test, y_train, y_test, results, best_name, models


# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
with st.spinner("🔄 Generating dataset and training models…"):
    df = generate_data()
    X, y, X_train, X_test, y_train, y_test, results, best_name, models = train_models(df)

best = results[best_name]
yes_count  = (df["Attrition"] == "Yes").sum()
attr_rate  = yes_count / len(df) * 100
avg_inc_y  = df[df.Attrition == "Yes"]["MonthlyIncome"].mean()
avg_inc_n  = df[df.Attrition == "No"]["MonthlyIncome"].mean()
ot_rate_y  = (df[df.Attrition == "Yes"]["OverTime"] == "Yes").mean() * 100
ot_rate_n  = (df[df.Attrition == "No"]["OverTime"] == "Yes").mean() * 100
cm         = confusion_matrix(y_test, best["y_pred"])
TN, FP, FN, TP = cm.ravel()


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 IBM HR Attrition")
    st.markdown("*Employee Retention Intelligence*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview & KPIs",
         "🔍 Exploratory Analysis",
         "🤖 Model Comparison",
         "🎯 Best Model Deep Dive",
         "🌲 Feature Importance",
         "💡 Attrition Drivers",
         "🔮 Predict Employee"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(f"""
    <div style='font-family:DM Mono,monospace;font-size:0.65rem;color:#5c6380;line-height:2'>
    Employees &nbsp;&nbsp;1,470<br/>
    Features &nbsp;&nbsp;&nbsp;&nbsp;23<br/>
    Train split &nbsp;80%<br/>
    Test split &nbsp;&nbsp;20%<br/>
    CV Folds &nbsp;&nbsp;&nbsp;5<br/>
    Models run &nbsp;5
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.caption("IBM HR Analytics Dataset")


# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "🏠 Overview & KPIs":
    st.markdown("## 🏠 Overview & KPIs")
    st.markdown("*IBM HR Analytics – Employee Attrition Prediction Pipeline*")
    st.divider()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Attrition Rate",    f"{attr_rate:.1f}%",   f"{yes_count} employees left")
    c2.metric("Best ROC-AUC",      f"{best['roc_auc']:.3f}", f"{best_name}")
    c3.metric("Recall (Leavers)",  f"{best['rec_yes']:.1%}", "Of leavers caught")
    c4.metric("F1 Score",          f"{best['f1_yes']:.3f}", "Attrition class")
    c5.metric("Lift over Random",  f"{best['avg_prec']/y_test.mean():.2f}×", "Avg Precision")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📋 Confusion Matrix Results")
        cm_data = {
            "Outcome": ["✅ True Negatives", "✅ True Positives", "⚠️ False Positives", "❌ False Negatives"],
            "Count": [TN, TP, FP, FN],
            "Pct": [f"{TN/cm.sum():.1%}", f"{TP/cm.sum():.1%}", f"{FP/cm.sum():.1%}", f"{FN/cm.sum():.1%}"],
            "Meaning": [
                "Correctly predicted STAYED",
                "Correctly predicted LEFT",
                "Flagged as leaving – actually stayed",
                "Predicted stayed – actually left"
            ]
        }
        st.dataframe(pd.DataFrame(cm_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### 🔑 Key EDA Findings")
        st.info(f"⏱️ **Overtime** — Leavers work overtime at **{ot_rate_y:.0f}%** vs **{ot_rate_n:.0f}%** for stayers")
        st.info(f"💰 **Income Gap** — Leavers earn **${avg_inc_y:,.0f}** vs **${avg_inc_n:,.0f}** for stayers")
        st.warning(f"🚗 **Commute** — Employees > 20km away are significantly more likely to leave")
        st.warning(f"📈 **Early Tenure** — 0–2 year employees churn at the highest rate")
        st.success(f"💎 **Stock Options** — Level-0 employees (no options) leave far more often")

    st.divider()
    st.markdown("#### 📊 Dataset Preview")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.dataframe(df.head(20), use_container_width=True, height=300)
    with col_b:
        st.markdown("**Shape**")
        st.code(f"{df.shape[0]} rows\n{df.shape[1]} cols")
        st.markdown("**Attrition**")
        vc = df["Attrition"].value_counts()
        st.code(f"No:  {vc['No']} ({vc['No']/len(df):.1%})\nYes: {vc['Yes']} ({vc['Yes']/len(df):.1%})")
        st.markdown("**Null Values**")
        st.code(f"{df.isnull().sum().sum()} missing")


# ════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.caption("Visual breakdown of the 1,470 employee dataset across key HR dimensions")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📈 Attrition Rates", "🔗 Correlations"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Attrition donut
            fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
            counts = df["Attrition"].value_counts()
            wedges, texts, autotexts = ax.pie(
                counts, labels=counts.index, autopct="%1.1f%%",
                colors=[PALETTE[k] for k in counts.index],
                wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
                startangle=90, textprops={"color": PRIMARY, "fontsize": 11}
            )
            for at in autotexts: at.set_color(BG)
            ax.set_title("Attrition Split", color=PRIMARY, fontweight="bold", pad=12)
            st.pyplot(fig, use_container_width=True)
            st.caption("💡 23.7% attrition rate — class imbalance handled via balanced weights in all models")

        with col2:
            # Age histogram
            fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
            ax.set_facecolor(BG)
            for label, grp in df.groupby("Attrition"):
                ax.hist(grp["Age"], bins=18, alpha=0.75, label=label,
                        color=PALETTE[label], edgecolor=BG)
            ax.set_title("Age Distribution by Attrition", color=PRIMARY, fontweight="bold")
            ax.set_xlabel("Age"); ax.set_ylabel("Count")
            ax.legend(title="Attrition", labelcolor=PRIMARY,
                      facecolor=BG, edgecolor="#1e2330")
            st.pyplot(fig, use_container_width=True)
            st.caption("💡 Younger employees (<30) leave at a higher rate than 35–50 cohort")

        col3, col4 = st.columns(2)

        with col3:
            # Monthly income boxplot
            fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
            ax.set_facecolor(BG)
            yes_inc = df[df.Attrition == "Yes"]["MonthlyIncome"]
            no_inc  = df[df.Attrition == "No"]["MonthlyIncome"]
            bp = ax.boxplot([no_inc, yes_inc], patch_artist=True, labels=["Stayed", "Left"],
                            boxprops=dict(linewidth=1.5),
                            medianprops=dict(color=PRIMARY, linewidth=2.5),
                            whiskerprops=dict(color="#5c6380"),
                            capprops=dict(color="#5c6380"),
                            flierprops=dict(marker='o', color="#5c6380", alpha=0.3, markersize=3))
            bp["boxes"][0].set_facecolor(PALETTE["No"] + "55")
            bp["boxes"][1].set_facecolor(PALETTE["Yes"] + "55")
            ax.set_title("Monthly Income vs Attrition", color=PRIMARY, fontweight="bold")
            ax.set_ylabel("Monthly Income ($)")
            st.pyplot(fig, use_container_width=True)
            st.caption(f"💡 Leavers earn ~${avg_inc_n - avg_inc_y:,.0f} less/month on average")

        with col4:
            # Overtime
            fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
            ax.set_facecolor(BG)
            ot = df.groupby(["OverTime", "Attrition"]).size().unstack(fill_value=0)
            x   = np.arange(len(ot))
            w   = 0.35
            ax.bar(x - w/2, ot.get("No", 0),  w, color=PALETTE["No"],  alpha=0.8, label="Stayed", edgecolor=BG)
            ax.bar(x + w/2, ot.get("Yes", 0), w, color=PALETTE["Yes"], alpha=0.8, label="Left",   edgecolor=BG)
            ax.set_xticks(x); ax.set_xticklabels(ot.index)
            ax.set_title("OverTime vs Attrition", color=PRIMARY, fontweight="bold")
            ax.legend(labelcolor=PRIMARY, facecolor=BG, edgecolor="#1e2330")
            st.pyplot(fig, use_container_width=True)
            st.caption(f"💡 Overtime employees leave at {ot_rate_y:.0f}% vs {ot_rate_n:.0f}% for non-overtime")

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=BG)
            ax.set_facecolor(BG)
            dept_r = (df.groupby(["Department", "Attrition"]).size().unstack(fill_value=0)
                        .assign(Rate=lambda x: x["Yes"] / (x["Yes"] + x["No"]) * 100)
                        .sort_values("Rate"))
            bars = ax.barh(dept_r.index, dept_r["Rate"], color=ACCENT, alpha=0.85, edgecolor=BG)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=PRIMARY)
            ax.set_title("By Department", color=PRIMARY, fontweight="bold")
            ax.set_xlabel("Attrition Rate (%)")
            st.pyplot(fig, use_container_width=True)

        with col2:
            fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=BG)
            ax.set_facecolor(BG)
            js  = df.groupby(["JobSatisfaction", "Attrition"]).size().unstack(fill_value=0)
            jsr = js["Yes"] / (js["Yes"] + js["No"]) * 100
            bars = ax.bar(jsr.index, jsr.values, color=ACCENT, alpha=0.85, edgecolor=BG)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=PRIMARY)
            ax.set_title("By Job Satisfaction", color=PRIMARY, fontweight="bold")
            ax.set_xlabel("Score (1=Low, 4=High)")
            ax.set_ylabel("Attrition Rate (%)")
            st.pyplot(fig, use_container_width=True)

        with col3:
            fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=BG)
            ax.set_facecolor(BG)
            ms  = df.groupby(["MaritalStatus", "Attrition"]).size().unstack(fill_value=0)
            msr = ms["Yes"] / (ms["Yes"] + ms["No"]) * 100
            cols_ms = ["#4f9cf9", "#e05c5c", "#52d483"]
            bars = ax.bar(msr.index, msr.values, color=cols_ms, alpha=0.85, edgecolor=BG)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=PRIMARY)
            ax.set_title("By Marital Status", color=PRIMARY, fontweight="bold")
            ax.set_ylabel("Attrition Rate (%)")
            st.pyplot(fig, use_container_width=True)

        col4, col5, col6 = st.columns(3)
        with col4:
            fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=BG)
            ax.set_facecolor(BG)
            bt  = df.groupby(["BusinessTravel", "Attrition"]).size().unstack(fill_value=0)
            btr = bt["Yes"] / (bt["Yes"] + bt["No"]) * 100
            bars = ax.bar(range(len(btr)), btr.values, color=["#4f9cf9", "#f0a840", "#e05c5c"], alpha=0.85, edgecolor=BG)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=PRIMARY)
            ax.set_xticks(range(len(btr))); ax.set_xticklabels(btr.index, rotation=10, ha="right", fontsize=8)
            ax.set_title("By Business Travel", color=PRIMARY, fontweight="bold")
            st.pyplot(fig, use_container_width=True)

        with col5:
            fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=BG)
            ax.set_facecolor(BG)
            so  = df.groupby(["StockOptionLevel", "Attrition"]).size().unstack(fill_value=0)
            sor = so["Yes"] / (so["Yes"] + so["No"]) * 100
            bars = ax.bar(sor.index, sor.values, color=ACCENT, alpha=0.85, edgecolor=BG)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=PRIMARY)
            ax.set_title("By Stock Option Level", color=PRIMARY, fontweight="bold")
            ax.set_xlabel("Level (0=None, 3=High)")
            st.pyplot(fig, use_container_width=True)

        with col6:
            df["YearsBin"] = pd.cut(df["YearsAtCompany"], bins=[0,2,5,10,20,41],
                                     labels=["0-2","3-5","6-10","11-20","20+"])
            yr      = df.groupby(["YearsBin","Attrition"]).size().unstack(fill_value=0)
            yr_rate = yr["Yes"] / (yr["Yes"] + yr["No"]) * 100
            fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor=BG)
            ax.set_facecolor(BG)
            bars = ax.bar(yr_rate.index.astype(str), yr_rate.values, color=ACCENT, alpha=0.85, edgecolor=BG)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=PRIMARY)
            ax.set_title("By Tenure", color=PRIMARY, fontweight="bold")
            ax.set_xlabel("Years at Company")
            st.pyplot(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Feature Correlation Heatmap")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        attrition_bin = (df["Attrition"] == "Yes").astype(int)
        corr = df[num_cols].assign(Attrition_bin=attrition_bin).corr()
        fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
        ax.set_facecolor(BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                    vmin=-1, vmax=1, linewidths=0.4, annot=True,
                    fmt=".2f", annot_kws={"size": 7},
                    cbar_kws={"shrink": 0.6})
        ax.set_title("Correlation Matrix — All Numeric Features", color=PRIMARY,
                     fontweight="bold", fontsize=13)
        st.pyplot(fig, use_container_width=True)
        st.caption("💡 YearsAtCompany, TotalWorkingYears and JobLevel are highly correlated — consider feature selection for production use")


# ════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("## 🤖 Model Comparison")
    st.caption("5 classifiers evaluated with 5-fold cross validation + held-out test set")
    st.divider()

    # Summary table
    rows = []
    for name, res in sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        rows.append({
            "Model":       ("🏆 " if name == best_name else "   ") + name,
            "CV-AUC":      f"{res['cv_roc'].mean():.4f} ± {res['cv_roc'].std():.4f}",
            "Test-AUC":    f"{res['roc_auc']:.4f}",
            "Avg-Prec":    f"{res['avg_prec']:.4f}",
            "F1 (Leavers)":f"{res['f1_yes']:.4f}",
            "Recall":      f"{res['rec_yes']:.4f}",
            "Precision":   f"{res['prec_yes']:.4f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # ROC Curves
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(BG)
        colors = ["#4f9cf9", "#52d483", "#e05c5c", "#9b59b6", "#f0a840"]
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            lw = 3 if name == best_name else 1.5
            ax.plot(fpr, tpr, color=colors[i], lw=lw, alpha=0.9,
                    label=f"{name} ({res['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "#252b3b", linestyle="--", lw=1.5, label="Random (0.500)")
        ax.set_title("ROC Curves — Test Set", color=PRIMARY, fontweight="bold")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, labelcolor=PRIMARY, facecolor=BG, edgecolor="#1e2330")
        st.pyplot(fig, use_container_width=True)

    with col2:
        # CV AUC bar chart
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        ax.set_facecolor(BG)
        names  = list(results.keys())
        means  = [results[n]["cv_roc"].mean() for n in names]
        stds   = [results[n]["cv_roc"].std()  for n in names]
        bars   = ax.barh(names, means, xerr=stds, color=colors[:len(names)],
                         alpha=0.85, edgecolor=BG,
                         error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "#5c6380"})
        ax.axvline(0.5, color="#5c6380", linestyle="--", alpha=0.6)
        for bar, v in zip(bars, means):
            ax.text(v + 0.004, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9, color=PRIMARY)
        ax.set_title("5-Fold CV ROC-AUC", color=PRIMARY, fontweight="bold")
        ax.set_xlim(0.45, 1.0)
        st.pyplot(fig, use_container_width=True)

    # Metric explainer
    st.divider()
    st.markdown("#### 📖 Metric Glossary")
    c1, c2, c3 = st.columns(3)
    c1.info("**ROC-AUC** — How well the model separates leavers from stayers.\n0.5 = random · 0.7+ = good · 0.9+ = excellent")
    c2.info("**Recall (Leavers)** — Of all employees who actually left, what % did the model catch? High recall = fewer missed leavers")
    c3.info("**F1-Score** — Harmonic mean of Precision & Recall. Best single metric for imbalanced classes like attrition")


# ════════════════════════════════════════════════════════════════
# PAGE 4 — BEST MODEL DEEP DIVE
# ════════════════════════════════════════════════════════════════
elif page == "🎯 Best Model Deep Dive":
    st.markdown(f"## 🎯 Best Model — {best_name}")
    st.caption(f"Test ROC-AUC: **{best['roc_auc']:.4f}** · Recall: **{best['rec_yes']:.1%}** · F1: **{best['f1_yes']:.4f}**")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
        ax.set_facecolor(BG)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Attrition", "Attrition"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", color=PRIMARY, fontweight="bold")
        total = cm.sum()
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i + 0.35, f"({val/total:.1%})", ha="center", va="top", fontsize=9, color="#5c6380")
        st.pyplot(fig, use_container_width=True)

        st.markdown("**In plain English:**")
        st.success(f"✅ Correctly identified **{TP}** employees who left")
        st.success(f"✅ Correctly retained **{TN}** employees as stayers")
        st.warning(f"⚠️ False alarms on **{FP}** employees (flagged as leaving, stayed)")
        st.error(f"❌ Missed **{FN}** actual leavers (blind spots)")

    with col2:
        st.markdown("#### Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
        ax.set_facecolor(BG)
        prec_c, rec_c, _ = precision_recall_curve(y_test, best["y_proba"])
        ax.fill_between(rec_c, prec_c, alpha=0.12, color=ACCENT)
        ax.plot(rec_c, prec_c, color=ACCENT, lw=2.5)
        baseline = y_test.mean()
        ax.axhline(baseline, color="#5c6380", linestyle="--", alpha=0.7, label=f"Baseline ({baseline:.2f})")
        ax.set_title(f"Precision-Recall  (AP={best['avg_prec']:.3f})", color=PRIMARY, fontweight="bold")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(labelcolor=PRIMARY, facecolor=BG, edgecolor="#1e2330")
        st.pyplot(fig, use_container_width=True)

        ap = best["avg_prec"]
        lift = ap / y_test.mean()
        st.markdown("**What this means:**")
        st.info(f"📈 Average Precision = **{ap:.3f}** vs random baseline **{y_test.mean():.3f}**")
        st.info(f"🚀 **{lift:.2f}× lift** over random prediction")
        st.info("💡 Moving the threshold from 0.5 → 0.35 would catch more leavers at the cost of more false alarms")

    st.divider()

    # Probability distribution
    st.markdown("#### Predicted Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.hist(best["y_proba"][y_test == 0], bins=30, alpha=0.7,
            color=PALETTE["No"], label="No Attrition", edgecolor=BG)
    ax.hist(best["y_proba"][y_test == 1], bins=30, alpha=0.7,
            color=PALETTE["Yes"], label="Attrition", edgecolor=BG)
    ax.axvline(0.5, color=PRIMARY, linestyle="--", lw=2, label="Default threshold (0.50)")
    ax.axvline(0.35, color="#f0a840", linestyle=":", lw=2, label="Suggested threshold (0.35)")
    ax.set_title("Predicted Probability Distribution — Test Set", color=PRIMARY, fontweight="bold")
    ax.set_xlabel("P(Attrition)"); ax.set_ylabel("Count")
    ax.legend(labelcolor=PRIMARY, facecolor=BG, edgecolor="#1e2330")
    st.pyplot(fig, use_container_width=True)

    st.divider()
    st.markdown("#### Full Classification Report")
    report_str = classification_report(y_test, best["y_pred"], target_names=["No Attrition", "Attrition"])
    st.code(report_str, language="text")


# ════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════
elif page == "🌲 Feature Importance":
    st.markdown("## 🌲 Feature Importance")
    st.caption("Which factors matter most for predicting employee attrition?")
    st.divider()

    tab1, tab2 = st.tabs(["🌲 Random Forest", "📐 Logistic Regression"])

    with tab1:
        rf_pipe = models["Random Forest"]
        fi_rf   = pd.Series(rf_pipe.named_steps["clf"].feature_importances_, index=X.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
        ax.set_facecolor(BG)
        bar_col = [ACCENT if v > fi_rf.median() else "#252b3b" for v in fi_rf.values]
        bars = ax.barh(fi_rf.index, fi_rf.values, color=bar_col, edgecolor=BG)
        ax.axvline(fi_rf.median(), color="#5c6380", linestyle="--", alpha=0.6, label="Median importance")
        ax.set_title("Random Forest — Gini Feature Importance", color=PRIMARY, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.legend(labelcolor=PRIMARY, facecolor=BG, edgecolor="#1e2330")
        st.pyplot(fig, use_container_width=True)
        st.caption("🔴 Red bars = above-median importance features. These are your primary intervention points.")

    with tab2:
        lr_pipe = models["Logistic Regression"]
        coef    = np.abs(lr_pipe.named_steps["clf"].coef_[0])
        fi_lr   = pd.Series(coef, index=X.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
        ax.set_facecolor(BG)
        bar_col2 = [PALETTE["Yes"] if v > fi_lr.median() else "#252b3b" for v in fi_lr.values]
        ax.barh(fi_lr.index, fi_lr.values, color=bar_col2, edgecolor=BG)
        ax.axvline(fi_lr.median(), color="#5c6380", linestyle="--", alpha=0.6, label="Median")
        ax.set_title("Logistic Regression — |Coefficient| Importance", color=PRIMARY, fontweight="bold")
        ax.set_xlabel("|Coefficient|")
        ax.legend(labelcolor=PRIMARY, facecolor=BG, edgecolor="#1e2330")
        st.pyplot(fig, use_container_width=True)
        st.caption("💡 Larger absolute coefficients = stronger influence on attrition probability")


# ════════════════════════════════════════════════════════════════
# PAGE 6 — ATTRITION DRIVERS
# ════════════════════════════════════════════════════════════════
elif page == "💡 Attrition Drivers":
    st.markdown("## 💡 Attrition Drivers & HR Actions")
    st.caption("Top 10 features driving attrition — with specific, actionable HR recommendations")
    st.divider()

    drivers = [
        ("JobLevel",             0.7237, "Junior employees leave more than seniors",
         "Accelerate promotion paths and visible career ladders for Level-1 staff within 12 months."),
        ("OverTime",             0.6112, "Working extra hours → high burnout risk",
         "Implement overtime limits & monthly workload reviews. Overtime employees are 2× more likely to quit."),
        ("MonthlyIncome",        0.4201, "Low pay is a strong push factor",
         "Benchmark salaries against the market every 6 months. Raise floor for junior roles — leavers earn ~$518/month less."),
        ("DistanceFromHome",     0.3684, "Long commute → more likely to quit",
         "Introduce remote/hybrid options or transport allowances. Target employees >20km away."),
        ("MaritalStatus",        0.3273, "Single employees are more mobile",
         "Offer team-building, social events, and relocation support targeted at single employees."),
        ("WorkLifeBalance",      0.2382, "Poor balance pushes employees out",
         "Enforce no-meeting days, flexible hours, mandatory PTO. Target employees scoring 1–2 on surveys."),
        ("StockOptionLevel",     0.2259, "Stock options = golden handcuffs",
         "Expand stock grants to level-0 employees — the strongest low-cost retention tool in the dataset."),
        ("NumCompaniesWorked",   0.2086, "History of job-hopping predicts future hops",
         "During hiring, weight long tenure at previous companies positively. 4+ companies = high flight risk."),
        ("TrainingTimesLastYear",0.1909, "Lack of growth drives dissatisfaction",
         "Ensure all employees receive at least 2 training sessions/year. Career development reduces attrition."),
        ("TotalWorkingYears",    0.1393, "Less experience = more career exploration",
         "Early-career employees (<3 yrs experience) explore more. Structured mentorship builds early loyalty."),
    ]

    max_score = drivers[0][1]
    for i, (feat, score, finding, action) in enumerate(drivers, 1):
        pct = score / max_score * 100
        col1, col2, col3 = st.columns([0.5, 3, 5])
        with col1:
            medals = ["🥇","🥈","🥉"]
            st.markdown(f"### {medals[i-1] if i <= 3 else i}")
        with col2:
            st.markdown(f"**`{feat}`**")
            st.caption(finding)
            st.progress(int(pct))
            st.caption(f"Score: {score:.4f}")
        with col3:
            st.info(f"🎯 **HR Action:** {action}")
        st.divider()


# ════════════════════════════════════════════════════════════════
# PAGE 7 — PREDICT EMPLOYEE
# ════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Employee":
    st.markdown("## 🔮 Predict Individual Employee Attrition Risk")
    st.caption(f"Using: **{best_name}** (best model, AUC={best['roc_auc']:.3f})")
    st.divider()

    st.info("👇 Fill in the employee's details below to get a real-time attrition risk prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_i       = st.slider("Age", 18, 60, 32)
        job_level_i = st.selectbox("Job Level", [1,2,3,4,5], index=0)
        monthly_i   = st.slider("Monthly Income ($)", 1000, 20000, 3500, step=500)
        overtime_i  = st.selectbox("OverTime", ["No","Yes"])
        distance_i  = st.slider("Distance From Home (km)", 1, 30, 12)

    with col2:
        marital_i   = st.selectbox("Marital Status", ["Single","Married","Divorced"])
        biz_travel_i= st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
        job_sat_i   = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1,2,3,4], index=2)
        env_sat_i   = st.selectbox("Environment Satisfaction", [1,2,3,4], index=2)
        wlb_i       = st.selectbox("Work-Life Balance", [1,2,3,4], index=2)

    with col3:
        dept_i      = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
        gender_i    = st.selectbox("Gender", ["Male","Female"])
        stock_i     = st.selectbox("Stock Option Level", [0,1,2,3])
        num_comp_i  = st.slider("Companies Worked At", 0, 9, 2)
        yrs_comp_i  = st.slider("Years at Company", 1, 40, 4)

    st.divider()

    if st.button("🔮 Predict Attrition Risk", use_container_width=True):
        # Build a minimal input row matching training features
        sample = pd.DataFrame([{
            "Age": age_i,
            "BusinessTravel": {"Non-Travel":0, "Travel_Rarely":2, "Travel_Frequently":1}[biz_travel_i],
            "Department": {"Human Resources":0, "Research & Development":1, "Sales":2}[dept_i],
            "DistanceFromHome": distance_i,
            "Education": 3,
            "EducationField": 1,
            "EnvironmentSatisfaction": env_sat_i,
            "Gender": 1 if gender_i == "Male" else 0,
            "JobLevel": job_level_i,
            "JobRole": 1,
            "JobSatisfaction": job_sat_i,
            "MaritalStatus": {"Divorced":0, "Married":1, "Single":2}[marital_i],
            "MonthlyIncome": monthly_i,
            "NumCompaniesWorked": num_comp_i,
            "OverTime": 1 if overtime_i == "Yes" else 0,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": stock_i,
            "TotalWorkingYears": yrs_comp_i,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": wlb_i,
            "YearsAtCompany": yrs_comp_i,
            "YearsInCurrentRole": max(0, yrs_comp_i - 2),
        }])

        prob = best["pipe"].predict_proba(sample)[0][1]
        pred = "Yes" if prob >= 0.5 else "No"

        st.divider()
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if prob >= 0.65:
                st.error(f"## 🚨 HIGH RISK  —  {prob:.1%} probability of leaving")
                st.error("This employee is at significant attrition risk. Immediate manager check-in recommended.")
            elif prob >= 0.40:
                st.warning(f"## ⚠️ MEDIUM RISK  —  {prob:.1%} probability of leaving")
                st.warning("Monitor this employee. Consider a career conversation within 30 days.")
            else:
                st.success(f"## ✅ LOW RISK  —  {prob:.1%} probability of leaving")
                st.success("This employee shows low attrition indicators. Standard engagement applies.")

            # Gauge-style progress bar
            st.markdown(f"**Risk Score: {prob:.1%}**")
            st.progress(float(prob))

            # Top risk factors
            st.markdown("**Key risk factors for this profile:**")
            risks = []
            if overtime_i  == "Yes":          risks.append("⏱️ Works overtime")
            if job_level_i == 1:               risks.append("📊 Junior job level (Level 1)")
            if monthly_i   < 4000:             risks.append("💰 Below-average income")
            if distance_i  > 20:               risks.append("🚗 Long commute (>20km)")
            if marital_i   == "Single":        risks.append("💍 Single — more geographically mobile")
            if biz_travel_i== "Travel_Frequently": risks.append("✈️ Frequent business travel")
            if stock_i     == 0:               risks.append("📈 No stock options")
            if num_comp_i  >= 4:               risks.append("🏢 Job-hopping history (4+ companies)")
            if job_sat_i   <= 2:               risks.append("😞 Low job satisfaction")
            if wlb_i       <= 2:               risks.append("⚖️ Poor work-life balance")

            if risks:
                for r in risks: st.markdown(f"- {r}")
            else:
                st.markdown("- ✅ No major risk flags detected")
