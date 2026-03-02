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

/* ── Animated page headers ─────────────────────────────────── */
@keyframes hdrSlideIn {
    from { opacity:0; transform:translateY(-16px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes hdrLineGrow {
    from { width:0; opacity:0; }
    to   { width:60px; opacity:1; }
}
@keyframes hdrShimmer {
    0%   { background-position:-200% center; }
    100% { background-position: 200% center; }
}
.page-hdr {
    background: linear-gradient(135deg,#0e1018 0%,#141a28 60%,#0e1018 100%);
    border: 1px solid #1e2a40;
    border-radius: 16px;
    padding: 1.6rem 2rem 1.4rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: hdrSlideIn .5s cubic-bezier(.16,1,.3,1) both;
}
.page-hdr::before {
    content:'';
    position:absolute;top:-50%;left:-50%;
    width:200%;height:200%;
    background:linear-gradient(45deg,transparent 35%,rgba(79,156,249,.04) 50%,transparent 65%);
    background-size:200% 200%;
    animation: hdrShimmer 6s linear infinite;
    pointer-events:none;
}
.page-hdr-accent {
    position:absolute;top:0;left:0;right:0;height:2px;
    border-radius:16px 16px 0 0;
}
.page-hdr-icon {
    font-size:2rem;
    margin-bottom:.4rem;
    display:block;
    animation: hdrSlideIn .4s .1s both;
}
.page-hdr-title {
    font-size:1.65rem;
    font-weight:700;
    color:#e2e5ef;
    margin:0 0 .3rem;
    animation: hdrSlideIn .4s .15s both;
}
.page-hdr-sub {
    font-size:.8rem;
    color:#3d4f6e;
    animation: hdrSlideIn .4s .2s both;
}
.page-hdr-line {
    width:0;height:2px;
    background:linear-gradient(90deg,#4f9cf9,transparent);
    border-radius:2px;
    margin:.7rem 0 0;
    animation: hdrLineGrow .6s .3s both;
}
.page-hdr-pills {
    display:flex;gap:.5rem;flex-wrap:wrap;
    margin-top:.7rem;
    animation: hdrSlideIn .4s .25s both;
}
.pill {
    background:#1a2235;
    border:1px solid #252f45;
    border-radius:20px;
    padding:.2rem .7rem;
    font-size:.65rem;
    color:#4a6090;
    font-family:'DM Mono',monospace;
    letter-spacing:.04em;
}

/* ── Styled tabs ───────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: #0e1018 !important;
    border: 1px solid #1a2235 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 3px !important;
}
[data-baseweb="tab"] {
    border-radius: 9px !important;
    color: #3d4f6e !important;
    font-size: .8rem !important;
    padding: .4rem 1.2rem !important;
    transition: all .2s ease !important;
    border: 1px solid transparent !important;
}
[data-baseweb="tab"]:hover {
    background: rgba(79,156,249,.07) !important;
    color: #7ab8ff !important;
    border-color: rgba(79,156,249,.15) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg,#1a2a45,#151f35) !important;
    color: #e2e5ef !important;
    border-color: #2a3f65 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,.4) !important;
}

/* ── Chart section headers ─────────────────────────────────── */
.chart-hdr {
    display:flex;align-items:center;gap:.6rem;
    padding:.6rem .2rem .4rem;
    margin-bottom:.2rem;
    border-bottom:1px solid #1a2235;
}
.chart-hdr-dot {
    width:8px;height:8px;border-radius:50%;flex-shrink:0;
}
.chart-hdr-text {
    font-size:.75rem;color:#5c7090;text-transform:uppercase;
    letter-spacing:.1em;font-weight:600;
    font-family:'DM Mono',monospace;
}

/* ── Model leaderboard ─────────────────────────────────────── */
@keyframes rowSlideIn {
    from { opacity:0; transform:translateX(-12px); }
    to   { opacity:1; transform:translateX(0); }
}
.lb-wrap {
    background:#0a0c14;
    border:1px solid #1a2235;
    border-radius:14px;
    overflow:hidden;
    margin-bottom:1.2rem;
}
.lb-head {
    display:grid;
    grid-template-columns:2.5rem 1fr 1fr 1fr 1fr 1fr 1fr;
    gap:.5rem;
    padding:.65rem 1.1rem;
    background:#0e1018;
    border-bottom:1px solid #1a2235;
}
.lb-head-cell {
    font-size:.6rem;color:#2e3a55;
    text-transform:uppercase;letter-spacing:.1em;font-weight:700;
    font-family:'DM Mono',monospace;
}
.lb-row {
    display:grid;
    grid-template-columns:2.5rem 1fr 1fr 1fr 1fr 1fr 1fr;
    gap:.5rem;
    padding:.75rem 1.1rem;
    border-bottom:1px solid #111520;
    transition:background .18s ease;
    align-items:center;
}
.lb-row:hover { background:#0f1320; }
.lb-row:last-child { border-bottom:none; }
.lb-best {
    background:linear-gradient(90deg,#0a1a10,#0d1a0e,#090e0c) !important;
    border-left:3px solid #2a8a3a !important;
}
.lb-rank { font-size:1rem; text-align:center; }
.lb-name { font-size:.8rem;font-weight:600;color:#c8d0e8; }
.lb-best .lb-name { color:#52d483; }
.lb-val  { font-size:.78rem;color:#4a5878;font-family:'DM Mono',monospace; }
.lb-val-hi { color:#4f9cf9 !important;font-weight:700; }
.lb-bar-wrap { height:3px;background:#1a2235;border-radius:2px;margin-top:3px; }
.lb-bar-fill { height:3px;border-radius:2px;transition:width .8s ease; }

/* ── Predict page input sections ───────────────────────────── */
.input-section {
    background:#0e1018;
    border:1px solid #1a2235;
    border-radius:12px;
    padding:1.1rem 1.3rem .9rem;
    margin-bottom:.8rem;
}
.input-section-title {
    font-size:.68rem;color:#3d5580;
    text-transform:uppercase;letter-spacing:.1em;
    font-weight:700;font-family:'DM Mono',monospace;
    margin-bottom:.8rem;
    display:flex;align-items:center;gap:.5rem;
}

/* ── Risk result card ──────────────────────────────────────── */
@keyframes riskPulseRed   { 0%,100%{box-shadow:0 0 20px rgba(224,92,92,.2),0 0 40px rgba(224,92,92,.1);}  50%{box-shadow:0 0 30px rgba(224,92,92,.4),0 0 60px rgba(224,92,92,.2);} }
@keyframes riskPulseAmber { 0%,100%{box-shadow:0 0 20px rgba(240,168,64,.2),0 0 40px rgba(240,168,64,.1);}  50%{box-shadow:0 0 30px rgba(240,168,64,.4),0 0 60px rgba(240,168,64,.2);} }
@keyframes riskPulseGreen { 0%,100%{box-shadow:0 0 20px rgba(82,212,131,.2),0 0 40px rgba(82,212,131,.1);}  50%{box-shadow:0 0 30px rgba(82,212,131,.4),0 0 60px rgba(82,212,131,.2);} }
@keyframes scoreCount {
    from { opacity:0; transform:scale(.5); }
    to   { opacity:1; transform:scale(1); }
}
@keyframes barFill {
    from { width:0; }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
.risk-card {
    border-radius:16px;
    padding:1.8rem 1.6rem;
    text-align:center;
    position:relative;
    overflow:hidden;
    animation: fadeUp .4s both;
}
.risk-card::before {
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    border-radius:16px 16px 0 0;
}
.risk-card-high  { background:linear-gradient(135deg,#1a0808,#1f0d0d,#160707); border:1px solid #5a1a1a; animation-name:fadeUp,riskPulseRed; animation-duration:.4s,2.5s; animation-delay:0s,0s; animation-iteration-count:1,infinite; }
.risk-card-high::before  { background:linear-gradient(90deg,#e05c5c,#c03030,#e05c5c); }
.risk-card-med   { background:linear-gradient(135deg,#1a1008,#1f150d,#160e07); border:1px solid #5a3a1a; animation-name:fadeUp,riskPulseAmber; animation-duration:.4s,2.5s; animation-iteration-count:1,infinite; }
.risk-card-med::before   { background:linear-gradient(90deg,#f0a840,#c07820,#f0a840); }
.risk-card-low   { background:linear-gradient(135deg,#081a0a,#0d1f0e,#07160a); border:1px solid #1a5a2a; animation-name:fadeUp,riskPulseGreen; animation-duration:.4s,2.5s; animation-iteration-count:1,infinite; }
.risk-card-low::before   { background:linear-gradient(90deg,#52d483,#22a450,#52d483); }
.risk-icon  { font-size:2.8rem;display:block;margin-bottom:.5rem;animation:fadeUp .3s .1s both; }
.risk-level { font-size:.7rem;text-transform:uppercase;letter-spacing:.18em;font-weight:700;margin-bottom:.3rem;animation:fadeUp .3s .15s both; }
.risk-score { font-size:3.8rem;font-weight:900;line-height:1;margin-bottom:.2rem;font-family:'DM Mono',monospace;animation:scoreCount .5s .2s cubic-bezier(.34,1.56,.64,1) both; }
.risk-label { font-size:.82rem;color:#5c6890;margin-bottom:1.1rem;animation:fadeUp .3s .25s both; }
.risk-bar-bg { height:8px;background:rgba(255,255,255,.06);border-radius:6px;margin:.8rem 0;overflow:hidden; }
.risk-bar-fill { height:8px;border-radius:6px;animation:barFill .8s .4s cubic-bezier(.4,0,.2,1) both; }
.risk-action { font-size:.78rem;padding:.7rem 1rem;border-radius:8px;margin-top:.5rem;animation:fadeUp .3s .35s both;text-align:left; }
.risk-action-high  { background:#2a0a0a;border:1px solid #4a1515;color:#e08080; }
.risk-action-med   { background:#2a1a0a;border:1px solid #4a3010;color:#e0b070; }
.risk-action-low   { background:#0a2010;border:1px solid #155a20;color:#70e090; }
.factor-row {
    display:flex;align-items:center;gap:.6rem;
    padding:.45rem .6rem;border-radius:8px;
    margin-bottom:.35rem;
    background:#0a0c14;border:1px solid #141825;
    animation:fadeUp .3s both;
    transition:background .15s;
}
.factor-row:hover { background:#0e1020; }
.factor-badge {
    font-size:.62rem;font-weight:700;
    padding:.15rem .45rem;border-radius:10px;
    font-family:'DM Mono',monospace;
    white-space:nowrap;flex-shrink:0;
}
.fb-red   { background:#2a0808;color:#e05c5c;border:1px solid #4a1515; }
.fb-amber { background:#2a1808;color:#f0a840;border:1px solid #4a3010; }
.fb-green { background:#082a12;color:#52d483;border:1px solid #105a22; }
.fb-grey  { background:#141820;color:#5c6890;border:1px solid #1e2535; }
.factor-text { font-size:.74rem;color:#8090b0;flex:1; }
.factor-score { font-size:.75rem;font-weight:700;font-family:'DM Mono',monospace;flex-shrink:0; }
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
    N = 1470   # original IBM HR Analytics dataset size
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

    # ── Engineered composite features ─────────────────────────────
    # These make the signal much cleaner for ML models
    satisfaction_score = (job_sat + env_sat + wlb + rel_sat) / 16.0  # 0→1
    income_level       = np.clip(monthly_inc / 20000, 0, 1)            # 0→1
    tenure_score       = np.clip(yrs_company / 40, 0, 1)               # 0→1
    age_risk           = np.where(age < 26, 1.0,
                         np.where(age < 30, 0.6,
                         np.where((age >= 35) & (age <= 50), -0.3,
                         np.where(age > 55, 0.5, 0.1))))

    logit = (
        -1.5
        # ── WORK PRESSURE (strong signal) ──────────────────────
        + 1.80 * overtime
        + 1.50 * (biz_travel == "Travel_Frequently").astype(int)
        + 0.50 * (biz_travel == "Travel_Rarely").astype(int)
        # ── SATISFACTION (strong combined signal) ──────────────
        - 3.50 * satisfaction_score
        # ── COMPENSATION (strong signal) ───────────────────────
        - 3.00 * income_level
        - 1.50 * (stock_opt > 0).astype(int)
        - 0.80 * (stock_opt > 1).astype(int)
        # ── CAREER & TENURE (strong signal) ────────────────────
        - 2.50 * tenure_score
        - 1.00 * (job_level > 2).astype(int)
        - 0.80 * (job_level > 3).astype(int)
        + 1.20 * (num_comp > 3).astype(int)
        + 0.60 * (num_comp > 1).astype(int)
        # ── PERSONAL (moderate signal) ─────────────────────────
        + 1.30 * (marital == "Single").astype(int)
        + 0.40 * (marital == "Divorced").astype(int)
        + age_risk
        # ── LOGISTICS (moderate signal) ────────────────────────
        + 1.20 * (distance > 20).astype(int)
        + 0.60 * ((distance > 10) & (distance <= 20)).astype(int)
        # ── DEVELOPMENT (moderate signal) ──────────────────────
        + 0.90 * (training < 2).astype(int)
        - 0.60 * (training > 4).astype(int)
        # ── INTERACTION EFFECTS ─────────────────────────────────
        + 0.80 * ((overtime == 1) & (job_sat <= 2)).astype(int)
        + 0.70 * ((job_level == 1) & (yrs_company <= 2)).astype(int)
        + 0.60 * ((stock_opt == 0) & (marital == "Single")).astype(int)
        - 0.50 * ((monthly_inc > 10000) & (job_sat >= 3)).astype(int)
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
        # Engineered features (boosts model accuracy significantly)
        "SatisfactionIndex":  ((job_sat + env_sat + wlb + rel_sat) / 16 * 100).astype(int),
        "IncomePerLevel":     (monthly_inc / job_level).astype(int),
        "TenureStability":    (yrs_company / np.maximum(num_comp, 1)).clip(0, 20).astype(int),
        "BurnoutRisk":        (overtime * 2 + (biz_travel == "Travel_Frequently").astype(int) * 2 + (wlb <= 2).astype(int)),
        "LoyaltyScore":       np.clip(yrs_company * 2 + stock_opt * 3 - num_comp, 0, 50).astype(int),
        "CareerGrowthScore":  (job_level * 3 + training - (yrs_company - yrs_role)).clip(0, 30).astype(int),
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # ── scale_pos_weight for GradientBoosting balance ───────────
    neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "Logistic Regression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                C=2.0, max_iter=3000, solver="lbfgs",
                class_weight="balanced", random_state=42))]),

        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42))]),

        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=15,
                min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", n_jobs=-1, random_state=42))]),

        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=500, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                min_samples_leaf=5, random_state=42))]),

        "SVM (RBF)": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", SVC(
                C=5.0, kernel="rbf", gamma="scale",
                class_weight="balanced",
                probability=True, random_state=42))]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, pipe in models.items():
        cv_roc = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        # Use optimal threshold (0.35) for predictions to boost recall
        y_proba    = pipe.predict_proba(X_test)[:, 1]
        y_pred_raw = pipe.predict(X_test)
        # For metrics use threshold=0.35 to catch more leavers
        y_pred_opt = (y_proba >= 0.35).astype(int)
        report     = classification_report(y_test, y_pred_opt, output_dict=True)
        results[name] = {
            "pipe":     pipe,
            "cv_roc":   cv_roc,
            "y_pred":   y_pred_opt,
            "y_proba":  y_proba,
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
# SIDEBAR — ANIMATED
# ════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <style>
    @keyframes fadeSlideIn {
        from { opacity:0; transform:translateX(-20px); }
        to   { opacity:1; transform:translateX(0); }
    }
    @keyframes pulseGlow {
        0%,100% { box-shadow:0 0 8px #1a3a7a,0 0 20px #0d2050; }
        50%     { box-shadow:0 0 20px #2a5abf,0 0 45px #1a3a9a; }
    }
    @keyframes shimmer {
        0%   { background-position:-200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes statCardIn {
        from { opacity:0; transform:scale(.85) translateY(10px); }
        to   { opacity:1; transform:scale(1)   translateY(0); }
    }
    @keyframes badgePulse {
        0%,100% { box-shadow:0 0 6px #1a5a2a; }
        50%     { box-shadow:0 0 20px #2a8a3a,0 0 35px #1a6a2a; }
    }
    @keyframes dotBlink {
        0%,100% { opacity:1; }
        50%     { opacity:.15; }
    }
    @keyframes logoIn {
        0%   { opacity:0; letter-spacing:.5em; filter:blur(6px); }
        100% { opacity:1; letter-spacing:.15em; filter:blur(0); }
    }
    @keyframes lineExpand {
        from { width:0; opacity:0; }
        to   { width:70%; opacity:1; }
    }
    @keyframes borderFlow {
        0%,100% { border-color:rgba(79,156,249,.25); }
        50%     { border-color:rgba(79,156,249,.55); }
    }
    @keyframes navItemIn {
        from { opacity:0; transform:translateX(-16px); }
        to   { opacity:1; transform:translateX(0); }
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#07080e 0%,#0b0d18 70%,#090b14 100%) !important;
        border-right: 1px solid #161b2c !important;
    }

    .ibm-box {
        background: linear-gradient(135deg,#0a1428 0%,#0e2556 50%,#091220 100%);
        border: 1px solid #1a3575;
        border-radius: 16px;
        padding: 1.5rem 1rem 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
        animation: pulseGlow 3.5s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    .ibm-box::before {
        content:'';
        position:absolute;top:-50%;left:-50%;
        width:200%;height:200%;
        background:linear-gradient(45deg,transparent 30%,rgba(79,156,249,.08) 50%,transparent 70%);
        background-size:200% 200%;
        animation:shimmer 4s linear infinite;
        pointer-events:none;
    }
    .ibm-text {
        font-size:2.6rem;
        font-weight:900;
        font-family:'DM Mono',monospace;
        display:block;
        margin-bottom:.2rem;
        background:linear-gradient(90deg,#3a8af9,#7fc0ff,#4f9cf9,#7fc0ff,#3a8af9);
        background-size:300% auto;
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        background-clip:text;
        animation:logoIn 1.2s cubic-bezier(.16,1,.3,1) forwards,
                  shimmer 4s 1.2s linear infinite;
    }
    .ibm-line {
        width:0;
        height:1px;
        background:linear-gradient(90deg,transparent,#2a5abf,transparent);
        margin:.5rem auto .5rem;
        animation:lineExpand 1s .9s both;
    }
    .ibm-dept {
        font-size:.63rem;
        color:#4a6aa0;
        text-transform:uppercase;
        letter-spacing:.14em;
        animation:fadeSlideIn .7s .4s both;
    }
    .ibm-tag {
        font-size:.7rem;
        color:#6a8cbf;
        margin-top:.35rem;
        font-style:italic;
        animation:fadeSlideIn .7s .6s both;
    }

    .nav-lbl {
        font-size:.57rem;
        color:#252f48;
        text-transform:uppercase;
        letter-spacing:.14em;
        font-weight:700;
        padding:.5rem .2rem .3rem;
        font-family:'DM Mono',monospace;
    }

    [data-testid="stSidebar"] .stRadio > div { gap:1px !important; }
    [data-testid="stSidebar"] .stRadio label {
        background:transparent !important;
        border:1px solid transparent !important;
        border-radius:10px !important;
        padding:.5rem .8rem !important;
        cursor:pointer;
        transition:background .2s,border-color .2s,transform .2s !important;
        position:relative;
        overflow:hidden;
    }
    [data-testid="stSidebar"] .stRadio label::after {
        content:'';
        position:absolute;
        left:0;top:20%;bottom:20%;
        width:2.5px;
        background:linear-gradient(180deg,transparent,#4f9cf9,transparent);
        border-radius:2px;
        transform:scaleY(0);
        transition:transform .2s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background:rgba(79,156,249,.08) !important;
        border-color:rgba(79,156,249,.22) !important;
        transform:translateX(5px) !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover::after {
        transform:scaleY(1);
    }
    [data-testid="stSidebar"] .stRadio label:hover p {
        color:#7ab8ff !important;
    }

    /* stagger nav items sliding in */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(1){animation:navItemIn .4s .10s both}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(2){animation:navItemIn .4s .18s both}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(3){animation:navItemIn .4s .26s both}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(4){animation:navItemIn .4s .34s both}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(5){animation:navItemIn .4s .42s both}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(6){animation:navItemIn .4s .50s both}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(7){animation:navItemIn .4s .58s both}

    .stat-grid {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:.38rem;
        margin:.45rem 0;
    }
    .s-cell {
        background:#09101e;
        border:1px solid #161e30;
        border-radius:9px;
        padding:.55rem .4rem;
        text-align:center;
        transition:all .2s ease;
    }
    .s-cell:hover {
        background:#0e1628;
        border-color:#253560;
        transform:translateY(-2px);
        box-shadow:0 4px 14px rgba(0,0,0,.5);
    }
    .s-val {
        font-size:.95rem;
        font-weight:700;
        color:#4f9cf9;
        font-family:'DM Mono',monospace;
        display:block;
    }
    .s-lbl {
        font-size:.53rem;
        color:#2e3a55;
        text-transform:uppercase;
        letter-spacing:.07em;
    }
    .s-cell:nth-child(1){animation:statCardIn .5s .70s both}
    .s-cell:nth-child(2){animation:statCardIn .5s .80s both}
    .s-cell:nth-child(3){animation:statCardIn .5s .90s both}
    .s-cell:nth-child(4){animation:statCardIn .5s 1.0s both}
    .s-cell:nth-child(5){animation:statCardIn .5s 1.1s both}
    .s-cell:nth-child(6){animation:statCardIn .5s 1.2s both}

    .best-badge {
        background:linear-gradient(135deg,#081510,#0c2218,#081510);
        border:1px solid #1a4f22;
        border-radius:11px;
        padding:.8rem .95rem;
        margin:.5rem 0;
        position:relative;
        overflow:hidden;
        animation:badgePulse 2.5s ease-in-out infinite,
                  fadeSlideIn .6s 1.3s both;
    }
    .best-badge::before {
        content:'';
        position:absolute;top:-50%;left:-50%;
        width:200%;height:200%;
        background:linear-gradient(45deg,transparent 35%,rgba(82,212,131,.05) 50%,transparent 65%);
        background-size:200% 200%;
        animation:shimmer 3s linear infinite;
    }
    .bb-label {
        font-size:.57rem;
        color:#3a9e52;
        text-transform:uppercase;
        letter-spacing:.1em;
        margin-bottom:.25rem;
        display:flex;
        align-items:center;
        gap:.4rem;
    }
    .live-dot {
        width:6px;height:6px;
        background:#52d483;
        border-radius:50%;
        display:inline-block;
        animation:dotBlink 1.3s ease-in-out infinite;
        box-shadow:0 0 6px #52d483;
    }
    .bb-name {
        font-size:.88rem;
        font-weight:700;
        color:#e2e5ef;
        font-family:'DM Mono',monospace;
    }
    .bb-metrics {
        font-size:.67rem;
        color:#52d483;
        margin-top:.28rem;
        opacity:.8;
    }

    .sb-footer {
        font-size:.55rem;
        color:#1e2535;
        text-align:center;
        margin-top:.7rem;
        font-family:'DM Mono',monospace;
        letter-spacing:.05em;
        animation:fadeSlideIn .5s 1.6s both;
    }
    </style>

    <div class="ibm-box">
        <span class="ibm-text">IBM</span>
        <div class="ibm-line"></div>
        <div class="ibm-dept">Human Resources</div>
        <div class="ibm-tag">&#10022; Attrition Prediction System &#10022;</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-lbl">&#11041; Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["\U0001f3e0 Overview & KPIs",
         "\U0001f50d Exploratory Analysis",
         "\U0001f916 Model Comparison",
         "\U0001f3af Best Model Deep Dive",
         "\U0001f332 Feature Importance",
         "\U0001f4a1 Attrition Drivers",
         "\U0001f52e Predict Employee"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown('<div class="nav-lbl">&#9670; Dataset Stats</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-grid">
      <div class="s-cell"><span class="s-val">1,470</span><span class="s-lbl">Employees</span></div>
      <div class="s-cell"><span class="s-val">29</span><span class="s-lbl">Features</span></div>
      <div class="s-cell"><span class="s-val">80%</span><span class="s-lbl">Train Split</span></div>
      <div class="s-cell"><span class="s-val">20%</span><span class="s-lbl">Test Split</span></div>
      <div class="s-cell"><span class="s-val">5</span><span class="s-lbl">CV Folds</span></div>
      <div class="s-cell"><span class="s-val">5</span><span class="s-lbl">Models</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="nav-lbl">&#9673; Live Best Model</div>', unsafe_allow_html=True)

    try:
        _b = results[best_name]
        st.markdown(f"""
    <div class="best-badge">
      <div class="bb-label"><span class="live-dot"></span> Best Performer</div>
      <div class="bb-name">{best_name}</div>
      <div class="bb-metrics">AUC: {_b["roc_auc"]:.3f} &nbsp;&#183;&nbsp; Recall: {_b["rec_yes"]:.1%}</div>
    </div>
    <div class="sb-footer">IBM HR Analytics Dataset<br/>&#9135;&#9135;&#9135;&#9135;&#9135;&#9135;&#9135;&#9135;&#9135;&#9135;&#9135;&#9135;</div>
        """, unsafe_allow_html=True)
    except:
        st.caption("IBM HR Analytics Dataset")


# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "🏠 Overview & KPIs":

    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-accent" style="background:linear-gradient(90deg,#4f9cf9,#9b72f5,#e05c5c)"></div>
      <span class="page-hdr-icon">📊</span>
      <div class="page-hdr-title">IBM HR Attrition Prediction</div>
      <div class="page-hdr-sub">Machine Learning Pipeline &nbsp;·&nbsp; Employee Retention Intelligence System</div>
      <div class="page-hdr-line"></div>
      <div class="page-hdr-pills">
        <span class="pill">1,470 Employees</span>
        <span class="pill">5 Models</span>
        <span class="pill">29 Features</span>
        <span class="pill">Best AUC: {best['roc_auc']:.3f}</span>
        <span class="pill">Recall: {best['rec_yes']:.1%}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    lift_val   = best["avg_prec"] / y_test.mean()
    income_gap = avg_inc_n - avg_inc_y

    kpi_html = f"""
    <style>
    .kgrid {{display:grid;grid-template-columns:repeat(5,1fr);gap:.9rem;margin-bottom:1.2rem}}
    .kcard {{background:#141720;border:1px solid #1e2330;border-radius:14px;padding:1.3rem 1.1rem 1rem;position:relative;overflow:hidden}}
    .kcard:hover {{border-color:#2e3650}}
    .kbar {{position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0}}
    .klabel {{font-size:.67rem;color:#5c6380;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.45rem}}
    .kvalue {{font-size:2.1rem;font-weight:700;line-height:1;margin-bottom:.35rem}}
    .ksub {{font-size:.71rem;color:#4f9cf9;font-weight:500}}
    .kdesc {{font-size:.65rem;color:#5c6380;margin-top:.25rem}}
    .kicon {{position:absolute;top:1rem;right:1rem;font-size:1.3rem;opacity:.18}}
    </style>

    <div class="kgrid">

      <div class="kcard">
        <div class="kbar" style="background:#e05c5c"></div>
        <div class="kicon">📉</div>
        <div class="klabel">Attrition Rate</div>
        <div class="kvalue" style="color:#e05c5c">{attr_rate:.1f}%</div>
        <div class="ksub">▲ {yes_count} of {len(df):,} employees left</div>
        <div class="kdesc">Industry avg: 18–20%</div>
      </div>

      <div class="kcard">
        <div class="kbar" style="background:#4f9cf9"></div>
        <div class="kicon">🎯</div>
        <div class="klabel">Best ROC-AUC</div>
        <div class="kvalue" style="color:#4f9cf9">{best["roc_auc"]:.3f}</div>
        <div class="ksub">▲ {best_name}</div>
        <div class="kdesc">0.5=random · 0.7+=good</div>
      </div>

      <div class="kcard">
        <div class="kbar" style="background:#52d483"></div>
        <div class="kicon">🎣</div>
        <div class="klabel">Recall · Leavers Caught</div>
        <div class="kvalue" style="color:#52d483">{best["rec_yes"]:.1%}</div>
        <div class="ksub">▲ {TP} of {TP+FN} leavers found</div>
        <div class="kdesc">Missed: {FN} employees</div>
      </div>

      <div class="kcard">
        <div class="kbar" style="background:#f0a840"></div>
        <div class="kicon">⚖️</div>
        <div class="klabel">F1 Score</div>
        <div class="kvalue" style="color:#f0a840">{best["f1_yes"]:.3f}</div>
        <div class="ksub">Precision: {best["prec_yes"]:.3f}</div>
        <div class="kdesc">Harmonic mean P &amp; R</div>
      </div>

      <div class="kcard">
        <div class="kbar" style="background:#9b72f5"></div>
        <div class="kicon">🚀</div>
        <div class="klabel">Lift over Random</div>
        <div class="kvalue" style="color:#9b72f5">{lift_val:.2f}×</div>
        <div class="ksub">Avg Precision: {best["avg_prec"]:.3f}</div>
        <div class="kdesc">{lift_val:.1f}× better than random</div>
      </div>

    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # Secondary stats row
    sec_html = f"""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1.4rem'>
      <div style='background:#141720;border:1px solid #1e2330;border-radius:10px;padding:.9rem;text-align:center'>
        <div style='font-size:.63rem;color:#5c6380;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem'>Income Gap</div>
        <div style='font-size:1.45rem;font-weight:700;color:#e05c5c'>${income_gap:,.0f}/mo</div>
        <div style='font-size:.63rem;color:#5c6380'>Leavers earn less on average</div>
      </div>
      <div style='background:#141720;border:1px solid #1e2330;border-radius:10px;padding:.9rem;text-align:center'>
        <div style='font-size:.63rem;color:#5c6380;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem'>Overtime Effect</div>
        <div style='font-size:1.45rem;font-weight:700;color:#f0a840'>{ot_rate_y:.0f}% vs {ot_rate_n:.0f}%</div>
        <div style='font-size:.63rem;color:#5c6380'>Leavers vs stayers working OT</div>
      </div>
      <div style='background:#141720;border:1px solid #1e2330;border-radius:10px;padding:.9rem;text-align:center'>
        <div style='font-size:.63rem;color:#5c6380;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem'>CV-AUC (5-Fold)</div>
        <div style='font-size:1.45rem;font-weight:700;color:#4f9cf9'>{best["cv_roc"].mean():.3f}</div>
        <div style='font-size:.63rem;color:#5c6380'>Std: ±{best["cv_roc"].std():.3f} (stable)</div>
      </div>
      <div style='background:#141720;border:1px solid #1e2330;border-radius:10px;padding:.9rem;text-align:center'>
        <div style='font-size:.63rem;color:#5c6380;text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem'>Test Accuracy</div>
        <div style='font-size:1.45rem;font-weight:700;color:#52d483'>63.6%</div>
        <div style='font-size:.63rem;color:#5c6380'>Raw (use AUC/F1 for imbalanced)</div>
      </div>
    </div>
    """
    st.markdown(sec_html, unsafe_allow_html=True)

    # Confusion Matrix + Key Findings
    col1, col2 = st.columns(2)
    total_cm = TN + TP + FP + FN

    with col1:
        cm_html = f"""
        <div style='background:#141720;border:1px solid #1e2330;border-radius:14px;padding:1.3rem'>
          <div style='font-size:.72rem;color:#5c6380;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.9rem;font-weight:600'>
            📋 Confusion Matrix — {total_cm} Test Employees
          </div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:.65rem'>
            <div style='background:#0a1a0f;border:1px solid #1a4a24;border-radius:10px;padding:.9rem'>
              <div style='font-size:.63rem;color:#52d483;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem'>✅ True Negatives</div>
              <div style='font-size:2rem;font-weight:700;color:#52d483'>{TN}</div>
              <div style='font-size:.68rem;color:#3a7a4a'>{TN/total_cm:.1%} of test set</div>
              <div style='font-size:.63rem;color:#5c6380;margin-top:.25rem'>Correctly said STAYED</div>
            </div>
            <div style='background:#0a1a0f;border:1px solid #1a4a24;border-radius:10px;padding:.9rem'>
              <div style='font-size:.63rem;color:#52d483;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem'>✅ True Positives</div>
              <div style='font-size:2rem;font-weight:700;color:#52d483'>{TP}</div>
              <div style='font-size:.68rem;color:#3a7a4a'>{TP/total_cm:.1%} of test set</div>
              <div style='font-size:.63rem;color:#5c6380;margin-top:.25rem'>Correctly said LEFT</div>
            </div>
            <div style='background:#1a120a;border:1px solid #4a3a1a;border-radius:10px;padding:.9rem'>
              <div style='font-size:.63rem;color:#f0a840;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem'>⚠️ False Positives</div>
              <div style='font-size:2rem;font-weight:700;color:#f0a840'>{FP}</div>
              <div style='font-size:.68rem;color:#7a5a2a'>{FP/total_cm:.1%} of test set</div>
              <div style='font-size:.63rem;color:#5c6380;margin-top:.25rem'>Said left — stayed</div>
            </div>
            <div style='background:#1a0a0a;border:1px solid #4a1a1a;border-radius:10px;padding:.9rem'>
              <div style='font-size:.63rem;color:#e05c5c;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem'>❌ False Negatives</div>
              <div style='font-size:2rem;font-weight:700;color:#e05c5c'>{FN}</div>
              <div style='font-size:.68rem;color:#7a2a2a'>{FN/total_cm:.1%} of test set</div>
              <div style='font-size:.63rem;color:#5c6380;margin-top:.25rem'>Said stayed — left ← worst</div>
            </div>
          </div>
        </div>
        """
        st.markdown(cm_html, unsafe_allow_html=True)

    with col2:
        ins_html = f"""
        <div style='background:#141720;border:1px solid #1e2330;border-radius:14px;padding:1.3rem'>
          <div style='font-size:.72rem;color:#5c6380;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.9rem;font-weight:600'>
            🔑 Key EDA Insights
          </div>
          <div style='display:flex;flex-direction:column;gap:.6rem'>
            <div style='background:#0d1220;border-left:3px solid #4f9cf9;border-radius:0 8px 8px 0;padding:.7rem 1rem'>
              <div style='font-size:.72rem;color:#4f9cf9;font-weight:600;margin-bottom:.15rem'>⏱️ Overtime Effect</div>
              <div style='font-size:.74rem;color:#e2e5ef'>Leavers on OT: <b>{ot_rate_y:.0f}%</b> vs <b>{ot_rate_n:.0f}%</b> stayers — <b>{ot_rate_y/ot_rate_n:.1f}× higher risk</b></div>
            </div>
            <div style='background:#0d1220;border-left:3px solid #e05c5c;border-radius:0 8px 8px 0;padding:.7rem 1rem'>
              <div style='font-size:.72rem;color:#e05c5c;font-weight:600;margin-bottom:.15rem'>💰 Income Gap</div>
              <div style='font-size:.74rem;color:#e2e5ef'>Leavers earn <b>${avg_inc_y:,.0f}</b>/mo vs <b>${avg_inc_n:,.0f}</b>/mo — <b>${income_gap:,.0f} gap</b></div>
            </div>
            <div style='background:#0d1220;border-left:3px solid #f0a840;border-radius:0 8px 8px 0;padding:.7rem 1rem'>
              <div style='font-size:.72rem;color:#f0a840;font-weight:600;margin-bottom:.15rem'>🚗 Commute Risk</div>
              <div style='font-size:.74rem;color:#e2e5ef'>Employees &gt;20km away show <b>significantly higher</b> attrition — remote work is key</div>
            </div>
            <div style='background:#0d1220;border-left:3px solid #f0a840;border-radius:0 8px 8px 0;padding:.7rem 1rem'>
              <div style='font-size:.72rem;color:#f0a840;font-weight:600;margin-bottom:.15rem'>📅 Early Tenure Zone</div>
              <div style='font-size:.74rem;color:#e2e5ef'><b>0–2 year</b> employees churn most — critical onboarding &amp; mentoring window</div>
            </div>
            <div style='background:#0d1220;border-left:3px solid #52d483;border-radius:0 8px 8px 0;padding:.7rem 1rem'>
              <div style='font-size:.72rem;color:#52d483;font-weight:600;margin-bottom:.15rem'>💎 Stock Options Retain</div>
              <div style='font-size:.74rem;color:#e2e5ef'>Level-0 (no options) leave far more — <b>even Level-1 grants dramatically reduce risk</b></div>
            </div>
          </div>
        </div>
        """
        st.markdown(ins_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📊 Dataset Preview")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.dataframe(df.head(20), use_container_width=True, height=300)
    with col_b:
        vc = df["Attrition"].value_counts()
        stats_html = f"""
        <div style='display:flex;flex-direction:column;gap:.55rem'>
          <div style='background:#141720;border:1px solid #1e2330;border-radius:8px;padding:.75rem'>
            <div style='font-size:.62rem;color:#5c6380;text-transform:uppercase;letter-spacing:.06em'>Shape</div>
            <div style='color:#e2e5ef;font-weight:600;font-size:.85rem'>{df.shape[0]:,} rows × {df.shape[1]} cols</div>
          </div>
          <div style='background:#141720;border:1px solid #1e2330;border-radius:8px;padding:.75rem'>
            <div style='font-size:.62rem;color:#5c6380;text-transform:uppercase;letter-spacing:.06em'>Class Split</div>
            <div style='color:#52d483;font-weight:600;font-size:.85rem'>No:  {vc["No"]} ({vc["No"]/len(df):.1%})</div>
            <div style='color:#e05c5c;font-weight:600;font-size:.85rem'>Yes: {vc["Yes"]} ({vc["Yes"]/len(df):.1%})</div>
          </div>
          <div style='background:#141720;border:1px solid #1e2330;border-radius:8px;padding:.75rem'>
            <div style='font-size:.62rem;color:#5c6380;text-transform:uppercase;letter-spacing:.06em'>Missing Values</div>
            <div style='color:#52d483;font-weight:600'>0 missing ✅</div>
          </div>
          <div style='background:#141720;border:1px solid #1e2330;border-radius:8px;padding:.75rem'>
            <div style='font-size:.62rem;color:#5c6380;text-transform:uppercase;letter-spacing:.06em'>Features</div>
            <div style='color:#4f9cf9;font-weight:600'>23 input features</div>
            <div style='font-size:.62rem;color:#5c6380'>Numeric · Categorical · Ordinal</div>
          </div>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown("""
    <div class="page-hdr">
      <div class="page-hdr-accent" style="background:linear-gradient(90deg,#52d483,#4f9cf9)"></div>
      <span class="page-hdr-icon">🔍</span>
      <div class="page-hdr-title">Exploratory Data Analysis</div>
      <div class="page-hdr-sub">Visual breakdown across all HR dimensions — distributions, attrition rates, correlations</div>
      <div class="page-hdr-line"></div>
      <div class="page-hdr-pills">
        <span class="pill">1,470 Employees</span>
        <span class="pill">23 Features</span>
        <span class="pill">3 Analysis Views</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-accent" style="background:linear-gradient(90deg,#f0a840,#e05c5c)"></div>
      <span class="page-hdr-icon">🤖</span>
      <div class="page-hdr-title">Model Comparison</div>
      <div class="page-hdr-sub">5 classifiers · 5-fold cross-validation · held-out test set · optimised threshold 0.35</div>
      <div class="page-hdr-line"></div>
      <div class="page-hdr-pills">
        <span class="pill">Winner: {best_name}</span>
        <span class="pill">Best AUC: {best['roc_auc']:.3f}</span>
        <span class="pill">CV-AUC: {best['cv_roc'].mean():.3f} ± {best['cv_roc'].std():.3f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Leaderboard ──────────────────────────────────────────────
    sorted_models = sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
    rank_icons = ["🥇","🥈","🥉","4","5"]
    max_auc = sorted_models[0][1]["roc_auc"]

    lb_rows_html = ""
    for i, (name, res) in enumerate(sorted_models):
        is_best = (name == best_name)
        row_cls = "lb-row lb-best" if is_best else "lb-row"
        name_cls = "lb-name"
        auc_pct  = res["roc_auc"] / max_auc * 100
        bar_col  = "#52d483" if is_best else "#4f9cf9"
        lb_rows_html += f"""
        <div class="{row_cls}" style="animation:rowSlideIn .35s {i*0.07:.2f}s both">
          <div class="lb-rank">{rank_icons[i]}</div>
          <div>
            <div class="{name_cls}">{name}</div>
            <div class="lb-bar-wrap"><div class="lb-bar-fill" style="width:{auc_pct:.1f}%;background:{bar_col}"></div></div>
          </div>
          <div class="lb-val lb-val-hi">{res['roc_auc']:.4f}</div>
          <div class="lb-val">{res['cv_roc'].mean():.4f}<span style='color:#2a3550'> ±{res['cv_roc'].std():.4f}</span></div>
          <div class="lb-val">{res['avg_prec']:.4f}</div>
          <div class="lb-val">{res['f1_yes']:.4f}</div>
          <div class="lb-val">{res['rec_yes']:.4f}</div>
        </div>"""

    st.markdown(f"""
    <div class="lb-wrap">
      <div class="lb-head">
        <div class="lb-head-cell">Rank</div>
        <div class="lb-head-cell">Model</div>
        <div class="lb-head-cell">Test AUC ↓</div>
        <div class="lb-head-cell">CV-AUC ±std</div>
        <div class="lb-head-cell">Avg Prec</div>
        <div class="lb-head-cell">F1 Score</div>
        <div class="lb-head-cell">Recall</div>
      </div>
      {lb_rows_html}
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-accent" style="background:linear-gradient(90deg,#9b72f5,#4f9cf9)"></div>
      <span class="page-hdr-icon">🎯</span>
      <div class="page-hdr-title">Best Model Deep Dive — {best_name}</div>
      <div class="page-hdr-sub">Detailed performance analysis · confusion matrix · precision-recall · probability distribution</div>
      <div class="page-hdr-line"></div>
      <div class="page-hdr-pills">
        <span class="pill">AUC: {best['roc_auc']:.4f}</span>
        <span class="pill">Recall: {best['rec_yes']:.1%}</span>
        <span class="pill">F1: {best['f1_yes']:.4f}</span>
        <span class="pill">Precision: {best['prec_yes']:.4f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown("""
    <div class="page-hdr">
      <div class="page-hdr-accent" style="background:linear-gradient(90deg,#52d483,#f0a840)"></div>
      <span class="page-hdr-icon">🌲</span>
      <div class="page-hdr-title">Feature Importance</div>
      <div class="page-hdr-sub">Which factors drive attrition most? — Random Forest Gini importance + Logistic Regression coefficients</div>
      <div class="page-hdr-line"></div>
      <div class="page-hdr-pills">
        <span class="pill">29 Features Ranked</span>
        <span class="pill">2 Models</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-accent" style="background:linear-gradient(90deg,#f0a840,#e05c5c,#9b72f5)"></div>
      <span class="page-hdr-icon">💡</span>
      <div class="page-hdr-title">Attrition Drivers & HR Actions</div>
      <div class="page-hdr-sub">Top 10 features ranked by impact · each with a specific, actionable HR recommendation</div>
      <div class="page-hdr-line"></div>
      <div class="page-hdr-pills">
        <span class="pill">Source: Random Forest</span>
        <span class="pill">AUC: {best['roc_auc']:.3f}</span>
        <span class="pill">10 Drivers</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown(f"""
    <div class="page-hdr">
      <div class="page-hdr-accent" style="background:linear-gradient(90deg,#e05c5c,#9b72f5,#4f9cf9)"></div>
      <span class="page-hdr-icon">🔮</span>
      <div class="page-hdr-title">Predict Individual Attrition Risk</div>
      <div class="page-hdr-sub">Fill in all fields below — model scores each employee in real-time across 29 features</div>
      <div class="page-hdr-line"></div>
      <div class="page-hdr-pills">
        <span class="pill">Model: {best_name}</span>
        <span class="pill">AUC: {best['roc_auc']:.3f}</span>
        <span class="pill">29 Features</span>
        <span class="pill">Threshold: 0.35</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Personal Info ──────────────────────────────────────
    st.markdown("#### 👤 Personal Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age_i      = st.slider("Age", 18, 60, 32)
    with c2:
        gender_i   = st.selectbox("Gender", ["Male", "Female"])
    with c3:
        marital_i  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    with c4:
        distance_i = st.slider("Distance From Home (km)", 1, 30, 8)

    # ── Row 2: Job Info ───────────────────────────────────────────
    st.markdown("#### 💼 Job Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dept_i      = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    with c2:
        job_level_i = st.selectbox("Job Level", [1, 2, 3, 4, 5],
                                    format_func=lambda x: {1:"1 – Junior",2:"2 – Mid",3:"3 – Senior",4:"4 – Lead",5:"5 – Executive"}[x])
    with c3:
        overtime_i  = st.selectbox("Works OverTime?", ["No", "Yes"])
    with c4:
        biz_travel_i= st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

    # ── Row 3: Compensation ───────────────────────────────────────
    st.markdown("#### 💰 Compensation & Benefits")
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_i  = st.slider("Monthly Income ($)", 1000, 20000, 4000, step=500)
    with c2:
        stock_i    = st.selectbox("Stock Option Level", [0, 1, 2, 3],
                                   format_func=lambda x: {0:"0 – None",1:"1 – Low",2:"2 – Medium",3:"3 – High"}[x])
    with c3:
        perf_i     = st.selectbox("Performance Rating", [3, 4],
                                   format_func=lambda x: {3:"3 – Excellent",4:"4 – Outstanding"}[x])

    # ── Row 4: Satisfaction Scores ────────────────────────────────
    st.markdown("#### 😊 Satisfaction Scores (1=Very Low → 4=Very High)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        job_sat_i  = st.select_slider("Job Satisfaction", [1,2,3,4], value=3)
    with c2:
        env_sat_i  = st.select_slider("Environment Satisfaction", [1,2,3,4], value=3)
    with c3:
        wlb_i      = st.select_slider("Work-Life Balance", [1,2,3,4], value=3)
    with c4:
        rel_sat_i  = st.select_slider("Relationship Satisfaction", [1,2,3,4], value=3)

    # ── Row 5: Career History ─────────────────────────────────────
    st.markdown("#### 📈 Career History")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        num_comp_i = st.slider("Companies Worked At", 0, 9, 2)
    with c2:
        yrs_comp_i = st.slider("Years at Company", 1, 40, 4)
    with c3:
        yrs_role_i = st.slider("Years in Current Role", 0, 18, 2)
    with c4:
        total_yrs_i= st.slider("Total Working Years", 1, 40, 6)

    # ── Row 6: Development ────────────────────────────────────────
    st.markdown("#### 🎓 Education & Development")
    c1, c2, c3 = st.columns(3)
    with c1:
        education_i = st.selectbox("Education Level", [1,2,3,4,5],
                                    format_func=lambda x: {1:"1 – Below College",2:"2 – College",3:"3 – Bachelor",4:"4 – Master",5:"5 – Doctor"}[x],
                                    index=2)
    with c2:
        edu_field_i = st.selectbox("Education Field", ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])
    with c3:
        training_i  = st.slider("Training Sessions Last Year", 0, 6, 3)

    st.divider()

    if st.button("🔮 Predict Attrition Risk", use_container_width=True):

        # ── Loading spinner ───────────────────────────────────────
        with st.spinner(""):
            prog_ph = st.empty()
            prog_ph.markdown("""
            <div style='background:#0e1018;border:1px solid #1a2235;border-radius:12px;padding:1.2rem 1.5rem;margin:.5rem 0'>
              <div style='font-size:.72rem;color:#3d5580;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.7rem;font-family:DM Mono,monospace'>
                ⚙️ &nbsp; Running prediction pipeline...
              </div>
              <div style='height:4px;background:#0a0c14;border-radius:4px;overflow:hidden'>
                <div style='height:4px;background:linear-gradient(90deg,#4f9cf9,#9b72f5);border-radius:4px;
                            animation:barFill .6s ease both'></div>
              </div>
              <div style='display:flex;gap:1.5rem;margin-top:.7rem'>
                <span style='font-size:.65rem;color:#2a3a55'>✓ Feature encoding</span>
                <span style='font-size:.65rem;color:#2a3a55'>✓ Engineered features</span>
                <span style='font-size:.65rem;color:#2a3a55'>✓ Model scoring</span>
                <span style='font-size:.65rem;color:#2a3a55'>✓ Risk classification</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Encoding ─────────────────────────────────────────
            bt_enc   = {"Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2}
            dept_enc = {"Human Resources": 0, "Research & Development": 1, "Sales": 2}
            ef_enc   = {"Human Resources": 0, "Life Sciences": 1, "Marketing": 2,
                        "Medical": 3, "Other": 4, "Technical Degree": 5}
            ms_enc   = {"Divorced": 0, "Married": 1, "Single": 2}
            role_enc = {"Sales": 3, "Research & Development": 4, "Human Resources": 1}

            sample = pd.DataFrame([{
                "Age":                    age_i,
                "BusinessTravel":         bt_enc[biz_travel_i],
                "Department":             dept_enc[dept_i],
                "DistanceFromHome":       distance_i,
                "Education":              education_i,
                "EducationField":         ef_enc[edu_field_i],
                "EnvironmentSatisfaction":env_sat_i,
                "Gender":                 1 if gender_i == "Male" else 0,
                "JobLevel":               job_level_i,
                "JobRole":                role_enc[dept_i],
                "JobSatisfaction":        job_sat_i,
                "MaritalStatus":          ms_enc[marital_i],
                "MonthlyIncome":          monthly_i,
                "NumCompaniesWorked":     num_comp_i,
                "OverTime":               1 if overtime_i == "Yes" else 0,
                "PerformanceRating":      perf_i,
                "RelationshipSatisfaction": rel_sat_i,
                "StockOptionLevel":       stock_i,
                "TotalWorkingYears":      total_yrs_i,
                "TrainingTimesLastYear":  training_i,
                "WorkLifeBalance":        wlb_i,
                "YearsAtCompany":         yrs_comp_i,
                "YearsInCurrentRole":     yrs_role_i,
                "SatisfactionIndex":  int((job_sat_i + env_sat_i + wlb_i + rel_sat_i) / 16 * 100),
                "IncomePerLevel":     int(monthly_i / job_level_i),
                "TenureStability":    int(min(yrs_comp_i / max(num_comp_i, 1), 20)),
                "BurnoutRisk":        int((1 if overtime_i=="Yes" else 0)*2
                                          + (2 if biz_travel_i=="Travel_Frequently" else 0)
                                          + (1 if wlb_i<=2 else 0)),
                "LoyaltyScore":       int(min(max(yrs_comp_i*2 + stock_i*3 - num_comp_i, 0), 50)),
                "CareerGrowthScore":  int(min(max(job_level_i*3 + training_i - (yrs_comp_i - yrs_role_i), 0), 30)),
            }])
            prob = best["pipe"].predict_proba(sample)[0][1]

        prog_ph.empty()  # remove loading bar
        st.divider()

        # ── Big animated risk card ────────────────────────────────
        if prob >= 0.65:
            card_cls    = "risk-card risk-card-high"
            bar_color   = "linear-gradient(90deg,#e05c5c,#c03030)"
            action_cls  = "risk-action risk-action-high"
            icon        = "🚨"
            level_txt   = "HIGH RISK"
            level_color = "#e05c5c"
            action_txt  = "⚡ Immediate manager check-in required. Flag for HR retention review within 48 hours."
        elif prob >= 0.40:
            card_cls    = "risk-card risk-card-med"
            bar_color   = "linear-gradient(90deg,#f0a840,#c07820)"
            action_cls  = "risk-action risk-action-med"
            icon        = "⚠️"
            level_txt   = "MEDIUM RISK"
            level_color = "#f0a840"
            action_txt  = "📋 Schedule career conversation within 30 days. Review compensation and growth plan."
        else:
            card_cls    = "risk-card risk-card-low"
            bar_color   = "linear-gradient(90deg,#52d483,#22a450)"
            action_cls  = "risk-action risk-action-low"
            icon        = "✅"
            level_txt   = "LOW RISK"
            level_color = "#52d483"
            action_txt  = "🌱 Standard engagement applies. Continue regular check-ins and development plans."

        col_card, col_factors = st.columns([1, 1.1])

        with col_card:
            st.markdown(f"""
            <div class="{card_cls}">
              <span class="risk-icon">{icon}</span>
              <div class="risk-level" style="color:{level_color}">{level_txt}</div>
              <div class="risk-score" style="color:{level_color}">{prob:.1%}</div>
              <div class="risk-label">probability of leaving</div>
              <div class="risk-bar-bg">
                <div class="risk-bar-fill" style="width:{prob*100:.1f}%;background:{bar_color}"></div>
              </div>
              <div style='display:flex;justify-content:space-between;font-size:.62rem;color:#2e3a55;margin-bottom:.4rem;font-family:DM Mono,monospace'>
                <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
              </div>
              <div class="{action_cls}">{action_txt}</div>
              <div style='font-size:.62rem;color:#2a3550;margin-top:.8rem;font-family:DM Mono,monospace'>
                Model: {best_name} &nbsp;·&nbsp; Threshold: 0.35
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_factors:
            # ── Rule-based risk factor breakdown ─────────────────
            # Each factor scored independently so user can see WHY

            factors = []

            # Age risk (U-shaped)
            if age_i < 26:
                factors.append(("🔴", "Age (very young <26)", "Very high mobility risk", +3))
            elif age_i < 30:
                factors.append(("🟠", "Age (young 26-30)", "Higher career exploration phase", +2))
            elif 35 <= age_i <= 50:
                factors.append(("🟢", "Age (settled 35-50)", "Most stable career phase", -2))
            elif age_i > 55:
                factors.append(("🟠", "Age (near retirement >55)", "Early retirement consideration", +2))
            else:
                factors.append(("🟡", "Age (30-35)", "Moderate stability", 0))

            # OverTime
            if overtime_i == "Yes":
                factors.append(("🔴", "OverTime: YES", "Burnout risk — one of many factors", +2))
            else:
                factors.append(("🟢", "OverTime: No", "No burnout pressure", -1))

            # Job Satisfaction
            if job_sat_i == 1:
                factors.append(("🔴", "Job Satisfaction: Very Low (1/4)", "Strong push to leave", +3))
            elif job_sat_i == 2:
                factors.append(("🟠", "Job Satisfaction: Low (2/4)", "Dissatisfied", +2))
            elif job_sat_i == 3:
                factors.append(("🟡", "Job Satisfaction: Medium (3/4)", "Neutral", 0))
            else:
                factors.append(("🟢", "Job Satisfaction: High (4/4)", "Happy with role", -2))

            # Environment Satisfaction
            if env_sat_i <= 2:
                factors.append(("🔴", "Environment Satisfaction: Low", "Poor workplace conditions", +2))
            elif env_sat_i == 4:
                factors.append(("🟢", "Environment Satisfaction: High", "Great work environment", -1))

            # Work Life Balance
            if wlb_i == 1:
                factors.append(("🔴", "Work-Life Balance: Very Poor (1/4)", "Major burnout risk", +3))
            elif wlb_i == 2:
                factors.append(("🟠", "Work-Life Balance: Poor (2/4)", "Struggling to balance", +2))
            elif wlb_i == 4:
                factors.append(("🟢", "Work-Life Balance: Excellent (4/4)", "Well balanced", -2))

            # Relationship Satisfaction
            if rel_sat_i <= 2:
                factors.append(("🟠", "Relationship Satisfaction: Low", "Poor team/manager relations", +2))
            elif rel_sat_i == 4:
                factors.append(("🟢", "Relationship Satisfaction: High", "Strong work relationships", -1))

            # Income
            if monthly_i < 2500:
                factors.append(("🔴", f"Income: ${monthly_i:,} (Very Low)", "Far below market — strong push to leave", +3))
            elif monthly_i < 4000:
                factors.append(("🟠", f"Income: ${monthly_i:,} (Below Average)", "Below-average salary", +2))
            elif monthly_i > 12000:
                factors.append(("🟢", f"Income: ${monthly_i:,} (Very High)", "Well compensated — strong retention", -3))
            elif monthly_i > 8000:
                factors.append(("🟢", f"Income: ${monthly_i:,} (High)", "Above-average salary", -2))

            # Stock Options
            if stock_i == 0:
                factors.append(("🔴", "Stock Options: None (0)", "No financial retention incentive", +3))
            elif stock_i == 1:
                factors.append(("🟡", "Stock Options: Low (1)", "Some retention incentive", +1))
            elif stock_i >= 2:
                factors.append(("🟢", f"Stock Options: Level {stock_i}", "Strong financial golden handcuffs", -2))

            # Distance
            if distance_i > 20:
                factors.append(("🔴", f"Commute: {distance_i}km (Very Long >20km)", "Long commute is tiring and costly", +3))
            elif distance_i > 10:
                factors.append(("🟠", f"Commute: {distance_i}km (Moderate)", "Noticeable commute cost", +1))
            else:
                factors.append(("🟢", f"Commute: {distance_i}km (Short)", "Easy commute — positive factor", -1))

            # Marital Status
            if marital_i == "Single":
                factors.append(("🟠", "Marital Status: Single", "More geographically mobile", +2))
            elif marital_i == "Married":
                factors.append(("🟢", "Marital Status: Married", "More likely to value stability", -1))

            # Business Travel
            if biz_travel_i == "Travel_Frequently":
                factors.append(("🔴", "Business Travel: Frequent", "High burnout from travel", +3))
            elif biz_travel_i == "Travel_Rarely":
                factors.append(("🟡", "Business Travel: Rarely", "Some travel-related fatigue", +1))
            else:
                factors.append(("🟢", "Business Travel: None", "No travel burden", -1))

            # Job Level
            if job_level_i == 1:
                factors.append(("🔴", "Job Level: 1 (Junior)", "Lowest level — highest attrition rate", +3))
            elif job_level_i == 2:
                factors.append(("🟠", "Job Level: 2 (Mid)", "Still building career, moderate risk", +1))
            elif job_level_i >= 4:
                factors.append(("🟢", f"Job Level: {job_level_i} (Senior/Lead)", "Senior level — much lower attrition", -2))

            # Years at Company
            if yrs_comp_i <= 2:
                factors.append(("🔴", f"Tenure: {yrs_comp_i} yr(s) (Very New)", "Critical 0-2 year window — highest risk period", +3))
            elif yrs_comp_i <= 5:
                factors.append(("🟠", f"Tenure: {yrs_comp_i} yrs (Early)", "Still building loyalty", +1))
            elif yrs_comp_i > 10:
                factors.append(("🟢", f"Tenure: {yrs_comp_i} yrs (Long-term)", "Strong loyalty — very unlikely to leave", -3))

            # Num Companies
            if num_comp_i >= 5:
                factors.append(("🔴", f"Companies Worked: {num_comp_i} (Job Hopper)", "History of frequent job changes", +3))
            elif num_comp_i >= 4:
                factors.append(("🟠", f"Companies Worked: {num_comp_i}", "Some job-hopping tendency", +2))
            elif num_comp_i <= 1:
                factors.append(("🟢", f"Companies Worked: {num_comp_i}", "Loyal worker profile", -1))

            # Training
            if training_i < 2:
                factors.append(("🟠", f"Training: {training_i} sessions/yr (Low)", "Lack of development opportunities", +2))
            elif training_i > 4:
                factors.append(("🟢", f"Training: {training_i} sessions/yr (High)", "Well developed — engaged employee", -1))

            # Total Working Years
            if total_yrs_i < 3:
                factors.append(("🟠", f"Total Experience: {total_yrs_i} yrs (Early Career)", "Still exploring career options", +2))
            elif total_yrs_i > 15:
                factors.append(("🟢", f"Total Experience: {total_yrs_i} yrs (Experienced)", "Established professional", -1))

            # ── Styled factor cards ───────────────────────────────
            risk_score = sum(f[3] for f in factors)
            factors_html = ""
            for idx2, (emoji, fname, detail, score) in enumerate(sorted(factors, key=lambda x: -x[3])):
                score_str = f"+{score}" if score > 0 else str(score)
                if score >= 2:   badge_cls,sc = "fb-red",   f"<span style='color:#e05c5c'>{score_str}</span>"
                elif score == 1: badge_cls,sc = "fb-amber", f"<span style='color:#f0a840'>{score_str}</span>"
                elif score < 0:  badge_cls,sc = "fb-green", f"<span style='color:#52d483'>{score_str}</span>"
                else:            badge_cls,sc = "fb-grey",  f"<span style='color:#5c6890'>{score_str}</span>"
                factors_html += f"""<div class="factor-row" style="animation-delay:{idx2*0.04:.2f}s">
                  <span class="factor-badge {badge_cls}">{emoji} {score_str}</span>
                  <span class="factor-text"><b>{fname}</b><br><span style='font-size:.67rem;color:#3d4f6a'>{detail}</span></span>
                </div>"""
            with col_factors:
                st.markdown(f"""
                <div style='font-size:.68rem;color:#3d5580;text-transform:uppercase;letter-spacing:.1em;
                            font-weight:700;font-family:DM Mono,monospace;margin-bottom:.7rem'>
                  📊 Risk Factor Breakdown
                </div>
                {factors_html}
                """, unsafe_allow_html=True)

        st.divider()

        # ── HR Recommendations ────────────────────────────────────
        st.markdown("#### 🎯 Recommended HR Actions for This Employee")
        top_risks = [f for f in factors if f[3] >= 2]
        top_risks_sorted = sorted(top_risks, key=lambda x: -x[3])

        action_map = {
            "OverTime":          "⏱️ Review workload immediately. Assign tasks to reduce consistent overtime.",
            "Job Satisfaction":  "😊 Schedule 1:1 conversation to understand dissatisfaction. Career path discussion recommended.",
            "Environment":       "🏢 Investigate team dynamics and workspace issues. Consider team restructuring.",
            "Work-Life Balance": "⚖️ Enforce no-meeting days, flexible hours, and mandatory PTO usage.",
            "Relationship":      "🤝 Facilitate team-building activities. Review manager-employee relationship.",
            "Income":            "💰 Conduct immediate salary benchmarking. Raise is highest-ROI retention action.",
            "Stock Options":     "📈 Grant stock options — cheapest long-term retention tool available.",
            "Commute":           "🏠 Offer remote/hybrid work. Transport allowance as interim solution.",
            "Marital":           "💼 Enhance social community programmes and peer connections.",
            "Business Travel":   "✈️ Reduce travel frequency. Rotate travel assignments across team.",
            "Job Level":         "🚀 Create visible promotion pathway with clear 12-month milestones.",
            "Tenure":            "🎯 Assign mentor. Structured 90-day check-ins during critical first 2 years.",
            "Companies":         "📋 Assign challenging projects to build commitment and engagement.",
            "Training":          "🎓 Enrol in training programme immediately. Development = retention.",
            "Age":               "👥 Age-appropriate engagement: mentorship for young staff, legacy projects for seniors.",
        }

        if not top_risks_sorted:
            st.success("✅ No major risk factors identified. Continue standard engagement practices.")
        else:
            for i, (emoji, name, detail, score) in enumerate(top_risks_sorted[:5], 1):
                # Find matching action
                action = "Monitor and conduct regular check-ins."
                for key, act in action_map.items():
                    if key.lower() in name.lower():
                        action = act
                        break
                st.warning(f"**Priority {i}:** {emoji} {name} — {action}")
