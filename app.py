import streamlit as st
import pandas as pd
import re

st.title("Crime Analytics Dashboard – South Africa")

# 🔹 Sidebar upload
crime_file = st.sidebar.file_uploader("Upload crime workbook (.xlsx)", type=["xlsx"])
pop_file   = st.sidebar.file_uploader("Upload population density (.csv)", type=["csv"])

if crime_file is not None:
    xls = pd.ExcelFile(crime_file)
    stations = pd.read_excel(xls, sheet_name="Stations")
    year_cols = [c for c in stations.columns if re.fullmatch(r"\d{4}-\d{4}", str(c))]
    st.success("Crime dataset loaded!")
    st.dataframe(stations.head())
else:
    st.warning("Please upload your crime Excel file.")# app.py
# South Africa Crime Analytics — Streamlit Dashboard
# - Hotspot classification (Logit + RandomForest)
# - Forecasting with Exponential Smoothing (Holt) (no ARIMA/NN)
# - Optional province-level contextual merge with population density

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="SA Crime Analytics", layout="wide")

# -----------------------------
# Config
# -----------------------------
CRIME_XLSX = "crime-statistics-20152016.xlsx"
POP_CSV = "cfafrica-_-data-team-_-covid-19-_-data-_-openafrica-uploads-_-south-africa-population-density (1).csv"

FEATURE_YEAR = "2014-2015"      # avoid leakage
TARGET_YEAR  = "2015-2016"      # label hotspots on this year
DEFAULT_CATEGORY = "Burglary at residential premises"
DEFAULT_PROVINCE = "Gauteng"
HOTSPOT_TOP_Q = 0.75            # top 25% = hotspot
RANDOM_SEED = 42

# -----------------------------
# Helpers
# -----------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    return df.drop(columns=drop_cols, errors="ignore")

def strip_strings(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def end_year(label: str):
    m = re.match(r"^(\d{4})-(\d{4})$", str(label))
    return int(m.group(2)) if m else None

@st.cache_data(show_spinner=False)
def load_crime_data(path: str):
    xls = pd.ExcelFile(path)
    stations = pd.read_excel(xls, sheet_name="Stations")
    stations = clean_columns(stations)
    year_cols = [c for c in stations.columns if re.fullmatch(r"\d{4}-\d{4}", c)]
    keep = ["Station", "Province", "Crime Category"] + year_cols
    stations = stations[keep].copy()
    stations = strip_strings(stations, ["Station", "Province", "Crime Category"])
    stations = coerce_numeric(stations, year_cols)
    stations = stations.dropna(subset=["Station", "Crime Category"])
    stations = stations.drop_duplicates(subset=["Station","Province","Crime Category"] + year_cols)
    return stations, year_cols

@st.cache_data(show_spinner=False)
def load_pop_data(path: str):
    if not Path(path).exists():
        return None
    pop = pd.read_csv(path)
    pop = clean_columns(pop)
    for c in ["geo_code","geo_level","name"]:
        if c in pop.columns:
            pop[c] = pop[c].astype(str).str.strip()
    pop = coerce_numeric(pop, ["population","square_kms","population_density"])
    pop = pop.drop_duplicates()
    return pop

def build_hotspot_dataset(stations: pd.DataFrame, year_cols):
    df = stations.groupby(["Station","Province","Crime Category"], as_index=False)[year_cols].sum()
    totals_by_station = df.groupby(["Station","Province"])[year_cols].sum().reset_index()
    cut = totals_by_station[TARGET_YEAR].quantile(HOTSPOT_TOP_Q)
    totals_by_station["is_hotspot"] = (totals_by_station[TARGET_YEAR] >= cut).astype(int)

    # features from FEATURE_YEAR (counts + proportions by crime type)
    features_wide = df.pivot_table(
        index=["Station","Province"],
        columns="Crime Category",
        values=FEATURE_YEAR,
        aggfunc="sum",
        fill_value=0
    )
    station_total_prev = features_wide.sum(axis=1).replace(0, 1)
    X_counts = features_wide.add_suffix("_count")
    X_props  = features_wide.div(station_total_prev, axis=0).add_suffix("_prop")
    X_all    = pd.concat([X_counts, X_props], axis=1)

    data = X_all.join(
        totals_by_station.set_index(["Station","Province"])[["is_hotspot", TARGET_YEAR]],
        how="inner"
    ).reset_index()

    X = data.drop(columns=["is_hotspot","Station","Province"])
    y = data["is_hotspot"]

    return data, X, y, totals_by_station, cut

def merge_province_context(data: pd.DataFrame, pop: pd.DataFrame):
    if pop is None or "geo_level" not in pop.columns:
        return None, None
    has_prov = pop["geo_level"].str.lower().eq("province").any()
    if not has_prov:
        return None, None
    prov = pop[pop["geo_level"].str.lower()=="province"].copy()
    prov["Province_norm"] = prov["name"].str.lower()
    data = data.copy()
    data["Province_norm"] = data["Province"].str.lower()
    merged = data.merge(
        prov[["Province_norm","population_density"]],
        on="Province_norm",
        how="left"
    ).drop(columns=["Province_norm"])
    return merged, "population_density"

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_SEED
    )
    logit = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=400, random_state=RANDOM_SEED))
    logit.fit(X_train, y_train)
    y_prob_log = logit.predict_proba(X_test)[:,1]
    y_pred_log = (y_prob_log >= 0.5).astype(int)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, class_weight="balanced", random_state=RANDOM_SEED
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:,1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)

    return (logit, rf, X_train, X_test, y_train, y_test, y_prob_log, y_pred_log, y_prob_rf, y_pred_rf)

def build_series(stations, year_cols, category, province):
    df_cat = stations[stations["Crime Category"] == category].copy()
    df_cat = df_cat[df_cat["Province"].str.strip().str.lower() == province.lower()]
    series_annual = df_cat[year_cols].sum()
    ts = series_annual.rename(index={c: end_year(c) for c in year_cols}).sort_index()
    ts.index = pd.PeriodIndex(ts.index, freq="Y")
    ts = ts.astype(float)
    return ts

def fit_holt(ts, n_ahead=2):
    model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(n_ahead)
    resid = fit.resid
    sigma = resid.std(ddof=1)
    ci_low = fc - 1.96*sigma
    ci_high = fc + 1.96*sigma
    return fit, fc, resid, ci_low, ci_high

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

crime_path = st.sidebar.text_input("Crime workbook path", CRIME_XLSX)
pop_path   = st.sidebar.text_input("Population density CSV (optional)", POP_CSV)
hotspot_q  = st.sidebar.slider("Hotspot top quantile", 0.5, 0.95, HOTSPOT_TOP_Q, 0.05)

# Load data
stations, year_cols = load_crime_data(crime_path)
pop = load_pop_data(pop_path)

# Overview
st.title("South Africa Crime Analytics")
st.caption("Hotspot classification + Exponential Smoothing forecast (no ARIMA/NN).")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔎 Data & Cleaning", "🔥 Hotspot Classification", "📈 Forecasting"])

# -----------------------------
# Tab 1: Data & Cleaning
# -----------------------------
with tab1:
    st.subheader("Crime data (cleaned)")
    st.write(f"Columns: {len(stations.columns)} | Rows: {len(stations)}")
    st.dataframe(stations.head(20), use_container_width=True)

    st.markdown("**Year columns detected:**")
    st.code(", ".join(year_cols))

    # Nulls / negatives quick checks
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Null checks (top 10):**")
        st.write(stations.isna().sum().sort_values(ascending=False).head(10))
    with col_right:
        st.markdown(f"**{TARGET_YEAR} summary:**")
        tmp = stations.groupby(["Station","Province"])[[TARGET_YEAR]].sum().reset_index()
        st.write(tmp[TARGET_YEAR].describe())

    if pop is not None:
        st.subheader("Population density (optional)")
        st.write(f"Columns: {len(pop.columns)} | Rows: {len(pop)}")
        st.dataframe(pop.head(10), use_container_width=True)
        if "geo_level" in pop.columns:
            st.write("Geo levels:", pop["geo_level"].dropna().unique())

# -----------------------------
# Tab 2: Hotspot Classification
# -----------------------------
with tab2:
    st.subheader("Define hotspots and train models")

    # Rebuild hotspot dataset with user quantile
    global HOTSPOT_TOP_Q
    HOTSPOT_TOP_Q = hotspot_q
    data, X, y, totals_by_station, cut = build_hotspot_dataset(stations, year_cols)

    st.write(f"Hotspot threshold for {TARGET_YEAR}: **{cut:.0f} incidents**")
    st.write("Share labeled as hotspots:", float(data["is_hotspot"].mean()))

    # Optional: merge province-level population density if present
    data_ctx, ctx_col = merge_province_context(data, pop)
    if data_ctx is not None and ctx_col in data_ctx.columns:
        st.info("Merged province-level population_density as contextual feature.")
        X = data_ctx.drop(columns=["is_hotspot","Station","Province", TARGET_YEAR])
        X[ctx_col] = X[ctx_col].fillna(X[ctx_col].median())

    # Train models
    (logit, rf, X_train, X_test, y_train, y_test,
     y_prob_log, y_pred_log, y_prob_rf, y_pred_rf) = train_models(X, y)

    # Metrics
    def metrics_table(name, probs, preds):
        rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        return {
            "model": name,
            "AUC": round(auc,3),
            "precision_hotspot": round(rep["1"]["precision"],3),
            "recall_hotspot": round(rep["1"]["recall"],3),
            "f1_hotspot": round(rep["1"]["f1-score"],3),
        }

    m1 = metrics_table("LogisticRegression", y_prob_log, y_pred_log)
    m2 = metrics_table("RandomForest",      y_prob_rf,  y_pred_rf)
    st.write(pd.DataFrame([m1,m2]))

    # Plots
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        # ROC
        fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
        fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
        auc_log = roc_auc_score(y_test, y_prob_log)
        auc_rf  = roc_auc_score(y_test, y_prob_rf)
        fig = plt.figure(figsize=(6,5))
        plt.plot(fpr_log, tpr_log, label=f"Logit (AUC={auc_log:.3f})")
        plt.plot(fpr_rf,  tpr_rf,  label=f"RF (AUC={auc_rf:.3f})")
        plt.plot([0,1],[0,1],"--", label="Chance")
        plt.title("ROC — Hotspot Classifiers")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
        st.pyplot(fig)

    with c2:
        # RF importances
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig = plt.figure(figsize=(7,6))
        plt.barh(fi.index[::-1], fi.values[::-1])
        plt.title("Random Forest — Top 15 Feature Importances")
        plt.xlabel("Importance"); plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    # Confusion matrices
    c3, c4 = st.columns(2)
    with c3:
        cm = confusion_matrix(y_test, y_pred_log)
        fig = plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix — Logistic")
        plt.colorbar()
        ticks = np.arange(2)
        plt.xticks(ticks, ["Non-hotspot","Hotspot"])
        plt.yticks(ticks, ["Non-hotspot","Hotspot"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
        st.pyplot(fig)
    with c4:
        cm = confusion_matrix(y_test, y_pred_rf)
        fig = plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix — RandomForest")
        plt.colorbar()
        ticks = np.arange(2)
        plt.xticks(ticks, ["Non-hotspot","Hotspot"])
        plt.yticks(ticks, ["Non-hotspot","Hotspot"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    # Top incident stations (for context)
    topN = st.slider("Top N stations by incidents", 5, 25, 10)
    totals = totals_by_station = data.groupby(["Station","Province"])[TARGET_YEAR].sum().reset_index()
    top_df = totals.sort_values(TARGET_YEAR, ascending=False).head(topN).iloc[::-1]
    fig = plt.figure(figsize=(10,5))
    plt.barh(top_df["Station"], top_df[TARGET_YEAR])
    plt.title(f"Top {topN} Stations by Incidents ({TARGET_YEAR})")
    plt.xlabel("Incidents"); plt.ylabel("Station"); plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# Tab 3: Forecasting
# -----------------------------
with tab3:
    st.subheader("Exponential Smoothing (Holt) — Annual series")

    # Pick category & province
    all_cats = sorted(stations["Crime Category"].dropna().unique().tolist())
    all_prov = sorted(stations["Province"].dropna().unique().tolist())
    category = st.selectbox("Crime Category", options=all_cats,
                            index=all_cats.index(DEFAULT_CATEGORY) if DEFAULT_CATEGORY in all_cats else 0)
    province = st.selectbox("Province", options=all_prov,
                            index=all_prov.index(DEFAULT_PROVINCE) if DEFAULT_PROVINCE in all_prov else 0)
    horizon = st.slider("Forecast horizon (years)", 1, 5, 2)

    # Build series & fit model
    ts = build_series(stations, year_cols, category, province)
    if len(ts) < 3:
        st.error("Not enough annual points to fit a trend model. Pick another category/province.")
    else:
        fit, fc, resid, ci_low, ci_high = fit_holt(ts, n_ahead=horizon)

        # Fig A: Observed series
        fig = plt.figure(figsize=(10,4))
        plt.plot(ts.index.to_timestamp(), ts.values, marker="o")
        plt.title(f"Observed Annual Trend — {category} in {province}")
        plt.xlabel("Year"); plt.ylabel("Incidents"); plt.tight_layout()
        st.pyplot(fig)

        # Fig B: Forecast + CI
        fig = plt.figure(figsize=(10,5))
        plt.plot(ts.index.to_timestamp(), ts.values, label="Observed")
        plt.plot(fc.index.to_timestamp(), fc.values, label="Forecast")
        plt.fill_between(fc.index.to_timestamp(), ci_low.values, ci_high.values, alpha=0.2, label="95% CI")
        plt.title(f"Forecast — {category} in {province} (+{len(fc)} yrs)")
        plt.xlabel("Year"); plt.ylabel("Incidents"); plt.legend(); plt.tight_layout()
        st.pyplot(fig)

        # Diagnostics
        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure(figsize=(10,4))
            plt.plot(resid.index.to_timestamp(), resid.values, marker="o")
            plt.axhline(0, color="red", linestyle="--")
            plt.title("Residuals")
            plt.xlabel("Year"); plt.ylabel("Residuals"); plt.tight_layout()
            st.pyplot(fig)
        with c2:
            pct_change = ts.pct_change()*100
            fig = plt.figure(figsize=(10,4))
            plt.plot(pct_change.index.to_timestamp(), pct_change.values, marker="o")
            plt.axhline(0, color="black", linestyle="--")
            plt.title("YoY % Change (Observed)")
            plt.xlabel("Year"); plt.ylabel("%"); plt.tight_layout()
            st.pyplot(fig)

        # Small metrics table
        rmse = np.sqrt(np.mean(resid.dropna()**2)) if resid.notna().any() else np.nan
        st.write(pd.DataFrame({"metric":["RMSE (in-sample)"], "value":[rmse]}))

import streamlit as st, os
if not os.path.exists("crime-statistics-20152016.xlsx"):
    st.error("Missing file: crime-statistics-20152016.xlsx"); st.stop()
if not os.path.exists("cfafrica-_-data-team-_-covid-19-_-data-_-openafrica-uploads-_-south-africa-population-density (1).csv"):
    st.warning("Context CSV not found — app will run crime-only.")

