# app.py — clean, single-version
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------- page config ----------
st.set_page_config(page_title="SA Crime Analytics", layout="wide")
st.title("South Africa Crime Analytics")
st.caption("Hotspot classification + Exponential Smoothing forecast (no ARIMA/NN).")

# ---------- constants ----------
FEATURE_YEAR = "2014-2015"
TARGET_YEAR  = "2015-2016"
DEFAULT_CATEGORY = "Burglary at residential premises"
DEFAULT_PROVINCE = "Gauteng"
HOTSPOT_TOP_Q_DEFAULT = 0.75
RANDOM_SEED = 42

# ---------- helpers ----------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    drop = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    return df.drop(columns=drop, errors="ignore")

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

@st.cache_data(show_spinner=True)
def load_crime(file_or_path):
    xls = pd.ExcelFile(file_or_path)
    # If your workbook uses a different sheet name, change here:
    stations = pd.read_excel(xls, sheet_name="Stations")
    stations = clean_columns(stations)
    year_cols = [c for c in stations.columns if re.fullmatch(r"\d{4}-\d{4}", str(c))]
    keep = ["Station", "Province", "Crime Category"] + year_cols
    stations = stations[keep].copy()
    stations = strip_strings(stations, ["Station","Province","Crime Category"])
    stations = coerce_numeric(stations, year_cols)
    stations = stations.dropna(subset=["Station","Crime Category"]).drop_duplicates()
    return stations, year_cols

@st.cache_data(show_spinner=True)
def load_pop(file_or_path):
    try:
        pop = pd.read_csv(file_or_path)
    except Exception:
        return None
    pop = clean_columns(pop)
    for c in ["geo_code","geo_level","name"]:
        if c in pop.columns:
            pop[c] = pop[c].astype(str).str.strip()
    pop = coerce_numeric(pop, ["population","square_kms","population_density"]).drop_duplicates()
    return pop

def build_hotspot_dataset(stations: pd.DataFrame, year_cols, hotspot_top_q):
    df = stations.groupby(["Station","Province","Crime Category"], as_index=False)[year_cols].sum()
    totals = df.groupby(["Station","Province"])[year_cols].sum().reset_index()
    cut = totals[TARGET_YEAR].quantile(hotspot_top_q)
    totals["is_hotspot"] = (totals[TARGET_YEAR] >= cut).astype(int)

    # features from FEATURE_YEAR
    wide = df.pivot_table(index=["Station","Province"], columns="Crime Category",
                          values=FEATURE_YEAR, aggfunc="sum", fill_value=0)
    tot_prev = wide.sum(axis=1).replace(0,1)
    X_counts = wide.add_suffix("_count")
    X_props  = wide.div(tot_prev, axis=0).add_suffix("_prop")
    X_all    = pd.concat([X_counts, X_props], axis=1)

    data = X_all.join(
        totals.set_index(["Station","Province"])[["is_hotspot", TARGET_YEAR]],
        how="inner"
    ).reset_index()

    X = data.drop(columns=["is_hotspot","Station","Province"])
    y = data["is_hotspot"]
    return data, X, y, totals, cut

def merge_province_context(data: pd.DataFrame, pop: pd.DataFrame):
    if pop is None or "geo_level" not in pop.columns:
        return None, None
    has_prov = pop["geo_level"].str.lower().eq("province").any()
    if not has_prov:
        return None, None
    prov = pop[pop["geo_level"].str.lower()=="province"].copy()
    prov["Province_norm"] = prov["name"].str.lower()
    d = data.copy()
    d["Province_norm"] = d["Province"].str.lower()
    merged = d.merge(
        prov[["Province_norm","population_density"]],
        on="Province_norm", how="left"
    ).drop(columns=["Province_norm"])
    return merged, "population_density"

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_SEED)
    logit = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=400, random_state=RANDOM_SEED))
    logit.fit(X_train, y_train)
    y_prob_log = logit.predict_proba(X_test)[:,1]
    y_pred_log = (y_prob_log >= 0.5).astype(int)

    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:,1]
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)

    return (logit, rf, X_train, X_test, y_train, y_test, y_prob_log, y_pred_log, y_prob_rf, y_pred_rf)

def build_series(stations, year_cols, category, province):
    sub = stations[(stations["Crime Category"]==category) & (stations["Province"].str.strip().str.lower()==province.lower())]
    series_annual = sub[year_cols].sum()
    ts = series_annual.rename(index={c: end_year(c) for c in year_cols}).sort_index()
    ts.index = pd.PeriodIndex(ts.index, freq="Y")
    return ts.astype(float)

def fit_holt(ts, n_ahead=2):
    model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
    fit = model.fit(optimized=True)
    fc = fit.forecast(n_ahead)
    resid = fit.resid
    sigma = resid.std(ddof=1)
    ci_low = fc - 1.96*sigma
    ci_high = fc + 1.96*sigma
    return fit, fc, resid, ci_low, ci_high

# ---------- sidebar: upload OR paths ----------
st.sidebar.header("Data")
crime_file = st.sidebar.file_uploader("Crime workbook (.xlsx)", type=["xlsx"])
pop_file   = st.sidebar.file_uploader("Population density (.csv)", type=["csv"])

crime_path_fallback = st.sidebar.text_input("OR path to crime workbook", "data/crime-statistics-20152016.xlsx")
pop_path_fallback   = st.sidebar.text_input("OR path to population CSV (optional)", "data/population.csv")

hotspot_top_q = st.sidebar.slider("Hotspot top quantile", 0.5, 0.95, HOTSPOT_TOP_Q_DEFAULT, 0.05)

# Resolve sources
crime_source = crime_file if crime_file is not None else crime_path_fallback
pop_source   = pop_file   if pop_file   is not None else (pop_path_fallback if pop_path_fallback else None)

# ---------- load data ----------
try:
    stations, year_cols = load_crime(crime_source)
except Exception as e:
    st.error(f"Could not load crime workbook. Check file & sheet name ('Stations'). Error: {e}")
    st.stop()

pop = None
if pop_source and Path(str(pop_source)).exists() or pop_file is not None:
    pop = load_pop(pop_source)

# ---------- tabs ----------
tab1, tab2, tab3 = st.tabs(["🔎 Data", "🔥 Classification", "📈 Forecasting"])

with tab1:
    st.subheader("Crime data (cleaned)")
    st.write(f"Rows: {len(stations)} | Year columns: {len(year_cols)}")
    st.dataframe(stations.head(20), use_container_width=True)
    st.code(", ".join(year_cols), language="text")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Nulls (top 10):**")
        st.write(stations.isna().sum().sort_values(ascending=False).head(10))
    with col2:
        tmp = stations.groupby(["Station","Province"])[[TARGET_YEAR]].sum().reset_index()
        st.markdown(f"**{TARGET_YEAR} overview:**")
        st.write(tmp[TARGET_YEAR].describe())
    if pop is not None:
        st.subheader("Population density (optional)")
        st.dataframe(pop.head(10), use_container_width=True)

with tab2:
    st.subheader("Train hotspot models")
    data, X, y, totals_by_station, cut = build_hotspot_dataset(stations, year_cols, hotspot_top_q)
    st.write(f"Hotspot threshold ({TARGET_YEAR}): **{cut:.0f}** incidents")
    st.write("Hotspot share:", float(data["is_hotspot"].mean()))

    # optional context
    data_ctx, ctx_col = merge_province_context(data.assign(Province=data["Province"]), pop) if pop is not None else (None, None)
    if data_ctx is not None and ctx_col in data_ctx.columns:
        st.info("Province-level population_density merged.")
        X = data_ctx.drop(columns=["is_hotspot","Station","Province", TARGET_YEAR])
        X[ctx_col] = X[ctx_col].fillna(X[ctx_col].median())

    (logit, rf, X_train, X_test, y_train, y_test,
     y_prob_log, y_pred_log, y_prob_rf, y_pred_rf) = train_models(X, y)

    # metrics
    def metrics_row(name, probs, preds):
        rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        return {"model": name, "AUC": round(auc,3),
                "precision_hotspot": round(rep["1"]["precision"],3),
                "recall_hotspot": round(rep["1"]["recall"],3),
                "f1_hotspot": round(rep["1"]["f1-score"],3)}
    st.write(pd.DataFrame([metrics_row("LogisticRegression", y_prob_log, y_pred_log),
                           metrics_row("RandomForest",      y_prob_rf,  y_pred_rf)]))

    # ROC
    c1, c2 = st.columns(2)
    with c1:
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
        fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig = plt.figure(figsize=(7,6))
        plt.barh(fi.index[::-1], fi.values[::-1])
        plt.title("Random Forest — Top 15 Feature Importances")
        plt.xlabel("Importance"); plt.tight_layout()
        st.pyplot(fig)

    # Confusion matrices
    c3, c4 = st.columns(2)
    with c3:
        cm = confusion_matrix(y_test, y_pred_log)
        fig = plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation="nearest"); plt.title("Confusion — Logistic")
        plt.colorbar(); ticks = np.arange(2)
        plt.xticks(ticks, ["Non-hotspot","Hotspot"]); plt.yticks(ticks, ["Non-hotspot","Hotspot"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
        st.pyplot(fig)
    with c4:
        cm = confusion_matrix(y_test, y_pred_rf)
        fig = plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation="nearest"); plt.title("Confusion — RandomForest")
        plt.colorbar(); ticks = np.arange(2)
        plt.xticks(ticks, ["Non-hotspot","Hotspot"]); plt.yticks(ticks, ["Non-hotspot","Hotspot"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout()
        st.pyplot(fig)

with tab3:
    st.subheader("Holt (Exponential Smoothing) — Annual")
    cats = sorted(stations["Crime Category"].dropna().unique().tolist())
    provs = sorted(stations["Province"].dropna().unique().tolist())
    category = st.selectbox("Crime Category", options=cats,
                            index=cats.index(DEFAULT_CATEGORY) if DEFAULT_CATEGORY in cats else 0)
    province = st.selectbox("Province", options=provs,
                            index=provs.index(DEFAULT_PROVINCE) if DEFAULT_PROVINCE in provs else 0)
    horizon = st.slider("Forecast horizon (years)", 1, 5, 2)

    ts = build_series(stations, year_cols, category, province)
    if len(ts) < 3:
        st.error("Not enough annual points to fit a trend model. Pick another category/province.")
    else:
        fit, fc, resid, ci_low, ci_high = fit_holt(ts, n_ahead=horizon)

        fig = plt.figure(figsize=(10,4))
        plt.plot(ts.index.to_timestamp(), ts.values, marker="o")
        plt.title(f"Observed Annual Trend — {category} in {province}")
        plt.xlabel("Year"); plt.ylabel("Incidents"); plt.tight_layout()
        st.pyplot(fig)

        fig = plt.figure(figsize=(10,5))
        plt.plot(ts.index.to_timestamp(), ts.values, label="Observed")
        plt.plot(fc.index.to_timestamp(), fc.values, label="Forecast")
        plt.fill_between(fc.index.to_timestamp(), ci_low.values, ci_high.values, alpha=0.2, label="95% CI")
        plt.title(f"Forecast — {category} in {province} (+{len(fc)} yrs)")
        plt.xlabel("Year"); plt.ylabel("Incidents"); plt.legend(); plt.tight_layout()
        st.pyplot(fig)

        c1, c2 = st.columns(2)
        with c1:
            fig = plt.figure(figsize=(10,4))
            plt.plot(resid.index.to_timestamp(), resid.values, marker="o")
            plt.axhline(0, color="red", linestyle="--")
            plt.title("Residuals"); plt.xlabel("Year"); plt.ylabel("Residuals"); plt.tight_layout()
            st.pyplot(fig)
        with c2:
            pct = ts.pct_change()*100
            fig = plt.figure(figsize=(10,4))
            plt.plot(pct.index.to_timestamp(), pct.values, marker="o")
            plt.axhline(0, color="black", linestyle="--")
            plt.title("YoY % Change"); plt.xlabel("Year"); plt.ylabel("%"); plt.tight_layout()
            st.pyplot(fig)
