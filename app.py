# app.py — SA Crime Analytics (Classification + Forecasting, no ARIMA/NN)

import os, re
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

# ---------------- Page setup (must be FIRST Streamlit call) ----------------
st.set_page_config(page_title="SA Crime Analytics", layout="wide")
st.title("South Africa Crime Analytics")
st.caption("Hotspot classification + Holt (Exponential Smoothing) forecast (no ARIMA/NN).")

# ---------------- Constants ----------------
FEATURE_YEAR = "2014-2015"
TARGET_YEAR  = "2015-2016"
DEFAULT_CATEGORY = "Burglary at residential premises"
DEFAULT_PROVINCE = "Gauteng"
RANDOM_SEED = 42

# ---------------- Helpers ----------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    drop = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    return df.drop(columns=drop, errors="ignore")

def _strip_strings(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _end_year(label: str):
    m = re.match(r"^(\d{4})-(\d{4})$", str(label))
    return int(m.group(2)) if m else None

@st.cache_data(show_spinner=True)
def load_crime(file_or_path, sheet_name):
    xls = pd.ExcelFile(file_or_path)
    if sheet_name == "<detect>":
        candidates = ["Stations", "Station Statistics", "Sheet1"]
        for s in candidates:
            if s in xls.sheet_names:
                sheet_name = s
                break
        if sheet_name == "<detect>":
            sheet_name = xls.sheet_names[0]

    stations = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
    stations = _clean_columns(stations)
    year_cols = [c for c in stations.columns if re.fullmatch(r"\d{4}-\d{4}", str(c))]
    keep = ["Station", "Province", "Crime Category"] + year_cols
    for col in ["Station","Province","Crime Category"]:
        if col not in stations.columns:
            raise ValueError(f"Missing required column: {col}. Found: {stations.columns.tolist()[:10]} ...")
    stations = stations[keep].copy()
    stations = _strip_strings(stations, ["Station","Province","Crime Category"])
    stations = _coerce_numeric(stations, year_cols)
    stations = stations.dropna(subset=["Station","Crime Category"]).drop_duplicates()
    if FEATURE_YEAR not in year_cols or TARGET_YEAR not in year_cols:
        raise ValueError(f"Required years not found. Need {FEATURE_YEAR} and {TARGET_YEAR}. Found: {year_cols}")
    return stations, year_cols

@st.cache_data(show_spinner=True)
def load_pop(file_or_path):
    if file_or_path is None:
        return None
    try:
        pop = pd.read_csv(file_or_path)
    except Exception:
        return None
    pop = _clean_columns(pop)
    for c in ["geo_code","geo_level","name"]:
        if c in pop.columns:
            pop[c] = pop[c].astype(str).str.strip()
    pop = _coerce_numeric(pop, ["population","square_kms","population_density"]).drop_duplicates()
    return pop

def build_hotspot_dataset(stations: pd.DataFrame, year_cols, hotspot_top_q):
    df = stations.groupby(["Station","Province","Crime Category"], as_index=False)[year_cols].sum()
    totals = df.groupby(["Station","Province"])[year_cols].sum().reset_index()
    cut = totals[TARGET_YEAR].quantile(hotspot_top_q)
    totals["is_hotspot"] = (totals[TARGET_YEAR] >= cut).astype(int)

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
    if not pop["geo_level"].astype(str).str.lower().eq("province").any():
        return None, None
    prov = pop[pop["geo_level"].astype(str).str.lower()=="province"].copy()
    if "name" not in prov.columns or "population_density" not in prov.columns:
        return None, None
    prov["Province_norm"] = prov["name"].str.lower()
    d = data.copy()
    d["Province_norm"] = d["Province"].str.lower()
    merged = d.merge(prov[["Province_norm","population_density"]], on="Province_norm", how="left")\
              .drop(columns=["Province_norm"])
    return merged, "population_density"

@st.cache_resource(show_spinner=True)
def train_models_cached(X, y, seed=RANDOM_SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=seed, test_size=0.2)
    logit = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=400, random_state=seed))
    logit.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed)
    rf.fit(X_train, y_train)
    return logit, rf, X_train, X_test, y_train, y_test

def build_series(stations, year_cols, category, province):
    sub = stations[(stations["Crime Category"]==category) &
                   (stations["Province"].str.strip().str.lower()==province.lower())]
    series_annual = sub[year_cols].sum()
    ts = series_annual.rename(index={c: _end_year(c) for c in year_cols}).sort_index()
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

# ---------------- Sidebar (data inputs) ----------------
st.sidebar.header("Data")
crime_file = st.sidebar.file_uploader("Crime workbook (.xlsx)", type=["xlsx"])
sheet_name = st.sidebar.selectbox("Crime sheet", options=["<detect>"], index=0)
pop_file   = st.sidebar.file_uploader("Population density (.csv) [optional]", type=["csv"])
hotspot_top_q = st.sidebar.slider("Hotspot top quantile", 0.50, 0.95, 0.75, 0.05)

# Load data
if not crime_file:
    st.warning("Please upload the **crime Excel file** (.xlsx).")
    st.stop()

try:
    stations, year_cols = load_crime(crime_file, sheet_name)
except Exception as e:
    st.error(f"Could not load crime workbook. Check file & columns. Error: {e}")
    st.stop()

pop = load_pop(pop_file) if pop_file is not None else None

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🔎 Data", "🔥 Classification", "📈 Forecasting"])

# -------- Tab 1: Data --------
with tab1:
    st.subheader("Crime data (cleaned)")
    st.write(f"Rows: {len(stations)} | Year columns: {len(year_cols)}")
    st.dataframe(stations.head(20), use_container_width=True)

    st.markdown("**Year columns detected:**")
    st.code(", ".join(year_cols), language="text")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Nulls (top 10):**")
        st.write(stations.isna().sum().sort_values(ascending=False).head(10))
    with c2:
        if TARGET_YEAR in year_cols:
            tmp = stations.groupby(["Station","Province"])[[TARGET_YEAR]].sum().reset_index()
            st.markdown(f"**{TARGET_YEAR} overview:**")
            st.write(tmp[TARGET_YEAR].describe())
        else:
            st.warning(f"{TARGET_YEAR} not found in detected year columns.")

    if pop is not None:
        st.subheader("Population density (optional)")
        st.dataframe(pop.head(10), use_container_width=True)

# -------- Tab 2: Classification --------
with tab2:
    st.subheader("Train hotspot models")

    data, X, y, totals_by_station, cut = build_hotspot_dataset(stations, year_cols, hotspot_top_q)
    st.write(f"Hotspot threshold ({TARGET_YEAR}): **{cut:.0f}** incidents")
    st.write("Hotspot share:", float(data["is_hotspot"].mean()))

    # Optional contextual feature
    data_ctx, ctx_col = merge_province_context(data.assign(Province=data["Province"]), pop) if pop is not None else (None, None)
    if data_ctx is not None and ctx_col in data_ctx.columns:
        st.info("Merged province-level **population_density** as contextual feature.")
        X = data_ctx.drop(columns=["is_hotspot","Station","Province", TARGET_YEAR])
        X[ctx_col] = X[ctx_col].fillna(X[ctx_col].median())

    if st.button("Run models"):
        logit, rf, X_train, X_test, y_train, y_test = train_models_cached(X, y)

        y_prob_log = logit.predict_proba(X_test)[:,1]; y_pred_log = (y_prob_log >= 0.5).astype(int)
        y_prob_rf  = rf.predict_proba(X_test)[:,1];    y_pred_rf  = (y_prob_rf  >= 0.5).astype(int)

        def _row(name, probs, preds):
            rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
            auc = roc_auc_score(y_test, probs)
            return {"model": name, "AUC": round(auc,3),
                    "precision_hotspot": round(rep["1"]["precision"],3),
                    "recall_hotspot":    round(rep["1"]["recall"],3),
                    "f1_hotspot":        round(rep["1"]["f1-score"],3)}
        st.write(pd.DataFrame([_row("LogisticRegression", y_prob_log, y_pred_log),
                               _row("RandomForest",      y_prob_rf,  y_pred_rf)]))

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
    else:
        st.info("Click **Run models** to train and view metrics.")

# -------- Tab 3: Forecasting --------
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
