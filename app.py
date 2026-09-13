"""
ReservoirIQ — Production Decline Curve Analysis
================================================
A self-contained Streamlit application for Arps decline curve analysis
(Exponential / Harmonic / Hyperbolic), EUR estimation, and multi-well
field portfolio review.

Run with:
    streamlit run app.py
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy optimize import curve_fit

# --------------------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="ReservoirIQ | Decline Curve Analysis",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b1220; }
    .block-container { padding-top: 1.6rem; max-width: 1300px; }

    h1, h2, h3, h4 { color: #eaf2ff !important; font-family: 'Segoe UI', sans-serif; }
    p, li, label, .stMarkdown { color: #c7d3e3 !important; }

    .riq-header {
        display:flex; align-items:center; justify-content:space-between;
        padding: 1.1rem 1.6rem; border-radius: 16px; margin-bottom: 1.4rem;
        background: linear-gradient(120deg, #0f2647 0%, #163a63 45%, #0e3b4a 100%);
        border: 1px solid #244a75;
    }
    .riq-title { font-size: 1.7rem; font-weight: 800; color: #ffffff; margin:0; letter-spacing:.3px;}
    .riq-sub { color:#9db6d6; font-size:.92rem; margin-top:2px;}
    .riq-badge {
        background: rgba(56, 217, 169, 0.14); color:#3fe0b0; border:1px solid #2c8a6d;
        padding: 5px 14px; border-radius: 999px; font-size:.78rem; font-weight:700;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #101d34, #0c1729);
        border: 1px solid #22354f;
        padding: 14px 16px 10px 16px;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] { color: #8ea6c9 !important; font-weight:600; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }

    section[data-testid="stSidebar"] { background-color: #0d1626; border-right: 1px solid #1c2b45; }

    .stTabs [data-baseweb="tab"] { color:#9db6d6; font-weight:600; }
    .stTabs [aria-selected="true"] { color:#3fe0b0 !important; }

    .riq-footer { text-align:center; color:#5c7291; font-size:.8rem; padding: 1.6rem 0 .6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_TEMPLATE = "plotly_dark"
COLORS = {
    "oil": "#3fe0b0",
    "fit": "#ff9f43",
    "forecast": "#5b8dff",
    "grid": "#1c2b45",
    "econ": "#ef5a6f",
}

# --------------------------------------------------------------------------------------
# ARPS DECLINE MODELS
# --------------------------------------------------------------------------------------

def exponential_q(t, qi, di):
    return qi * np.exp(-di * t)


def harmonic_q(t, qi, di):
    return qi / (1.0 + di * t)


def hyperbolic_q(t, qi, di, b):
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def cumulative(t, qi, di, b, model):
    """Analytic cumulative production Np(t) for each Arps model."""
    if model == "Exponential":
        return (qi / di) * (1 - np.exp(-di * t))
    if model == "Harmonic":
        return (qi / di) * np.log(1 + di * t)
    # hyperbolic
    b = max(b, 1e-6)
    qt = hyperbolic_q(t, qi, di, b)
    return (qi ** b / ((1 - b) * di)) * (qi ** (1 - b) - qt ** (1 - b))


def time_to_econ_limit(qi, di, b, q_lim, model):
    if q_lim <= 0 or q_lim >= qi:
        return None
    if model == "Exponential":
        return np.log(qi / q_lim) / di
    if model == "Harmonic":
        return (qi / q_lim - 1) / di
    b = max(b, 1e-6)
    return ((qi / q_lim) ** b - 1) / (b * di)


def fit_model(t, q, model):
    """Fit a given Arps model to (t, q). t in years, q in rate units."""
    qi0 = max(q[0], 1e-3)
    try:
        if model == "Exponential":
            popt, _ = curve_fit(
                exponential_q, t, q, p0=[qi0, 0.5],
                bounds=([1e-6, 1e-6], [qi0 * 5, 10]), maxfev=10000,
            )
            qi, di = popt
            b = 0.0
        elif model == "Harmonic":
            popt, _ = curve_fit(
                harmonic_q, t, q, p0=[qi0, 0.5],
                bounds=([1e-6, 1e-6], [qi0 * 5, 10]), maxfev=10000,
            )
            qi, di = popt
            b = 1.0
        else:  # Hyperbolic
            popt, _ = curve_fit(
                hyperbolic_q, t, q, p0=[qi0, 0.5, 0.5],
                bounds=([1e-6, 1e-6, 1e-4], [qi0 * 5, 10, 2.0]), maxfev=10000,
            )
            qi, di, b = popt

        q_hat = {
            "Exponential": exponential_q(t, qi, di),
            "Harmonic": harmonic_q(t, qi, di),
            "Hyperbolic": hyperbolic_q(t, qi, di, b),
        }[model]

        ss_res = np.sum((q - q_hat) ** 2)
        ss_tot = np.sum((q - np.mean(q)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"qi": qi, "di": di, "b": b, "r2": r2, "q_hat": q_hat}
    except Exception:
        return None


def best_fit(t, q):
    """Try all three models, return the one with the highest R2."""
    results = {}
    for m in ["Exponential", "Harmonic", "Hyperbolic"]:
        fit = fit_model(t, q, m)
        if fit is not None:
            results[m] = fit
    if not results:
        return None, None
    best_name = max(results, key=lambda k: results[k]["r2"])
    return best_name, results[best_name]


# --------------------------------------------------------------------------------------
# DATA HELPERS
# --------------------------------------------------------------------------------------

def generate_sample_data(n_wells=4, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2021-01-01")
    for i in range(n_wells):
        well_name = f"WELL-{i+1:02d}"
        qi = rng.uniform(600, 1600)
        di = rng.uniform(0.35, 0.75)
        b = rng.uniform(0.2, 1.1)
        n_months = rng.integers(24, 42)
        t = np.arange(n_months)
        t_years = t / 12.0
        q_true = hyperbolic_q(t_years, qi, di, b)
        noise = rng.normal(0, q_true * 0.06 + 3)
        q_obs = np.clip(q_true + noise, 1, None)
        dates = start + pd.to_timedelta(t * 30, unit="D") + pd.DateOffset(months=0)
        for d, qv in zip(dates, q_obs):
            rows.append({"Well": well_name, "Date": d, "Oil_Rate_BOPD": round(qv, 1)})
    return pd.DataFrame(rows)


def guess_column(columns, keywords):
    cols_lower = {c: c.lower() for c in columns}
    for c, cl in cols_lower.items():
        for k in keywords:
            if k in cl:
                return c
    return None


def load_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# --------------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------------
st.markdown(
    """
    <div class="riq-header">
        <div>
            <p class="riq-title">📉 ReservoirIQ</p>
            <p class="riq-sub">Production Decline Curve Analysis &nbsp;•&nbsp; Arps Modeling &nbsp;•&nbsp; EUR Forecasting</p>
        </div>
        <div class="riq-badge">ENGINEERING GRADE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------
# SIDEBAR — DATA INPUT
# --------------------------------------------------------------------------------------
st.sidebar.header("1 · Load Production Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel", type=["csv", "xlsx", "xls"],
    help="Needs at minimum a date column and a production rate column. A well-name column is optional.",
)

using_sample = False
if uploaded_file is not None:
    try:
        df_raw = load_uploaded_file(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Couldn't read that file: {e}")
        st.stop()
else:
    df_raw = generate_sample_data()
    using_sample = True
    st.sidebar.info("No file uploaded — showing demo data for 4 sample wells. Upload your own dataset above.")

if df_raw.empty:
    st.error("The dataset is empty.")
    st.stop()

st.sidebar.header("2 · Map Columns")
columns = list(df_raw.columns)

default_well = guess_column(columns, ["well", "uwi", "api"])
default_date = guess_column(columns, ["date", "time", "month", "day"])
default_rate = guess_column(columns, ["oil", "rate", "bopd", "prod", "bbl"])

well_col = st.sidebar.selectbox(
    "Well identifier column (optional)", ["<none — single well>"] + columns,
    index=(columns.index(default_well) + 1) if default_well in columns else 0,
)
date_col = st.sidebar.selectbox(
    "Date column", columns,
    index=columns.index(default_date) if default_date in columns else 0,
)
rate_col = st.sidebar.selectbox(
    "Production rate column", columns,
    index=columns.index(default_rate) if default_rate in columns else 0,
)

st.sidebar.header("3 · Decline Model")
model_choice = st.sidebar.selectbox(
    "Model", ["Auto (best fit)", "Exponential", "Harmonic", "Hyperbolic"], index=0,
)
econ_limit = st.sidebar.number_input(
    "Economic limit rate (same units as rate column)", min_value=0.0, value=10.0, step=1.0,
)
forecast_years = st.sidebar.slider("Forecast horizon (years)", 1, 30, 15)

# --------------------------------------------------------------------------------------
# CLEAN DATA
# --------------------------------------------------------------------------------------
df = df_raw.copy()
try:
    df[date_col] = pd.to_datetime(df[date_col])
except Exception:
    st.error(f"Couldn't parse '{date_col}' as dates. Pick a different date column.")
    st.stop()

df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
df = df.dropna(subset=[date_col, rate_col])
df = df[df[rate_col] > 0]

if well_col == "<none — single well>":
    df["__well__"] = "Well-1"
else:
    df["__well__"] = df[well_col].astype(str)

if df.empty:
    st.error("After cleaning, no valid rows remain. Check your column mapping.")
    st.stop()

all_wells = sorted(df["__well__"].unique())

# --------------------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Single Well Analysis", "🌍 Field Portfolio", "📋 Data & Export"])

# Storage for field-level results, built once
field_results = []
for w in all_wells:
    wdf = df[df["__well__"] == w].sort_values(date_col)
    if len(wdf) < 4:
        continue
    t0 = wdf[date_col].iloc[0]
    t_years = (wdf[date_col] - t0).dt.days.values / 365.25
    q = wdf[rate_col].values.astype(float)

    if model_choice == "Auto (best fit)":
        name, fit = best_fit(t_years, q)
    else:
        name, fit = model_choice, fit_model(t_years, q, model_choice)

    if fit is None:
        continue

    t_econ = time_to_econ_limit(fit["qi"], fit["di"], fit["b"], econ_limit, name)
    eur = cumulative(t_econ, fit["qi"], fit["di"], fit["b"], name) if t_econ else cumulative(
        forecast_years, fit["qi"], fit["di"], fit["b"], name
    )
    cum_to_date = np.trapz(q, t_years) if len(q) > 1 else 0.0

    field_results.append(
        {
            "well": w, "model": name, "qi": fit["qi"], "di": fit["di"], "b": fit["b"],
            "r2": fit["r2"], "t_years": t_years, "q": q, "dates": wdf[date_col].values,
            "eur": eur, "t_econ": t_econ, "cum_to_date": cum_to_date, "t0": t0,
        }
    )

if not field_results:
    st.error("Not enough data points per well to fit a decline curve (need at least 4).")
    st.stop()

results_by_well = {r["well"]: r for r in field_results}

# --------------------------------------------------------------------------------------
# TAB 1 — SINGLE WELL
# --------------------------------------------------------------------------------------
with tab1:
    sel_well = st.selectbox("Select well", list(results_by_well.keys()))
    r = results_by_well[sel_well]

    di_annual_pct = (1 - np.exp(-r["di"])) * 100 if r["model"] == "Exponential" else r["di"] * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best-fit Model", r["model"])
    c2.metric("Initial Rate (qi)", f"{r['qi']:.0f}")
    c3.metric("Decline Rate", f"{di_annual_pct:.1f} %/yr")
    c4.metric("b-factor", f"{r['b']:.2f}")
    c5.metric("Fit Quality (R²)", f"{r['r2']:.3f}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Cumulative to Date", f"{r['cum_to_date']:,.0f}")
    c7.metric("Estimated EUR", f"{r['eur']:,.0f}")
    remaining = max(r["eur"] - r["cum_to_date"], 0)
    c8.metric("Remaining Reserves", f"{remaining:,.0f}")

    # Build forecast series
    t_max = r["t_econ"] if r["t_econ"] and r["t_econ"] < forecast_years else forecast_years
    t_fc = np.linspace(0, max(t_max, r["t_years"].max()), 300)
    model_fn = {
        "Exponential": lambda t: exponential_q(t, r["qi"], r["di"]),
        "Harmonic": lambda t: harmonic_q(t, r["qi"], r["di"]),
        "Hyperbolic": lambda t: hyperbolic_q(t, r["qi"], r["di"], r["b"]),
    }[r["model"]]
    q_fc = model_fn(t_fc)
    dates_fc = r["t0"] + pd.to_timedelta(t_fc * 365.25, unit="D")

    colA, colB = st.columns(2)

    with colA:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=r["dates"], y=r["q"], mode="markers", name="Observed",
            marker=dict(color=COLORS["oil"], size=7, line=dict(width=1, color="#0b1220")),
        ))
        fig1.add_trace(go.Scatter(
            x=dates_fc, y=q_fc, mode="lines", name=f"{r['model']} fit",
            line=dict(color=COLORS["fit"], width=3),
        ))
        if econ_limit > 0:
            fig1.add_hline(y=econ_limit, line_dash="dot", line_color=COLORS["econ"],
                            annotation_text="Economic limit", annotation_font_color=COLORS["econ"])
        fig1.update_layout(
            title="Production Rate vs. Time (Cartesian)", template=PLOT_TEMPLATE,
            height=420, legend=dict(orientation="h", y=1.1),
            xaxis=dict(gridcolor=COLORS["grid"]), yaxis=dict(title="Rate", gridcolor=COLORS["grid"]),
            margin=dict(t=60, l=10, r=10, b=10),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=r["dates"], y=r["q"], mode="markers", name="Observed",
            marker=dict(color=COLORS["oil"], size=7, line=dict(width=1, color="#0b1220")),
        ))
        fig2.add_trace(go.Scatter(
            x=dates_fc, y=q_fc, mode="lines", name=f"{r['model']} fit",
            line=dict(color=COLORS["fit"], width=3),
        ))
        fig2.update_layout(
            title="Production Rate vs. Time (Semi-Log)", template=PLOT_TEMPLATE,
            height=420, legend=dict(orientation="h", y=1.1),
            xaxis=dict(gridcolor=COLORS["grid"]),
            yaxis=dict(title="Rate (log)", type="log", gridcolor=COLORS["grid"]),
            margin=dict(t=60, l=10, r=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Cumulative production forecast
    np_fc = cumulative(t_fc, r["qi"], r["di"], r["b"], r["model"])
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=dates_fc, y=np_fc, mode="lines", name="Cumulative production",
        line=dict(color=COLORS["forecast"], width=3), fill="tozeroy",
        fillcolor="rgba(91,141,255,0.12)",
    ))
    fig3.add_hline(y=r["eur"], line_dash="dash", line_color="#3fe0b0",
                    annotation_text=f"EUR ≈ {r['eur']:,.0f}", annotation_font_color="#3fe0b0")
    fig3.update_layout(
        title="Cumulative Production Forecast to Economic Limit", template=PLOT_TEMPLATE,
        height=380, xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(title="Cumulative volume", gridcolor=COLORS["grid"]),
        margin=dict(t=60, l=10, r=10, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Model diagnostics"):
        diag = pd.DataFrame({
            "Date": r["dates"], "Observed": r["q"],
            "Fitted": model_fn(r["t_years"]),
        })
        diag["Residual"] = diag["Observed"] - diag["Fitted"]
        diag["Residual %"] = (diag["Residual"] / diag["Observed"] * 100).round(2)
        st.dataframe(diag, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# TAB 2 — FIELD PORTFOLIO
# --------------------------------------------------------------------------------------
with tab2:
    summary = pd.DataFrame([
        {
            "Well": r["well"], "Model": r["model"], "qi": round(r["qi"], 1),
            "Decline %/yr": round((1 - np.exp(-r["di"])) * 100 if r["model"] == "Exponential" else r["di"] * 100, 1),
            "b-factor": round(r["b"], 2), "R²": round(r["r2"], 3),
            "Cumulative to Date": round(r["cum_to_date"], 0),
            "EUR": round(r["eur"], 0),
        }
        for r in field_results
    ]).sort_values("EUR", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Wells Analyzed", len(field_results))
    k2.metric("Field Cumulative", f"{summary['Cumulative to Date'].sum():,.0f}")
    k3.metric("Field EUR", f"{summary['EUR'].sum():,.0f}")
    k4.metric("Avg. Fit Quality", f"{summary['R²'].mean():.3f}")

    fig4 = go.Figure(go.Bar(
        x=summary["Well"], y=summary["EUR"], marker_color=COLORS["oil"],
        text=summary["EUR"].map(lambda v: f"{v:,.0f}"), textposition="outside",
    ))
    fig4.update_layout(
        title="Estimated Ultimate Recovery (EUR) by Well", template=PLOT_TEMPLATE,
        height=420, xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(title="EUR", gridcolor=COLORS["grid"]),
        margin=dict(t=60, l=10, r=10, b=10),
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = go.Figure()
    for r in field_results:
        fig5.add_trace(go.Scatter(
            x=r["dates"], y=r["q"], mode="lines+markers", name=r["well"],
            line=dict(width=2), marker=dict(size=4),
        ))
    fig5.update_layout(
        title="All Wells — Production Rate History", template=PLOT_TEMPLATE,
        height=440, xaxis=dict(gridcolor=COLORS["grid"]),
        yaxis=dict(title="Rate (log)", type="log", gridcolor=COLORS["grid"]),
        margin=dict(t=60, l=10, r=10, b=10), legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Field Summary Table")
    st.dataframe(summary, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------------------
# TAB 3 — DATA & EXPORT
# --------------------------------------------------------------------------------------
with tab3:
    st.subheader("Cleaned Input Data")
    st.dataframe(df.drop(columns="__well__"), use_container_width=True, hide_index=True)

    st.subheader("Download Results")
    csv_buf = io.StringIO()
    summary.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download field summary (CSV)", data=csv_buf.getvalue(),
        file_name="reservoiriq_field_summary.csv", mime="text/csv",
    )

    per_well_frames = []
    for r in field_results:
        model_fn = {
            "Exponential": lambda t, r=r: exponential_q(t, r["qi"], r["di"]),
            "Harmonic": lambda t, r=r: harmonic_q(t, r["qi"], r["di"]),
            "Hyperbolic": lambda t, r=r: hyperbolic_q(t, r["qi"], r["di"], r["b"]),
        }[r["model"]]
        t_max = r["t_econ"] if r["t_econ"] and r["t_econ"] < forecast_years else forecast_years
        t_fc = np.linspace(0, max(t_max, r["t_years"].max()), 120)
        dates_fc = r["t0"] + pd.to_timedelta(t_fc * 365.25, unit="D")
        per_well_frames.append(pd.DataFrame({
            "Well": r["well"], "Date": dates_fc, "Forecast_Rate": model_fn(t_fc),
            "Cumulative": cumulative(t_fc, r["qi"], r["di"], r["b"], r["model"]),
        }))
    forecast_all = pd.concat(per_well_frames, ignore_index=True)
    csv_buf2 = io.StringIO()
    forecast_all.to_csv(csv_buf2, index=False)
    st.download_button(
        "⬇️ Download full forecast (all wells, CSV)", data=csv_buf2.getvalue(),
        file_name="reservoiriq_forecast.csv", mime="text/csv",
    )

    if using_sample:
        st.caption("Tip: this is demo data. Upload your own CSV/Excel from the sidebar for real analysis.")

st.markdown(
    '<div class="riq-footer">ReservoirIQ — Arps decline curve analysis (Exponential · Harmonic · Hyperbolic). '
    'Built for production engineers, by production engineers.</div>',
    unsafe_allow_html=True,
)
