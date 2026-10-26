"""
app.py
------
Main Streamlit entry-point for the NILM Dashboard.

Run with:
    streamlit run app.py

Pages
-----
  1. Overview Dashboard
  2. Dataset Management
  3. Data Preprocessing
  4. Model Training
  5. Prediction Interface
  6. Energy Analysis
  7. Model Evaluation
  8. Reports & Export
  9. Settings & Info
"""

import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── project root on sys.path ──────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import (
    AGGREGATE_COL,
    APPLIANCE_DISPLAY_NAMES,
    SUPPORTED_APPLIANCES,
    DataLoader,
    generate_sample_dataset,
    sample_csv_bytes,
    validate_csv_bytes,
)
from src.energy import (
    aggregate_discrepancy,
    appliance_energy_summary,
    compute_cost,
    compute_energy_kwh,
    daily_energy_breakdown,
)
from src.evaluate import evaluate_all, metrics_to_csv
from src.features import build_features, features_summary
from src.predict import predict, validate_schema_compatibility
from src.preprocessing import (
    NILMPreprocessor,
    chronological_split,
    preprocessing_summary,
)
from src.train import NILMTrainer

logging.basicConfig(level=logging.WARNING)

# ═════════════════════════════════════════════════════════════
# Page config & global CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NILM Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .main { background-color: #f4f6fb; }
    section[data-testid="stSidebar"] { background-color: #0d1b2a; }

    /* ── Sidebar text ── */
    section[data-testid="stSidebar"] * { color: #e0e8f0 !important; }
    section[data-testid="stSidebar"] .sidebar-title {
        color: #38bdf8 !important; font-weight: 700; font-size: 1.15rem;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        border: 1px solid #d0e4ff;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(37,99,235,0.07);
    }
    .kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.95rem; font-weight: 700; color: #0f172a; margin: 4px 0 2px; }
    .kpi-sub   { font-size: 0.75rem; color: #94a3b8; }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 2px solid #bfdbfe; padding-bottom: 6px; margin-bottom: 14px;
    }

    /* ── Status badges ── */
    .badge-ok   { background:#d1fae5; color:#065f46; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }
    .badge-warn { background:#fef3c7; color:#92400e; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }
    .badge-err  { background:#fee2e2; color:#991b1b; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }

    /* ── Empty state ── */
    .empty-state {
        text-align:center; padding:60px 20px; color:#94a3b8;
        border:2px dashed #cbd5e1; border-radius:16px; margin:20px 0;
    }
    .empty-state h3 { color:#475569; margin-bottom:8px; }

    /* ── Page title ── */
    .page-title { font-size:1.75rem; font-weight:700; color:#0f172a; margin-bottom:4px; }
    .page-sub   { font-size:0.9rem;  color:#64748b; margin-bottom:24px; }

    /* ── Table ── */
    .dataframe { border-radius:8px !important; }

    /* Plotly chart border */
    .js-plotly-plot { border-radius:12px; }

    /* remove default padding on top */
    .block-container { padding-top: 1.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# Session-state initialisation
# ═════════════════════════════════════════════════════════════
_DEFAULTS = {
    "raw_df": None,
    "loader": None,
    "quality_report": None,
    "appliance_col_map": {},
    "train_df": None,
    "val_df": None,
    "test_df": None,
    "preprocessor": None,
    "trainer": None,
    "pred_df": None,
    "eval_report": None,
    "energy_df": None,
    "tariff": 0.12,
    "resample_interval": "1min",
    "page": "Overview Dashboard",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════
# Sidebar navigation
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-title">⚡ NILM Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    pages = [
        "Overview Dashboard",
        "Dataset Management",
        "Data Preprocessing",
        "Model Training",
        "Prediction Interface",
        "Energy Analysis",
        "Model Evaluation",
        "Reports & Export",
        "Settings & Info",
    ]
    icons = ["🏠","📂","⚙️","🤖","🔮","⚡","📊","📥","ℹ️"]
    for icon, page in zip(icons, pages):
        if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
            st.session_state["page"] = page

    st.markdown("---")
    # Quick status
    has_data    = st.session_state["raw_df"] is not None
    has_splits  = st.session_state["train_df"] is not None
    has_model   = st.session_state["trainer"] is not None
    has_preds   = st.session_state["pred_df"] is not None

    def _badge(ok): return "badge-ok" if ok else "badge-warn"

    st.markdown(f"""
    <div style='font-size:0.78rem;line-height:2'>
      <span class='{_badge(has_data)}'>{'✓' if has_data else '○'} Data loaded</span><br>
      <span class='{_badge(has_splits)}'>{'✓' if has_splits else '○'} Splits ready</span><br>
      <span class='{_badge(has_model)}'>{'✓' if has_model else '○'} Model trained</span><br>
      <span class='{_badge(has_preds)}'>{'✓' if has_preds else '○'} Predictions</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Academic NILM prototype — predictions are estimates, not measurements.")

page = st.session_state["page"]


# ═════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════

def _kpi(label, value, sub="", col=None):
    html = f"""<div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>"""
    (col or st).markdown(html, unsafe_allow_html=True)


def _empty(title, body=""):
    st.markdown(
        f"<div class='empty-state'><h3>{title}</h3><p>{body}</p></div>",
        unsafe_allow_html=True,
    )


def _section(title):
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


APP_COLORS = {
    "refrigerator":    "#2563eb",
    "ceiling_fan":     "#0891b2",
    "air_conditioner": "#7c3aed",
    "washing_machine": "#059669",
    "electric_kettle": "#d97706",
}

DISPLAY = APPLIANCE_DISPLAY_NAMES


# ═════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "Overview Dashboard":
    st.markdown("<div class='page-title'>⚡ NILM Energy Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Non-Intrusive Load Monitoring · Appliance-Level Energy Disaggregation</div>", unsafe_allow_html=True)

    raw_df   = st.session_state["raw_df"]
    pred_df  = st.session_state["pred_df"]
    trainer  = st.session_state["trainer"]
    tariff   = st.session_state["tariff"]

    # ── KPI row ──────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)

    if raw_df is not None:
        agg_kwh = compute_energy_kwh(raw_df[AGGREGATE_COL])
        agg_avg = raw_df[AGGREGATE_COL].mean()
        _kpi("Total Measured Energy", f"{agg_kwh:.2f} kWh", "Aggregate meter", c1)
        _kpi("Avg. Aggregate Power",  f"{agg_avg:.0f} W",   "All appliances",  c2)
    else:
        _kpi("Total Measured Energy", "—", "Upload data first", c1)
        _kpi("Avg. Aggregate Power",  "—", "Upload data first", c2)

    if pred_df is not None and trainer:
        keys = trainer.trained_appliances()
        e_df = appliance_energy_summary(pred_df, keys, tariff)
        total_pred_kwh = e_df["pred_energy_kwh"].sum() if len(e_df) else 0.0
        total_cost     = compute_cost(total_pred_kwh, tariff)
        _kpi("Est. Appliance Energy", f"{total_pred_kwh:.2f} kWh", "From predictions",  c3)
        _kpi("Est. Electricity Cost", f"₹{total_cost:.2f}",        f"@₹{tariff}/kWh",   c4)
        _kpi("Appliances Tracked",    str(len(keys)),               "Trained models",    c5)
    else:
        _kpi("Est. Appliance Energy", "—", "Train model first", c3)
        _kpi("Est. Electricity Cost", "—", "Train model first", c4)
        _kpi("Appliances Tracked",    "—", "Train model first", c5)

    model_status = "✓ Ready" if trainer else "○ Not trained"
    _kpi("Model Status", model_status, "Random Forest", c6)

    st.markdown("---")

    if raw_df is None:
        _empty(
            "No data loaded yet",
            "Go to → 📂 Dataset Management to upload your CSV or load the sample dataset.",
        )
        st.stop()

    # ── Aggregate power over time ─────────────────────────────
    _section("Aggregate Power Over Time")
    display_df = raw_df.copy()
    if len(display_df) > 5000:
        display_df = display_df.resample("5min").mean()

    fig_agg = go.Figure()
    fig_agg.add_trace(go.Scatter(
        x=display_df.index, y=display_df[AGGREGATE_COL],
        name="Aggregate (measured)", line=dict(color="#2563eb", width=1.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    ))
    fig_agg.update_layout(
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        xaxis_title="Time", yaxis_title="Power (W)",
        paper_bgcolor="white", plot_bgcolor="#f8faff",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_agg, use_container_width=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        if pred_df is not None and trainer:
            _section("Predicted Appliance Consumption")
            keys = trainer.trained_appliances()
            fig_app = go.Figure()
            fig_app.add_trace(go.Scatter(
                x=display_df.index, y=display_df[AGGREGATE_COL],
                name="Aggregate", line=dict(color="#94a3b8", width=1, dash="dot"),
            ))
            for k in keys:
                col = f"{k}_pred_w"
                if col in pred_df.columns:
                    ps = pred_df[col]
                    if len(ps) > 5000:
                        ps = ps.resample("5min").mean()
                    fig_app.add_trace(go.Scatter(
                        x=ps.index, y=ps,
                        name=DISPLAY.get(k, k),
                        line=dict(color=APP_COLORS.get(k,"#555"), width=1.5),
                    ))
            fig_app.update_layout(
                height=280, margin=dict(l=0,r=0,t=10,b=0),
                xaxis_title="Time", yaxis_title="Power (W)",
                paper_bgcolor="white", plot_bgcolor="#f8faff",
                legend=dict(orientation="h", y=1.12, font_size=10),
            )
            st.plotly_chart(fig_app, use_container_width=True)
        else:
            _empty("Appliance Predictions", "Train a model to see per-appliance estimates.")

    with col_right:
        if pred_df is not None and trainer:
            _section("Energy Share by Appliance")
            keys = trainer.trained_appliances()
            e_df = appliance_energy_summary(pred_df, keys, tariff)
            if len(e_df):
                fig_pie = px.pie(
                    e_df, values="pred_energy_kwh",
                    names=e_df["appliance"].map(lambda k: DISPLAY.get(k, k)),
                    color="appliance",
                    color_discrete_map={k: APP_COLORS.get(k,"#999") for k in keys},
                    hole=0.5,
                )
                fig_pie.update_traces(textposition="outside", textinfo="percent+label")
                fig_pie.update_layout(
                    height=280, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="white", showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            _empty("Energy Distribution", "Available after training.")

    # ── Daily energy trend ────────────────────────────────────
    if pred_df is not None and trainer:
        _section("Daily Energy Trend (kWh)")
        keys = trainer.trained_appliances()
        daily_df = daily_energy_breakdown(pred_df, keys)
        if not daily_df.empty:
            daily_df.columns = [DISPLAY.get(c, c) for c in daily_df.columns]
            fig_daily = px.bar(
                daily_df.reset_index(), x="date",
                y=list(daily_df.columns),
                color_discrete_sequence=list(APP_COLORS.values()),
                barmode="stack",
            )
            fig_daily.update_layout(
                height=260, margin=dict(l=0,r=0,t=10,b=0),
                xaxis_title="Date", yaxis_title="Energy (kWh)",
                paper_bgcolor="white", plot_bgcolor="#f8faff",
                legend=dict(orientation="h", y=1.12, font_size=10),
            )
            st.plotly_chart(fig_daily, use_container_width=True)

    # ── Appliance ON/OFF summary ──────────────────────────────
    if pred_df is not None and trainer:
        _section("Appliance Operating State Summary")
        keys = trainer.trained_appliances()
        cols = st.columns(len(keys))
        for i, k in enumerate(keys):
            scol = f"{k}_pred_state"
            if scol in pred_df.columns:
                on_pct = float(pred_df[scol].mean()) * 100
                bar_color = APP_COLORS.get(k, "#2563eb")
                cols[i].markdown(
                    f"<div class='kpi-card' style='border-left-color:{bar_color}'>"
                    f"<div class='kpi-label'>{DISPLAY.get(k,k)}</div>"
                    f"<div class='kpi-value'>{on_pct:.1f}%</div>"
                    f"<div class='kpi-sub'>Estimated ON time</div></div>",
                    unsafe_allow_html=True,
                )


# ═════════════════════════════════════════════════════════════
# PAGE 2 — DATASET MANAGEMENT
# ═════════════════════════════════════════════════════════════
elif page == "Dataset Management":
    st.markdown("<div class='page-title'>📂 Dataset Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Upload and validate your electricity CSV dataset</div>", unsafe_allow_html=True)

    # ── Download sample ───────────────────────────────────────
    with st.expander("📥 Download Sample CSV Template", expanded=False):
        st.markdown("""
        The sample dataset is a **synthetic labeled dataset** (1 day, 1-minute intervals).
        It contains aggregate power and five appliance columns.  
        ⚠️ *Synthetic data — do not use to claim real-world accuracy.*
        """)
        st.download_button(
            "⬇️ Download sample_nilm_data.csv",
            data=sample_csv_bytes(),
            file_name="sample_nilm_data.csv",
            mime="text/csv",
        )

    st.markdown("---")
    tab_upload, tab_path, tab_sample = st.tabs(["📤 Upload CSV", "📁 Load from Path", "🧪 Use Sample Dataset"])

    # ── Load from local path tab (best for large files > 200 MB) ──────────
    with tab_path:
        st.info(
            "**Best for large datasets (> 200 MB).**  "
            "Place your CSV anywhere on this machine and paste the full file path below. "
            "The file is read directly from disk — no browser upload limit applies.",
            icon="💡",
        )
        local_path = st.text_input(
            "Full path to CSV file",
            placeholder=r"e.g.  C:\Users\You\Downloads\my_dataset.csv",
            key="local_csv_path",
        )
        load_path_btn = st.button("Load File", key="load_local_path_btn", type="primary")

        if load_path_btn and local_path:
            _lp = Path(local_path.strip())
            if not _lp.exists():
                st.error(f"❌ File not found: `{_lp}`")
            elif _lp.suffix.lower() not in {".csv"}:
                st.error("❌ Only CSV files are supported.")
            else:
                with st.spinner(f"Reading {_lp.name} …"):
                    try:
                        raw_bytes = _lp.read_bytes()
                        ok, msg = validate_csv_bytes(raw_bytes)
                        if not ok:
                            st.error(f"❌ File rejected: {msg}")
                            st.stop()
                        size_mb = len(raw_bytes) / 1_048_576
                        st.success(f"✅ Loaded **{_lp.name}** ({size_mb:.1f} MB) — configure column mapping below.")
                        preview_df = pd.read_csv(io.BytesIO(raw_bytes), nrows=5)
                        st.dataframe(preview_df, use_container_width=True)
                        all_cols = list(preview_df.columns)

                        st.markdown("### Column Mapping")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            ts_col  = st.selectbox("Timestamp column", all_cols, index=0, key="lp_ts_col")
                            agg_col = st.selectbox("Aggregate power column", all_cols, index=1 if len(all_cols) > 1 else 0, key="lp_agg_col")
                            unit    = st.radio("Power unit", ["W", "kW"], horizontal=True, key="lp_power_unit")
                            resamp  = st.selectbox("Resample interval", ["None","30s","1min","5min","15min","30min"], key="lp_resamp")
                        with col_b:
                            st.markdown("**Appliance columns (optional — needed for training)**")
                            app_map = {}
                            for app_key in SUPPORTED_APPLIANCES:
                                choice = st.selectbox(
                                    DISPLAY.get(app_key, app_key),
                                    ["(none)"] + all_cols,
                                    key=f"lp_app_{app_key}",
                                )
                                if choice != "(none)":
                                    app_map[app_key] = choice

                        if st.button("✅ Confirm & Save Dataset", key="lp_confirm_btn", type="primary"):
                            with st.spinner("Processing …"):
                                df_loaded = load_and_preprocess(
                                    raw_bytes, ts_col, agg_col, unit, resamp, app_map
                                )
                            st.session_state["dataset"] = df_loaded
                            st.session_state["dataset_name"] = _lp.name
                            st.session_state["app_map"] = app_map
                            st.success(f"🎉 Dataset saved — {len(df_loaded):,} rows ready.")
                            st.rerun()
                    except Exception as _e:
                        st.error(f"Error reading file: {_e}")

    # ── Upload tab ────────────────────────────────────────────
    with tab_upload:
        st.info(
            "**Upload limit is 2 GB** (configured via `.streamlit/config.toml`).  "
            "For files larger than that, use the **📁 Load from Path** tab instead.",
            icon="📤",
        )
        uploaded = st.file_uploader(
            "Upload your NILM CSV file", type=["csv"],
            help="CSV must contain a timestamp column and an aggregate power column.",
        )

        if uploaded:
            raw_bytes = uploaded.read()
            ok, msg = validate_csv_bytes(raw_bytes)
            if not ok:
                st.error(f"❌ File rejected: {msg}")
                st.stop()

            # Preview columns
            try:
                preview_df = pd.read_csv(io.BytesIO(raw_bytes), nrows=5)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                st.stop()

            st.success("✅ File uploaded — configure column mapping below.")
            st.dataframe(preview_df, use_container_width=True)
            all_cols = list(preview_df.columns)

            st.markdown("### Column Mapping")
            col_a, col_b = st.columns(2)
            with col_a:
                ts_col  = st.selectbox("Timestamp column", all_cols, index=0, key="ts_col")
                agg_col = st.selectbox("Aggregate power column", all_cols, index=1 if len(all_cols)>1 else 0, key="agg_col_sel")
                unit    = st.radio("Power unit", ["W", "kW"], horizontal=True, key="power_unit")
                resamp  = st.selectbox("Resample interval", ["None","30s","1min","5min","15min","30min"], key="resamp")

            with col_b:
                st.markdown("**Appliance columns (optional — needed for training)**")
                app_map = {}
                for app_key in SUPPORTED_APPLIANCES:
                    choice = st.selectbox(
                        DISPLAY.get(app_key, app_key),
                        ["(none)"] + all_cols,
                        key=f"app_{app_key}",
                    )
                    if choice != "(none)":
                        app_map[app_key] = choice

            if st.button("✅ Load Dataset", type="primary", key="btn_load"):
                try:
                    loader = DataLoader(
                        timestamp_col=ts_col,
                        aggregate_col=agg_col,
                        appliance_col_map=app_map,
                        power_unit=unit,
                        resample_interval=None if resamp == "None" else resamp,
                    )
                    df = loader.load(raw_bytes)
                    qr = loader.quality_report(df)
                    st.session_state.update({
                        "raw_df": df, "loader": loader,
                        "quality_report": qr, "appliance_col_map": app_map,
                        "train_df": None, "val_df": None, "test_df": None,
                        "trainer": None, "pred_df": None, "eval_report": None,
                    })
                    st.success(f"✅ Loaded {len(df):,} rows.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Load failed: {e}")

    # ── Sample tab ────────────────────────────────────────────
    with tab_sample:
        st.info("ℹ️ The synthetic sample dataset is for **software testing only** and does not represent real-world measurements.")
        n_days = st.slider("Days of synthetic data", 1, 7, 1)
        seed   = st.number_input("Random seed", value=42, step=1)

        if st.button("🧪 Load Sample Dataset", type="primary", key="btn_sample"):
            sample_df = generate_sample_dataset(
                n_samples=n_days * 1440, interval_seconds=60, seed=int(seed)
            )
            app_map = {k: k for k in SUPPORTED_APPLIANCES}
            loader = DataLoader(
                timestamp_col="timestamp",
                aggregate_col=AGGREGATE_COL,
                appliance_col_map=app_map,
            )
            qr = loader.quality_report(sample_df)
            st.session_state.update({
                "raw_df": sample_df, "loader": loader,
                "quality_report": qr, "appliance_col_map": app_map,
                "train_df": None, "val_df": None, "test_df": None,
                "trainer": None, "pred_df": None, "eval_report": None,
            })
            st.success(f"✅ Synthetic dataset loaded — {len(sample_df):,} rows, {n_days} day(s).")
            st.rerun()

    # ── Quality report ────────────────────────────────────────
    qr = st.session_state["quality_report"]
    if qr:
        st.markdown("---")
        _section("Dataset Quality Report")
        c1,c2,c3,c4 = st.columns(4)
        _kpi("Total Rows",      f"{qr['total_rows']:,}",              "", c1)
        _kpi("Duration",        qr["duration"],                       "", c2)
        _kpi("Sampling",        f"{qr['median_interval_s']}s median", "", c3)
        _kpi("Irregular Gaps",  str(qr["irregular_intervals"]),       "", c4)

        col_warn, col_info = st.columns(2)
        with col_warn:
            if qr["missing_aggregate"] > 0:
                st.warning(f"⚠️ {qr['missing_aggregate']} missing aggregate readings.")
            if qr["negative_aggregate"] > 0:
                st.warning(f"⚠️ {qr['negative_aggregate']} negative aggregate readings (clipped to 0 during preprocessing).")
            if qr["irregular_intervals"] > 0:
                st.warning(f"⚠️ {qr['irregular_intervals']} irregular sampling intervals detected.")

        with col_info:
            apps = qr.get("appliances_available", [])
            if apps:
                st.success(f"✅ Labeled appliances: {', '.join(DISPLAY.get(k,k) for k in apps)}")
            else:
                st.info("ℹ️ No appliance labels found.  Predictions can be generated but supervised training requires labeled data.")

        # Preview
        raw_df = st.session_state["raw_df"]
        if raw_df is not None:
            _section("Data Preview (first 20 rows)")
            st.dataframe(raw_df.head(20), use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 3 — DATA PREPROCESSING
# ═════════════════════════════════════════════════════════════
elif page == "Data Preprocessing":
    st.markdown("<div class='page-title'>⚙️ Data Preprocessing</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Clean data and create chronological train / val / test splits</div>", unsafe_allow_html=True)

    raw_df = st.session_state["raw_df"]
    if raw_df is None:
        _empty("No dataset loaded", "Go to 📂 Dataset Management first.")
        st.stop()

    col_l, col_r = st.columns([1, 2])
    with col_l:
        _section("Split Configuration")
        train_frac = st.slider("Training fraction", 0.5, 0.85, 0.70, 0.05, key="train_frac")
        val_frac   = st.slider("Validation fraction", 0.05, 0.25, 0.15, 0.05, key="val_frac")
        test_frac  = round(1 - train_frac - val_frac, 2)
        st.info(f"Test fraction: **{test_frac:.0%}** (held out, never used for fitting)")

        if st.button("▶️ Run Preprocessing", type="primary", key="btn_preproc"):
            try:
                train, val, test = chronological_split(raw_df, train_frac, val_frac)
                prep = NILMPreprocessor(scale_aggregate=False)
                train_c = prep.fit_transform(train)
                val_c   = prep.transform(val)
                test_c  = prep.transform(test)
                app_keys = [k for k in SUPPORTED_APPLIANCES if k in raw_df.columns]
                summ = preprocessing_summary(raw_df, train_c, val_c, test_c, app_keys)
                st.session_state.update({
                    "train_df": train_c, "val_df": val_c, "test_df": test_c,
                    "preprocessor": prep,
                    "trainer": None, "pred_df": None, "eval_report": None,
                })
                st.success("✅ Preprocessing complete.")
            except Exception as e:
                st.error(f"❌ {e}")

    with col_r:
        train_df = st.session_state["train_df"]
        if train_df is not None:
            val_df   = st.session_state["val_df"]
            test_df  = st.session_state["test_df"]
            _section("Split Summary")

            col1,col2,col3 = st.columns(3)
            for name, sdf, c in [("Train",train_df,col1),("Val",val_df,col2),("Test",test_df,col3)]:
                _kpi(f"{name} Set", f"{len(sdf):,} rows",
                     f"{sdf.index[0].date()} → {sdf.index[-1].date()}", c)

            _section("Aggregate Power — Split View")
            fig_splits = go.Figure()
            for sdf, name, color in [
                (train_df,"Train","#2563eb"),
                (val_df,  "Val",  "#0891b2"),
                (test_df, "Test", "#7c3aed"),
            ]:
                ds = sdf if len(sdf) <= 5000 else sdf.resample("5min").mean()
                fig_splits.add_trace(go.Scatter(
                    x=ds.index, y=ds[AGGREGATE_COL],
                    name=name, line=dict(color=color, width=1.2),
                ))
            fig_splits.update_layout(
                height=260, margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="white", plot_bgcolor="#f8faff",
                xaxis_title="Time", yaxis_title="Power (W)",
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_splits, use_container_width=True)

            # Feature preview
            _section("Feature Matrix Preview")
            X_sample = build_features(train_df.head(200))
            fsum = features_summary(X_sample)
            st.write(f"**{fsum['n_features']} features** · {fsum['n_samples']} sample rows · any NaN: {fsum['any_nan']}")
            st.dataframe(X_sample.head(5), use_container_width=True)
        else:
            _empty("Run preprocessing to see split summary")


# ═════════════════════════════════════════════════════════════
# PAGE 4 — MODEL TRAINING
# ═════════════════════════════════════════════════════════════
elif page == "Model Training":
    st.markdown("<div class='page-title'>🤖 Model Training</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Train Random Forest models and a median baseline per appliance</div>", unsafe_allow_html=True)

    train_df = st.session_state["train_df"]
    if train_df is None:
        _empty("Run preprocessing first", "Go to ⚙️ Data Preprocessing.")
        st.stop()

    app_keys_available = [k for k in SUPPORTED_APPLIANCES if k in train_df.columns]
    if not app_keys_available:
        st.error("❌ No appliance label columns found in the training data.  Supervised training requires labeled appliance columns.")
        st.stop()

    col_l, col_r = st.columns([1, 2])
    with col_l:
        _section("Model Configuration")
        selected_apps = st.multiselect(
            "Appliances to train",
            app_keys_available,
            default=app_keys_available,
            format_func=lambda k: DISPLAY.get(k, k),
            key="sel_apps",
        )
        n_estimators = st.slider("Number of trees", 10, 300, 100, 10, key="n_est")
        max_depth    = st.selectbox("Max tree depth", ["None",4,6,8,12,16], key="max_dep")
        max_depth_v  = None if max_depth == "None" else int(max_depth)
        random_state = st.number_input("Random seed", value=42, step=1, key="rnd_seed")

        if st.button("🚀 Train Models", type="primary", key="btn_train"):
            if not selected_apps:
                st.error("Select at least one appliance.")
            else:
                with st.spinner("Training… this may take a moment."):
                    try:
                        trainer = NILMTrainer(
                            appliance_keys=selected_apps,
                            n_estimators=int(n_estimators),
                            max_depth=max_depth_v,
                            random_state=int(random_state),
                        )
                        trainer.fit(train_df, val_df=st.session_state["val_df"])
                        model_dir = ROOT / "models"
                        trainer.save(model_dir)
                        st.session_state.update({
                            "trainer": trainer,
                            "pred_df": None, "eval_report": None,
                        })
                        st.success("✅ Models trained and saved to `models/`.")
                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")

        # Load existing
        st.markdown("---")
        if st.button("📂 Load Saved Models", key="btn_load_model"):
            try:
                trainer = NILMTrainer.load(ROOT / "models")
                st.session_state["trainer"] = trainer
                st.success(f"✅ Loaded models for: {', '.join(trainer.trained_appliances())}")
            except Exception as e:
                st.error(f"❌ Could not load: {e}")

    with col_r:
        trainer = st.session_state["trainer"]
        if trainer:
            _section("Training Summary")
            summ = trainer.training_summary
            rows = []
            for k, v in summ.items():
                rows.append({
                    "Appliance": DISPLAY.get(k,k),
                    "Train samples": v.get("train_samples","—"),
                    "Mean power (W)": v.get("power_mean_train","—"),
                    "ON fraction": v.get("on_fraction_train","—"),
                    "Baseline median (W)": v.get("baseline_median_w","—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            _section("Feature Importances")
            selected_fi = st.selectbox(
                "Select appliance",
                trainer.trained_appliances(),
                format_func=lambda k: DISPLAY.get(k,k),
                key="fi_app",
            )
            if selected_fi in trainer.regressors:
                reg = trainer.regressors[selected_fi]
                from src.features import FEATURE_COLUMNS
                fi_df = pd.DataFrame({
                    "Feature": FEATURE_COLUMNS,
                    "Importance": reg.feature_importances_,
                }).sort_values("Importance", ascending=True)
                fig_fi = px.bar(
                    fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance",
                    color_continuous_scale=["#bfdbfe","#2563eb"],
                )
                fig_fi.update_layout(
                    height=340, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="white", plot_bgcolor="#f8faff",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            _empty("No model trained yet", "Configure and click 🚀 Train Models.")


# ═════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTION INTERFACE
# ═════════════════════════════════════════════════════════════
elif page == "Prediction Interface":
    st.markdown("<div class='page-title'>🔮 Prediction Interface</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Generate appliance-level power predictions from aggregate data</div>", unsafe_allow_html=True)

    raw_df  = st.session_state["raw_df"]
    trainer = st.session_state["trainer"]

    if raw_df is None:
        _empty("No data loaded", "Upload data in 📂 Dataset Management.")
        st.stop()
    if trainer is None:
        _empty("No trained model", "Train a model in 🤖 Model Training.")
        st.stop()

    # Schema check
    warns = validate_schema_compatibility(raw_df, trainer)
    for w in warns:
        st.warning(f"⚠️ {w}")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        _section("Options")
        trained_apps = trainer.trained_appliances()
        sel_apps = st.multiselect(
            "Appliances", trained_apps, default=trained_apps,
            format_func=lambda k: DISPLAY.get(k,k), key="pred_apps",
        )
        date_min = raw_df.index.min().date()
        date_max = raw_df.index.max().date()
        d_start  = st.date_input("From date", date_min, min_value=date_min, max_value=date_max, key="pred_start")
        d_end    = st.date_input("To date",   date_max, min_value=date_min, max_value=date_max, key="pred_end")

        if st.button("▶️ Run Predictions", type="primary", key="btn_predict"):
            subset = raw_df[str(d_start):str(d_end)]
            if subset.empty:
                st.error("No data in the selected date range.")
            else:
                try:
                    with st.spinner("Predicting…"):
                        pred_df = predict(trainer, subset, sel_apps)
                    st.session_state["pred_df"] = pred_df
                    st.success(f"✅ Predictions generated for {len(pred_df):,} time steps.")
                except Exception as e:
                    st.error(f"❌ {e}")

    with col_r:
        pred_df = st.session_state["pred_df"]
        if pred_df is None:
            _empty("No predictions yet", "Select options and click ▶️ Run Predictions.")
        else:
            trained_apps = trainer.trained_appliances()
            sel_vis = st.selectbox(
                "Visualise appliance",
                [k for k in trained_apps if f"{k}_pred_w" in pred_df.columns],
                format_func=lambda k: DISPLAY.get(k,k),
                key="vis_app",
            )

            disp = pred_df if len(pred_df) <= 5000 else pred_df.resample("5min").mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=disp.index, y=disp[AGGREGATE_COL],
                name="Aggregate (measured)", line=dict(color="#94a3b8", dash="dot", width=1.5),
            ))
            pred_col = f"{sel_vis}_pred_w"
            true_col = f"{sel_vis}_true_w"
            if pred_col in disp.columns:
                fig.add_trace(go.Scatter(
                    x=disp.index, y=disp[pred_col],
                    name=f"{DISPLAY.get(sel_vis,sel_vis)} — Predicted",
                    line=dict(color=APP_COLORS.get(sel_vis,"#2563eb"), width=2),
                ))
            if true_col in disp.columns:
                fig.add_trace(go.Scatter(
                    x=disp.index, y=disp[true_col],
                    name=f"{DISPLAY.get(sel_vis,sel_vis)} — Ground Truth",
                    line=dict(color="#059669", width=1.5, dash="dash"),
                ))
            fig.update_layout(
                height=320, margin=dict(l=0,r=0,t=10,b=0),
                xaxis_title="Time", yaxis_title="Power (W)",
                paper_bgcolor="white", plot_bgcolor="#f8faff",
                legend=dict(orientation="h", y=1.12, font_size=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ON/OFF state
            state_col = f"{sel_vis}_pred_state"
            if state_col in pred_df.columns:
                on_pct = float(pred_df[state_col].mean()) * 100
                st.info(f"📟 Predicted ON time for **{DISPLAY.get(sel_vis,sel_vis)}**: **{on_pct:.1f}%** of the selected period.")

            if true_col in pred_df.columns:
                st.caption("✅ Ground-truth labels available — shown as dashed green line above.")
            else:
                st.caption("ℹ️ No ground-truth labels for this appliance in the loaded dataset.")


# ═════════════════════════════════════════════════════════════
# PAGE 6 — ENERGY ANALYSIS
# ═════════════════════════════════════════════════════════════
elif page == "Energy Analysis":
    st.markdown("<div class='page-title'>⚡ Energy Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Appliance energy estimates, cost calculations, and daily trends</div>", unsafe_allow_html=True)

    pred_df = st.session_state["pred_df"]
    trainer = st.session_state["trainer"]
    tariff  = st.session_state["tariff"]

    if pred_df is None or trainer is None:
        _empty("No predictions available", "Run predictions in 🔮 Prediction Interface first.")
        st.stop()

    col_l, col_r = st.columns([1, 3])
    with col_l:
        _section("Tariff Settings")
        tariff_new = st.number_input(
            "Electricity tariff (₹/kWh)", value=float(tariff),
            min_value=0.0, step=0.01, format="%.2f", key="tariff_input",
        )
        st.session_state["tariff"] = tariff_new
        tariff = tariff_new
        st.caption("Cost values are estimates.  Tariff is user-configurable.")

    keys = trainer.trained_appliances()
    e_df = appliance_energy_summary(pred_df, keys, tariff)
    st.session_state["energy_df"] = e_df

    with col_r:
        if len(e_df):
            total_pred = e_df["pred_energy_kwh"].sum()
            total_cost = e_df["estimated_cost"].sum()
            agg_kwh    = compute_energy_kwh(pred_df[AGGREGATE_COL])

            c1,c2,c3,c4 = st.columns(4)
            _kpi("Aggregate Energy",      f"{agg_kwh:.3f} kWh",    "Measured",   c1)
            _kpi("Total Predicted Energy",f"{total_pred:.3f} kWh", "Estimated",  c2)
            _kpi("Est. Total Cost",       f"₹{total_cost:.2f}",    f"@₹{tariff}/kWh", c3)
            disc = agg_kwh - total_pred
            _kpi("Disaggregation Gap",    f"{disc:+.3f} kWh",      "agg − pred", c4)

    st.markdown("---")
    _section("Per-Appliance Energy & Cost (Predicted)")
    disp_e = e_df.copy()
    disp_e["appliance"] = disp_e["appliance"].map(lambda k: DISPLAY.get(k,k))
    st.dataframe(disp_e.rename(columns={
        "pred_energy_kwh":  "Pred. Energy (kWh)",
        "peak_pred_w":      "Peak Power (W)",
        "estimated_cost":   "Est. Cost (₹)",
        "energy_share_pct": "Share (%)",
        "true_energy_kwh":  "True Energy (kWh)",
    }), use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        _section("Energy Comparison: Predicted vs. Ground Truth")
        has_true = "true_energy_kwh" in e_df.columns
        y_cols = ["pred_energy_kwh"] + (["true_energy_kwh"] if has_true else [])
        labels = {"pred_energy_kwh": "Predicted", "true_energy_kwh": "Ground Truth"}
        fig_comp = px.bar(
            e_df, x="appliance", y=y_cols, barmode="group",
            color_discrete_sequence=["#2563eb","#059669"],
            labels=labels,
        )
        fig_comp.update_xaxes(ticktext=[DISPLAY.get(k,k) for k in e_df["appliance"]], tickvals=e_df["appliance"])
        fig_comp.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="white", plot_bgcolor="#f8faff",
            yaxis_title="Energy (kWh)", legend=dict(orientation="h",y=1.15))
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_right:
        _section("Daily Energy Trend")
        daily_df = daily_energy_breakdown(pred_df, keys)
        if not daily_df.empty:
            daily_df.columns = [DISPLAY.get(c,c) for c in daily_df.columns]
            fig_d = px.area(
                daily_df.reset_index(), x="date", y=list(daily_df.columns),
                color_discrete_sequence=list(APP_COLORS.values()),
            )
            fig_d.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor="white", plot_bgcolor="#f8faff",
                yaxis_title="Energy (kWh)", legend=dict(orientation="h",y=1.15,font_size=10))
            st.plotly_chart(fig_d, use_container_width=True)

    # Discrepancy
    _section("Aggregate Disaggregation Discrepancy")
    disc_series = aggregate_discrepancy(pred_df, keys)
    if not disc_series.empty:
        fig_disc = go.Figure(go.Scatter(
            x=disc_series.index, y=disc_series,
            name="Discrepancy", line=dict(color="#d97706", width=1.2),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.08)",
        ))
        fig_disc.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig_disc.update_layout(
            height=200, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="white", plot_bgcolor="#f8faff",
            xaxis_title="Time", yaxis_title="Discrepancy (W)",
        )
        st.plotly_chart(fig_disc, use_container_width=True)
        st.caption("Discrepancy = Aggregate − Σ(Appliance predictions).  Non-zero values indicate unmodelled loads or model error.  Predictions are NOT forced to sum to aggregate.")


# ═════════════════════════════════════════════════════════════
# PAGE 7 — MODEL EVALUATION
# ═════════════════════════════════════════════════════════════
elif page == "Model Evaluation":
    st.markdown("<div class='page-title'>📊 Model Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Metrics computed on the held-out test set only</div>", unsafe_allow_html=True)

    test_df = st.session_state["test_df"]
    trainer = st.session_state["trainer"]

    if test_df is None or trainer is None:
        _empty("Preprocessing and training required", "Complete ⚙️ Preprocessing and 🤖 Training first.")
        st.stop()

    if st.button("🔄 Evaluate on Test Set", type="primary", key="btn_eval"):
        with st.spinner("Evaluating…"):
            try:
                keys = trainer.trained_appliances()
                # Only evaluate appliances that have ground-truth in test set
                eval_keys = [k for k in keys if k in test_df.columns]
                if not eval_keys:
                    st.error("No ground-truth appliance columns in the test set.  Evaluation requires labeled data.")
                    st.stop()
                pred_test = predict(trainer, test_df, eval_keys)
                report = evaluate_all(pred_test, trainer, eval_keys)
                st.session_state["eval_report"] = report
                st.success("✅ Evaluation complete.")
            except Exception as e:
                st.error(f"❌ {e}")

    report = st.session_state["eval_report"]
    if report is None:
        _empty("Click 🔄 Evaluate on Test Set to compute metrics")
        st.stop()

    # ── Metrics table ────────────────────────────────────────
    _section("Regression Metrics (Power Estimation)")
    reg_rows = []
    for app, m in report.items():
        if "regression" in m:
            rg = m["regression"]
            bl = m.get("baseline_regression", {})
            reg_rows.append({
                "Appliance":     DISPLAY.get(app,app),
                "MAE (W)":       rg.get("mae_w","—"),
                "RMSE (W)":      rg.get("rmse_w","—"),
                "R²":            rg.get("r2","—"),
                "Baseline MAE":  bl.get("mae_w","—"),
                "Baseline RMSE": bl.get("rmse_w","—"),
            })
    if reg_rows:
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    _section("Classification Metrics (ON/OFF State)")
    clf_rows = []
    for app, m in report.items():
        if "classification" in m:
            cl = m["classification"]
            clf_rows.append({
                "Appliance": DISPLAY.get(app,app),
                "Precision": cl.get("precision","—"),
                "Recall":    cl.get("recall","—"),
                "F1-Score":  cl.get("f1_score","—"),
            })
    if clf_rows:
        st.dataframe(pd.DataFrame(clf_rows), use_container_width=True, hide_index=True)

    _section("Energy Estimation Accuracy")
    energy_rows = []
    for app, m in report.items():
        if "energy" in m:
            eg = m["energy"]
            energy_rows.append({
                "Appliance":         DISPLAY.get(app,app),
                "True Energy (kWh)": eg.get("energy_true_kwh","—"),
                "Pred Energy (kWh)": eg.get("energy_pred_kwh","—"),
                "Abs Error (kWh)":   eg.get("absolute_error_kwh","—"),
                "Rel Error (%)":     eg.get("relative_error_pct","—"),
            })
    if energy_rows:
        st.dataframe(pd.DataFrame(energy_rows), use_container_width=True, hide_index=True)

    # Confusion matrices
    _section("Confusion Matrices")
    has_cm = [app for app,m in report.items() if "classification" in m and "confusion_matrix" in m["classification"]]
    if has_cm:
        cm_cols = st.columns(min(len(has_cm), 3))
        for i, app in enumerate(has_cm):
            cm = report[app]["classification"]["confusion_matrix"]
            cm_df = pd.DataFrame(cm, index=["True OFF","True ON"], columns=["Pred OFF","Pred ON"])
            fig_cm = px.imshow(
                cm_df, text_auto=True, color_continuous_scale="Blues",
                title=DISPLAY.get(app,app),
            )
            fig_cm.update_layout(height=240, margin=dict(l=0,r=0,t=30,b=0), coloraxis_showscale=False)
            cm_cols[i % 3].plotly_chart(fig_cm, use_container_width=True)

    # MAE comparison chart
    if reg_rows:
        _section("ML Model vs. Baseline — MAE Comparison")
        df_cmp = pd.DataFrame(reg_rows)[["Appliance","MAE (W)","Baseline MAE"]]
        df_cmp = df_cmp.dropna()
        fig_cmp = px.bar(
            df_cmp, x="Appliance", y=["MAE (W)","Baseline MAE"], barmode="group",
            color_discrete_sequence=["#2563eb","#94a3b8"],
        )
        fig_cmp.update_layout(
            height=260, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="white", plot_bgcolor="#f8faff",
            yaxis_title="MAE (W)", legend=dict(orientation="h",y=1.15),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.caption("Lower MAE is better.  ML improvement is shown only when the test results support that conclusion.")


# ═════════════════════════════════════════════════════════════
# PAGE 8 — REPORTS & EXPORT
# ═════════════════════════════════════════════════════════════
elif page == "Reports & Export":
    st.markdown("<div class='page-title'>📥 Reports & Export</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Download predictions, energy summaries, and evaluation metrics</div>", unsafe_allow_html=True)

    pred_df   = st.session_state["pred_df"]
    energy_df = st.session_state["energy_df"]
    report    = st.session_state["eval_report"]
    trainer   = st.session_state["trainer"]

    col1, col2 = st.columns(2)

    with col1:
        _section("Timestamped Predictions CSV")
        if pred_df is not None:
            st.write(f"**{len(pred_df):,} rows**, columns include aggregate + predicted power per appliance.")
            export_pred = pred_df.reset_index()
            export_pred.columns = [c.replace("_pred_w", " Predicted (W)")
                                     .replace("_true_w", " Ground Truth (W)")
                                     .replace("_pred_state", " ON/OFF State") for c in export_pred.columns]
            st.download_button(
                "⬇️ Download predictions.csv",
                data=export_pred.to_csv(index=False).encode(),
                file_name="nilm_predictions.csv",
                mime="text/csv",
            )
        else:
            st.info("Generate predictions first.")

        _section("Energy Summary CSV")
        if energy_df is not None and len(energy_df):
            disp_e = energy_df.copy()
            disp_e["appliance"] = disp_e["appliance"].map(lambda k: DISPLAY.get(k,k))
            st.download_button(
                "⬇️ Download energy_summary.csv",
                data=disp_e.to_csv(index=False).encode(),
                file_name="nilm_energy_summary.csv",
                mime="text/csv",
            )
        else:
            st.info("Complete Energy Analysis first.")

    with col2:
        _section("Evaluation Metrics CSV / JSON")
        if report:
            csv_str = metrics_to_csv(report)
            st.download_button(
                "⬇️ Download metrics.csv",
                data=csv_str.encode(),
                file_name="nilm_metrics.csv",
                mime="text/csv",
            )
            json_str = json.dumps(report, indent=2)
            st.download_button(
                "⬇️ Download metrics.json",
                data=json_str.encode(),
                file_name="nilm_metrics.json",
                mime="application/json",
            )
        else:
            st.info("Run Model Evaluation first.")

        _section("Training Summary JSON")
        if trainer:
            ts_data = {
                "appliance_keys": trainer.appliance_keys,
                "n_estimators":   trainer.n_estimators,
                "max_depth":      trainer.max_depth,
                "on_threshold_w": trainer.on_threshold_w,
                "random_state":   trainer.random_state,
                "per_appliance":  trainer.training_summary,
            }
            st.download_button(
                "⬇️ Download training_summary.json",
                data=json.dumps(ts_data, indent=2).encode(),
                file_name="nilm_training_summary.json",
                mime="application/json",
            )
        else:
            st.info("Train a model first.")

    st.markdown("---")
    st.caption("""
    **Column legend**:
    - `aggregate_w` — measured aggregate household power (W)
    - `*_pred_w` — **ML-predicted** appliance power (W); these are estimates, not measurements
    - `*_true_w` — ground-truth appliance label from the dataset (W)
    - `*_pred_state` — predicted ON (1) / OFF (0) state
    - All energy values in kWh, cost values are estimates based on user-configured tariff
    """)


# ═════════════════════════════════════════════════════════════
# PAGE 9 — SETTINGS & INFO
# ═════════════════════════════════════════════════════════════
elif page == "Settings & Info":
    st.markdown("<div class='page-title'>ℹ️ Settings & Project Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Configuration, methodology, and limitations</div>", unsafe_allow_html=True)

    tab_set, tab_method, tab_lim = st.tabs(["⚙️ Settings","📐 Methodology","⚠️ Limitations & Privacy"])

    with tab_set:
        col_l, col_r = st.columns(2)
        with col_l:
            _section("Global Settings")
            tariff = st.number_input("Electricity tariff (₹/kWh)", value=float(st.session_state["tariff"]),
                min_value=0.0, step=0.01, format="%.2f", key="s_tariff")
            st.session_state["tariff"] = tariff

            resamp_opts = ["30s","1min","5min","15min","30min","None"]
            ri = st.selectbox("Default resample interval", resamp_opts,
                index=resamp_opts.index(st.session_state.get("resample_interval","1min")))
            st.session_state["resample_interval"] = ri

        with col_r:
            _section("Supported Appliances")
            for k in SUPPORTED_APPLIANCES:
                st.markdown(f"- **{DISPLAY.get(k,k)}** — key: `{k}`")

            _section("Model Configuration")
            trainer = st.session_state["trainer"]
            if trainer:
                st.write(f"- Trees: `{trainer.n_estimators}`")
                st.write(f"- Max depth: `{trainer.max_depth}`")
                st.write(f"- ON threshold: `{trainer.on_threshold_w}` W")
                st.write(f"- Trained: {', '.join(DISPLAY.get(k,k) for k in trainer.trained_appliances())}")
            else:
                st.info("No model trained yet.")

    with tab_method:
        st.markdown("""
        ## Methodology

        ### Energy Disaggregation Pipeline
        1. **Ingest** — Upload a CSV with a timestamp column and an aggregate power column.
        2. **Validate** — Schema checks, unit parsing, column mapping.
        3. **Preprocess** — Clip negatives, forward-fill gaps, sort chronologically.
        4. **Split** — Chronological train / val / test (no random shuffling to prevent temporal leakage).
        5. **Feature Engineering** — Aggregate-only features: current power, lags (1,2,5,10 steps), rolling mean & max (5, 15 steps), rolling std, first difference, cyclical time encodings, night flag.
        6. **Train** — Random Forest regressor (power estimation) + classifier (ON/OFF) per appliance. Ground-truth appliance columns are **labels only**, never input features.
        7. **Baseline** — Training-set median power is used as the simplest baseline.
        8. **Evaluate** — Regression: MAE, RMSE, R². Classification: Precision, Recall, F1, Confusion matrix. Energy: absolute and relative kWh error.  All metrics on held-out test set only.
        9. **Predict** — Features built from aggregate signal at inference time; no future information used.
        10. **Energy & Cost** — `Energy (kWh) = Σ[Power(W) × Δt(s)] / 3,600,000`. Actual elapsed time used for irregular timestamps.

        ### Data Splitting Rule
        ```
        ├── Train   70%   ← used to fit models and preprocessing
        ├── Val     15%   ← used for early-stopping / monitoring (not hyperparameter tuning in this prototype)
        └── Test    15%   ← held out; used once for final evaluation
        ```

        ### Disclaimer on Predictions
        > Appliance-level outputs are **ML estimates**, not direct electrical measurements.
        > Always compare against ground truth when available and report actual measured metrics.
        """)

    with tab_lim:
        st.markdown("""
        ## Limitations

        | Area | Limitation |
        |------|-----------|
        | **Signature overlap** | Different appliances may have similar power signatures, confusing the model |
        | **Sampling rate** | Low-frequency datasets miss short events (e.g., kettle on for 3 minutes at 30-min resolution) |
        | **Simultaneous use** | Multiple appliances running together reduce disaggregation accuracy |
        | **Household variability** | A model trained on one home may not generalise to another |
        | **Synthetic data** | Does not establish real-world performance; used for software testing only |
        | **Cost estimates** | Depend on tariff accuracy and measurement completeness |
        | **No real-time** | This prototype processes historical CSV data, not live meter feeds |

        ## Privacy Notes

        - Electricity consumption patterns can reveal household routines and occupancy.
        - This application processes data **locally in your browser** via Streamlit — data is not sent to external servers.
        - Do not upload identifiable household data to shared or public deployments without appropriate consent.
        - Predictions must not be used as the sole basis for billing, legal, or safety-critical decisions.

        ## This Is an Academic Prototype
        This project is intended for educational purposes in electrical engineering, machine learning, and energy monitoring.
        It is not a certified meter, billing system, or safety device.
        """)
