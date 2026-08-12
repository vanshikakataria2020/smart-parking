"""
Smart Parking Availability Prediction System
A professional Streamlit dashboard for ML-based parking occupancy prediction.
Author: Academic ML Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Parking AI Dashboard",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary:   #0b0f1a;
        --bg-card:      #111827;
        --bg-card2:     #161d2e;
        --accent-blue:  #3b82f6;
        --accent-cyan:  #06b6d4;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red:   #ef4444;
        --text-primary: #f1f5f9;
        --text-muted:   #94a3b8;
        --border:       rgba(255,255,255,0.07);
        --glow-blue:    rgba(59,130,246,0.25);
        --glow-cyan:    rgba(6,182,212,0.20);
    }

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    .stApp { background-color: var(--bg-primary); }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; max-width: 1400px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0d1424 0%, #111827 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--accent-cyan);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* ── Widget labels ── */
    .stSlider label, .stNumberInput label,
    .stSelectbox label, .stRadio label {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em;
    }

    /* ── Predict button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #0891b2);
        color: #fff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.03em;
        width: 100%;
        cursor: pointer;
        box-shadow: 0 4px 20px var(--glow-blue);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px var(--glow-blue);
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .metric-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
    .metric-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .metric-card.red::before   { background: linear-gradient(90deg, #ef4444, #f87171); }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
        color: var(--text-primary);
    }
    .metric-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }
    .metric-icon {
        position: absolute;
        right: 1.2rem;
        top: 1.2rem;
        font-size: 1.6rem;
        opacity: 0.35;
    }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 2rem 0 1rem 0;
    }
    .section-header span {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.01em;
    }
    .section-dot {
        width: 8px; height: 8px;
        background: var(--accent-cyan);
        border-radius: 50%;
        box-shadow: 0 0 8px var(--accent-cyan);
    }

    /* ── Prediction result ── */
    .result-LOW {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-MEDIUM {
        background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
        border: 1px solid rgba(245,158,11,0.4);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-HIGH {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    .result-level {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        line-height: 1;
        margin-bottom: 0.4rem;
    }
    .result-LOW    .result-level { color: #10b981; }
    .result-MEDIUM .result-level { color: #f59e0b; }
    .result-HIGH   .result-level { color: #ef4444; }
    .result-desc {
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0d1424 0%, #0f1f3d 50%, #0b1a2f 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '🅿️';
        position: absolute;
        right: 2rem; top: 50%;
        transform: translateY(-50%);
        font-size: 6rem;
        opacity: 0.08;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        max-width: 600px;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.3);
        color: #60a5fa;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* ── Chart containers ── */
    .chart-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .chart-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.8rem;
    }

    /* ── Footer ── */
    .footer-bar {
        margin-top: 3rem;
        padding: 1.2rem 2rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        text-align: center;
    }
    .footer-bar p {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin: 0;
    }
    .footer-bar strong {
        color: var(--accent-cyan);
    }

    /* ── Plotly charts ── */
    .js-plotly-plot .plotly { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS – MODEL & DATA
# ─────────────────────────────────────────────
FEATURES = [
    "total_slots", "occupied_slots", "avg_parking_duration_minutes",
    "entry_count", "exit_count", "parking_fee_collected",
    "Hour", "Day", "Month"
]
CLASSES = ["Low", "Medium", "High"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora, sans-serif", color="#94a3b8", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=10),
    ),
)

AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")


@st.cache_resource
def load_model():
    """Load trained RandomForest model from pickle, or return a dummy model."""
    model_paths = ["parking_model.pkl", "model.pkl", "rf_model.pkl"]
    for path in model_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), True
    # ── Fallback: create a simple rule-based estimator for demo ──
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    rng = np.random.default_rng(42)
    n = 1000
    total  = rng.integers(50, 300, n)
    occ    = (total * rng.uniform(0.0, 1.3, n)).astype(int)
    dur    = rng.integers(10, 360, n)
    entry  = rng.integers(0, 150, n)
    exit_  = rng.integers(0, 150, n)
    fee    = rng.uniform(500, 8000, n)
    hour   = rng.integers(0, 24, n)
    day    = rng.integers(0, 7, n)
    month  = rng.integers(1, 13, n)
    X = np.column_stack([total, occ, dur, entry, exit_, fee, hour, day, month])
    rate = np.clip(occ / np.where(total == 0, 1, total), 0, 1)
    y = np.where(rate < 0.4, 0, np.where(rate < 0.75, 1, 2))
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    clf.classes_ = np.array([0, 1, 2])
    return clf, False


@st.cache_data
def generate_demo_trend():
    """Generate synthetic 24-hour occupancy trend data."""
    hours = list(range(24))
    low    = [max(0, 30 + 15*np.sin(h/3) + np.random.randint(-5, 5)) for h in hours]
    medium = [max(0, 40 - 10*np.cos(h/4) + np.random.randint(-5, 5)) for h in hours]
    high   = [max(0, 30 + 8*np.sin((h-12)/3) + np.random.randint(-5, 5)) for h in hours]
    return hours, low, medium, high


def predict_occupancy(model, inputs: dict) -> tuple[str, np.ndarray]:
    """Run model prediction and return label + probability array."""
    X = np.array([[inputs[f] for f in FEATURES]])
    pred_idx  = model.predict(X)[0]
    try:
        proba = model.predict_proba(X)[0]
    except Exception:
        proba = np.array([0.33, 0.33, 0.34])
    label = CLASSES[pred_idx]
    return label, proba


# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
def make_pie(label: str, proba: np.ndarray) -> go.Figure:
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    fig = go.Figure(go.Pie(
        labels=CLASSES,
        values=(proba * 100).round(1),
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0b0f1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, family="Sora, sans-serif", color="#f1f5f9"),
        hovertemplate="<b>%{label}</b><br>Probability: %{value}%<extra></extra>",
    ))
    color_map = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    fig.add_annotation(
        text=f"<b>{label}</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=color_map.get(label, "#f1f5f9"), family="Sora, sans-serif"),
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=260, title_text="Occupancy Distribution", xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
    return fig


def make_trend(hours, low, medium, high) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=low, name="Low",
        line=dict(color="#10b981", width=2.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="Hour %{x}:00 | Low: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=medium, name="Medium",
        line=dict(color="#f59e0b", width=2.5),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
        hovertemplate="Hour %{x}:00 | Medium: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=high, name="High",
        line=dict(color="#ef4444", width=2.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
        hovertemplate="Hour %{x}:00 | High: %{y}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280, title_text="24-Hour Occupancy Trend")
    fig.update_xaxes(title="Hour of Day", tickmode="linear", dtick=4, **AXIS_STYLE)
    fig.update_yaxes(title="Count", **AXIS_STYLE)
    return fig


def make_feature_importance(model) -> go.Figure:
    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.random.dirichlet(np.ones(len(FEATURES)))

    labels = [
        "Total Slots", "Occupied Slots", "Avg Duration",
        "Entry Count", "Exit Count", "Fee Collected",
        "Hour", "Day", "Month"
    ]
    idx = np.argsort(importances)[::-1]
    sorted_imp   = importances[idx]
    sorted_label = [labels[i] for i in idx]

    colors = [
        f"rgba(59,130,246,{0.4 + 0.6*(v/max(sorted_imp))})"
        for v in sorted_imp
    ]

    fig = go.Figure(go.Bar(
        x=sorted_imp,
        y=sorted_label,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)")),
        text=[f"{v:.3f}" for v in sorted_imp],
        textposition="outside",
        textfont=dict(size=10, color="#94a3b8"),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300, title_text="Feature Importance (RandomForest)")
    fig.update_xaxes(title="Importance Score", **AXIS_STYLE)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", zerolinecolor="rgba(0,0,0,0)")
    return fig


def make_gauge(pct: float) -> go.Figure:
    color = "#10b981" if pct < 40 else ("#f59e0b" if pct < 75 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(pct, 1),
        number=dict(suffix="%", font=dict(size=28, color=color, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#475569", tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(255,255,255,0.03)",
            borderwidth=0,
            steps=[
                dict(range=[0, 40],  color="rgba(16,185,129,0.08)"),
                dict(range=[40, 75], color="rgba(245,158,11,0.08)"),
                dict(range=[75, 100],color="rgba(239,68,68,0.08)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=pct),
        ),
        title=dict(text="Occupancy %", font=dict(size=11, color="#94a3b8")),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200,
                      margin=dict(l=20, r=20, t=30, b=10))
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar() -> dict:
    st.markdown("""<div class="section-header"><div class="section-dot"></div>
    <span>Input Controls</span></div>""", unsafe_allow_html=True)

    with st.expander("⚙️  Configure Inputs & Predict", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**🏗️ Slot Configuration**")
            total_slots = st.number_input("Total Slots", min_value=10, max_value=1000, value=200, step=10)
            occupied_slots = st.number_input(
                "Occupied Slots", min_value=0, max_value=int(total_slots * 1.3),
                value=min(120, total_slots), step=1
            )

        with col2:
            st.markdown("**🚗 Traffic Metrics**")
            entry_count = st.slider("Entry Count", 0, 200, 65)
            exit_count  = st.slider("Exit Count",  0, 200, 50)
            avg_dur     = st.slider("Avg Duration (min)", 5, 480, 120)

        with col3:
            st.markdown("**💰 Revenue**")
            fee = st.number_input("Fee Collected (₹)", min_value=0.0, max_value=20000.0,
                                  value=3500.0, step=100.0)
            st.markdown("**🕒 Date & Time**")
            hour  = st.selectbox("Hour of Day", list(range(24)), index=9,
                                  format_func=lambda h: f"{h:02d}:00")

        with col4:
            st.markdown("**📅 Day & Month**")
            day   = st.selectbox("Day of Week", list(range(7)), index=0,
                                  format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d])
            month = st.selectbox("Month", list(range(1, 13)), index=0,
                                  format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                          "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔮  Predict Occupancy Level")

    return dict(
        total_slots=total_slots, occupied_slots=occupied_slots,
        avg_parking_duration_minutes=avg_dur,
        entry_count=entry_count, exit_count=exit_count,
        parking_fee_collected=fee,
        Hour=hour, Day=day, Month=month,
    ), predict_btn


# ─────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────
def render_metric_cards(inputs: dict):
    total    = inputs["total_slots"]
    occupied = inputs["occupied_slots"]
    avail    = max(0, total - occupied)
    pct      = min(100.0, (occupied / total * 100) if total > 0 else 0.0)

    cols = st.columns(4)
    cards = [
        ("blue",  "🏢", "Total Slots",     total,    "parking facility capacity"),
        ("red",   "🚗", "Occupied Slots",  occupied, "vehicles currently parked"),
        ("green", "✅", "Available Slots", avail,    "open spaces right now"),
        ("amber", "📊", "Occupancy Rate",  f"{pct:.1f}%", "current fill level"),
    ]
    for col, (color, icon, label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
    return pct


# ─────────────────────────────────────────────
# PREDICTION RESULT CARD
# ─────────────────────────────────────────────
def render_prediction(label: str, proba: np.ndarray):
    descriptions = {
        "Low":    "🟢 Parking is comfortably available. No immediate action needed.",
        "Medium": "🟡 Moderate occupancy — monitor closely for peak buildup.",
        "High":   "🔴 Critical occupancy! Consider directing drivers to alternate zones.",
    }
    icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    pct_str = f"{proba[CLASSES.index(label)]*100:.1f}%"
    st.markdown(f"""
    <div class="result-{label}">
        <div class="result-title">Predicted Occupancy Level &nbsp; · &nbsp; Confidence: {pct_str}</div>
        <div class="result-level">{icons[label]} &nbsp; {label.upper()}</div>
        <div class="result-desc">{descriptions[label]}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    model, model_loaded = load_model()
    inputs, predict_btn = render_sidebar()

    # ── Hero ──────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🤖 AI-Powered · RandomForest Classifier</div>
        <div class="hero-title">Smart Parking Availability<br>Prediction System</div>
        <div class="hero-subtitle">
            A real-time machine learning dashboard that predicts parking occupancy levels
            (Low / Medium / High) based on live slot analytics, traffic flow, and temporal features —
            built to demonstrate smart city infrastructure intelligence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.info(
            "ℹ️  **Demo mode** — no `parking_model.pkl` found in the working directory. "
            "A surrogate RandomForest model trained on synthetic data is being used. "
            "Place your trained `parking_model.pkl` alongside `app.py` to use your real model."
        )

    # ── Metric cards ──────────────────────────
    st.markdown("""<div class="section-header"><div class="section-dot"></div>
    <span>Live Parking Overview</span></div>""", unsafe_allow_html=True)
    occupancy_pct = render_metric_cards(inputs)

    # ── Prediction ────────────────────────────
    st.markdown("""<div class="section-header"><div class="section-dot"></div>
    <span>Occupancy Prediction</span></div>""", unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Running ML inference…"):
            label, proba = predict_occupancy(model, inputs)
        st.session_state["pred_label"] = label
        st.session_state["pred_proba"] = proba

    if "pred_label" in st.session_state:
        render_prediction(st.session_state["pred_label"], st.session_state["pred_proba"])
    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px; padding:1.5rem; text-align:center; color:#475569; font-size:0.85rem;'>
            ← Adjust inputs in the sidebar and click <b>Predict Occupancy Level</b> to see the result.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1: Pie + Gauge ──────────────
    st.markdown("""<div class="section-header"><div class="section-dot"></div>
    <span>Dashboard Analytics</span></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.4, 1])

    proba_disp = (
        st.session_state["pred_proba"]
        if "pred_proba" in st.session_state
        else np.array([0.33, 0.34, 0.33])
    )
    label_disp = st.session_state.get("pred_label", "—")

    with c1:
        st.markdown('<div class="chart-box"><div class="chart-title">🥧 Occupancy Probability Distribution</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_pie(label_disp, proba_disp), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-box"><div class="chart-title">⚡ Occupancy Rate Gauge</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_gauge(occupancy_pct), use_container_width=True, config={"displayModeBar": False})

        # Quick stats
        net_flow = inputs["entry_count"] - inputs["exit_count"]
        flow_color = "#ef4444" if net_flow > 0 else "#10b981"
        flow_arrow = "↑" if net_flow > 0 else "↓"
        st.markdown(f"""
        <div style='display:flex; gap:0.8rem; margin-top:0.5rem;'>
            <div style='flex:1; background:rgba(255,255,255,0.03); border-radius:10px; padding:0.8rem; text-align:center;'>
                <div style='font-size:0.65rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;'>Net Flow</div>
                <div style='font-size:1.3rem; font-weight:800; color:{flow_color}; font-family:JetBrains Mono;'>{flow_arrow}{abs(net_flow)}</div>
            </div>
            <div style='flex:1; background:rgba(255,255,255,0.03); border-radius:10px; padding:0.8rem; text-align:center;'>
                <div style='font-size:0.65rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;'>Fee / Slot</div>
                <div style='font-size:1.3rem; font-weight:800; color:#60a5fa; font-family:JetBrains Mono;'>
                    ₹{(inputs["parking_fee_collected"]/max(1,inputs["occupied_slots"])):.0f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Charts row 2: Trend + Importance ──────
    c3, c4 = st.columns(2)
    hours, low, medium, high = generate_demo_trend()

    with c3:
        st.markdown('<div class="chart-box"><div class="chart-title">📈 24-Hour Occupancy Trend</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_trend(hours, low, medium, high), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-box"><div class="chart-title">🧠 Feature Importance</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_feature_importance(model), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Model info strip ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-header"><div class="section-dot"></div>
    <span>Model Information</span></div>""", unsafe_allow_html=True)

    info_cols = st.columns(4)
    info_items = [
        ("🤖 Algorithm",    "RandomForestClassifier"),
        ("📦 Output",       "Low · Medium · High"),
        ("🔢 Features",     "9 input features"),
        ("📚 Framework",    "scikit-learn + Streamlit"),
    ]
    for col, (icon_label, val) in zip(info_cols, info_items):
        with col:
            label_txt, icon = icon_label.split(" ", 1) if " " in icon_label else (icon_label, "")
            st.markdown(f"""
            <div style='background:var(--bg-card); border:1px solid var(--border);
                        border-radius:12px; padding:1rem; text-align:center;'>
                <div style='font-size:1.4rem; margin-bottom:0.3rem;'>{icon_label.split()[0]}</div>
                <div style='font-size:0.65rem; color:#475569; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.08em;'>{" ".join(icon_label.split()[1:])}</div>
                <div style='font-size:0.82rem; color:#f1f5f9; font-weight:600; margin-top:0.25rem;'>{val}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────
    now = datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(f"""
    <div class="footer-bar" style="margin-top:2.5rem;">
        <p>
            <strong>Smart Parking Availability Prediction System</strong> &nbsp;·&nbsp;
            Developed for Academic ML Project Demonstration &nbsp;·&nbsp;
            🅿️ ParkSense AI Dashboard &nbsp;·&nbsp;
            <span style='color:#334155;'>Last updated: {now}</span>
        </p>
        <p style='margin-top:0.4rem; color:#1e293b; font-size:0.65rem;'>
            Powered by RandomForest · scikit-learn · Streamlit · Plotly
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()