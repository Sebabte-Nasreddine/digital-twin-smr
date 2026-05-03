"""
app.py — Tableau de bord Streamlit pour le Jumeau Numérique SMR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import get_config, predict, predict_with_uncertainty, predict_grid
from utils import (
    OUTPUT_COLORS,
    OUTPUT_LABELS,
    carbon_deposition_color,
    kpi_figure,
    response_surface_heatmap,
    sensitivity_bar,
    sobol_bar,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SMR Digital Twin",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}
.main .block-container {
    padding: 2rem 2rem 3rem;
    max-width: 1300px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #94A3B8 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] hr {
    border-color: #1E293B !important;
    margin: 1.25rem 0 !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #CBD5E1 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.75rem;
}
.app-header-left { display: flex; align-items: center; gap: 1rem; }
.app-header-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #6366F1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2);
    flex-shrink: 0;
}
.app-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}
.app-subtitle {
    font-size: 0.78rem;
    color: #94A3B8;
    margin: 0;
    font-weight: 400;
}
.live-badge {
    display: flex; align-items: center; gap: 0.4rem;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 20px;
    padding: 0.3rem 0.75rem;
    font-size: 0.72rem;
    color: #16A34A;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22C55E;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Condition bar ── */
.cond-bar {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    font-size: 0.875rem;
}
.cond-item { display: flex; align-items: baseline; gap: 0.4rem; }
.cond-label { color: #94A3B8; font-size: 0.75rem; font-weight: 500; }
.cond-value { color: #0F172A; font-weight: 600; font-size: 0.9rem; }
.cond-hint { margin-left: auto; font-size: 0.72rem; color: #CBD5E1; }

/* ── KPI cards ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.75rem;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.25rem 1rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow .15s ease;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.kpi-accent {
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 12px 12px 0 0;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94A3B8;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
}
.kpi-unit { font-size: 1rem; font-weight: 500; opacity: 0.55; margin-left: 1px; }
.kpi-sigma {
    font-size: 0.7rem;
    color: #CBD5E1;
    margin-top: 0.4rem;
    font-weight: 500;
}

/* ── Section title ── */
.sec-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 0.25rem;
}
.sec-sub {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-bottom: 1.25rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #E2E8F0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    color: #94A3B8;
    padding: 0.65rem 1.25rem;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    letter-spacing: 0;
    text-transform: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #6366F1 !important;
    border-bottom-color: #6366F1 !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0 !important;
    transition: all .15s ease !important;
    height: 38px !important;
}
.stButton > button[kind="primary"] {
    background: #6366F1 !important;
    border: none !important;
    box-shadow: 0 1px 3px rgba(99,102,241,.35) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4F46E5 !important;
    box-shadow: 0 4px 12px rgba(99,102,241,.4) !important;
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    color: #475569 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #CBD5E1 !important;
    background: #F8FAFC !important;
}

/* ── Inputs / selects ── */
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Result metrics ── */
.result-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.result-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.result-card-label {
    font-size: 0.7rem; font-weight: 600;
    color: #94A3B8; letter-spacing: 0.05em;
    text-transform: uppercase; margin-bottom: 0.35rem;
}
.result-card-value {
    font-size: 1.4rem; font-weight: 700;
    color: #0F172A; letter-spacing: -0.03em;
}
.result-card-unit { font-size: 0.8rem; font-weight: 400; color: #94A3B8; }

/* ── Sidebar meta card ── */
.meta-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-top: 1rem;
}
.meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.22rem 0;
    font-size: 0.75rem;
}
.meta-key { color: #475569 !important; }
.meta-val { color: #94A3B8 !important; font-weight: 600; }

/* ── Divider ── */
hr { border-color: #E2E8F0 !important; margin: 1.5rem 0 !important; }

/* ── Plots ── */
.js-plotly-plot {
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def _load_config():
    return get_config()

try:
    config = _load_config()
except Exception as exc:
    st.error(f"Impossible de charger les modèles : {exc}")
    st.stop()

x1_min = config["inputs"]["X1"]["min"]
x1_max = config["inputs"]["X1"]["max"]
x2_min = config["inputs"]["X2"]["min"]
x2_max = config["inputs"]["X2"]["max"]
x1_mid = (x1_min + x1_max) / 2
x2_mid = (x2_min + x2_max) / 2


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:0.25rem 0 1.25rem">
        <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                    color:#6366F1;text-transform:uppercase;margin-bottom:0.4rem">
            Digital Twin
        </div>
        <div style="font-size:1.05rem;font-weight:700;color:#F1F5F9;
                    letter-spacing:-0.02em;line-height:1.3">
            Reformage Méthane<br>à la Vapeur
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p style="font-size:0.7rem;font-weight:700;color:#475569;'
                'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.75rem">'
                'Conditions opératoires</p>', unsafe_allow_html=True)

    X1 = st.slider(
        "Température — X1 (°C)",
        min_value=float(x1_min), max_value=float(x1_max),
        value=float(x1_mid), step=10.0,
        help=config["inputs"]["X1"]["description"],
    )
    X2 = st.slider(
        "Temps de contact — X2 (g·h/mol)",
        min_value=float(x2_min), max_value=float(x2_max),
        value=float(x2_mid), step=0.5,
        help=config["inputs"]["X2"]["description"],
    )

    st.divider()

    show_uncertainty = st.checkbox("Incertitude ±1σ", value=True)

    st.markdown(f"""
    <div class="meta-card">
        <div class="meta-row">
            <span class="meta-key">Modèle</span>
            <span class="meta-val">Random Forest</span>
        </div>
        <div class="meta-row">
            <span class="meta-key">Échantillons</span>
            <span class="meta-val">{config['n_training_samples']}</span>
        </div>
        <div class="meta-row">
            <span class="meta-key">Source</span>
            <span class="meta-val">{config['data_source']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Predictions ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _predict(x1, x2):
    return predict(x1, x2)

@st.cache_data(show_spinner=False)
def _predict_unc(x1, x2):
    return predict_with_uncertainty(x1, x2)

with st.spinner(""):
    preds = _predict(X1, X2)
    if show_uncertainty:
        unc = _predict_unc(X1, X2)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="app-header">
    <div class="app-header-left">
        <div class="app-header-dot"></div>
        <div>
            <div class="app-title">Reformage du Méthane à la Vapeur</div>
            <div class="app-subtitle">Jumeau Numérique · Modèle de substitution RandomForest</div>
        </div>
    </div>
    <div class="live-badge">
        <div class="live-dot"></div>
        Simulation en direct
    </div>
</div>
""", unsafe_allow_html=True)


# ── Condition bar ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="cond-bar">
    <div class="cond-item">
        <span class="cond-label">Température</span>
        <span class="cond-value">{X1:.0f} °C</span>
    </div>
    <div class="cond-item">
        <span class="cond-label">Temps de contact</span>
        <span class="cond-value">{X2:.2f} g·h/mol</span>
    </div>
    <span class="cond-hint">Déplacez les curseurs pour mettre à jour</span>
</div>
""", unsafe_allow_html=True)


# ── KPI cards ─────────────────────────────────────────────────────────────────

KPI_DEFS = [
    ("Y1", "Conversion CH₄",    "#3B82F6"),
    ("Y2", "Rendement H₂",      "#10B981"),
    ("Y3", "Sélectivité CO₂",   "#F59E0B"),
    ("Y4", "Dépôt de carbone",  None),
]

kpi_html = '<div class="kpi-row">'
for key, label, color in KPI_DEFS:
    val = preds[key]
    col = color if color else carbon_deposition_color(val)
    if show_uncertainty:
        s     = unc[key]["std"]
        sigma = f'<div class="kpi-sigma">± {s:.2f} %</div>'
    else:
        sigma = '<div class="kpi-sigma" style="height:1rem"></div>'

    kpi_html += (
        f'<div class="kpi-card">'
        f'<div class="kpi-accent" style="background:{col}"></div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{col}">{val:.2f}<span class="kpi-unit"> %</span></div>'
        f'{sigma}'
        f'</div>'
    )

kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_surface, tab_sensitivity, tab_optimize = st.tabs([
    "Cartes de réponse",
    "Analyse de sensibilité",
    "Optimisation",
])


# ── Tab 1 : Response surface ──────────────────────────────────────────────────

with tab_surface:
    st.markdown('<div class="sec-title">Cartes de réponse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Comment chaque sortie évolue sur la grille Température × Temps de contact.</div>', unsafe_allow_html=True)

    col_ctrl, col_opts = st.columns([2, 1])
    with col_ctrl:
        target_map = {
            "Conversion CH₄ (Y1)":     "Y1",
            "Rendement H₂ (Y2)":       "Y2",
            "Sélectivité CO₂ (Y3)":    "Y3",
            "Dépôt de carbone (Y4)":   "Y4",
        }
        selected_label  = st.selectbox("Variable", list(target_map.keys()))
        selected_target = target_map[selected_label]
    with col_opts:
        n_grid  = st.slider("Résolution grille", 20, 80, 40, step=10)
        plot_3d = st.checkbox("Vue 3D", value=False)

    @st.cache_data(show_spinner="Calcul en cours…")
    def _grid(n):
        return predict_grid((x1_min, x1_max), (x2_min, x2_max), n_points=n)

    grid_df = _grid(n_grid)

    if plot_3d:
        from utils import response_surface_3d
        fig = response_surface_3d(grid_df, selected_target)
    else:
        fig = response_surface_heatmap(grid_df, selected_target)
        fig.add_trace(go.Scatter(
            x=[X1], y=[X2],
            mode="markers",
            marker=dict(color="white", size=12, symbol="cross",
                        line=dict(color="#0F172A", width=2)),
            name="Point actuel",
            hovertemplate=f"T={X1}°C · W/F={X2:.2f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=40, b=40, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Afficher les 4 sorties simultanément", value=False):
        cols = st.columns(2)
        for i, t in enumerate(["Y1", "Y2", "Y3", "Y4"]):
            with cols[i % 2]:
                f = response_surface_heatmap(grid_df, t)
                f.update_layout(
                    paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                    font=dict(family="Inter, sans-serif", size=11),
                    margin=dict(t=40, b=40, l=40, r=40),
                )
                st.plotly_chart(f, use_container_width=True)


# ── Tab 2 : Sensitivity ───────────────────────────────────────────────────────

with tab_sensitivity:
    st.markdown('<div class="sec-title">Analyse de sensibilité</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Importance relative de chaque variable d\'entrée sur chaque sortie du modèle.</div>', unsafe_allow_html=True)

    fig_sens = sensitivity_bar(config)
    fig_sens.update_layout(
        paper_bgcolor="white", plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=40, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    if st.button("Calculer les indices de Sobol (~10 s)", type="secondary"):
        with st.spinner("Calcul Monte-Carlo en cours…"):
            fig_sobol = sobol_bar(config, n_samples=2_000)
        fig_sobol.update_layout(
            paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            font=dict(family="Inter, sans-serif", size=12),
            margin=dict(t=40, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_sobol, use_container_width=True)


# ── Tab 3 : Optimisation ─────────────────────────────────────────────────────

with tab_optimize:
    st.markdown('<div class="sec-title">Optimisation des conditions opératoires</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Définissez vos contraintes et vos priorités — l\'algorithme trouve automatiquement les meilleures conditions.</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("**Contraintes opératoires**")
        min_Y1 = st.number_input(
            "Conversion CH₄ minimale (%)",
            min_value=50.0, max_value=100.0, value=90.0, step=1.0,
            help="Le modèle cherchera uniquement les conditions dépassant ce seuil.",
        )
        min_Y2 = st.number_input(
            "Rendement H₂ minimal (%)",
            min_value=0.0, max_value=40.0, value=10.0, step=0.5,
        )
        max_Y4 = st.number_input(
            "Dépôt de carbone maximal (%)",
            min_value=0.0, max_value=100.0, value=40.0, step=1.0,
            help="Les solutions au-dessus de ce seuil sont rejetées.",
        )

    with col_r:
        st.markdown("**Importance des objectifs**")
        w_Y1 = st.slider("Conversion CH₄",  0.0, 3.0, 1.0, step=0.1)
        w_Y2 = st.slider("Rendement H₂",    0.0, 3.0, 1.0, step=0.1)
        w_Y3 = st.slider("Sélectivité CO₂", 0.0, 3.0, 0.5, step=0.1)
        w_Y4 = st.slider("Pénalité carbone (négatif = minimiser)", -3.0, 0.0, -2.0, step=0.1)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_opt = st.button("Lancer l'optimisation", type="primary", use_container_width=False)

    if run_opt:
        with st.spinner("Recherche des meilleures conditions…"):
            for _mod in list(sys.modules.keys()):
                if _mod == "optimize":
                    sys.modules.pop(_mod)
            from optimize import run_single_objective, find_optimal
            try:
                st.session_state["opt_result"] = run_single_objective(
                    weights={"Y1": w_Y1, "Y2": w_Y2, "Y3": w_Y3, "Y4": w_Y4}
                )
                st.session_state["opt_constrained"] = find_optimal(
                    min_Y1=min_Y1, min_Y2=min_Y2, max_Y4=max_Y4
                )
                st.session_state["opt_error"] = None
            except Exception as _e:
                st.session_state["opt_error"] = str(_e)

    if st.session_state.get("opt_error"):
        st.error(f"Erreur : {st.session_state['opt_error']}")

    if "opt_result" in st.session_state and not st.session_state.get("opt_error"):
        opt = st.session_state["opt_result"]
        con = st.session_state["opt_constrained"]

        st.divider()

        # ── Global result ──
        st.markdown("**Optimum global** — sans contraintes opératoires")
        st.markdown(f"""
        <div class="result-grid">
            <div class="result-card">
                <div class="result-card-label">Température</div>
                <div class="result-card-value">{opt['X1']:.1f}<span class="result-card-unit"> °C</span></div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Temps de contact</div>
                <div class="result-card-value">{opt['X2']:.3f}<span class="result-card-unit"> g·h/mol</span></div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Conversion CH₄</div>
                <div class="result-card-value" style="color:#3B82F6">{opt['Y1']:.2f}<span class="result-card-unit"> %</span></div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Rendement H₂</div>
                <div class="result-card-value" style="color:#10B981">{opt['Y2']:.2f}<span class="result-card-unit"> %</span></div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Sélectivité CO₂</div>
                <div class="result-card-value" style="color:#F59E0B">{opt['Y3']:.2f}<span class="result-card-unit"> %</span></div>
            </div>
            <div class="result-card">
                <div class="result-card-label">Dépôt de carbone</div>
                <div class="result-card-value" style="color:#EF4444">{opt['Y4']:.2f}<span class="result-card-unit"> %</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Constrained result ──
        st.markdown("**Optimum contraint** — avec vos contraintes opératoires")

        if con and not con.get("infeasible"):
            st.markdown(f"""
            <div class="result-grid">
                <div class="result-card">
                    <div class="result-card-label">Température</div>
                    <div class="result-card-value">{con['X1']:.1f}<span class="result-card-unit"> °C</span></div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">Temps de contact</div>
                    <div class="result-card-value">{con['X2']:.3f}<span class="result-card-unit"> g·h/mol</span></div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">Conversion CH₄</div>
                    <div class="result-card-value" style="color:#3B82F6">{con['Y1']:.2f}<span class="result-card-unit"> %</span></div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">Rendement H₂</div>
                    <div class="result-card-value" style="color:#10B981">{con['Y2']:.2f}<span class="result-card-unit"> %</span></div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">Sélectivité CO₂</div>
                    <div class="result-card-value" style="color:#F59E0B">{con['Y3']:.2f}<span class="result-card-unit"> %</span></div>
                </div>
                <div class="result-card">
                    <div class="result-card-label">Dépôt de carbone</div>
                    <div class="result-card-value" style="color:#EF4444">{con['Y4']:.2f}<span class="result-card-unit"> %</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif con and con.get("infeasible"):
            reason = con.get("reason")
            if reason == "y4":
                min_y4 = con["min_y4_achievable"]
                st.warning(
                    f"Contrainte dépôt de carbone ≤ {max_Y4:.1f}% impossible à satisfaire.\n\n"
                    f"Avec Y1 ≥ {min_Y1:.0f}% et Y2 ≥ {min_Y2:.1f}%, "
                    f"le minimum atteignable est **{min_y4:.1f}%** "
                    f"(T = {con['best_X1']:.0f}°C, W/F = {con['best_X2']:.2f} g·h/mol).\n\n"
                    f"Conseil : relevez le seuil à **{min_y4 + 1:.0f}%** minimum."
                )
            else:
                st.warning(
                    f"Y1 ≥ {min_Y1:.0f}% et Y2 ≥ {min_Y2:.1f}% ne peuvent pas être "
                    f"atteints simultanément. Max modèle : Y1 = {con['max_y1']:.1f}%, "
                    f"Y2 = {con['max_y2']:.1f}%."
                )

        # ── Diagnostic ──
        with st.expander("Diagnostic — plages réelles du modèle"):
            from predict import predict_grid as _pg
            _d  = _pg((x1_min, x1_max), (x2_min, x2_max), n_points=40)
            _n  = len(_d)
            _c1 = int((_d["Y1"] >= min_Y1).sum())
            _c2 = int((_d["Y2"] >= min_Y2).sum())
            _c4 = int((_d["Y4"] <= max_Y4).sum())
            _ca = int(((_d["Y1"] >= min_Y1) & (_d["Y2"] >= min_Y2) & (_d["Y4"] <= max_Y4)).sum())
            st.markdown(f"""
| Contrainte | Seuil | Points OK / {_n} | Min modèle | Max modèle |
|:---|:---:|:---:|:---:|:---:|
| Conversion CH₄ (Y1) | ≥ {min_Y1:.1f}% | **{_c1}** | {_d['Y1'].min():.1f}% | {_d['Y1'].max():.1f}% |
| Rendement H₂ (Y2)   | ≥ {min_Y2:.1f}% | **{_c2}** | {_d['Y2'].min():.1f}% | {_d['Y2'].max():.1f}% |
| Dépôt carbone (Y4)  | ≤ {max_Y4:.1f}% | **{_c4}** | {_d['Y4'].min():.1f}% | {_d['Y4'].max():.1f}% |
| **Toutes ensemble** | — | **{_ca}** | — | — |
""")
            if _ca == 0:
                st.info(
                    f"Aucun des {_n} points de grille ne satisfait toutes les contraintes. "
                    f"Y4 minimum atteignable : **{_d['Y4'].min():.1f}%**."
                )
