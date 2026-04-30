"""
app.py — Tableau de bord Streamlit pour le Jumeau Numérique SMR.

Lancer :
    cd digital_twin_smr
    streamlit run app.py
"""

from __future__ import annotations

import io
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
    pareto_scatter,
    response_surface_heatmap,
    sensitivity_bar,
    sobol_bar,
)

# ── Configuration de la page ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Jumeau Numérique SMR",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS professionnel ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Bannière d'en-tête ── */
    .smr-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1B3A4B 50%, #0D2137 100%);
        border-bottom: 3px solid #00BCD4;
        padding: 2rem 2.5rem 1.6rem;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .smr-header-icon svg {
        filter: drop-shadow(0 0 12px rgba(0,188,212,0.6));
    }
    .smr-header-text h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .smr-header-text p {
        margin: 0.3rem 0 0;
        font-size: 0.85rem;
        color: #90CAF9;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .smr-header-badge {
        margin-left: auto;
        background: rgba(0,188,212,0.15);
        border: 1px solid rgba(0,188,212,0.4);
        border-radius: 6px;
        padding: 0.4rem 0.9rem;
        font-size: 0.78rem;
        color: #00BCD4;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* ── Barre de conditions ── */
    .op-bar {
        background: #F0F4F8;
        border: 1px solid #DDE3EA;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-size: 0.9rem;
        color: #2D3748;
        margin-bottom: 1.5rem;
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    .op-bar span { font-weight: 600; color: #0D1B2A; }
    .op-label { font-size: 0.78rem; color: #718096; margin-right: 0.25rem; }

    /* ── Cartes KPI ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.25rem 1rem 1rem;
        border-top: 4px solid #2196F3;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07), 0 0 0 1px rgba(0,0,0,0.04);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .kpi-icon {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 0.6rem;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .kpi-unit {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.7;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.4rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .kpi-sigma {
        font-size: 0.72rem;
        color: #A0AEC0;
        margin-top: 0.25rem;
    }

    /* ── Titre de section ── */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1A202C;
        letter-spacing: -0.01em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #EDF2F7;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0D1B2A !important;
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E0 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #90CAF9 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .sidebar-brand-text {
        font-size: 1rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .sidebar-brand-sub {
        font-size: 0.7rem;
        color: #4FC3F7 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .sidebar-meta {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.75rem;
        margin-top: 1rem;
        font-size: 0.78rem;
    }
    .sidebar-meta-row {
        display: flex;
        justify-content: space-between;
        padding: 0.2rem 0;
    }
    .sidebar-meta-key { color: #718096 !important; }
    .sidebar-meta-val { color: #90CAF9 !important; font-weight: 600; }

    /* ── Onglets ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 2px solid #EDF2F7;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.6rem 1.1rem;
        border-radius: 6px 6px 0 0;
        color: #718096;
    }
    .stTabs [aria-selected="true"] {
        color: #0D1B2A !important;
        background: #EDF2F7;
        border-bottom: 2px solid #0D1B2A;
    }

    /* ── Boutons ── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1565C0, #0D47A1) !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(21,101,192,0.35) !important;
    }

    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        overflow: hidden;
    }

    .stDownloadButton > button {
        background: #FFFFFF !important;
        border: 1.5px solid #CBD5E0 !important;
        color: #2D3748 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #4A5568 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    hr { border-color: #EDF2F7 !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Icônes SVG ────────────────────────────────────────────────────────────────

REACTOR_ICON = """
<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="16" y="4" width="16" height="6" rx="2" fill="#00BCD4"/>
  <rect x="12" y="10" width="24" height="28" rx="4" fill="none" stroke="#00BCD4" stroke-width="2"/>
  <rect x="16" y="38" width="16" height="6" rx="2" fill="#00BCD4"/>
  <line x1="12" y1="20" x2="6" y2="20" stroke="#4FC3F7" stroke-width="2"/>
  <line x1="6" y1="20" x2="6" y2="28" stroke="#4FC3F7" stroke-width="2"/>
  <line x1="6" y1="28" x2="12" y2="28" stroke="#4FC3F7" stroke-width="2"/>
  <line x1="36" y1="20" x2="42" y2="20" stroke="#4FC3F7" stroke-width="2"/>
  <line x1="42" y1="20" x2="42" y2="28" stroke="#4FC3F7" stroke-width="2"/>
  <line x1="42" y1="28" x2="36" y2="28" stroke="#4FC3F7" stroke-width="2"/>
  <circle cx="24" cy="24" r="6" fill="rgba(0,188,212,0.2)" stroke="#00BCD4" stroke-width="1.5"/>
  <circle cx="24" cy="24" r="2.5" fill="#00BCD4"/>
</svg>"""

ICON_CONVERSION = """<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <path d="M3 10 Q5 4 10 4 Q15 4 17 10" stroke="#1565C0" stroke-width="2" stroke-linecap="round" fill="none"/>
  <path d="M3 10 Q5 16 10 16 Q15 16 17 10" stroke="#1565C0" stroke-width="1" stroke-dasharray="2 2" stroke-linecap="round" fill="none"/>
  <circle cx="10" cy="10" r="2.5" fill="#1565C0"/>
</svg>"""

ICON_YIELD = """<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <polyline points="3,15 7,9 11,12 17,4" stroke="#2E7D32" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="17" cy="4" r="2" fill="#2E7D32"/>
</svg>"""

ICON_SELECTIVITY = """<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <circle cx="10" cy="10" r="7" stroke="#E65100" stroke-width="2" fill="none"/>
  <path d="M10 3 L10 10 L15 10" stroke="#E65100" stroke-width="2" stroke-linecap="round"/>
</svg>"""

ICON_CARBON = """<svg width="20" height="20" viewBox="0 0 20 20" fill="none">
  <path d="M10 2 L18 16 H2 Z" stroke="#C62828" stroke-width="2" stroke-linejoin="round" fill="rgba(198,40,40,0.15)"/>
  <line x1="10" y1="8" x2="10" y2="12" stroke="#C62828" stroke-width="2" stroke-linecap="round"/>
  <circle cx="10" cy="14.5" r="1" fill="#C62828"/>
</svg>"""

KPI_ICONS = {"Y1": ICON_CONVERSION, "Y2": ICON_YIELD, "Y3": ICON_SELECTIVITY, "Y4": ICON_CARBON}

SIDEBAR_ICON_TEMP = """<svg width="18" height="18" viewBox="0 0 18 18" fill="none">
  <rect x="7" y="2" width="4" height="9" rx="2" stroke="#4FC3F7" stroke-width="1.5" fill="none"/>
  <circle cx="9" cy="13" r="3" stroke="#4FC3F7" stroke-width="1.5" fill="rgba(79,195,247,0.2)"/>
  <line x1="9" y1="7" x2="9" y2="11" stroke="#4FC3F7" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""

SIDEBAR_ICON_TIME = """<svg width="18" height="18" viewBox="0 0 18 18" fill="none">
  <circle cx="9" cy="9" r="7" stroke="#4FC3F7" stroke-width="1.5" fill="none"/>
  <polyline points="9,5 9,9 12,12" stroke="#4FC3F7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


# ── Chargement du modèle ──────────────────────────────────────────────────────

@st.cache_resource
def _load_config():
    return get_config()


try:
    config = _load_config()
except Exception as exc:
    st.error(f"Impossible de charger les modèles : {exc}\n\nLancez d'abord `python src/extract_models.py`.")
    st.stop()

x1_min = config["inputs"]["X1"]["min"]
x1_max = config["inputs"]["X1"]["max"]
x2_min = config["inputs"]["X2"]["min"]
x2_max = config["inputs"]["X2"]["max"]
x1_mid = (x1_min + x1_max) / 2
x2_mid = (x2_min + x2_max) / 2


# ── Barre latérale ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-brand">
        {REACTOR_ICON}
        <div>
            <div class="sidebar-brand-text">Jumeau Numérique SMR</div>
            <div class="sidebar-brand-sub">Simulateur de procédé</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem">
        {SIDEBAR_ICON_TEMP}
        <span style="font-size:0.78rem;font-weight:600;color:#90CAF9;text-transform:uppercase;letter-spacing:0.04em">
            Température X1 (°C)
        </span>
    </div>""", unsafe_allow_html=True)

    X1 = st.slider(
        "Température X1",
        min_value=float(x1_min),
        max_value=float(x1_max),
        value=float(x1_mid),
        step=10.0,
        help=config["inputs"]["X1"]["description"],
        label_visibility="collapsed",
    )

    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;margin-top:0.75rem">
        {SIDEBAR_ICON_TIME}
        <span style="font-size:0.78rem;font-weight:600;color:#90CAF9;text-transform:uppercase;letter-spacing:0.04em">
            Temps de contact X2 (g·h/mol)
        </span>
    </div>""", unsafe_allow_html=True)

    X2 = st.slider(
        "Temps de contact X2",
        min_value=float(x2_min),
        max_value=float(x2_max),
        value=float(x2_mid),
        step=0.5,
        help=config["inputs"]["X2"]["description"],
        label_visibility="collapsed",
    )

    st.markdown("<hr/>", unsafe_allow_html=True)

    show_uncertainty = st.checkbox("Afficher l'incertitude ±1σ", value=True)

    st.markdown(f"""
    <div class="sidebar-meta">
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Modèle</span>
            <span class="sidebar-meta-val">{config['model_type']}</span>
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Points d'entraînement</span>
            <span class="sidebar-meta-val">{config['n_training_samples']}</span>
        </div>
        <div class="sidebar-meta-row">
            <span class="sidebar-meta-key">Source</span>
            <span class="sidebar-meta-val">{config['data_source']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Calcul des prédictions ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _predict(x1, x2):
    return predict(x1, x2)


@st.cache_data(show_spinner=False)
def _predict_unc(x1, x2):
    return predict_with_uncertainty(x1, x2)


with st.spinner("Calcul en cours ..."):
    preds = _predict(X1, X2)
    if show_uncertainty:
        unc = _predict_unc(X1, X2)


# ── En-tête principal ─────────────────────────────────────────────────────────

st.markdown(f"""
<div class="smr-header">
    <div class="smr-header-icon">{REACTOR_ICON}</div>
    <div class="smr-header-text">
        <h1>Reformage du Méthane à la Vapeur</h1>
        <p>Jumeau Numérique — Tableau de bord du modèle de substitution</p>
    </div>
    <div class="smr-header-badge">Simulation en direct</div>
</div>
""", unsafe_allow_html=True)

# ── Barre de conditions opératoires ──────────────────────────────────────────

st.markdown(f"""
<div class="op-bar">
    <div>
        <span class="op-label">Température</span>
        <span>T = {X1:.0f} °C</span>
    </div>
    <div>
        <span class="op-label">Temps de contact</span>
        <span>W/F = {X2:.2f} g<sub>cat</sub>·h/mol</span>
    </div>
    <div style="margin-left:auto;font-size:0.78rem;color:#A0AEC0">
        Les résultats se mettent à jour automatiquement en déplaçant les curseurs
    </div>
</div>
""", unsafe_allow_html=True)

# ── Cartes KPI ────────────────────────────────────────────────────────────────

kpi_defs = [
    ("Y1", preds["Y1"], "Conversion CH₄", OUTPUT_COLORS["Y1"], KPI_ICONS["Y1"]),
    ("Y2", preds["Y2"], "Rendement H₂",   OUTPUT_COLORS["Y2"], KPI_ICONS["Y2"]),
    ("Y3", preds["Y3"], "Sélectivité CO₂", OUTPUT_COLORS["Y3"], KPI_ICONS["Y3"]),
    ("Y4", preds["Y4"], "Dépôt de carbone", carbon_deposition_color(preds["Y4"]), KPI_ICONS["Y4"]),
]

kpi_html = '<div class="kpi-grid">'
for key, val, label, color, icon in kpi_defs:
    sigma_html = ""
    if show_uncertainty:
        sigma = unc[key]["std"]
        sigma_html = f'<div class="kpi-sigma">&plusmn; {sigma:.2f}%</div>'

    kpi_html += f"""
    <div class="kpi-card" style="border-top-color:{color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="color:{color}">{val:.2f}<span class="kpi-unit">%</span></div>
        <div class="kpi-label">{label}</div>
        {sigma_html}
    </div>"""

kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────

tab_surface, tab_sensitivity, tab_optimize, tab_export = st.tabs([
    "Cartes de réponse",
    "Analyse de sensibilité",
    "Optimisation",
    "Exporter les données",
])


# ── Onglet 1 : Cartes de réponse ──────────────────────────────────────────────

with tab_surface:
    st.markdown('<div class="section-title">Cartes de réponse du procédé</div>', unsafe_allow_html=True)
    st.caption("Visualisez comment chaque sortie évolue en fonction de la température et du temps de contact.")

    col_ctrl, col_opts = st.columns([2, 1])

    with col_ctrl:
        target_map = {
            "Conversion CH₄ (Y1)": "Y1",
            "Rendement H₂ (Y2)":   "Y2",
            "Sélectivité CO₂ (Y3)": "Y3",
            "Dépôt de carbone (Y4)": "Y4",
        }
        selected_label = st.selectbox("Variable à afficher", list(target_map.keys()))
        selected_target = target_map[selected_label]

    with col_opts:
        n_grid = st.slider("Précision de la grille", 20, 80, 40, step=10)
        plot_3d = st.checkbox("Vue en 3D", value=False)

    @st.cache_data(show_spinner="Calcul de la carte de réponse ...")
    def _grid(n):
        return predict_grid((x1_min, x1_max), (x2_min, x2_max), n_points=n)

    grid_df = _grid(n_grid)

    if plot_3d:
        from utils import response_surface_3d
        fig = response_surface_3d(grid_df, selected_target)
    else:
        fig = response_surface_heatmap(grid_df, selected_target)

    if not plot_3d:
        fig.add_trace(go.Scatter(
            x=[X1], y=[X2],
            mode="markers",
            marker=dict(color="white", size=14, symbol="cross",
                        line=dict(color="#0D1B2A", width=2.5)),
            name="Point actuel",
            hovertemplate=f"T={X1}°C, W/F={X2:.2f}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Afficher les 4 sorties en même temps", value=False):
        cols = st.columns(2)
        for i, t in enumerate(["Y1", "Y2", "Y3", "Y4"]):
            with cols[i % 2]:
                f = response_surface_heatmap(grid_df, t)
                f.update_layout(paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                                font=dict(family="Inter, sans-serif", size=11))
                st.plotly_chart(f, use_container_width=True)


# ── Onglet 2 : Analyse de sensibilité ────────────────────────────────────────

with tab_sensitivity:
    st.markdown('<div class="section-title">Analyse de sensibilité</div>', unsafe_allow_html=True)
    st.caption(
        "Quelle variable (Température ou Temps de contact) a le plus d'influence sur chaque résultat ? "
        "Les barres montrent l'importance relative de chaque entrée."
    )

    fig_sens = sensitivity_bar(config)
    fig_sens.update_layout(paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                            font=dict(family="Inter, sans-serif", size=12))
    st.plotly_chart(fig_sens, use_container_width=True)

    if st.button("Calculer les indices de Sobol (environ 10 secondes)", type="secondary"):
        with st.spinner("Calcul en cours — Monte-Carlo ..."):
            fig_sobol = sobol_bar(config, n_samples=2_000)
        fig_sobol.update_layout(paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                                 font=dict(family="Inter, sans-serif", size=12))
        st.plotly_chart(fig_sobol, use_container_width=True)


# ── Onglet 3 : Optimisation ───────────────────────────────────────────────────

with tab_optimize:
    st.markdown('<div class="section-title">Optimisation des conditions opératoires</div>', unsafe_allow_html=True)
    st.caption(
        "Définissez vos contraintes et l'importance de chaque objectif. "
        "L'algorithme cherche automatiquement les meilleures conditions."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Contraintes minimales/maximales**")
        min_Y1 = st.number_input(
            "Conversion CH₄ minimale (%)",
            min_value=50.0, max_value=100.0, value=90.0, step=1.0,
            help="Le modèle cherchera uniquement les conditions qui dépassent ce seuil."
        )
        min_Y2 = st.number_input(
            "Rendement H₂ minimal (%)",
            min_value=0.0, max_value=40.0, value=10.0, step=0.5,
        )
        max_Y4 = st.number_input(
            "Dépôt de carbone maximal (%)",
            min_value=0.0, max_value=100.0, value=40.0, step=1.0,
            help="Au-dessus de ce seuil, la solution est rejetée."
        )

    with col_r:
        st.markdown("**Importance de chaque objectif**")
        w_Y1 = st.slider("Importance — Conversion CH₄", 0.0, 3.0, 1.0, step=0.1)
        w_Y2 = st.slider("Importance — Rendement H₂",   0.0, 3.0, 1.0, step=0.1)
        w_Y3 = st.slider("Importance — Sélectivité CO₂", 0.0, 3.0, 0.5, step=0.1)
        w_Y4 = st.slider("Pénalité — Dépôt de carbone (négatif = minimiser)", -3.0, 0.0, -2.0, step=0.1)

    run_opt = st.button("Lancer l'optimisation", type="primary")

    if run_opt:
        with st.spinner("Recherche des meilleures conditions en cours ..."):
            from optimize import run_single_objective, find_optimal
            opt_result = run_single_objective(
                weights={"Y1": w_Y1, "Y2": w_Y2, "Y3": w_Y3, "Y4": w_Y4}
            )
            constrained = find_optimal(min_Y1=min_Y1, min_Y2=min_Y2, max_Y4=max_Y4)

        st.markdown("#### Meilleure solution globale (sans contraintes)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Température optimale", f"{opt_result['X1']:.1f} °C")
            st.metric("Conversion CH₄", f"{opt_result['Y1']:.2f} %")
            st.metric("Sélectivité CO₂", f"{opt_result['Y3']:.2f} %")
        with col_b:
            st.metric("Temps de contact optimal", f"{opt_result['X2']:.3f} g·h/mol")
            st.metric("Rendement H₂", f"{opt_result['Y2']:.2f} %")
            st.metric("Dépôt de carbone", f"{opt_result['Y4']:.2f} %")

        st.divider()

        if constrained:
            st.markdown("#### Meilleure solution avec vos contraintes")
            st.success(
                f"Température = **{constrained['X1']:.1f} °C**  |  "
                f"Temps de contact = **{constrained['X2']:.3f} g·h/mol**  |  "
                f"Conversion = **{constrained['Y1']:.2f}%**  |  "
                f"Rendement H₂ = **{constrained['Y2']:.2f}%**  |  "
                f"Carbone = **{constrained['Y4']:.2f}%**"
            )
        else:
            st.warning(
                "Aucune solution trouvée avec ces contraintes. "
                "Essayez d'assouplir les limites (baisser la conversion minimale ou augmenter le dépôt maximal)."
            )

    st.divider()
    st.markdown("#### Front de Pareto : Conversion vs Dépôt de carbone")
    st.caption(
        "Le front de Pareto montre le compromis entre maximiser la conversion "
        "et minimiser le dépôt de carbone. Chaque point est une solution optimale différente."
    )

    if st.button("Calculer le front de Pareto (environ 5 secondes)", type="secondary"):
        with st.spinner("Calcul du front de Pareto ..."):
            from optimize import grid_pareto_front
            pareto_df = grid_pareto_front(n_grid=50)
        fig_par = pareto_scatter(pareto_df)
        fig_par.update_layout(paper_bgcolor="white", plot_bgcolor="#FAFAFA",
                               font=dict(family="Inter, sans-serif", size=12))
        st.plotly_chart(fig_par, use_container_width=True)

        with st.expander("Voir le tableau de données du front de Pareto"):
            st.dataframe(pareto_df.round(3), use_container_width=True)


# ── Onglet 4 : Exporter ───────────────────────────────────────────────────────

with tab_export:
    st.markdown('<div class="section-title">Exporter les résultats</div>', unsafe_allow_html=True)
    st.caption(
        "Téléchargez les prédictions du modèle sur toute la grille de conditions, "
        "ou récupérez uniquement le point de fonctionnement actuel."
    )

    n_export = st.slider("Taille de la grille d'export", 10, 100, 30, step=10)

    @st.cache_data(show_spinner="Génération de la grille ...")
    def _export_grid(n):
        return predict_grid((x1_min, x1_max), (x2_min, x2_max), n_points=n)

    export_df = _export_grid(n_export)
    export_df = export_df.rename(columns={
        "X1": "Temperature_C",
        "X2": "TempsContact_gcat_h_mol",
        "Y1": "Conversion_CH4_pct",
        "Y2": "Rendement_H2_pct",
        "Y3": "Selectivite_CO2_pct",
        "Y4": "DepotCarbone_pct",
    })

    st.dataframe(export_df.round(4), height=300, use_container_width=True)

    csv_buf = io.BytesIO()
    export_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Télécharger en CSV",
        data=csv_buf.getvalue(),
        file_name="jumeau_numerique_smr_predictions.csv",
        mime="text/csv",
    )

    st.markdown("**Conditions actuelles**")
    point_df = pd.DataFrame([{
        "Temperature_C": X1,
        "TempsContact_gcat_h_mol": X2,
        "Conversion_CH4_pct": preds["Y1"],
        "Rendement_H2_pct": preds["Y2"],
        "Selectivite_CO2_pct": preds["Y3"],
        "DepotCarbone_pct": preds["Y4"],
    }])
    st.dataframe(point_df.round(4), use_container_width=True)
