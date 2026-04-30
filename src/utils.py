"""
utils.py — Shared plotting and helper utilities for the SMR Digital Twin.

All charts use Plotly for browser-native interactivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color scheme ──────────────────────────────────────────────────────────────

OUTPUT_COLORS = {
    "Y1": "#1565C0",  # deep blue   — CH4 conversion
    "Y2": "#2E7D32",  # deep green  — H2 yield
    "Y3": "#E65100",  # deep orange — CO2 selectivity
    "Y4": "#C62828",  # deep red    — carbon deposition
}

OUTPUT_LABELS = {
    "Y1": "CH₄ Conversion (%)",
    "Y2": "H₂ Yield (%)",
    "Y3": "CO₂ Selectivity (%)",
    "Y4": "Carbon Deposition (%)",
}


def carbon_deposition_color(y4_value: float) -> str:
    """Return a CSS colour string based on carbon deposition level (%)."""
    if y4_value < 10.0:
        return "#4CAF50"   # green  — low risk
    elif y4_value < 50.0:
        return "#FF9800"   # orange — moderate risk
    return "#F44336"       # red    — high risk


# ── KPI card ─────────────────────────────────────────────────────────────────

def kpi_figure(value: float, label: str, unit: str = "%", color: str = "#2196F3") -> go.Figure:
    """Single-value indicator figure for Streamlit KPI cards."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"size": 32}},
        title={"text": label, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "lightgray",
            "steps": [{"range": [0, 100], "color": "#f0f2f6"}],
        },
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=200)
    return fig


# ── Response surface ──────────────────────────────────────────────────────────

def response_surface_heatmap(
    grid_df: pd.DataFrame,
    target: str,
    x1_col: str = "X1",
    x2_col: str = "X2",
    x1_label: str = "Temperature (°C)",
    x2_label: str = "Contact Time W/F (g_cat·h/mol)",
) -> go.Figure:
    """
    2-D heatmap of a single output variable over the (X1, X2) input space.

    Args:
        grid_df: DataFrame from predict_grid() with columns X1, X2, Y1–Y4.
        target: Which output to plot ('Y1', 'Y2', 'Y3', or 'Y4').
    """
    x1_unique = np.sort(grid_df[x1_col].unique())
    x2_unique = np.sort(grid_df[x2_col].unique())

    pivot = grid_df.pivot_table(index=x2_col, columns=x1_col, values=target, aggfunc="mean")

    colorscale = "RdYlGn" if target != "Y4" else "RdYlGn_r"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
        colorbar={"title": OUTPUT_LABELS.get(target, target)},
        hovertemplate=(
            f"{x1_label}: %{{x:.1f}}<br>"
            f"{x2_label}: %{{y:.2f}}<br>"
            f"{OUTPUT_LABELS.get(target, target)}: %{{z:.2f}}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"Response Surface — {OUTPUT_LABELS.get(target, target)}",
        xaxis_title=x1_label,
        yaxis_title=x2_label,
        height=420,
        margin=dict(l=60, r=40, t=50, b=60),
    )
    return fig


def response_surface_3d(
    grid_df: pd.DataFrame,
    target: str,
    x1_col: str = "X1",
    x2_col: str = "X2",
) -> go.Figure:
    """3-D surface plot of an output over (X1, X2)."""
    pivot = grid_df.pivot_table(index=x2_col, columns=x1_col, values=target, aggfunc="mean")
    colorscale = "Viridis" if target != "Y4" else "Magma"
    fig = go.Figure(go.Surface(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
        colorbar={"title": OUTPUT_LABELS.get(target, target)},
    ))
    fig.update_layout(
        title=f"3-D Surface — {OUTPUT_LABELS.get(target, target)}",
        scene=dict(
            xaxis_title="Temperature (°C)",
            yaxis_title="Contact Time W/F",
            zaxis_title=OUTPUT_LABELS.get(target, target),
        ),
        height=500,
    )
    return fig


# ── Sensitivity / feature importance ─────────────────────────────────────────

def sensitivity_bar(config: dict) -> go.Figure:
    """
    Bar chart of Random Forest feature importances for Y1
    (the only model whose inputs are raw experimental variables).
    """
    fi = config.get("feature_importance_Y1", {})
    if not fi:
        return go.Figure()

    labels = {"X1": "Temperature (X1)", "X2": "Contact Time (X2)"}
    names = [labels.get(k, k) for k in fi]
    values = list(fi.values())

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=["#1565C0", "#E65100"],
        marker_line_color=["#0D47A1", "#BF360C"],
        marker_line_width=1.5,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance — CH₄ Conversion (Y1)",
        yaxis_title="Importance",
        yaxis_range=[0, max(values) * 1.35],
        height=350,
        margin=dict(t=50, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
    )
    return fig


def sobol_bar(
    config: dict,
    n_samples: int = 1_000,
    seed: int = 42,
) -> go.Figure:
    """
    Approximate first-order Sobol' sensitivity indices via Monte-Carlo variance.

    Samples are drawn from the training input ranges. Works without SALib.
    """
    from predict import predict_batch

    rng = np.random.RandomState(seed)
    x1_lo = config["inputs"]["X1"]["min"]
    x1_hi = config["inputs"]["X1"]["max"]
    x2_lo = config["inputs"]["X2"]["min"]
    x2_hi = config["inputs"]["X2"]["max"]

    X1 = rng.uniform(x1_lo, x1_hi, n_samples)
    X2 = rng.uniform(x2_lo, x2_hi, n_samples)
    X1_fixed = np.full(n_samples, (x1_lo + x1_hi) / 2)
    X2_fixed = np.full(n_samples, (x2_lo + x2_hi) / 2)

    df_base   = pd.DataFrame({"X1": X1,       "X2": X2})
    df_fix_x1 = pd.DataFrame({"X1": X1_fixed, "X2": X2})
    df_fix_x2 = pd.DataFrame({"X1": X1,       "X2": X2_fixed})

    p_base   = predict_batch(df_base)
    p_fix_x1 = predict_batch(df_fix_x1)
    p_fix_x2 = predict_batch(df_fix_x2)

    records = []
    for y in ["Y1", "Y2", "Y3", "Y4"]:
        col = f"{y}_pred"
        var_total = p_base[col].var()
        if var_total < 1e-12:
            records.append({"Output": y, "Variable": "X1", "S1": 0.0})
            records.append({"Output": y, "Variable": "X2", "S1": 0.0})
            continue
        # First-order index approximation: 1 - E[Var(Y|Xi)] / Var(Y)
        S1_x1 = 1 - (p_fix_x1[col] - p_base[col]).var() / var_total
        S1_x2 = 1 - (p_fix_x2[col] - p_base[col]).var() / var_total
        records.append({"Output": y, "Variable": "Temperature (X1)", "S1": max(S1_x1, 0)})
        records.append({"Output": y, "Variable": "Contact Time (X2)", "S1": max(S1_x2, 0)})

    df_sobol = pd.DataFrame(records)
    fig = px.bar(
        df_sobol, x="Output", y="S1", color="Variable",
        barmode="group",
        color_discrete_map={"Temperature (X1)": "#1565C0", "Contact Time (X2)": "#E65100"},
        labels={"S1": "Approx. First-Order Sobol' Index", "Output": "Output Variable"},
        title="Sensitivity Analysis — Approx. Sobol' Indices",
    )
    fig.update_layout(height=380, margin=dict(t=50, b=40),
                      paper_bgcolor="white", plot_bgcolor="#FAFAFA")
    return fig


# ── Pareto front ─────────────────────────────────────────────────────────────

def pareto_scatter(pareto_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of the Pareto front: Y1 (conversion) vs Y4 (carbon deposition).

    Points are coloured by H2 yield (Y2) to show the trade-off context.
    """
    fig = px.scatter(
        pareto_df,
        x="Y4", y="Y1",
        color="Y2",
        size="Y3",
        hover_data=["X1", "X2", "Y3"],
        color_continuous_scale="Viridis",
        labels={
            "Y4": "Carbon Deposition (%)",
            "Y1": "CH₄ Conversion (%)",
            "Y2": "H₂ Yield (%)",
            "Y3": "CO₂ Selectivity (%)",
        },
        title="Pareto Front: CH₄ Conversion vs Carbon Deposition",
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(height=460, margin=dict(t=50, b=60))
    return fig


# ── Parity plot ───────────────────────────────────────────────────────────────

def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> go.Figure:
    """Predicted vs actual scatter with perfect-prediction diagonal."""
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color="#2196F3", opacity=0.6, size=6),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="red", dash="dash"), name="Perfect",
    ))
    fig.update_layout(
        title=f"Parity Plot — {label}",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=380,
    )
    return fig
