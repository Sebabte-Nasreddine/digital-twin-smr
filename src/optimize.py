"""
optimize.py — Optimisation pour le jumeau numérique SMR.

Stratégie : grille vectorisée (predict_grid) + affinement local scipy.
La grille est évaluée en un seul appel batch, donc très rapide.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


def _import_predict():
    from predict import get_config, predict, predict_grid
    return predict, predict_grid, get_config


def _get_bounds():
    _, _, get_config = _import_predict()
    cfg = get_config()
    return (
        (cfg["inputs"]["X1"]["min"], cfg["inputs"]["X1"]["max"]),
        (cfg["inputs"]["X2"]["min"], cfg["inputs"]["X2"]["max"]),
    )


# ── Optimisation mono-objectif ────────────────────────────────────────────────

def run_single_objective(weights: dict | None = None, seed: int = 42) -> dict:
    """
    Trouve les conditions optimales par pondération des objectifs.
    Maximise Y1, Y2, Y3 et minimise Y4 selon les poids donnés.
    """
    predict, predict_grid, get_config = _import_predict()
    cfg = get_config()
    bounds = _get_bounds()

    if weights is None:
        weights = {"Y1": 1.0, "Y2": 1.0, "Y3": 0.5, "Y4": -2.0}

    y_ranges = {k: (v["min"], v["max"]) for k, v in cfg["outputs"].items()}

    # Évaluation vectorisée de toute la grille en un seul appel
    df = predict_grid((bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1]), n_points=80)

    scores = np.zeros(len(df))
    for k, w in weights.items():
        lo, hi = y_ranges[k]
        p_norm = (df[k].values - lo) / max(hi - lo, 1e-9)
        scores -= w * p_norm

    # Top 5 candidats → affinement local L-BFGS-B
    top_idx = np.argsort(scores)[:5]
    top_pts = df[["X1", "X2"]].values[top_idx]

    def _f(x):
        try:
            p = predict(float(x[0]), float(x[1]))
        except Exception:
            return 1e6
        s = 0.0
        for k, w in weights.items():
            lo, hi = y_ranges[k]
            p_norm = (p[k] - lo) / max(hi - lo, 1e-9)
            s -= w * p_norm
        return s

    best_x, best_score = None, np.inf
    for pt in top_pts:
        res = minimize(_f, x0=pt, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-9})
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x

    preds = predict(float(best_x[0]), float(best_x[1]))
    return {
        "X1": round(float(best_x[0]), 2),
        "X2": round(float(best_x[1]), 4),
        **{k: round(v, 4) for k, v in preds.items()},
        "score_objectif": round(-best_score, 6),
    }


# ── Optimisation avec contraintes ─────────────────────────────────────────────

def find_optimal(
    min_Y1: float = 90.0,
    min_Y2: float = 10.0,
    max_Y4: float = 40.0,
    seed: int = 42,
) -> dict | None:
    """
    Cherche les conditions qui respectent les contraintes et maximisent
    la conversion + rendement - pénalité carbone.

    Retourne :
      - un dict avec les résultats si une solution est trouvée
      - un dict {"infeasible": True, ...} avec les infos de diagnostic sinon
    """
    predict, predict_grid, get_config = _import_predict()
    bounds = _get_bounds()

    # Évaluation vectorisée de toute la grille en un seul appel
    df = predict_grid((bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1]), n_points=80)

    mask = (df["Y1"] >= min_Y1) & (df["Y2"] >= min_Y2) & (df["Y4"] <= max_Y4)
    feasible = df[mask]

    if feasible.empty:
        # Diagnostic : trouver le minimum Y4 atteignable avec les autres contraintes
        partial_mask = (df["Y1"] >= min_Y1) & (df["Y2"] >= min_Y2)
        partial = df[partial_mask]
        if partial.empty:
            # Même Y1 et Y2 simultanément infaisables
            best_y1_row = df.loc[df["Y1"].idxmax()]
            return {
                "infeasible": True,
                "reason": "y1_y2",
                "max_y1": round(float(df["Y1"].max()), 2),
                "max_y2": round(float(df["Y2"].max()), 2),
                "best_X1": round(float(best_y1_row["X1"]), 1),
                "best_X2": round(float(best_y1_row["X2"]), 3),
            }
        # Y1 et Y2 OK mais Y4 trop élevé
        min_y4 = float(partial["Y4"].min())
        best_row = partial.loc[partial["Y4"].idxmin()]
        return {
            "infeasible": True,
            "reason": "y4",
            "min_y4_achievable": round(min_y4, 2),
            "best_X1": round(float(best_row["X1"]), 1),
            "best_X2": round(float(best_row["X2"]), 3),
            "best_Y1": round(float(best_row["Y1"]), 2),
            "best_Y2": round(float(best_row["Y2"]), 2),
            "best_Y4": round(float(best_row["Y4"]), 2),
        }

    scores = feasible["Y1"].values + feasible["Y2"].values - 0.5 * feasible["Y4"].values
    best_idx = int(np.argmax(scores))
    best_row = feasible.iloc[best_idx]
    x0 = np.array([best_row["X1"], best_row["X2"]])

    def _neg_score(x):
        try:
            p = predict(float(x[0]), float(x[1]))
        except Exception:
            return 1e6
        if p["Y1"] < min_Y1 or p["Y2"] < min_Y2 or p["Y4"] > max_Y4:
            return 1e6
        return -(p["Y1"] + p["Y2"] - 0.5 * p["Y4"])

    res = minimize(_neg_score, x0=x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300})

    try:
        p_final = predict(float(res.x[0]), float(res.x[1]))
    except Exception:
        p_final = None

    if p_final and p_final["Y1"] >= min_Y1 and p_final["Y2"] >= min_Y2 and p_final["Y4"] <= max_Y4:
        return {
            "X1": round(float(res.x[0]), 2),
            "X2": round(float(res.x[1]), 4),
            **{k: round(v, 4) for k, v in p_final.items()},
            "score": round(-res.fun, 4),
        }

    return {
        "X1": round(float(x0[0]), 2),
        "X2": round(float(x0[1]), 4),
        "Y1": round(float(best_row["Y1"]), 4),
        "Y2": round(float(best_row["Y2"]), 4),
        "Y3": round(float(best_row["Y3"]), 4),
        "Y4": round(float(best_row["Y4"]), 4),
        "score": round(float(scores[best_idx]), 4),
    }


# ── Front de Pareto ───────────────────────────────────────────────────────────

def _is_pareto_dominated(costs: np.ndarray) -> np.ndarray:
    n = len(costs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                dominated[i] = True
                break
    return ~dominated


def grid_pareto_front(n_grid: int = 50) -> pd.DataFrame:
    """Extrait le front de Pareto Y1 vs Y4 à partir d'une grille dense."""
    from predict import predict_grid, get_config
    cfg = get_config()
    df = predict_grid(
        (cfg["inputs"]["X1"]["min"], cfg["inputs"]["X1"]["max"]),
        (cfg["inputs"]["X2"]["min"], cfg["inputs"]["X2"]["max"]),
        n_points=n_grid,
    )
    costs = np.column_stack([-df["Y1"].values, df["Y4"].values])
    mask = _is_pareto_dominated(costs)
    return df[mask].sort_values("Y1").reset_index(drop=True)
