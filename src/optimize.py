"""
optimize.py — Optimisation pour le jumeau numérique SMR.

Stratégie : grille dense (80×80) + affinement local scipy.
Beaucoup plus rapide que differential_evolution pour seulement 2 variables.
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


# ── Recherche sur grille + affinement local ───────────────────────────────────

def _grid_then_polish(score_fn, bounds, n_grid=80, n_best=5, seed=42):
    """
    Évalue score_fn sur une grille n_grid×n_grid, puis affine les n_best
    meilleurs points avec L-BFGS-B. Retourne (x_opt, score_opt).
    """
    x1_lo, x1_hi = bounds[0]
    x2_lo, x2_hi = bounds[1]

    X1v = np.linspace(x1_lo, x1_hi, n_grid)
    X2v = np.linspace(x2_lo, x2_hi, n_grid)
    X1g, X2g = np.meshgrid(X1v, X2v)
    grid_pts = np.column_stack([X1g.ravel(), X2g.ravel()])

    scores = np.array([score_fn(p) for p in grid_pts])

    # Meilleurs points sur la grille → affinement local
    top_idx = np.argsort(scores)[:n_best]
    best_x, best_score = None, np.inf

    for idx in top_idx:
        res = minimize(
            score_fn,
            x0=grid_pts[idx],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x

    return best_x, best_score


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

    def _f(x):
        try:
            p = predict(float(x[0]), float(x[1]))
        except Exception:
            return 1e6
        score = 0.0
        for k, w in weights.items():
            lo, hi = y_ranges[k]
            p_norm = (p[k] - lo) / max(hi - lo, 1e-9)
            score -= w * p_norm
        return score

    best_x, best_score = _grid_then_polish(_f, bounds)
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
    """
    predict, predict_grid, get_config = _import_predict()
    cfg = get_config()
    bounds = _get_bounds()

    # Grille dense
    x1_lo, x1_hi = bounds[0]
    x2_lo, x2_hi = bounds[1]
    X1v = np.linspace(x1_lo, x1_hi, 80)
    X2v = np.linspace(x2_lo, x2_hi, 80)
    X1g, X2g = np.meshgrid(X1v, X2v)
    pts = np.column_stack([X1g.ravel(), X2g.ravel()])

    best_score = -np.inf
    best = None

    for pt in pts:
        try:
            p = predict(float(pt[0]), float(pt[1]))
        except Exception:
            continue
        if p["Y1"] < min_Y1 or p["Y2"] < min_Y2 or p["Y4"] > max_Y4:
            continue
        score = p["Y1"] + p["Y2"] - 0.5 * p["Y4"]
        if score > best_score:
            best_score = score
            best = {"x": pt.copy(), "preds": p}

    if best is None:
        return None

    # Affinage local
    x0 = best["x"]

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
        p_final = best["preds"]

    if (p_final["Y1"] >= min_Y1 and p_final["Y2"] >= min_Y2
            and p_final["Y4"] <= max_Y4):
        return {
            "X1": round(float(res.x[0]), 2),
            "X2": round(float(res.x[1]), 4),
            **{k: round(v, 4) for k, v in p_final.items()},
            "score": round(-res.fun, 4),
        }

    # Retourner le meilleur point de la grille si l'affinement sort des contraintes
    return {
        "X1": round(float(best["x"][0]), 2),
        "X2": round(float(best["x"][1]), 4),
        **{k: round(v, 4) for k, v in best["preds"].items()},
        "score": round(best_score, 4),
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
