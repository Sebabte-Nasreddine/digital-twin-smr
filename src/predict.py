"""
predict.py — Prediction functions for the SMR Digital Twin cascade model.

Load once, predict many. The cascade order is:
  X1, X2 → Y1 → Y2 → Y3, Y4
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent.parent / "models"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple:
    """Load models, scaler, and config from disk (cached after first call)."""
    models = {}
    for target in ["Y1", "Y2", "Y3", "Y4"]:
        models[target] = joblib.load(MODELS_DIR / f"model_{target}.pkl")

    scaler = joblib.load(MODELS_DIR / "scaler_X.pkl")

    with open(MODELS_DIR / "model_config.json") as fh:
        config = json.load(fh)

    return models, scaler, config


def predict(X1: float, X2: float) -> dict:
    """
    Run the full cascade prediction for a single operating point.

    Args:
        X1: Temperature (°C)
        X2: Contact time W/F°CH4 (g_cat·h/mol)

    Returns:
        dict with keys Y1, Y2, Y3, Y4 (all in %)
    """
    models, scaler, config = load_artifacts()

    # Validate against training ranges (1% tolerance for rounding on boundaries)
    for name, val in [("X1", X1), ("X2", X2)]:
        bounds = config["inputs"][name]
        lo, hi = bounds["min"], bounds["max"]
        tol = (hi - lo) * 0.01
        if not (lo - tol <= val <= hi + tol):
            raise ValueError(
                f"{name}={val} is outside training range [{lo}, {hi}]"
            )

    X_raw = np.array([[X1, X2]])
    X_scaled = scaler.transform(X_raw)

    # Cascade prediction
    Y1 = float(models["Y1"].predict(X_scaled)[0])
    X_Y2 = np.column_stack([X_scaled, [[Y1]]])
    Y2 = float(models["Y2"].predict(X_Y2)[0])
    X_Y3Y4 = np.column_stack([X_scaled, [[Y2]]])
    Y3 = float(models["Y3"].predict(X_Y3Y4)[0])
    Y4 = float(models["Y4"].predict(X_Y3Y4)[0])

    # Clip to physical bounds
    Y1 = np.clip(Y1, 0.0, 100.0)
    Y2 = np.clip(Y2, 0.0, 100.0)
    Y3 = np.clip(Y3, 0.0, 100.0)
    Y4 = np.clip(Y4, 0.0, 100.0)

    return {"Y1": round(Y1, 4), "Y2": round(Y2, 4), "Y3": round(Y3, 4), "Y4": round(Y4, 4)}


def predict_with_uncertainty(X1: float, X2: float) -> dict:
    """
    Cascade prediction with per-tree uncertainty estimate from the Random Forest.

    Returns mean prediction and ±1σ interval for each output.
    """
    models, scaler, _ = load_artifacts()

    X_raw = np.array([[X1, X2]])
    X_scaled = scaler.transform(X_raw)

    def _rf_stats(model, X_in):
        tree_preds = np.array([t.predict(X_in)[0] for t in model.estimators_])
        return float(tree_preds.mean()), float(tree_preds.std())

    mean_Y1, std_Y1 = _rf_stats(models["Y1"], X_scaled)
    mean_Y1 = np.clip(mean_Y1, 0, 100)

    X_Y2 = np.column_stack([X_scaled, [[mean_Y1]]])
    mean_Y2, std_Y2 = _rf_stats(models["Y2"], X_Y2)
    mean_Y2 = np.clip(mean_Y2, 0, 100)

    X_Y3Y4 = np.column_stack([X_scaled, [[mean_Y2]]])
    mean_Y3, std_Y3 = _rf_stats(models["Y3"], X_Y3Y4)
    mean_Y4, std_Y4 = _rf_stats(models["Y4"], X_Y3Y4)
    mean_Y3 = np.clip(mean_Y3, 0, 100)
    mean_Y4 = np.clip(mean_Y4, 0, 100)

    return {
        "Y1": {"mean": round(mean_Y1, 4), "std": round(std_Y1, 4)},
        "Y2": {"mean": round(mean_Y2, 4), "std": round(std_Y2, 4)},
        "Y3": {"mean": round(mean_Y3, 4), "std": round(std_Y3, 4)},
        "Y4": {"mean": round(mean_Y4, 4), "std": round(std_Y4, 4)},
    }


def predict_batch(df: pd.DataFrame, X1_col: str = "X1", X2_col: str = "X2") -> pd.DataFrame:
    """
    Batch prediction over a DataFrame.

    Returns a copy of df with Y1–Y4 columns appended.
    """
    models, scaler, _ = load_artifacts()

    X_raw = df[[X1_col, X2_col]].values
    X_scaled = scaler.transform(X_raw)

    Y1 = models["Y1"].predict(X_scaled)
    X_Y2 = np.column_stack([X_scaled, Y1])
    Y2 = models["Y2"].predict(X_Y2)
    X_Y3Y4 = np.column_stack([X_scaled, Y2])
    Y3 = models["Y3"].predict(X_Y3Y4)
    Y4 = models["Y4"].predict(X_Y3Y4)

    result = df.copy()
    result["Y1_pred"] = np.clip(Y1, 0, 100)
    result["Y2_pred"] = np.clip(Y2, 0, 100)
    result["Y3_pred"] = np.clip(Y3, 0, 100)
    result["Y4_pred"] = np.clip(Y4, 0, 100)
    return result


def predict_grid(
    X1_range: tuple[float, float],
    X2_range: tuple[float, float],
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Predict on a 2-D grid (for response surface plots).

    Returns a tidy DataFrame with columns X1, X2, Y1, Y2, Y3, Y4.
    """
    x1_vals = np.linspace(*X1_range, n_points)
    x2_vals = np.linspace(*X2_range, n_points)
    X1_grid, X2_grid = np.meshgrid(x1_vals, x2_vals)

    grid_df = pd.DataFrame({
        "X1": X1_grid.ravel(),
        "X2": X2_grid.ravel(),
    })
    result = predict_batch(grid_df)
    result = result.rename(columns={
        "Y1_pred": "Y1", "Y2_pred": "Y2", "Y3_pred": "Y3", "Y4_pred": "Y4",
    })
    return result


def get_config() -> dict:
    """Return the model configuration metadata."""
    _, _, config = load_artifacts()
    return config
