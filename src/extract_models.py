"""
extract_models.py — Step 1: Train and export surrogate models for the SMR Digital Twin.

Execution order: run this script FIRST before anything else.

Pipeline:
  1. Attempt to load real data from the notebook's Excel source
  2. Fall back to physics-inspired synthetic data if not found
  3. Train cascade RandomForest models (Y1→Y2→Y3/Y4)
  4. Export models as .pkl files and save model_config.json
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Data generation / loading ─────────────────────────────────────────────────

DATA_CANDIDATES = [
    Path.home() / "Downloads" / "Base_2A.xlsx",
    PROJECT_ROOT / "data" / "Base_2A.xlsx",
    PROJECT_ROOT.parent / "Base_2A.xlsx",
]

# Contact-time grid observed in the notebook (W/F°CH4, g_cat·h/mol)
X2_LEVELS = np.array([
    7.702, 10.101, 12.500, 14.899, 17.298, 19.697, 22.096, 24.495,
    26.894, 29.293, 31.338, 34.091, 36.490, 38.889, 41.288, 43.687,
    46.086, 50.884, 55.668,
])

TEMP_LEVELS = np.array([500, 600, 700, 750, 800])  # °C


def _smr_model(T: float, x2: float, rng: np.random.RandomState) -> dict:
    """
    Physics-inspired empirical model for SMR outputs.

    Calibrated so that the RF surrogate achieves R² close to notebook values:
      - Y1 R² ≈ 0.97  (CH4 conversion)
      - Y2 R² ≈ 0.85  (H2 yield, from [X1,X2])
      - Y4 R² ≈ 0.98  (carbon deposition, from cascade)

    Observed extremes (notebook cell 10 output):
      T=500°C, X2=55.67: Y1≈77–99%, Y2≈0.15–0.66%, Y3≈3–15%, Y4≈98–100%
      T=800°C, X2=7.70:  Y1≈96–98%, Y2≈32–33%,    Y3≈10–11%, Y4≈13–15%

    Noise is kept small (~0.7–1.5%) so between-group variance dominates.
    """
    T_n = (T - 500) / 300          # 0 → 1 as T goes 500 → 800
    x2_n = (x2 - 7.702) / 47.966  # 0 → 1 as X2 goes min → max

    # Y1 — CH4 conversion (%)
    # Mean range calibrated to 77–100% (strong T effect, moderate X2 effect)
    Y1 = 77.0 + 19.0 * T_n + 3.5 * x2_n - 1.5 * T_n * x2_n
    Y1 += rng.normal(0, 0.8)
    Y1 = float(np.clip(Y1, 70.0, 100.0))

    # Y2 — H2 yield (%): very strong temperature effect (endothermic pathway)
    # Range: 0.15% at T=500 → ~33% at T=800; slightly decreases with X2
    Y2 = 0.15 + 33.0 * T_n * (1.0 - 0.07 * x2_n)
    Y2 += rng.normal(0, 1.8)
    Y2 = float(np.clip(Y2, 0.0, 40.0))

    # Y3 — CO2 selectivity (%): peaks at high X2/low T (strong WGS pathway)
    Y3 = 3.0 + 8.5 * x2_n * (1.0 - 0.45 * T_n) + 4.5 * T_n * (1.0 - x2_n) + 3.0 * T_n * x2_n
    Y3 += rng.normal(0, 0.7)
    Y3 = float(np.clip(Y3, 1.0, 20.0))

    # Y4 — Carbon deposition (%): strongly anti-correlated with Y2 (r≈-0.93)
    # Range: ~14% at high T/low X2 → ~99.6% at low T/high X2
    Y4 = 99.7 - 2.63 * (Y2 - 0.15)
    Y4 += rng.normal(0, 1.5)
    Y4 = float(np.clip(Y4, 5.0, 100.0))

    return {"X1": T, "X2": x2, "Y1": Y1, "Y2": Y2, "Y3": Y3, "Y4": Y4}


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate ~605 synthetic SMR rows matching the observed data statistics.

    5 temperatures × 19 contact-time levels × ~6.4 replicates each.
    """
    print("[generate_synthetic_data] Building physics-inspired dataset ...")
    rng = np.random.RandomState(seed)

    counts = {500: 113, 600: 113, 700: 113, 750: 113, 800: 153}
    rows = []
    for T, n in counts.items():
        x2_pool = np.tile(X2_LEVELS, n // len(X2_LEVELS) + 1)[:n]
        rng.shuffle(x2_pool)
        for x2 in x2_pool:
            rows.append(_smr_model(float(T), float(x2), rng))

    df = pd.DataFrame(rows)
    print(f"  Generated {len(df)} rows  |  columns: {list(df.columns)}")
    return df


def load_real_data() -> pd.DataFrame | None:
    """Try to load the original Excel data file from known candidate paths."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            print(f"[load_real_data] Found data at {candidate}")
            df = pd.read_excel(candidate)
            # Drop header-as-row artefact (row 0 contains column labels as strings)
            df.columns = ["X1", "X2", "Y1", "Y2", "Y3", "Y4"]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna()
            print(f"  Loaded {len(df)} rows after cleaning")
            return df
    return None


# ── Feature engineering ────────────────────────────────────────────────────────

def prepare_cascade_datasets(df: pd.DataFrame):
    """
    Build (X, y) pairs for the cascade model architecture:

      Model_Y1 : [X1, X2]       → Y1
      Model_Y2 : [X1, X2, Y1]   → Y2
      Model_Y3 : [X1, X2, Y2]   → Y3
      Model_Y4 : [X1, X2, Y2]   → Y4

    All Y columns are kept in [0, 100] percentage units throughout.
    A shared StandardScaler is fit on [X1, X2] only.
    """
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(df[["X1", "X2"]].values)
    X_base_scaled = pd.DataFrame(X_base_scaled, columns=["X1", "X2"], index=df.index)

    datasets = {
        "Y1": {
            "X": X_base_scaled[["X1", "X2"]].values,
            "y": df["Y1"].values,
            "features": ["X1", "X2"],
        },
        "Y2": {
            "X": np.column_stack([X_base_scaled[["X1", "X2"]].values, df["Y1"].values]),
            "y": df["Y2"].values,
            "features": ["X1", "X2", "Y1"],
        },
        "Y3": {
            "X": np.column_stack([X_base_scaled[["X1", "X2"]].values, df["Y2"].values]),
            "y": df["Y3"].values,
            "features": ["X1", "X2", "Y2"],
        },
        "Y4": {
            "X": np.column_stack([X_base_scaled[["X1", "X2"]].values, df["Y2"].values]),
            "y": df["Y4"].values,
            "features": ["X1", "X2", "Y2"],
        },
    }
    return datasets, scaler


# ── Training ──────────────────────────────────────────────────────────────────

RF_PARAMS = dict(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=3)


def train_single(target: str, X: np.ndarray, y: np.ndarray, features: list[str]):
    """Train a RandomForestRegressor and report metrics."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2  = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)

    print(
        f"  [{target}]  R²={r2:.4f}  MSE={mse:.6f}  MAE={mae:.4f}"
        f"  features={features}"
    )
    return model, {"R2": r2, "MSE": mse, "MAE": mae}


def train_all_models(datasets: dict) -> tuple[dict, dict]:
    """Train cascade models for all four outputs."""
    print("\n[train_all_models] Training cascade RandomForest models ...")
    models, metrics = {}, {}
    for target in ["Y1", "Y2", "Y3", "Y4"]:
        d = datasets[target]
        model, m = train_single(target, d["X"], d["y"], d["features"])
        models[target] = model
        metrics[target] = m
    return models, metrics


# ── Persistence ───────────────────────────────────────────────────────────────

def save_models(models: dict, scaler: StandardScaler, df: pd.DataFrame, metrics: dict):
    """Serialize models, scaler, and config to disk."""
    print(f"\n[save_models] Writing artefacts to {MODELS_DIR} ...")

    for target, model in models.items():
        path = MODELS_DIR / f"model_{target}.pkl"
        joblib.dump(model, path)
        print(f"  Saved {path.name}")

    scaler_path = MODELS_DIR / "scaler_X.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Saved {scaler_path.name}")

    # Feature importances (for Y1 only, since it uses only raw inputs)
    fi = dict(zip(
        ["X1", "X2"],
        models["Y1"].feature_importances_.tolist(),
    ))

    config = {
        "version": "1.0",
        "model_type": "RandomForestRegressor (cascade)",
        "data_source": "Base_2A.xlsx" if load_real_data() is not None else "synthetic_smr",
        "n_training_samples": len(df),
        "inputs": {
            "X1": {
                "label": "Temperature",
                "unit": "°C",
                "min": float(df["X1"].min()),
                "max": float(df["X1"].max()),
                "description": "Reactor temperature",
            },
            "X2": {
                "label": "Contact Time W/F°CH4",
                "unit": "g_cat·h/mol",
                "min": float(df["X2"].min()),
                "max": float(df["X2"].max()),
                "description": "Space time (weight/molar flow rate of CH4)",
            },
        },
        "outputs": {
            "Y1": {
                "label": "CH4 Conversion",
                "unit": "%",
                "min": float(df["Y1"].min()),
                "max": float(df["Y1"].max()),
                "description": "Methane conversion percentage",
                "features": ["X1", "X2"],
            },
            "Y2": {
                "label": "H2 Yield",
                "unit": "%",
                "min": float(df["Y2"].min()),
                "max": float(df["Y2"].max()),
                "description": "Hydrogen yield percentage",
                "features": ["X1", "X2", "Y1"],
            },
            "Y3": {
                "label": "CO2 Selectivity",
                "unit": "%",
                "min": float(df["Y3"].min()),
                "max": float(df["Y3"].max()),
                "description": "CO2 selectivity percentage",
                "features": ["X1", "X2", "Y2"],
            },
            "Y4": {
                "label": "Carbon Deposition",
                "unit": "%",
                "min": float(df["Y4"].min()),
                "max": float(df["Y4"].max()),
                "description": "Carbon deposition risk — minimize for stable operation",
                "features": ["X1", "X2", "Y2"],
                "color_thresholds": {"green": 10.0, "orange": 50.0, "red": 100.0},
            },
        },
        "cascade_order": ["Y1", "Y2", "Y3", "Y4"],
        "scaler": "StandardScaler on [X1, X2]",
        "feature_importance_Y1": fi,
        "metrics": metrics,
    }

    config_path = MODELS_DIR / "model_config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"  Saved {config_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SMR Digital Twin — Model Extraction")
    print("=" * 60)

    # 1. Load data
    df = load_real_data()
    if df is None:
        print("[WARNING] Real data file not found — using synthetic SMR data.")
        df = generate_synthetic_data()

    print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.describe().round(3).to_string())

    # 2. Build cascade datasets
    datasets, scaler = prepare_cascade_datasets(df)

    # 3. Train
    models, metrics = train_all_models(datasets)

    # 4. Save
    save_models(models, scaler, df, metrics)

    print("\n✓ All models exported successfully.")
    print(f"  → {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
