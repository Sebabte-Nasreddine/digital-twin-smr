بِسْمِ ٱللهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ

# SMR Digital Twin

Simulation-driven Digital Twin for a **Steam Methane Reforming (SMR)** reactor,
built on surrogate Random Forest models trained from `best.ipynb`.

## Inputs / Outputs

| Variable | Description | Unit | Range |
|----------|-------------|------|-------|
| **X1** | Reactor temperature | °C | 500–800 |
| **X2** | Contact time W/F°CH₄ | g_cat·h/mol | 7.7–55.7 |
| **Y1** | CH₄ conversion | % | — |
| **Y2** | H₂ yield | % | — |
| **Y3** | CO₂ selectivity | % | — |
| **Y4** | Carbon deposition | % | minimize |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Export trained models (run ONCE)
cd digital_twin_smr
python src/extract_models.py

# 3. Launch dashboard
streamlit run app.py

# 4. (Optional) Start REST API
python api.py          # → http://localhost:8000/docs
```

## Project structure

```
digital_twin_smr/
├── notebooks/best.ipynb     ← source notebook (read-only)
├── models/
│   ├── model_Y1.pkl         ← cascade RF: [X1,X2] → Y1
│   ├── model_Y2.pkl         ← cascade RF: [X1,X2,Y1] → Y2
│   ├── model_Y3.pkl         ← cascade RF: [X1,X2,Y2] → Y3
│   ├── model_Y4.pkl         ← cascade RF: [X1,X2,Y2] → Y4
│   ├── scaler_X.pkl         ← StandardScaler for [X1,X2]
│   └── model_config.json    ← metadata, ranges, metrics
├── src/
│   ├── extract_models.py    ← Step 1: train & export
│   ├── predict.py           ← prediction functions
│   ├── optimize.py          ← multi-objective optimisation
│   └── utils.py             ← Plotly chart helpers
├── app.py                   ← Streamlit dashboard
├── api.py                   ← FastAPI REST API
└── requirements.txt
```

## Model architecture (cascade)

```
X1, X2
  │
  ├─[RF]──→ Y1 (CH₄ conversion)
  │             │
  │         [RF]──→ Y2 (H₂ yield)
  │                     │
  └─────────────[RF]──→ Y3 (CO₂ selectivity)
  └─────────────[RF]──→ Y4 (carbon deposition)
```

## API usage

```bash
# Health check
curl http://localhost:8000/health

# Point prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"X1": 700, "X2": 31.3}'

# With uncertainty
curl -X POST http://localhost:8000/predict/uncertainty \
  -H "Content-Type: application/json" \
  -d '{"X1": 700, "X2": 31.3}'
```

## Dashboard features

- **KPI cards** — live Y1–Y4 predictions with ±1σ intervals
- **Response surfaces** — 2-D heatmaps and optional 3-D surfaces
- **Sensitivity analysis** — RF feature importances + Sobol' indices
- **Optimisation** — constrained/unconstrained DE + Pareto front
- **Export** — download full grid as CSV
