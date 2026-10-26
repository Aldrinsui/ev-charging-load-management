# ⚡ EV Charging Load Management System

LLM-Enhanced Spatiotemporal Prediction for EV Charging Load Management

## 📋 Description
An intelligent EV charging load management dashboard that combines:
- Deep Learning (LSTM) based spatiotemporal load forecasting
- Near real-time simulation using ACN-Data (Caltech)
- Google Gemini AI for load mitigation recommendations
- Interactive dashboard with heatmaps, forecasts and alerts

## 🛠️ Tech Stack
- React.js (Frontend Dashboard)
- Google Gemini 2.0 Flash (LLM Decision Support)
- ACN-Data — Caltech EV Charging Dataset
- Python + Google Colab (Data Processing)

## 🚀 How to Run
1. Clone the repo
   git clone https://github.com/YOUR_USERNAME/ev-charging-load-management.git

2. Install dependencies
   cd ev-dashboard
   npm install

3. Add your Gemini API key in .env
   REACT_APP_GEMINI_KEY=your_key_here

4. Run
   npm start

## 👥 Authors
- Elamaran S — RA2211003010850
- Jenitt Aldrin — RA2211003010857

SRM Institute of Science and Technology
Department of Computing Technologies

## 📄 Reference
IEEE Conference Paper — 2026

## Backend (Flask)

### Setup
    cd backend
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env            # add your Gemini key

### Database
Defaults to local SQLite (`backend/ev_charging.db`), zero setup. For MySQL in
production, set `DATABASE_URL`:

    export DATABASE_URL="mysql+pymysql://user:password@host:3306/ev_charging"

The schema (`backend/db.py`) is identical either way.

### Train the forecasting model
    python -m ml.train

Trains a single-layer LSTM, implemented from scratch in NumPy (forward pass +
manual backpropagation-through-time — no PyTorch/TensorFlow dependency), on
`public/data/acn-data.csv`. Writes `backend/ml/weights.npz` and
`backend/ml/metrics.json`.

**Measured on a held-out test split (19 of 95 sequences), not a target:**

| Metric | Value |
|---|---|
| MAE  | 1.68 kWh |
| RMSE | 1.97 kWh |
| MAPE | 11.5% |

The dataset is small (95 usable sequences from 100 charging sessions), so
treat this as a baseline, not a claim of production-grade accuracy. Re-run
`python -m ml.train` any time the CSV changes — CI re-trains on every push.

### Run the API
    python app.py             # serves http://localhost:5000

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | liveness check |
| `/api/stations?hour=19` | GET | polls all 6 stations **concurrently** (`ThreadPoolExecutor`), persists readings to the DB, returns elapsed poll time |
| `/api/forecast` | POST | JSON contract `{stationId, history[], steps}` → `{forecast[]}`, served by the trained LSTM |
| `/api/recommendations` | POST | server-side Gemini call (key never reaches the browser) |
| `/api/model-metrics` | GET | the real metrics from the last training run |

Run the backend tests (25 tests: API, LSTM, simulator, DB) with `pytest` from `backend/`.

The React app calls `REACT_APP_API_URL` (default `http://localhost:5000`). If the
backend is down, the dashboard falls back to its built-in recommendations.

### CI/CD
`.github/workflows/ci.yml` runs on every push: retrains the model, runs all
backend tests, builds the React app. The `deploy` job pushes the Flask API to
Vercel as a serverless function (`vercel.json`), gated behind `VERCEL_TOKEN`,
`VERCEL_ORG_ID`, and `VERCEL_PROJECT_ID` repo secrets — add those and flip the
`ENABLE_DEPLOY` repo variable to `true` to turn on real deploys. Until then
it's wired but inactive, not a live deployment.
