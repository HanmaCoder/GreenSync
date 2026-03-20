# ⚡ GreenSync — Real-Time Energy Demand Forecaster for Microgrids

> **Hackathon 2026 · Track: ML & AI · Duration: 36 Hours**

GreenSync is a machine-learning-powered platform that ingests historical electricity data and live weather feeds to predict energy demand for the next 6 hours. It helps microgrid operators make real-time switching decisions between solar, battery, and grid power — reducing waste and preventing blackouts.

---

## 🏗 Architecture Overview

```
greensync/
├── data/
│   ├── raw/energy_data.csv          # Historical energy + weather time-series
│   └── processed/clean_data.csv     # Preprocessed & feature-engineered data
├── models/
│   ├── train_model.py               # Multi-step XGBoost/GBM trainer
│   └── trained_model.pkl            # Serialised model bundle (post-training)
├── services/
│   ├── weather_service.py           # OpenWeatherMap API wrapper (+ simulation)
│   ├── solar_model.py               # Physics-based PV output estimator
│   ├── battery_model.py             # Li-ion SoC simulator & charge/discharge
│   └── optimization.py             # Dispatch optimisation engine
├── api/
│   ├── main.py                      # FastAPI app with auto-train on startup
│   └── routes.py                    # REST endpoints: /forecast /train /weather
├── dashboard/
│   └── app.py                       # Streamlit interactive dashboard
├── utils/
│   ├── preprocessing.py             # Full data pipeline: ingest → clean → features
│   └── helpers.py                   # MAPE/RMSE/MAE, classify_demand, etc.
├── config/
│   └── settings.py                  # Central configuration (env-aware)
├── requirements.txt
├── run.sh                           # One-command launcher
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) OpenWeatherMap API key — falls back to simulation without one

### 1. Clone & install
```bash
git clone https://github.com/your-team/greensync.git
cd greensync
pip install -r requirements.txt
```

### 2. Configure (optional)
```bash
cp .env.example .env
# Edit .env:
# WEATHER_API_KEY=your_key_here
# DEFAULT_CITY=London
```

### 3. Launch everything
```bash
chmod +x run.sh && ./run.sh
```

This will:
1. Install dependencies
2. Run the preprocessing pipeline
3. Train the forecasting model
4. Start the FastAPI backend (port 8000)
5. Launch the Streamlit dashboard (port 8501)

| Service   | URL                              |
|-----------|----------------------------------|
| Dashboard | http://localhost:8501            |
| API Docs  | http://localhost:8000/docs       |
| Health    | http://localhost:8000/api/v1/health |

---

## 📦 Core Components

### 1. Data Ingestion & Processing (`utils/preprocessing.py`)
- Loads CSV time-series with `parse_dates`
- Linear interpolation for missing values; median fill for residuals
- Z-score outlier removal (threshold: 3.5σ)
- Feature engineering: hour/day cyclical encodings, lag features (1h, 24h), rolling means, effective solar calculation
- Reproducible, parameterised pipeline

### 2. Predictive Model (`models/train_model.py`)
- **Algorithm**: XGBoost (falls back to Scikit-learn GradientBoostingRegressor)
- **Strategy**: Multi-output — one model per forecast horizon (h+1 … h+6)
- **Features**: temperature, cloud cover, wind speed, humidity, hour-of-day (sin/cos), day-of-week (sin/cos), lag demand, rolling mean, effective solar, month
- **Target MAPE**: < 15% on held-out test set (typical: 7–11%)
- **Persistence**: Full bundle (models + scaler + metadata) saved as `.pkl`

### 3. Optimization Engine (`services/optimization.py`)
Priority dispatch:
1. **Solar** (near-zero marginal cost)
2. **Battery** (clean stored energy)
3. **Grid** (fallback)

Cost-optimization mode:
- Off-peak: pre-charge battery from cheap grid power
- Peak hours (17:00–20:59): aggressively discharge battery to avoid peak tariffs (£0.30/kWh vs £0.15/kWh)

### 4. Dashboard (`dashboard/app.py`)
- **Tab 1 — Forecast**: 6-hour demand chart with colour-coded status bands (safe / warning / critical), solar overlay, hour-by-hour outlook panel
- **Tab 2 — Optimization**: Stacked bar showing solar/battery/grid split per hour, battery SoC trajectory, cost breakdown pie chart
- **Tab 3 — Historical**: Actual vs predicted comparison with MAPE/RMSE/MAE metrics
- **Tab 4 — Anomaly Detection**: Flags demand spikes not explained by weather (>35 kW above baseline)
- **Export**: Downloadable CSV for forecast and historical data

---

## 🌐 REST API

| Method | Endpoint                    | Description                          |
|--------|-----------------------------|--------------------------------------|
| GET    | `/api/v1/health`            | Service health check                 |
| POST   | `/api/v1/train`             | Trigger model (re)training           |
| GET    | `/api/v1/forecast`          | 6-hour demand forecast + optimization|
| GET    | `/api/v1/current`           | Current microgrid state              |
| GET    | `/api/v1/weather`           | Raw weather forecast                 |
| GET    | `/api/v1/solar`             | Solar output forecast                |

**Example:**
```bash
curl "http://localhost:8000/api/v1/forecast?battery_soc=80&cost_mode=true"
```

---

## 📊 Evaluation Criteria Mapping

| Criterion                    | Weight | Implementation                                           |
|------------------------------|--------|----------------------------------------------------------|
| Model Accuracy (MAPE/RMSE)   | 30%    | XGBoost multi-step, target MAPE < 15%                   |
| Dashboard Quality            | 25%    | Streamlit — 4 tabs, live charts, colour-coded outlook    |
| Optimization Logic           | 20%    | Rule-based dispatch + cost-opt mode with TOU tariffs     |
| Data Pipeline Robustness     | 15%    | Interpolation, outlier removal, reproducible pipeline    |
| Innovation & Bonus Features  | 10%    | Anomaly detection, cost-opt mode, REST API               |

---

## 🔮 Bonus Features Implemented

- ✅ **Anomaly Detection** — flags unexplained demand spikes vs weather baseline
- ✅ **Cost-Optimization Mode** — TOU tariff-aware battery pre-charging
- ✅ **REST API** — FastAPI with Swagger UI (`/docs`)
- 🔄 **Multi-microgrid** — architecture supports zone aggregation (config extension)

---

## 🛠 Tech Stack

| Category       | Choice                                    |
|----------------|-------------------------------------------|
| ML Framework   | XGBoost + Scikit-learn (GBM fallback)     |
| Data Processing| Pandas, NumPy                             |
| Weather API    | OpenWeatherMap (+ built-in simulation)    |
| Backend        | FastAPI + Uvicorn                         |
| Frontend / Viz | Streamlit + Plotly                        |
| Storage        | CSV / Pickle (SQLite-ready)               |
| Deployment     | Shell script / Docker-ready               |

---

## 🔑 Key Design Decisions

1. **Multi-output per horizon** — training a separate model per forecast step gives better accuracy than a single multi-target model for this dataset size.
2. **Physics solar model** — avoids needing labelled solar data; uses panel geometry + cloud cover for realistic output estimates.
3. **Graceful API fallback** — all external dependencies (weather API, trained model) have simulation/fallback paths so the system always produces output.
4. **Cyclical time encoding** — sin/cos encoding of hour and day avoids discontinuities (e.g., 23→0h) that confuse tree models.
5. **Cost-optimisation as toggle** — keeping TOU logic opt-in avoids unintended side effects on battery health when cost optimisation is not the operator's priority.

---

## 📄 License
MIT — Built for GreenSync Hackathon 2026
