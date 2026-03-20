"""
GreenSync — API Routes
"""
import logging
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    MODEL_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH,
    FEATURE_COLUMNS, TARGET_COLUMN, FORECAST_HOURS,
    BATTERY_CAPACITY_KWH, DEFAULT_LAT, DEFAULT_LON, DEFAULT_CITY, WEATHER_API_KEY,
)
from services.weather_service import WeatherService
from services.solar_model import SolarModel
from services.battery_model import BatteryModel
from services.optimization import OptimizationEngine
from utils.helpers import classify_demand, generate_forecast_timestamps, simulate_current_state
from utils.preprocessing import run_pipeline, engineer_features, load_raw_data

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Singleton services ────────────────────────────────────────────────────────
_weather_svc = WeatherService(WEATHER_API_KEY, DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON)
_solar_model  = SolarModel(latitude=DEFAULT_LAT)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ForecastPoint(BaseModel):
    timestamp: str
    predicted_demand_kw: float
    status: str

class OptimizationPoint(BaseModel):
    timestamp: str
    demand_kw: float
    solar_kw: float
    battery_kw: float
    grid_kw: float
    source_priority: str
    battery_soc_pct: float
    estimated_cost_inr: float
    status: str
    is_peak: bool
    notes: List[str]

class ForecastResponse(BaseModel):
    generated_at: str
    forecast: List[ForecastPoint]
    optimization: List[OptimizationPoint]
    battery_state: Dict
    metrics: Optional[Dict] = None

class CurrentStateResponse(BaseModel):
    timestamp: str
    battery_level_pct: float
    solar_output_kw: float
    current_demand_kw: float
    grid_connected: bool


# ── Helper: load model bundle ─────────────────────────────────────────────────
def _load_bundle():
    if not Path(MODEL_PATH).exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. POST /train first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _make_feature_row(weather: Dict, hour: int, day_of_week: int,
                      solar_kw: float, last_demand: float = 185.0) -> np.ndarray:
    """Build a single feature vector from weather + time context."""
    import math
    is_weekend = int(day_of_week >= 5)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    day_sin  = math.sin(2 * math.pi * day_of_week / 7)
    day_cos  = math.cos(2 * math.pi * day_of_week / 7)
    eff_solar = solar_kw * (1 - weather.get("cloud_cover_pct", 0) / 100)

    row = {
        "temperature_c":   weather.get("temperature_c", 15.0),
        "cloud_cover_pct": weather.get("cloud_cover_pct", 30.0),
        "wind_speed_ms":   weather.get("wind_speed_ms", 4.0),
        "humidity_pct":    weather.get("humidity_pct", 60.0),
        "hour_of_day":     hour,
        "day_of_week":     day_of_week,
        "is_weekend":      is_weekend,
        "solar_output_kw": solar_kw,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "day_sin":  day_sin,  "day_cos":  day_cos,
        "demand_lag_1h":   last_demand,
        "demand_lag_24h":  last_demand * 0.98,
        "demand_roll_6h":  last_demand * 1.01,
        "effective_solar": eff_solar,
        "month": datetime.utcnow().month,
    }
    return row


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "service": "GreenSync", "time": datetime.utcnow().isoformat()}


@router.post("/train")
def train_model():
    """Trigger model (re)training."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.train_model import train
        metrics = train()
        return {"status": "trained", "metrics": metrics}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    battery_soc: float = Query(75.0, ge=0, le=100, description="Current battery SoC %"),
    cost_mode: bool = Query(False, description="Enable cost-optimization mode"),
):
    """Return 6-hour demand forecast with optimization recommendations."""
    bundle = _load_bundle()
    models     = bundle["models"]
    scaler_X   = bundle["scaler_X"]
    feat_cols  = bundle["feature_columns"]

    # Fetch weather & solar forecast
    weather_forecast = _weather_svc.get_forecast(FORECAST_HOURS)
    solar_forecast   = _solar_model.forecast(weather_forecast)

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    forecast_points  = []
    opt_results      = []

    battery = BatteryModel(capacity_kwh=BATTERY_CAPACITY_KWH, initial_soc_pct=battery_soc)
    opt_engine = OptimizationEngine(battery, cost_mode=cost_mode)

    last_demand = 185.0
    for i, (w, s) in enumerate(zip(weather_forecast, solar_forecast)):
        ts = now + timedelta(hours=i + 1)
        row = _make_feature_row(w, ts.hour, ts.weekday(), s["solar_output_kw"], last_demand)

        feat_vec = np.array([[row.get(c, 0.0) for c in feat_cols]])
        feat_scaled = scaler_X.transform(feat_vec)

        # Predict h+1 only (use models[0] for next-hour, then chain)
        pred_kw = float(models[i].predict(feat_scaled)[0]) if i < len(models) else last_demand
        pred_kw = max(50, pred_kw)   # floor
        last_demand = pred_kw

        status = classify_demand(pred_kw)
        ts_str = ts.isoformat() + "Z"

        forecast_points.append(ForecastPoint(
            timestamp=ts_str,
            predicted_demand_kw=round(pred_kw, 2),
            status=status,
        ))

        # Optimization
        rec = opt_engine.recommend(ts.hour, pred_kw, s["solar_output_kw"])
        rec["timestamp"] = ts_str
        opt_results.append(OptimizationPoint(**{k: rec[k] for k in OptimizationPoint.__fields__}))

    return ForecastResponse(
        generated_at=datetime.utcnow().isoformat() + "Z",
        forecast=forecast_points,
        optimization=opt_results,
        battery_state=battery.get_state(),
        metrics=bundle.get("metrics"),
    )


@router.get("/current", response_model=CurrentStateResponse)
def get_current_state():
    """Return simulated current microgrid state."""
    state = simulate_current_state()
    return CurrentStateResponse(**state)


@router.get("/weather")
def get_weather():
    """Return raw weather forecast."""
    data = _weather_svc.get_forecast(FORECAST_HOURS)
    return {"forecast": data}


@router.get("/solar")
def get_solar_forecast():
    """Return solar output forecast."""
    weather = _weather_svc.get_forecast(FORECAST_HOURS)
    solar   = _solar_model.forecast(weather)
    return {"forecast": solar}