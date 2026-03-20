"""
GreenSync — Helper Utilities
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%), ignoring near-zero actuals."""
    mask = np.abs(y_true) > 1.0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def classify_demand(demand_kw: float, warning_threshold: float = 200.0, critical_threshold: float = 240.0) -> str:
    """Return 'safe', 'warning', or 'critical' based on demand level."""
    if demand_kw >= critical_threshold:
        return "critical"
    if demand_kw >= warning_threshold:
        return "warning"
    return "safe"


def generate_forecast_timestamps(start: datetime, hours: int = 6) -> List[datetime]:
    """Generate list of hourly timestamps starting from 'start'."""
    return [start + timedelta(hours=i) for i in range(1, hours + 1)]


def format_forecast_response(timestamps: List[datetime], predictions: List[float],
                              statuses: List[str], recommendations: List[Dict]) -> Dict[str, Any]:
    """Package forecast data into a clean API response dict."""
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "forecast": [
            {
                "timestamp": ts.isoformat(),
                "predicted_demand_kw": round(pred, 2),
                "status": status,
                "recommendation": rec,
            }
            for ts, pred, status, rec in zip(timestamps, predictions, statuses, recommendations)
        ],
    }


def simulate_current_state(battery_pct: float = 75.0, solar_kw: float = 45.0) -> Dict[str, Any]:
    """Return a simulated current microgrid state (used when live sensors unavailable)."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "battery_level_pct": battery_pct,
        "solar_output_kw": solar_kw,
        "grid_connected": True,
        "current_demand_kw": 185.4,
    }


def df_to_json_records(df: pd.DataFrame) -> List[Dict]:
    """Convert a DataFrame to JSON-serialisable list of dicts."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))
