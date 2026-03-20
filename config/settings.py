"""
GreenSync Configuration Settings
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Data paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "energy_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "clean_data.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_model.pkl"

# Weather API Configuration
# Using Open-Meteo (https://open-meteo.com) — FREE, no API key required.
# Just set your city coordinates below (or via environment variables).
WEATHER_API_KEY = ""                                    # not needed for Open-Meteo
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Dadri")
DEFAULT_LAT  = float(os.getenv("DEFAULT_LAT", "28.5533"))   # Dadri, Uttar Pradesh, India
DEFAULT_LON  = float(os.getenv("DEFAULT_LON", "77.5550"))

# Model Configuration
FORECAST_HOURS = 6
SEQUENCE_LENGTH = 24          # hours of history to use for prediction
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42
MAPE_THRESHOLD = 15.0         # % — target max MAPE

# Feature columns used for ML
FEATURE_COLUMNS = [
    "temperature_c",
    "cloud_cover_pct",
    "wind_speed_ms",
    "humidity_pct",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "solar_output_kw",
]
TARGET_COLUMN = "energy_demand_kw"

# Optimization Engine
BATTERY_CAPACITY_KWH = 200.0       # total battery capacity
BATTERY_MIN_LEVEL = 20.0           # % — don't drain below this
BATTERY_MAX_LEVEL = 95.0           # % — don't charge above this
SOLAR_PANEL_CAPACITY_KW = 150.0    # installed solar capacity
GRID_COST_PER_KWH = 8.00           # ₹/kWh baseline (off-peak)
PEAK_GRID_COST_PER_KWH = 15.00     # ₹/kWh during peak hours

# Peak hours (24-hour format)
PEAK_HOURS = list(range(17, 21))   # 17:00–20:59

# Alert thresholds
WARNING_THRESHOLD_KW = 200.0       # demand above this → warning
CRITICAL_THRESHOLD_KW = 240.0      # demand above this → critical

# API Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"

# Dashboard
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
DASHBOARD_THEME = "dark"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"