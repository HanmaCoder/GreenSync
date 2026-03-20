"""
GreenSync — Data Preprocessing Pipeline
Handles ingestion, cleaning, feature engineering, and normalisation.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV energy data."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or interpolate missing values."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Interpolate time-series gaps (linear), then fill remaining edges
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # Any remaining NaNs → column median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    logger.info("Missing values handled.")
    return df


def remove_outliers(df: pd.DataFrame, columns: list, z_thresh: float = 3.5) -> pd.DataFrame:
    """Remove rows where any column exceeds z_thresh standard deviations."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        if std == 0:
            continue
        z = (df[col] - mean) / std
        mask &= z.abs() <= z_thresh

    removed = (~mask).sum()
    if removed:
        logger.warning(f"Removed {removed} outlier rows.")
    return df[mask].reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and derived features."""
    df = df.copy()
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features
    df["demand_lag_1h"]  = df["energy_demand_kw"].shift(1)
    df["demand_lag_24h"] = df["energy_demand_kw"].shift(24)
    df["demand_roll_6h"] = df["energy_demand_kw"].rolling(6, min_periods=1).mean()

    # Effective solar (reduced by cloud cover)
    df["effective_solar"] = df["solar_output_kw"] * (1 - df["cloud_cover_pct"] / 100)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def normalize_data(df: pd.DataFrame, feature_cols: list, target_col: str, scaler_path: str = None):
    """
    Fit MinMaxScaler on features and target, return scaled arrays and scaler.
    Optionally save the fitted scaler to disk.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(df[feature_cols])
    y = scaler_y.fit_transform(df[[target_col]])

    if scaler_path:
        scalers = {"X": scaler_X, "y": scaler_y}
        with open(scaler_path, "wb") as f:
            pickle.dump(scalers, f)
        logger.info(f"Scalers saved to {scaler_path}")

    return X, y, scaler_X, scaler_y


def load_scalers(scaler_path: str):
    """Load saved scalers from disk."""
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)
    return scalers["X"], scalers["y"]


def run_pipeline(raw_path: str, processed_path: str) -> pd.DataFrame:
    """End-to-end preprocessing pipeline."""
    df = load_raw_data(raw_path)
    df = handle_missing_values(df)

    numeric_cols = [
        "energy_demand_kw", "temperature_c", "cloud_cover_pct",
        "wind_speed_ms", "humidity_pct", "solar_output_kw", "battery_level_pct",
    ]
    df = remove_outliers(df, numeric_cols)
    df = engineer_features(df)

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info(f"Clean data saved to {processed_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import RAW_DATA_PATH, PROCESSED_DATA_PATH

    df = run_pipeline(str(RAW_DATA_PATH), str(PROCESSED_DATA_PATH))
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
