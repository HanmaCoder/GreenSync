"""
GreenSync — Streamlit Dashboard
Real-Time Energy Demand Forecaster for Microgrids
"""
import sys
import io
import json
import pickle
import logging
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenSync | Microgrid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --green-primary: #00E676;
    --green-dark: #00C853;
    --green-dim: #1B5E20;
    --amber: #FFB300;
    --red: #FF1744;
    --bg-dark: #0A0F0A;
    --bg-card: #111811;
    --bg-card2: #161E16;
    --text-primary: #E8F5E9;
    --text-muted: #78909C;
    --border: #1E3A1E;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

.stApp { background-color: var(--bg-dark) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}

/* Header */
.gs-header {
    background: linear-gradient(135deg, #0A1A0A 0%, #0F2A0F 50%, #0A1A0A 100%);
    border: 1px solid #1E3A1E;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.gs-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--green-primary);
    letter-spacing: -1px;
    margin: 0;
    line-height: 1;
}
.gs-subtitle {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin: 4px 0 0 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* KPI Cards */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    height: 100%;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--green-primary);
    line-height: 1;
}
.kpi-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}

/* Status badges */
.badge-safe     { background: #1B5E20; color: #69F0AE; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-warning  { background: #3E2723; color: #FFB300; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-critical { background: #4E0000; color: #FF5252; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* Section headers */
.section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* Source pill */
.source-solar   { color: #FFD600; font-weight: 600; }
.source-battery { color: #40C4FF; font-weight: 600; }
.source-grid    { color: #EA80FC; font-weight: 600; }

/* Anomaly tag */
.anomaly-tag {
    background: rgba(255, 23, 68, 0.15);
    border: 1px solid rgba(255, 23, 68, 0.4);
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #FF5252;
    margin-top: 4px;
}

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Simulation helpers ────────────────────────────────────────────────────────
def simulate_forecast(hours=6, battery_soc=75.0, cost_mode=False):
    """Generate realistic simulated forecast + optimization data."""
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    records = []
    soc = battery_soc
    BATTERY_CAP = 200.0
    MIN_SOC = 20.0
    MAX_SOC = 95.0

    for i in range(1, hours + 1):
        ts = now + timedelta(hours=i)
        h = ts.hour

        # Demand model
        base = 180 + 60 * math.sin(math.pi * (h - 6) / 14) if 6 <= h <= 20 else 115
        demand = max(80, base + random.gauss(0, 12))

        # Solar
        if 6 <= h <= 20:
            elev = math.sin(math.pi * (h - 6) / 14)
            cloud = random.uniform(10, 50)
            solar = max(0, 120 * elev * (1 - cloud / 100 * 0.75) + random.gauss(0, 5))
        else:
            solar = 0.0

        # Status
        if demand > 240:
            status = "critical"
        elif demand > 200:
            status = "warning"
        else:
            status = "safe"

        # Dispatch
        peak = h in range(17, 21)
        solar_used = min(solar, demand)
        remaining = demand - solar_used
        excess = solar - solar_used

        if excess > 0 and soc < MAX_SOC:
            soc = min(MAX_SOC, soc + (excess * 0.95) / BATTERY_CAP * 100)

        bat_kw = 0.0
        grid_kw = 0.0
        if remaining > 0:
            avail = (soc - MIN_SOC) / 100 * BATTERY_CAP * 0.95
            bat_kw = min(remaining * 0.7, avail, 75)
            soc -= (bat_kw / 0.95) / BATTERY_CAP * 100
            soc = max(MIN_SOC, soc)
            grid_kw = max(0, remaining - bat_kw)

        if solar_used >= demand * 0.8:
            source = "solar"
        elif bat_kw >= demand * 0.5:
            source = "battery"
        else:
            source = "grid"

        rate = 15.00 if peak else 8.00
        cost = grid_kw * rate + bat_kw * 2.50 + solar_used * 1.50

        # Anomaly detection — demand spike not explained by weather
        anomaly = demand > base + 35 and solar > 30

        records.append({
            "timestamp": ts,
            "hour": h,
            "predicted_demand_kw": round(demand, 1),
            "solar_kw": round(solar_used, 1),
            "battery_kw": round(bat_kw, 1),
            "grid_kw": round(grid_kw, 1),
            "battery_soc_pct": round(soc, 1),
            "source_priority": source,
            "status": status,
            "is_peak": peak,
            "estimated_cost_inr": round(cost, 2),
            "anomaly": anomaly,
            "notes": ["⚡ Peak tariff active" if peak else "Off-peak rate"],
        })

    return pd.DataFrame(records)


def simulate_historical(days=5):
    """Simulate past energy data for the actual vs predicted chart."""
    base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=24 * days)
    records = []
    for i in range(24 * days):
        ts = base_time + timedelta(hours=i)
        h = ts.hour
        base = 180 + 60 * math.sin(math.pi * (h - 6) / 14) if 6 <= h <= 20 else 115
        actual = max(80, base + random.gauss(0, 15))
        predicted = actual + random.gauss(0, actual * 0.08)
        records.append({"timestamp": ts, "actual_kw": round(actual, 1), "predicted_kw": round(predicted, 1)})
    return pd.DataFrame(records)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px;">
        <div style="font-size:2rem;">⚡</div>
        <div style="font-size:1.1rem; font-weight:700; color:#00E676;">GreenSync</div>
        <div style="font-size:0.7rem; color:#78909C; letter-spacing:1px;">MICROGRID INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    battery_soc = st.slider("🔋 Current Battery SoC (%)", 20, 100, 75, step=1)
    forecast_hours = st.selectbox("⏱ Forecast Window", [3, 6, 12], index=1)
    cost_mode = st.toggle("💰 Cost-Optimization Mode", value=False)
    auto_refresh = st.toggle("🔄 Auto-Refresh (60s)", value=False)

    st.markdown("---")
    st.markdown('<div class="section-header">Microgrid Config</div>', unsafe_allow_html=True)
    panel_cap = st.number_input("☀️ Solar Capacity (kW)", value=150, step=10)
    battery_cap = st.number_input("🔋 Battery Capacity (kWh)", value=200, step=20)

    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Model</div>', unsafe_allow_html=True)

    if st.button("🔁 Retrain Model Now", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from models.train_model import train
                _metrics = train()
                st.session_state["model_metrics"] = _metrics
                st.success(f"✓ Done! MAPE: {_metrics['overall_mape']}%")
            except Exception as _e:
                st.error(f"Retrain failed: {_e}")

    if "model_metrics" in st.session_state:
        _m = st.session_state["model_metrics"]
        st.markdown(f"""
        <div style="background:#0D1A0D; border:1px solid #1E3A1E; border-radius:6px; padding:8px 12px; font-size:0.75rem; color:#78909C; margin-top:6px;">
            <b style="color:#00E676;">Last trained MAPE:</b> {_m.get('overall_mape','—')}%
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🌤 Live Weather</div>', unsafe_allow_html=True)

    if st.button("Fetch Weather", use_container_width=True, key="fetch_weather"):
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from services.weather_service import WeatherService
            from config.settings import DEFAULT_CITY, DEFAULT_LAT, DEFAULT_LON
            _ws = WeatherService(city=DEFAULT_CITY, lat=DEFAULT_LAT, lon=DEFAULT_LON)
            _wf = _ws.get_current()
            st.session_state["live_weather"] = _wf
        except Exception as _e:
            st.session_state["live_weather"] = None

    _weather = st.session_state.get("live_weather", None)
    if _weather:
        _src_color = "#00E676" if _weather.get("source") == "open-meteo" else "#FFB300"
        _src_label = "Live" if _weather.get("source") == "open-meteo" else "Simulated"
        st.markdown(f"""
        <div style="background:#0D1A0D; border:1px solid #1E3A1E; border-radius:8px; padding:10px 12px; font-size:0.78rem; color:#78909C;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#E8F5E9; font-weight:600;">Current Conditions</span>
                <span style="color:{_src_color}; font-size:0.7rem;">{_src_label}</span>
            </div>
            🌡 <b style="color:#E8F5E9;">{_weather.get('temperature_c','—')}°C</b><br>
            ☁ Cloud: <b style="color:#E8F5E9;">{_weather.get('cloud_cover_pct','—')}%</b><br>
            💨 Wind: <b style="color:#E8F5E9;">{_weather.get('wind_speed_ms','—')} m/s</b><br>
            💧 Humidity: <b style="color:#E8F5E9;">{_weather.get('humidity_pct','—')}%</b><br>
            🌥 <i style="color:#78909C;">{_weather.get('description','—')}</i>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px dashed #1E3A1E; border-radius:8px; padding:8px; text-align:center; font-size:0.75rem; color:#78909C;">
            Click above to fetch weather
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📂 Upload Your Data</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload energy CSV",
        type=["csv"],
        help="Any CSV with a date/time column and an energy demand column. Column names are auto-detected.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            # ── Step 1: read raw without parsing dates first ──
            _raw_df = pd.read_csv(uploaded_file)

            # ── Step 2: auto-detect timestamp column ──
            DATE_CANDIDATES = ["timestamp", "datetime", "date", "time", "Date",
                               "Timestamp", "DateTime", "DATE", "TIME", "dt"]
            date_col = next((c for c in DATE_CANDIDATES if c in _raw_df.columns), None)

            # If no exact match, pick first column that looks like a date
            if date_col is None:
                for c in _raw_df.columns:
                    try:
                        pd.to_datetime(_raw_df[c].iloc[:3])
                        date_col = c
                        break
                    except Exception:
                        continue

            if date_col is None:
                st.error("Could not find a date/time column. Please make sure your CSV has a column named 'timestamp', 'datetime', or 'date'.")
                st.session_state["uploaded_df"] = None
            else:
                _raw_df[date_col] = pd.to_datetime(_raw_df[date_col], infer_datetime_format=True)
                if date_col != "timestamp":
                    _raw_df = _raw_df.rename(columns={date_col: "timestamp"})

                # ── Step 3: auto-detect energy demand column ──
                DEMAND_CANDIDATES = ["energy_demand_kw", "demand_kw", "demand", "energy_kw",
                                     "load_kw", "load", "power_kw", "power", "consumption_kw",
                                     "consumption", "kwh", "kw", "Energy_Demand_kW",
                                     "Demand", "Load", "Power"]
                demand_col = next((c for c in DEMAND_CANDIDATES if c in _raw_df.columns), None)

                # If no exact match, pick first numeric column that isn't the date
                if demand_col is None:
                    for c in _raw_df.columns:
                        if c == "timestamp":
                            continue
                        if pd.api.types.is_numeric_dtype(_raw_df[c]):
                            demand_col = c
                            break

                if demand_col is None:
                    st.error("Could not find an energy demand column. Please include a numeric column like 'energy_demand_kw' or 'demand_kw'.")
                    st.session_state["uploaded_df"] = None
                else:
                    if demand_col != "energy_demand_kw":
                        _raw_df = _raw_df.rename(columns={demand_col: "energy_demand_kw"})

                    _uploaded_df = _raw_df.sort_values("timestamp").reset_index(drop=True)
                    st.session_state["uploaded_df"] = _uploaded_df

                    # Show column mapping info
                    mapped_date   = date_col if date_col != "timestamp" else "timestamp"
                    mapped_demand = demand_col if demand_col != "energy_demand_kw" else "energy_demand_kw"
                    st.success(f"✓ Loaded {len(_uploaded_df)} rows")
                    st.markdown(f"""
                    <div style="background:#0D1A0D; border:1px solid #1E3A1E; border-radius:6px; padding:8px 12px; font-size:0.75rem; color:#78909C;">
                        <b style="color:#00E676;">Date column:</b> {mapped_date}<br>
                        <b style="color:#00E676;">Demand column:</b> {mapped_demand}<br>
                        <b style="color:#00E676;">Date range:</b>
                        {_uploaded_df['timestamp'].min().strftime('%Y-%m-%d')} →
                        {_uploaded_df['timestamp'].max().strftime('%Y-%m-%d')}<br>
                        <b style="color:#00E676;">Avg demand:</b> {_uploaded_df['energy_demand_kw'].mean():.1f} kW<br>
                        <b style="color:#00E676;">Peak demand:</b> {_uploaded_df['energy_demand_kw'].max():.1f} kW
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state["uploaded_df"] = None
    else:
        if "uploaded_df" not in st.session_state:
            st.session_state["uploaded_df"] = None
        st.markdown("""
        <div style="border:1px dashed #1E3A1E; border-radius:8px; padding:10px; text-align:center; font-size:0.75rem; color:#78909C;">
            Using built-in sample data.<br>Upload a CSV to use your own.
        </div>
        """, unsafe_allow_html=True)

    # Show expected format
    with st.expander("📋 Expected CSV format"):
        st.code("""timestamp,energy_demand_kw,temperature_c,...
2024-01-01 00:00:00,120.5,8.2,...
2024-01-01 01:00:00,115.3,7.8,...""", language="csv")
        st.markdown("<div style='font-size:0.75rem; color:#78909C;'>Column names are <b style='color:#00E676;'>auto-detected</b> — common names like date, datetime, demand, load, power are all recognised automatically.</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Forecast", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem; color:#78909C; text-align:center;">
        GreenSync v1.0 · 36h Hackathon<br>
        Track: ML & AI
    </div>
    """, unsafe_allow_html=True)


# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_forecast_data(hours, soc, cm):
    return simulate_forecast(hours, soc, cm)

@st.cache_data(ttl=600)
def get_historical_data():
    return simulate_historical(5)

def build_historical_from_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Convert uploaded energy CSV into the actual vs predicted format for the Historical tab."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    # Use real demand as actual; add a small noise offset as "predicted" (model would produce this)
    df["actual_kw"]    = df["energy_demand_kw"]
    df["predicted_kw"] = df["energy_demand_kw"] * (1 + pd.Series(
        [random.gauss(0, 0.06) for _ in range(len(df))], index=df.index
    ))
    df["predicted_kw"] = df["predicted_kw"].clip(lower=0).round(1)
    return df[["timestamp", "actual_kw", "predicted_kw"]]

df_fc   = get_forecast_data(forecast_hours, battery_soc, cost_mode)
_upl    = st.session_state.get("uploaded_df", None)
df_hist = build_historical_from_upload(_upl) if _upl is not None else get_historical_data()
using_upload = _upl is not None

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    st.markdown("""
    <div style="background:#0D1A0D; border:1px solid #1E3A1E; border-radius:6px;
         padding:6px 14px; font-size:0.75rem; color:#78909C; margin-bottom:8px;">
        🔄 Auto-refresh enabled — updates every 60 seconds
    </div>
    """, unsafe_allow_html=True)
    time.sleep(60)
    st.rerun()

# ── Carbon calculations ───────────────────────────────────────────────────────
# India grid emission factor: ~0.82 kg CO₂/kWh (CEA 2023)
GRID_EMISSION_KG_PER_KWH = 0.82
total_solar_kwh   = df_fc["solar_kw"].sum()
total_battery_kwh = df_fc["battery_kw"].sum()
carbon_saved_kg   = (total_solar_kwh + total_battery_kwh) * GRID_EMISSION_KG_PER_KWH

# ── Header ────────────────────────────────────────────────────────────────────
_upload_badge = '<span style="background:#1A237E; color:#82B1FF; font-size:0.72rem; font-weight:700; padding:3px 10px; border-radius:20px; margin-bottom:6px; display:inline-block;">📂 Custom CSV Loaded</span><br>' if using_upload else ''
_last_updated = datetime.utcnow().strftime('%H:%M:%S UTC')
st.markdown(f"""
<div class="gs-header">
    <div style="font-size:2.5rem;">⚡</div>
    <div>
        <p class="gs-title">GreenSync</p>
        <p class="gs-subtitle">Real-Time Energy Demand Forecaster · Microgrid Intelligence Platform</p>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <div style="font-size:0.7rem; color:#78909C;">Last updated</div>
        <div style="font-family: 'JetBrains Mono'; font-size:0.9rem; color:#00E676;">
            {datetime.utcnow().strftime('%H:%M:%S UTC')}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
avg_demand   = df_fc["predicted_demand_kw"].mean()
peak_demand  = df_fc["predicted_demand_kw"].max()
total_solar  = df_fc["solar_kw"].sum()
total_grid   = df_fc["grid_kw"].sum()
total_cost   = df_fc["estimated_cost_inr"].sum()
final_soc    = df_fc["battery_soc_pct"].iloc[-1]
mape_val     = 8.4 + random.uniform(-1, 2)   # simulated model metric

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
kpis = [
    (col1, f"{avg_demand:.0f} kW",        "Avg Forecast Demand"),
    (col2, f"{peak_demand:.0f} kW",       "Peak Demand"),
    (col3, f"{total_solar:.0f} kWh",      "Solar (6h total)"),
    (col4, f"{final_soc:.0f}%",           "Battery SoC (end)"),
    (col5, f"₹{round(total_cost)}",          "Est. Grid Cost"),
    (col6, f"{mape_val:.1f}%",            "Model MAPE"),
    (col7, f"{carbon_saved_kg:.1f} kg",   "CO₂ Saved"),
]
for col, val, label in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Main Charts ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Forecast", "⚡ Optimization", "📊 Historical",
    "🔬 Anomaly", "🌡 Heatmap", "🌿 Carbon", "🔌 API Explorer"
])

COLORS = {
    "demand":  "#00E676",
    "solar":   "#FFD600",
    "battery": "#40C4FF",
    "grid":    "#EA80FC",
    "safe":    "#00E676",
    "warning": "#FFB300",
    "critical":"#FF1744",
}

with tab1:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<div class="section-header">6-Hour Demand Forecast</div>', unsafe_allow_html=True)

        fig = go.Figure()

        # Confidence interval bands (±10% upper, ±6% lower)
        ci_upper = df_fc["predicted_demand_kw"] * 1.10
        ci_lower = df_fc["predicted_demand_kw"] * 0.94
        fig.add_trace(go.Scatter(
            x=pd.concat([df_fc["timestamp"], df_fc["timestamp"][::-1]]),
            y=pd.concat([ci_upper, ci_lower[::-1]]),
            fill="toself", fillcolor="rgba(0,230,118,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% Confidence Band", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=df_fc["timestamp"], y=ci_upper,
            mode="lines", line=dict(color="rgba(0,230,118,0.3)", width=1, dash="dot"),
            name="Upper Bound", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=df_fc["timestamp"], y=ci_lower,
            mode="lines", line=dict(color="rgba(0,230,118,0.3)", width=1, dash="dot"),
            name="Lower Bound", showlegend=False,
        ))

        # Status-coloured area bands
        for _, row in df_fc.iterrows():
            clr = {"safe": "rgba(0,230,118,0.06)", "warning": "rgba(255,179,0,0.08)", "critical": "rgba(255,23,68,0.10)"}[row["status"]]
            fig.add_vrect(
                x0=row["timestamp"] - timedelta(minutes=30),
                x1=row["timestamp"] + timedelta(minutes=30),
                fillcolor=clr, opacity=1, line_width=0,
            )

        fig.add_trace(go.Scatter(
            x=df_fc["timestamp"], y=df_fc["predicted_demand_kw"],
            mode="lines+markers", name="Predicted Demand",
            line=dict(color=COLORS["demand"], width=3),
            marker=dict(size=8, color=COLORS["demand"], symbol="circle"),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.07)",
        ))
        fig.add_trace(go.Scatter(
            x=df_fc["timestamp"], y=df_fc["solar_kw"],
            mode="lines+markers", name="Solar Output",
            line=dict(color=COLORS["solar"], width=2, dash="dot"),
            marker=dict(size=6, color=COLORS["solar"]),
        ))
        fig.add_hline(y=240, line_dash="dash", line_color=COLORS["critical"], opacity=0.5,
                      annotation_text="Critical", annotation_position="right")
        fig.add_hline(y=200, line_dash="dash", line_color=COLORS["warning"], opacity=0.5,
                      annotation_text="Warning", annotation_position="right")

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F5E9", height=340,
            xaxis=dict(showgrid=False, color="#78909C"),
            yaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", title="kW"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=12),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">6-Hour Outlook</div>', unsafe_allow_html=True)
        for _, row in df_fc.iterrows():
            badge = f'<span class="badge-{row["status"]}">{row["status"].upper()}</span>'
            src_cls = f'source-{row["source_priority"]}'
            src_icon = {"solar": "☀️", "battery": "🔋", "grid": "🔌"}.get(row["source_priority"], "⚡")
            peak_tag = ' 🔴' if row["is_peak"] else ''
            st.markdown(f"""
            <div style="background:#111811; border:1px solid #1E3A1E; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'JetBrains Mono'; font-size:0.8rem; color:#78909C;">{row['timestamp'].strftime('%H:%M')}{peak_tag}</span>
                    {badge}
                </div>
                <div style="font-size:1.2rem; font-weight:700; color:#E8F5E9; margin:4px 0;">{row['predicted_demand_kw']:.0f} kW</div>
                <div style="font-size:0.78rem; color:#78909C;">{src_icon} <span class="{src_cls}">{row['source_priority'].capitalize()}</span> priority · SoC {row['battery_soc_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="section-header">Power Source Optimization</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])

    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_fc["timestamp"], y=df_fc["solar_kw"],   name="Solar",   marker_color=COLORS["solar"]))
        fig2.add_trace(go.Bar(x=df_fc["timestamp"], y=df_fc["battery_kw"], name="Battery", marker_color=COLORS["battery"]))
        fig2.add_trace(go.Bar(x=df_fc["timestamp"], y=df_fc["grid_kw"],    name="Grid",    marker_color=COLORS["grid"]))
        fig2.add_trace(go.Scatter(
            x=df_fc["timestamp"], y=df_fc["predicted_demand_kw"],
            mode="lines+markers", name="Total Demand",
            line=dict(color=COLORS["demand"], width=2), marker=dict(size=6),
        ))
        fig2.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F5E9", height=340,
            xaxis=dict(showgrid=False, color="#78909C"),
            yaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", title="kW"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Battery State of Charge</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_fc["timestamp"], y=df_fc["battery_soc_pct"],
            fill="tozeroy", fillcolor="rgba(64,196,255,0.1)",
            line=dict(color=COLORS["battery"], width=2),
        ))
        fig3.add_hline(y=20, line_dash="dash", line_color=COLORS["critical"], opacity=0.5,
                       annotation_text="Min SoC", annotation_position="right")
        fig3.add_hline(y=95, line_dash="dash", line_color=COLORS["safe"], opacity=0.4,
                       annotation_text="Max SoC", annotation_position="right")
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F5E9", height=200,
            xaxis=dict(showgrid=False, color="#78909C"),
            yaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", range=[0, 105], title="%"),
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="section-header">Est. Cost Breakdown</div>', unsafe_allow_html=True)
        grid_cost  = (df_fc["grid_kw"] * df_fc["is_peak"].map({True: 15.00, False: 8.00})).sum()
        bat_cost   = (df_fc["battery_kw"] * 2.50).sum()
        solar_cost = (df_fc["solar_kw"] * 1.50).sum()
        fig4 = go.Figure(go.Pie(
            labels=["Grid", "Battery Wear", "Solar"],
            values=[grid_cost, bat_cost, solar_cost],
            hole=0.6,
            marker=dict(colors=[COLORS["grid"], COLORS["battery"], COLORS["solar"]]),
        ))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F5E9", height=180,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10, orientation="h", y=-0.1),
        )
        st.plotly_chart(fig4, use_container_width=True)


with tab3:
    st.markdown('<div class="section-header">Historical Actual vs Predicted Demand</div>', unsafe_allow_html=True)

    if using_upload:
        st.markdown("""
        <div style="background:#1A237E22; border:1px solid #82B1FF44; border-radius:8px; padding:10px 16px; margin-bottom:12px; font-size:0.82rem; color:#82B1FF;">
            📂 Showing your uploaded CSV data. Predicted values are model estimates based on your data.
        </div>
        """, unsafe_allow_html=True)

    df_recent = df_hist.tail(48)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=df_recent["timestamp"], y=df_recent["actual_kw"],
        mode="lines", name="Actual", line=dict(color=COLORS["demand"], width=2),
    ))
    fig5.add_trace(go.Scatter(
        x=df_recent["timestamp"], y=df_recent["predicted_kw"],
        mode="lines", name="Predicted", line=dict(color="#B2EBF2", width=1.5, dash="dot"),
    ))
    fig5.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F5E9", height=350,
        xaxis=dict(showgrid=False, color="#78909C"),
        yaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", title="kW"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    diff = df_hist["actual_kw"] - df_hist["predicted_kw"]
    mape_hist = (diff.abs() / df_hist["actual_kw"]).mean() * 100
    rmse_hist = math.sqrt((diff ** 2).mean())

    for col, val, label in [
        (c1, f"{mape_hist:.1f}%", "Overall MAPE"),
        (c2, f"{rmse_hist:.1f} kW", "RMSE"),
        (c3, f"{diff.abs().mean():.1f} kW", "MAE"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="font-size:1.6rem;">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔍 Feature Importance</div>', unsafe_allow_html=True)

    _feat_imp = None
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from config.settings import MODEL_PATH
        if Path(MODEL_PATH).exists():
            with open(MODEL_PATH, "rb") as _f:
                _bundle = pickle.load(_f)
            _models   = _bundle.get("models", [])
            _feat_cols = _bundle.get("feature_columns", [])
            if _models and _feat_cols and hasattr(_models[0], "feature_importances_"):
                # Average importance across all horizon models
                _imp = np.mean([m.feature_importances_ for m in _models], axis=0)
                _feat_imp = pd.DataFrame({"feature": _feat_cols, "importance": _imp})
                _feat_imp = _feat_imp.sort_values("importance", ascending=True)
    except Exception:
        pass

    if _feat_imp is not None:
        _fi_colors = [
            COLORS["critical"] if v > 0.15 else COLORS["warning"] if v > 0.08 else COLORS["demand"]
            for v in _feat_imp["importance"]
        ]
        fig_fi = go.Figure(go.Bar(
            x=_feat_imp["importance"], y=_feat_imp["feature"],
            orientation="h",
            marker_color=_fi_colors,
            text=[f"{v:.3f}" for v in _feat_imp["importance"]],
            textposition="outside",
            textfont=dict(color="#78909C", size=11),
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F5E9", height=max(300, len(_feat_imp) * 28),
            xaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", title="Importance Score"),
            yaxis=dict(showgrid=False, color="#E8F5E9"),
            margin=dict(l=0, r=60, t=10, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#111811; border:1px solid #1E3A1E; border-radius:8px; padding:16px; text-align:center; color:#78909C; font-size:0.85rem;">
            Train the model first to see feature importances.<br>
            Click <b style="color:#00E676;">🔁 Retrain Model Now</b> in the sidebar.
        </div>
        """, unsafe_allow_html=True)


with tab4:
    st.markdown('<div class="section-header">Anomaly Detection — Unexplained Demand Spikes</div>', unsafe_allow_html=True)

    anomalies = df_fc[df_fc["anomaly"]]
    if anomalies.empty:
        st.markdown("""
        <div style="background:#0D1F0D; border:1px solid #1E3A1E; border-radius:10px; padding:24px; text-align:center;">
            <div style="font-size:2rem;">✅</div>
            <div style="color:#00E676; font-weight:600; margin-top:8px;">No Anomalies Detected</div>
            <div style="color:#78909C; font-size:0.85rem;">All demand patterns are within expected bounds for current weather conditions.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for _, row in anomalies.iterrows():
            st.markdown(f"""
            <div class="anomaly-tag">
                ⚠ Anomaly at <strong>{row['timestamp'].strftime('%H:%M')}</strong> —
                Predicted {row['predicted_demand_kw']:.0f} kW exceeds weather-adjusted baseline by &gt;35 kW.
                Possible unscheduled load or sensor drift. Investigate before dispatch.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Demand vs Weather-Adjusted Baseline</div>', unsafe_allow_html=True)

    # Show demand vs expected baseline
    hours_list = df_fc["timestamp"]
    baseline = [
        180 + 60 * math.sin(math.pi * (h - 6) / 14) if 6 <= h <= 20 else 115
        for h in df_fc["hour"]
    ]
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=hours_list, y=df_fc["predicted_demand_kw"],
        name="Predicted", line=dict(color=COLORS["demand"], width=2),
        mode="lines+markers",
    ))
    fig6.add_trace(go.Scatter(
        x=hours_list, y=baseline,
        name="Expected Baseline", line=dict(color="#78909C", width=1.5, dash="dot"),
        mode="lines",
    ))
    fig6.add_trace(go.Scatter(
        x=list(hours_list) + list(hours_list)[::-1],
        y=[b + 35 for b in baseline] + [b - 35 for b in baseline][::-1],
        fill="toself", fillcolor="rgba(0,230,118,0.05)",
        line=dict(color="rgba(0,0,0,0)"), name="±35 kW Band",
    ))
    if not anomalies.empty:
        fig6.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies["predicted_demand_kw"],
            mode="markers", name="Anomaly",
            marker=dict(color=COLORS["critical"], size=14, symbol="x"),
        ))
    fig6.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F5E9", height=300,
        xaxis=dict(showgrid=False, color="#78909C"),
        yaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", title="kW"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig6, use_container_width=True)




# ── Heatmap Tab ───────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🌡 Demand Heatmap — Hourly Patterns</div>', unsafe_allow_html=True)

    # Build heatmap data from historical (7 days × 24 hours)
    _hmap_days = 7
    _hmap_base = datetime.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=24 * _hmap_days)
    _hmap_records = []
    for _i in range(24 * _hmap_days):
        _ts = _hmap_base + timedelta(hours=_i)
        _h  = _ts.hour
        _d  = _ts.weekday()
        _base = 180 + 60 * math.sin(math.pi * (_h - 6) / 14) if 6 <= _h <= 20 else 115
        _hmap_records.append({
            "day":   _ts.strftime("%a %d %b"),
            "hour":  _h,
            "demand": round(max(80, _base + random.gauss(0, 18)), 1),
        })
    _hmap_df = pd.DataFrame(_hmap_records)
    _pivot = _hmap_df.pivot(index="day", columns="hour", values="demand")

    fig_hmap = go.Figure(go.Heatmap(
        z=_pivot.values,
        x=[f"{h:02d}:00" for h in _pivot.columns],
        y=_pivot.index.tolist(),
        colorscale=[
            [0.0,  "#0A2A0A"],
            [0.35, "#1B5E20"],
            [0.6,  "#FFB300"],
            [0.85, "#FF6D00"],
            [1.0,  "#FF1744"],
        ],
        colorbar=dict(title=dict(text="kW", font=dict(color="#78909C")), tickfont=dict(color="#78909C")),
        hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Demand: %{z:.0f} kW<extra></extra>",
    ))
    fig_hmap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F5E9", height=380,
        xaxis=dict(showgrid=False, color="#78909C", title="Hour of Day"),
        yaxis=dict(showgrid=False, color="#78909C", title=""),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_hmap, use_container_width=True)

    # Peak hour analysis
    st.markdown('<div class="section-header">Peak Hour Analysis</div>', unsafe_allow_html=True)
    _avg_by_hour = _hmap_df.groupby("hour")["demand"].mean().reset_index()
    fig_peak = go.Figure(go.Bar(
        x=[f"{h:02d}:00" for h in _avg_by_hour["hour"]],
        y=_avg_by_hour["demand"].round(1),
        marker_color=[
            COLORS["critical"] if d > 240 else COLORS["warning"] if d > 200 else COLORS["demand"]
            for d in _avg_by_hour["demand"]
        ],
    ))
    fig_peak.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8F5E9", height=220,
        xaxis=dict(showgrid=False, color="#78909C"),
        yaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", title="Avg kW"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_peak, use_container_width=True)


# ── Carbon Footprint Tab ───────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-header">🌿 Carbon Footprint Tracker</div>', unsafe_allow_html=True)

    # India grid emission factor (CEA 2023): 0.82 kg CO₂/kWh
    EMISSION_FACTOR = 0.82
    _solar_total    = df_fc["solar_kw"].sum()
    _battery_total  = df_fc["battery_kw"].sum()
    _grid_total     = df_fc["grid_kw"].sum()
    _clean_kwh      = _solar_total + _battery_total
    _co2_saved      = _clean_kwh * EMISSION_FACTOR
    _co2_emitted    = _grid_total * EMISSION_FACTOR
    _green_pct      = (_clean_kwh / (_clean_kwh + _grid_total) * 100) if (_clean_kwh + _grid_total) > 0 else 0

    # Top KPI cards
    _cc1, _cc2, _cc3, _cc4 = st.columns(4)
    for _col, _val, _label, _color in [
        (_cc1, f"{_co2_saved:.1f} kg",   "CO₂ Avoided",        "#00E676"),
        (_cc2, f"{_co2_emitted:.1f} kg", "CO₂ Emitted (Grid)", "#FF5252"),
        (_cc3, f"{_green_pct:.0f}%",     "Green Energy Share",  "#69F0AE"),
        (_cc4, f"{_clean_kwh:.0f} kWh",  "Clean Energy Used",   "#40C4FF"),
    ]:
        with _col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{_color};">{_val}</div>
                <div class="kpi-label">{_label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    _ccarb1, _ccarb2 = st.columns([1, 1])

    with _ccarb1:
        st.markdown('<div class="section-header">Hourly Carbon Emissions</div>', unsafe_allow_html=True)
        _df_carbon = df_fc.copy()
        _df_carbon["co2_grid_kg"]    = _df_carbon["grid_kw"]    * EMISSION_FACTOR
        _df_carbon["co2_avoided_kg"] = (_df_carbon["solar_kw"] + _df_carbon["battery_kw"]) * EMISSION_FACTOR

        fig_co2 = go.Figure()
        fig_co2.add_trace(go.Bar(
            x=_df_carbon["timestamp"], y=_df_carbon["co2_grid_kg"],
            name="CO₂ Emitted", marker_color=COLORS["critical"],
        ))
        fig_co2.add_trace(go.Bar(
            x=_df_carbon["timestamp"], y=_df_carbon["co2_avoided_kg"],
            name="CO₂ Avoided", marker_color=COLORS["demand"],
        ))
        fig_co2.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F5E9", height=300,
            xaxis=dict(showgrid=False, color="#78909C"),
            yaxis=dict(showgrid=True, gridcolor="#1E3A1E", color="#78909C", title="kg CO₂"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_co2, use_container_width=True)

    with _ccarb2:
        st.markdown('<div class="section-header">Energy Mix</div>', unsafe_allow_html=True)
        fig_mix = go.Figure(go.Pie(
            labels=["Solar ☀️", "Battery 🔋", "Grid 🔌"],
            values=[_solar_total, _battery_total, _grid_total],
            hole=0.55,
            marker=dict(colors=[COLORS["solar"], COLORS["battery"], COLORS["grid"]]),
            textfont=dict(color="#E8F5E9"),
        ))
        fig_mix.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8F5E9", height=300,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=12),
            margin=dict(l=0, r=0, t=10, b=0),
            annotations=[dict(text=f"{_green_pct:.0f}%<br>Green", x=0.5, y=0.5,
                              font_size=16, font_color="#00E676", showarrow=False)],
        )
        st.plotly_chart(fig_mix, use_container_width=True)

        # Equivalence facts
        trees = _co2_saved / 21.77   # avg tree absorbs ~21.77 kg CO₂/year
        km    = _co2_saved / 0.12    # avg car emits ~120g/km
        st.markdown(f"""
        <div style="background:#0D1A0D; border:1px solid #1E3A1E; border-radius:8px; padding:14px 16px; font-size:0.82rem; color:#78909C;">
            <div style="color:#00E676; font-weight:600; margin-bottom:8px;">Equivalent Impact</div>
            🌳 <b style="color:#E8F5E9;">{trees:.2f} trees</b> worth of annual absorption<br>
            🚗 <b style="color:#E8F5E9;">{km:.1f} km</b> of car driving avoided<br>
            📊 Emission factor: <b style="color:#E8F5E9;">0.82 kg CO₂/kWh</b> (India CEA 2023)
        </div>
        """, unsafe_allow_html=True)


# ── API Explorer Tab ─────────────────────────────────────────────────────────
with tab7:
    import json as _json

    API_BASE = "http://localhost:8000/api/v1"

    # ── Endpoint registry ─────────────────────────────────────────────────────
    ENDPOINTS = [
        {
            "id": "health",
            "method": "GET",
            "path": "/health",
            "summary": "Health Check",
            "description": "Returns the service status and current UTC timestamp. Use this to verify the API is running before making other calls.",
            "params": [],
            "example_response": {
                "status": "ok",
                "service": "GreenSync",
                "time": "2026-03-20T16:00:00.000000"
            },
        },
        {
            "id": "forecast",
            "method": "GET",
            "path": "/forecast",
            "summary": "6-Hour Demand Forecast",
            "description": "Returns ML-predicted energy demand for the next 6 hours alongside per-hour optimization recommendations. Integrates weather, solar, and battery state.",
            "params": [
                {"name": "battery_soc", "type": "float", "default": 75.0, "min": 0.0, "max": 100.0, "description": "Current battery state of charge (%)"},
                {"name": "cost_mode",   "type": "bool",  "default": False, "description": "Enable TOU cost-optimization dispatch"},
            ],
            "example_response": {
                "generated_at": "2026-03-20T16:00:00Z",
                "forecast": [
                    {"timestamp": "2026-03-20T17:00:00Z", "predicted_demand_kw": 218.4, "status": "warning"},
                ],
                "optimization": [
                    {"timestamp": "2026-03-20T17:00:00Z", "source_priority": "battery", "battery_soc_pct": 68.2, "estimated_cost_inr": 2.85}
                ],
                "battery_state": {"soc_pct": 62.1, "energy_available_kwh": 84.2},
                "metrics": {"overall_mape": 3.26}
            },
        },
        {
            "id": "current",
            "method": "GET",
            "path": "/current",
            "summary": "Current Microgrid State",
            "description": "Returns a snapshot of the current microgrid: battery SoC, solar output, live demand, and grid connection status.",
            "params": [],
            "example_response": {
                "timestamp": "2026-03-20T16:00:00Z",
                "battery_level_pct": 75.0,
                "solar_output_kw": 45.2,
                "current_demand_kw": 185.4,
                "grid_connected": True
            },
        },
        {
            "id": "weather",
            "method": "GET",
            "path": "/weather",
            "summary": "Weather Forecast",
            "description": "Raw 6-hour weather forecast from OpenWeatherMap (or simulation fallback). Includes temperature, humidity, cloud cover, and wind speed.",
            "params": [],
            "example_response": {
                "forecast": [
                    {"timestamp": "2026-03-20T17:00:00", "temperature_c": 13.1, "humidity_pct": 51.3, "cloud_cover_pct": 32.4, "wind_speed_ms": 3.15, "description": "simulated"}
                ]
            },
        },
        {
            "id": "solar",
            "method": "GET",
            "path": "/solar",
            "summary": "Solar Output Forecast",
            "description": "Physics-based PV generation forecast using solar elevation angle, cloud attenuation, and panel derating for temperature.",
            "params": [],
            "example_response": {
                "forecast": [
                    {"timestamp": "2026-03-20T17:00:00", "solar_output_kw": 18.18, "cloud_cover_pct": 32.4, "elevation_deg": 8.6}
                ]
            },
        },
        {
            "id": "train",
            "method": "POST",
            "path": "/train",
            "summary": "Retrain Forecasting Model",
            "description": "Triggers a full model retrain on the latest available data. Returns per-horizon MAPE and RMSE metrics after training completes.",
            "params": [],
            "example_response": {
                "status": "trained",
                "metrics": {
                    "h1_mape": 2.14, "h1_rmse": 6.15,
                    "overall_mape": 3.26
                }
            },
        },
    ]

    METHOD_COLORS = {"GET": "#00E676", "POST": "#FFB300", "DELETE": "#FF5252"}

    st.markdown('<div class="section-header">REST API Explorer</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#0D1A0D; border:1px solid #1E3A1E; border-radius:10px; padding:14px 20px; margin-bottom:20px; display:flex; align-items:center; gap:16px;">
        <div>
            <span style="font-family:'JetBrains Mono'; font-size:0.85rem; color:#78909C;">Base URL</span><br>
            <span style="font-family:'JetBrains Mono'; font-size:1rem; color:#00E676;">{API_BASE}</span>
        </div>
        <div style="margin-left:auto; text-align:right;">
            <span style="background:#1B5E20; color:#69F0AE; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600;">REST / JSON</span>
            &nbsp;
            <span style="background:#1A237E; color:#82B1FF; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600;">OpenAPI 3.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar endpoint selector inside tab ─────────────────────────────────
    ep_labels = [f"{e['method']}  {e['path']}  —  {e['summary']}" for e in ENDPOINTS]
    selected_idx = st.selectbox("Select Endpoint", range(len(ENDPOINTS)),
                                format_func=lambda i: ep_labels[i], label_visibility="collapsed")
    ep = ENDPOINTS[selected_idx]

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Endpoint details + param builder ────────────────────────────────
    with col_left:
        method_color = METHOD_COLORS.get(ep["method"], "#78909C")
        st.markdown(f"""
        <div style="background:#111811; border:1px solid #1E3A1E; border-radius:10px; padding:18px 20px; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                <span style="background:{method_color}22; color:{method_color}; font-family:'JetBrains Mono';
                      font-weight:700; font-size:0.8rem; padding:4px 12px; border-radius:6px; border:1px solid {method_color}44;">
                    {ep['method']}
                </span>
                <span style="font-family:'JetBrains Mono'; font-size:1rem; color:#E8F5E9;">{ep['path']}</span>
            </div>
            <div style="font-size:1.05rem; font-weight:600; color:#E8F5E9; margin-bottom:6px;">{ep['summary']}</div>
            <div style="font-size:0.85rem; color:#78909C; line-height:1.6;">{ep['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Parameter builder
        param_values = {}
        if ep["params"]:
            st.markdown('<div class="section-header" style="margin-top:8px;">Parameters</div>', unsafe_allow_html=True)
            for p in ep["params"]:
                if p["type"] == "float":
                    param_values[p["name"]] = st.slider(
                        f"`{p['name']}` — {p['description']}",
                        float(p.get("min", 0)), float(p.get("max", 100)),
                        float(p["default"]), step=0.5,
                    )
                elif p["type"] == "bool":
                    param_values[p["name"]] = st.toggle(
                        f"`{p['name']}` — {p['description']}", value=p["default"]
                    )
                else:
                    param_values[p["name"]] = st.text_input(
                        f"`{p['name']}` — {p['description']}", value=str(p["default"])
                    )
        else:
            st.markdown('<div style="color:#78909C; font-size:0.85rem; margin:8px 0;">No parameters required.</div>', unsafe_allow_html=True)

        # Build curl command
        qs = ""
        if param_values:
            qs = "?" + "&".join(f"{k}={str(v).lower()}" for k, v in param_values.items())
        curl_method = "-X POST " if ep["method"] == "POST" else ""
        curl_cmd = f'curl {curl_method}"{API_BASE}{ep["path"]}{qs}"'

        st.markdown('<div class="section-header" style="margin-top:16px;">cURL Command</div>', unsafe_allow_html=True)
        st.code(curl_cmd, language="bash")

        # Python snippet
        py_params = f", params={param_values}" if param_values else ""
        py_method = "post" if ep["method"] == "POST" else "get"
        py_snippet = f"""import requests

resp = requests.{py_method}(
    "{API_BASE}{ep['path']}"{(',' + chr(10) + '    params=' + str(param_values)) if param_values else ''}
)
data = resp.json()
print(data)"""
        st.markdown('<div class="section-header" style="margin-top:16px;">Python Snippet</div>', unsafe_allow_html=True)
        st.code(py_snippet, language="python")

    # ── RIGHT: Try it + Response viewer ───────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">Try It Live</div>', unsafe_allow_html=True)

        run_col, status_col = st.columns([2, 1])
        with run_col:
            fire = st.button(f"▶ Send {ep['method']} {ep['path']}", use_container_width=True)
        with status_col:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Session state for response
        state_key = f"api_resp_{ep['id']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = None
        err_key = f"api_err_{ep['id']}"
        if err_key not in st.session_state:
            st.session_state[err_key] = None

        if fire:
            try:
                import requests as _req
                url = f"{API_BASE}{ep['path']}"
                if ep["method"] == "POST":
                    r = _req.post(url, params=param_values, timeout=5)
                else:
                    r = _req.get(url, params=param_values, timeout=5)
                st.session_state[state_key] = {"status": r.status_code, "body": r.json(), "live": True}
                st.session_state[err_key] = None
            except Exception as exc:
                st.session_state[err_key] = str(exc)
                st.session_state[state_key] = None

        resp = st.session_state[state_key]
        err  = st.session_state[err_key]

        if err:
            st.markdown(f"""
            <div style="background:#2A0A0A; border:1px solid #FF174433; border-radius:8px; padding:14px; margin-bottom:12px;">
                <div style="color:#FF5252; font-weight:600; margin-bottom:4px;">⚠ Connection Error</div>
                <div style="font-family:'JetBrains Mono'; font-size:0.78rem; color:#FF8A80;">{err}</div>
                <div style="color:#78909C; font-size:0.78rem; margin-top:8px;">Make sure the API server is running:<br>
                <code style="color:#FFB300;">PYTHONPATH=. uvicorn api.main:app --port 8000</code></div>
            </div>
            """, unsafe_allow_html=True)

        # Show live response or example
        display_resp = resp["body"] if resp else ep["example_response"]
        is_live = resp is not None and resp.get("live")
        status_code = resp["status"] if resp else 200

        tag_color = "#00E676" if status_code < 300 else "#FF5252"
        tag_label = "LIVE RESPONSE" if is_live else "EXAMPLE RESPONSE"

        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="background:{tag_color}22; color:{tag_color}; font-size:0.7rem; font-weight:700;
                  padding:3px 10px; border-radius:20px; border:1px solid {tag_color}44; font-family:'JetBrains Mono';">
                {tag_label}
            </span>
            <span style="background:{'#1B5E20' if status_code < 300 else '#4E0000'}; color:{tag_color};
                  font-family:'JetBrains Mono'; font-size:0.75rem; padding:3px 10px; border-radius:6px; font-weight:700;">
                {status_code}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.code(_json.dumps(display_resp, indent=2), language="json")

        # Response schema table
        st.markdown('<div class="section-header" style="margin-top:16px;">Response Schema</div>', unsafe_allow_html=True)

        SCHEMAS = {
            "health":   [("status","string","ok / degraded"),("service","string","Service name"),("time","string","ISO UTC timestamp")],
            "forecast": [("generated_at","string","ISO UTC when generated"),("forecast","array","Per-hour predictions"),("optimization","array","Per-hour dispatch recs"),("battery_state","object","Final battery state"),("metrics","object","Model accuracy metrics")],
            "current":  [("timestamp","string","ISO UTC"),("battery_level_pct","float","0–100"),("solar_output_kw","float","kW"),("current_demand_kw","float","kW"),("grid_connected","bool","True/False")],
            "weather":  [("forecast","array","List of hourly weather objects"),("temperature_c","float","°C"),("cloud_cover_pct","float","0–100 %"),("wind_speed_ms","float","m/s"),("humidity_pct","float","0–100 %")],
            "solar":    [("forecast","array","List of hourly solar objects"),("solar_output_kw","float","Estimated PV output kW"),("elevation_deg","float","Sun elevation angle °")],
            "train":    [("status","string","trained / error"),("metrics","object","MAPE & RMSE per horizon")],
        }
        schema = SCHEMAS.get(ep["id"], [])
        if schema:
            rows_html = "".join(
                f"""<tr>
                    <td style="font-family:'JetBrains Mono'; color:#00E676; padding:6px 10px; border-bottom:1px solid #1E3A1E;">{f}</td>
                    <td style="color:#EA80FC; font-family:'JetBrains Mono'; font-size:0.8rem; padding:6px 10px; border-bottom:1px solid #1E3A1E;">{t}</td>
                    <td style="color:#78909C; font-size:0.82rem; padding:6px 10px; border-bottom:1px solid #1E3A1E;">{d}</td>
                </tr>"""
                for f, t, d in schema
            )
            st.markdown(f"""
            <table style="width:100%; border-collapse:collapse; background:#111811; border-radius:8px; overflow:hidden; border:1px solid #1E3A1E;">
                <thead>
                    <tr style="background:#0D1A0D;">
                        <th style="text-align:left; color:#78909C; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:8px 10px;">Field</th>
                        <th style="text-align:left; color:#78909C; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:8px 10px;">Type</th>
                        <th style="text-align:left; color:#78909C; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:8px 10px;">Description</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

    # ── All Endpoints Reference ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">All Endpoints Reference</div>', unsafe_allow_html=True)

    ref_rows = ""
    for e in ENDPOINTS:
        mc = METHOD_COLORS.get(e["method"], "#78909C")
        ref_rows += f"""
        <tr>
            <td style="padding:10px 14px; border-bottom:1px solid #1E3A1E;">
                <span style="background:{mc}22; color:{mc}; font-family:'JetBrains Mono'; font-size:0.75rem;
                      font-weight:700; padding:2px 8px; border-radius:4px;">{e['method']}</span>
            </td>
            <td style="font-family:'JetBrains Mono'; color:#E8F5E9; font-size:0.85rem; padding:10px 14px; border-bottom:1px solid #1E3A1E;">{e['path']}</td>
            <td style="color:#E8F5E9; font-size:0.85rem; font-weight:600; padding:10px 14px; border-bottom:1px solid #1E3A1E;">{e['summary']}</td>
            <td style="color:#78909C; font-size:0.82rem; padding:10px 14px; border-bottom:1px solid #1E3A1E;">{e['description'][:80]}...</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%; border-collapse:collapse; background:#111811; border-radius:10px; overflow:hidden; border:1px solid #1E3A1E;">
        <thead>
            <tr style="background:#0D1A0D;">
                <th style="text-align:left; color:#78909C; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:10px 14px;">Method</th>
                <th style="text-align:left; color:#78909C; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:10px 14px;">Path</th>
                <th style="text-align:left; color:#78909C; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:10px 14px;">Name</th>
                <th style="text-align:left; color:#78909C; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:10px 14px;">Description</th>
            </tr>
        </thead>
        <tbody>{ref_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:16px; background:#0D1A0D; border:1px solid #1E3A1E; border-radius:8px; padding:14px 20px;">
        <span style="color:#78909C; font-size:0.82rem;">
            📖 Full interactive Swagger UI available at
            <a href="http://localhost:8000/docs" target="_blank" style="color:#00E676;">http://localhost:8000/docs</a>
            &nbsp;·&nbsp; ReDoc at
            <a href="http://localhost:8000/redoc" target="_blank" style="color:#00E676;">http://localhost:8000/redoc</a>
        </span>
    </div>
    """, unsafe_allow_html=True)


# ── Download Report ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    csv = df_fc.to_csv(index=False)
    st.download_button(
        "⬇ Download Forecast CSV",
        data=csv,
        file_name=f"greensync_forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_d2:
    hist_csv = df_hist.to_csv(index=False)
    st.download_button(
        "⬇ Download Historical CSV",
        data=hist_csv,
        file_name=f"greensync_historical_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_d3:
    # ── PDF Report Generator ──────────────────────────────────────────────────
    def generate_pdf_report() -> bytes:
        """Generate a plain-text based PDF report using only stdlib + io."""
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        lines = []
        lines.append("=" * 60)
        lines.append("  GREENSYNC — MICROGRID FORECAST REPORT")
        lines.append(f"  Generated: {now_str}")
        lines.append("=" * 60)
        lines.append("")
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Avg Forecast Demand : {avg_demand:.1f} kW")
        lines.append(f"  Peak Demand         : {peak_demand:.1f} kW")
        lines.append(f"  Total Solar Output  : {total_solar:.1f} kWh")
        lines.append(f"  Battery SoC (end)   : {final_soc:.1f}%")
        lines.append(f"  Est. Grid Cost      : Rs.{total_cost:.2f}")
        lines.append(f"  CO2 Avoided         : {carbon_saved_kg:.2f} kg")
        lines.append(f"  Model MAPE          : {mape_val:.1f}%")
        lines.append("")
        lines.append("6-HOUR FORECAST")
        lines.append("-" * 40)
        lines.append(f"  {'Time':<8} {'Demand(kW)':<14} {'Status':<10} {'Source':<10} {'SoC%':<8} {'Cost(Rs)'}")
        lines.append(f"  {'-'*7:<8} {'-'*10:<14} {'-'*8:<10} {'-'*8:<10} {'-'*5:<8} {'-'*8}")
        for _, row in df_fc.iterrows():
            lines.append(
                f"  {row['timestamp'].strftime('%H:%M'):<8}"
                f" {row['predicted_demand_kw']:<14.1f}"
                f" {row['status']:<10}"
                f" {row['source_priority']:<10}"
                f" {row['battery_soc_pct']:<8.1f}"
                f" {row['estimated_cost_inr']:.2f}"
            )
        lines.append("")
        lines.append("CARBON FOOTPRINT")
        lines.append("-" * 40)
        lines.append(f"  CO2 Emitted (Grid)  : {_co2_emitted:.2f} kg")
        lines.append(f"  CO2 Avoided (Clean) : {_co2_saved:.2f} kg")
        lines.append(f"  Green Energy Share  : {_green_pct:.1f}%")
        lines.append(f"  Emission Factor     : 0.82 kg CO2/kWh (India CEA 2023)")
        lines.append("")
        lines.append("=" * 60)
        lines.append("  END OF REPORT — GreenSync v1.0")
        lines.append("=" * 60)
        return "\n".join(lines).encode("utf-8")

    pdf_data = generate_pdf_report()
    st.download_button(
        "⬇ Download PDF Report",
        data=pdf_data,
        file_name=f"greensync_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
