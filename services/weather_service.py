"""
GreenSync — Weather Service
Uses Open-Meteo API (https://open-meteo.com) — FREE, no API key required.
Falls back to simulated data if the API is unreachable.
"""
import logging
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

# Open-Meteo free forecast endpoint
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherService:
    """
    Fetches real hourly weather forecasts from Open-Meteo.
    No API key needed — just pass latitude & longitude.
    """

    def __init__(self, api_key: str = "",
                 city: str = "London",
                 lat: float = 51.5074,
                 lon: float = -0.1278):
        self.city = city
        self.lat  = lat
        self.lon  = lon
        logger.info(f"WeatherService initialised for {city} ({lat}, {lon}) via Open-Meteo")

    def _fetch_live(self, hours: int = 6) -> List[Dict]:
        params = {
            "latitude":   self.lat,
            "longitude":  self.lon,
            "hourly":     "temperature_2m,relativehumidity_2m,cloudcover,windspeed_10m,weathercode",
            "wind_speed_unit": "ms",
            "forecast_days":   2,
            "timezone":        "UTC",
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        hourly    = data["hourly"]
        times     = hourly["time"]
        temps     = hourly["temperature_2m"]
        humidity  = hourly["relativehumidity_2m"]
        cloud     = hourly["cloudcover"]
        wind      = hourly["windspeed_10m"]
        wmo_codes = hourly["weathercode"]

        now_utc   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        next_hour = now_utc + timedelta(hours=1)

        results = []
        for i, t in enumerate(times):
            ts = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
            if ts < next_hour:
                continue
            results.append({
                "timestamp":       ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "temperature_c":   round(temps[i], 1),
                "humidity_pct":    round(humidity[i], 1),
                "cloud_cover_pct": round(cloud[i], 1),
                "wind_speed_ms":   round(wind[i], 2),
                "description":     _wmo_description(wmo_codes[i]),
                "source":          "open-meteo",
            })
            if len(results) >= hours:
                break

        logger.info(f"Open-Meteo returned {len(results)} hourly forecast points.")
        return results

    def _simulate_weather(self, hours: int = 6) -> List[Dict]:
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        base_cloud = 30.0 + random.uniform(-15, 15)
        results = []
        for i in range(hours):
            ts  = now + timedelta(hours=i + 1)
            h   = ts.hour
            temp = 12.0 + 8.0 * math.sin(math.pi * (h - 6) / 12) + random.gauss(0, 1.5)
            cld  = max(0.0, min(100.0, base_cloud + random.gauss(0, 10)))
            wnd  = max(0.0, 4.0 + random.gauss(0, 1.5))
            hum  = max(20.0, min(100.0, 60.0 + random.gauss(0, 8)))
            results.append({
                "timestamp":       ts.strftime("%Y-%m-%dT%H:%M:%S"),
                "temperature_c":   round(temp, 1),
                "humidity_pct":    round(hum, 1),
                "cloud_cover_pct": round(cld, 1),
                "wind_speed_ms":   round(wnd, 2),
                "description":     "simulated",
                "source":          "simulation",
            })
        return results

    def get_forecast(self, hours: int = 6) -> List[Dict]:
        try:
            data = self._fetch_live(hours)
            if data:
                return data
            logger.warning("Open-Meteo returned empty data — using simulation.")
        except Exception as exc:
            logger.error(f"Open-Meteo API error: {exc} — falling back to simulation.")
        return self._simulate_weather(hours)

    def get_current(self) -> Dict:
        result = self.get_forecast(1)
        return result[0] if result else {}


def _wmo_description(code: int) -> str:
    WMO = {
        0: "clear sky",
        1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "icy fog",
        51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
        61: "slight rain",   63: "moderate rain",   65: "heavy rain",
        71: "slight snow",   73: "moderate snow",   75: "heavy snow",
        80: "slight showers",81: "moderate showers",82: "violent showers",
        95: "thunderstorm",  96: "thunderstorm with hail",
    }
    return WMO.get(int(code), f"wmo:{code}")