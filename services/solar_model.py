"""
GreenSync — Solar Output Model
Estimates PV generation based on weather conditions and time of day.
"""
import math
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class SolarModel:
    """Physics-inspired solar output estimator."""

    def __init__(self, panel_capacity_kw: float = 150.0, panel_efficiency: float = 0.20,
                 panel_area_m2: float = 800.0, latitude: float = 51.5):
        self.panel_capacity_kw = panel_capacity_kw
        self.panel_efficiency = panel_efficiency
        self.panel_area_m2 = panel_area_m2
        self.latitude = latitude

    def _solar_elevation_angle(self, dt: datetime) -> float:
        """Approximate solar elevation angle (degrees) for a given UTC datetime."""
        day_of_year = dt.timetuple().tm_yday
        hour_angle = 15 * (dt.hour + dt.minute / 60 - 12)  # degrees
        declination = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))

        lat_rad = math.radians(self.latitude)
        decl_rad = math.radians(declination)
        ha_rad = math.radians(hour_angle)

        sin_elev = (math.sin(lat_rad) * math.sin(decl_rad) +
                    math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad))
        elevation_deg = math.degrees(math.asin(max(-1, min(1, sin_elev))))
        return max(0, elevation_deg)

    def estimate_output(self, dt: datetime, cloud_cover_pct: float,
                        temperature_c: float = 20.0) -> float:
        """
        Estimate solar output in kW.

        Parameters
        ----------
        dt : datetime (UTC)
        cloud_cover_pct : 0–100
        temperature_c : ambient temperature (affects panel efficiency)
        """
        elevation = self._solar_elevation_angle(dt)
        if elevation <= 0:
            return 0.0

        # Direct Normal Irradiance (simplified)
        irradiance_clear = 1000 * math.sin(math.radians(elevation))   # W/m²
        cloud_factor = 1 - (cloud_cover_pct / 100) * 0.75
        irradiance = irradiance_clear * cloud_factor

        # Temperature derating: -0.4 %/°C above 25 °C
        temp_derate = 1 - max(0, (temperature_c - 25) * 0.004)

        output_kw = (irradiance * self.panel_area_m2 * self.panel_efficiency * temp_derate) / 1000
        return min(round(output_kw, 2), self.panel_capacity_kw)

    def forecast(self, weather_data: List[Dict]) -> List[Dict]:
        """
        Generate solar output forecast from weather forecast list.

        Parameters
        ----------
        weather_data : list of dicts with keys timestamp, cloud_cover_pct, temperature_c
        """
        results = []
        for w in weather_data:
            dt = datetime.fromisoformat(w["timestamp"].replace("Z", ""))
            output = self.estimate_output(dt, w["cloud_cover_pct"], w.get("temperature_c", 20))
            results.append({
                "timestamp": w["timestamp"],
                "solar_output_kw": output,
                "cloud_cover_pct": w["cloud_cover_pct"],
                "elevation_deg": round(self._solar_elevation_angle(dt), 1),
            })
        return results
