"""
GreenSync — Optimization Engine
Recommends power source switching decisions for each forecast hour.
Supports cost-optimization mode with time-of-use tariffs.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

from services.battery_model import BatteryModel
from utils.helpers import classify_demand

logger = logging.getLogger(__name__)


PEAK_HOURS = set(range(17, 21))          # 17:00–20:59 peak tariff
GRID_COST_PEAK = 15.00                   # ₹/kWh (peak tariff)
GRID_COST_OFF_PEAK = 8.00               # ₹/kWh (off-peak tariff)
SOLAR_COST = 1.50                        # ₹/kWh near-zero marginal cost
BATTERY_COST = 2.50                      # ₹/kWh wear cost per kWh


class OptimizationEngine:
    """
    Rule-based + cost-aware optimization for microgrid dispatch.

    Priority order (without cost mode):
      1. Solar (free, use first)
      2. Battery (stored clean energy)
      3. Grid (fallback)

    With cost mode:
      - During off-peak, prefer to charge battery from grid if cheap
      - During peak, drain battery aggressively to avoid expensive grid
    """

    def __init__(self, battery_model: BatteryModel, cost_mode: bool = False,
                 warning_threshold: float = 200.0, critical_threshold: float = 240.0):
        self.battery = battery_model
        self.cost_mode = cost_mode
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def _is_peak(self, hour: int) -> bool:
        return hour in PEAK_HOURS

    def _grid_cost(self, hour: int) -> float:
        return GRID_COST_PEAK if self._is_peak(hour) else GRID_COST_OFF_PEAK

    def recommend(self, hour: int, demand_kw: float, solar_kw: float) -> Dict:
        """
        Recommend dispatch strategy for a single hour.

        Returns
        -------
        dict with keys: source_priority, solar_kw, battery_kw, grid_kw,
                        battery_soc_after, estimated_cost_inr, notes
        """
        solar_kw = max(0, solar_kw)
        demand_kw = max(0, demand_kw)
        peak = self._is_peak(hour)
        grid_rate = self._grid_cost(hour)
        notes = []

        # --- Solar allocation ---
        solar_used = min(solar_kw, demand_kw)
        remaining_demand = demand_kw - solar_used
        excess_solar = solar_kw - solar_used

        # --- Charge battery with excess solar ---
        if excess_solar > 0:
            charged = self.battery.charge(excess_solar)
            notes.append(f"Excess solar {charged:.1f} kWh stored in battery.")

        # --- Cost-mode: cheap off-peak — pre-charge battery from grid ---
        grid_precharge = 0.0
        if self.cost_mode and not peak and self.battery.energy_headroom_kwh > 20:
            precharge_kw = min(self.battery.max_charge_rate_kw, self.battery.energy_headroom_kwh)
            charged_pre = self.battery.charge(precharge_kw)
            grid_precharge = charged_pre / self.battery.charge_efficiency
            notes.append(f"Off-peak pre-charge: {charged_pre:.1f} kWh added from grid.")

        # --- Battery to cover remaining demand ---
        battery_kw = 0.0
        grid_kw = 0.0

        if remaining_demand > 0:
            # Use battery more aggressively during peak
            bat_priority = 1.0 if (peak and self.cost_mode) else 0.7
            desired_from_bat = remaining_demand * bat_priority
            discharged = self.battery.discharge(desired_from_bat)
            battery_kw = discharged
            remaining_demand -= discharged
            if remaining_demand > 0:
                grid_kw = remaining_demand
                notes.append(f"Grid supplementing {grid_kw:.1f} kW.")

        total_grid = grid_kw + grid_precharge
        cost = (total_grid * grid_rate) + (battery_kw * BATTERY_COST) + (solar_used * SOLAR_COST)

        # Source priority label
        if solar_used >= demand_kw * 0.8:
            source_priority = "solar"
        elif battery_kw >= demand_kw * 0.5:
            source_priority = "battery"
        else:
            source_priority = "grid"

        status = classify_demand(demand_kw, self.warning_threshold, self.critical_threshold)
        if status == "critical":
            notes.append("⚠ CRITICAL demand — consider load shedding.")
        elif status == "warning":
            notes.append("⚡ WARNING: demand approaching capacity.")

        return {
            "hour": hour,
            "demand_kw": round(demand_kw, 2),
            "solar_kw": round(solar_used, 2),
            "battery_kw": round(battery_kw, 2),
            "grid_kw": round(total_grid, 2),
            "source_priority": source_priority,
            "battery_soc_pct": round(self.battery.soc_pct, 1),
            "estimated_cost_inr": round(cost, 2),
            "status": status,
            "is_peak": peak,
            "notes": notes,
        }

    def run_forecast(self, forecast_hours: List[Dict], solar_forecast: List[Dict]) -> List[Dict]:
        """
        Run optimization across all forecast hours.

        Parameters
        ----------
        forecast_hours : list of dicts with keys timestamp, predicted_demand_kw
        solar_forecast  : list of dicts with keys timestamp, solar_output_kw
        """
        solar_map = {s["timestamp"]: s["solar_output_kw"] for s in solar_forecast}
        results = []

        for f in forecast_hours:
            ts = f["timestamp"]
            demand = f["predicted_demand_kw"]
            solar = solar_map.get(ts, 0.0)
            hour = datetime.fromisoformat(ts.replace("Z", "")).hour
            rec = self.recommend(hour, demand, solar)
            rec["timestamp"] = ts
            results.append(rec)

        logger.info(f"Optimization complete for {len(results)} hours.")
        return results