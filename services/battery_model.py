"""
GreenSync — Battery State Model
Simulates and tracks battery charge/discharge cycles.
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class BatteryModel:
    """
    Lithium-ion battery storage model.

    Tracks state of charge (SoC) and simulates charge/discharge
    based on net power flow from solar + grid vs demand.
    """

    def __init__(self, capacity_kwh: float = 200.0, charge_efficiency: float = 0.95,
                 discharge_efficiency: float = 0.95, max_charge_rate_kw: float = 50.0,
                 max_discharge_rate_kw: float = 75.0, min_soc_pct: float = 20.0,
                 max_soc_pct: float = 95.0, initial_soc_pct: float = 75.0):
        self.capacity_kwh = capacity_kwh
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.max_charge_rate_kw = max_charge_rate_kw
        self.max_discharge_rate_kw = max_discharge_rate_kw
        self.min_soc_pct = min_soc_pct
        self.max_soc_pct = max_soc_pct
        self.soc_pct = initial_soc_pct

    @property
    def energy_available_kwh(self) -> float:
        """Energy available for discharge (kWh)."""
        return (self.soc_pct - self.min_soc_pct) / 100 * self.capacity_kwh

    @property
    def energy_headroom_kwh(self) -> float:
        """Space available for charging (kWh)."""
        return (self.max_soc_pct - self.soc_pct) / 100 * self.capacity_kwh

    def charge(self, power_kw: float, duration_h: float = 1.0) -> float:
        """
        Charge battery at `power_kw` for `duration_h` hours.
        Returns actual energy absorbed (kWh).
        """
        power_kw = min(power_kw, self.max_charge_rate_kw)
        energy_in = power_kw * duration_h * self.charge_efficiency
        headroom = self.energy_headroom_kwh
        actual_energy = min(energy_in, headroom)
        self.soc_pct += actual_energy / self.capacity_kwh * 100
        self.soc_pct = min(self.soc_pct, self.max_soc_pct)
        return actual_energy

    def discharge(self, power_kw: float, duration_h: float = 1.0) -> float:
        """
        Discharge battery at `power_kw` for `duration_h` hours.
        Returns actual energy delivered (kWh).
        """
        power_kw = min(power_kw, self.max_discharge_rate_kw)
        energy_out = power_kw * duration_h
        available = self.energy_available_kwh * self.discharge_efficiency
        actual_energy = min(energy_out, available)
        self.soc_pct -= (actual_energy / self.discharge_efficiency) / self.capacity_kwh * 100
        self.soc_pct = max(self.soc_pct, self.min_soc_pct)
        return actual_energy

    def simulate_hour(self, solar_kw: float, demand_kw: float,
                      grid_import_kw: float = 0.0) -> Dict:
        """
        Simulate one hour of battery operation.

        Returns dict with updated SoC and power flows.
        """
        net_solar = solar_kw - demand_kw

        if net_solar > 0:
            # Excess solar — charge battery
            charged = self.charge(net_solar)
            action = "charging"
        elif net_solar < 0:
            # Deficit — try to cover from battery first
            deficit = abs(net_solar)
            discharged = self.discharge(deficit)
            remaining_deficit = deficit - discharged / self.discharge_efficiency
            grid_import_kw = max(0, remaining_deficit)
            action = "discharging"
        else:
            action = "idle"
            charged = discharged = 0

        return {
            "solar_kw": solar_kw,
            "demand_kw": demand_kw,
            "grid_import_kw": round(grid_import_kw, 2),
            "soc_pct": round(self.soc_pct, 1),
            "action": action,
        }

    def get_state(self) -> Dict:
        return {
            "soc_pct": round(self.soc_pct, 1),
            "energy_available_kwh": round(self.energy_available_kwh, 1),
            "energy_headroom_kwh": round(self.energy_headroom_kwh, 1),
            "capacity_kwh": self.capacity_kwh,
        }
