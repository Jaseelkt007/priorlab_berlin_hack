"""Vacuum pneumatics for the suction cup (gate 1 of docs/MODEL_SPEC.md).

Everything here follows docs/research/model-suction.md §3; the section numbers below point to it.
- Chamber: isothermal lumped ODE (Richer & Hurmuzlu 2000), cup node plus optional line node
  through a hose (§3.1, §3.6).
- Ejector: vendor flow-vs-vacuum tables, monotone (PCHIP) interpolation, no assumed curve (§3.2).
- Leaks per lip segment: min(viscous slit flow, ISO 6358 nozzle flow) (§3.3 a, b). Viscous flow
  uses Tiwari & Persson eq. 13 with the Knudsen correction (eq. 19). Plus an equivalent
  roughness gap g0 (§3.3 d) and a porous Darcy term K_p (§3.3 c).
- Seal at touch: root of Q_ej = Q_leak (SMC adsorption-test method, §3.4).
- Sensor: first-order lag, sampling, Gaussian noise (§3.7).
Units: SI internally (Pa, m, m^3, kg/s). Vacuum levels are "kPa below ambient" only at the API edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

# constants (model-suction.md §5)
P_ATM = 101_300.0      # Pa, Piab test ambient
RHO0 = 1.185           # kg/m^3, ISO 6358 ANR reference density
R_AIR = 287.0          # J/(kg K)
GAMMA = 1.4
T0 = 293.15            # K
ETA_AIR = 1.81e-5      # Pa s
MFP_AIR = 0.068e-6     # m, mean free path of air at 20 °C, 101.3 kPa (standard value)
B_IDEAL = (2 / (GAMMA + 1)) ** (GAMMA / (GAMMA - 1))  # 0.528, ideal-nozzle critical pressure ratio


# ---------------------------------------------------------------- ejectors (§3.2)
class EjectorCurve:
    """Suction flow (normal litres per second, ANR) vs vacuum (kPa below ambient), from a vendor table."""

    def __init__(self, name: str, dp_kpa, q_nl_s, dp_max_kpa: float):
        dp = np.r_[np.asarray(dp_kpa, float), dp_max_kpa]
        q = np.r_[np.asarray(q_nl_s, float), 0.0]
        order = np.argsort(dp)
        self.name, self.dp_max = name, dp_max_kpa * 1e3
        self._f = PchipInterpolator(dp[order] * 1e3, q[order] * 1e-3, extrapolate=False)

    def q_m3s(self, dp_pa):
        """Normal volumetric suction flow (m^3/s ANR) at vacuum dp (Pa below ambient)."""
        dp = np.clip(np.asarray(dp_pa, float), 0.0, self.dp_max)
        return np.nan_to_num(self._f(dp), nan=0.0)

    def mdot(self, dp_pa):
        return RHO0 * self.q_m3s(dp_pa)


# Piab COAX cartridges [26][27] (model-suction.md §3.2 table). Si02-2: scfm at inHg converted (1 inHg = 3.386 kPa, 1 scfm = 0.47195 l/s).
EJECTORS = {
    "Si08-3": EjectorCurve("Piab COAX MINI Si08-3 @0.6 MPa", [0, 10, 20, 30, 40, 50, 60, 70],
                           [1.34, 0.73, 0.55, 0.35, 0.23, 0.17, 0.13, 0.08], 75.0),
    "Pi12-2": EjectorCurve("Piab COAX MINI Pi12-2 @0.34 MPa", [0, 10, 20, 30, 40, 50, 60, 70, 80],
                           [0.68, 0.60, 0.44, 0.27, 0.19, 0.14, 0.10, 0.06, 0.03], 90.0),
    "Si02-2": EjectorCurve("Piab COAX MICRO Si02-2 @0.6 MPa", np.array([0, 2, 5, 8, 11, 14, 17, 20]) * 3.386,
                           np.array([0.083, 0.062, 0.035, 0.024, 0.021, 0.018, 0.012, 0.0059]) * 0.47195, 75.0),
}
# Piab published evacuation times, seconds per litre, to the listed vacuum levels [26]
EVAC_TABLES = {
    "Si08-3": ([10, 20, 30, 40, 50, 60, 70], [0.10, 0.25, 0.48, 0.80, 1.30, 2.00, 2.90]),
    "Pi12-2": ([10, 20, 30, 40, 50, 60, 70, 80], [0.17, 0.32, 0.58, 1.1, 1.8, 2.7, 4.0, 6.4]),
}


# ---------------------------------------------------------------- leak flows (§3.3)
def sonic_conductance(area_m2) -> np.ndarray:
    """ISO 6358-3 ideal converging nozzle: C [m^3/(s Pa)] for throat area A (≈ 0.20 dm^3/(s bar) per mm^2)."""
    return np.asarray(area_m2) * np.sqrt(GAMMA / (R_AIR * T0)) * (2 / (GAMMA + 1)) ** ((GAMMA + 1) / (2 * (GAMMA - 1))) / RHO0


def iso6358_mdot(C, p_up, p_down, b: float = B_IDEAL, m: float = 0.5, T_up: float = T0):
    """ISO 6358-3 §5.2 mass flow (kg/s) from p_up to p_down through conductance C [m^3/(s Pa)]."""
    p_up, p_down = np.asarray(p_up, float), np.asarray(p_down, float)
    ratio = np.clip(p_down / p_up, 0.0, 1.0)
    base = np.asarray(C) * RHO0 * p_up * np.sqrt(T0 / T_up)
    sub = np.clip(1 - ((ratio - b) / (1 - b)) ** 2, 0.0, 1.0) ** m
    return np.where(ratio <= b, base, base * sub)


def slit_mdot(gap_m, L_y, L_x, p_a, p_c):
    """Tiwari & Persson eq. 13 + Knudsen correction (eq. 19): compressible laminar flow through a gap (kg/s)."""
    g = np.maximum(np.asarray(gap_m, float), 0.0)
    lam = MFP_AIR * P_ATM / (0.5 * (p_a + p_c))
    corr = 1 + (32 / np.pi) * lam / np.maximum(g, 1e-12)
    return (np.asarray(L_y) / L_x) * (p_a ** 2 - p_c ** 2) * g ** 3 / (24 * ETA_AIR * R_AIR * T0) * corr


@dataclass
class LipLeak:
    """Leak through the lip ring: n segments of arc length L_y, radial land L_x, discharge coefficient C_d."""
    r_lip: float = 0.0107      # m (SPB1 20: Ds/2)
    n_seg: int = 32
    L_x: float = 1.0e-3        # m, lip land width (not published; 0.5–2 mm)
    C_d: float = 0.7           # (0.6–0.8)
    g0: float = 0.0            # m, equivalent roughness gap added to every segment
    K_p: float = 0.0           # kg/(s Pa^2), porous product coefficient: mdot = K_p (p_a^2 - p_c^2)
    combine: str = "min"       # "min" (spec) or "series" (alternative for the V3 sensitivity check)

    def mdot(self, p_cup: float, gaps=None, p_a: float = P_ATM) -> float:
        L_y = 2 * np.pi * self.r_lip / self.n_seg
        g = self.g0 + (np.zeros(self.n_seg) if gaps is None else np.maximum(np.asarray(gaps, float), 0.0))
        visc = slit_mdot(g, L_y, self.L_x, p_a, p_cup)
        orif = iso6358_mdot(sonic_conductance(self.C_d * L_y * g), p_a, p_cup)
        if self.combine == "min":
            seg = np.minimum(visc, orif)
        else:  # resistances in series
            seg = np.where((visc > 0) & (orif > 0), 1 / (1 / np.maximum(visc, 1e-30) + 1 / np.maximum(orif, 1e-30)), 0.0)
        return float(seg.sum() + self.K_p * max(p_a ** 2 - p_cup ** 2, 0.0))


# ---------------------------------------------------------------- vacuum circuit (§3.1, §3.6)
@dataclass
class VacuumCircuit:
    """Cup node (+ optional line node behind a hose) evacuated by an ejector. State: absolute pressures."""
    ejector: EjectorCurve
    V_cup: float                    # m^3 (cup + short hose if single node)
    V_line: float = 0.0             # m^3; 0 => single node (ejector at the cup)
    C_hose: float | None = None     # m^3/(s Pa), sonic conductance cup<->line (ISO 6358-3)
    b_hose: float = B_IDEAL
    p_a: float = P_ATM
    n_sub: int = 20                 # RK4 substeps per call to step()
    p_cup: float = field(default=P_ATM)
    p_line: float = field(default=P_ATM)
    on: bool = True

    def _hose(self, p_cup, p_line):
        if self.V_line <= 0:
            return 0.0
        if p_cup >= p_line:
            return float(iso6358_mdot(self.C_hose, p_cup, p_line, self.b_hose))
        return -float(iso6358_mdot(self.C_hose, p_line, p_cup, self.b_hose))

    def deriv(self, p_cup, p_line, mdot_leak_fn, dVdt_cup):
        k = R_AIR * T0
        ej = float(self.ejector.mdot(self.p_a - (p_line if self.V_line > 0 else p_cup))) if self.on else 0.0
        leak = mdot_leak_fn(p_cup)
        if self.V_line <= 0:
            return k / self.V_cup * (leak - ej) - p_cup * dVdt_cup / self.V_cup, 0.0
        h = self._hose(p_cup, p_line)
        return (k / self.V_cup * (leak - h) - p_cup * dVdt_cup / self.V_cup, k / self.V_line * (h - ej))

    def step(self, dt: float, mdot_leak_fn, dVdt_cup: float = 0.0) -> float:
        """Advance dt seconds; mdot_leak_fn(p_cup) -> leak mass flow into the cup (kg/s). Returns vacuum at the cup (Pa)."""
        h = dt / self.n_sub
        pc, pl = self.p_cup, self.p_line
        f = lambda a, b: self.deriv(a, b, mdot_leak_fn, dVdt_cup)
        lo = self.p_a - max(self.ejector.dp_max, 1.0) - 1.0
        for _ in range(self.n_sub):
            k1 = f(pc, pl)
            k2 = f(pc + 0.5 * h * k1[0], pl + 0.5 * h * k1[1])
            k3 = f(pc + 0.5 * h * k2[0], pl + 0.5 * h * k2[1])
            k4 = f(pc + h * k3[0], pl + h * k3[1])
            pc = min(max(pc + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), lo), self.p_a)
            pl = min(max(pl + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), lo), self.p_a)
        self.p_cup, self.p_line = pc, pl
        return self.p_a - pc

    @property
    def vacuum(self) -> float:
        return self.p_a - self.p_cup


def steady_seal(ejector: EjectorCurve, leak_fn, p_a: float = P_ATM) -> float:
    """Steady vacuum (Pa below ambient) at the cup where ejector flow equals leak flow (single node, SMC §3.4)."""
    g = lambda dp: float(ejector.mdot(dp)) - leak_fn(p_a - dp)
    hi = ejector.dp_max - 1e-6
    if g(hi) >= 0:
        return hi
    return brentq(g, 0.0, hi, xtol=1e-3)


# ---------------------------------------------------------------- sensor (§3.7)
@dataclass
class VacuumSensor:
    tau: float = 1.5e-3       # s, response (SMC ZSE20 "1.5 ms or less")
    f_s: float = 435.0        # Hz (IO-Link 2.3 ms cycle)
    sigma: float = 5.0        # Pa noise (MPRLS; industrial value not found)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _y: float = 0.0
    _t_next: float = 0.0
    readings: list = field(default_factory=list)  # (t, kPa)

    def update(self, t: float, dt: float, vacuum_pa: float) -> None:
        self._y += (vacuum_pa - self._y) * (1 - np.exp(-dt / self.tau))
        if t >= self._t_next:
            self.readings.append((t, (self._y + self.rng.normal(0, self.sigma)) / 1e3))
            self._t_next += 1.0 / self.f_s
