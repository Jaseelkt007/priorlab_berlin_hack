"""Product catalogue and line drift (docs/research/model-products.md §2, §3, §5; docs/MODEL_SPEC.md §3).

Each pick draws one product instance plus the line's current state and returns the CupConfig the physics runs on,
together with what a real line would log BEFORE the move (checkweigher, vision, thermometers, cup cycle counter) and
the hidden physics that is never shown to the model.

Tags as in the research documents: S sourced, D derived, E estimate. Modelling deviations from the real tooling
are listed per product (`deviation`) and in DEBT.md.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .cup import CupConfig
from functools import lru_cache

from .pneumatics import EJECTORS, LipLeak, steady_seal

REF_DS = 21.4                          # mm: Schmalz SPB1 20, the calibrated cup (stiffness, mass, travel scale from it)
HOSE_CM3 = 1.0                         # cm³: short hose between cup and generator (4 mm bore, ~8 cm; model-suction.md §5)
# Schmalz SPG 33 chocolate bell (datasheet, generated 16.07.2026) [S]: Ds 32.1 mm, H 38 mm, volume 8.1 cm³, silicone
# 45/65 Shore A, max. operating vacuum -150 mbar. From its dimension drawing [D]: conical wall from the Ø32.1 mouth to a
# Ø15.7 neck over 15.2 mm, i.e. 28° from the axis.
SPG = dict(r_mouth=16.05, r_neck=7.85, h_cone=15.2, vol_cm3=8.1, shore=45.0, t_wall=1.0)   # t_wall [E]: "extremely
# thin sealing lip"; the drawing's section shows a wall of about 1 mm


def shore_a_modulus(shore: float) -> float:
    """Young's modulus (Pa) of a rubber from its Shore A hardness: Gent (1958), E[MPa] = 0.0981 (56 + 7.62336 S) /
    (0.137505 (254 - 2.54 S))."""
    return 1e6 * 0.0981 * (56 + 7.62336 * shore) / (0.137505 * (254 - 2.54 * shore))
SEAL_CLASS = {                         # kPa at touch with the Si08-3 ejector, unregulated (model-suction.md §3.4, [I])
    "smooth": (70.0, 75.0),            # foil, glass, film-like smooth
    "film": (55.0, 70.0),              # textured film, creased flow-wrap, decorated tops
    "porous": (20.0, 50.0),            # porous baked goods (through-product leak)
}


@dataclass(frozen=True)
class Product:
    key: str
    mass_g: tuple            # N(mean, sd), S/E per research §5
    size_mm: tuple           # L, W, H (cyl: D, D, H)
    shape: str
    cup: str                 # "flat" | "bellows" | "bell" (Schmalz SPG)
    cup_ds_mm: float         # cup outer (lip) diameter
    cup_deff_mm: float       # effective diameter -> A_eff (vendor suction force / 60 kPa, or the sealed product face)
    cup_vol_cm3: float       # vendor cup volume (without hose)
    p_set_kpa: float         # vacuum set point
    mu: tuple                # U(lo, hi) dry
    seal: str                # seal class
    mu_wet: tuple | None = None        # U(lo, hi) when condensation forms (product colder than the dew point)
    temp_c: tuple | None = None        # N(mean, sd) product temperature; None = room temperature
    com_off_mm: tuple = (0.0, 0.0)     # U(lo, hi) lateral CoM offset from the grasp point
    slosh: tuple | None = None         # (mass fraction, U(f_lo, f_hi) Hz)
    wrinkle_um_per_mm: float = 0.0     # film wrinkle: extra leak gap per mm of pick offset [E]
    deviation: str = ""


# Cup data. Flat: Schmalz SPF (ED-65) [S]: volume 0.8 / 1.13 / 1.70 / 2.53 / 5.09 cm³ for SPF 15 / 20 / 25 / 30 / 40;
# SPF 15 9.81 N and SPF 40 49.5 N at -0.6 bar (d_eff 14.4 / 32.4 mm). Bellows: Schmalz SPB1 table (model-suction.md
# §2.1) [S], log-log interpolated in Ds [D]; SPB2f 50 d_eff 29.5 mm [S].
CATALOGUE = {p.key: p for p in [
    Product("blister", (5.0, 0.1), (95, 46, 4), "box", "flat", 16, 14.4, 0.8, 60, (0.4, 0.6), "smooth",
            deviation="real tool uses 2 x Ø16 cups; one SPF 15-class cup modelled"),
    Product("praline", (10.5, 0.2), (22, 22, 20), "cyl", "bell", 2 * SPG["r_mouth"], 22.0, SPG["vol_cm3"], 15,
            (0.3, 0.7), "film", mu_wet=(0.2, 0.3), temp_c=(16.0, 1.5),
            deviation="SPG wall as 16 capsules on axial springs with the wall's bending stiffness; the seated product "
                      "rests on the joint stops; vacuum on the whole Ø22 top inside the seal line (research P2: 5.7 N)"),
    Product("digestive", (15.0, 0.3), (72, 72, 7.2), "cyl", "bellows", 28, 19.0, 6.1, 30, (0.5, 0.7), "porous"),
    Product("bar", (45.0, 0.7), (120, 30, 15), "box", "flat", 25, 17.0, 1.70, 50, (0.3, 0.6), "film",
            wrinkle_um_per_mm=2.0, deviation="real tool is a 20x45 rectangular bellows; a Ø25 round flat cup is modelled"),
    Product("cheese_wedge", (200.0, 3.0), (70, 50, 50), "box", "flat", 41.4, 32.4, 5.091, 60, (0.4, 0.6), "film",
            mu_wet=(0.2, 0.3), temp_c=(5.0, 1.5), com_off_mm=(10.0, 17.0),
            deviation="triangular 100x70 wedge as an equal-area 70x50 box with the measured CoM offset; Schmalz SPF 40 "
                      "flat cup (research: Ø40-50 bellows [E]; the bellows variant swings ~40° under the off-centre load, "
                      "and Ø41.4 keeps the lip > 2σ of the vision error inside the 50 mm face)"),
    Product("pouch85", (89.0, 1.5), (140, 90, 15), "box", "bellows", 50, 29.5, 32.4, 50, (0.3, 0.6), "film",
            slosh=(0.7, (2.0, 5.0)), deviation="film drawn into the cup not modelled; contents as one slosh mode; "
                                               "SPB2f 50 volume extrapolated from SPB1 (Ds³ law of SPB1 30-40) [D]"),
    Product("vial10r", (18.0, 0.3), (24, 24, 45), "cyl", "flat", 12, 8.2, 0.5, 70, (0.4, 0.6), "smooth",
            deviation="peel stress test only (research P7), not part of the line's product mix"),
]}
LINE_PRODUCTS = ["blister", "praline", "digestive", "bar", "cheese_wedge", "pouch85"]


def cup_geometry(p: Product, rim_mm: float | None = None) -> dict:
    """CupConfig fields for the product's cup. Flat and bellows cups scale from the calibrated SPB1 20 [E]: ring
    stiffness ∝ Ds, mass ∝ Ds², travel and bellows dimensions ∝ Ds. The bell's wall stiffness is derived below. A_eff from the effective
    diameter; volume from the vendor table plus the hose. A bell seals on the product rim (rim_mm, the instance's top
    radius), so only the part of the bell above the seal circle is evacuated [D: vendor volume minus the cone frustum
    below the seal line]."""
    s = p.cup_ds_mm / REF_DS
    vol = p.cup_vol_cm3
    g = dict(A_eff=np.pi * (p.cup_deff_mm / 2e3) ** 2, cup_mass=2.2e-3 * s ** 2, delta_max=0.004 * s,
             bellows=p.cup == "bellows")
    if p.cup == "bell":
        r_seal = rim_mm if rim_mm is not None else p.size_mm[0] / 2
        tb = (SPG["r_mouth"] - SPG["r_neck"]) / SPG["h_cone"]
        h = (SPG["r_mouth"] - r_seal) / tb
        vol -= np.pi * h / 3 * (SPG["r_mouth"] ** 2 + SPG["r_mouth"] * r_seal + r_seal ** 2) / 1e3
        beta = float(np.arctan(tb))
        # wall stiffness [D]: each segment is a strip of the thin silicone wall (width 2πr/n, thickness t) cantilevered
        # from the neck to the seal circle, loaded normal to the wall: k_n = 3EI/L³. A normal deflection δn equals an
        # axial slide δn/sin β of the segment, so the slide joint gets k_n·sin²β. An off-centre product only bends the
        # wall locally, so this is the stiffness that governs capture and tilt.
        L = (r_seal - SPG["r_neck"]) / np.sin(beta) / 1e3
        b, t = 2 * np.pi * r_seal / 1e3 / 16, SPG["t_wall"] / 1e3
        k_seg = 3 * shore_a_modulus(SPG["shore"]) * b * t ** 3 / 12 / L ** 3 * np.sin(beta) ** 2
        g.update(lip="cone", r_lip=r_seal / 1e3, cone_beta=beta, cone_r_mouth=SPG["r_mouth"] / 1e3,
                 cone_r_neck=SPG["r_neck"] / 1e3, k_ring=16 * k_seg,
                 delta_max=(r_seal - SPG["r_neck"]) / tb / 1e3)       # travel until the rim reaches the neck (seated)
    else:
        g.update(r_lip=p.cup_ds_mm / 2e3, k_ring=4000.0 * s)
    g.update(V_cup=(vol + HOSE_CM3) * 1e-6)
    if p.cup == "bellows":
        g.update(k_bz=1240.0 * s, R_b=(p.cup_ds_mm + p.cup_deff_mm) / 4e3, L_b=0.012 * s, Z_stroke=0.008 * s)
    return g


def n_cartridges(ds_mm: float) -> int:
    """Generator sized to the cup: lip leak scales with the lip circumference, so the flow capacity does too, in whole
    Si08-3 cartridges (SPB1 20 = 1) [E: vendors size the generator to the cup's leak, Schmalz/Piab sizing guides]."""
    return max(1, int(np.ceil(ds_mm / REF_DS - 1e-9)))


def _seal_kpa(r_lip: float, n_gen: int, g0: float = 0.0, K_p: float = 0.0, n_seg: int = 16) -> float:
    """Steady seal (kPa below ambient) of a closed lip ring with roughness gap g0 and parallel viscous leak K_p, on n_gen
    unregulated Si08-3 cartridges (SMC adsorption-test balance)."""
    leak = LipLeak(r_lip=r_lip, n_seg=n_seg, g0=g0, K_p=K_p)
    return steady_seal(EJECTORS["Si08-3"].parallel(n_gen), leak.mdot) / 1e3


def _bisect_log(f, target: float, lo: float, hi: float) -> float:
    """x in [lo, hi] (log space) with f(x) = target, for f falling monotonically in x."""
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return float(np.sqrt(lo * hi))


def _invert_seal(target_kpa: float, r_lip: float, n_seg: int, porous: bool, n_gen: int = 1) -> float:
    """g0 (m) or K_p giving the target steady seal at this cup with its generator, unregulated (SMC adsorption-test
    balance; the seal classes are touch-down levels of a correctly sized generator)."""
    if porous:
        return _bisect_log(lambda x: _seal_kpa(r_lip, n_gen, K_p=x, n_seg=n_seg), target_kpa, 1e-14, 1e-8)
    return _bisect_log(lambda x: _seal_kpa(r_lip, n_gen, g0=x, n_seg=n_seg), target_kpa, 1e-8, 2e-4)


@dataclass
class LineState:
    """Slow drift of the line (research §3): the per-pick features a real line logs, plus hidden drivers."""
    cup_cycles: int = 0            # picks since the cup was changed (counter on the controller)
    room_t: float = 22.0           # °C
    room_rh: float = 50.0          # %
    batch_mass_shift: float = 0.0  # relative shift of the batch mean mass
    batch_seal_shift: float = 0.0  # kPa shift of the batch seal class (porosity / film batch)


def dew_point(t_c: float, rh: float) -> float:
    """Magnus formula (Alduchov & Eskridge 1996 coefficients)."""
    a, b = 17.625, 243.04
    g = np.log(rh / 100.0) + a * t_c / (b + t_c)
    return b * g / (a - g)


WEAR_LIFE = 750_000      # cycles: vendor replacement interval (500k–750k, research §3.4, S weak)


@lru_cache(maxsize=None)
def wear_leak(key: str, r_lip: float, n_gen: int) -> float:
    """Leak coefficient K_w (kg/(s·Pa²)) of the product's cup at end of life. Wear (abraded, flattened, cracked lip)
    opens leak paths through the lip itself, in parallel with the product's roughness and porosity, and their area
    grows ~linearly with use (research §3.4, E). A fixed narrow channel carries viscous flow K_w·(p_a² - p_c²), the
    same law as the slit and Darcy terms. Calibrated to the maintenance rule "replace at 15 % leak-down" [S11] in the
    cup's own application: on the product's typical seal (mid seal class), K_w brings the seal to 85 % of new."""
    p = CATALOGUE[key]
    lo, hi = SEAL_CLASS[p.seal]
    mid = 0.5 * (lo + hi)
    porous = p.seal == "porous"
    x = _invert_seal(mid, r_lip, 16, porous, n_gen)
    g0, K_p = (0.0, x) if porous else (x, 0.0)
    return _bisect_log(lambda k: _seal_kpa(r_lip, n_gen, g0=g0, K_p=K_p + k), 0.85 * mid, 1e-16, 1e-8)


def sample(key: str, rng: np.random.Generator, st: LineState, dt: float = 2e-4) -> tuple[CupConfig, dict, dict]:
    """One product instance under the current line state -> (CupConfig, logged features, hidden truth)."""
    p = CATALOGUE[key]
    mass = rng.normal(p.mass_g[0], p.mass_g[1]) * (1 + st.batch_mass_shift) / 1e3
    size = tuple(np.array(p.size_mm) * (1 + rng.normal(0, 0.01, 3)) / 1e3)
    if p.shape == "cyl":
        size = (size[0], size[0], size[2])
    geo = cup_geometry(p, rim_mm=size[0] / 2 * 1e3)
    t_prod = st.room_t if p.temp_c is None else rng.normal(*p.temp_c)
    # condensation (research §3.3): only below the room dew point, and then "with probability rising with RH" [E:
    # linear from 0 at 35 % RH to 1 at 85 % RH; the film's effect on grip is unpublished]
    below = p.mu_wet is not None and t_prod < dew_point(st.room_t, st.room_rh)
    wet = bool(below and rng.uniform() < np.clip((st.room_rh - 35.0) / 50.0, 0.0, 1.0))
    mu = rng.uniform(*(p.mu_wet if wet else p.mu))
    w = st.cup_cycles / WEAR_LIFE
    mu *= 1 - 0.15 * min(w, 1.5)                       # worn lips: friction falls 10–20 % at end of life (E)
    lo, hi = SEAL_CLASS[p.seal]
    target = float(np.clip(rng.uniform(lo, hi) + st.batch_seal_shift, 5.0, 75.0 - 1e-3))
    porous = p.seal == "porous"
    n_gen = n_cartridges(p.cup_ds_mm)
    x = _invert_seal(target, geo["r_lip"], 16, porous, n_gen)
    g0, K_p = (0.0, x) if porous else (x, 0.0)
    K_wear = min(w, 1.5) * wear_leak(key, round(geo["r_lip"], 4), n_gen)
    ang = rng.uniform(0, 2 * np.pi)
    off = rng.uniform(*p.com_off_mm) / 1e3
    com = (off * np.cos(ang), off * np.sin(ang), 0.0)
    slosh = {}
    if p.slosh:
        slosh = dict(slosh_frac=p.slosh[0], slosh_hz=rng.uniform(*p.slosh[1]))
    cfg = CupConfig(**geo, n_gen=n_gen, p_set_kpa=p.p_set_kpa, shape=p.shape, size=size, mass=mass, com=com, mu=mu, g0=g0, K_p=K_p + K_wear,
                    dt=dt, **slosh)
    logged = dict(product=key, weight_g=mass * 1e3 + rng.normal(0, 0.05 if mass < 0.05 else 0.2),  # checkweigher
                  len_mm=size[0] * 1e3 + rng.normal(0, 0.5), wid_mm=size[1] * 1e3 + rng.normal(0, 0.5),
                  hgt_mm=size[2] * 1e3 + rng.normal(0, 0.5),                                       # 3-D vision ±0.5 mm
                  prod_t=t_prod + rng.normal(0, 0.3), room_t=st.room_t, room_rh=st.room_rh,
                  cup_cycles=st.cup_cycles, cup_d_mm=p.cup_ds_mm, p_set_kpa=p.p_set_kpa)
    hidden = dict(_mu=mu, _wet=wet, _g0_um=g0 * 1e6, _K_p=K_p, _K_wear=K_wear, _seal_target=target,
                  _com_off_mm=off * 1e3)
    return cfg, logged, hidden
