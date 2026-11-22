"""Evaluation streams for the speed policies (one stream = one product running through a drift scenario).

Each instance is simulated at every speed setting (sim.picklog.simulate_instance), so policies replay on identical
items. The scenarios are the drift drivers of the research (docs/research/model-products.md §3):

- wear:  the cup ages from 300k to 1.05M cycles (vendor interval 500k–750k; overdue cups happen) [§3.4]
- humid: a humid afternoon, room RH 40 → 80 % at 23 °C, chilled products below the dew point condense [§3.3]
- new:   a product the line has never logged (absent from the commissioning log), random line states
- batch: a new film batch after 20 picks: seal class −12 kPa (creased film) and mass +3 % [§3, batch changeovers]
"""
from __future__ import annotations

import numpy as np

from .products import LINE_PRODUCTS

SCENARIOS = {
    "wear": dict(products=LINE_PRODUCTS, n=60),
    "humid": dict(products=["cheese_wedge", "praline"], n=90),
    "new": dict(products=["pouch85", "cheese_wedge", "bar"], n=80),
    "batch": dict(products=["bar", "pouch85"], n=80),
}


def stream_tasks(scenario: str, product: str) -> list[dict]:
    """Ordered instance tasks (for sim.picklog.simulate_instance) of one stream."""
    sc = SCENARIOS[scenario]
    n = sc["n"]
    sid = list(SCENARIOS).index(scenario) * 10 + LINE_PRODUCTS.index(product)
    tasks = []
    for i in range(n):
        u = i / max(n - 1, 1)
        rng = np.random.default_rng(10_000 * sid + i)
        t = dict(stream=f"{scenario}:{product}", scenario=scenario, idx=i, product=product,
                 seed=1_000_000 + 10_000 * sid + i, belt_mps=float(rng.uniform(0.2, 0.5)))
        if scenario == "wear":
            t["state"] = dict(cup_cycles=int(300_000 + u * 750_000), room_t=float(rng.uniform(21, 24)),
                              room_rh=float(rng.uniform(45, 55)))
        elif scenario == "humid":
            t["state"] = dict(cup_cycles=int(rng.uniform(100_000, 300_000)), room_t=23.0, room_rh=40.0 + 40.0 * u)
        elif scenario == "batch":
            new = i >= 20
            t["state"] = dict(cup_cycles=int(rng.uniform(100_000, 400_000)), room_t=float(rng.uniform(21, 24)),
                              room_rh=float(rng.uniform(45, 55)), batch_mass_shift=0.03 if new else 0.0,
                              batch_seal_shift=-12.0 if new else 0.0)
        # "new": no state given -> simulate_instance draws a random line state (sim.picklog.random_state)
        tasks.append(t)
    return tasks


def all_tasks() -> list[dict]:
    return [t for sc, d in SCENARIOS.items() for p in d["products"] for t in stream_tasks(sc, p)]
