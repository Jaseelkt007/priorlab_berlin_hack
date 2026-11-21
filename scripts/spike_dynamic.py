"""Spike 3: does transfer acceleration cause slip and drops where physics says it should?

A 1 kg item, mu 0.5 (static F_crit = 9.81 N). Grid over alpha and grip force.
Holding needs 2 mu F >= m (g + a_up), so higher alpha needs more grip.
Also checks the test-lift estimates (mass, centre of mass).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sim.cell import Cell, Item
from sim.episode import run_episode

cell = Cell(seed=1)
item = Item(width=0.06, depth=0.05, height=0.08, mass=1.0, mu=0.5, com=(0.0, 0.015, 0.0))
fcrit = item.mass * 9.81 / (2 * item.mu)
print(f"item 1.0 kg, mu 0.5, CoM offset 15 mm -> static F_crit {fcrit:.1f} N")
print(f"{'alpha':>5} {'F/crit':>6} {'F N':>6} | {'est kg':>6} {'CoM mm':>6} | {'slip mm':>8} {'tilt deg':>8} {'drop':>4} {'placed':>6} {'transfer s':>10}")
t0 = time.perf_counter()
n = 0
for alpha in (0.2, 0.5, 0.8):
    for k in (1.1, 1.4, 2.0, 3.0):
        F = k * fcrit
        r = run_episode(cell, item, alpha, F)
        n += 1
        print(f"{alpha:5.1f} {k:6.1f} {F:6.1f} | {r['est_mass_kg']:6.2f} {r['com_offset_mm']:6.1f} | "
              f"{r['slip_max_mm']:8.2f} {r['tilt_max_deg']:8.2f} {r['dropped']:4d} {r['placed']:6d} {r['transfer_s']:10.2f}")
print(f"{(time.perf_counter() - t0) / n:.2f} s wall time per episode")
