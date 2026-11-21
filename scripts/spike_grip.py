"""Spike 2: is grip slip in MuJoCo physically believable?

Static test: grasp an item with squeeze force F, lift it slowly, hold still for 2 s.
Physics says it holds if 2 * mu * F > m * g, so the critical force is F_crit = m g / (2 mu).
We check: (1) no creep well above F_crit, (2) slip close to F_crit, (3) the measured pad
normal force equals the commanded grasp force.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sim.cell import Cell, Item
from sim.franka import CONVEYOR_TOP, PICK_XY, ContactParams

noslip = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cell = Cell(ContactParams(noslip_iterations=noslip))
item = Item(width=0.06, depth=0.05, height=0.08, mass=1.0, mu=0.5)
fcrit = item.mass * 9.81 / (2 * item.mu)
print(f"noslip_iterations={noslip} | item {item.mass} kg, mu {item.mu} -> F_crit = {fcrit:.2f} N per finger")

top = CONVEYOR_TOP + item.height
grasp_z = top - 0.03
for F in [0.6 * fcrit, 0.85 * fcrit, 0.95 * fcrit, 1.05 * fcrit, 1.2 * fcrit, 2.0 * fcrit, 4.0 * fcrit]:
    cell.reset(item)
    cell.move_tcp([*PICK_XY, top + 0.08], alpha=0.5)
    cell.move_tcp([*PICK_XY, grasp_z], alpha=0.2)
    cell.grip.grasp(F)
    cell.hold(0.4)
    fn = cell.finger_normal_force() / 2  # per finger
    cell.mark_grasp()
    cell.move_tcp([*PICK_XY, grasp_z + 0.06], alpha=0.1, record=True)
    s_lift = cell.trace.slip[-1] if cell.trace.slip else 0
    n0 = len(cell.trace.slip)
    cell.hold(2.0, record=True)
    s = np.array(cell.trace.slip)
    creep = (s[-1] - s[n0]) * 1000
    item_z = cell.item_pose()[0][2]
    held = item_z > CONVEYOR_TOP + item.height / 2 + 0.03
    print(f"F = {F:6.2f} N ({F / fcrit:4.2f}x crit) | measured pad force/finger {fn:6.2f} N | "
          f"slip during lift {s_lift * 1000:6.2f} mm | creep in 2 s hold {creep:7.3f} mm | {'HELD' if held else 'DROPPED'}")
