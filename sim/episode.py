"""One item through the cell = one row of the log (see docs/PROBLEM_SPEC.md).

Sequence: approach -> grasp at the probe force -> 2 cm test lift (wrist force/torque,
like Franka's O_F_ext_hat_K) -> re-grip at the chosen force -> transfer at the chosen
alpha -> lower into the bin -> release. Inputs are what the robot knows before the
transfer; outputs are measured during and after it.
"""
from __future__ import annotations

import numpy as np

from .cell import Cell, Item
from .franka import BIN_INNER, BIN_XY, CONVEYOR_TOP, PICK_XY, SHELF_TOP

G = 9.81
SAFE_Z = 0.45           # TCP height for the transfer
APPROACH_ALPHA = 0.6    # empty-hand moves, not part of the decision
PROBE_FORCE = 30.0      # N, grasp force used for the test lift
FT_NOISE = (1.0, 0.05)  # N, Nm: order of the real external-wrench estimate noise


def _ft_world(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    sid = cell.m.site("ft").id
    R = cell.d.site_xmat[sid].reshape(3, 3)
    ft = cell.ft()
    return R @ ft[:3], R @ ft[3:]


def _avg_ft(cell: Cell, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    fs, ts = [], []
    for _ in range(int(seconds / 0.002)):
        cell.hold(0.002)
        f, t = _ft_world(cell)
        fs.append(f)
        ts.append(t)
    return np.mean(fs, axis=0), np.mean(ts, axis=0)


def run_episode(cell: Cell, item: Item, alpha: float, grip_n: float, rng: np.random.Generator | None = None) -> dict:
    rng = rng or cell.rng
    cell.reset(item)
    top = CONVEYOR_TOP + item.height
    grasp_z = top - min(0.03, item.height / 2)
    out = {"width_mm": item.width * 1000, "height_mm": item.height * 1000}

    cell.move_tcp([*PICK_XY, top + 0.08], APPROACH_ALPHA)
    cell.move_tcp([*PICK_XY, grasp_z], 0.3)
    cell.grip.grasp(min(PROBE_FORCE, grip_n if grip_n > 0 else PROBE_FORCE))
    cell.hold(0.3)
    out["grasp_width_mm"] = cell.grip.width(cell.d) * 1000

    # test lift: tare while the item rests on the conveyor, then lift 2 cm and hold
    f0, t0 = _avg_ft(cell, 0.1)
    cell.mark_grasp()
    cell.move_tcp([*PICK_XY, grasp_z + 0.02], 0.1, record=True)
    f1, t1 = _avg_ft(cell, 0.3)
    df = f1 - f0 + rng.normal(0, FT_NOISE[0], 3)
    dt = t1 - t0 + rng.normal(0, FT_NOISE[1], 3)
    est_mass = abs(df[2]) / G
    out["est_mass_kg"] = est_mass
    out["com_offset_mm"] = 1000 * np.linalg.norm(dt[:2]) / max(est_mass * G, 1e-3)
    out["probe_slip_mm"] = 1000 * (cell.trace.slip[-1] if cell.trace.slip else 0.0)

    # re-grip at the chosen force, then the transfer we are deciding about
    cell.grip.grasp(grip_n)
    cell.hold(0.15)
    cell.mark_grasp()
    t_transfer = 0.0
    t_transfer += cell.move_tcp([*PICK_XY, SAFE_Z], alpha, record=True)
    t_transfer += cell.move_tcp([*BIN_XY, SAFE_Z], alpha, record=True)
    place_z = SHELF_TOP + item.height - min(0.03, item.height / 2) + 0.01
    t_transfer += cell.move_tcp([*BIN_XY, place_z], alpha, record=True, settle=0.1)

    rel, _ = cell.item_in_tcp()
    rel0, _ = cell.trace.rel0
    dropped = bool(np.linalg.norm(rel - rel0) > 0.03)
    out["dropped"] = int(dropped)
    out["slip_max_mm"] = 1000 * max(cell.trace.slip) if cell.trace.slip else 0.0
    out["tilt_max_deg"] = max(cell.trace.tilt) if cell.trace.tilt else 0.0
    out["transfer_s"] = t_transfer

    cell.grip.move(0.08)
    cell.hold(0.3)
    p, _ = cell.item_pose()
    inside = np.all(np.abs(p[:2] - BIN_XY) < BIN_INNER / 2) and p[2] > SHELF_TOP
    out["placed"] = int(bool(inside) and not dropped)
    out["place_err_mm"] = 1000 * float(np.linalg.norm(p[:2] - BIN_XY)) if inside else float("nan")
    return out
