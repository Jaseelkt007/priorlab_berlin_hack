"""Spike: does simulated suction hold, slide and peel where physics says it should?

Tests (product held by the lip ring, adhesion = vacuum holding force F):
  lift : pull up with acceleration a_z; fails when m (g + a_z) > F
  shear: flat product, sideways acceleration a; fails when m a > mu F  (plus lift share)
  peel : tall product, sideways acceleration a; fails when m a h > F r
Prints measured failure points next to the theory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np

from sim.suction import DT, LIP_R, Effector, Product, build, lip_contacts

G = 9.81


def trial(prod: Product, cup_d: float, F: float, a_lat: float = 0.0, a_up: float = 0.0, settle_t: float = 0.05):
    """Grip, then a pure vertical (a_up) or a lifted sideways (a_lat) move. Returns held?, max tilt, min contacts."""
    m, ids = build(cup_d, prod)
    d = mujoco.MjData(m)
    eff = Effector(m)
    top = prod.size[2]
    z_touch = top + LIP_R - 0.0003 - (prod.size[2] + 0.002 + LIP_R)  # joint offset relative to the start height
    m.actuator_gainprm[3, 0] = F
    mujoco.mj_forward(m, d)

    def run(T, ref, vac=1.0, rec=None):
        for k in range(int(T / DT)):
            p, v, a = ref(k * DT)
            d.ctrl[:3] = eff.force(d, p, v, a)
            d.ctrl[3] = vac
            mujoco.mj_step(m, d)
            if rec is not None:
                rec(d)

    # descend and seal
    run(0.06, lambda t: ([0, 0, z_touch * min(1, t / 0.04)], [0, 0, 0], [0, 0, 0]), vac=0.0)
    run(settle_t, lambda t: ([0, 0, z_touch], [0, 0, 0], [0, 0, 0]))
    stats = {"tilt": 0.0, "minc": 8}
    q0 = ids["prod_qpos"]

    def rec(d):
        quat = d.qpos[q0 + 3:q0 + 7]
        stats["tilt"] = max(stats["tilt"], np.degrees(2 * np.arccos(min(1.0, abs(quat[0])))))
        stats["minc"] = min(stats["minc"], lip_contacts(m, d, ids))

    if a_lat == 0.0:
        # vertical pull: accelerate up at a_up for 40 ms, then hold
        T = 0.04
        run(T, lambda t: ([0, 0, z_touch + 0.5 * a_up * t * t], [0, 0, a_up * t], [0, 0, a_up]), rec=rec)
        z1, v1 = z_touch + 0.5 * a_up * T * T, a_up * T
        run(0.04, lambda t: ([0, 0, z1 + v1 * t - 0.5 * a_up * t * t], [0, 0, v1 - a_up * t], [0, 0, -a_up]), rec=rec)
    else:
        # lift 2 cm gently, then sideways bang-bang: +a for 60 ms, -a for 60 ms
        run(0.1, lambda t: ([0, 0, z_touch + 0.02 * (3 * (t / 0.1) ** 2 - 2 * (t / 0.1) ** 3)], [0, 0, 0], [0, 0, 0]), rec=rec)
        zl, T = z_touch + 0.02, 0.06

        def lat(t):
            if t < T:
                return [0.5 * a_lat * t * t, 0, zl], [a_lat * t, 0, 0], [a_lat, 0, 0]
            t2 = t - T
            return [0.5 * a_lat * T * T + a_lat * T * t2 - 0.5 * a_lat * t2 * t2, 0, zl], [a_lat * (T - t2), 0, 0], [-a_lat, 0, 0]
        run(2 * T, lat, rec=rec)
    run(0.05, lambda t: (d.qpos[:3].copy(), [0, 0, 0], [0, 0, 0]), rec=rec)
    p_eff = d.xpos[ids["eff"]]
    p_prod = d.xpos[ids["prod"]]
    held = np.linalg.norm(p_eff[:2] - p_prod[:2]) < 0.01 and (p_eff[2] - p_prod[2]) < top / 2 + 0.01
    return held, stats["tilt"], stats["minc"]


def critical(fn, lo, hi, iters=12):
    """Smallest value of the swept parameter at which the trial fails (bisection)."""
    if not fn(lo):
        return lo
    if fn(hi):
        return float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    print("== 1. LIFT: product mass m, holding force F; critical upward accel vs theory a* = F/m - g")
    for F in (5.0, 10.0):
        prod = Product(size=(0.05, 0.05, 0.02), mass=0.2, mu=0.6)
        a_star = critical(lambda a: trial(prod, 0.02, F, a_up=a)[0], 0, 200)
        print(f"   F={F:5.1f} N, m=0.2 kg: measured a* = {a_star:6.1f} m/s²   theory {F / 0.2 - G:6.1f}")

    print("== 2. SHEAR: flat product (CoM just below the cup); theory a* ≈ mu F / m  (lift share small)")
    for mu in (0.3, 0.6):
        prod = Product(size=(0.06, 0.045, 0.006), mass=0.05, mu=mu)
        F = 5.0
        a_star = critical(lambda a: trial(prod, 0.02, F, a_lat=a)[0], 1, 400)
        print(f"   mu={mu:.1f}, F={F} N, m=0.05 kg: measured a* = {a_star:6.1f} m/s²   theory {mu * F / 0.05:6.1f}")

    print("== 3. PEEL: tall product, CoM h below the lip; theory a* ≈ F r / (m h)")
    for h_box in (0.04, 0.08):
        prod = Product(size=(0.04, 0.04, h_box), mass=0.1, mu=0.9)
        F, r = 8.0, 0.01
        a_star = critical(lambda a: trial(prod, 2 * r, F, a_lat=a)[0], 1, 400)
        h = h_box / 2
        print(f"   box height {h_box * 100:.0f} cm (h={h * 100:.0f} cm), F={F} N, r={r * 1000:.0f} mm, m=0.1 kg: measured a* = {a_star:6.1f}   theory peel {F * r / (0.1 * h):6.1f}, shear {0.9 * F / 0.1:6.1f}")
