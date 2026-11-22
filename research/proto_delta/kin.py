"""Clavel delta IK/FK (Williams 2016 notation adapted). z down positive below motor plane."""
import numpy as np
PHI = np.deg2rad([0, 120, 240])

def ik(p, rb, ra, L, l):
    """p: platform centre (x,y,z) in motor frame, z<0 below motors. returns theta (rad, + = arm down)."""
    th = np.empty(3)
    for i, ph in enumerate(PHI):
        c, s = np.cos(ph), np.sin(ph)
        x = c * p[0] + s * p[1]; y = -s * p[0] + c * p[1]; z = p[2]  # arm i frame: arm along +x
        # elbow E = (rb + L cos t, 0, -L sin t); joint at platform J = (x+ra, y, z); |E-J| = l
        a = x + ra - rb
        # (L cos t - a)^2 + y^2 + (-L sin t - z)^2 = l^2  ->  -2La cos t + 2Lz sin t = l^2 - L^2 - a^2 - y^2 - z^2
        K = (l**2 - L**2 - a**2 - y**2 - z**2) / (2 * L)
        A, B = -a, z  # A cos t + B sin t = K
        R = np.hypot(A, B)
        if abs(K) > R: return None
        g = np.arctan2(B, A)
        t1 = g + np.arccos(K / R); t2 = g - np.arccos(K / R)
        # choose elbow-out solution (elbow outside, i.e. E_x > rb - small): pick the one with larger cos(t)
        cand = [t for t in (t1, t2)]
        cand = [((t + np.pi) % (2*np.pi)) - np.pi for t in cand]
        th[i] = max(cand, key=lambda t: np.cos(t))
    return th

def jac(p, th, rb, ra, L, l):
    """J_x xdot = J_q qdot ; returns Jx, Jq (3x3)."""
    Jx = np.zeros((3, 3)); Jq = np.zeros((3, 3))
    for i, ph in enumerate(PHI):
        c, s = np.cos(ph), np.sin(ph)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        E = R @ np.array([rb + L*np.cos(th[i]), 0, -L*np.sin(th[i])])
        J = p + R @ np.array([ra, 0, 0])
        d = J - E  # forearm vector
        dE = R @ np.array([-L*np.sin(th[i]), 0, -L*np.cos(th[i])])  # dE/dtheta
        Jx[i] = d
        Jq[i, i] = d @ dE
    return Jx, Jq

if __name__ == "__main__":
    import itertools
    geoms = {
        "A: ROS IRB360 (TU Gdansk) rb200 ra45 L350 l800": (0.200, 0.045, 0.350, 0.800),
        "B: Williams 1/1600 scaled": (0.567*np.sqrt(3)/6, 0.076*np.sqrt(3)/6, 0.524, 1.244),
        "C: delta-hil CAD (1600?) rb175 ra76 L508 l1046": (0.175, 0.076, 0.508, 1.046),
    }
    # ABB 1130 cylinder: TCP z 0.865..1.115 below base frame, motor plane at 0.275 => 0.590..0.840 below motors, r<=0.565
    for name, g in geoms.items():
        print(name)
        for off in (0.0, 0.05, 0.10):
            ths, ok = [], True
            for r, ang, zt in itertools.product(np.linspace(0, 0.565, 6), np.linspace(0, 2*np.pi, 13), np.linspace(0.590, 0.840, 6)):
                p = np.array([r*np.cos(ang), r*np.sin(ang), -(zt - off)])
                t = ik(p, *g)
                if t is None: ok = False; break
                ths.append(t)
            if ok:
                ths = np.rad2deg(np.array(ths))
                print(f"  TCP offset {off*1000:.0f} mm: theta range [{ths.min():.1f}, {ths.max():.1f}] deg")
            else:
                print(f"  TCP offset {off*1000:.0f} mm: unreachable point r={r:.3f} z={zt:.3f}")
