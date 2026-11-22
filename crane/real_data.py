"""Real lab-crane data (Nazarbayev University tower crane, Mendeley Data V2, CC BY 4.0) -> TabPFN rows.

Standard library only, so it runs before any environment exists:
    python3 -I crane/real_data.py            # summary of every trolley run + row counts

Credit: Alhassan, Arif, Talapiden, Shehu, Haruna, Do (2025), doi:10.17632/x69hyyv3zw.2, CC BY 4.0.

Each run: 99 s at 100 Hz with columns L (m), x_r (m, reference), x (m, actual trolley), theta (rad).
Rows are made at 10 Hz. Features use only what a crane controller knows at time t:
the swing now and in the last 0.5 s, the trolley state, the cable length, and the *planned*
trolley displacement over the horizon (the controller commands it, so it is known).
"""
import math
import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'tower_crane_mendeley_v2')
G = 9.81
DT = 0.01            # sampling step of the dataset, s
STEP = 10            # 10 Hz rows
HORIZONS = (0.5, 1.0)

# Trolley-motion runs only (jib rotation is a different axis). payload: default / ball75 / cyl170 / beam500
RUNS = {
    1: dict(L=0.1, payload='default', hoist=False), 2: dict(L=0.2, payload='default', hoist=False),
    3: dict(L=0.3, payload='default', hoist=False), 4: dict(L=0.4, payload='default', hoist=False),
    5: dict(L=0.5, payload='default', hoist=False), 6: dict(L=0.6, payload='default', hoist=False),
    7: dict(L=0.7, payload='default', hoist=False), 8: dict(L=0.8, payload='default', hoist=False),
    9: dict(L=None, payload='default', hoist=True), 10: dict(L=0.5, payload='ball75', hoist=False),
    11: dict(L=0.5, payload='cyl170', hoist=False), 12: dict(L=0.5, payload='beam500', hoist=False),
    13: dict(L=None, payload='beam500', hoist=True),
}
FEATURES = ['th', 'th_m02', 'th_m05', 'dth', 'amp2s', 'x', 'dx', 'ddx', 'L', 'dL', 'plan_dx05', 'plan_dx10']


def run_path(i):
    for f in os.listdir(DATA):
        if f.startswith(f'({i})'):
            return os.path.join(DATA, f)
    raise FileNotFoundError(i)


def load_run(i):
    """-> dict of lists: L, xr, x, th (100 Hz)."""
    xml = zipfile.ZipFile(run_path(i)).read('xl/worksheets/sheet1.xml').decode()
    L, xr, x, th = [], [], [], []
    for r in re.findall(r'<row[^>]*>(.*?)</row>', xml)[1:]:           # skip header
        v = [float(m) for m in re.findall(r'<v>(.*?)</v>', r)]
        if len(v) == 4:
            L.append(v[0]); xr.append(v[1]); x.append(v[2]); th.append(v[3])
    meta = {k: v for k, v in RUNS[i].items() if k != 'L'}
    return dict(i=i, L_nom=RUNS[i]['L'], L=L, xr=xr, x=x, th=th, **meta)


def _mean(a, i, k):  # mean over a[i-k+1 .. i]
    return sum(a[i - k + 1:i + 1]) / k


def rows(run, horizons=HORIZONS):
    """Feature dicts + targets at 10 Hz. Smoothing: 5-sample (50 ms) means before differencing."""
    th, x, L = run['th'], run['x'], run['L']
    n, hmax = len(th), int(round(max(horizons) / DT))
    out = []
    for i in range(200, n - hmax, STEP):                                    # need 2 s of history
        thm = _mean(th, i, 5)
        f = dict(
            th=thm, th_m02=_mean(th, i - 20, 5), th_m05=_mean(th, i - 50, 5),
            dth=(thm - _mean(th, i - 5, 5)) / 0.05,
            amp2s=max(abs(v) for v in th[i - 200:i + 1]),
            x=_mean(x, i, 5), dx=(_mean(x, i, 5) - _mean(x, i - 10, 5)) / 0.1,
            ddx=(_mean(x, i, 5) - 2 * _mean(x, i - 10, 5) + _mean(x, i - 20, 5)) / 0.01,
            L=L[i], dL=(L[i] - L[i - 10]) / 0.1,
            plan_dx05=x[i + 50] - x[i], plan_dx10=x[i + 100] - x[i],
        )
        y = {f'th_p{int(h * 10):02d}': th[i + int(round(h / DT))] for h in horizons}
        out.append(dict(run=run['i'], t=i * DT, f=f, y=y))
    return out


def pendulum_predict(run, i, h, Le, zeta, sign):
    """Physics baseline: integrate theta'' = -(g/Le) sin th - sign*(xdd/Le) cos th - 2 zeta w th'
    from the state at sample i, driven by the *logged* (planned) trolley motion. Returns theta(t+h)."""
    th, x = run['th'], run['x']
    w = math.sqrt(G / Le)
    a = _mean(th, i, 5); da = (a - _mean(th, i - 5, 5)) / 0.05
    k = int(round(h / DT))
    for j in range(i, i + k):                                                 # semi-implicit Euler, 10 ms
        xdd = (x[min(j + 1, len(x) - 1)] - 2 * x[j] + x[j - 1]) / DT ** 2
        acc = -(G / Le) * math.sin(a) - sign * (xdd / Le) * math.cos(a) - 2 * zeta * w * da
        da += acc * DT; a += da * DT
    return a


if __name__ == '__main__':
    tot = 0
    for i in RUNS:
        r = load_run(i); R = rows(r); tot += len(R)
        moves = sum(1 for k in range(1, len(r['xr'])) if r['xr'][k] != r['xr'][k - 1])
        print(f"run {i:2d} {r['payload']:8s} hoist={str(r['hoist']):5s} L {min(r['L']):.2f}-{max(r['L']):.2f} m  "
              f"x {min(r['x']):+.2f}..{max(r['x']):+.2f} m  ref changes {moves:3d}  max|th| {max(map(abs, r['th'])):.3f}  rows {len(R)}")
    print('total rows', tot, 'features', FEATURES)
