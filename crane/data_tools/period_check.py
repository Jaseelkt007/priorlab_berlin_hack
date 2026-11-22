"""Measured swing period vs. textbook 2*pi*sqrt(L/g) for the trolley-motion runs.

    python3 -I crane/data_tools/period_check.py crane/data/tower_crane_mendeley_v2

Method: upward zero crossings of theta over the last 40 s of each 99 s run, after the trolley has stopped.
L_eff = g * (T_meas / 2pi)^2 is the pendulum length that would explain the measured period.
"""
import glob, math, os, re, sys, zipfile

def load(path):
    xml = zipfile.ZipFile(path).read('xl/worksheets/sheet1.xml').decode()
    out = []
    for r in re.findall(r'<row[^>]*>(.*?)</row>', xml)[1:]:          # skip header row
        v = [float(m) for m in re.findall(r'<v>(.*?)</v>', r)]
        if len(v) == 4: out.append(v)                                  # L, x_r, x, theta
    return out

for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.xlsx'))):
    n = os.path.basename(f)
    if 'Trolley motion' not in n or 'Simultaneous' in n: continue
    R = load(f); th = [r[3] for r in R]; L = sum(r[0] for r in R) / len(R)
    s = len(th) - 4000; ups = [i for i in range(s + 1, len(th)) if th[i - 1] < 0 <= th[i]]
    T = (ups[-1] - ups[0]) / (len(ups) - 1) * 0.01 if len(ups) > 2 else float('nan')
    Leff = 9.81 * (T / (2 * math.pi)) ** 2
    print(f"{n[:60]:60s} L={L:.2f} T_meas={T:.2f}s textbook={2*math.pi*math.sqrt(L/9.81):.2f}s L_eff={Leff:.2f} (+{Leff-L:.2f}) m")
