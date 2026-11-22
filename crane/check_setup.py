"""Pre-flight check for presenting the live crane demo on a new machine. Run it once, well before presenting:

    crane/.venv/bin/python crane/check_setup.py          (Linux / macOS)
    crane\\.venv\\Scripts\\python.exe crane\\check_setup.py  (Windows)

It checks Python, the packages, the GPU, the TabPFN key and licence (and downloads the model weights once, so
the demo starts fast), the cached tables the demo needs, and MuJoCo physics. It never prints the key.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True


def check(name, fn):
    global ok
    try:
        msg = fn(); print(f'  OK    {name}' + (f': {msg}' if msg else ''))
    except Exception as e:                                  # noqa: BLE001 - report anything that would stop the demo
        ok = False; print(f'  FAIL  {name}: {e}')


def load_env():
    p = os.path.join(os.path.dirname(HERE), '.env')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8-sig'):
            if '=' in line and not line.lstrip().startswith('#'):
                k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ.setdefault('TABPFN_NO_BROWSER', '1')


def main():
    load_env()
    print('Crane demo pre-flight check')
    check('Python 3.10+', lambda: (sys.version.split()[0] if sys.version_info >= (3, 10) else (_ for _ in ()).throw(RuntimeError(sys.version))))
    check('packages', lambda: (__import__('mujoco'), __import__('tabpfn'), __import__('sklearn'), __import__('numpy')) and 'mujoco, tabpfn, scikit-learn, numpy')

    def gpu():
        import torch
        if torch.cuda.is_available(): return f'CUDA, {torch.cuda.get_device_name(0)}'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): return 'Apple GPU (MPS)'
        return 'no GPU found: the demo still runs on CPU, but the green forecast ghost will lag'
    check('GPU', gpu)

    def token():
        t = os.environ.get('TABPFN_TOKEN', '')
        if not t or '<' in t or 'your key' in t: raise RuntimeError('put TABPFN_TOKEN=<your key> in .env (no < >, no quotes)')
        return 'set (not shown)'
    check('TabPFN key in .env', token)

    for f in ('live_fleet_rows.json', 'pickplace_decision_fleet.json'):
        check(f'cached table results/{f}', lambda f=f: (os.path.getsize(os.path.join(HERE, 'results', f)) // 1024 and f'{os.path.getsize(os.path.join(HERE, "results", f)) // 1024} KB'))

    def physics():
        sys.path.insert(0, HERE)
        import mujoco_crane as mc
        import mujoco
        m, d = mc.make(dict(L1=4.0, L2=2.0, m_hook=250.0, m_load=500.0, beam=2.0))
        for _ in range(200): mujoco.mj_step(m, d)
        return f'MuJoCo {mujoco.__version__} steps fine'
    check('MuJoCo physics', physics)

    def tabpfn():
        import numpy as np
        from tabpfn import TabPFNRegressor
        t0 = time.time(); X = np.random.rand(50, 3); y = X.sum(1)
        TabPFNRegressor(n_estimators=1).fit(X, y).predict(X[:3])
        return f'licence OK, weights ready ({time.time() - t0:.0f} s; first run downloads them)'
    check('TabPFN prediction', tabpfn)

    print('\nAll good: start the demo with crane/run_live.sh (Linux/macOS) or crane\\run_live.ps1 (Windows).' if ok else
          '\nFix the FAIL lines above before presenting.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
