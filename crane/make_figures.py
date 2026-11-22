"""Slide-ready figures for the crane pitch, from the saved result files (no simulation, no TabPFN needed).

    crane/.venv/bin/python crane/make_figures.py      # needs matplotlib (pip install matplotlib)

Writes crane/figures/*.png (1600 px wide, white background, same colours as the explainer page:
teal = TabPFN, orange = physics / industry anti-sway, grey = no anti-sway / slow).
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results'); OUT = os.path.join(HERE, 'figures')
TAB, PHYS, GREY, BAD, INK, MUTED = '#0a7a7e', '#c45a17', '#8a8f94', '#c0392b', '#0f1a22', '#586672'
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9aa7b1',
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': 'white'})
J = lambda f: json.load(open(os.path.join(R, f)))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches='tight'); plt.close(fig); print('wrote figures/' + name)


def antisway():
    """Two half pushes cancel the swing; wrong timing leaves 18%."""
    g = 9.81; Lr = 4.5 + 0.3 + 2.0 + 0.18; Lw = 4.5 + 0.3 + 0.5 + 0.18; w = math.sqrt(g / Lr); a, ta, dt = 0.4, 3.0, 0.01; T = np.arange(0, 16, dt)
    acc = lambda t: np.where((t >= 0) & (t < ta), a, 0.0)
    def sw(u):
        x = v = 0.0; o = []
        for k in u: v += (-w * w * x - k) * dt; x += v * dt; o.append(x)
        return np.array(o) * 100
    dr, dw = math.pi * math.sqrt(Lr / g), math.pi * math.sqrt(Lw / g)
    plain = sw(acc(T)); right = sw(0.5 * acc(T) + 0.5 * acc(T - dr)); wrong = sw(0.5 * acc(T) + 0.5 * acc(T - dw))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for k, (y, col, title) in enumerate([(right, TAB, 'Anti-sway timed for the real sling: swing cancels'), (wrong, PHYS, 'Anti-sway set for a typed-in 0.5 m sling: swing survives')]):
        ax[k].plot(T, plain, color=GREY, lw=1.2, label='no anti-sway'); ax[k].plot(T, y, color=col, lw=2.6, label='with anti-sway')
        ax[k].axhspan(-5, 5, color=TAB, alpha=0.08); ax[k].set_title(title, fontsize=13, color=INK, loc='left'); ax[k].set_xlabel('time (s)')
        left = np.abs(y[T > ta + dr + 0.5]).max()
        ax[k].text(15.8, 45, f'swing left: {left:.0f} cm', ha='right', color=col, fontsize=13, fontweight='bold')
    ax[0].set_ylabel('load offset from trolley (cm)'); ax[0].legend(frameon=False, loc='lower right'); ax[0].text(0.2, 6.5, '±5 cm set-down zone', color=TAB, fontsize=10)
    save(fig, 'antisway.png')


def period():
    """Real lab crane: measured swing period vs textbook pendulum."""
    L = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.71, 0.81]); Tm = np.array([1.03, 1.19, 1.35, 1.48, 1.61, 1.71, 1.83, 1.93])
    x = np.linspace(0.05, 0.9, 100); fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, 2 * np.pi * np.sqrt(x / 9.81), color=GREY, lw=2, label='textbook 2π√(L/g)')
    ax.plot(x, 2 * np.pi * np.sqrt((x + 0.14) / 9.81), color=TAB, lw=1.5, ls='--', label='same with L + 0.14 m')
    ax.scatter(L, Tm, color=TAB, s=60, zorder=3, label='measured (lab crane)')
    ax.scatter([0.5, 0.5], [1.71, 1.77], color=[PHYS, BAD], marker='s', s=60, zorder=3, label='cylinder / beam payload')
    ax.set_xlabel('logged cable length L (m)'); ax.set_ylabel('swing period (s)'); ax.legend(frameon=False, fontsize=11)
    ax.set_title('On a real crane the textbook pendulum is wrong in every run', loc='left', fontsize=13); save(fig, 'real_crane_period.png')


def adaptation():
    """Real lab crane, unseen beam: error vs seconds of its own rows."""
    c = J('exp2_real.json')['step1_adaptation_beam']['curve']; s = [r['seconds_watched'] for r in c]; xs = range(len(s))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key, col, lab, lw in [('pendulum_cm', PHYS, 'pendulum, refitted', 2), ('trees_cm', GREY, 'standard ML (boosted trees), retrained', 2), ('tabpfn_cm', TAB, 'TabPFN, rows added only', 3.2)]:
        ax.plot(xs, [r['+1.0s'][key] for r in c], 'o-', color=col, lw=lw, label=lab)
    for i, r in zip(xs, c): ax.text(i, -0.075, f"{round(100 * r['+1.0s']['coverage90'])}%", ha='center', color=TAB, fontsize=10)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f'{v:g} s' for v in s]); ax.set_ylim(-0.1, 0.85)
    ax.set_xlabel('seconds of the new load added to the table   (teal % = how often TabPFN\'s 90% range held)'); ax.set_ylabel('error 1 s ahead (cm at the load)')
    ax.set_title('Real lab crane, unseen beam: 3× better after 5 s, no retraining', loc='left', fontsize=13); ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=11)
    save(fig, 'real_crane_adaptation.png')


def forecast_by_sling():
    """MuJoCo: physics vs TabPFN error as the hidden sling gets longer."""
    e = J('influence.json')['slider_effects']['sling']; x = np.arange(len(e)); wdt = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - wdt / 2, [r['physics_cm'] for r in e], wdt, color=PHYS, label='physics, typed-in 0.5 m sling')
    ax.bar(x + wdt / 2, [r['tabpfn_cm'] for r in e], wdt, color=TAB, label='TabPFN')
    for i, r in enumerate(e):
        ax.text(i - wdt / 2, r['physics_cm'] + 0.08, f"{r['physics_cm']:.1f}", ha='center', color=PHYS); ax.text(i + wdt / 2, r['tabpfn_cm'] + 0.08, f"{r['tabpfn_cm']:.1f}", ha='center', color=TAB)
    ax.set_xticks(x); ax.set_xticklabels([f"real sling {r['value']:g} m" for r in e]); ax.set_ylabel('error 1 s ahead (cm)')
    ax.set_title('MuJoCo crane: the hidden sling breaks physics, not TabPFN', loc='left', fontsize=13); ax.legend(frameon=False); save(fig, 'forecast_by_sling.png')


def closed_loop():
    s = J('closed_loop.json')['summary']; names = [('industry', 'Industry anti-sway\n(set for 0.5 m)', PHYS), ('slow', 'Slow and safe\n(40%)', GREY), ('tabpfn', 'TabPFN chooses\nspeed + anti-sway', TAB)]
    fig, ax = plt.subplots(figsize=(8, 4.4)); v = [s[k]['total_s'] for k, _, _ in names]
    ax.barh(range(3), v, color=[c for _, _, c in names]); ax.set_yticks(range(3)); ax.set_yticklabels([n for _, n, _ in names]); ax.invert_yaxis()
    for i, (k, _, _) in enumerate(names): ax.text(v[i] + 5, i, f"{v[i]:.0f} s · {s[k]['moves_per_hour']:.0f} moves/h", va='center', fontsize=12)
    ax.set_xlim(0, max(v) * 1.35); ax.set_xlabel('time for 12 new loads (s)')
    ax.set_title('MuJoCo closed loop: 35% faster on 12 new loads', loc='left', fontsize=13); save(fig, 'closed_loop.png')


def live():
    run = J('live_runs.json')['runs']['latest_9_cycles']; c = run['cycles']; x = np.arange(1, len(c) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, [r['cycle_s'] for r in c], color=[TAB if r['ok'] else BAD for r in c], width=0.65)
    ax.axhline(68, color=PHYS, ls='--', lw=1.5); ax.text(len(c) + 0.45, 69, 'typed-in 0.5 m: about 68 s', ha='right', color=PHYS, fontsize=11)
    for i, r in zip(x, c):
        ax.text(i, r['cycle_s'] + 1.2, f"{r['antisway']:g} m", ha='center', fontsize=11, color=INK)
        ax.text(i, 3, 'OK' if r['ok'] else 'MISS', ha='center', color='white', fontsize=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xlabel('cycle (label = anti-sway setting TabPFN chose)'); ax.set_ylabel('cycle time (s)'); ax.set_ylim(0, 82)
    ax.set_title(f"Live pick and place: learns the setting, then {sum(r['ok'] for r in c)}/{len(c)} OK, {run['mean_cycle_s']} s average", loc='left', fontsize=13)
    save(fig, 'live_cycles.png')


def prediction_simple():
    """One plain comparison for the pitch: how far off the 1-second prediction was, physics vs TabPFN."""
    a = J('exp2_real.json')['step1_adaptation_beam']['curve'][3]['+1.0s']        # 5 s of the new load's rows
    e = J('influence.json')['slider_effects']['sling'][-1]                         # hidden 2.5 m sling
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    panels = [(ax[0], 'Real lab crane: a load it never saw\n(after watching it for 5 s)', [('physics', a['pendulum_cm'], PHYS), ('standard ML\n(retrained)', a['trees_cm'], GREY), ('TabPFN', a['tabpfn_cm'], TAB)]),
              (ax[1], 'Full-size crane (MuJoCo):\nsling longer than typed in', [('physics', e['physics_cm'], PHYS), ('TabPFN', e['tabpfn_cm'], TAB)])]
    for a_, title, bars in panels:
        x = np.arange(len(bars)); v = [b[1] for b in bars]
        a_.bar(x, v, color=[b[2] for b in bars], width=0.6)
        for i, b in enumerate(bars): a_.text(i, b[1] * 1.03, f'{b[1]:.2f} cm' if b[1] < 1 else f'{b[1]:.1f} cm', ha='center', va='bottom', fontsize=15, fontweight='bold', color=b[2])
        a_.set_xticks(x); a_.set_xticklabels([b[0] for b in bars], fontsize=14); a_.set_ylim(0, max(v) * 1.25); a_.set_yticks([])
        a_.spines['left'].set_visible(False); a_.set_title(title, fontsize=14, loc='left', color=INK)
        ratio = bars[0][1] / bars[-1][1]
        a_.text(len(bars) - 1, max(v) * 1.13, f'{ratio:.0f}× closer than physics', ha='center', fontsize=14, color=TAB, fontweight='bold')
    fig.suptitle('How far off was the 1-second prediction?  (smaller is better)', x=0.01, ha='left', fontsize=15, color=INK)
    fig.tight_layout(); save(fig, 'prediction_simple.png')


if __name__ == '__main__':
    antisway(); period(); adaptation(); forecast_by_sling(); closed_loop(); live(); prediction_simple()
