"""Demo clip: TabPFN choosing the speed of each pick, next to the simulated pick it chose.

Replays the TabPFN policy along one bank stream (as in scripts/run_eval.py), keeps a few picks as scenes, then
re-simulates each scene's chosen pick with recording (deterministic per seed: the same pick the bank holds). Each
frame: the MuJoCo render (left) and the decision for that pick (right): predicted drop risk at every speed setting,
the risk target, the chosen setting, and the outcome.
Usage: .venv/bin/python scripts/demo_loop.py BANK.csv[,..] LOG.csv[,..] STREAM OUT.mp4 [--target 0.05] [--scenes 1,8,25,60]
"""
import argparse
import io
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

from policy.policies import X_COLS, TabPFNPolicy, replay
from scripts.view_pick import camera, record
from sim.line import PickCell
from sim.picklog import instance_setup
from sim.scenarios import stream_tasks

SURF, INK, INK2, GRID, BLUE, RED, GREEN = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#e34948", "#008300"
W, H, PW = 960, 720, 640          # render size and side-panel width


def panel(row, probs, target, n_logged, title, cost=None):
    fig, ax = plt.subplots(figsize=(PW / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    al = np.round(np.arange(0.2, 1.01, 0.1), 2)
    cols = [BLUE if np.isclose(a, row["alpha"]) else "#b9cdea" for a in al]
    ax.bar([f"{a:.0%}" for a in al], 100 * probs, color=cols, width=0.7)
    if cost is None:
        ax.axhline(100 * target, color=INK2, lw=1.2, ls="--")
        ax.text(8.4, 100 * target + 1.5, f"risk target {target:.0%}", ha="right", color=INK2, fontsize=10)
    else:
        ax.text(8.4, 92, f"chooses the best expected value per second\n(a drop costs {cost:g} picks)", ha="right",
                va="top", color=INK2, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("predicted drop risk (%)", color=INK2)
    ax.set_xlabel("speed setting (share of 150 m/s²)", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ok = not bool(row["dropped"])
    verdict = f"placed, {row['place_offset_mm']:.1f} mm off" if ok else "DROPPED"
    ax.set_title(f"{title}\nseal at touch {row['seal_kpa']:.1f} kPa · cup {int(row['cup_cycles']):,} cycles · "
                 f"{n_logged:,} picks in context", loc="left", fontsize=11, color=INK, pad=28)
    ax.text(0, 1.015, f"chose {row['alpha']:.0%} → {verdict}", transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=GREEN if ok else RED)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=100)
    plt.close(fig)
    return np.frombuffer(buf.getvalue(), np.uint8).reshape(H, PW, 4)[:, :, :3]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bank")
    ap.add_argument("log")
    ap.add_argument("stream")
    ap.add_argument("out")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--cost", type=float, default=None, help="value rule: a drop costs this many picks")
    ap.add_argument("--scenes", default="0,5,20,50")
    ap.add_argument("--slow", type=float, default=8.0)
    a = ap.parse_args()
    bank = pd.concat([pd.read_csv(f) for f in a.bank.split(",")], ignore_index=True)
    log = pd.concat([pd.read_csv(f) for f in a.log.split(",")], ignore_index=True)
    b = bank[bank["stream"] == a.stream]
    b = b[b.groupby("idx")["alpha"].transform("nunique") == 9]
    scen, prod = a.stream.split(":")
    ctx = log[log["product"] != prod] if scen == "new" else log

    # replay, keeping each scene's risk vector (the policy's own predictions at decision time)
    pol = TabPFNPolicy(None, cost=a.cost) if a.cost is not None else TabPFNPolicy(a.target)
    scenes = {int(x) for x in a.scenes.split(",")}
    probs, sizes = {}, {}
    orig = pol.choose

    def choose(cand):
        i = int(cand["idx"].iloc[0])
        if i in scenes:
            probs[i] = pol._proba(cand[X_COLS].to_numpy(float)) if pol.const is None else np.full(9, pol.const)
            sizes[i] = len(pol.log)
        return orig(cand)
    pol.choose = choose
    runs = replay(b, pol, ctx).set_index("idx")
    tasks = {t["idx"]: t for t in stream_tasks(scen, prod)}

    fps = 30
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{W + PW}x{H}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                           "-crf", "18", a.out], stdin=subprocess.PIPE)
    for i in sorted(scenes):
        row = runs.loc[i]
        setup, logged, _ = instance_setup(tasks[i], row["alpha"])
        cell, frames, out = record(PickCell(setup))
        assert bool(out["dropped"]) == bool(row["dropped"]), "re-simulated pick differs from the bank"
        side = panel(row, probs[i], a.target, sizes[i], f"{prod.replace('_', ' ')}, pick {i + 1} ({scen})", a.cost)
        m, d = cell.m, mujoco.MjData(cell.m)
        r = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera()
        camera(cam)
        step = max(1, int(round(1000 / fps / a.slow)))
        for q in frames[::step] + [frames[-1]] * fps:
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            ff.stdin.write(np.concatenate([r.render(), side], axis=1).tobytes())
        print(f"scene pick {i + 1}: alpha {row['alpha']:.1f}, p {row['p_drop']:.3f}, dropped {bool(row['dropped'])}",
              flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)
