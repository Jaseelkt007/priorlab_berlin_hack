"""Experiment F: DAgger in context. Teach TabPFN to recover from its own mistakes, no training.

    MUJOCO_GL=egl conda run -n flybody python collect/dagger.py --rounds 4

Round 0 uses 512 random recorded steps as context (as in experiment E). Every round TabPFN
flies the smooth synthetic path (experiment E) until it falls or hits --round-steps; the
original policy labels every state TabPFN visited (what it would have done there); those
rows join a pool. The next context is 256 random recorded steps + up to 256 steps sampled
from the pool (512 steps = 6,144 rows, the most a 4 GB GPU fits). After the last round the
final context also flies the two saccade paths, which it never practised on.
Writes results/exp_f/: per-round drift curves, summary.json.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from closed_loop import OBS_KEYS, TabPFNController, recorded_steps  # noqa: E402
from preview_flight import Policy, find  # noqa: E402
from synthetic_paths import DT, DURATION, PATHS, make_env, make_path  # noqa: E402

BUDGET = 512  # context steps that fit in 4 GB with TabPFN 3.5 Fast + cache


def fly(env, loader, path, act, label=None, max_steps=None):
    """Fly one synthetic path. Returns drift (mm) per step, and visited obs + labels if `label`."""
    qpos, qvel, _ = make_path(PATHS[path], int(round(DURATION / DT)) + 10)
    loader.set_next_trajectory(qpos, qvel)
    timestep = env.reset()
    drift, obs, labels = [], [], []
    while timestep.step_type != 2 and (max_steps is None or len(drift) < max_steps):
        if label is not None:
            obs.append(np.concatenate([np.ravel(timestep.observation[k]) for k in OBS_KEYS]))
            labels.append(label(timestep.observation))
        timestep = env.step(act(timestep.observation))
        drift.append(10 * float(np.linalg.norm(timestep.observation["walker/ref_displacement"][0])))
    return np.array(drift), np.array(obs, np.float32), np.array(labels, np.float32)


def report(tag, drift):
    marks = {f"{int(ms)}ms": round(float(drift[int(ms / 1000 / DT) - 1]), 2)
             for ms in (20, 50, 100, 200, 500, 1000) if int(ms / 1000 / DT) <= len(drift)}
    print(f"{tag}: flew {len(drift) * DT * 1000:.0f} ms, drift {marks} mm", flush=True)
    return dict(flown_ms=len(drift) * DT * 1000, drift_at=marks, mean=float(drift.mean()), max=float(drift.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--round-steps", type=int, default=2500, help="cap per DAgger rollout (0.5 s)")
    parser.add_argument("--path", default="smooth", choices=list(PATHS))
    args = parser.parse_args()

    data = ROOT / "data"
    env, loader = make_env(str(find(data, "wing_pattern_fmech.npy")), DURATION)
    low, high = env.action_spec().minimum, env.action_spec().maximum
    expert = Policy(find(data, "flight", want_dir=True), env.action_spec())
    rec_obs, rec_act, _ = recorded_steps()
    rng = np.random.default_rng(0)
    pool_obs, pool_act = np.zeros((0, rec_obs.shape[1]), np.float32), np.zeros((0, rec_act.shape[1]), np.float32)

    out = ROOT / "results/exp_f"
    out.mkdir(parents=True, exist_ok=True)
    summary, ctrl = {"rounds": {}}, None
    for r in range(args.rounds + 1):
        n_pool = min(len(pool_obs), BUDGET // 2)
        base = rng.choice(len(rec_obs), BUDGET - n_pool, replace=False)
        take = rng.choice(len(pool_obs), n_pool, replace=False) if n_pool else np.array([], int)
        context = (np.vstack([rec_obs[base], pool_obs[take]]), np.vstack([rec_act[base], pool_act[take]]))
        del ctrl
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        ctrl = TabPFNController(None, "cuda", seed=r, context=context)
        act = lambda o: np.clip(ctrl(o), low, high)  # noqa: E731

        last = r == args.rounds
        t0 = time.time()
        drift, obs, labels = fly(env, loader, args.path, act, label=None if last else expert,
                                 max_steps=None if last else args.round_steps)
        summary["rounds"][r] = dict(context_recorded=int(len(base)), context_own_mistakes=int(n_pool),
                                    wall_s=round(time.time() - t0, 1),
                                    **report(f"round {r} ({n_pool} of {BUDGET} context steps from own flights)", drift))
        np.save(out / f"round{r}_{args.path}_drift.npy", drift)
        if not last:
            pool_obs, pool_act = np.vstack([pool_obs, obs]), np.vstack([pool_act, labels])

    # Generalisation: the final context on the paths it never practised on.
    summary["transfer"] = {}
    for path in PATHS:
        if path == args.path:
            continue
        drift, _, _ = fly(env, loader, path, act)
        np.save(out / f"final_{path}_drift.npy", drift)
        summary["transfer"][path] = report(f"final context on {path}", drift)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
