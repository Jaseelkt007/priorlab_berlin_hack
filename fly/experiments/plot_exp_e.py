"""Plot experiment E: distance to the reference over time, policy vs TabPFN, saccade shaded.

    uv run python experiments/plot_exp_e.py
Reads results/exp_e/<path>_<controller>.npz, writes results/exp_e/drift_vs_time.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results/exp_e"
PATHS = ["smooth", "smooth_then_saccade", "saccade_then_smooth"]


def main():
    # synthetic_paths.py writes one summary_<controller>.json per controller run.
    controllers = sorted(p.stem.removeprefix("summary_") for p in OUT.glob("summary_*.json"))
    fig, axes = plt.subplots(len(PATHS), 1, figsize=(10, 8), sharex=True)
    for ax, path in zip(axes, PATHS):
        yaw_ax = ax.twinx()
        for ctrl in controllers:
            f = OUT / f"{path}_{ctrl}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            ax.plot(d["t"] * 1000, d["drift_mm"], label=ctrl, lw=1.5)
            if ctrl == "policy":
                yaw_ax.fill_between(d["t"] * 1000, d["ref_yaw_rate_deg_s"], color="0.85", zorder=0)
        yaw_ax.set_ylabel("ref yaw rate (deg/s)", color="0.5")
        yaw_ax.tick_params(axis="y", colors="0.5")
        ax.set_zorder(yaw_ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_title(path.replace("_", " "))
        ax.set_ylabel("distance to ghost (mm)")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("Experiment E: does a sudden turn (grey: reference yaw rate) break the controller?")
    fig.tight_layout()
    fig.savefig(OUT / "drift_vs_time.png", dpi=120)
    print(f"wrote {OUT / 'drift_vs_time.png'}")


if __name__ == "__main__":
    main()
