"""
plot_mlp.py
Test-set figures of trained surrogates (train_mlp.py), on the test states of
both branches: parity of the first model, and the median relative error against
energy for every model given.

Usage:
    python surrogate/plot_mlp.py data/production_shell/nn_dataset.npz MODEL_DIR [MODEL_DIR ...]
Writes parity.png and error_vs_energy.png into the first MODEL_DIR.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from train_mlp import load_model, load_rows, predict

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots, in order
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
QUANTITIES = [("energy", 0, "U"), ("moment", 1, "M₁ (head)"),
              ("fx", 3, "fx = −∂U/∂d"), ("fy", 4, "fy = (∂U/∂φ₁ + ∂U/∂φ₂)/d")]


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def rel_error(pred, true):
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-2 * np.sqrt(np.mean(true ** 2)))


def main():
    dataset, dirs = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    x, y, exists, is_test, _ = load_rows(dataset)
    x, y = x[is_test & exists], y[is_test & exists]
    preds = [predict(load_model(d / "model.pt"), x[:, 0], x[:, 1], x[:, 2], x[:, 3]) for d in dirs]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), facecolor="white")
    for ax, (key, col, label) in zip(axes, QUANTITIES):
        style(ax)
        true, pred = y[:, col], preds[0][key]
        if key == "energy":
            hb = ax.hexbin(true, pred, gridsize=120, xscale="log", yscale="log", cmap=BLUES,
                           norm=LogNorm(), mincnt=1, linewidths=0)
        else:
            t = np.arcsinh(true)                    # bin on the symlog-like scale
            hb = ax.hexbin(t, np.arcsinh(pred), gridsize=120, cmap=BLUES, norm=LogNorm(),
                           mincnt=1, linewidths=0)
            ticks = np.array([-1e4, -1e3, -100, -10, 0, 10, 100])
            ticks = ticks[(ticks >= true.min() * 1.5) & (ticks <= true.max() * 1.5)]
            ax.set_xticks(np.arcsinh(ticks), [f"{v:g}" for v in ticks], rotation=45)
            ax.set_yticks(np.arcsinh(ticks), [f"{v:g}" for v in ticks])
            true, pred = t, np.arcsinh(pred)
        lo, hi = min(true.min(), pred.min()), max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("FEM (test)", color=MUTED, fontsize=9)
        ax.set_ylabel("MLP", color=MUTED, fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
    cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("test rows per hexagon", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.savefig(dirs[0] / "parity.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {dirs[0] / 'parity.png'}")

    edges = np.geomspace(y[:, 0].min(), y[:, 0].max() * 1.0001, 40)
    centers = np.sqrt(edges[:-1] * edges[1:])
    which = np.digitize(y[:, 0], edges) - 1
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), facecolor="white", sharex=True)
    for ax, (key, col, label) in zip(axes, QUANTITIES):
        style(ax)
        for n, (run, p) in enumerate(zip(dirs, preds)):
            err = rel_error(p[key], y[:, col])
            med = [np.median(err[which == b]) if np.any(which == b) else np.nan
                   for b in range(len(centers))]
            ax.plot(centers, med, color=SERIES[n], linewidth=2, label=run.name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("U (FEM)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("median relative error (test)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(dirs[0] / "error_vs_energy.png", dpi=160, bbox_inches="tight")
    print(f"Wrote {dirs[0] / 'error_vs_energy.png'}")


if __name__ == "__main__":
    main()
