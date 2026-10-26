"""Figures of the neural-network surrogate section of the deck (surrogate/train_mlp.py),
written for readers who have not met neural networks before.

usage: python make_surrogate.py [data/production_shell]      (in the elastica-ml env: needs torch)

Against the two trained runs RUN/mlp (the force in the fit) and RUN/mlp_no_fx (without it),
the dataset RUN/nn_dataset.npz and the solver's grid data/production_nn/branches_grid.npz.
Writes to presentation/figures/:
  nn_speed.png      time per state, solver against network
  nn_unit.png       what one unit of a network computes, and the smooth bend it applies
  nn_io.png         what goes in and what comes out
  nn_slopes.png     U(d) and U(phi1) with the tangents whose slopes are -fx and -M1
  nn_data.png       the d = 0.90 layer: training states, test states, points without a state
  nn_log.png        why log U: the spread of U, the weight of a 1 % error, U = e^z > 0
  nn_asinh.png      why asinh M: asinh against its limits, the spread of M, what an output error means
  nn_training.png   the mismatch on held-back rows while fitting
  nn_rel.png        relative error against energy
  nn_abs.png        absolute error against energy, against the size of the quantity
  nn_map.png        where the absolute errors are at d = 0.90
  nn_exist.png      where each shape exists at d = 0.90, network against solver
  nn_force.png      why the force is in the fit
  nn_verdict.png    the errors against the solver's own accuracy
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

REPO = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "production_shell"
GRID = REPO / "data" / "production_nn" / "branches_grid.npz"           # the solver's bent-up branch
OUT = REPO / "presentation" / "figures"
sys.path.insert(0, str(REPO / "surrogate"))
from train_mlp import load_model, load_rows, predict  # noqa: E402

INK, SLATE, ORANGE, TEAL, GREY, LIGHT = "#16222E", "#2E4A5C", "#D9822B", "#2A9D8F", "#5C6B7A", "#EEF2F4"
PURPLE, RED, PALE = "#7A5195", "#C8553D", "#C9D3DB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "legend.fontsize": 8.5, "mathtext.fontset": "dejavusans"})
GRIDLINE = {"color": "#E3E7EA", "lw": 0.8}
A, B = (6.4, 4.5), (11.0, 3.3)                     # figure frames: beside the text, full width
QUANTITIES = (("energy", 0, "energy U"), ("moment", 1, "moment M₁"), ("fx", 3, "force fx"),
              ("fy", 4, "force fy"))
BVP = (1.2e-4, 4.8e-4)          # refine.py check: solver against an independent BVP solve, median and max


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, facecolor="white")      # fixed size: the slide frames keep it
    plt.close(fig)
    print("wrote", name)


def rel_error(pred, true):
    """|error| / |true|, with a floor of 1 % of the RMS where the target crosses zero."""
    return np.abs(pred - true) / np.maximum(np.abs(true), 1e-2 * np.sqrt(np.mean(true ** 2)))


def binned(values, which, n, q):
    return np.array([np.percentile(values[which == b], q) if np.count_nonzero(which == b) > 50 else np.nan
                     for b in range(n)])


def canvas(size):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, face=LIGHT, edge="none", colour=INK, size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=face, edgecolor=edge, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=colour, fontsize=size,
            weight=weight, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, colour=GREY, ls="-", lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11, color=colour,
                                 lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- data
x_all, y_all, exists_all, test_all, _ = load_rows(RUN / "nn_dataset.npz")
keep = test_all & exists_all
x, y = x_all[keep], y_all[keep]
models = {name: load_model(RUN / name / "model.pt") for name in ("mlp", "mlp_no_fx")}
preds = {name: predict(m, x[:, 0], x[:, 1], x[:, 2], x[:, 3]) for name, m in models.items()}
metrics = {name: json.loads((RUN / name / "metrics.json").read_text()) for name in models}
model = models["mlp"]
p = preds["mlp"]
edges = np.geomspace(y[:, 0].min(), y[:, 0].max() * 1.0001, 36)
centers = np.sqrt(edges[:-1] * edges[1:])
which = np.digitize(y[:, 0], edges) - 1
with np.load(GRID) as g:
    shape = tuple(int(n) for n in g["shape"])
    d_vals, phi = g["d_vals"], g["phi1_vals"]
    grid = {k: g[k].reshape(shape) for k in ("exists", "energy", "fx", "moment")}
layer = int(np.argmin(np.abs(d_vals - 0.90)))
P1, P2 = np.meshgrid(phi, phi, indexing="ij")

# ---------------------------------------------------------------- 1: speed
fig, ax = plt.subplots(figsize=A, constrained_layout=True)
rows = (("solver: Newton on 203 unknowns,\none CPU core", 17.9e-3, SLATE),
        ("network on a CPU", 7.6e-6, ORANGE), ("network on a GPU (RTX A6000)", 0.46e-6, ORANGE))
for k, (label, t, colour) in enumerate(rows):
    ax.barh(k, t, color=colour, height=0.55)
    million = t * 1e6
    per_million = f"{million / 3600:.0f} h" if million > 3600 else f"{million:.2g} s"
    per_state = f"{t * 1e3:.3g} ms" if t > 1e-4 else f"{t * 1e6:.2g} µs"
    ax.text(t * 1.4, k, f"{per_state} per state\n{per_million} per million", va="center", fontsize=9,
            color=INK, linespacing=1.3)
ax.set_yticks(range(3), [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(1e-7, 30)
ax.set_xlabel("seconds per state (log scale)")
ax.grid(axis="x", **GRIDLINE)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=INK)
ax.set_title("Time to get U, M₁, M₂, fx, fy at one clamp setting")
save(fig, "nn_speed.png")

# ---------------------------------------------------------------- 2: one unit
fig = plt.figure(figsize=A)
ax = fig.add_axes((0, 0.38, 1, 0.62))
ax.set_xlim(0, 6.4)
ax.set_ylim(0, 2.8)
ax.axis("off")
columns = ((0.55, ("d", "φ₁", "φ₂")), (2.0, ("", "", "", "")), (3.45, ("", "", "", "")), (4.9, ("U",)))
pos = [[(xc, yy) for yy in (np.linspace(2.35, 0.35, 4) if len(labels) == 4 else
                            np.linspace(2.35, 0.35, len(labels) + 2)[1:-1])] for xc, labels in columns]
lit = pos[1][1]                                      # the unit spelled out below
for a, b in zip(pos[:-1], pos[1:]):
    for p0 in a:
        for p1 in b:
            hl = p1 == lit and a is pos[0]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ORANGE if hl else PALE, lw=1.8 if hl else 0.8,
                    zorder=1)
for kk, column in enumerate(pos):
    for jj, (xc, yy) in enumerate(column):
        inner = 0 < kk < len(pos) - 1
        ax.add_patch(plt.Circle((xc, yy), 0.17, color=ORANGE if (xc, yy) == lit else (SLATE if inner else "white"),
                                ec=SLATE, lw=1.2, zorder=2))
        if columns[kk][1][jj]:
            ax.text(xc, yy, columns[kk][1][jj], ha="center", va="center", fontsize=10, color=INK, zorder=3)
for xc, text in ((0.55, "inputs"), (2.0, "layer 1"), (3.45, "layer 2, … 6"), (4.9, "output")):
    ax.text(xc, 2.65, text, ha="center", color=GREY, fontsize=9)
box(ax, 5.35, 1.15, 0.95, 1.25, "one unit:\na weighted\nsum, then\na bend", face="white", edge=ORANGE,
    size=8.5)
ax2 = fig.add_axes((0.11, 0.1, 0.48, 0.24))
u = np.linspace(-5, 5, 400)
ax2.plot(u, u / (1 + np.exp(-u)), color=ORANGE, lw=2)
ax2.axhline(0, color=GREY, lw=0.6)
ax2.axvline(0, color=GREY, lw=0.6)
ax2.set_xlabel("u = w₁d + w₂φ₁ + w₃φ₂ + c", fontsize=8.5)
ax2.set_ylabel("output of the unit", fontsize=8.5)
ax2.set_title("the bend: SiLU(u) = u / (1 + e⁻ᵘ)", fontsize=9)
ax2.grid(**GRIDLINE)
fig.text(0.65, 0.22, "w₁, w₂, w₃ and c are numbers\nset by fitting. The two networks\nhave 662 thousand of them.",
         fontsize=8.5, color=INK, va="center", linespacing=1.45)
save(fig, "nn_unit.png")

# ---------------------------------------------------------------- 3: in and out
fig, ax = canvas(A)
ax.text(0.2, 4.15, "in", color=GREY, fontsize=9)
for kk, (label, note) in enumerate((("d", "end distance"), ("φ₁, φ₂", "clamp angles"),
                                    ("b = +1 or −1", "bent up or bent down"))):
    box(ax, 0.2, 3.3 - kk * 0.95, 1.35, 0.6, label, face="white", edge=SLATE, size=10)
    ax.text(0.2, 3.2 - kk * 0.95, note, color=GREY, fontsize=8, va="top")
box(ax, 2.25, 2.55, 1.6, 1.15, "energy\nnetwork", face=SLATE, colour="white", size=10, weight="bold")
box(ax, 2.25, 0.55, 1.6, 1.15, "existence\nnetwork", face=TEAL, colour="white", size=10, weight="bold")
for kk in range(3):
    for yy in (3.12, 1.12):
        arrow(ax, 1.55, 3.6 - kk * 0.95, 2.25, yy, lw=0.9)
ax.text(4.35, 4.15, "out", color=GREY, fontsize=9)
for kk, label in enumerate(("U", "M₁", "M₂")):
    box(ax, 4.35, 3.5 - kk * 0.6, 0.75, 0.45, label, face=ORANGE, colour="white", size=10, weight="bold")
    arrow(ax, 3.85, 3.12, 4.35, 3.72 - kk * 0.6, lw=0.9)
box(ax, 4.35, 0.85, 1.85, 0.55, "P(this shape exists)", face=TEAL, colour="white", size=9, weight="bold")
arrow(ax, 3.85, 1.12, 4.35, 1.12, lw=0.9)
box(ax, 5.35, 2.0, 0.85, 0.75, "fx, fy", face="white", edge=PURPLE, colour=PURPLE, size=10, weight="bold")
arrow(ax, 4.75, 2.3, 5.35, 2.37, colour=PURPLE, ls="--")
ax.text(5.35, 1.85, "slopes of U\n(slide 55)", color=PURPLE, fontsize=8, va="top", linespacing=1.3)
save(fig, "nn_io.png")

# ---------------------------------------------------------------- 4: forces are slopes of U
j, k = int(np.argmin(np.abs(phi - 0.3))), int(np.argmin(np.abs(phi + 0.2)))
i = layer
fig, axes = plt.subplots(2, 1, figsize=A, constrained_layout=True)
ax = axes[0]
dd = np.linspace(d_vals[0], d_vals[-1], 300)
net = predict(model, dd, phi[j], phi[k], 1)
ax.plot(d_vals, grid["energy"][:, j, k], "o", ms=4, color=SLATE, label="solver, grid points")
ax.plot(dd, net["energy"], color=ORANGE, lw=1.6, label="network")
slope, span = -grid["fx"][i, j, k], np.array([-0.05, 0.05])
ax.plot(d_vals[i] + span, grid["energy"][i, j, k] + slope * span, color=PURPLE, lw=2.2,
        label=f"tangent at d = {d_vals[i]:.2f}: slope −fx = {slope:.1f}".replace("= -", "= −"))
ax.set_xlabel("end distance d")
ax.set_ylabel("U")
ax.set_title(f"U along d, at φ₁ = {phi[j]:.2f}, φ₂ = {phi[k]:.2f}, bent up".replace("-", "−"))
ax.legend(loc="upper right")
ax.grid(**GRIDLINE)
ax = axes[1]
pp = np.linspace(-0.4, 1.2, 300)
net = predict(model, d_vals[i], pp, phi[k], 1)
sel = (phi > -0.4) & (phi < 1.2) & grid["exists"][i, :, k]
ax.plot(phi[sel][::4], grid["energy"][i, sel, k][::4], "o", ms=4, color=SLATE, label="solver, grid points")
ax.plot(pp, np.where(net["exists_prob"] > 0.5, net["energy"], np.nan), color=ORANGE, lw=1.6, label="network")
slope, span = -grid["moment"][i, j, k], np.array([-0.25, 0.25])
ax.plot(phi[j] + span, grid["energy"][i, j, k] + slope * span, color=PURPLE, lw=2.2,
        label=f"tangent at φ₁ = {phi[j]:.2f}: slope −M₁ = {slope:.2f}".replace("= -", "= −"))
ax.set_xlabel("clamp angle φ₁")
ax.set_ylabel("U")
ax.set_title(f"U along φ₁, at d = {d_vals[i]:.2f}, φ₂ = {phi[k]:.2f}, bent up".replace("-", "−"))
ax.legend(loc="upper right")
ax.grid(**GRIDLINE)
save(fig, "nn_slopes.png")

# ---------------------------------------------------------------- 5: the data and the test set
step = phi[1] - phi[0]
near = (x_all[:, 3] == 1) & (x_all[:, 0] >= d_vals[layer]) & (x_all[:, 0] < d_vals[layer] + 0.0025)
CLASSES = ((near & exists_all & ~test_all, PALE, "fitted to (80 %)", 1.0),
           (near & exists_all & test_all, ORANGE, "held back for testing (20 %)", 1.0),
           (near & ~exists_all, "#8A96A3", "no bent-up state: an existence example", 1.0))
window = (0.25, 0.4, -0.45, -0.3)
fig, axes = plt.subplots(1, 2, figsize=A, constrained_layout=True)
for ax, zoom in zip(axes, (False, True)):
    for rows_, colour, label, size in (CLASSES if zoom else (CLASSES[1], CLASSES[0], CLASSES[2])):
        ax.scatter(x_all[rows_, 1], x_all[rows_, 2], s=size * (6 if zoom else 1), color=colour,
                   linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("φ₁")
    if zoom:
        held = near & test_all & (x_all[:, 1] > window[0] - step) & (x_all[:, 1] < window[1] + step) \
            & (x_all[:, 2] > window[2] - step) & (x_all[:, 2] < window[3] + step)
        cells = {(int(np.floor((a - phi[0]) / step + 1e-9)), int(np.floor((b - phi[0]) / step + 1e-9)))
                 for a, b in x_all[held, 1:3]}
        for ci, cj in cells:                        # every held-back cell, shaded whole
            ax.add_patch(plt.Rectangle((phi[ci], phi[cj]), step, step, color="#F6E3CF", lw=0, zorder=0))
        ax.set_xlim(window[:2])
        ax.set_ylim(window[2:])
        for v in phi[(phi > window[0] - 0.01) & (phi < window[1] + 0.01)]:
            ax.axvline(v, color="#E3E7EA", lw=0.5, zorder=0)
        for v in phi[(phi > window[2] - 0.01) & (phi < window[3] + 0.01)]:
            ax.axhline(v, color="#E3E7EA", lw=0.5, zorder=0)
        ax.set_title("close up: whole grid cells\nare held back", fontsize=9.5)
    else:
        ax.set_xlim(-np.pi / 2, np.pi / 2)
        ax.set_ylim(-np.pi / 2, np.pi / 2)
        ax.add_patch(plt.Rectangle((window[0], window[2]), window[1] - window[0], window[3] - window[2],
                                   fill=False, ec=INK, lw=1))
        ax.set_title(f"bent-up points with\n{d_vals[layer]:.2f} ≤ d < {d_vals[layer] + 0.0025:.4f}", fontsize=9.5)
        ax.set_ylabel("φ₂")
handles = [Line2D([], [], marker="o", ls="", color=c, ms=5, label=l) for _, c, l, _ in CLASSES]
fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8)
save(fig, "nn_data.png")

# ---------------------------------------------------------------- 6: why log U
fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.75), constrained_layout=True)
ax = axes[0]
u_all = y_all[exists_all, 0]
ax.hist(u_all, bins=np.geomspace(u_all.min(), u_all.max(), 60), color=SLATE)
ax.set_xscale("log")
ax.set_xlabel("energy U (log scale)")
ax.set_ylabel("states")
ax.set_title("U spans three decades, 0.1 to 142")
ax.grid(axis="y", **GRIDLINE)
ax = axes[1]
uu = np.geomspace(0.1, 142, 200)
ax.plot(uu, uu ** 2, color=SLATE, lw=2, label="fitting U: (0.01 U)², grows as U²")
ax.plot(uu, np.ones_like(uu), color=ORANGE, lw=2, label="fitting ln U: 0.01², the same at every U")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(3e-3, 1e5)
ax.set_xlabel("energy U")
ax.set_ylabel("weight, relative to U = 1")
ax.set_title("weight of a 1 % miss in the squared error")
ax.legend(loc="upper left", fontsize=8)
ax.text(1.2, 3e-2, f"U = 142 weighs {142 ** 2 / 0.1 ** 2:.0e} × U = 0.1".replace("e+06", " × 10⁶"),
        fontsize=8, color=SLATE)
ax.grid(**GRIDLINE)
ax = axes[2]
zz = np.linspace(-3, 5.2, 200)
ax.plot(zz, np.exp(zz), color=ORANGE, lw=2)
ax.set_yscale("log")
ax.set_xlabel("network output z, any real number")
ax.set_ylabel("U = eᶻ")
ax.set_title("U = eᶻ is positive for every output")
ax.grid(**GRIDLINE)
save(fig, "nn_log.png")

# ---------------------------------------------------------------- 7: why asinh M
fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.75), constrained_layout=True)
ax = axes[0]
mm = np.linspace(-12, 12, 400)
ax.plot(mm, mm, color=GREY, lw=1, ls=(0, (3, 2)), label="M, near 0")
for sgn in (1, -1):
    side = mm * sgn > 1.5
    ax.plot(mm[side], sgn * np.log(2 * np.abs(mm[side])), color=PURPLE, lw=1.2, ls=(0, (1, 1.5)),
            label="± ln 2|M|, far out" if sgn == 1 else None)
ax.plot(mm, np.arcsinh(mm), color=ORANGE, lw=2.2, label="asinh M")
ax.set_ylim(-4, 4)
ax.set_xlabel("moment M")
ax.set_title("asinh: linear near 0, a log far out")
ax.legend(loc="lower right", fontsize=8)
ax.grid(**GRIDLINE)
ax = axes[1]
m1 = y_all[exists_all, 1]
ax.hist(np.arcsinh(m1), bins=80, color=SLATE)
ticks = np.array([-100, -10, -1, 0, 1, 10, 100])
ax.set_xticks(np.arcsinh(ticks), [f"{t:g}".replace("-", "−") for t in ticks])
ax.set_xlabel("moment M₁ (asinh scale)")
ax.set_ylabel("states")
ax.set_title("M₁ changes sign and reaches ±168")
ax.grid(axis="y", **GRIDLINE)
ax = axes[2]
mabs = np.geomspace(1e-3, 200, 200)
ax.plot(mabs, 1e-3 * np.sqrt(1 + mabs ** 2), color=ORANGE, lw=2.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.axvspan(1e-3, 1, color=LIGHT, zorder=0)
ax.text(2e-3, 6e-2, "|M| < 1: absolute,\nδM ≈ 0.001", fontsize=8.5, color=INK, va="top")
ax.text(4, 2.5e-3, "|M| > 1: relative,\nδM ≈ 0.001 |M|", fontsize=8.5, color=INK, va="top")
ax.set_xlabel("|M|")
ax.set_ylabel("error δM")
ax.set_title("what an output miss of 0.001 means")
ax.grid(**GRIDLINE)
save(fig, "nn_asinh.png")

# ---------------------------------------------------------------- 8: fitting converges
TERMS = (("energy", "energy (ln U)", ORANGE), ("moments", "moments (asinh M)", TEAL),
         ("grad_moments", "slopes of U against the moments", SLATE),
         ("grad_force", "slope of U against the force", PURPLE), ("exists", "existence", RED))
rows = []
for line in (RUN / "mlp" / "train.log").read_text().splitlines():
    if line.startswith("epoch"):
        test = line.split("| test(5%)")[1]
        rows.append([int(line.split()[1])] + [float(re.search(rf"{key} (\S+)", test).group(1))
                                               for key, _, _ in TERMS])
rows = np.array(rows)
fig, ax = plt.subplots(figsize=A, constrained_layout=True)
for kk, (key, label, colour) in enumerate(TERMS):
    ax.plot(rows[:, 0], rows[:, kk + 1], color=colour, lw=1.8, label=label)
ax.set_yscale("log")
ax.set_xlabel("passes through the training data")
ax.set_ylabel("mean squared mismatch, held-back rows")
ax.set_title("Each term of the loss, on rows the fit never used")
ax.legend(loc="lower left")
ax.grid(axis="y", **GRIDLINE)
save(fig, "nn_training.png")

# ---------------------------------------------------------------- 9, 10: errors against energy
fig, axes = plt.subplots(1, 4, figsize=B, sharex=True, sharey=True, constrained_layout=True)
for ax, (key, col, label) in zip(axes, QUANTITIES):
    err = rel_error(p[key], y[:, col])
    ax.fill_between(centers, binned(err, which, len(centers), 50), binned(err, which, len(centers), 99),
                    color=ORANGE, alpha=0.18, lw=0, label="median to 99th percentile")
    ax.plot(centers, binned(err, which, len(centers), 50), color=ORANGE, lw=2, label="median")
    for level, text in ((1e-3, "0.1 %"), (1e-4, "0.01 %")):
        ax.axhline(level, color=GREY, lw=0.8, ls=(0, (3, 2)))
        ax.text(0.105, level * 1.15, text, fontsize=8, color=GREY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(2e-6, 2e-1)
    ax.set_title(label)
    ax.set_xlabel("energy U")
    ax.grid(**GRIDLINE)
axes[0].set_ylabel("relative error")
axes[0].legend(loc="upper left", fontsize=8)
save(fig, "nn_rel.png")

fig, axes = plt.subplots(1, 4, figsize=B, sharex=True, constrained_layout=True)
for ax, (key, col, label) in zip(axes, QUANTITIES):
    err = np.abs(p[key] - y[:, col])
    ax.plot(centers, binned(np.abs(y[:, col]), which, len(centers), 50), color=SLATE, lw=1.6,
            ls=(0, (1, 1.2)), label="the quantity itself (median |value|)")
    ax.fill_between(centers, binned(err, which, len(centers), 50), binned(err, which, len(centers), 99),
                    color=ORANGE, alpha=0.18, lw=0, label="error: median to 99th percentile")
    ax.plot(centers, binned(err, which, len(centers), 50), color=ORANGE, lw=2, label="error: median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(label)
    ax.set_xlabel("energy U")
    ax.grid(**GRIDLINE)
axes[0].set_ylabel("|network − solver|  (EI = L = 1)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8.5)
save(fig, "nn_abs.png")

# ---------------------------------------------------------------- 11: where the errors are
HEAT = LinearSegmentedColormap.from_list("heat", ["#F6E3CF", ORANGE, "#7A3E0B"])
near_t = (x[:, 3] == 1) & (np.abs(x[:, 0] - d_vals[layer]) < 0.0025)
fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.4), constrained_layout=True)
for ax, (key, col, label) in zip(axes, QUANTITIES):
    err = np.abs(p[key][near_t] - y[near_t, col])
    ax.contour(P1, P2, grid["exists"][layer].astype(float), levels=(0.5,), colors=GREY, linewidths=0.8)
    hb = ax.hexbin(x[near_t, 1], x[near_t, 2], C=err, reduce_C_function=np.median, gridsize=26,
                   extent=(-np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2), mincnt=3, cmap=HEAT,
                   norm=LogNorm(), linewidths=0.2, edgecolors="white")
    ax.set_aspect("equal")
    ax.set_xlim(-np.pi / 2, np.pi / 2)
    ax.set_ylim(-np.pi / 2, np.pi / 2)
    ax.set_title(label)
    ax.set_xlabel("φ₁")
    cb = fig.colorbar(hb, ax=ax, orientation="horizontal", shrink=0.9, pad=0.02, aspect=25)
    cb.set_label("median |error| per hexagon", color=GREY, fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
axes[0].set_ylabel("φ₂")
save(fig, "nn_map.png")

# ---------------------------------------------------------------- 12: existence
truth = {"up": grid["exists"][layer], "down": grid["exists"][layer, ::-1, ::-1]}   # down: the reflection
truth["both"] = truth["up"] & truth["down"]
prob = {name: predict(model, d_vals[layer], P1.ravel(), P2.ravel(), b)["exists_prob"].reshape(P1.shape)
        for name, b in (("up", 1), ("down", -1))}
prob["both"] = np.minimum(prob["up"], prob["down"])
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
titles = {"up": "bent up exists", "down": "bent down exists", "both": "both exist: two stable shapes"}
for ax, key in zip(axes, ("up", "down", "both")):
    ax.contourf(P1, P2, prob[key], levels=(0.5, 1.01), colors=(PALE,))
    ax.contour(P1, P2, truth[key].astype(float), levels=(0.5,), colors=ORANGE, linewidths=1.8)
    ax.contour(P1, P2, prob[key], levels=(0.5,), colors=INK, linewidths=1.0, linestyles=[(0, (3, 2))])
    wrong = (prob[key] > 0.5) != truth[key]
    ax.plot(P1[wrong], P2[wrong], "o", ms=2, color=RED)
    ax.set_aspect("equal")
    ax.set_title(f"{titles[key]}\n{wrong.sum()} of {wrong.size:,} grid points disagree", fontsize=9.5)
    ax.set_xlabel("φ₁")
axes[0].set_ylabel("φ₂")
handles = [Patch(color=PALE, label="network: probability > 0.5"),
           Line2D([], [], color=ORANGE, lw=1.8, label="solver: where the shape ends"),
           Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)), label="network: probability = 0.5"),
           Line2D([], [], color=RED, marker="o", ls="", ms=3, label="grid point where they disagree")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=8.5)
save(fig, "nn_exist.png")

# ---------------------------------------------------------------- 13: why the force is in the fit
fig, axes = plt.subplots(2, 1, figsize=A, sharex=True, constrained_layout=True)
ax = axes[0]
ax.plot(centers, binned(np.abs(y[:, 3]), which, len(centers), 50), color=SLATE, lw=2, label="median")
ax.plot(centers, binned(np.abs(y[:, 3]), which, len(centers), 99), color=SLATE, lw=1, ls=(0, (3, 2)),
        label="99th percentile")
ax.set_yscale("log")
ax.set_ylabel("|fx|, solver")
ax.set_title("Above U ≈ 20 the tension, the slope −∂U/∂d, grows to 10⁴")
ax.legend(loc="upper left")
ax.grid(**GRIDLINE)
ax = axes[1]
for name, colour, label in (("mlp_no_fx", SLATE, "fitted to U and the moments only"),
                            ("mlp", ORANGE, "the force in the fit too")):
    err = rel_error(preds[name]["fx"], y[:, 3])
    ax.plot(centers, binned(err, which, len(centers), 50), color=colour, lw=2, label=label)
    ax.plot(centers, binned(err, which, len(centers), 99), color=colour, lw=1, ls=(0, (3, 2)))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("energy U")
ax.set_ylabel("relative error of fx")
ax.set_title("fx = −∂U/∂d from each fit: median (solid), 99th percentile (dashed)", fontsize=9)
ax.set_ylim(3e-5, 2e-1)
ax.legend(loc="upper left", fontsize=8)
ax.grid(**GRIDLINE)
save(fig, "nn_force.png")

# ---------------------------------------------------------------- 14: verdict
names = (("energy", "energy U"), ("moment (output)", "moment M₁"), ("moment_right (output)", "moment M₂"),
         ("fx (autodiff)", "force fx"), ("fy (autodiff)", "force fy"))
fig, ax = plt.subplots(figsize=(11.0, 2.9), constrained_layout=True)
ax.axvspan(*BVP, color=LIGHT, zorder=0)
ax.text(np.sqrt(BVP[0] * BVP[1]), -0.75, "the solver against an independent\nBVP solver: median to max",
        ha="center", va="center", fontsize=8.5, color=GREY)
for kk, (key, label) in enumerate(names):
    test = metrics["mlp"]["test"]["fields"][key]["both [0, inf)"]
    train = metrics["mlp"]["train"]["fields"][key]["both [0, inf)"]
    ax.plot([test["median_rel"], test["p99_rel"]], [kk, kk], color=ORANGE, lw=2, alpha=0.4)
    ax.plot(test["median_rel"], kk, "o", color=ORANGE, ms=8,
            label="held-back states: median" if kk == 0 else None)
    ax.plot(test["p99_rel"], kk, "|", color=ORANGE, ms=12, mew=2,
            label="held-back states: 99th percentile" if kk == 0 else None)
    ax.plot(train["median_rel"], kk, "o", mfc="none", mec=INK, ms=13, mew=1.2,
            label="fitted states: median" if kk == 0 else None)
ax.set_yticks(range(len(names)), [n[1] for n in names])
ax.set_ylim(len(names) - 0.4, -1.1)
ax.set_xscale("log")
ax.set_xlim(3e-6, 6e-2)
ax.set_xlabel("relative error")
ax.tick_params(axis="y", length=0, labelcolor=INK)
ax.grid(axis="x", **GRIDLINE)
fig.legend(loc="outside upper right", ncol=3, fontsize=8.5)
save(fig, "nn_verdict.png")
