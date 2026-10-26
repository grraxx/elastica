"""Figures of the energy-shell refinement scheme (solver/refine_shell.py) for the deck.

usage: python make_refinement_shell.py [data/production_shell]

Against the iso-energy scheme (data/production_iso) and the cell scheme
(data/production_nn). Writes shell_comb.png (the energies of the new points on fine
bins, and the points of a thin slab of d, for the iso and the shell scheme),
shell_points.png (one shell's points in 3D, and their nearest-neighbour spacing over
all bins) and shell_histograms.png (the three datasets on the deck's bins) to
presentation/figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "production_shell"
ISO = REPO / "data" / "production_iso"
CELLS = REPO / "data" / "production_nn"
OUT = REPO / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, SLATE, ORANGE, TEAL, GREY, LIGHT = "#16222E", "#2E4A5C", "#D9822B", "#2A9D8F", "#5C6B7A", "#EEF2F4"
PURPLE = "#7A5195"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.titlecolor": INK, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
BANDS = ((0, 1, "U < 1"), (1, 3, "1 ≤ U < 3"), (3, np.inf, "U ≥ 3"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, facecolor="white")      # fixed size: the deck's frames keep it
    plt.close(fig)
    print("wrote", name)


with open(RUN / "config.json") as f:
    config = json.load(f)
with np.load((RUN / config.get("branches_npz", "branches_grid.npz")).resolve()) as b:
    shape = tuple(int(n) for n in b["shape"])
    axes = [b[k] for k in ("d_vals", "phi1_vals", "phi2_vals")]
cells = tuple(n - 1 for n in shape)
steps = np.array([np.diff(a).mean() for a in axes])
origin = np.array([a[0] for a in axes])


def samples(run):
    """New states of a run: coordinates (d, phi1, phi2), energy and level/bin."""
    with np.load(run / "refined_samples.npz") as s:
        ijk = np.column_stack(np.unravel_index(s["cell"], cells)) + s["offsets"]
        return origin + ijk * steps, s["energy"], s["level"], s["level_values"], s["level_edges"]


iso_x, iso_u, _, iso_levels, _ = samples(ISO)
sh_x, sh_u, sh_bin, _, edges = samples(RUN)
datasets = {}
for name, run in (("cells", CELLS), ("iso", ISO), ("shell", RUN)):
    with open(run / "config.json") as f:
        path = run / json.load(f).get("dataset_npz", "nn_dataset.npz")
    with np.load(path) as ds:
        datasets[name] = ds["energy"][ds["exists"]]

# F1: the comb, in energy and in space -------------------------------------------------------------
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.4, 4.51), constrained_layout=True,
                                 gridspec_kw={"width_ratios": (1.5, 1, 1)})
lo, hi = 0.9, 1.2
bins = np.geomspace(lo, hi, 121)                       # 0.24 % wide
for u, colour, label in ((iso_u, ORANGE, "iso-energy: on the levels"),
                         (sh_u, PURPLE, "energy shells: between them")):
    a1.hist(u[(u >= lo) & (u < hi)], bins, color=colour, alpha=0.85 if colour == PURPLE else 1, lw=0,
            label=label)
for level in iso_levels[(iso_levels > lo) & (iso_levels < hi)]:
    a1.axvline(level, color=GREY, lw=0.5, ls=(0, (2, 3)), zorder=0)
a1.set_xlim(lo, hi)
a1.set_xlabel("bending energy U of the new states  (bins of 0.24 %)")
a1.set_ylabel("new states per bin")
a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
a1.legend(fontsize=8.5, frameon=False, loc="upper right")
a1.set_title("Energies: a comb, and a continuum", loc="left")
d0, half = 0.945, 0.0015                               # a slab of d, 0.3 grid steps thick
window = (0.0, 0.45, -0.45, 0.0)
for ax, x, colour, title in ((a2, iso_x, ORANGE, "iso-energy points"), (a3, sh_x, PURPLE, "energy-shell points")):
    sel = (np.abs(x[:, 0] - d0) < half) & (x[:, 1] >= window[0]) & (x[:, 1] <= window[1]) \
        & (x[:, 2] >= window[2]) & (x[:, 2] <= window[3])
    ax.plot(x[sel, 1], x[sel, 2], ".", ms=1.6, color=colour)
    ax.set_xlim(window[:2]); ax.set_ylim(window[2:]); ax.set_aspect("equal")
    ax.set_xlabel("φ₁")
    ax.set_title(f"{title}: {sel.sum():,}", loc="left", fontsize=9.5)
    ax.tick_params(labelsize=8)
a2.set_ylabel("φ₂")
a3.set_yticklabels([])
fig.text(0.79, 0.015, f"points with |d − {d0}| < {half} (φ₁ + φ₂ ≥ 0 side)", ha="center", fontsize=8.5,
         color=GREY)
save(fig, "shell_comb.png")
near = lambda u, lv: np.minimum(*(np.abs(u / lv[np.clip(np.searchsorted(lv, u) - s, 0, len(lv) - 1)] - 1)
                                  for s in (0, 1)))
print(f"within 0.5 % of an iso level value: iso {np.mean(near(iso_u, iso_levels) < 0.005):.1%}, "
      f"shell {np.mean(near(sh_u, iso_levels) < 0.005):.1%}")

# F2: one shell's points, and the spacing over all bins --------------------------------------------
with np.load(RUN / "shell_plan.npz") as p:
    plan_x = origin + (np.column_stack(np.unravel_index(p["cell"], cells)) + p["offsets"]) * steps
    plan_bin, spacing = p["level"], p["level_spacing"]
ratios = []
for k in range(len(spacing)):
    pts = (plan_x[plan_bin == k] - origin) / steps                 # grid steps
    if len(pts) > 1:
        ratios.append(cKDTree(pts).query(pts, k=2)[0][:, 1] / spacing[k])
ratios = np.concatenate(ratios)
print(f"nearest neighbour / r over {len(ratios):,} points: median {np.median(ratios):.3f}, "
      f"99.9 % {np.percentile(ratios, 99.9):.2f}, max {ratios.max():.2f}")
k1 = int(np.argmin(np.abs(np.sqrt(edges[:-1] * edges[1:]) - 1.0)))
xyz = plan_x[plan_bin == k1]
mirror = np.column_stack((xyz[:, 0], -xyz[:, 2], -xyz[:, 1]))
fig = plt.figure(figsize=(6.7, 3.72))
a1 = fig.add_axes([-0.03, -0.04, 0.64, 1.06], projection="3d")
rng = np.random.default_rng(0)
for points, colour in ((mirror, "#C9D3DB"), (xyz, PURPLE)):
    pick = rng.permutation(len(points))[:5000]
    a1.scatter(points[pick, 1], points[pick, 2], points[pick, 0], s=0.7, c=colour, depthshade=False, lw=0)
a1.set_zlim(xyz[:, 0].min() - 0.005, axes[0][-1])
a1.set_xticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"]); a1.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"])
a1.set_xlabel("φ₁", labelpad=-7); a1.set_ylabel("φ₂", labelpad=-7); a1.set_zlabel("d", labelpad=-5)
a1.tick_params(labelsize=7, pad=-3)
a1.view_init(elev=22, azim=-50)
for pane in (a1.xaxis, a1.yaxis, a1.zaxis):
    pane.set_pane_color((1, 1, 1, 0))
a1.set_title(f"{edges[k1]:.2f} ≤ U < {edges[k1 + 1]:.2f}: {len(xyz):,} points on φ₁ + φ₂ ≥ 0,\n"
             "the mirror gives the rest (grey); 5,000 of each drawn", fontsize=8.5, pad=0, y=0.93)
a2 = fig.add_axes([0.73, 0.17, 0.25, 0.66])
a2.hist(ratios, bins=np.arange(0.95, 1.65, 0.025), color=SLATE, lw=0)
a2.axvline(1, color=PURPLE, lw=1.4)
a2.text(1.03, 0.95, "r: no two\npoints closer", transform=a2.get_xaxis_transform(), color=PURPLE,
        fontsize=8, va="top")
a2.text(0.97, 0.55, f"99.9 % below\n{np.percentile(ratios, 99.9):.2f} r", transform=a2.transAxes, ha="right",
        color=INK, fontsize=8)
a2.set_xlabel("nearest neighbour / r", fontsize=9)
a2.set_ylabel("points", fontsize=9)
a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
a2.tick_params(labelsize=8)
a2.set_title(f"all {len(spacing)} bins:\nmedian {np.median(ratios):.2f} r", fontsize=9, loc="left")
save(fig, "shell_points.png")

# F3: the three datasets on the deck's bins ----------------------------------------------------------
edges25 = np.arange(0, 25.25, 0.25)
fig, axs = plt.subplots(1, 3, figsize=(10.4, 4.51), sharey=True, constrained_layout=True)
for ax, (name, colour, title) in zip(axs, (("cells", TEAL, "Cell scheme"), ("iso", ORANGE, "Iso-energy"),
                                            ("shell", PURPLE, "Energy shells"))):
    energy = datasets[name]
    counts, _ = np.histogram(energy, edges25)
    ax.bar(edges25[:-1], counts, width=0.25, align="edge", color=colour, lw=0)
    for lo_, _, _ in BANDS[1:]:
        ax.axvline(lo_, color=GREY, lw=0.8, ls=(0, (3, 3)))
    lines = [f"{label}: {np.count_nonzero((energy >= a) & (energy < b)) / len(energy):5.1%}"
             for a, b, label in BANDS]
    ax.text(0.97, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=INK, family="DejaVu Sans Mono", linespacing=1.5)
    ax.set_title(f"{title} ({len(energy) / 1e6:.2f} M states)", loc="left", fontsize=10)
    ax.set_xlim(0, 25)
    ax.set_xlabel("bending energy U  (bins of 0.25)")
    ax.grid(axis="y", color="#E3E7EA", lw=0.8)
    ax.set_axisbelow(True)
axs[0].set_ylabel("states per bin")
axs[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
save(fig, "shell_histograms.png")
for name, energy in datasets.items():
    print(name, f"{len(energy):,}", {label: round(np.count_nonzero((energy >= a) & (energy < b)) / len(energy), 4)
                                     for a, b, label in BANDS})
with np.load(RUN / "refined_samples.npz") as s:
    print(f"projections: mean {s['projections'].mean():.2f}; share 0 steps {np.mean(s['projections'] == 0):.1%}")
