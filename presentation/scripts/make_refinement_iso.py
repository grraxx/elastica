"""Figures of the iso-energy refinement scheme (solver/refine_iso.py) for the deck.

usage: python make_refinement_iso.py [data/production_iso]

The run's plan (iso_plan.npz: points on the levels before solving), its solves
(refined_samples.npz) and its dataset (nn_dataset.npz), against the grid's bent-up
branch (branches_grid.npz) and the dataset of the cell scheme (data/production_nn).

Writes iso_target.png (rows per unit U: the grid, the target and the two datasets),
iso_surface.png (one level's points, and their nearest-neighbour spacing over all
levels), iso_projection.png (how close the solves land on their level) and
iso_histograms.png (the cell scheme and the iso scheme on the deck's bins) to
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
sys.path.insert(0, str(REPO / "solver"))
from refine_iso import Target, settings  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "production_iso"
OUT = REPO / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, SLATE, ORANGE, TEAL, GREY, LIGHT = "#16222E", "#2E4A5C", "#D9822B", "#2A9D8F", "#5C6B7A", "#EEF2F4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.titlecolor": INK, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
BANDS = ((0, 1, "U < 1"), (1, 3, "1 ≤ U < 3"), (3, np.inf, "U ≥ 3"))


def save(fig, name):
    # A fixed figure size (no tight bounding box): the deck's picture frames keep its aspect.
    fig.savefig(OUT / name, dpi=220, facecolor="white")
    plt.close(fig)
    print("wrote", name)


class Run:                                    # what refine_iso.settings reads
    def __init__(self, config):
        self.config = config


with open(RUN / "config.json") as f:
    config = json.load(f)
cfg = settings(Run(config))
branches = (RUN / config.get("branches_npz", "branches_grid.npz")).resolve()
with np.load(branches) as b:
    shape = tuple(int(n) for n in b["shape"])
    axes = [b[k] for k in ("d_vals", "phi1_vals", "phi2_vals")]
    grid_up = b["energy"][b["exists"]]
grid_rows = np.concatenate([grid_up, grid_up])             # both branches: the reflection
target = Target(grid_rows, cfg)
with np.load(RUN / "iso_plan.npz") as p:
    plan = {k: p[k] for k in ("cell", "offsets", "level", "level_values", "level_spacing",
                              "level_target", "level_placed")}
with np.load(RUN / "refined_samples.npz") as s:
    solved = {k: s[k] for k in ("energy", "level", "projections")}
with np.load(RUN / "nn_dataset.npz") as ds:
    have = ds["exists"]
    iso_all = ds["energy"][have]
    iso_new = ds["energy"][have & (ds["source"] == 6)]
cells_run = REPO / "data" / "production_nn"
with open(cells_run / "config.json") as f:
    cells_dataset = cells_run / json.load(f).get("dataset_npz", "nn_dataset.npz")
with np.load(cells_dataset) as ds:
    cells_all = ds["energy"][ds["exists"]]

# F1: rows per unit energy: the grid, the target and the two datasets ---------------------------
# Bins of two level intervals, then the same ratio up to the largest energy, so every bin
# below U_p holds two whole levels.
edges = target.edges[::2]
ratio = edges[1] / edges[0]
edges = np.append(edges, edges[-1] * ratio ** np.arange(1, 1 + int(np.ceil(
    np.log(grid_rows.max() / edges[-1]) / np.log(ratio)))))
width = np.diff(edges)
per_u = {k: np.histogram(v, edges)[0] / width for k, v in
         (("grid", grid_rows), ("cells", cells_all), ("iso", iso_all))}
fig, ax = plt.subplots(figsize=(6.7, 3.72), constrained_layout=True)
ax.axvspan(target.u_min, target.u_max, color=LIGHT, zorder=0)
ax.stairs(per_u["grid"], edges, color=SLATE, lw=1.8, label="grid, both branches (5.9 M)")
ax.stairs(per_u["cells"], edges, color=TEAL, lw=1.2, ls=(0, (3, 2)),
          label=f"cell scheme dataset ({len(cells_all) / 1e6:.1f} M)")
ax.stairs(per_u["iso"], edges, color=ORANGE, lw=2.2, label=f"iso scheme dataset ({len(iso_all) / 1e6:.1f} M)")
u = np.geomspace(target.u_min, target.u_max, 50)
ax.plot(u, target.density(u), color=INK, lw=1.1, ls=(0, (5, 3)),
        label=f"target  H(U) = {target.c / 1e6:.2f} M / U")
ax.text(np.sqrt(target.u_min * target.u_max), 3.4e7,
        f"{len(target.levels)} levels, {cfg['level_step']:.0%} apart", ha="center", color=GREY, fontsize=8.5)
ax.annotate(f"grid's peak, $U_p$ = {target.u_peak:.1f}", (target.u_peak, target.density(target.u_peak)),
            (9, 3.5e6), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(edges[0], 150); ax.set_ylim(1e3, 6e7)
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1", "0.3", "1", "3", "10", "30", "100"])
ax.set_xlabel("bending energy U")
ax.set_ylabel("dataset rows per unit U")
ax.legend(fontsize=8, frameon=False, loc="lower left")
save(fig, "iso_target.png")
low = (edges[:-1] >= 0.12) & (edges[1:] <= target.u_max)
fit = np.exp(np.mean(np.log(per_u["iso"][low] * np.sqrt(edges[:-1] * edges[1:])[low])))
print(f"target C = {target.c:,.0f}; dataset rows per unit U x U on 0.12-{target.u_max:.2f}: "
      f"geometric mean {fit:,.0f}")

# F2: one level's points, and the spacing over all levels -------------------------------------------
cells = tuple(n - 1 for n in shape)
ijk = np.column_stack(np.unravel_index(plan["cell"], cells)) + plan["offsets"]   # grid steps
steps = np.array([np.diff(a).mean() for a in axes])
ratios = []
for k in range(len(plan["level_values"])):
    pts = ijk[plan["level"] == k]
    if len(pts) > 1:
        dist = cKDTree(pts).query(pts, k=2)[0][:, 1]
        ratios.append(dist / plan["level_spacing"][k])
ratios = np.concatenate(ratios)
print(f"nearest-neighbour distance / r over {len(ratios):,} points: min {ratios.min():.3f}, "
      f"median {np.median(ratios):.3f}, 99.9 % {np.percentile(ratios, 99.9):.2f}, max {ratios.max():.2f}")
k1 = int(np.argmin(np.abs(plan["level_values"] - 1.0)))
pts = ijk[plan["level"] == k1]
xyz = np.array([a[0] for a in axes]) + pts * steps                             # (d, phi1, phi2)
mirror = np.column_stack((xyz[:, 0], -xyz[:, 2], -xyz[:, 1]))              # the other half
fig = plt.figure(figsize=(6.7, 3.72))
a1 = fig.add_axes([-0.03, -0.04, 0.64, 1.06], projection="3d")
rng = np.random.default_rng(0)
for points, colour in ((mirror, "#C9D3DB"), (xyz, ORANGE)):
    pick = rng.permutation(len(points))[:5000]
    a1.scatter(points[pick, 1], points[pick, 2], points[pick, 0], s=0.7, c=colour, depthshade=False, lw=0)
a1.set_zlim(xyz[:, 0].min() - 0.005, axes[0][-1])
a1.set_xticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"]); a1.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"])
a1.set_xlabel("φ₁", labelpad=-7); a1.set_ylabel("φ₂", labelpad=-7); a1.set_zlabel("d", labelpad=-5)
a1.tick_params(labelsize=7, pad=-3)
a1.view_init(elev=22, azim=-50)
for pane in (a1.xaxis, a1.yaxis, a1.zaxis):
    pane.set_pane_color((1, 1, 1, 0))
a1.set_title(f"U = {plan['level_values'][k1]:.2f}: {len(xyz):,} points on φ₁ + φ₂ ≥ 0 (orange),\n"
             "the mirror gives the rest (grey); 5,000 of each drawn", fontsize=8.5, pad=0, y=0.93)
a2 = fig.add_axes([0.73, 0.17, 0.25, 0.66])
a2.hist(ratios, bins=np.arange(0.95, 1.65, 0.025), color=SLATE, lw=0)
a2.axvline(1, color=ORANGE, lw=1.4)
a2.text(1.03, 0.95, "r: no two\npoints closer", transform=a2.get_xaxis_transform(), color=ORANGE,
        fontsize=8, va="top")
a2.text(0.97, 0.55, f"99.9 % below\n{np.percentile(ratios, 99.9):.2f} r", transform=a2.transAxes, ha="right",
        color=INK, fontsize=8)
a2.set_xlabel("nearest neighbour / r", fontsize=9)
a2.set_ylabel("points", fontsize=9)
a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
a2.tick_params(labelsize=8)
a2.set_title(f"all {len(plan['level_values'])} levels:\nmedian {np.median(ratios):.2f} r", fontsize=9,
             loc="left")
save(fig, "iso_surface.png")

# F3: how close the solves land on their level -------------------------------------------------
miss = np.abs(solved["energy"] / plan["level_values"][solved["level"]] - 1)
tol = cfg["project_tol"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.7, 3.72), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.55, 1)})
bins = np.logspace(-14, -1, 53)
a1.hist(np.clip(miss, 1.01e-14, None), bins=bins, color=SLATE, lw=0)
a1.axvline(tol, color=ORANGE, lw=1.4)
a1.text(tol * 1.4, 0.95, f"{tol:g}: {np.mean(miss <= tol):.2%}\nwithin", transform=a1.get_xaxis_transform(),
        color=ORANGE, fontsize=8.5, va="top")
a1.set_xscale("log")
a1.set_xticks([1e-12, 1e-9, 1e-6, 1e-3], ["10⁻¹²", "10⁻⁹", "10⁻⁶", "10⁻³"])
a1.set_xlabel("$|U\\,/\\,U_k - 1|$ of the accepted state")
a1.set_ylabel("solves")
a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
a1.set_title("Distance from the level", loc="left")
n_steps = int(cfg["project_steps"])
share = np.bincount(solved["projections"], minlength=n_steps + 1) / len(miss) * 100
a2.bar(np.arange(n_steps + 1), share, color=ORANGE, width=0.7)
for x, v in enumerate(share):
    a2.text(x, v + 1, f"{v:.0f} %", ha="center", fontsize=8, color=INK)
a2.set_xticks(np.arange(n_steps + 1))
a2.set_xlabel("projection steps")
a2.set_ylabel("share of solves (%)")
a2.set_ylim(0, share.max() * 1.18)
a2.set_title(f"mean {solved['projections'].mean():.2f} steps", loc="left")
save(fig, "iso_projection.png")
print(f"on level: {np.mean(miss <= tol):.4%}; projections mean {solved['projections'].mean():.2f}, "
      f"share by steps {np.round(share, 1).tolist()}")

# F4: the cell scheme and the iso scheme on the deck's bins ------------------------------------------
edges = np.arange(0, 25.25, 0.25)
fig, axes2 = plt.subplots(1, 2, figsize=(10.4, 4.51), sharey=True, gridspec_kw={"wspace": 0.14},
                          constrained_layout=True)
for ax, energy, color, title in (
        (axes2[0], cells_all, TEAL, f"Cell scheme ({len(cells_all) / 1e6:.2f} M states)"),
        (axes2[1], iso_all, ORANGE, f"Iso-energy scheme ({len(iso_all) / 1e6:.2f} M states)")):
    counts, _ = np.histogram(energy, edges)
    ax.bar(edges[:-1], counts, width=0.25, align="edge", color=color, lw=0)
    for lo, _, _ in BANDS[1:]:
        ax.axvline(lo, color=GREY, lw=0.8, ls=(0, (3, 3)))
    lines = [f"{label}:  {np.count_nonzero((energy >= lo) & (energy < hi)) / len(energy):5.1%}"
             for lo, hi, label in BANDS]
    lines.append(f"beyond 25 (not shown): {np.count_nonzero(energy >= edges[-1]) / len(energy):.1%}")
    ax.text(0.97, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=INK, family="DejaVu Sans Mono", linespacing=1.5)
    ax.set_title(title, loc="left")
    ax.set_xlim(0, 25)
    ax.set_xlabel("bending energy U  (bins of 0.25)")
    ax.grid(axis="y", color="#E3E7EA", lw=0.8)
    ax.set_axisbelow(True)
axes2[0].set_ylabel("states per bin")
axes2[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
save(fig, "iso_histograms.png")
for name, energy in (("cells", cells_all), ("iso", iso_all), ("iso new", iso_new)):
    print(name, len(energy), {label: round(np.count_nonzero((energy >= lo) & (energy < hi)) / len(energy), 4)
                              for lo, hi, label in BANDS})
first = plan["level_placed"][:6].astype(int).tolist(), plan["level_target"][:6].astype(int).tolist()
print(f"levels {len(plan['level_values'])}, U {plan['level_values'][0]:.3f}-{plan['level_values'][-1]:.3f}; "
      f"first placed/target {first}; planned points {len(plan['cell']):,}")
