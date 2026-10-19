"""Refinement figures for the deck: the points-per-cell rule and the energy histograms
before and after refinement.

usage: python make_refinement.py [data/production_nn]

Before is the production run (one state per grid node); after is the training dataset
of solver/refine.py (both branches at every node, the refined points and their images).
The rule is cell_counts in refine.py: n = clip(round((u_ref / U_min)^power), 1, max_points)
per grid cell, and at least fold_min_points in cells where the bent-up branch holds at
some corners only.

The explainer figures follow sample_points and the solve in refine.py: a cell's n - 1 new
points are the first scrambled Sobol points (seed 0, as in the run) with a random shift per
cell, each solved from the corners holding the bent-up branch, nearest first.

The 3D views draw the same random 1 % of the points before and after, so the density of dots
is the density of data, and every point of a 5 x 5 x 3 block of cells near d = 0.97.

Writes refine_rule.png, refine_histograms.png, refine_rule_eq.png (the image the deck's
equation falls back to), refine_cell.png, refine_spread.png, refine_solve.png, refine_3d.png
and refine_3d_zoom.png to presentation/figures/.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import qmc

REPO = Path(__file__).resolve().parents[2]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "production_nn"
OUT = REPO / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, SLATE, ORANGE, TEAL, GREY, LIGHT = "#16222E", "#2E4A5C", "#D9822B", "#2A9D8F", "#5C6B7A", "#EEF2F4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.titlecolor": INK, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
BANDS = ((0, 1, "U < 1"), (1, 3, "1 ≤ U < 3"), (3, np.inf, "U ≥ 3"))


def save(fig, name, **kw):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor=kw.pop("facecolor", "white"), **kw)
    plt.close(fig)
    print("wrote", name)


with open(RUN / "config.json") as f:
    config = json.load(f)
rule = config["density"]
source = (RUN / config.get("source_run", "../production")).resolve()
with np.load(source / "solution_space_results.npz") as r:
    before = r["energy"].astype(float)
with np.load(RUN / config.get("dataset_npz", "nn_dataset.npz")) as ds:
    after = ds["energy"][ds["exists"]]

# F1: points per cell against the lowest corner energy ------------------------------------------
u = np.logspace(-1.3, 1.5, 600)
n = np.clip(np.rint((rule["u_ref"] / u) ** rule["power"]), 1, rule["max_points"])
fig, ax = plt.subplots(figsize=(5.8, 1.85))
ax.step(u, n, where="mid", color=ORANGE, lw=2)
ax.axhline(rule["fold_min_points"], color=TEAL, lw=1.3, ls=(0, (4, 3)))
ax.text(0.052, rule["fold_min_points"] * 1.18, f"partial cells: at least {rule['fold_min_points']}",
        color=TEAL, fontsize=8.5, va="bottom")
one = rule["u_ref"] / np.sqrt(1.5) ** (2 / rule["power"])        # round() gives 1 above this
full = rule["u_ref"] / rule["max_points"] ** (1 / rule["power"])
ax.axvspan(one, u[-1], color=LIGHT, zorder=0)
ax.text(np.sqrt(one * u[-1]), 30, "1 point:\nthe node alone", color=GREY, fontsize=8.5, ha="center")
ax.annotate(f"{rule['max_points']} for $U_\\mathrm{{min}}$ ≤ {full:.3g}", (full, rule["max_points"]), (0.75, 44),
            fontsize=8.5, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(u[0], u[-1]); ax.set_ylim(0.8, 100)
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
ax.set_yticks([1, 4, 16, 64], ["1", "4", "16", "64"])
ax.minorticks_off()
ax.set_xlabel("lowest bent-up energy at the cell's 8 corners, $U_\\mathrm{min}$")
ax.set_ylabel("points per cell")
fig.patch.set_alpha(0)
save(fig, "refine_rule.png", facecolor="none", transparent=True)

# F2: energy histograms before and after ----------------------------------------------------------
edges = np.arange(0, 25.25, 0.25)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True, gridspec_kw={"wspace": 0.14})
for ax, energy, color, title in (
        (axes[0], before, SLATE, f"Before: production run, one state per node ({len(before) / 1e6:.2f} M)"),
        (axes[1], after, ORANGE, f"After: training dataset ({len(after) / 1e6:.2f} M states)")):
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
axes[0].set_ylabel("states per bin")
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{v / 1e3:.0f}k" if v else "0"))
save(fig, "refine_histograms.png")
for name, energy in (("before", before), ("after", after)):
    print(name, len(energy), {label: int(np.count_nonzero((energy >= lo) & (energy < hi)))
                              for lo, hi, label in BANDS})

# F3: the rule as an image, the fallback of the deck's equation -----------------------------------
fig = plt.figure(figsize=(4.2, 0.6))
fig.text(0, 0.5, r"$n=\mathrm{clip}\!\left(\left(\dfrac{u_\mathrm{ref}}{U_\mathrm{min}}\right)^{2},\,1,\,64\right)$",
         fontsize=17, va="center", color=INK, math_fontfamily="cm")
save(fig, "refine_rule_eq.png", facecolor="none", transparent=True, pad_inches=0.02)


# F4: one grid cell, its corners and its new points -------------------------------------------------
SOBOL = qmc.Sobol(d=3, scramble=True, seed=int(config.get("seed", 0))).random_base2(6)
n_new = 15                                       # n = 16: U_min about 0.75
pts = (SOBOL[:n_new] + np.random.default_rng(7).random(3)) % 1.0
fig = plt.figure(figsize=(4.3, 4.6))
ax = fig.add_subplot(projection="3d", computed_zorder=False)
corners = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)], float)
for a in corners:
    for b in corners:
        if np.sum(np.abs(a - b)) == 1 and tuple(a) < tuple(b):
            ax.plot(*zip(a, b), color=GREY, lw=1.2, zorder=1)
ax.scatter(pts[:, 1], pts[:, 2], pts[:, 0], s=34, color=ORANGE, depthshade=False, zorder=3,
           label=f"new points to solve ({n_new})")
ax.scatter(*corners[1:, [1, 2, 0]].T, s=70, color=SLATE, depthshade=False, zorder=4,
           label="corners: rods already solved on the grid")
ax.scatter([0], [0], [0], s=110, facecolor="white", edgecolor=SLATE, lw=2.6, depthshade=False,
           zorder=5, label="the cell's own corner: counted, not solved again")
ax.set_xlabel("φ₁  (+0.01)", labelpad=-8); ax.set_ylabel("φ₂  (+0.01)", labelpad=-8)
ax.set_zlabel("d  (+0.01)", labelpad=-10)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_ticklabels([]); axis.pane.fill = False; axis.pane.set_edgecolor("none")
    axis.line.set_color("none")
ax.grid(False)
ax.view_init(elev=18, azim=-58)
ax.set_box_aspect((1, 1, 1), zoom=0.95)
fig.legend(loc="lower center", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02),
           handletextpad=0.3, markerscale=0.9)
fig.subplots_adjust(left=0, right=1, top=1.12, bottom=0.2)
save(fig, "refine_cell.png")

# F5: why Sobol points, and why a shift per cell ------------------------------------------------------
def square(ax, title, sub, grid=True):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(True); side.set_color(GREY)
    if grid:
        for v in (0.25, 0.5, 0.75):
            ax.axhline(v, color="#D5DCE1", lw=0.8, zorder=0); ax.axvline(v, color="#D5DCE1", lw=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=15)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.3, color=GREY, va="bottom")


def mark_squares(ax, x):
    """Shade 4 x 4 squares that are empty or hold more than one point."""
    cnt = np.zeros((4, 4), int)
    np.add.at(cnt, tuple(np.floor(x * 4).astype(int).T), 1)
    for i in range(4):
        for j in range(4):
            if cnt[i, j] != 1:
                ax.add_patch(Rectangle((i / 4, j / 4), 0.25, 0.25, color="#F3D6D0" if cnt[i, j] == 0 else "#F6E3BF",
                                       lw=0, zorder=0))
    return np.count_nonzero(cnt == 0), cnt.max()


fig, axes = plt.subplots(1, 4, figsize=(7.55, 2.6), gridspec_kw={"wspace": 0.1})
rnd = np.random.default_rng(3).random((16, 2))
empty, most = mark_squares(axes[0], rnd)
square(axes[0], "Random", f"{empty} empty, one holds {most}")
axes[0].scatter(*rnd.T, s=16, color=GREY, zorder=3)
lat = (np.stack(np.meshgrid(np.arange(4), np.arange(4)), -1).reshape(-1, 2) + 0.5) / 4
square(axes[1], "Regular lattice", "same spots in every cell")
axes[1].scatter(*lat.T, s=16, color=GREY, zorder=3)
sob = SOBOL[:16, 1:]
square(axes[2], "Sobol sequence", "one point per square")
axes[2].scatter(*sob.T, s=16, color=ORANGE, zorder=3)
square(axes[3], "Shifted per cell", "4 cells, 8 points each", grid=False)
rng = np.random.default_rng(11)
for i in range(2):
    for j in range(2):
        q = (SOBOL[:8, 1:] + rng.random(2)) % 1.0
        axes[3].scatter((i + q[:, 0]) / 2, (j + q[:, 1]) / 2, s=11, color=ORANGE, zorder=3)
axes[3].axhline(0.5, color=GREY, lw=1); axes[3].axvline(0.5, color=GREY, lw=1)
axes[0].text(0, -0.06, "16 points in one cell, seen flat (φ₁ across, φ₂ up). Red: empty square; "
             "yellow: crowded square", transform=axes[0].transAxes, fontsize=8.3, color=GREY, va="top")
save(fig, "refine_spread.png")

# F6: solving a new point, and the three free images ---------------------------------------------------
def rod(ax, x0, y0, flip=1, reverse=False, color=ORANGE, w=0.22, h=0.07):
    t = np.linspace(0, 1, 40)
    shape = np.sin(np.pi * t) * (0.55 + 0.9 * t)       # lopsided, so a swap of the ends shows
    if reverse:
        shape = shape[::-1]
    ax.plot(x0 - w / 2 + w * t, y0 + flip * h * shape / shape.max(), color=color, lw=2.2,
            solid_capstyle="round", zorder=4)
    ax.plot([x0 - w / 2, x0 + w / 2], [y0, y0], ls="none", marker="o", ms=3.2, color=INK, zorder=5)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 3.55), gridspec_kw={"wspace": 0.3})
# (a) one face of a cell
a1.set_xlim(-0.32, 1.32); a1.set_ylim(-0.3, 1.3); a1.set_aspect("equal"); a1.axis("off")
a1.add_patch(Rectangle((0, 0), 1, 1, fill=False, ec=GREY, lw=1.2))
held = {(0, 0): True, (1, 0): True, (0, 1): True, (1, 1): False}
P = np.array([0.36, 0.58])
for (cx, cy), ok in held.items():
    a1.scatter([cx], [cy], s=95, zorder=5, color=SLATE if ok else "white", edgecolor=SLATE, lw=2)
    if ok:
        rod(a1, cx + (0.16 if cx else -0.16), cy + (0.1 if cy else -0.16), color=SLATE, w=0.2, h=0.06)
a1.text(1.04, 1.04, "no bent-up rod\nhere (past\nthe fold)", fontsize=8, color=GREY, va="bottom")
order = sorted((np.hypot(cx - P[0], cy - P[1]), (cx, cy)) for (cx, cy), ok in held.items() if ok)
for rank, (_, (cx, cy)) in enumerate(order, 1):
    a1.add_patch(FancyArrowPatch((cx, cy), P, arrowstyle="-|>", mutation_scale=12, lw=1.4,
                                 color=[ORANGE, "#E3A869", "#EDC9A1"][rank - 1], shrinkA=7, shrinkB=9,
                                 ls="-" if rank == 1 else (0, (4, 3)), zorder=3))
    mid = (np.array([cx, cy]) * 0.55 + P * 0.45)
    a1.text(*mid + (0.035, 0.02), f"try {rank}", fontsize=8.3, color=INK, zorder=6)
a1.scatter(*P, s=170, marker="*", color=ORANGE, zorder=6)
a1.text(P[0] + 0.05, P[1] + 0.07, "new point", fontsize=9, color=INK, fontweight="bold")
a1.set_title("Start from the nearest solved corner", loc="left", fontsize=9.5, fontweight="bold")
a1.text(-0.32, -0.36, "one face of a cell, seen flat;\nat the corners: the solved rods", fontsize=8, color=GREY, va="top")
# (b) the four images in the angle square
a2.set_xlim(-1.75, 1.75); a2.set_ylim(-1.75, 1.75); a2.set_aspect("equal")
a2.fill([-1.6, 1.6, 1.6], [1.6, 1.6, -1.6], color="#FBEBDC", zorder=0)
a2.plot([-1.6, 1.6], [1.6, -1.6], color=GREY, lw=0.9, ls=(0, (4, 3)))
a2.text(1.52, -0.6, "solved\nhalf", fontsize=8, color=ORANGE, ha="right", va="top")
x, y = 0.95, 0.35
images = (((x, y), 1, False, ORANGE, "solved"),
          ((-y, -x), 1, True, ORANGE, "mirror:\nswap the ends"),
          ((-x, -y), -1, False, SLATE, "upside down:\nbent down"),
          ((y, x), -1, True, SLATE, "both"))
for (px, py), flip, rev, color, label in images:
    a2.scatter([px], [py], s=60, color=color, zorder=5)
    rod(a2, px, py + (0.2 if flip > 0 else -0.2), flip=flip, reverse=rev, color=color, w=0.5, h=0.16)
    a2.text(px, py + (-0.33 if flip > 0 else 0.33), label, fontsize=8, color=INK, ha="center",
            va="top" if flip > 0 else "bottom")
for v in (-np.pi / 2, np.pi / 2):
    a2.axhline(v, color="#D5DCE1", lw=0.8, zorder=0); a2.axvline(v, color="#D5DCE1", lw=0.8, zorder=0)
a2.set_xticks([-np.pi / 2, 0, np.pi / 2], ["−π/2", "0", "π/2"])
a2.set_yticks([-np.pi / 2, 0, np.pi / 2], ["−π/2", "0", "π/2"])
a2.set_xlabel("φ₁ (left clamp angle)", fontsize=9); a2.set_ylabel("φ₂ (right clamp angle)", fontsize=9)
a2.tick_params(labelsize=8.5)
a2.set_title("One solve, four rods", loc="left", fontsize=9.5, fontweight="bold")
save(fig, "refine_solve.png")


# F7: the whole domain in 3D, before and after, the same share of points drawn from each --------
SHARE = 0.01
rng = np.random.default_rng(5)
with np.load(source / "solution_space_results.npz") as r:
    grid_axes = (r["d_vals"], r["phi1_vals"], r["phi2_vals"])
gi, gj, gk = np.unravel_index(np.flatnonzero(rng.random(len(before)) < SHARE),
                              tuple(len(a) for a in grid_axes))
before_xyz = np.column_stack((grid_axes[1][gj], grid_axes[2][gk], grid_axes[0][gi]))
with np.load(RUN / config.get("dataset_npz", "nn_dataset.npz")) as ds:
    keep = ds["exists"] & (rng.random(len(ds["exists"])) < SHARE)
    after_xyz = np.column_stack((ds["phi1"][keep], ds["phi2"][keep], ds["d"][keep]))
    after_refined = ds["source"][keep] >= 4


def box3d(ax, title, sub):
    ax.set_xlim(-np.pi / 2, np.pi / 2); ax.set_ylim(-np.pi / 2, np.pi / 2); ax.set_zlim(0.70, 0.99)
    ax.set_xticks([-np.pi / 2, 0, np.pi / 2], ["−π/2", "0", ""])      # the corner keeps φ₂'s label
    ax.set_yticks([-np.pi / 2, 0, np.pi / 2], ["−π/2", "0", "π/2"])
    ax.set_zticks([0.7, 0.8, 0.9, 0.99], ["0.70", "0.80", "0.90", "0.99"])
    ax.set_xlabel("φ₁", labelpad=-4); ax.set_ylabel("φ₂", labelpad=-4); ax.set_zlabel("d", labelpad=-2)
    ax.tick_params(labelsize=8, pad=-1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor("#D5DCE1")
    ax.grid(False)
    ax.view_init(elev=16, azim=-50)
    ax.set_box_aspect((1, 1, 0.95), zoom=0.92)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, y=1.0)
    ax.text2D(0.0, 0.955, sub, transform=ax.transAxes, fontsize=8.6, color=GREY)


fig = plt.figure(figsize=(11.0, 5.2))
a1 = fig.add_subplot(1, 2, 1, projection="3d")
a2 = fig.add_subplot(1, 2, 2, projection="3d")
box3d(a1, f"Before: {len(before) / 1e6:.2f} M grid states",
      f"one per grid node; 1 % drawn ({len(before_xyz):,} dots)")
a1.scatter(*before_xyz.T, s=0.6, color=SLATE, alpha=0.5, linewidths=0, depthshade=False)
box3d(a2, f"After: {len(after) / 1e6:.2f} M states",
      f"both branches + refined points; 1 % drawn ({len(after_xyz):,} dots)")
a2.scatter(*after_xyz[~after_refined].T, s=0.6, color=SLATE, alpha=0.5, linewidths=0, depthshade=False)
a2.scatter(*after_xyz[after_refined].T, s=0.6, color=ORANGE, alpha=0.6, linewidths=0, depthshade=False)
for label, color in (("grid node", SLATE), ("refined point", ORANGE)):
    a2.scatter([], [], [], s=18, color=color, label=label)
a2.legend(loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(1.02, 0.93))
fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0.02, wspace=0.0)
save(fig, "refine_3d.png")

# F8: a few cells, every point -------------------------------------------------------------------
# A 5 x 5 x 3 block of cells (by node index) on the edge of a low-energy well at d = 0.96-0.99,
# where the points per cell run from 5 to 64. Bent-up branch only.
I0, J0, K0, NC, ND = 26, 152, 154, 5, 3
gd, g1, g2 = grid_axes[0][I0:I0 + ND + 1], grid_axes[1][J0:J0 + NC + 1], grid_axes[2][K0:K0 + NC + 1]
TOL = 1e-9
with np.load(RUN / config.get("dataset_npz", "nn_dataset.npz")) as ds:
    inside = ds["exists"] & (ds["branch"] > 0)
    for key, axis in (("d", gd), ("phi1", g1), ("phi2", g2)):
        inside &= (ds[key] >= axis[0] - TOL) & (ds[key] <= axis[-1] + TOL)
    zoom = {k: ds[k][inside] for k in ("d", "phi1", "phi2", "energy", "source")}
on_node = zoom["source"] <= 3
ENERGY = matplotlib.colors.LinearSegmentedColormap.from_list("energy", ["#7A3208", ORANGE, "#E9C46A"])


def cells3d(ax, title, sub):
    light = dict(color="#D3DAE0", lw=0.6, zorder=0)
    for a in g1:                                    # the cell edges
        for b in g2:
            ax.plot([a, a], [b, b], [gd[0], gd[-1]], **light)
        for c in gd:
            ax.plot([a, a], [g2[0], g2[-1]], [c, c], **light)
    for b in g2:
        for c in gd:
            ax.plot([g1[0], g1[-1]], [b, b], [c, c], **light)
    ax.set_xlim(g1[0], g1[-1]); ax.set_ylim(g2[0], g2[-1]); ax.set_zlim(gd[0], gd[-1])
    ax.set_xlabel("φ₁", labelpad=-2); ax.set_ylabel("φ₂", labelpad=-2); ax.set_zlabel("d", labelpad=0)
    ax.set_xticks(g1[::5], [f"{v:+.2f}" for v in g1[::5]])
    ax.set_yticks(g2[::5], ["", f"{g2[-1]:+.2f}"])     # its first label would sit on φ₁'s last
    ax.set_zticks(gd, [f"{v:.2f}" for v in gd])
    ax.tick_params(labelsize=8, pad=-1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=26, azim=-54)                 # no two nodes project close together
    ax.set_box_aspect((1, 1, 0.62), zoom=1.0)
    x = ax.get_position().x0 + 0.01                  # headings at one height for both panels
    ax.figure.text(x, 0.97, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    ax.figure.text(x, 0.915, sub, fontsize=8.6, color=GREY, va="top")


fig = plt.figure(figsize=(11.0, 5.0))
a1 = fig.add_axes([0.0, 0.0, 0.45, 1.0], projection="3d")
a2 = fig.add_axes([0.48, 0.0, 0.43, 1.0], projection="3d")
GD, G1, G2 = np.meshgrid(gd, g1, g2, indexing="ij")
cells3d(a1, f"Before: {GD.size} grid nodes", f"{NC} × {NC} × {ND} cells, a rod at every corner")
a1.scatter(G1.ravel(), G2.ravel(), GD.ravel(), s=18, color=SLATE, depthshade=False)
cells3d(a2, f"After: {len(zoom['d']):,} bent-up states",
        "the same corners, plus points inside each cell")
norm = matplotlib.colors.LogNorm(0.3, 2)
sc = a2.scatter(zoom["phi1"][~on_node], zoom["phi2"][~on_node], zoom["d"][~on_node], s=6,
                c=zoom["energy"][~on_node], cmap=ENERGY, norm=norm, depthshade=False, linewidths=0)
a2.scatter(zoom["phi1"][on_node], zoom["phi2"][on_node], zoom["d"][on_node], s=18, color=SLATE,
           depthshade=False)
cax = fig.add_axes([0.955, 0.3, 0.012, 0.4])
cb = fig.colorbar(sc, cax=cax)
cb.set_label("bending energy U", fontsize=9)
cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
cb.set_ticks([0.3, 0.5, 1, 2], labels=["0.3", "0.5", "1", "2"])
cb.ax.tick_params(labelsize=8)
save(fig, "refine_3d_zoom.png")
print("zoom:", len(zoom["d"]), "states;", int(on_node.sum()), "on nodes; U range",
      round(float(zoom["energy"].min()), 3), round(float(zoom["energy"].max()), 3))
