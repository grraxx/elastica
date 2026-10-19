"""Bistable-region figures and the branch-following animation for the deck.

usage: python make_bistable.py [data/production]

The solve grew from one bent-up cold start. A bent-down cold start gives its image under
theta -> -theta: the node (phi1, phi2) holds the solution at (-phi1, -phi2) with theta and
p negated, which on the map is the mirror in phi1 = phi2 with the branches swapped. Where
both branches are stable the two solves differ: that is the bistable region. Elsewhere they
hold the same rod, and swapping phi1 and phi2 turns it over (theta(s) -> theta(L - s)).

The animation follows one rod through the clamp angles at d = 0.82: each grid step it stays
on the equilibrium continuous with the one it is on (no jump in q, the solver's rule) and
snaps where its branch ends. The map shows the landscape of the branch it is on.

Writes bistable_*.png, branch_walk.mp4 and branch_walk_poster.png to presentation/figures/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, TwoSlopeNorm
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solver"))
from branch import is_jump  # noqa: E402
from validate_results import read_rows  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "production"
OUT = REPO / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 0.17                                      # the solver's jump_tol, on q
SAME = 0.02                                     # two solves hold one rod when q agrees this well

INK, SLATE, ORANGE, TEAL, RED, GREY, LIGHT = (
    "#16222E", "#2E4A5C", "#D9822B", "#2A9D8F", "#C8553D", "#5C6B7A", "#EEF2F4")
UP_FILL, DOWN_FILL, BI_FILL = "#F6D9BC", "#C9D3DB", "#BFE3DE"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.titlecolor": INK, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
UPDOWN = LinearSegmentedColormap.from_list("updown", [SLATE, "#9DB0BF", "#F7F7F7", "#F0C08F", ORANGE])
RESID = LinearSegmentedColormap.from_list("resid", ["#F7F7F7", "#9DB0BF", SLATE, INK])
PI_TICKS = ([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2], ["−π/2", "−π/4", "0", "π/4", "π/2"])

# ------------------------------------------------------------------ data
RESULTS = RUN / "solution_space_results.npz"
with np.load(RESULTS) as r:
    shape = tuple(int(n) for n in r["shape"])
    d_vals, p1, p2, s = r["d_vals"], r["phi1_vals"], r["phi2_vals"], r["beam_x"]
    Q = r["q"].reshape(shape).astype(float)
    E = r["energy"].reshape(shape).astype(float)
N = shape[1]
EXTENT = (p1[0], p1[-1], p2[0], p2[-1])
H = np.pi / 2
BR = {"up": (Q, E), "down": (-Q[:, ::-1, ::-1], E[:, ::-1, ::-1])}     # q and U of each cold start
QD = BR["down"][0]
DISTINCT = np.abs(Q - QD) >= SAME                 # bistable: the two cold starts hold different rods
MAIN = DISTINCT & (Q > 0) & (QD < 0)              # ... one bent up, one bent down


def layer(d):
    return int(np.argmin(np.abs(d_vals - d)))


def at(v):
    return int(np.argmin(np.abs(p1 - v)))


def theta_rows(nodes):
    """theta of (i, j, k, branch) nodes; bent down reads the mirrored node, negated."""
    index = [int(np.ravel_multi_index((i, j, k) if b == "up" else (i, N - 1 - j, N - 1 - k), shape))
             for i, j, k, b in nodes]
    rows = dict(zip(sorted(set(index)), read_rows(RESULTS, "theta", sorted(set(index)))))
    return [rows[n] * (1 if b == "up" else -1) for n, (*_, b) in zip(index, nodes)]


def centreline(theta):
    x = np.concatenate(([0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * np.diff(s))))
    y = np.concatenate(([0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * np.diff(s))))
    return x, y


def snap_edges(q):
    """Midpoints of grid edges of one layer where q changes sign, with and without a jump."""
    out = {"snap": [], "smooth": []}
    for axis in (0, 1):
        a, b = (q[:-1], q[1:]) if axis == 0 else (q[:, :-1], q[:, 1:])
        change = np.sign(a) != np.sign(b)
        snap = change & is_jump(a, b, TOL)
        for name, mask in (("snap", snap), ("smooth", change & ~snap)):
            j, k = np.nonzero(mask)
            x = 0.5 * (p1[j] + p1[j + 1]) if axis == 0 else p1[j]
            y = p2[k] if axis == 0 else 0.5 * (p2[k] + p2[k + 1])
            out[name].append(np.column_stack((x, y)))
    return {k: np.concatenate(v) for k, v in out.items()}


def pi_axes(ax, y=True):
    ax.set_xticks(*PI_TICKS)
    if y:
        ax.set_yticks(*PI_TICKS)
        ax.set_ylabel("φ₂")
    else:
        ax.set_yticks(PI_TICKS[0], [""] * len(PI_TICKS[0]))
    ax.set_xlabel("φ₁")
    ax.set_aspect("equal")


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


I82, I70 = layer(0.82), layer(0.70)
ZERO = at(0.0)
snaps82 = {b: snap_edges(BR[b][0][I82]) for b in BR}

# F1: the two cold starts, mirrored in phi1 = phi2 -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
for ax, b, title in ((axes[0], "up", "Bent-up cold start (the solve)"),
                     (axes[1], "down", "Bent-down cold start (its mirror)")):
    im = ax.imshow(BR[b][0][I82].T, origin="lower", extent=EXTENT, cmap=UPDOWN,
                   norm=TwoSlopeNorm(0, -1.15, 1.15), interpolation="nearest")
    ax.plot(*snaps82[b]["snap"].T, ".", ms=1.6, color=RED)
    ax.plot(*snaps82[b]["smooth"].T, ".", ms=1.2, color=INK)
    for x, y, n in ((1.2, 1.3, "1"), (-1.2, 1.3, "2"), (-1.2, -1.3, "3"), (1.2, -1.3, "4")):
        ax.text(x, y, n, ha="center", va="center", fontsize=13, fontweight="bold", color=INK, alpha=0.45)
    ax.set_title(title)
    pi_axes(ax, y=ax is axes[0])
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cb.set_label("q   (orange: bent up, slate: bent down)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
save(fig, "bistable_coldstarts.png")

# F2: outside the region, swapping the clamps turns the rod over --------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1, 1.25)})
resid = np.abs(Q[I82] + Q[I82].T) + 1e-6
im = a1.imshow(resid.T, origin="lower", extent=EXTENT, cmap=RESID, norm=LogNorm(1e-5, 2),
               interpolation="nearest")
a1.plot([-H, H], [-H, H], ":", color=GREY, lw=0.8)
cb = fig.colorbar(im, ax=a1, shrink=0.85, pad=0.02)
cb.set_label("|q(φ₁, φ₂) + q(φ₂, φ₁)|", fontsize=8.5); cb.ax.tick_params(labelsize=8)
a1.set_title("d = 0.82, bent-up cold start")
pi_axes(a1)
pa, pb = (at(-1.2), at(0.35)), (at(0.35), at(-1.2))
ta, tb = theta_rows([(I82, *pa, "up"), (I82, *pb, "up")])
xa, ya = centreline(ta); xb, yb = centreline(tb)
a2.plot(xa, ya, color=SLATE, lw=2.4,
        label=f"φ = ({p1[pa[0]]:+.2f}, {p2[pa[1]]:+.2f}):  bent down,  U = {E[I82][pa]:.3f}")
a2.plot(xb, yb, color=ORANGE, lw=2.4,
        label=f"φ = ({p1[pb[0]]:+.2f}, {p2[pb[1]]:+.2f}):  bent up,  U = {E[I82][pb]:.3f}")
a2.plot(d_vals[I82] - xa, -ya, color=INK, lw=1.1, ls=(0, (3, 3)), label="first rod turned 180° about mid-span")
a2.axhline(0, color="#D5DBE0", lw=0.8, zorder=0)
a2.set_aspect("equal"); a2.set_xlabel("x"); a2.set_ylabel("y")
a2.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "bistable_mirror.png")

# F3: the region at d = 0.82 and its four edges ------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
cls = np.where(DISTINCT[I82], 2, np.where(Q[I82] > 0, 1, 0))
ax.imshow(cls.T, origin="lower", extent=EXTENT, interpolation="nearest",
          cmap=ListedColormap([DOWN_FILL, UP_FILL, BI_FILL]), vmin=0, vmax=2)
ax.plot(*snaps82["up"]["snap"].T, ".", ms=2.4, color=ORANGE, label="bent-up branch ends (fold)")
ax.plot(*snaps82["down"]["snap"].T, ".", ms=2.4, color=SLATE, label="bent-down branch ends (fold)")
ax.plot([-H, H], [-H, H], "--", color=GREY, lw=0.7)
ax.plot([-H, H], [H, -H], ":", color=GREY, lw=0.7)
diag = np.flatnonzero(MAIN[I82][np.arange(N), np.arange(N)])
anti = np.flatnonzero(MAIN[I82][np.arange(N), N - 1 - np.arange(N)])
cusp, corner = p1[diag[-1]], p1[anti[-1]]
ax.text(0, 0.1, "bistable:\nbent up and bent down\nboth stable", ha="center", va="center", fontsize=9.5,
        color=INK, fontweight="bold")
for x, y, text in ((-1.2, 1.42, "only bent down"), (1.15, -1.3, "only bent up"),
                   (1.35, 0.95, "bent up"), (0.95, 1.42, "bent down"),
                   (-1.35, -0.95, "bent down"), (-0.9, -1.42, "bent up")):
    ax.text(x, y, text, ha="center", fontsize=8.5, color=INK)
for x, y, text, dx, dy in ((cusp, cusp, "cusp", 0.25, -0.35), (-cusp, -cusp, "cusp", -0.1, 0.35),
                           (-corner, corner, "corner", 0.35, 0.28), (corner, -corner, "corner", -0.3, -0.32)):
    ax.plot(x, y, "o", ms=6, mfc="white", mec=INK, mew=1.3, zorder=5)
    ax.annotate(text, (x, y), (x + dx, y + dy), fontsize=8.5, ha="center", color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
ax.annotate("φ₁ = φ₂ beyond the cusp:\nsmooth crossing (S-shape)", (1.3, 1.3), (0.45, 1.12), fontsize=8,
            ha="center", color=INK, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:])
pi_axes(ax)
ax.set_title("d = 0.82")
ax.legend(fontsize=8, frameon=False, markerscale=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
save(fig, "bistable_region.png")

# F4: two stable rods at the same clamps ---------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 2.3), constrained_layout=True)
j, k = at(0.2), ZERO
for (b, colour), th in zip((("up", ORANGE), ("down", SLATE)), theta_rows([(I82, j, k, "up"), (I82, j, k, "down")])):
    ax.plot(*centreline(th), color=colour, lw=2.4, label=f"bent {b}:  U = {BR[b][1][I82, j, k]:.3f}")
ax.plot([0, d_vals[I82]], [0, 0], "o", color=INK, ms=5, zorder=5)
ax.axhline(0, color="#D5DBE0", lw=0.8, zorder=0)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"φ = ({p1[j]:+.2f}, {p2[k]:+.2f}),  d = 0.82", fontsize=10)
ax.legend(fontsize=8.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
save(fig, "bistable_tworods.png")

# F5: hysteresis along phi2 = 0 ------------------------------------------------------------
row = MAIN[I82][:, ZERO]
lo, hi = np.flatnonzero(row)[[0, -1]]
up_side, down_side = np.arange(N) >= lo, np.arange(N) <= hi
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
for ax, (key, label) in zip(axes, ((0, "q = p / a(d)"), (1, "bending energy U"))):
    yu, yd = BR["up"][key][I82][:, ZERO], BR["down"][key][I82][:, ZERO]
    ax.axvspan(p1[lo], p1[hi], color=BI_FILL, alpha=0.6, lw=0)
    ax.plot(p1[up_side], yu[up_side], color=ORANGE, lw=2.2, label="bent-up branch")
    ax.plot(p1[down_side], yd[down_side], color=SLATE, lw=2.2, label="bent-down branch")
    for j0, src, dst in ((lo, yu, yd), (hi, yd, yu)):
        ax.annotate("", (p1[j0], dst[j0]), (p1[j0], src[j0]),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6, shrinkA=0, shrinkB=0))
    for a, b, y in ((0.4, 0.1, yu), (-0.4, -0.1, yd)):
        ax.annotate("", (p1[at(b)], y[at(b)]), (p1[at(a)], y[at(a)]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
    ax.set_xticks(*PI_TICKS); ax.set_xlabel("φ₁   (φ₂ = 0, d = 0.82)"); ax.set_ylabel(label)
    ax.grid(color="#E3E8EC", lw=0.6); ax.set_axisbelow(True)
axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
axes[0].text(0, 0, f"bistable\n{p1[lo]:+.2f} < φ₁ < {p1[hi]:+.2f}", ha="center", va="center", fontsize=8.5, color=INK)
u0, u1 = E[I82][lo, ZERO], BR["down"][1][I82][lo, ZERO]
axes[1].set_title(f"each snap releases U {u0:.1f} → {u1:.1f}", fontsize=10)
axes[0].set_title("left on the bent-up branch, then back right", fontsize=10)
save(fig, "bistable_hysteresis.png")

# F6: the region closes in as d -> 1 -------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1, 1.15)})
picks = [layer(d) for d in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)]
colours = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(picks)))
for i, colour in zip(picks, colours):
    a1.contour(p1, p2, DISTINCT[i].T.astype(float), levels=[0.5], colors=[colour], linewidths=1.5)
    a1.plot([], [], color=colour, label=f"d = {d_vals[i]:.2f}")
a1.plot([-H, H], [-H, H], "--", color=GREY, lw=0.7)
a1.set_xlim(EXTENT[:2]); a1.set_ylim(EXTENT[2:])
pi_axes(a1); a1.set_title("Edge of the bistable region")
a1.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
share = DISTINCT.mean(axis=(1, 2)) * 100
half = [p1[np.flatnonzero(MAIN[i][:, ZERO])[-1]] for i in range(len(d_vals))]
a2.plot(d_vals, share, color=TEAL, lw=2.2, marker="o", ms=3, label="share of the layer (%)")
a3 = a2.twinx()
a3.plot(d_vals, half, color=INK, lw=1.4, ls="--", label="half-width along φ₂ = 0 (rad)")
a3.spines["right"].set_visible(True)
a2.set_ylim(0, 45); a3.set_ylim(0, 1.1)
a2.set_xlabel("d"); a2.set_ylabel("bistable share of the layer (%)"); a3.set_ylabel("half-width along φ₂ = 0 (rad)")
a2.grid(color="#E3E8EC", lw=0.6); a2.set_axisbelow(True)
h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels()
a2.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper right")
a2.set_title("It closes in on the origin as d → 1")
save(fig, "bistable_shrink.png")

# F7: the tip along the diagonal at d = 0.70 -----------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1, 1.3)})
cls = np.where(DISTINCT[I70], 2, np.where(Q[I70] > 0, 1, 0))
a1.imshow(cls.T, origin="lower", extent=EXTENT, interpolation="nearest",
          cmap=ListedColormap([DOWN_FILL, UP_FILL, BI_FILL]), vmin=0, vmax=2)
for b, colour in (("up", ORANGE), ("down", SLATE)):
    a1.plot(*snap_edges(BR[b][0][I70])["snap"].T, ".", ms=4, color=colour,
            label=f"bent-{b} branch ends")
a1.plot([-H, H], [-H, H], "--", color=GREY, lw=0.7)
tip = p1[np.flatnonzero(DISTINCT[I70][np.arange(N), np.arange(N)])[-1]]
a1.annotate(f"tip (cusp), φ ≈ {tip:.2f}", (tip, tip), (tip - 0.08, tip + 0.12), fontsize=8.5, ha="center",
            color=INK, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7))
a1.set_xlim(0.7, 1.3); a1.set_ylim(0.7, 1.3); a1.set_aspect("equal")
a1.set_xlabel("φ₁"); a1.set_ylabel("φ₂"); a1.set_title("d = 0.70, upper-right corner")
a1.legend(fontsize=7.5, frameon=True, framealpha=0.9, markerscale=2.5, loc="lower right")
jt = at(0.97)
for (b, colour), th in zip((("up", ORANGE), ("down", SLATE)), theta_rows([(I70, jt, jt, "up"), (I70, jt, jt, "down")])):
    a2.plot(*centreline(th), color=colour, lw=2.2,
            label=f"bent-{b} cold start:  q = {BR[b][0][I70, jt, jt]:+.2f},  U = {BR[b][1][I70, jt, jt]:.2f}")
a2.axhline(0, color="#D5DBE0", lw=0.8, zorder=0)
a2.set_aspect("equal"); a2.set_xlabel("x"); a2.set_ylabel("y")
a2.set_title(f"two stable S-shapes at φ = ({p1[jt]:+.2f}, {p2[jt]:+.2f})", fontsize=10)
a2.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
save(fig, "bistable_tip.png")

# 3D: the snap surface each branch can meet -----------------------------------------------
pv.OFF_SCREEN = True
ZS = 8.0
surface = pv.read(RUN / "post" / "snap_surface.vtp").triangulate()
is_snap = np.asarray(surface.point_data["snap"])[surface.regular_faces].mean(axis=1) > 0.5
surface.points[:, 2] *= ZS
sheet = {"up": (surface.extract_cells(np.flatnonzero(is_snap)).extract_surface(algorithm="dataset_surface")
                .clean().smooth_taubin(n_iter=80, pass_band=0.02))}
sheet["down"] = sheet["up"].copy()
sheet["down"].points[:, :2] *= -1                 # theta -> -theta: (phi1, phi2) -> (-phi1, -phi2)
BOUNDS = (p1[0], p1[-1], p2[0], p2[-1], d_vals[0] * ZS, d_vals[-1] * ZS)


def crop(path, pad=12):
    image = Image.open(path).convert("RGB")
    box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    if box:
        image.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                    min(box[2] + pad, image.width), min(box[3] + pad, image.height))).save(path)


for b, colour in (("up", ORANGE), ("down", "#5A7A8E")):          # slate, lightened to show the shading
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 1000))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(sheet[b], color=colour, smooth_shading=True, specular=0.2, ambient=0.35, diffuse=0.75)
    plane = pv.Plane(center=(0, 0, 0.82 * ZS), direction=(0, 0, 1), i_size=np.pi, j_size=np.pi)
    plotter.add_mesh(plane, color=TEAL, opacity=0.25)
    plotter.add_mesh(pv.Box(bounds=BOUNDS).extract_all_edges(), color="#9AA5B1", line_width=1.5)
    plotter.add_point_labels(np.array([(1.25, 1.25, BOUNDS[4]), (-1.25, 1.25, BOUNDS[4]),
                                       (-1.25, -1.25, BOUNDS[4]), (1.25, -1.25, BOUNDS[4])]),
                             ["Q1", "Q2", "Q3", "Q4"], font_size=34, text_color=GREY, bold=True, show_points=False,
                             shape=None, always_visible=True)
    plotter.add_point_labels(np.array([(p1[-1] + 0.25, 0, BOUNDS[4]), (0, p2[-1] + 0.25, BOUNDS[4]),
                                       (-H, -H, 0.82 * ZS)]), ["phi1", "phi2", "d = 0.82"], font_size=26,
                             text_color=INK, show_points=False, shape=None, always_visible=True)
    centre = np.array([0.0, 0.0, 0.845 * ZS])
    a, e = np.radians(-62), np.radians(58)
    eye = centre + 11.5 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    plotter.camera_position = [tuple(eye), tuple(centre), (0.0, 0.0, 1.0)]
    plotter.screenshot(OUT / f"bistable_snap3d_{b}.png")
    plotter.close()
    crop(OUT / f"bistable_snap3d_{b}.png")
    print("wrote", f"bistable_snap3d_{b}.png")

# Animation: one rod walked around the snap surface at d = 0.82 ----------------------------
def follow(q_from, branch, q):
    """The branch a rod on `branch` (with q = q_from) is on after one step to a node where the
    two cold starts give q = {up, down}, and whether it snapped on the way."""
    if abs(q["up"] - q["down"]) < SAME:                # one rod here: its sign names the branch
        return ("up" if q["up"] > 0 else "down" if q["up"] < 0 else branch), bool(is_jump(q_from, q["up"], TOL))
    other = "down" if branch == "up" else "up"
    if not is_jump(q_from, q[branch], TOL):
        return branch, False
    if not is_jump(q_from, q[other], TOL):
        return other, False
    return (branch if abs(q[branch] - q_from) <= abs(q[other] - q_from) else other), True


LEGS = [("Move φ₁ left, along φ₂ = 0", (-1.25, 0.0)),
        ("Move φ₁ right, back across the region", (1.25, 0.0)),
        ("Raise φ₂: go round the end of the snap curve", (1.25, 1.40)),
        ("Move φ₁ back to 0", (0.0, 1.40)),
        ("Lower φ₂ to 0: back at the start", (0.0, 0.0)),
        ("Move φ₁ left again", (-1.25, 0.0))]
qL = {b: BR[b][0][I82] for b in BR}
uL = {b: BR[b][1][I82] for b in BR}
path = [dict(j=ZERO, k=ZERO, branch="up", leg=0, event=None)]
for n, (_, (t1, t2)) in enumerate(LEGS):
    for axis, target in ((0, at(t1)), (1, at(t2))):
        while (path[-1]["j"], path[-1]["k"])[axis] != target:
            a = path[-1]
            j, k = a["j"], a["k"]
            if axis == 0:
                j += np.sign(target - j)
            else:
                k += np.sign(target - k)
            q_from = qL[a["branch"]][a["j"], a["k"]]
            branch, snap = follow(q_from, a["branch"], {b: qL[b][j, k] for b in BR})
            q_to = qL[branch][j, k]
            event = "snap" if snap else ("smooth" if np.sign(q_to) != np.sign(q_from) else None)
            path.append(dict(j=int(j), k=int(k), branch=branch, leg=n, event=event))
for p in path:
    p["q"], p["U"] = qL[p["branch"]][p["j"], p["k"]], uL[p["branch"]][p["j"], p["k"]]
print("path:", len(path), "steps;", [(p1[p["j"]].round(2), p2[p["k"]].round(2), p["event"], p["branch"])
                                       for p in path if p["event"]])
rods = theta_rows([(I82, p["j"], p["k"], p["branch"]) for p in path])
shapes = [centreline(th) for th in rods]

# Frames: every second step, with pauses at the start of each leg and at each event.
frames = []
for n, p in enumerate(path):
    if n and p["leg"] != path[n - 1]["leg"]:
        frames += [(n, "leg")] * 14
    if p["event"]:
        frames += [(n, p["event"])] * 32
    elif n % 2 == 0 or n == len(path) - 1:
        frames.append((n, None))
frames = [(0, "leg")] * 22 + frames + [(len(path) - 1, "end")] * 50

WORD = {"up": "bent up", "down": "bent down"}
QUADS = {"up": "quadrants 1, 2, 3", "down": "quadrants 1, 3, 4"}
fig = plt.figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
gs = fig.add_gridspec(2, 2, width_ratios=(1.12, 1), height_ratios=(1, 1.15), left=0.05, right=0.98,
                      top=0.83, bottom=0.07, wspace=0.12, hspace=0.32)
ax_map, ax_rod, ax_3d = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
maps = {b: ax_map.imshow(qL[b].T, origin="lower", extent=EXTENT, cmap=UPDOWN, norm=TwoSlopeNorm(0, -1.15, 1.15),
                         interpolation="nearest") for b in BR}
active = {b: ax_map.plot(*snaps82[b]["snap"].T, ".", ms=3.2, color=RED, zorder=3)[0] for b in BR}
inactive = {b: ax_map.plot(*snaps82[b]["snap"].T, ".", ms=1.4, color=GREY, alpha=0.6, zorder=3)[0] for b in BR}
ax_map.plot([-H, H], [-H, H], ":", color=GREY, lw=0.7)
trail, = ax_map.plot([], [], color=INK, lw=1.6, zorder=4)
marks, = ax_map.plot([], [], "X", ms=11, color=RED, mec="white", mew=1.2, zorder=6)
dot, = ax_map.plot([], [], "o", ms=12, mec="white", mew=2, zorder=7)
ax_map.set_xlim(EXTENT[:2]); ax_map.set_ylim(EXTENT[2:])
pi_axes(ax_map)
ax_map.plot([], [], ".", ms=8, color=RED, label="snap curve of the branch the rod is on")
ax_map.plot([], [], ".", ms=8, color=GREY, label="the other branch's snap curve")
ax_map.plot([], [], "X", ms=9, color=RED, mec="white", label="the rod snapped")
ax_map.legend(fontsize=8.5, frameon=True, framealpha=0.92, loc="lower left")
map_title = ax_map.set_title("", fontsize=11, loc="left")

allx = np.concatenate([x for x, _ in shapes]); ally = np.concatenate([y for _, y in shapes])
ax_rod.axhline(0, color="#D5DBE0", lw=0.8, zorder=0)
ghost, = ax_rod.plot([], [], color=GREY, lw=1.6, ls=(0, (4, 3)), label="just before the snap")
rod_line, = ax_rod.plot([], [], lw=2.8)
ends, = ax_rod.plot([0, d_vals[I82]], [0, 0], "o", color=INK, ms=5, zorder=5)
ax_rod.set_xlim(allx.min() - 0.03, allx.max() + 0.03); ax_rod.set_ylim(ally.min() - 0.04, ally.max() + 0.04)
ax_rod.set_aspect("equal"); ax_rod.set_xlabel("x"); ax_rod.set_ylabel("y")
rod_title = ax_rod.set_title("", fontsize=10.5, loc="left")
rod_legend = ax_rod.legend(fontsize=8.5, frameon=False, loc="lower right")

ims3d = {b: ax_3d.imshow(Image.open(OUT / f"bistable_snap3d_{b}.png")) for b in BR}
ax_3d.axis("off")
title3d = ax_3d.set_title("", fontsize=10.5, loc="left")

head = fig.text(0.05, 0.94, "", fontsize=17, fontweight="bold", color=INK, va="center")
status = fig.text(0.05, 0.885, "", fontsize=12, color=INK, va="center")
event_text = fig.text(0.98, 0.94, "", fontsize=14, fontweight="bold", color=RED, ha="right", va="center")
snap_pts, last_snap = [], None


def draw(frame):
    global last_snap
    n, kind = frame
    p = path[n]
    b = p["branch"]
    for x in BR:
        maps[x].set_visible(x == b); active[x].set_visible(x == b); inactive[x].set_visible(x != b)
        ims3d[x].set_visible(x == b)
    leg = LEGS[p["leg"]][0]
    head.set_text(f"Leg {p['leg'] + 1} of {len(LEGS)}:  {leg}")
    status.set_text(f"Rod: {WORD[b]}   ·   map: the {WORD[b].replace(' ', '-')} landscape   ·   "
                    f"the snap curve it can meet is in {QUADS[b]}")
    xs = [p1[path[m]["j"]] for m in range(n + 1)]; ys = [p2[path[m]["k"]] for m in range(n + 1)]
    trail.set_data(xs, ys)
    if kind == "snap" and (not snap_pts or snap_pts[-1][2] != n):
        a = path[n - 1]
        snap_pts.append((0.5 * (p1[a["j"]] + p1[p["j"]]), 0.5 * (p2[a["k"]] + p2[p["k"]]), n))
        last_snap = n
    shown = [sp for sp in snap_pts if sp[2] <= n]
    marks.set_data([sp[0] for sp in shown], [sp[1] for sp in shown])
    colour = ORANGE if b == "up" else SLATE
    dot.set_data([p1[p["j"]]], [p2[p["k"]]]); dot.set_color(colour); dot.set_markeredgecolor("white")
    map_title.set_text(f"d = 0.82   ·   φ = ({p1[p['j']]:+.2f}, {p2[p['k']]:+.2f})")
    rod_line.set_data(*shapes[n]); rod_line.set_color(colour)
    rod_title.set_text(f"The rod: {WORD[b]},  q = {p['q']:+.2f},  U = {p['U']:.2f}")
    show_ghost = last_snap is not None and last_snap <= n and p["leg"] == path[last_snap]["leg"]
    ghost.set_data(*(shapes[last_snap - 1] if show_ghost else ([], [])))
    rod_legend.set_visible(show_ghost)
    title3d.set_text(f"The snap surface a {WORD[b]} rod can meet")
    if kind == "snap":
        a = path[n - 1]
        event_text.set_text(f"SNAP: {WORD[a['branch']]} → {WORD[b]},  U {a['U']:.1f} → {p['U']:.1f}")
        event_text.set_color(RED)
    elif kind == "smooth":
        event_text.set_text(f"No snap: smooth crossing through the S-shape, now {WORD[b]}")
        event_text.set_color(TEAL)
    elif kind == "end":
        event_text.set_text("Leg 6 repeats leg 1, but bent down: no snap")
        event_text.set_color(TEAL)
    elif kind == "leg" or (n and path[n]["leg"] != path[max(n - 1, 0)]["leg"]):
        event_text.set_text("")
    return []


# Pick out the poster: the frame right after the first snap.
writer = FFMpegWriter(fps=24, codec="libx264", bitrate=-1,
                      extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart"])
first_snap = next(i for i, (n, kind) in enumerate(frames) if kind == "snap")
with writer.saving(fig, OUT / "branch_walk.mp4", dpi=150):
    for i, frame in enumerate(frames):
        draw(frame)
        writer.grab_frame(facecolor="white")
        if i == first_snap + 10:
            fig.savefig(OUT / "branch_walk_poster.png", dpi=150, facecolor="white")
plt.close(fig)
print("wrote branch_walk.mp4 (", len(frames), "frames ) and branch_walk_poster.png")
