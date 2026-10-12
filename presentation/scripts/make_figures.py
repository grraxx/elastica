"""Figures for presentation/elastica_snap_surface.pptx, from the production run.

usage: python make_figures.py [data/production]

Reads <run>/solution_space_results.npz and <run>/post/ (written by
postprocessing/postprocess.py) and writes PNGs to presentation/figures/.
theta is read row by row, so the 2.4 GB archive is never loaded whole.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, TwoSlopeNorm
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solver"))
from branch import first_mode_projection, is_jump  # noqa: E402
from validate_results import read_rows  # noqa: E402

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data" / "production"
OUT = REPO / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 0.17                                      # the solver's jump_tol, on q

INK, SLATE, ORANGE, TEAL, RED, GREY, LIGHT = (
    "#16222E", "#2E4A5C", "#D9822B", "#2A9D8F", "#C8553D", "#5C6B7A", "#EEF2F4")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.titlecolor": INK, "axes.titlesize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})
FIELD = LinearSegmentedColormap.from_list("field", ["#16222E", "#2E4A5C", TEAL, "#E9C46A", ORANGE])
UPDOWN = LinearSegmentedColormap.from_list("updown", [SLATE, "#9DB0BF", "#F7F7F7", "#F0C08F", ORANGE])
PI_TICKS = ([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2], ["−π/2", "−π/4", "0", "π/4", "π/2"])

# ------------------------------------------------------------------ data
RESULTS = RUN / "solution_space_results.npz"
with np.load(RESULTS) as r:
    shape = tuple(int(n) for n in r["shape"])
    d_vals, p1, p2, s = r["d_vals"], r["phi1_vals"], r["phi2_vals"], r["beam_x"]
    F = {k: r[k].reshape(shape) for k in ("fx", "fy", "energy", "p", "q", "solve_order", "waited")}
EXTENT = (p1[0], p1[-1], p2[0], p2[-1])
ZERO = int(np.argmin(np.abs(p1)))                 # phi = 0 column


def layer(d):
    return int(np.argmin(np.abs(d_vals - d)))


def node(d, phi1, phi2):
    i, j, k = layer(d), int(np.argmin(np.abs(p1 - phi1))), int(np.argmin(np.abs(p2 - phi2)))
    return i, j, k, int(np.ravel_multi_index((i, j, k), shape))


def centreline(theta):
    x = np.concatenate(([0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * np.diff(s))))
    y = np.concatenate(([0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * np.diff(s))))
    return x, y


def pi_axes(ax, y=True):
    ax.set_xticks(*PI_TICKS)
    if y:
        ax.set_yticks(*PI_TICKS)
    ax.set_xlabel("φ₁")


def snap_edges(i):
    """Midpoints of in-layer grid edges at layer i: snaps and smooth sign changes of p."""
    Q = F["q"][i]
    out = {"snap": [], "smooth": []}
    for axis in (0, 1):
        a, b = (Q[:-1], Q[1:]) if axis == 0 else (Q[:, :-1], Q[:, 1:])
        change = np.sign(a) != np.sign(b)
        snap = change & is_jump(a, b, TOL)
        for name, mask in (("snap", snap), ("smooth", change & ~snap)):
            j, k = np.nonzero(mask)
            x = 0.5 * (p1[j] + p1[j + 1]) if axis == 0 else p1[j]
            y = p2[k] if axis == 0 else 0.5 * (p2[k] + p2[k + 1])
            out[name].append(np.column_stack((x, y)))
    return {k: np.concatenate(v) for k, v in out.items()}


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# The snap on the line d = 0.90, phi2 = 0 ------------------------------------------------
I90 = layer(0.90)
line_q = F["q"][I90, :, ZERO]
jumps = np.flatnonzero(is_jump(line_q[:-1], line_q[1:], TOL)
                       & (np.sign(line_q[:-1]) != np.sign(line_q[1:])))
J = int(jumps[0])                                   # snap between J and J + 1
pair = [node(0.90, p1[J], 0.0), node(0.90, p1[J + 1], 0.0)]

# Rows of theta needed below, read in one pass.
shapes = {"up": node(0.90, 0.30, -0.30), "down": node(0.90, -0.70, 0.70),
          "S": node(0.90, 0.90, 0.90), "mirror": node(0.90, 0.45, -0.30)}
rows = [n[3] for n in pair] + [n[3] for n in shapes.values()]
theta = dict(zip(rows, read_rows(RESULTS, "theta", rows)))

# F1: two equilibria 0.01 rad apart -------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
for (i, j, k, n), colour in zip(pair, (SLATE, ORANGE)):
    x, y = centreline(theta[n])
    branch = "bent up" if F["p"][i, j, k] > 0 else "bent down"
    ax.plot(x, y, color=colour, lw=2.6,
            label=f"φ₁ = {p1[j]:+.3f}:  U = {F['energy'][i, j, k]:.2f},  {branch}")
ax.plot([0, d_vals[I90]], [0, 0], "o", color=INK, ms=6, zorder=5)
ax.axhline(0, color="#D5DBE0", lw=0.8, zorder=0)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
save(fig, "snap_pair.png")

# F2: bent up, bent down and the S-shape, with the first mode ----------------------------
fig, ax = plt.subplots(figsize=(5.8, 3.9), constrained_layout=True)
for key, colour, label in (("up", ORANGE, "bent up"), ("down", SLATE, "bent down"),
                           ("S", GREY, "S-shape")):
    i, j, k, n = shapes[key]
    x, y = centreline(theta[n])
    p = float(first_mode_projection(theta[n], s))
    ax.plot(x, y, color=colour, lw=2.4,
            label=f"{label}: φ = ({p1[j]:+.2f}, {p2[k]:+.2f}),  "
                  + ("p ≈ 0" if abs(p) < 5e-4 else f"p = {p:+.3f}"))
ax.plot(s * d_vals[I90], 0.08 * (1 - np.cos(2 * np.pi * s)), color=GREY, lw=1.1, ls=":",
        label="first mode w₁ = 1 − cos 2πs (scaled)")
ax.axhline(0, color="#D5DBE0", lw=0.8, zorder=0)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("d = 0.90", fontsize=10)
ax.legend(fontsize=8, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.2), ncol=1)
save(fig, "rods_p.png")

# F3: U and q along phi1 at d = 0.90, phi2 = 0 -------------------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 4.3), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": (1, 1.1)})
U_line = F["energy"][I90, :, ZERO]
a1.plot(p1, U_line, color=INK, lw=1.6)
a1.annotate(f"U jumps {U_line[J]:.2f} → {U_line[J + 1]:.2f}", (p1[J + 1], U_line[J + 1]),
            (p1[J + 1] + 0.25, U_line[J + 1] + 1.5), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
a1.set_ylabel("U")
a2.axhspan(-TOL, TOL, color=LIGHT, zorder=0)
a2.axhline(0, color="#C9D1D8", lw=0.8)
a2.plot(p1[: J + 1], line_q[: J + 1], color=SLATE, lw=2)
a2.plot(p1[J + 1:], line_q[J + 1:], color=ORANGE, lw=2)
a2.plot([p1[J], p1[J + 1]], [line_q[J], line_q[J + 1]], color=RED, lw=1, ls="--")
a2.annotate(f"snap: q {line_q[J]:+.2f} → {line_q[J + 1]:+.2f}", (p1[J], line_q[J]),
            (p1[J] + 0.3, -0.75), fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
a2.text(p1[-1], TOL + 0.05, f"±{TOL}", ha="right", fontsize=8, color=GREY)
a2.set_ylabel("q = p / a(d)")
pi_axes(a2, y=False)
a2.set_xlabel("φ₁   (d = 0.90, φ₂ = 0)")
save(fig, "jump_line.png")

# F4: energy and axial force on the d = 0.90 layer ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), constrained_layout=True)
for ax, key, title, norm in (
        (axes[0], "energy", "Bending energy U (log scale)", LogNorm(F["energy"][I90].min(), F["energy"][I90].max())),
        (axes[1], "fx", "Axial force fₓ",
         TwoSlopeNorm(0.0, F["fx"][I90].min(), F["fx"][I90].max()))):
    cmap = FIELD if key == "energy" else LinearSegmentedColormap.from_list(
        "fx", [TEAL, "#BFE3DE", "#F7F7F7", "#F0C08F", ORANGE])
    im = ax.imshow(F[key][I90].T, origin="lower", extent=EXTENT, cmap=cmap, norm=norm,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    if key == "energy":
        cb.set_ticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
        cb.minorticks_off()
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    pi_axes(ax); ax.set_ylabel("φ₂"); ax.set_aspect("equal")
save(fig, "fields_d090.png")

# F5: branch map at d = 0.90 and the snap curve on every fifth layer ---------------------
fig, ax = plt.subplots(figsize=(4.6, 4.1), constrained_layout=True)
im = ax.imshow(F["q"][I90].T, origin="lower", extent=EXTENT, cmap=UPDOWN,
               norm=TwoSlopeNorm(0, -1.15, 1.15), interpolation="nearest")
e = snap_edges(I90)
ax.plot(*e["snap"].T, ".", ms=1.5, color=RED)
ax.plot(*e["smooth"].T, ".", ms=1.5, color=INK)
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("q   (orange: bent up, slate: bent down)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
ax.set_title("d = 0.90")
pi_axes(ax); ax.set_ylabel("φ₂"); ax.set_aspect("equal")
save(fig, "branch_map_d090.png")

fig, ax = plt.subplots(figsize=(4.6, 4.1), constrained_layout=True)
picks = [layer(d) for d in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)]
colours = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(picks)))
for i, colour in zip(picks, colours):
    ax.plot(*snap_edges(i)["snap"].T, ".", ms=1.6, color=colour, label=f"d = {d_vals[i]:.2f}")
ax.plot([p1[0], p1[-1]], [p2[-1], p2[0]], ":", color=GREY, lw=0.8)
ax.text(-1.5, 1.42, "φ₁ = −φ₂", fontsize=8, color=GREY, rotation=-45, va="top")
ax.set_xlim(EXTENT[:2]); ax.set_ylim(EXTENT[2:])
pi_axes(ax); ax.set_ylabel("φ₂"); ax.set_aspect("equal")
ax.set_title("Snap curve per d layer")
ax.legend(fontsize=7.5, frameon=False, markerscale=5, loc="lower right")
save(fig, "snap_curves.png")

# F6: the front at d = 0.99: solve order and the nodes that had to wait ------------------
I99 = layer(0.99)
order = F["solve_order"][I99].astype(float)
rank = np.argsort(np.argsort(order.ravel())).reshape(order.shape) / order.size
fig, ax = plt.subplots(figsize=(4.6, 4.1), constrained_layout=True)
im = ax.imshow(rank.T, origin="lower", extent=EXTENT, cmap=LinearSegmentedColormap.from_list(
    "order", ["#F7F7F7", "#9DB0BF", SLATE]), interpolation="nearest")
j, k = np.nonzero(F["waited"][I99] > 0.5)
ax.plot(p1[j], p2[k], "s", ms=1.4, color=ORANGE, label=f"deferred first ({len(j)} nodes)")
ax.plot(*snap_edges(I99)["snap"].T, ".", ms=1.2, color=RED, label="snap")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("order in which the layer was solved", fontsize=8.5)
cb.set_ticks([0, 1], labels=["first", "last"]); cb.ax.tick_params(labelsize=8)
ax.set_title("d = 0.99")
pi_axes(ax); ax.set_ylabel("φ₂"); ax.set_aspect("equal")
ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, markerscale=4, loc="lower left")
save(fig, "front_d099.png")

# F7: the mirror map ---------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1, 1.35)})
h = np.pi / 2
a1.fill([-h, h, h], [h, h, -h], color="#D9EFEC")
a1.plot([-h, h], [h, -h], ":", color=GREY, lw=0.8)
a1.text(0.55, 0.75, "solved\nφ₁ + φ₂ ≥ 0", ha="center", fontsize=8.5, color=TEAL)
a1.text(-0.75, -0.95, "mirrored", ha="center", fontsize=8.5, color=GREY)
i, j, k, n = shapes["mirror"]
a1.plot(p1[j], p2[k], "o", color=INK, ms=5)
a1.plot(-p2[k], -p1[j], "o", color=ORANGE, ms=5)
a1.annotate("", (-p2[k], -p1[j]), (p1[j], p2[k]),
            arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
a1.set_xlim(-h, h); a1.set_ylim(-h, h); a1.set_aspect("equal")
pi_axes(a1); a1.set_ylabel("φ₂")
th = theta[n]
a2.plot(s, th, color=INK, lw=2, label=f"θ(s) at (φ₁, φ₂) = ({p1[j]:.2f}, {p2[k]:.2f})")
a2.plot(s, -th[::-1], color=ORANGE, lw=2, label=f"−θ(L − s) at ({-p2[k]:.2f}, {-p1[j]:.2f})")
a2.axhline(0, color="#D5DBE0", lw=0.8, zorder=0)
a2.set_xlabel("s"); a2.set_ylabel("θ")
a2.legend(fontsize=8, frameon=False, loc="lower left")
a2.set_title("d = 0.90", fontsize=10)
save(fig, "mirror.png")

# 3D: the p = 0 surface ------------------------------------------------------------------
pv.OFF_SCREEN = True
ZS = 8.0
surface = pv.read(RUN / "post" / "snap_surface.vtp").triangulate()
is_snap = np.asarray(surface.point_data["snap"])[surface.regular_faces].mean(axis=1) > 0.5
surface.points[:, 2] *= ZS
# Only the snap part: the smooth p = 0 set includes the whole S-shape region on the
# diagonal phi1 = phi2, where p vanishes identically. Taubin smoothing removes the
# grid terraces of the contour without shrinking the sheet.
snap_part = (surface.extract_cells(np.flatnonzero(is_snap)).extract_surface(algorithm="dataset_surface")
             .clean().smooth_taubin(n_iter=80, pass_band=0.02))
BOUNDS = (p1[0], p1[-1], p2[0], p2[-1], d_vals[0] * ZS, d_vals[-1] * ZS)
frame = pv.Box(bounds=BOUNDS).extract_all_edges()


def render(name, meshes, azimuth, elevation, size=(1500, 1250), zoom=1.0, parallel=False,
           d_axis=True):
    plotter = pv.Plotter(off_screen=True, window_size=size)
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    for mesh, kwargs in meshes:
        plotter.add_mesh(mesh, **kwargs)
    plotter.add_mesh(frame, color="#9AA5B1", line_width=1.5)
    plotter.show_bounds(bounds=BOUNDS, xtitle="phi1", ytitle="phi2", ztitle="", color=INK,
                        font_size=12, location="outer", grid=False, n_xlabels=3, n_ylabels=3,
                        show_zlabels=False, fmt="%.2f")
    if d_axis:            # d is drawn ZS times taller; label its edge with true d
        az = np.radians(azimuth)
        right = np.array([-np.sin(az), np.cos(az)])
        corners = [(x, y) for x in BOUNDS[:2] for y in BOUNDS[2:4]]
        corner = min(corners, key=lambda c: float(np.dot(c, right)))
        lo, hi = BOUNDS[4] / ZS, BOUNDS[5] / ZS
        ticks = [lo] + [t / 10 for t in range(int(np.floor(lo * 10)) + 1, int(np.ceil(hi * 10)))
                        if lo + 0.02 < t / 10 < hi - 0.02] + [hi]
        plotter.add_point_labels(
            np.array([(*corner, t * ZS) for t in ticks] + [(*corner, BOUNDS[5] + 0.3)]),
            [f"{t:.2f}" for t in ticks] + ["d"], font_size=16, text_color=INK,
            show_points=False, shape=None, always_visible=True)
    centre = np.array([0.0, 0.0, 0.845 * ZS])
    a, e = np.radians(azimuth), np.radians(elevation)
    eye = centre + 11.5 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    plotter.camera_position = [tuple(eye), tuple(centre), (0.0, 0.0, 1.0) if elevation < 89
                               else (0.0, 1.0, 0.0)]
    if parallel:
        plotter.enable_parallel_projection()
    plotter.camera.zoom(zoom)
    plotter.screenshot(OUT / name)
    plotter.close()
    crop(OUT / name)
    print("wrote", name)


def crop(path, pad=12):
    image = Image.open(path).convert("RGB")
    box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    if box:
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
               min(box[2] + pad, image.width), min(box[3] + pad, image.height))
        image.crop(box).save(path)


lit = dict(smooth_shading=True, specular=0.2, ambient=0.35, diffuse=0.75)
snap_part.point_data["d"] = snap_part.points[:, 2] / ZS
render("snap3d.png", [(snap_part, dict(scalars="d", cmap="viridis", clim=(d_vals[0], d_vals[-1]),
                                       show_scalar_bar=False, **lit))],
       azimuth=-50, elevation=24)
render("snap3d_top.png", [(snap_part, dict(scalars="d", cmap="viridis", clim=(d_vals[0], d_vals[-1]),
                                           scalar_bar_args=dict(title="d", color=INK, vertical=True,
                                                                position_x=0.86, position_y=0.2,
                                                                height=0.6, width=0.05,
                                                                fmt="%.2f", label_font_size=16,
                                                                title_font_size=18),
                                           **lit))],
       azimuth=-90, elevation=90, size=(1400, 1250), parallel=True, zoom=1.35, d_axis=False)

# 3D: the axial-force step under the sheet -----------------------------------------------
plane = pv.ImageData(dimensions=(len(p1), len(p2), 1), spacing=(p1[1] - p1[0], p2[1] - p2[0], 1),
                     origin=(p1[0], p2[0], 0.90 * ZS))
plane.point_data["fx"] = F["fx"][I90].ravel(order="F")
upper = snap_part.clip(normal=(0, 0, -1), origin=(0, 0, 0.90 * ZS))
fx_cmap = LinearSegmentedColormap.from_list("fx", [TEAL, "#BFE3DE", "#F7F7F7", "#F0C08F", ORANGE])
render("fx_step3d.png",
       [(plane, dict(scalars="fx", cmap=fx_cmap, clim=(-80, 80), smooth_shading=False,
                     scalar_bar_args=dict(title="fx", color=INK, vertical=True, position_x=0.87,
                                          position_y=0.25, height=0.5, width=0.04, fmt="%.0f",
                                          label_font_size=16, title_font_size=18))),
        (upper, dict(color=RED, opacity=0.35, **lit))],
       azimuth=-50, elevation=30)

# Production level-set figures from postprocess.py, cropped ------------------------------
for name in ("iso_energy", "iso_force_fx", "iso_force_fy", "iso_moment"):
    for suffix in ("", "_layers"):
        source = RUN / "post" / f"{name}{suffix}.png"
        image = Image.open(source).convert("RGB")
        image.save(OUT / source.name)
        crop(OUT / source.name)
        print("wrote", source.name)
