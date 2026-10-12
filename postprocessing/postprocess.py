"""Post-process a results archive: a ParaView VTU and four families of surfaces.

Input: solution_space_results.npz from solver/main.py (no theta needed).
Points are written as x = phi1, y = phi2, z = d (unscaled; scale z by about 10 in ParaView).

Writes to the output directory:
    elastica_solution_space.vtu   every node and tetrahedron, with point fields
        p, q            projection on the first clamped mode; q = p / a(d) (branch.py)
        branch          sign of p: +1 bent up, -1 bent down
        energy          bending energy / EI (L = 1)
        fx, fy          end force components / (EI / L^2); fx > 0 compresses the rod
        force           |(fx, fy)|
        moment_left     theta'(0), clamp moment / (EI / L)
        moment_right    theta'(L)
        snap            1 on nodes touching a snap edge (p changes sign with a jump)
        solve_order, waited
      and the cell field smooth (0 on tetrahedra with a jump on an edge).
    snap_surface.vtp/.stl         p = 0, with point field snap (1: snap, 0: smooth crossing)
    snap_surface_jump.stl         only its snap part
    iso_energy.vtp, iso_fx.vtp, iso_fy.vtp, iso_moment_left.vtp, iso_moment_right.vtp
        level sets over the tetrahedra without a jump, so they end at the snap
        surface instead of running along it; point field "level"
    snap_surface.png, snap_layers.png          the p = 0 surface; q per d layer
    iso_energy.png, iso_force_fx.png, iso_force_fy.png, iso_moment.png
        one panel per level, coloured by d; *_layers.png: level lines per d layer
    summary.json                  field ranges and the area of every level set

usage: python postprocess.py results.npz [-o outdir] [--tol 0.17]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "solver"))
from branch import first_mode_amplitude, is_jump  # noqa: E402

vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)
pv.OFF_SCREEN = True
ZS = 8.0                                     # d exaggeration in the 3D renders
BOUNDS = None                                # set in main: the box in render coordinates
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

LAYER_LEVELS = {
    "energy": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64],
    "fx": [-3000, -1000, -300, -100, -30, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80],
    "fy": [-30, -20, -10, -5, 0, 5, 10, 20, 30],
    "moment_left": [-16, -12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12, 16],
}
LEVELS = {
    "energy": [2, 4, 8, 16, 32, 64],
    "fx": [-1000, -100, -10, 0, 20, 40, 60, 80],
    "fy": [-20, -10, -5, 0, 5, 10, 20],
    "moment_left": [-12, -8, -4, 0, 4, 8, 12],
    "moment_right": [-12, -8, -4, 0, 4, 8, 12],
}


def write_stl(path, points, triangles, title):
    tri = points[triangles].astype(np.float32)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    record = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    record["n"], record["v"] = normal, tri
    with open(path, "wb") as file:
        file.write(title.encode()[:80].ljust(80))
        file.write(np.uint32(len(tri)).tobytes())
        file.write(record.tobytes())


def unstructured_grid(points, tets, point_fields, cell_fields):
    grid = vtk.vtkUnstructuredGrid()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    grid.SetPoints(vtk_points)
    offsets = numpy_to_vtk(np.arange(0, 4 * len(tets) + 1, 4, dtype=np.int64), deep=True,
                           array_type=vtk.VTK_ID_TYPE)
    connectivity = numpy_to_vtk(np.ascontiguousarray(tets.ravel(), dtype=np.int64), deep=True,
                                array_type=vtk.VTK_ID_TYPE)
    cells = vtk.vtkCellArray()
    cells.SetData(offsets, connectivity)
    grid.SetCells(vtk.VTK_TETRA, cells)
    for data, fields in ((grid.GetPointData(), point_fields), (grid.GetCellData(), cell_fields)):
        for name, values in fields.items():
            array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
            array.SetName(name)
            data.AddArray(array)
    return grid


def contour(grid, name, levels):
    """Level sets of one point field; point field 'level' holds the value."""
    grid.GetPointData().SetActiveScalars(name)
    flt = vtk.vtkContourGrid() if isinstance(grid, vtk.vtkUnstructuredGrid) else vtk.vtkContourFilter()
    flt.SetInputData(grid)
    for n, level in enumerate(levels):
        flt.SetValue(n, level)
    flt.ComputeScalarsOn()
    flt.Update()
    surface = pv.wrap(flt.GetOutput())
    if surface.n_points:
        surface.point_data["level"] = np.asarray(surface.point_data[name])
    return surface


def area_per_level(surface, levels):
    if not surface.n_cells:
        return {str(level): 0.0 for level in levels}
    sizes = surface.compute_cell_sizes(length=False, volume=False)
    cell_level = np.asarray(surface.point_data["level"])[surface.regular_faces[:, 0]]
    area = np.asarray(sizes.cell_data["Area"])
    return {str(level): float(area[np.isclose(cell_level, level)].sum()) for level in levels}


def set_camera(plotter, azimuth, elevation=24.0, distance=11.5):
    centre = np.array([0.0, 0.0, 0.845 * ZS])
    az, el = np.radians(azimuth), np.radians(elevation)
    eye = centre + distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    plotter.camera_position = [tuple(eye), tuple(centre), (0.0, 0.0, 1.0)]


def render_panels(panels, path, title, size_per_panel=(760, 640), columns=None):
    """panels: list of (subtitle, [(mesh, add_mesh kwargs), ...], azimuth)."""
    columns = columns or len(panels)
    rows = int(np.ceil(len(panels) / columns))
    plotter = pv.Plotter(off_screen=True, shape=(rows, columns), border=False,
                         window_size=(size_per_panel[0] * columns, size_per_panel[1] * rows))
    for n, (subtitle, meshes, azimuth) in enumerate(panels):
        plotter.subplot(n // columns, n % columns)
        plotter.set_background("white")
        for mesh, kwargs in meshes:
            if mesh.n_points:
                plotter.add_mesh(mesh, **kwargs)
        plotter.show_bounds(bounds=BOUNDS, xtitle="phi1", ytitle="phi2", ztitle=" ",
                            color="#16222E", font_size=9, location="outer", grid=False,
                            n_xlabels=3, n_ylabels=3, show_zlabels=False, fmt="%.2f")
        # d is drawn ZS times taller; label the vertical edge with true d.
        lo, hi = BOUNDS[4] / ZS, BOUNDS[5] / ZS
        ticks = [lo] + [t / 10 for t in range(int(np.floor(lo * 10)) + 1, int(np.ceil(hi * 10)))
                        if lo + 0.02 < t / 10 < hi - 0.02] + [hi]
        right = np.array([-np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))])
        corners = [(x, y) for x in BOUNDS[:2] for y in BOUNDS[2:4]]
        corner = min(corners, key=lambda c: float(np.dot(c, right)))     # left silhouette edge
        plotter.add_point_labels(
            np.array([(*corner, t * ZS) for t in ticks] + [(*corner, BOUNDS[5] + 0.35)]),
            [f"{t:.2f}" for t in ticks] + ["d"], font_size=13, text_color="#16222E",
            show_points=False, shape=None, always_visible=True)
        set_camera(plotter, azimuth)
        plotter.add_text(subtitle, position="upper_edge", font_size=11, color="#16222E")
    plotter.subplot(0, 0)
    if title:
        plotter.add_text(title, position="lower_left", font_size=9, color="#5C6B7A")
    plotter.screenshot(path)
    plotter.close()


def scaled(mesh):
    out = mesh.copy()
    if out.n_points:
        out.points[:, 2] *= ZS
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", type=Path)
    ap.add_argument("-o", "--output-dir", type=Path)
    ap.add_argument("--tol", type=float, default=0.17, help="the solver's jump_tol, on q")
    args = ap.parse_args()
    out = args.output_dir or args.results.parent / "post"
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.results) as r:
        shape = tuple(int(n) for n in r["shape"])
        d_vals, p1_vals, p2_vals = (r[k] for k in ("d_vals", "phi1_vals", "phi2_vals"))
        coords, tets = r["coords"], r["tets"].astype(np.int64)
        f = {k: r[k] for k in ("fx", "fy", "moment", "energy", "p", "solve_order", "waited",
                               "solved")}
        moment_right = r["moment_right"] if "moment_right" in r.files else None
    if not (f["solved"] > 0.5).all():
        raise SystemExit(f"{np.count_nonzero(f['solved'] < 0.5):,} nodes are unsolved")
    n = shape[1]
    if moment_right is None:
        i, j, k = np.unravel_index(np.arange(len(coords)), shape)
        moment_right = f["moment"][np.ravel_multi_index((i, n - 1 - k, n - 1 - j), shape)]
    amplitude = first_mode_amplitude(d_vals)[np.arange(len(coords)) // (shape[1] * shape[2])]
    q = f["p"] / amplitude
    points = coords[:, [1, 2, 0]].copy()                       # x = phi1, y = phi2, z = d

    # Jumps on tetrahedron edges: the snap indicator and the smooth cells.
    jump_cell = np.zeros(len(tets), dtype=bool)
    snap_node = np.zeros(len(coords), dtype=np.uint8)
    for a, b in EDGES:
        qa, qb = q[tets[:, a]], q[tets[:, b]]
        jump = is_jump(qa, qb, args.tol)
        jump_cell |= jump
        snap = jump & (np.sign(qa) != np.sign(qb))
        snap_node[tets[snap, a]] = 1
        snap_node[tets[snap, b]] = 1
    print(f"{len(tets):,} tetrahedra, {jump_cell.sum():,} with a jump on an edge "
          f"({jump_cell.mean():.2%}); {snap_node.sum():,} snap nodes")

    point_fields = {
        "p": f["p"], "q": q, "branch": np.sign(f["p"]), "energy": f["energy"],
        "fx": f["fx"], "fy": f["fy"], "force": np.hypot(f["fx"], f["fy"]),
        "moment_left": f["moment"], "moment_right": moment_right,
        "snap": snap_node.astype(np.float64), "solve_order": f["solve_order"].astype(np.float64),
        "waited": f["waited"].astype(np.float64),
    }
    grid = unstructured_grid(points, tets, point_fields,
                             {"smooth": (~jump_cell).astype(np.uint8)})
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(out / "elastica_solution_space.vtu"))
    writer.SetInputData(grid)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    writer.Write()
    print(f"wrote {out / 'elastica_solution_space.vtu'}")

    summary = {"tol_q": args.tol, "nodes": len(coords), "tetrahedra": len(tets),
               "jump_tetrahedra": int(jump_cell.sum()),
               "ranges": {k: [float(np.min(v)), float(np.max(v))] for k, v in point_fields.items()},
               "bent_up_per_layer": {f"{d:.2f}": float((f["p"].reshape(shape)[i] > 0).mean())
                                     for i, d in enumerate(d_vals)}}

    # 1. Snap surface: p = 0, coloured by whether p jumps there.
    snap_surface = contour(grid, "p", [0.0])
    snap_surface.point_data["snap"] = (np.asarray(snap_surface.point_data["snap"]) > 0.5).astype(float)
    tri = snap_surface.triangulate().regular_faces
    is_snap = np.asarray(snap_surface.point_data["snap"])[tri].mean(axis=1) > 0.5
    snap_surface.save(out / "snap_surface.vtp")
    write_stl(out / "snap_surface.stl", snap_surface.points, tri,
              "p = 0 surface of the clamped elastica, x = phi1, y = phi2, z = d")
    write_stl(out / "snap_surface_jump.stl", snap_surface.points, tri[is_snap],
              "snap part of the p = 0 surface, x = phi1, y = phi2, z = d")
    sizes = snap_surface.triangulate().compute_cell_sizes(length=False, volume=False)
    area = np.asarray(sizes.cell_data["Area"])
    summary["snap_surface"] = {"triangles": int(len(tri)), "snap_triangles": int(is_snap.sum()),
                               "area_snap": float(area[is_snap].sum()),
                               "area_smooth": float(area[~is_snap].sum())}
    global BOUNDS
    BOUNDS = (p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1], d_vals[0] * ZS, d_vals[-1] * ZS)
    surf = scaled(snap_surface.triangulate())
    snap_part = surf.extract_cells(np.flatnonzero(is_snap)).extract_surface()
    smooth_part = surf.extract_cells(np.flatnonzero(~is_snap)).extract_surface()
    box = pv.Box(bounds=BOUNDS).extract_all_edges()
    frame = (box, dict(color="#9AA5B1", line_width=1))
    snap_style = dict(color="#C8553D", smooth_shading=True, specular=0.25)
    smooth_style = dict(color="#7F8C99", smooth_shading=True, specular=0.25)
    render_panels([(view, [(snap_part, snap_style), (smooth_part, smooth_style), frame], az)
                   for view, az in (("from phi1 > 0, phi2 < 0", -50), ("from phi1 > 0, phi2 > 0", 40),
                                    ("from above", -50.0001))][:2],
                  out / "snap_surface.png",
                  "p = 0 surface. Red: snap (p jumps, bent up <-> bent down). "
                  "Grey: smooth zero crossing through an S-shape.", size_per_panel=(1000, 820))

    # Per-layer maps of q, with snap nodes.
    P = f["p"].reshape(shape); Q = q.reshape(shape); S = snap_node.reshape(shape)
    picks = np.unique(np.linspace(0, shape[0] - 1, 6).round().astype(int))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), constrained_layout=True)
    for ax, i in zip(axes.flat, picks):
        im = ax.imshow(Q[i].T, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0, -1.6, 1.6),
                       extent=(p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]))
        ax.contour(p1_vals, p2_vals, P[i].T, [0.0], colors="k", linewidths=0.6)
        j, k = np.nonzero(S[i])
        ax.plot(p1_vals[j], p2_vals[k], ".", ms=1.2, color="#FFD23F")
        ax.set_title(f"d = {d_vals[i]:.2f}: bent up {(P[i] > 0).mean():.0%}")
        ax.set_xlabel("φ1"); ax.set_ylabel("φ2"); ax.set_aspect("equal")
    fig.colorbar(im, ax=axes, shrink=0.7, label="q = p / a(d)   (red: bent up, blue: bent down)")
    fig.suptitle("Branch per d layer; black: p = 0, yellow: snap nodes (p changes sign with a jump)")
    fig.savefig(out / "snap_layers.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2-4. Level sets over the jump-free tetrahedra.
    smooth_grid = grid
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(grid)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "smooth")
    threshold.SetLowerThreshold(0.5); threshold.SetUpperThreshold(1.5)
    threshold.Update()
    smooth_grid = threshold.GetOutput()
    print(f"jump-free grid: {smooth_grid.GetNumberOfCells():,} tetrahedra")
    surfaces, summary["level_sets"] = {}, {}
    for name, levels in LEVELS.items():
        lo, hi = summary["ranges"][name]
        levels = [level for level in levels if lo < level < hi]
        surfaces[name] = contour(smooth_grid, name, levels)
        surfaces[name].save(out / f"iso_{name}.vtp")
        summary["level_sets"][name] = area_per_level(surfaces[name], levels)
        print(f"iso_{name}: {surfaces[name].n_cells:,} triangles, levels {levels}")
    ghost = (snap_part, dict(color="#C8553D", opacity=0.12))
    titles = {"energy": "U", "fx": "fx", "fy": "fy", "moment_left": "M1", "moment_right": "M2"}
    picks = {"energy": [2, 8, 16, 64], "fx": [-100, 0, 20, 60], "fy": [-10, 0, 10, 20],
             "moment_left": [-8, -4, 0, 4], "moment_right": [-8, -4, 0, 4]}
    captions = {
        "energy": "Constant bending energy U = 1/2 int theta'^2 ds (EI = L = 1). Colour: d. "
                  "Faint red: snap part of the p = 0 surface; level sets stop at it.",
        "fx": "Constant axial end force fx (> 0 compression, < 0 tension). Colour: d.",
        "fy": "Constant transverse end force fy. Colour: d.",
        "moment_left": "Constant clamp moment M1 = theta'(0) at the left clamp. Colour: d. "
                       "M2 = theta'(L) is its mirror image under (phi1, phi2) -> (-phi2, -phi1).",
    }
    files = {"energy": "iso_energy.png", "fx": "iso_force_fx.png", "fy": "iso_force_fy.png",
             "moment_left": "iso_moment.png"}
    for name, file in files.items():
        surface = scaled(surfaces[name])
        panels = []
        for level in picks[name]:
            part = surface.threshold((level - 1e-9, level + 1e-9), scalars="level").extract_surface() \
                if surface.n_points else surface
            if not part.n_points:
                continue
            part.point_data["d"] = part.points[:, 2] / ZS
            panels.append((f"{titles[name]} = {level:g}",
                           [(part, dict(scalars="d", cmap="viridis", clim=(d_vals[0], d_vals[-1]),
                                        smooth_shading=True, show_scalar_bar=False)),
                            ghost, frame], -50))
        render_panels(panels, out / file, captions[name], size_per_panel=(760, 680))

    # Layer maps of each field, contour lines at the levels, broken at snap nodes.
    layer_picks = np.unique(np.linspace(0, shape[0] - 1, 4).round().astype(int))
    for name, file in files.items():
        F = point_fields[name].reshape(shape).astype(float).copy()
        F[S.astype(bool)] = np.nan
        levels = [float(v) for v in LAYER_LEVELS[name]
                  if summary["ranges"][name][0] < v < summary["ranges"][name][1]]
        line_colours = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 0.75, 256)))
        fig, axes = plt.subplots(1, len(layer_picks), figsize=(4.4 * len(layer_picks), 4.6),
                                 constrained_layout=True)
        for ax, i in zip(axes, layer_picks):
            sign = np.where(P[i] > 0, 1.0, -1.0)
            ax.imshow(sign.T, origin="lower", cmap=ListedColormap(["#DCE4EC", "#F8E3CF"]),
                      extent=(p1_vals[0], p1_vals[-1], p2_vals[0], p2_vals[-1]), vmin=-1, vmax=1)
            cs = ax.contour(p1_vals, p2_vals, F[i].T, levels=levels, cmap=line_colours,
                            linewidths=1.1)
            ax.clabel(cs, fmt="%g", fontsize=7)
            j, k = np.nonzero(S[i])
            ax.plot(p1_vals[j], p2_vals[k], ".", ms=0.8, color="#C8553D")
            ax.set_title(f"d = {d_vals[i]:.2f}"); ax.set_xlabel("φ1"); ax.set_aspect("equal")
        axes[0].set_ylabel("φ2")
        fig.suptitle(f"{titles[name]} level lines per d layer (background: bent up orange, "
                     "bent down blue; red: snap)")
        fig.savefig(out / file.replace(".png", "_layers.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["snap_surface"]), json.dumps(summary["level_sets"]))


if __name__ == "__main__":
    main()
