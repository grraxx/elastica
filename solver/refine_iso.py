"""
refine_iso.py
The iso-energy scheme of refine.py's sample stage (config "scheme": "iso"): new
points on level sets of the up branch's bending energy U, regularly spaced on
each surface, as many per level as a target energy histogram asks.

Target histogram. The dataset (grid rows of both branches plus the new points'
four images) should hold

    H(U) = C U^-alpha   rows per unit U, for U_min <= U <= U_peak,

and the grid alone above U_peak, where the grid's own histogram peaks and then
falls. C = scale * H_grid(U_peak) U_peak^alpha, so at scale 1 the power law
meets the grid at its peak and the whole histogram decreases monotonically.
alpha = 1 (the default) gives every decade of U the same number of rows: a
network fitting U to a fixed relative accuracy (MSE on log U, or on U weighted
by 1/U^2) then has as much data at U ~ 0.1 as at U ~ 1, which means ten times
the density per unit U, and an absolute error that shrinks with U.

Levels. U_k = sqrt(e_k e_k+1) on edges e_k = U_min (1 + level_step)^k up to
U_peak. Level k gets the rows H assigns to [e_k, e_k+1) less the grid rows
already there, divided by four (each solve gives four rows by symmetry).

Surfaces. Marching tetrahedra (the explorer's 6-tetrahedron split of every
cell) over the tetrahedra whose four corners hold the up branch with no jump in
q between them, so a surface ends at the fold. Only phi1 + phi2 >= 0 is kept;
the mirror gives the other half.

Spacing. The surface's own vertices (one per crossed grid edge) come first,
then a barycentric lattice in every triangle at half the target spacing. A
greedy Poisson-disk selection (vertices first, then lattice points, each in a
random order) keeps a point only if no kept point lies within r. The kept set
has no two points closer than r and no gap wider than r, so it is regularly
spaced on the surface. r is tuned per level until the count matches.

Solve. Each point is warm-started from the cell corners that hold up, nearest
first, with main.py's tests (no jump in q, stable). The solve lands near but
not on U_k, because the surface came from linear interpolation; with
project_steps > 0 the point then moves along the energy gradient,
dU = -fx dd - M1 dphi1 + M2 dphi2 (the envelope theorem), and is re-solved from
the state just found, until |U - U_k| < project_tol U_k.
"""

import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import refine
from branch import is_jump
from refine import FIELDS, Branches, band_counts, save_npz
from stability import unstable_directions

# Corner offsets of the six tetrahedra of a cell, around its 000-111 diagonal
# (postprocessing/web/server.py uses the same split).
_C = {"000": (0, 0, 0), "100": (1, 0, 0), "010": (0, 1, 0), "001": (0, 0, 1),
      "110": (1, 1, 0), "101": (1, 0, 1), "011": (0, 1, 1), "111": (1, 1, 1)}
CELL_TETS = np.array([[_C[c] for c in tet] for tet in (
    ("000", "100", "110", "111"), ("000", "110", "010", "111"), ("000", "010", "011", "111"),
    ("000", "011", "001", "111"), ("000", "001", "101", "111"), ("000", "101", "100", "111"))])
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
SOURCE_ISO = 6
PLAN_KEYS = ("alpha", "scale", "u_max", "level_step", "min_spacing", "count_tol")
DEFAULTS = {"alpha": 1.0, "scale": 1.0, "u_max": None, "level_step": 0.05,
            "min_spacing": 0.05, "count_tol": 0.03, "project_steps": 4, "project_tol": 1e-4}


def settings(run):
    return {**DEFAULTS, **run.config.get("iso", {})}


# ---------------------------------------------------------------- target

class Target:
    """The target histogram of dataset rows over U, and the levels."""

    def __init__(self, grid_rows, cfg):
        self.alpha, self.grid_rows = float(cfg["alpha"]), np.sort(grid_rows)
        self.u_min = float(self.grid_rows[0])
        # The grid's density per unit U, on log bins (20 per decade), to find its peak.
        edges = np.geomspace(self.u_min, self.grid_rows[-1] * (1 + 1e-9), 1 + int(np.ceil(
            20 * np.log10(self.grid_rows[-1] / self.u_min))))
        density = np.histogram(self.grid_rows, edges)[0] / np.diff(edges)
        peak = int(np.argmax(density))
        self.u_peak = float(np.sqrt(edges[peak] * edges[peak + 1]))
        self.c = float(cfg["scale"]) * density[peak] * self.u_peak ** self.alpha
        self.u_max = float(cfg["u_max"] or self.u_peak)
        ratio = 1.0 + float(cfg["level_step"])
        n = int(np.ceil(np.log(self.u_max / self.u_min) / np.log(ratio)))
        self.edges = np.append(self.u_min * ratio ** np.arange(n), self.u_max)
        self.levels = np.sqrt(self.edges[:-1] * self.edges[1:])
        grid_in = np.diff(np.searchsorted(self.grid_rows, self.edges))
        self.rows = np.maximum(0.0, self.integral(self.edges[:-1], self.edges[1:]) - grid_in)
        self.points = np.rint(self.rows / 4).astype(np.int64)        # canonical solves

    def density(self, u):
        return self.c * np.asarray(u, dtype=float) ** -self.alpha

    def integral(self, lo, hi):
        if np.isclose(self.alpha, 1.0):
            return self.c * np.log(hi / lo)
        return self.c * (hi ** (1 - self.alpha) - lo ** (1 - self.alpha)) / (1 - self.alpha)


# ---------------------------------------------------------------- surfaces

class Tets:
    """The canonical tetrahedra over which the up branch's energy is linear and
    jump-free, with what marching tetrahedra needs."""

    def __init__(self, br, tol):
        shape, cells = br.shape, br.cell_shape
        nc = cells[1]
        ci, cj, ck = np.unravel_index(np.arange(int(np.prod(cells))), cells)
        keep = np.flatnonzero(cj + ck >= nc - 1)           # canonical cells, the diagonal ones too
        cell = np.repeat(keep, 6)
        kind = np.tile(np.arange(6), len(keep))
        base = np.ravel_multi_index((ci[cell], cj[cell], ck[cell]), shape)
        step = CELL_TETS @ np.array([shape[1] * shape[2], shape[2], 1])          # (6, 4)
        nodes = base[:, None] + step[kind]                                        # (t, 4)
        del base
        ok = br.exists[nodes].all(axis=1)
        q = br.up["q"]
        for a, b in TET_EDGES:
            ok[ok] &= ~is_jump(q[nodes[ok, a]], q[nodes[ok, b]], tol)
        self.nodes, self.cell, self.kind = nodes[ok], cell[ok], kind[ok].astype(np.int8)
        self.energy = br.up["energy"]
        e = self.energy[self.nodes]
        self.lo, self.hi = e.min(axis=1), e.max(axis=1)
        self.shape, self.n_phi = shape, shape[1]
        # Gradient of the linear interpolant per tetrahedron kind, in grid steps:
        # grad = inv(P1 - P0, P2 - P0, P3 - P0) (E1 - E0, E2 - E0, E3 - E0).
        self.inv = np.linalg.inv(CELL_TETS[:, 1:] - CELL_TETS[:, :1])
        print(f"{len(self.nodes):,} jump-free tetrahedra with up at all corners, "
              f"phi1 + phi2 >= 0 ({ok.mean():.1%} of the canonical cells' tetrahedra)")

    def surface(self, level):
        """Triangles of U = level: vertices (grid-index coordinates), triangles, the
        tetrahedron of each triangle and the energy gradient there (per grid step)."""
        t = np.flatnonzero((self.lo < level) & (self.hi > level))
        V = self.nodes[t]
        above = self.energy[V] > level
        count = above.sum(axis=1)
        order = np.argsort(~above, axis=1, kind="stable")            # above-level corners first
        V = np.take_along_axis(V, order, axis=1)
        tris, src = [], []
        for n, edge_lists in ((1, [((0, 1), (0, 2), (0, 3))]),
                              (3, [((3, 0), (3, 1), (3, 2))]),
                              (2, [((0, 2), (0, 3), (1, 3)), ((0, 2), (1, 3), (1, 2))])):
            rows = np.flatnonzero(count == n)
            for edges in edge_lists:
                tris.append(np.stack([V[rows][:, list(e)] for e in edges], axis=1))  # (m, 3, 2)
                src.append(rows)
        tris, src = np.concatenate(tris), np.concatenate(src)
        a, b = tris[..., 0].ravel(), tris[..., 1].ravel()
        key = np.minimum(a, b) * len(self.energy) + np.maximum(a, b)
        _, first, index = np.unique(key, return_index=True, return_inverse=True)
        a, b = a[first], b[first]
        ea, eb = self.energy[a], self.energy[b]
        s = ((level - ea) / (eb - ea))[:, None]
        pa = np.column_stack(np.unravel_index(a, self.shape)).astype(float)
        pb = np.column_stack(np.unravel_index(b, self.shape)).astype(float)
        vertices = pa + s * (pb - pa)
        tet = t[src]
        e = self.energy[self.nodes[tet]]
        grad = np.einsum("tij,tj->ti", self.inv[self.kind[tet]], e[:, 1:] - e[:, :1])
        # The vertex's tetrahedron: that of its first triangle.
        vertex_tet = np.empty(len(vertices), dtype=np.int64)
        vertex_tet[index[::-1]] = np.repeat(tet, 3)[::-1]
        return vertices, index.reshape(-1, 3), tet, np.linalg.norm(grad, axis=1), vertex_tet


def lattice(vertices, triangles, tet, spacing):
    """Barycentric lattice points of every triangle, at most `spacing` apart along
    its longest edge, corners left out (they are the vertices)."""
    P = vertices[triangles]
    longest = np.max(np.linalg.norm(P - np.roll(P, 1, axis=1), axis=2), axis=1)
    steps = np.maximum(1, np.ceil(longest / spacing)).astype(np.int64)
    points, owner = [], []
    for s in np.unique(steps):
        if s == 1:
            continue
        i, j = np.meshgrid(np.arange(s + 1), np.arange(s + 1), indexing="ij")
        i, j = i.ravel(), j.ravel()
        k = s - i - j
        inside = (k >= 0) & ~(((i == s) | (j == s) | (k == s)))
        w = np.column_stack((i[inside], j[inside], k[inside])) / s                     # (m, 3)
        rows = np.flatnonzero(steps == s)
        points.append(np.einsum("mc,tcx->tmx", w, P[rows]).reshape(-1, 3))
        owner.append(np.repeat(tet[rows], len(w)))
    if not points:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return np.concatenate(points), np.concatenate(owner)


def poisson_disk(points, radius, priority):
    """The greedy r-separated subset: points taken in priority order, each kept
    unless a kept point lies within radius. Computed in parallel rounds (a point
    whose priority beats every live neighbour's is kept; its neighbours die),
    which gives the same set as the sequential greedy pass."""
    n = len(points)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    alive, keep = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
    i, j = pairs[:, 0], pairs[:, 1]
    while alive.any():
        best = np.full(n, np.inf)
        np.minimum.at(best, i, priority[j])
        np.minimum.at(best, j, priority[i])
        chosen = alive & (priority < best)
        keep |= chosen
        alive &= ~chosen
        alive[i[chosen[j]]] = False
        alive[j[chosen[i]]] = False
        live = alive[i] & alive[j]
        i, j = i[live], j[live]
    return keep


def place(tets, level, n_target, cfg, rng):
    """Regularly spaced points on one level: (grid-index coordinates, tetrahedron,
    radius, stats)."""
    vertices, triangles, tet, grad, vertex_tet = tets.surface(level)
    P = vertices[triangles]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    side = P[..., 1].sum(axis=1) / 3 + P[..., 2].sum(axis=1) / 3 - (tets.n_phi - 1)
    canonical_area = float(area[side >= 0].sum())
    # Volume per unit energy between neighbouring levels, and the normal gap per unit U.
    stats = {"area": canonical_area, "triangles": len(triangles),
             "gap_per_u": float((area / np.maximum(grad, 1e-12)).sum() / max(area.sum(), 1e-300))}
    if n_target <= 0 or canonical_area == 0:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), np.nan, stats
    radius = max(cfg["min_spacing"], np.sqrt(0.7 * canonical_area / n_target))
    for attempt in range(6):
        extra, extra_tet = lattice(vertices, triangles, tet, 0.5 * radius)
        cand = np.concatenate((vertices, extra))
        owner = np.concatenate((vertex_tet, extra_tet))
        later = np.concatenate((np.zeros(len(vertices)), np.ones(len(extra))))  # vertices first
        on_side = cand[:, 1] + cand[:, 2] >= tets.n_phi - 1 - 1e-9
        cand, owner, later = cand[on_side], owner[on_side], later[on_side]
        keep = poisson_disk(cand, radius, later + rng.random(len(cand)))
        got = int(keep.sum())
        if abs(got / n_target - 1) <= cfg["count_tol"] or (got < n_target
                                                            and radius <= cfg["min_spacing"]):
            break
        radius = max(cfg["min_spacing"], radius * np.sqrt(got / n_target))
    stats.update(radius=float(radius), candidates=len(cand))
    return cand[keep], owner[keep], float(radius), stats


def plan(run, br, cfg, seed):
    grid_energy = br.up["energy"][br.exists]
    target = Target(np.concatenate([grid_energy, grid_energy]), cfg)
    print(f"Target: H(U) = {target.c:,.0f} U^-{target.alpha:g} dataset rows per unit U for "
          f"{target.u_min:.4f} <= U <= {target.u_max:.3f}, the grid alone above "
          f"(the grid's histogram peaks at U = {target.u_peak:.2f})")
    print(f"{len(target.levels)} levels, {target.edges[1] / target.edges[0] - 1:.0%} apart; "
          f"{target.points.sum():,} new points to solve (up branch, phi1 + phi2 >= 0)")
    tets = Tets(br, run.solver_settings["jump_tol"])
    rng = np.random.default_rng(seed)
    coords, owners, level_index, table = [], [], [], []
    start = time.perf_counter()
    print(f"{'level':>4} {'U':>8} {'target':>8} {'placed':>8} {'area':>9} {'spacing':>8} "
          f"{'gap':>7}")
    for k, (level, n_target) in enumerate(zip(target.levels, target.points)):
        pts, owner, radius, stats = place(tets, level, int(n_target), cfg, rng)
        coords.append(pts)
        owners.append(owner)
        level_index.append(np.full(len(pts), k, dtype=np.int32))
        # Normal distance to the next level, in grid steps: dU / |grad U|.
        gap = (target.edges[k + 1] - target.edges[k]) * stats["gap_per_u"]
        table.append((level, n_target, len(pts), stats["area"], radius, gap))
        print(f"{k:4d} {level:8.4f} {n_target:8,d} {len(pts):8,d} {stats['area']:9.1f} "
              f"{radius:8.3f} {gap:7.3f}", flush=True)
    print(f"Placed in {time.perf_counter() - start:.0f} s (spacing and gap in grid steps)")
    coords = np.concatenate(coords)
    owner = np.concatenate(owners)
    level_index = np.concatenate(level_index)
    cell = tets.cell[owner]
    # Offsets of each point in its tetrahedron's cell, in [0, 1].
    ijk = br.cell_ijk[cell]
    offsets = np.clip(coords - ijk, 0.0, 1.0)
    order = np.lexsort((offsets[:, 0], cell))                  # neighbouring points together
    table = np.array(table)
    return target, dict(cell=cell[order], offsets=offsets[order], level=level_index[order],
                        level_values=target.levels, level_edges=target.edges,
                        level_target=table[:, 1], level_placed=table[:, 2],
                        level_area=table[:, 3], level_spacing=table[:, 4], level_gap=table[:, 5])


# ---------------------------------------------------------------- solve

def _solve_chunk(task):
    """Solve one chunk: nearest-corner warm start, then, while U is outside the
    point's window [lo, hi], projection towards its goal energy. The iso scheme's
    goal is the level and its window the level within project_tol; refine_shell.py
    passes its bin and the point's planned energy."""
    chunk, part_path, cell, offsets, level, goal, lo_u, hi_u, project = task
    w = refine._WORKER
    br, beam, amplitude_at = w["br"], w["beam"], w["amplitude"]
    lo = np.array([a[0] for a in br.axes])
    hi = np.array([a[-1] for a in br.axes])
    coords = refine.point_coords(br, cell, offsets)
    m = len(cell)
    out = {k: np.full(m, np.nan) for k in FIELDS}
    out.update(exists=np.zeros(m, dtype=bool), source=np.full(m, project["source"], dtype=np.int8),
               warm_start=np.full(m, -1, dtype=np.int64), iterations=np.zeros(m, dtype=np.int16),
               projections=np.zeros(m, dtype=np.int8), goal=np.array(goal, dtype=float),
               coords=coords.copy(),
               theta=np.full((m, len(br.beam_x)), np.nan, dtype=np.float32))
    counts = {"solved": 0, project["hit"]: 0, "dropped": 0}
    for row in range(m):
        corners = br.corner_nodes[:, cell[row]]
        distance = np.sum((refine.CORNERS - offsets[row]) ** 2, axis=1)
        order = [int(corners[c]) for c in np.argsort(distance, kind="stable")
                 if br.exists[corners[c]]]
        x = coords[row]
        amplitude = float(amplitude_at(x[0]))
        state = None
        for node in order[:beam.max_failures]:
            trial = beam.warm(x, br.theta[node], br.up["fx"][node], br.up["fy"][node])
            if beam.keeps_branch(trial, br.up["q"][node], amplitude):
                state, out["warm_start"][row] = trial, node
                break
        if state is None:
            counts["dropped"] += 1
            continue
        for step in range(project["steps"]):
            if lo_u[row] <= state["energy"] <= hi_u[row]:
                break
            miss = state["energy"] - goal[row]
            grad = np.array([-state["fx"], -state["moment"], state["moment_right"]])
            for _ in range(3):                     # along the box face if the step leaves it
                if not grad.any():
                    break
                x_new = x - miss * grad / (grad @ grad)
                out_of_box = (x_new < lo - 1e-12) | (x_new > hi + 1e-12)
                if not out_of_box.any():
                    break
                grad[out_of_box] = 0.0
            if not grad.any() or out_of_box.any():
                break
            x_new = np.clip(x_new, lo, hi)
            q_from = state["p"] / amplitude
            amplitude_new = float(amplitude_at(x_new[0]))
            trial = beam.warm(x_new, state["theta"], state["fx"], state["fy"])
            if not beam.keeps_branch(trial, q_from, amplitude_new):
                break
            x, state, amplitude = x_new, trial, amplitude_new
            out["projections"][row] = step + 1
        counts[project["hit"]] += bool(lo_u[row] <= state["energy"] <= hi_u[row])
        for k in FIELDS:
            out[k][row] = state[k]
        out["coords"][row] = x
        out["theta"][row] = state["theta"]
        out["iterations"][row] = state["iterations"]
        out["exists"][row] = True
        counts["solved"] += 1
    keep = out["exists"]
    # The cell and offsets of the final (projected) point.
    ijk = np.column_stack([np.clip(np.floor((out["coords"][:, a] - br.axes[a][0]) / br.steps[a]
                                            + 1e-9).astype(np.int64), 0, br.cell_shape[a] - 1)
                           for a in range(3)])
    new_cell = np.ravel_multi_index(ijk.T, br.cell_shape)
    new_offsets = np.clip((out.pop("coords") - np.column_stack(
        [br.axes[a][ijk[:, a]] for a in range(3)])) / br.steps, 0.0, 1.0)
    save_npz(part_path, cell=new_cell[keep], offsets=new_offsets[keep], level=level[keep],
             fold=np.zeros(int(keep.sum()), dtype=bool), **{k: v[keep] for k, v in out.items()})
    return chunk, counts


def stage_sample(run, dry_run):
    cfg = settings(run)
    seed = int(run.config.get("seed", 0))
    br = Branches(run.branches_npz)
    plan_path = Path(run.config.get("iso_plan_npz", "iso_plan.npz"))
    plan_key = repr(sorted((k, cfg[k]) for k in PLAN_KEYS)) + f" seed={seed}"
    if plan_path.exists():
        with np.load(plan_path) as p:
            points = {k: p[k] for k in p.files}
        if points.get("settings", np.array("")).item() != plan_key:
            raise SystemExit(f"{plan_path} was planned with other iso settings; delete it (and "
                             f"{run.parts_dir}/) to plan again")
        print(f"Plan from {plan_path}: {len(points['cell']):,} points on "
              f"{len(points['level_values'])} levels")
    else:
        target, points = plan(run, br, cfg, seed)
        points["settings"] = np.array(plan_key)
        save_npz(plan_path, **points)
        print(f"Wrote {plan_path}")
    cell, offsets, level = points["cell"], points["offsets"], points["level"]
    on_plane = np.isclose(offsets[:, 1] + offsets[:, 2] + br.cell_ijk[cell, 1]
                          + br.cell_ijk[cell, 2], br.shape[1] - 1, atol=1e-9)
    grid_energy = br.up["energy"][br.exists]
    planned = points["level_values"][level]
    rows_energy = np.concatenate([grid_energy, grid_energy, np.repeat(planned, 4)[
        ~np.repeat(on_plane, 4) | np.tile([True, False, True, False], len(planned))]])
    print(f"Dataset rows with a state (planned): {len(rows_energy):,}; {band_counts(rows_energy)}")
    workers = int(run.config.get("workers", max(1, (os.cpu_count() or 2) - 4)))
    per_solve = float(run.config.get("seconds_per_solve", 0.011))
    solves = len(cell) * (1 + 0.5 * (cfg["project_steps"] > 0))
    print(f"Estimated time: {solves * per_solve / 3600:.1f} h serial, "
          f"{solves * per_solve / 3600 / workers * 60:.0f} min on {workers} workers")
    if dry_run:
        Path("post").mkdir(exist_ok=True)
        histogram_figure(np.concatenate([grid_energy, grid_energy]), rows_energy[
            2 * len(grid_energy):], cfg, "post/energy_histogram_iso_planned.png", "planned")
        print("Wrote post/energy_histogram_iso_planned.png")
        return

    goal = points["level_values"][level]
    tol = float(cfg["project_tol"])
    merged = solve_plan(run, cell, offsets, level, goal, goal * (1 - tol), goal * (1 + tol),
                        {"steps": int(cfg["project_steps"]), "source": SOURCE_ISO, "hit": "on level"},
                        workers)
    for k in ("level_values", "level_edges", "level_target", "level_placed"):
        merged[k] = points[k]
    save_npz(run.samples_npz, **merged)
    miss = np.abs(merged["energy"] / points["level_values"][merged["level"]] - 1)
    print(f"Wrote {run.samples_npz}: {len(merged['cell']):,} states of {len(cell):,} planned "
          f"points; {np.count_nonzero(miss <= cfg['project_tol']):,} within "
          f"{cfg['project_tol']:g} of their level (median |U / U_k - 1| {np.median(miss):.1e}, "
          f"max {miss.max():.1e})")


def solve_plan(run, cell, offsets, level, goal, lo_u, hi_u, project, workers):
    """Solve planned points on a pool of workers (_solve_chunk), one part file per
    chunk so that a rerun resumes, and merge the parts."""
    parts_dir = run.parts_dir
    parts_dir.mkdir(exist_ok=True)
    chunk_size = int(run.config.get("chunk_points", 5000))
    tasks = []
    for chunk, start in enumerate(range(0, len(cell), chunk_size)):
        part = parts_dir / f"part_{chunk:05d}.npz"
        if not part.exists():
            rows = slice(start, start + chunk_size)
            tasks.append((chunk, str(part), cell[rows], offsets[rows], level[rows], goal[rows],
                          lo_u[rows], hi_u[rows], project))
    total_chunks = -(-len(cell) // chunk_size)
    print(f"{total_chunks - len(tasks)} of {total_chunks} chunks already done; "
          f"solving {len(tasks)} on {workers} workers")
    totals = {}
    start_time = time.perf_counter()
    if tasks:
        import multiprocessing as mp
        context = mp.get_context("spawn")
        with context.Pool(workers, initializer=refine._init_worker,
                          initargs=(run.branches_npz, run.solver_settings)) as pool:
            for done, (chunk, counts) in enumerate(pool.imap_unordered(_solve_chunk, tasks), 1):
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
                if done % max(1, len(tasks) // 50) == 0 or done == len(tasks):
                    elapsed = time.perf_counter() - start_time
                    print(f"  {done}/{len(tasks)} chunks, {elapsed / 60:.1f} min, eta "
                          f"{elapsed / done * (len(tasks) - done) / 60:.1f} min: "
                          + ", ".join(f"{v:,} {k}" for k, v in totals.items()), flush=True)
            pool.close()        # let the workers exit; the with block would kill them
            pool.join()

    parts = sorted(parts_dir.glob("part_*.npz"))
    if len(parts) != total_chunks:
        raise SystemExit(f"{len(parts)} of {total_chunks} parts present; rerun to finish")
    merged = {}
    for part in parts:
        with np.load(part) as p:
            for k in p.files:
                merged.setdefault(k, []).append(p[k])
    return {k: np.concatenate(v) for k, v in merged.items()}


# ---------------------------------------------------------------- figure

def histogram_figure(grid_rows, iso_rows, cfg, path, what, name="iso-energy points"):
    """Rows per unit U on log axes against the target, and the counts on linear bins
    of 0.25 as in postprocess.energy_histogram; the pre-refined grid and the
    iso-energy rows stacked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid_colour, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    target = Target(grid_rows, cfg)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=surface)
    for ax in (a1, a2):
        ax.set_facecolor(surface)
        ax.grid(axis="y", color=grid_colour, lw=0.8, which="major")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # Left: per unit U, on bins that are pairs of level intervals, then the same
    # ratio up to the largest energy.
    edges = target.edges[::2]
    ratio = edges[1] / edges[0]
    top = max(grid_rows.max(), iso_rows.max() if len(iso_rows) else 0)
    edges = np.append(edges, edges[-1] * ratio ** np.arange(1, 1 + int(np.ceil(
        np.log(top / edges[-1]) / np.log(ratio)))))
    width = np.diff(edges)
    g = np.histogram(grid_rows, edges)[0] / width
    t = g + np.histogram(iso_rows, edges)[0] / width
    a1.stairs(t, edges, color=orange, lw=2, label=f"dataset: grid + {name}")
    a1.stairs(g, edges, color=blue, lw=2, label="pre-refined grid (both branches)")
    u = np.geomspace(target.u_min, target.u_max, 50)
    a1.plot(u, target.density(u), color=ink, lw=1, ls="--",
            label=f"target {target.c:,.0f} U^-{target.alpha:g}")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylim(max(1.0, g[g > 0].min() * 0.5), t.max() * 2)
    a1.set_xlabel("bending energy U (log scale)")
    a1.set_ylabel("rows per unit U (log scale)")
    a1.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=ink)
    a1.set_title(f"Rows per unit energy ({what})", loc="left", color=ink)

    # Right: counts on linear bins of 0.25, stacked.
    x_max = float(np.ceil(np.percentile(np.concatenate([grid_rows, iso_rows]), 99.5) / 5) * 5)
    lin = np.arange(0.0, x_max + 0.25, 0.25)
    cg = np.histogram(grid_rows, lin)[0]
    ci = np.histogram(iso_rows, lin)[0]
    a2.bar(lin[:-1], cg, width=0.25, align="edge", color=blue, edgecolor=surface, lw=0.5,
           label="pre-refined grid")
    a2.bar(lin[:-1], ci, width=0.25, bottom=cg, align="edge", color=orange, edgecolor=surface,
           lw=0.5, label=name)
    a2.set_xlim(0, x_max)
    a2.set_xlabel("bending energy U (bin width 0.25)")
    a2.set_ylabel("rows per bin")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    total = cg + ci
    rises = np.flatnonzero(np.diff(total) > 0)
    all_rows = np.concatenate([grid_rows, iso_rows])
    lines = [f"{len(all_rows):,} rows, {len(iso_rows):,} from {name}"]
    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, np.inf)):
        n = np.count_nonzero((all_rows >= lo) & (all_rows < hi))
        lines.append(f"{lo:g} ≤ U < {hi:g}: {n / len(all_rows):.1%}" if hi < np.inf
                     else f"U ≥ {lo:g}: {n / len(all_rows):.1%}")
    lines.append("bins never rise" if not len(rises) else
                 f"{len(rises)} bins rise, largest by {np.max(np.diff(total)[rises]):,}")
    a2.text(0.98, 0.96, "\n".join(lines), transform=a2.transAxes, ha="right", va="top",
            color=ink, fontsize=9, linespacing=1.5)
    a2.legend(frameon=False, fontsize=9, loc="center right", labelcolor=ink)
    a2.set_title("Rows per bin", loc="left", color=ink)
    for ax in (a1, a2):
        ax.tick_params(colors=ink2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


def check(run, data, have):
    """The states on their levels, and the dataset's histogram against the target.
    Returns the names of failed checks."""
    cfg = settings(run)
    with np.load(run.samples_npz) as s:
        miss = np.abs(s["energy"] / s["level_values"][s["level"]] - 1)
        projections = s["projections"]
    on = miss <= cfg["project_tol"]
    ok = on.mean() > 0.99
    print(f"[{'PASS' if ok else 'FAIL'}] on level: {on.sum():,} of {len(on):,} iso-energy states "
          f"({on.mean():.2%}) within {cfg['project_tol']:g} of their level after at most "
          f"{cfg['project_steps']} projections (mean {projections.mean():.2f}); the rest are "
          f"valid states off their level, median |U / U_k - 1| "
          f"{np.median(miss[~on]) if (~on).any() else 0:.1e}")
    grid = have & np.isin(data["source"], (1, 2, 3))
    iso = have & (data["source"] == SOURCE_ISO)
    histogram_figure(data["energy"][grid], data["energy"][iso], cfg,
                     "post/energy_histogram_iso.png", "dataset")
    print("Wrote post/energy_histogram_iso.png")
    return [] if ok else ["on level"]
