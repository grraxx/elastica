"""
refine_shell.py
The energy-shell scheme of refine.py's sample stage (config "scheme": "shell"):
new points spread through the volume between consecutive energy levels of the
up branch, as many per energy bin as the target histogram of refine_iso.py asks.

Why. The iso scheme puts every new point on one of its levels U_k, 5 % apart,
so between two levels the dataset holds only grid states: on any energy scale
finer than the level step its new points form a comb. Here the levels are the
edges of energy bins, and the points fill the shell between two edges.

Target and bins. As refine_iso.py: H(U) = C U^-alpha dataset rows per unit U up
to the grid's histogram peak U_peak; bin edges e_k = U_min (1 + level_step)^k;
bin k gets the target's rows in [e_k, e_k+1) less the grid rows already there,
divided by four (each solve gives four rows by symmetry).

Shells. Bin k's shell is where the linear interpolant of the up branch's energy
over the jump-free tetrahedra (refine_iso.Tets) lies in [e_k, e_k+1), so it ends
at the fold and at the box. Candidates: the same number of uniform random points
in every tetrahedron the bin crosses (the six tetrahedra of a cell all have
volume 1/6), kept where the interpolant lies in the bin; so they are uniform in
the shell, about `oversample` times the bin's target. A greedy Poisson-disk
selection in 3D (refine_iso.poisson_disk, in random order) keeps a candidate
unless a kept point lies within r, in grid steps; r is tuned per bin until the
count is within count_tol of the target, with a floor of min_spacing. Only
phi1 + phi2 >= 0; the mirror gives the other half.

Solve. refine_iso's: a warm start from the cell's corners holding up, nearest
first, with main.py's tests, then projection along the energy gradient onto the
point's planned energy (the interpolant at the point), until |U / U_plan - 1| <
project_tol, as the iso scheme does onto its level. The planned energies fill
each bin continuously, and so do the states. (Projecting only the states that
leave their bin, and only until they are back in it, piles them up just inside
the bin edges: U lies below its linear interpolant, so the states planned near
a bin's lower edge are the ones that leave it.)
"""

import os
import time
from pathlib import Path

import numpy as np

from refine import Branches, band_counts, save_npz
from refine_iso import Target, Tets, histogram_figure, poisson_disk, solve_plan

SOURCE_SHELL = 7
PLAN_KEYS = ("alpha", "scale", "u_max", "level_step", "min_spacing", "count_tol", "oversample")
DEFAULTS = {"alpha": 1.0, "scale": 1.0, "u_max": None, "level_step": 0.05,
            "min_spacing": 0.05, "count_tol": 0.03, "oversample": 12, "project_steps": 4,
            "project_tol": 1e-4}
PILOT = 4                # candidates per tetrahedron in the first pass, to measure the shell


def settings(run):
    return {**DEFAULTS, **run.config.get("shell", {})}


# ---------------------------------------------------------------- shells

def candidates(tets, t, m, lo, hi, rng):
    """m uniform random points in every tetrahedron of t, kept where the energy
    interpolant lies in [lo, hi) and phi1 + phi2 >= 0: grid-index coordinates,
    interpolated energies and tetrahedra."""
    xs, us, owners = [], [], []
    batch = max(1, 2_000_000 // m)
    for start in range(0, len(t), batch):
        tt = t[start:start + batch]
        corners = np.stack(np.unravel_index(tets.nodes[tt], tets.shape), axis=-1).astype(float)
        energy = tets.energy[tets.nodes[tt]]                                   # (b, 4)
        w = rng.exponential(size=(len(tt), m, 4))                              # uniform on
        w /= w.sum(axis=2, keepdims=True)                                      # the simplex
        u = np.einsum("bmc,bc->bm", w, energy)
        b, k = np.nonzero((u >= lo) & (u < hi))
        x = np.einsum("nc,ncx->nx", w[b, k], corners[b])
        side = x[:, 1] + x[:, 2] >= tets.n_phi - 1 - 1e-9
        xs.append(x[side])
        us.append(u[b, k][side])
        owners.append(tt[b][side])
    return np.concatenate(xs), np.concatenate(us), np.concatenate(owners)


def place(tets, lo, hi, n_target, cfg, rng):
    """Spread points through one shell: (grid-index coordinates, planned energy,
    tetrahedron, radius, stats)."""
    empty = (np.empty((0, 3)), np.empty(0), np.empty(0, dtype=np.int64), np.nan)
    t = np.flatnonzero((tets.lo < hi) & (tets.hi >= lo))
    if not len(t):
        return *empty, {"volume": 0.0}
    x, u, owner = candidates(tets, t, PILOT, lo, hi, rng)
    stats = {"volume": len(x) / (6.0 * PILOT)}                 # grid steps^3, phi1 + phi2 >= 0
    if n_target <= 0 or not len(x):
        return *empty, stats
    want = cfg["oversample"] * n_target
    if len(x) < want:                       # more candidates: the same number per tetrahedron
        extra = int(np.ceil(PILOT * want / len(x))) - PILOT
        more = candidates(tets, t, extra, lo, hi, rng)
        x, u, owner = (np.concatenate(pair) for pair in zip((x, u, owner), more))
    elif len(x) > want:                     # fewer: a uniform random subset
        pick = rng.choice(len(x), int(want), replace=False)
        x, u, owner = x[pick], u[pick], owner[pick]
    # A saturated random packing of r-balls holds about 0.7 points per r^3.
    radius = max(cfg["min_spacing"], (0.6 * stats["volume"] / n_target) ** (1 / 3))
    for attempt in range(8):
        keep = poisson_disk(x, radius, rng.random(len(x)))
        got = int(keep.sum())
        if abs(got / n_target - 1) <= cfg["count_tol"] or (got < n_target
                                                            and radius <= cfg["min_spacing"]):
            break
        radius = max(cfg["min_spacing"], radius * (got / n_target) ** (1 / 3))
    stats.update(candidates=len(x))
    return x[keep], u[keep], owner[keep], float(radius), stats


def plan(run, br, cfg, seed):
    grid_energy = br.up["energy"][br.exists]
    target = Target(np.concatenate([grid_energy, grid_energy]), cfg)
    print(f"Target: H(U) = {target.c:,.0f} U^-{target.alpha:g} dataset rows per unit U for "
          f"{target.u_min:.4f} <= U <= {target.u_max:.3f}, the grid alone above "
          f"(the grid's histogram peaks at U = {target.u_peak:.2f})")
    print(f"{len(target.levels)} bins, {target.edges[1] / target.edges[0] - 1:.0%} wide; "
          f"{target.points.sum():,} new points to solve (up branch, phi1 + phi2 >= 0)")
    tets = Tets(br, run.solver_settings["jump_tol"])
    rng = np.random.default_rng(seed)
    coords, energies, owners, bins, table = [], [], [], [], []
    start = time.perf_counter()
    print(f"{'bin':>4} {'from U':>8} {'to U':>8} {'target':>8} {'placed':>8} {'volume':>9} "
          f"{'spacing':>8}")
    for k, n_target in enumerate(target.points):
        lo, hi = target.edges[k], target.edges[k + 1]
        pts, u, owner, radius, stats = place(tets, lo, hi, int(n_target), cfg, rng)
        coords.append(pts)
        energies.append(u)
        owners.append(owner)
        bins.append(np.full(len(pts), k, dtype=np.int32))
        table.append((n_target, len(pts), stats["volume"], radius))
        print(f"{k:4d} {lo:8.4f} {hi:8.4f} {n_target:8,d} {len(pts):8,d} {stats['volume']:9.2f} "
              f"{radius:8.3f}", flush=True)
    print(f"Placed in {time.perf_counter() - start:.0f} s (volume and spacing in grid steps)")
    coords, owner = np.concatenate(coords), np.concatenate(owners)
    cell = tets.cell[owner]
    offsets = np.clip(coords - br.cell_ijk[cell], 0.0, 1.0)
    order = np.lexsort((offsets[:, 0], cell))                  # neighbouring points together
    table = np.array(table)
    return dict(cell=cell[order], offsets=offsets[order], level=np.concatenate(bins)[order],
                u_plan=np.concatenate(energies)[order], level_values=target.levels,
                level_edges=target.edges, level_target=table[:, 0], level_placed=table[:, 1],
                level_volume=table[:, 2], level_spacing=table[:, 3])


# ---------------------------------------------------------------- sample

def stage_sample(run, dry_run):
    cfg = settings(run)
    seed = int(run.config.get("seed", 0))
    br = Branches(run.branches_npz)
    plan_path = Path(run.config.get("shell_plan_npz", "shell_plan.npz"))
    plan_key = repr(sorted((k, cfg[k]) for k in PLAN_KEYS)) + f" seed={seed}"
    if plan_path.exists():
        with np.load(plan_path) as p:
            points = {k: p[k] for k in p.files}
        if points.get("settings", np.array("")).item() != plan_key:
            raise SystemExit(f"{plan_path} was planned with other shell settings; delete it (and "
                             f"{run.parts_dir}/) to plan again")
        print(f"Plan from {plan_path}: {len(points['cell']):,} points in "
              f"{len(points['level_values'])} bins")
    else:
        points = plan(run, br, cfg, seed)
        points["settings"] = np.array(plan_key)
        save_npz(plan_path, **points)
        print(f"Wrote {plan_path}")
    cell, offsets, level, planned = points["cell"], points["offsets"], points["level"], points["u_plan"]
    on_plane = np.isclose(offsets[:, 1] + offsets[:, 2] + br.cell_ijk[cell, 1]
                          + br.cell_ijk[cell, 2], br.shape[1] - 1, atol=1e-9)
    grid_energy = br.up["energy"][br.exists]
    rows_energy = np.concatenate([grid_energy, grid_energy, np.repeat(planned, 4)[
        ~np.repeat(on_plane, 4) | np.tile([True, False, True, False], len(planned))]])
    print(f"Dataset rows with a state (planned): {len(rows_energy):,}; {band_counts(rows_energy)}")
    workers = int(run.config.get("workers", max(1, (os.cpu_count() or 2) - 4)))
    per_solve = float(run.config.get("seconds_per_solve", 0.011))
    print(f"Estimated time: {len(cell) * per_solve / 3600:.1f} h serial, "
          f"{len(cell) * per_solve / 3600 / workers * 60:.0f} min on {workers} workers (no projections)")
    if dry_run:
        Path("post").mkdir(exist_ok=True)
        histogram_figure(np.concatenate([grid_energy, grid_energy]), rows_energy[2 * len(grid_energy):],
                         cfg, "post/energy_histogram_shell_planned.png", "planned",
                         name="energy-shell points")
        print("Wrote post/energy_histogram_shell_planned.png")
        return

    tol = float(cfg["project_tol"])
    merged = solve_plan(run, cell, offsets, level, planned, planned * (1 - tol), planned * (1 + tol),
                        {"steps": int(cfg["project_steps"]), "source": SOURCE_SHELL, "hit": "on plan"},
                        workers)
    for k in ("level_values", "level_edges", "level_target", "level_placed"):
        merged[k] = points[k]
    save_npz(run.samples_npz, **merged)
    inside = in_bin(merged["energy"], merged["level"], points["level_edges"])
    on_plan = np.abs(merged["energy"] / merged["goal"] - 1) <= tol
    print(f"Wrote {run.samples_npz}: {len(merged['cell']):,} states of {len(cell):,} planned "
          f"points; {on_plan.sum():,} ({on_plan.mean():.2%}) within {tol:g} of their planned "
          f"energy, {inside.sum():,} ({inside.mean():.2%}) in their bin")


def in_bin(energy, level, edges):
    return (energy >= edges[level]) & (energy <= edges[level + 1])


def check(run, data, have):
    """The states in their bins, how they fill them, and the dataset's histogram
    against the target. Returns the names of failed checks."""
    cfg = settings(run)
    with np.load(run.samples_npz) as s:
        energy, level, edges, goal = s["energy"], s["level"], s["level_edges"], s["goal"]
        projections = s["projections"]
    inside = in_bin(energy, level, edges)
    on_plan = np.abs(energy / goal - 1) <= cfg["project_tol"]
    ok = inside.mean() > 0.99 and on_plan.mean() > 0.99
    # Where in its bin each state lies, on a log scale: 0 at the lower edge, 1 at the upper.
    position = np.log(energy / edges[level]) / np.log(edges[level + 1] / edges[level])
    quarters = np.histogram(position[inside], np.linspace(0, 1, 5))[0] / inside.sum()
    print(f"[{'PASS' if ok else 'FAIL'}] in bin: {on_plan.sum():,} of {len(inside):,} energy-shell "
          f"states ({on_plan.mean():.2%}) within {cfg['project_tol']:g} of their planned energy and "
          f"{inside.mean():.2%} in their bin, after at most {cfg['project_steps']} projections "
          f"(mean {projections.mean():.2f}); by quarter of the bin (log U): "
          + ", ".join(f"{q:.1%}" for q in quarters))
    grid = have & np.isin(data["source"], (1, 2, 3))
    new = have & (data["source"] == SOURCE_SHELL)
    histogram_figure(data["energy"][grid], data["energy"][new], cfg,
                     "post/energy_histogram_shell.png", "dataset", name="energy-shell points")
    print("Wrote post/energy_histogram_shell.png")
    return [] if ok else ["in bin"]
