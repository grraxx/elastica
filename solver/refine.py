"""
refine.py
Training data for a neural surrogate of the clamped elastica, built from a
finished run of main.py: both stable branches at every grid node, and extra
solves inside the grid cells, more of them where the bending energy U is low.

usage: python refine.py path/to/config.json {branches,sample,export,check} [--dry-run]
       (outputs go next to the config; the source run is only read)

Two sampling schemes, chosen by the config's "scheme": "cells" (the default,
below) or "iso" (refine_iso.py: regularly spaced points on iso-energy surfaces
of the grid, following a target energy histogram). Both write the same
refined_samples.npz, so export and check serve either.

Branches are labelled by the sign of p (branch.py): up (p > 0) and down. Two
maps take solutions to solutions, and the solver's grid is symmetric under both:
    mirror M:      (phi1, phi2) -> (-phi2, -phi1), theta~(s) = -theta(L - s):
                   same fx, energy and p, opposite fy, clamp moments swapped
    reflection R:  (phi1, phi2) -> (-phi1, -phi2), theta~(s) = -theta(s):
                   same fx and energy, opposite fy, M1, M2 and p
So the up branch everywhere determines the down branch everywhere (down at x
is R of up at -x), and the up branch needs solving only where phi1 + phi2 >= 0.

branches   The up branch at every grid node. main.py keeps one state per node;
           where it is up, that is it, and where the state at -x is down, its
           reflection is an up state at x. A front then extends up into the
           nodes still without it, warm-started from up neighbours, until
           every attempt jumps (onto the down state already there), diverges
           or is unstable: past the fold. Those nodes are negative examples.
           Writes branches_grid.npz.
sample     Extra points inside each grid cell, for the up branch where any
           corner holds it: n = clip(round((u_ref / U)^power), 1, max_points)
           points per cell, U the lowest up energy at its corners, and at
           least fold_min_points in fold cells (up at some corners only). The
           cell's own node is the first point; the rest are a scrambled Sobol
           set with a random shift per cell. Each point is warm-started from
           the corners holding up, nearest first, with main.py's tests (no
           jump in q against that corner, stable). A point in a fold cell that
           fails from every corner tried is a negative example. Runs on a pool
           of workers; every chunk is saved as it finishes, so a rerun resumes.
           --dry-run counts the points and predicts the histogram, no solves.
export     nn_dataset.npz: one row per (point, branch), both branches, all
           four images of every refined point, with a train / validation /
           test split by symmetry orbit of whole grid cells.
check      Reflection, bistability, interpolation, BVP and symmetry checks,
           and the energy histogram of the dataset.
"""

import os

# One Newton solve has ~200 DOFs; threaded BLAS only adds overhead.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import struct
import sys
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from branch import first_mode_amplitude, first_mode_projection, is_jump
from main import end_moments, grid_neighbors, precompute_elastica_spaces
from solver import ElasticaSolver
from stability import unstable_directions

FIELDS = ("fx", "fy", "moment", "moment_right", "energy", "p")
# Row sources in the dataset.
SOURCES = {1: "grid", 2: "grid reflected", 3: "extended", 4: "refined", 5: "refined cold start",
           6: "iso-energy"}
# Corner offsets of a grid cell, and the energy bands of the split and the reports.
CORNERS = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])
BANDS = (0.0, 1.0, 3.0, 10.0, np.inf)
SPLITS = ("train", "validation", "test")


def npz_memmap(path, name):
    """A member of an uncompressed .npz as a read-only memory map, or None."""
    with zipfile.ZipFile(path) as archive:
        if f"{name}.npy" not in archive.namelist():
            return None
        info = archive.getinfo(f"{name}.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    with open(path, "rb") as file:
        file.seek(info.header_offset)
        name_len, extra_len = struct.unpack("<HH", file.read(30)[26:30])
        file.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(file)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                 else np.lib.format.read_array_header_2_0)(file)
        offset = file.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape,
                     order="F" if fortran else "C")


def save_npz(path, compress=False, **arrays):
    """Write next to the target, then rename, so a stopped write leaves no broken file."""
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as file:
        (np.savez_compressed if compress else np.savez)(file, **arrays)
    os.replace(temporary, path)


def amplitude_between_layers(d_vals):
    """a(d) at any d within the layers: a(d) / sqrt(1 - d) is smooth, so it is
    interpolated on a table ten times finer than the layers."""
    table = np.linspace(d_vals[0], d_vals[-1], 10 * (len(d_vals) - 1) + 1)
    scaled = first_mode_amplitude(table) / np.sqrt(1.0 - table)

    def amplitude(d):
        d = np.asarray(d, dtype=float)
        return np.interp(d, table, scaled) * np.sqrt(1.0 - d)
    return amplitude


def band_counts(energy):
    return ", ".join(f"[{lo:g}, {hi:g}): {np.count_nonzero((energy >= lo) & (energy < hi)):,}"
                     for lo, hi in zip(BANDS[:-1], BANDS[1:]))


class Run:
    """The run directory, its config and the source run's config."""

    def __init__(self, config_path):
        config_path = os.path.abspath(config_path)
        with open(config_path) as f:
            self.config = json.load(f)
        os.chdir(os.path.dirname(config_path))
        source = Path(self.config.get("source_run", "../production"))
        with open(source / "config.json") as f:
            self.source_config = json.load(f)
        self.source_results = source / self.source_config.get("results_npz",
                                                              "solution_space_results.npz")
        self.source_live = source / self.source_config.get("live_results_npz",
                                                           "solution_space_live.npz")
        self.branches_npz = self.config.get("branches_npz", "branches_grid.npz")
        self.samples_npz = self.config.get("samples_npz", "refined_samples.npz")
        self.dataset_npz = self.config.get("dataset_npz", "nn_dataset.npz")
        self.parts_dir = Path(self.config.get("parts_dir", "sample_parts"))
        # The solver settings come from the source run, so that refined and grid
        # values are computed alike.
        sc = self.source_config
        self.solver_settings = {
            "beam_elements": int(sc.get("beam_elements", 100)),
            "beam_grading": float(sc.get("beam_grading", 0.0)),
            "snes_rtol": float(sc.get("snes_rtol", 1e-8)),
            "snes_atol": float(sc.get("snes_atol", 1e-10)),
            "warm_start_max_it": int(sc.get("warm_start_max_it", 50)),
            "jump_tol": float(sc.get("jump_tol", 0.17)),
            "max_failures": int(self.config.get("max_failures", 3)),
        }


class Grid:
    """The source run's results on its grid, or on the box in the config."""

    def __init__(self, run, with_theta=True):
        with np.load(run.source_results) as r:
            shape = tuple(int(n) for n in r["shape"])
            axes = [r[k] for k in ("d_vals", "phi1_vals", "phi2_vals")]
            self.beam_x = r["beam_x"]
            if not (r["solved"] > 0.5).all():
                raise SystemExit(f"{run.source_results} has unsolved nodes")
            fields = {k: r[k] for k in FIELDS}
        theta = None
        if with_theta:
            theta = npz_memmap(run.source_live, "theta") if run.source_live.exists() else None
            live_energy = npz_memmap(run.source_live, "energy") if theta is not None else None
            if theta is None or not np.array_equal(live_energy, fields["energy"]):
                print(f"Reading theta from {run.source_results} (no matching uncompressed copy)")
                with np.load(run.source_results) as r:
                    theta = r["theta"]

        box = run.config.get("box")
        if box:
            # A sub-box, symmetric in phi so that both symmetries still hold.
            d_lo, d_hi = box["d_bounds"]
            keep = [np.flatnonzero((axes[0] >= d_lo - 1e-9) & (axes[0] <= d_hi + 1e-9)),
                    np.flatnonzero(np.abs(axes[1]) <= box["phi_max"] + 1e-9),
                    np.flatnonzero(np.abs(axes[2]) <= box["phi_max"] + 1e-9)]
            index = np.ravel_multi_index(np.meshgrid(*keep, indexing="ij"), shape).ravel()
            shape = tuple(len(k) for k in keep)
            axes = [a[k] for a, k in zip(axes, keep)]
            fields = {k: v[index] for k, v in fields.items()}
            if theta is not None:
                theta = np.asarray(theta[index])
        self.shape, self.axes, self.fields, self.theta = shape, axes, fields, theta
        self.num_nodes = int(np.prod(shape))
        n = shape[1]
        if shape[1] != shape[2] or not np.allclose(axes[1], -axes[1][::-1]) \
                or not np.allclose(axes[1], axes[2]):
            raise ValueError("phi1 and phi2 need the same grid, symmetric about 0")
        i, j, k = np.unravel_index(np.arange(self.num_nodes), shape)
        self.mirror = np.ravel_multi_index((i, n - 1 - k, n - 1 - j), shape)
        self.reflect = np.ravel_multi_index((i, n - 1 - j, n - 1 - k), shape)
        self.canonical = j + k >= n - 1
        self.coords = np.column_stack((axes[0][i], axes[1][j], axes[2][k]))
        self.steps = np.array([np.diff(a).mean() for a in axes])
        self.amplitude = first_mode_amplitude(axes[0])[i]


class Beam:
    """The elastica solver with the source run's settings, and main.py's tests."""

    def __init__(self, settings):
        V, R, self.x, self.order = precompute_elastica_spaces(
            L=1.0, N=settings["beam_elements"], grading=settings["beam_grading"])
        self.solver = ElasticaSolver(V, R, petsc_options={
            "snes_max_it": 500, "snes_rtol": settings["snes_rtol"],
            "snes_atol": settings["snes_atol"]})
        self.max_it = settings["warm_start_max_it"]
        self.tol = settings["jump_tol"]
        self.max_failures = settings["max_failures"]

    def state(self, solution):
        theta = solution.theta[self.order]
        m1, m2 = end_moments(theta, self.x, solution.fx, solution.fy)
        return {"theta": theta, "fx": solution.fx, "fy": solution.fy, "moment": m1,
                "moment_right": m2, "energy": solution.energy,
                "p": float(first_mode_projection(theta, self.x)),
                "iterations": solution.iterations}

    def warm(self, coords, theta, fx, fy, max_it=None):
        """Solve at coords from a stored state (theta in beam order); None if it diverges."""
        theta0 = np.empty(len(self.order))
        theta0[self.order] = theta
        try:
            return self.state(self.solver.solve(*coords, theta0, fx, fy,
                                                max_it=max_it or self.max_it))
        except RuntimeError:
            return None

    def cold(self, coords):
        try:
            return self.state(self.solver.cold_solve(*coords))
        except RuntimeError:
            return None

    def keeps_branch(self, state, q_from, amplitude):
        """main.py's acceptance: no jump in q from the warm start, and stable."""
        return (state is not None
                and not is_jump(q_from, state["p"] / amplitude, self.tol)
                and not unstable_directions(state["theta"], self.x, state["fx"], state["fy"]))


# ---------------------------------------------------------------- branches

def stage_branches(run, dry_run):
    grid = Grid(run)
    n, f = grid.num_nodes, grid.fields
    R, Mi = grid.reflect, grid.mirror
    print(f"Grid {grid.shape}, {n:,} nodes")

    # The up state at each node: the stored one if it is up, else the reflection
    # of the stored state at -x if that is down.
    stored_up = f["p"] > 0
    reflected_up = f["p"][R] < 0
    exists = stored_up | reflected_up
    both = stored_up & reflected_up
    same = np.isclose(f["energy"], f["energy"][R], rtol=1e-6) & np.isclose(f["fx"], f["fx"][R],
                                                                            rtol=1e-6, atol=1e-8)
    print(f"Up branch known at {exists.mean():.1%} of nodes: stored at {stored_up.sum():,}, "
          f"from the reflection at {(reflected_up & ~stored_up).sum():,} more; where both "
          f"apply they are the same state at {same[both].mean():.4%} of {both.sum():,} nodes")
    print(f"Both branches known at {(exists & exists[R]).mean():.1%} of nodes")
    lacking = ~exists.reshape(grid.shape)
    near = np.zeros(grid.shape, dtype=bool)
    up3 = exists.reshape(grid.shape)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                shifted = np.zeros_like(up3)
                dst = tuple(slice(max(0, -s), m - max(0, s)) for s, m in zip((di, dj, dk), grid.shape))
                src = tuple(slice(max(0, s), m - max(0, -s)) for s, m in zip((di, dj, dk), grid.shape))
                shifted[dst] = up3[src]
                near |= shifted
    seeds = np.flatnonzero((lacking & near).ravel() & grid.canonical)
    print(f"Front seeds (lacking up, beside an up node, phi1 + phi2 >= 0): {len(seeds):,}")
    if dry_run:
        return

    up = {k: np.full(n, np.nan) for k in FIELDS}
    theta = np.full((n, len(grid.beam_x)), np.nan, dtype=np.float32)
    source = np.zeros(n, dtype=np.int8)
    for start in range(0, n, 200_000):              # chunks: theta may be a memory map
        rows = np.arange(start, min(n, start + 200_000))
        own = rows[stored_up[rows]]
        other = rows[reflected_up[rows] & ~stored_up[rows]]
        theta[own] = grid.theta[own]
        theta[other] = -np.asarray(grid.theta[R[other]])
    for k in FIELDS:
        sign = -1.0 if k in ("fy", "moment", "moment_right", "p") else 1.0
        up[k] = np.where(stored_up, f[k], np.where(reflected_up, sign * f[k][R], np.nan))
    source[reflected_up] = 2
    source[stored_up] = 1

    beam = Beam(run.solver_settings)
    if not np.allclose(beam.x, grid.beam_x):
        raise ValueError("The beam mesh differs from the source run's")
    attempted = np.zeros(n, dtype=bool)
    failures = np.zeros(n, dtype=np.int8)
    tried = {}
    queue, queued = deque(), np.zeros(n, dtype=bool)

    def push(node):
        node = node if grid.canonical[node] else int(Mi[node])
        if not exists[node] and failures[node] < beam.max_failures and not queued[node]:
            queued[node] = True
            queue.append(node)

    def store(node, state):
        for k in FIELDS:
            up[k][node] = state[k]
        theta[node] = state["theta"]
        exists[node], source[node] = True, 3
        m = int(Mi[node])
        if m != node:
            for k in ("fx", "energy", "p"):
                up[k][m] = state[k]
            up["fy"][m] = -state["fy"]
            up["moment"][m], up["moment_right"][m] = state["moment_right"], state["moment"]
            theta[m] = -state["theta"][::-1]
            exists[m], source[m] = True, 3

    for node in seeds:
        push(int(node))
    q_up = up["p"] / grid.amplitude
    stats = {"extended": 0, "diverged": 0, "jumped": 0, "unstable": 0}
    start_time, solves, last_report = time.perf_counter(), 0, 0
    while queue:
        node = queue.popleft()
        queued[node] = False
        if exists[node] or failures[node] >= beam.max_failures:
            continue
        attempted[node] = attempted[Mi[node]] = True
        node_tried = tried.setdefault(node, set())
        for neighbor in grid_neighbors(node, grid.shape):
            if not exists[neighbor] or neighbor in node_tried:
                continue
            node_tried.add(neighbor)
            state = beam.warm(grid.coords[node], theta[neighbor], up["fx"][neighbor],
                              up["fy"][neighbor])
            solves += 1
            # The node already holds a down state, so an accepted up state (p > 0,
            # no jump from an up neighbour) is a second, different one.
            if state is None:
                stats["diverged"] += 1
            elif state["p"] <= 0 or is_jump(q_up[neighbor], state["p"] / grid.amplitude[node],
                                            beam.tol):
                stats["jumped"] += 1
            elif unstable_directions(state["theta"], beam.x, state["fx"], state["fy"]):
                stats["unstable"] += 1
            else:
                store(node, state)
                q_up[node] = q_up[Mi[node]] = state["p"] / grid.amplitude[node]
                stats["extended"] += 1
                for nb in grid_neighbors(node, grid.shape):
                    push(nb)
                for nb in grid_neighbors(int(Mi[node]), grid.shape):
                    push(nb)
                break
            failures[node] += 1
            if failures[node] >= beam.max_failures:
                break
        if solves - last_report >= 20000:
            last_report = solves
            print(f"  {solves:,} solves, {time.perf_counter() - start_time:.0f} s, queue "
                  f"{len(queue):,}: " + ", ".join(f"{v:,} {k}" for k, v in stats.items()),
                  flush=True)
    elapsed = time.perf_counter() - start_time
    print(f"Front done: {solves:,} solves in {elapsed:.0f} s; "
          + ", ".join(f"{v:,} {k}" for k, v in stats.items()))
    print(f"Up branch now at {exists.mean():.1%} of nodes, both branches at "
          f"{(exists & exists[R]).mean():.1%}; {np.count_nonzero(attempted & ~exists):,} "
          "nodes past a fold (attempted, failed)")
    save_npz(run.branches_npz, shape=np.asarray(grid.shape), d_vals=grid.axes[0],
             phi1_vals=grid.axes[1], phi2_vals=grid.axes[2], beam_x=grid.beam_x,
             exists=exists, attempted=attempted, source=source,
             q=up["p"] / grid.amplitude, theta=theta, **up)
    print(f"Wrote {run.branches_npz}")


class Branches:
    """branches_grid.npz: the up branch at every grid node."""

    def __init__(self, path, with_theta=False):
        with np.load(path) as b:
            self.shape = tuple(int(n) for n in b["shape"])
            self.axes = [b[k] for k in ("d_vals", "phi1_vals", "phi2_vals")]
            self.beam_x = b["beam_x"]
            self.up = {k: b[k] for k in (*FIELDS, "q")}
            self.exists, self.attempted, self.source = b["exists"], b["attempted"], b["source"]
        self.theta = npz_memmap(path, "theta") if with_theta else None
        self.steps = np.array([np.diff(a).mean() for a in self.axes])
        cells = tuple(m - 1 for m in self.shape)
        self.cell_shape = cells
        # Corner node of every cell, per corner offset: (8, cells)
        ci, cj, ck = np.unravel_index(np.arange(int(np.prod(cells))), cells)
        self.corner_nodes = np.stack([np.ravel_multi_index((ci + a, cj + b, ck + c), self.shape)
                                      for a, b, c in CORNERS])
        self.cell_ijk = np.column_stack((ci, cj, ck))
        nc = cells[1]
        self.cell_mirror = np.ravel_multi_index((ci, nc - 1 - ck, nc - 1 - cj), cells)
        self.cell_reflect = np.ravel_multi_index((ci, nc - 1 - cj, nc - 1 - ck), cells)
        self.cell_canonical = cj + ck >= nc - 1
        self.cell_self_mirror = cj + ck == nc - 1


def cell_counts(br, density):
    """Points per cell for the up branch (the cell's node included), and the fold flag."""
    corner_up = br.exists[br.corner_nodes]                    # (8, cells)
    corner_energy = np.where(corner_up, br.up["energy"][br.corner_nodes], np.inf)
    lowest = corner_energy.min(axis=0)
    has = corner_up.any(axis=0)
    fold = has & ~corner_up.all(axis=0)
    with np.errstate(divide="ignore"):
        n = np.clip(np.rint((density["u_ref"] / lowest) ** density["power"]), 1,
                    density["max_points"]).astype(np.int64)
    n[fold] = np.maximum(n[fold], density["fold_min_points"])
    n[~has] = 0
    return n, fold


def sample_points(br, density, seed):
    """New points in the canonical cells: (cell index, offsets in [0, 1)^3).

    Scrambled Sobol points with a random shift per cell (Cranley-Patterson), so
    each cell's first m points are well spread for every m. In a cell on the
    diagonal phi1 + phi2 = 0, only the half phi1 + phi2 >= 0 is kept; the
    mirror fills the other."""
    counts, fold = cell_counts(br, density)
    cells = np.flatnonzero(br.cell_canonical & (counts > 1))
    extra = counts[cells] - 1
    sobol = qmc.Sobol(d=3, scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(max(2, density["max_points"])))))
    shift = np.random.default_rng(seed + 1).random((len(cells), 3))
    owner = np.repeat(np.arange(len(cells)), extra)
    rank = np.arange(len(owner)) - np.repeat(np.cumsum(extra) - extra, extra)
    offsets = (sobol[rank] + shift[owner]) % 1.0
    cell = cells[owner]
    keep = ~br.cell_self_mirror[cell] | (offsets[:, 1] + offsets[:, 2] >= 1.0)
    return cell[keep], offsets[keep], counts, fold


def point_coords(br, cell, offsets):
    ijk = br.cell_ijk[cell]
    return np.column_stack([br.axes[a][ijk[:, a]] + offsets[:, a] * br.steps[a] for a in range(3)])


def trilinear(br, cell, offsets, values, mask=None):
    """Trilinear interpolation of a node field at points in cells, over the corners
    where mask holds (weights renormalised)."""
    corner = br.corner_nodes[:, cell]                          # (8, m)
    weights = np.prod(np.where(CORNERS[:, None, :] == 1, offsets[None], 1.0 - offsets[None]),
                      axis=2)
    if mask is not None:
        weights = weights * mask[corner]
    vals = np.where(weights > 0, values[corner], 0.0)
    with np.errstate(invalid="ignore"):
        return (weights * vals).sum(axis=0) / weights.sum(axis=0)


_WORKER = {}


def _init_worker(branches_path, settings):
    br = Branches(branches_path, with_theta=True)
    _WORKER.update(br=br, beam=Beam(settings), amplitude=amplitude_between_layers(br.axes[0]))


def _solve_chunk(task):
    """Solve one chunk of points; writes its part file and returns its counts."""
    chunk, part_path, cell, offsets, fold = task
    br, beam = _WORKER["br"], _WORKER["beam"]
    coords = point_coords(br, cell, offsets)
    m = len(cell)
    out = {k: np.full(m, np.nan) for k in FIELDS}
    out.update(exists=np.zeros(m, dtype=bool), source=np.zeros(m, dtype=np.int8),
               warm_start=np.full(m, -1, dtype=np.int64), iterations=np.zeros(m, dtype=np.int16),
               theta=np.full((m, len(br.beam_x)), np.nan, dtype=np.float32))
    counts = {"solved": 0, "cold": 0, "fold negatives": 0, "dropped": 0}
    for row in range(m):
        corners = br.corner_nodes[:, cell[row]]
        distance = np.sum((CORNERS - offsets[row]) ** 2, axis=1)
        order = [int(corners[c]) for c in np.argsort(distance, kind="stable")
                 if br.exists[corners[c]]]
        amplitude = float(_WORKER["amplitude"](coords[row, 0]))
        accepted, source = None, 4
        for node in order[:beam.max_failures]:
            state = beam.warm(coords[row], br.theta[node], br.up["fx"][node], br.up["fy"][node])
            if beam.keeps_branch(state, br.up["q"][node], amplitude):
                accepted, out["warm_start"][row] = state, node
                break
        if accepted is None and not fold[row]:
            # Inside the branch every corner should do; fall back to a cold start,
            # kept only if it lands on the branch of the nearest corner.
            state = beam.cold(coords[row])
            if order and beam.keeps_branch(state, br.up["q"][order[0]], amplitude):
                accepted, source = state, 5
                out["warm_start"][row] = order[0]
        if accepted is None:
            counts["fold negatives" if fold[row] else "dropped"] += 1
            out["source"][row] = 4
            continue
        for k in FIELDS:
            out[k][row] = accepted[k]
        out["theta"][row] = accepted["theta"]
        out["iterations"][row] = accepted["iterations"]
        out["exists"][row], out["source"][row] = True, source
        counts["cold" if source == 5 else "solved"] += 1
    keep = out["exists"] | fold                  # dropped points leave no row
    save_npz(part_path, cell=cell[keep], offsets=offsets[keep], fold=fold[keep],
             **{k: v[keep] for k, v in out.items()})
    return chunk, counts


def stage_sample(run, dry_run):
    if run.config.get("scheme", "cells") == "iso":
        import refine_iso
        return refine_iso.stage_sample(run, dry_run)
    density = {"u_ref": 3.0, "power": 2.0, "max_points": 64, "fold_min_points": 16,
               **run.config.get("density", {})}
    seed = int(run.config.get("seed", 0))
    br = Branches(run.branches_npz)
    cell, offsets, counts, fold_cells = sample_points(br, density, seed)
    fold = fold_cells[cell]
    print(f"Density {density}; {np.count_nonzero(counts):,} cells hold up at a corner, "
          f"{fold_cells.sum():,} fold cells")
    print(f"New points to solve (up branch, phi1 + phi2 >= 0): {len(cell):,}, "
          f"{fold.sum():,} of them in fold cells")
    predicted = trilinear(br, cell, offsets, br.up["energy"], br.exists)
    print(f"  by predicted energy: {band_counts(predicted)}")
    # Rows of the dataset: both branches at every node, and four images of every
    # refined state (up and down, by the mirror and the reflection).
    grid_energy = np.concatenate([br.up["energy"][br.exists],
                                  br.up["energy"][br.exists]])      # down: the reflection
    rows_energy = np.concatenate([grid_energy, np.repeat(predicted[~fold], 4)])
    print(f"Dataset rows with a state (predicted): {len(rows_energy):,}; {band_counts(rows_energy)}")
    workers = int(run.config.get("workers", max(1, (os.cpu_count() or 2) - 4)))
    per_solve = float(run.config.get("seconds_per_solve", 0.011))
    print(f"Estimated time: {len(cell) * per_solve / 3600:.1f} h serial, "
          f"{len(cell) * per_solve / 3600 / workers * 60:.0f} min on {workers} workers")
    if dry_run:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postprocessing"))
        from postprocess import energy_histogram
        Path("post").mkdir(exist_ok=True)
        d_nodes = np.repeat(br.axes[0], np.prod(br.shape[1:]))[br.exists]
        d_rows = np.concatenate([d_nodes, d_nodes,
                                 np.repeat(point_coords(br, cell[~fold], offsets[~fold])[:, 0], 4)])
        energy_histogram(rows_energy, d_rows, br.axes[0], "post/energy_histogram_predicted.png",
                         what="states (predicted)")
        print("Wrote post/energy_histogram_predicted.png")
        return

    run.parts_dir.mkdir(exist_ok=True)
    chunk_size = int(run.config.get("chunk_points", 5000))
    tasks = []
    for chunk, start in enumerate(range(0, len(cell), chunk_size)):
        part = run.parts_dir / f"part_{chunk:05d}.npz"
        if not part.exists():
            rows = slice(start, start + chunk_size)
            tasks.append((chunk, str(part), cell[rows], offsets[rows], fold[rows]))
    total_chunks = -(-len(cell) // chunk_size)
    print(f"{total_chunks - len(tasks)} of {total_chunks} chunks already done; "
          f"solving {len(tasks)} on {workers} workers")
    totals = {}
    start_time = time.perf_counter()
    if tasks:
        context = mp.get_context("spawn")
        with context.Pool(workers, initializer=_init_worker,
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

    parts = sorted(run.parts_dir.glob("part_*.npz"))
    if len(parts) != total_chunks:
        raise SystemExit(f"{len(parts)} of {total_chunks} parts present; rerun to finish")
    merged = {}
    for part in parts:
        with np.load(part) as p:
            for k in p.files:
                merged.setdefault(k, []).append(p[k])
    merged = {k: np.concatenate(v) for k, v in merged.items()}
    save_npz(run.samples_npz, **merged)
    solved = merged["exists"]
    print(f"Wrote {run.samples_npz}: {solved.sum():,} states "
          f"({np.count_nonzero(merged['source'] == 5):,} by cold start), "
          f"{np.count_nonzero(~solved):,} fold negatives")


# ------------------------------------------------------------------ export

def images(d, phi1, phi2, branch, fields):
    """The four images of states at (d, phi1, phi2): identity, mirror, reflection,
    reflection of the mirror. fields: dict of FIELDS arrays."""
    fx, fy, m1, m2, energy, p = (fields[k] for k in FIELDS)
    out = []
    for name, (a, b), sign_b, f in (
            ("x", (phi1, phi2), 1, (fx, fy, m1, m2, energy, p)),
            ("M", (-phi2, -phi1), 1, (fx, -fy, m2, m1, energy, p)),
            ("R", (-phi1, -phi2), -1, (fx, -fy, -m1, -m2, energy, -p)),
            ("RM", (phi2, phi1), -1, (fx, fy, -m2, -m1, energy, -p))):
        out.append((np.column_stack((d, a, b)), sign_b * branch, dict(zip(FIELDS, f))))
    return out


def stage_export(run):
    br = Branches(run.branches_npz)
    n = int(np.prod(br.shape))
    i, j, k = np.unravel_index(np.arange(n), br.shape)
    m = br.shape[1]
    R = np.ravel_multi_index((i, m - 1 - j, m - 1 - k), br.shape)
    coords = np.column_stack((br.axes[0][i], br.axes[1][j], br.axes[2][k]))
    up = br.up
    signs = {"fx": 1, "fy": -1, "moment": -1, "moment_right": -1, "energy": 1, "p": -1}
    # Grid rows: up at every node, down at x = reflection of up at -x.
    down_source = np.choose(br.source[R], [0, 2, 1, 3]).astype(np.int8)
    blocks = [
        dict(coords=coords, branch=np.ones(n, np.int8), exists=br.exists,
             attempted=br.attempted, source=br.source, **{f: up[f] for f in FIELDS}),
        dict(coords=coords, branch=-np.ones(n, np.int8), exists=br.exists[R],
             attempted=br.attempted[R], source=down_source,
             **{f: signs[f] * up[f][R] for f in FIELDS}),
    ]
    with np.load(run.samples_npz) as s:
        sample = {key: s[key] for key in ("cell", "offsets", "fold", "exists", "source", *FIELDS)}
    pts = point_coords(br, sample["cell"], sample["offsets"])
    label = np.where(sample["exists"], np.sign(sample["p"]), 1).astype(np.int8)
    # On phi1 + phi2 = 0 the mirror image is the point itself: keep x and R only.
    on_plane = np.abs(pts[:, 1] + pts[:, 2]) < 1e-9
    for n, (xyz, branch, f) in enumerate(images(pts[:, 0], pts[:, 1], pts[:, 2], label,
                                                {k: sample[k] for k in FIELDS})):
        rows = ~on_plane if n in (1, 3) else np.ones(len(pts), dtype=bool)
        blocks.append(dict(coords=xyz[rows], branch=branch[rows].astype(np.int8),
                           exists=sample["exists"][rows], attempted=np.ones(rows.sum(), bool),
                           source=sample["source"][rows], **{k: v[rows] for k, v in f.items()}))
    rows = {key: np.concatenate([b[key] for b in blocks]) for key in blocks[0]}
    coords = rows.pop("coords")
    for f in FIELDS:                                # targets of negatives are undefined
        rows[f] = np.where(rows["exists"], rows[f], np.nan)

    # Cell of every row, its symmetry orbit, and the split by orbit and energy band.
    def cell_of(xyz):
        ijk = [np.clip(np.floor((xyz[:, a] - br.axes[a][0]) / br.steps[a] + 1e-9).astype(np.int64),
                       0, br.cell_shape[a] - 1) for a in range(3)]
        return np.ravel_multi_index(ijk, br.cell_shape)

    cell = cell_of(coords)
    orbit = np.minimum.reduce([np.arange(len(br.cell_mirror)), br.cell_mirror, br.cell_reflect,
                               br.cell_mirror[br.cell_reflect]])
    # A node on a cell face belongs to the cell above it, and its images to cells
    # that are not the images of that cell; so take the orbit of every image.
    d, a, b = coords.T
    row_orbit = np.minimum.reduce([orbit[cell_of(np.column_stack(xyz))] for xyz in
                                   ((d, a, b), (d, -b, -a), (d, -a, -b), (d, b, a))])
    _, fold_up = cell_counts(br, {"u_ref": 1.0, "power": 1.0, "max_points": 1,
                                       "fold_min_points": 1})
    corner_energy = np.where(br.exists[br.corner_nodes], up["energy"][br.corner_nodes], np.inf)
    cell_energy = corner_energy.min(axis=0)
    orbit_energy = np.minimum.reduce([cell_energy, cell_energy[br.cell_mirror],
                                      cell_energy[br.cell_reflect]])
    split_config = {"validation": 0.1, "test": 0.1, **run.config.get("split", {})}
    rng = np.random.default_rng(int(run.config.get("seed", 0)) + 2)
    orbit_split = np.zeros(len(orbit), dtype=np.int8)
    roots = np.flatnonzero(orbit == np.arange(len(orbit)))
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        members = rng.permutation(roots[(orbit_energy[roots] >= lo) & (orbit_energy[roots] < hi)])
        n_val = int(round(split_config["validation"] * len(members)))
        n_test = int(round(split_config["test"] * len(members)))
        orbit_split[members[:n_val]] = 1
        orbit_split[members[n_val:n_val + n_test]] = 2
    rows["split"] = orbit_split[row_orbit]
    rows["cell"] = cell
    rows["fold_cell"] = np.where(rows["branch"] > 0, fold_up[cell], fold_up[br.cell_reflect[cell]])
    rows["q"] = rows["p"] / amplitude_between_layers(br.axes[0])(coords[:, 0])
    save_npz(run.dataset_npz, compress=True, d=coords[:, 0], phi1=coords[:, 1],
             phi2=coords[:, 2], sources=np.array([f"{k}: {v}" for k, v in SOURCES.items()]),
             splits=np.array(SPLITS), **rows)
    have = rows["exists"]
    print(f"Wrote {run.dataset_npz}: {len(have):,} rows, {have.sum():,} states, "
          f"{np.count_nonzero(~have):,} negatives")
    for code, name in SOURCES.items():
        print(f"  {name}: {np.count_nonzero(have & (rows['source'] == code)):,} states")
    for s, name in enumerate(SPLITS):
        print(f"  {name}: {np.count_nonzero(rows['split'] == s):,} rows")
    print(f"  states by energy: {band_counts(rows['energy'][have])}")


# ------------------------------------------------------------------- check

def stage_check(run):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_results import bvp_check
    br = Branches(run.branches_npz, with_theta=True)
    beam = Beam(run.solver_settings)
    rng = np.random.default_rng(int(run.config.get("seed", 0)) + 3)
    n = int(np.prod(br.shape))
    i, j, k = np.unravel_index(np.arange(n), br.shape)
    m = br.shape[1]
    R = np.ravel_multi_index((i, m - 1 - j, m - 1 - k), br.shape)
    coords = np.column_stack((br.axes[0][i], br.axes[1][j], br.axes[2][k]))
    up = br.up
    failed = []

    def report(name, ok, text):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {text}")
        if not ok:
            failed.append(name)

    # 1. Reflected states solve the problem: Newton from them stays put.
    reflected = np.flatnonzero(br.source == 2)
    picks = rng.choice(reflected, min(1000, len(reflected)), replace=False)
    worst, iterations, unstable = 0.0, 0, 0
    for node in picks:
        state = beam.warm(coords[node], br.theta[node], up["fx"][node], up["fy"][node])
        if state is None:
            worst = np.inf
            continue
        iterations = max(iterations, state["iterations"])
        scale = max(abs(state["fx"]), abs(state["fy"]), 1.0)
        worst = max(worst, abs(state["fx"] - up["fx"][node]) / scale,
                    abs(state["fy"] - up["fy"][node]) / scale,
                    abs(state["energy"] / up["energy"][node] - 1))
        unstable += bool(unstable_directions(br.theta[node], beam.x, up["fx"][node],
                                             up["fy"][node]))
    report("reflection", worst < 1e-6 and iterations <= 1 and not unstable,
           f"{len(picks)} reflected states re-solved: max rel change {worst:.2e}, "
           f"max {iterations} Newton iterations, {unstable} unstable")

    # 2. Both ends of every snap edge of the source run hold both branches, or are
    # on a fold (tried and failed); where both branches exist they differ by a jump.
    exists_down = br.exists[R]
    q_down = -up["q"][R]
    both = br.exists & exists_down
    stored_sign = np.where(br.source == 1, 1, -1)       # branch of the source run's state
    shape = br.shape
    snap_nodes = np.zeros(n, dtype=bool)
    for axis in range(3):
        idx = np.arange(n).reshape(shape)
        a = np.take(idx, range(shape[axis] - 1), axis=axis).ravel()
        b = np.take(idx, range(1, shape[axis]), axis=axis).ravel()
        q_a = np.where(stored_sign[a] > 0, up["q"][a], q_down[a])
        q_b = np.where(stored_sign[b] > 0, up["q"][b], q_down[b])
        snap = (np.sign(q_a) != np.sign(q_b)) & is_jump(q_a, q_b, beam.tol)
        snap_nodes[a[snap]] = snap_nodes[b[snap]] = True
    on_fold = (br.attempted & ~br.exists) | (br.attempted[R] & ~exists_down)
    covered = both | on_fold
    report("snap edges", not snap_nodes.any() or covered[snap_nodes].mean() > 0.99,
           f"{snap_nodes.sum():,} snap-edge nodes: {both[snap_nodes].mean():.1%} hold both "
           f"branches, {(on_fold & ~both)[snap_nodes].mean():.1%} on a fold, "
           f"{(~covered)[snap_nodes].sum():,} neither")
    # Without a jump between them the two must be one state with p ~ 0, labelled
    # both ways: on phi1 = phi2 the S-shape theta(s) = theta(L - s) has p = 0.
    distinct = is_jump(up["q"][both], q_down[both], beam.tol)
    one_state = (np.isclose(up["energy"][both], up["energy"][R][both], rtol=1e-6)
                 & np.isclose(up["fx"][both], up["fx"][R][both], rtol=1e-6, atol=1e-8))
    report("bistable", (distinct | one_state).all(),
           f"{both.sum():,} nodes ({both.mean():.1%}) with both branches; "
           f"{np.count_nonzero(~distinct & one_state):,} hold one p ~ 0 state under both "
           f"labels; {np.count_nonzero(~distinct & ~one_state):,} pairs of different states "
           "without a jump between them")
    for i_d in range(0, shape[0], max(1, shape[0] // 6)):
        layer = slice(i_d * shape[1] * shape[2], (i_d + 1) * shape[1] * shape[2])
        print(f"    d = {br.axes[0][i_d]:.3f}: both branches at {both[layer].mean():.1%}, "
              f"extended {np.count_nonzero(br.source[layer] == 3):,}")

    # 3. Refined states in cells whose corners all hold up. Their distance from
    # trilinear interpolation of the corners is only reported: near d = 1 the
    # fields vary steeply and U has valleys between the nodes, which is what the
    # refinement is for. The test is that the state does not depend on the corner
    # it was warm-started from: re-solved from the farthest one, it is the same.
    with np.load(run.samples_npz) as s:
        sample = {key: s[key] for key in ("cell", "offsets", "fold", "exists", "source",
                                          *FIELDS)}
        theta_rows = s["theta"]
    solved = sample["exists"]
    corner_ok = br.exists[br.corner_nodes].all(axis=0)
    smooth = solved & ~sample["fold"] & corner_ok[sample["cell"]] & (sample["p"] > 0)
    pts = point_coords(br, sample["cell"], sample["offsets"])
    energy_rel = np.abs(sample["energy"][smooth] - trilinear(
        br, sample["cell"][smooth], sample["offsets"][smooth], up["energy"])) / \
        sample["energy"][smooth]
    for lo in np.unique(np.floor(pts[smooth, 0] / br.steps[0] + 1e-9)):
        layer = np.floor(pts[smooth, 0] / br.steps[0] + 1e-9) == lo
        print(f"    d in [{lo * br.steps[0]:.3f}, {(lo + 1) * br.steps[0]:.3f}): energy vs "
              f"trilinear, median rel {np.median(energy_rel[layer]):.1e}, "
              f"p99 {np.percentile(energy_rel[layer], 99):.1e}")
    picks = rng.choice(np.flatnonzero(smooth), min(500, smooth.sum()), replace=False)
    differ = rejected = 0
    amplitude = amplitude_between_layers(br.axes[0])
    for row in picks:
        corners = br.corner_nodes[:, sample["cell"][row]]
        far = int(corners[np.argmax(np.sum((CORNERS - sample["offsets"][row]) ** 2, axis=1))])
        state = beam.warm(pts[row], br.theta[far], up["fx"][far], up["fy"][far], max_it=200)
        # A start that diverges, jumps or ends unstable would have been rejected.
        if not beam.keeps_branch(state, up["q"][far], float(amplitude(pts[row, 0]))):
            rejected += 1
        elif not np.isclose(state["energy"], sample["energy"][row], rtol=1e-6):
            differ += 1
    report("corner independence", differ == 0,
           f"{len(picks)} refined states re-solved from the farthest corner: {differ} reach "
           f"a different accepted state, {rejected} starts rejected (diverged, jumped or "
           "unstable)")

    # 4. Independent BVP solves of a random sample of refined states.
    picks = rng.choice(np.flatnonzero(solved), min(200, solved.sum()), replace=False)
    errors, branch_kept, converged = [], 0, 0
    for row in picks:
        out = bvp_check(*pts[row], theta_rows[row].astype(float), br.beam_x,
                        sample["fx"][row], sample["fy"][row])
        if out is None:
            continue
        converged += 1
        fx, fy, m0, m1, energy, p = out
        branch_kept += np.sign(p) == np.sign(sample["p"][row])
        sf = max(abs(fx), abs(fy), 1.0)
        sm = max(abs(m0), abs(m1), 1.0)
        errors.append(max(abs(sample["fx"][row] - fx) / sf, abs(sample["fy"][row] - fy) / sf,
                          abs(sample["moment"][row] - m0) / sm,
                          abs(sample["moment_right"][row] - m1) / sm,
                          abs(sample["energy"][row] / energy - 1)))
    errors = np.array(errors or [np.inf])
    report("bvp", converged >= 0.98 * len(picks) and branch_kept == converged
           and errors.max() < 2e-2,
           f"{len(picks)} sampled, {converged} converged, {branch_kept} same branch, "
           f"median rel {np.median(errors):.1e}, max {errors.max():.1e}")

    # 5. Both symmetries on the dataset: every state's three images are rows too.
    with np.load(run.dataset_npz) as ds:
        data = {key: ds[key] for key in ("d", "phi1", "phi2", "branch", "exists", *FIELDS,
                                         "split", "source")}
    have = data["exists"]
    rows_have = np.flatnonzero(have)
    key = np.column_stack([np.rint(data[c][rows_have] * 1e8).astype(np.int64)
                           for c in ("d", "phi1", "phi2")]
                          + [data["branch"][rows_have].astype(np.int64)])

    def hashed(keys):                    # int64 arithmetic wraps; hits are verified below
        primes = np.array([1000003, 998244353, 1000000007, 7919], dtype=np.int64)
        return (keys * primes).sum(axis=1)

    order = np.argsort(hashed(key))
    sorted_hash = hashed(key)[order]
    picks = rng.choice(len(rows_have), min(20000, len(rows_have)), replace=False)
    d_i, a, b, s_ = key[picks].T
    bad, leak = 0, 0
    for image, expect in ((np.column_stack((d_i, -b, -a, s_)), {"fy": -1, "p": 1}),
                          (np.column_stack((d_i, -a, -b, -s_)), {"fy": -1, "p": -1}),
                          (np.column_stack((d_i, b, a, -s_)), {"fy": 1, "p": -1})):
        h = hashed(image)
        pos = np.clip(np.searchsorted(sorted_hash, h), 0, len(sorted_hash) - 1)
        hit = order[pos]
        found = (sorted_hash[pos] == h) & np.all(key[hit] == image, axis=1)
        src_rows, other = rows_have[picks], rows_have[hit]
        scale_e = data["energy"][src_rows]
        scale_f = np.maximum(1.0, np.abs(data["fx"][src_rows]))
        # On phi1 = -phi2 the first image is the state itself, so symmetry forces fy = 0;
        # the solver resolves that zero only to ~1e-8, hence the absolute floor.
        scale_y = np.maximum(100.0, np.abs(data["fy"][src_rows]))
        same = (found
                & (np.abs(data["energy"][other] - scale_e) <= 1e-9 * scale_e)
                & (np.abs(data["fx"][other] - data["fx"][src_rows]) <= 1e-9 * scale_f)
                & (np.abs(data["fy"][other] - expect["fy"] * data["fy"][src_rows]) <= 1e-9 * scale_y)
                & (np.abs(data["p"][other] - expect["p"] * data["p"][src_rows]) <= 1e-9))
        bad += int(np.count_nonzero(~same))
        leak += int(np.count_nonzero(found & (data["split"][other] != data["split"][src_rows])))
    report("symmetry", bad == 0 and leak == 0,
           f"{len(picks):,} states checked: {bad} images missing or different, "
           f"{leak} images in another split")

    # 6. The energy histogram of the dataset.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postprocessing"))
    from postprocess import energy_histogram
    Path("post").mkdir(exist_ok=True)
    energy_histogram(data["energy"][have], data["d"][have], br.axes[0],
                     "post/energy_histogram.png", what="states")
    print(f"Wrote post/energy_histogram.png; states by energy: {band_counts(data['energy'][have])}")
    if run.config.get("scheme", "cells") == "iso":
        import refine_iso
        failed += refine_iso.check(run, data, have)
    print("FAILED: " + ", ".join(failed) if failed else "All checks passed")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("config")
    ap.add_argument("stage", choices=("branches", "sample", "export", "check"))
    ap.add_argument("--dry-run", action="store_true", help="count and predict; no solves")
    args = ap.parse_args()
    run = Run(args.config)
    print(f"Run directory: {os.getcwd()}")
    if args.stage == "branches":
        stage_branches(run, args.dry_run)
    elif args.stage == "sample":
        stage_sample(run, args.dry_run)
    elif args.stage == "export":
        stage_export(run)
    else:
        sys.exit(stage_check(run))


if __name__ == "__main__":
    main()
