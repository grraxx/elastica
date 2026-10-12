"""
main.py
Solve the clamped elastica over a uniform tetrahedral (d, phi1, phi2) mesh.

usage: python main.py [path/to/config.json]   (outputs go next to the config)

Only the half phi1 + phi2 >= 0 is solved; the mirror (phi1, phi2) ->
(-phi2, -phi1) fills the other half.

There is one planned cold start: the exact first clamped mode at the angular
origin (phi1 = phi2 = 0) at the smallest d. From there the solved region grows
as a front, nearest grid distance first. Each node is warm-started from its
solved grid neighbours, nearest first. A result is kept unless p, the
projection on the first buckling mode, jumps (judged on p / a(d), a(d) = |p|
of the exact first mode at that d; see branch.py): then Newton has left the
neighbour's branch, for the other hump or for the unstable S-shaped family.

A node that cannot be reached without a jump is deferred, and retried with
each new neighbour that gets solved. After max_failures_per_branch failed
attempts from one branch it stops trying that branch, so a node past a fold
soon just waits for the other. The branch a fold cuts off is therefore not
entered by jumping across the snap surface; the front reaches it the long way
round, through the region where p crosses zero smoothly, and approaches it
from the far side. Only when the front stops altogether is a deferred node
cold-started (exact first mode, continued in the clamp angles).

A result that passes the jump test is also rejected if it is unstable (see
stability.py): on the anti-diagonal phi1 = -phi2 Newton follows the
symmetric bent-up branch smoothly past the bifurcation where it loses
stability, with no jump in p to give it away.
"""

import os

# One Newton solve has ~100 DOFs, so threaded BLAS/MUMPS only adds overhead
# (about 40x slower per solve). Must be set before PETSc is loaded.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import heapq
import itertools
import json
import signal
import sys
import time

import basix.ufl
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from branch import first_mode_amplitude, first_mode_projection, is_jump
from stability import unstable_directions
from mesh_generator import build_rectilinear_mesh, mesh_matches, mesh_settings
from solver import ElasticaSolver
from vtu import write_vtu


ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"

# 26-neighbourhood in grid-index space, face neighbours first. It contains
# every edge of the tetrahedral mesh, and is symmetric under the mirror.
NEIGHBOR_OFFSETS = sorted(
    (offset for offset in itertools.product((-1, 0, 1), repeat=3) if any(offset)),
    key=lambda offset: sum(c * c for c in offset),
)


def colored_status(message, color):
    return f"{color}{message}{ANSI_RESET}"


def precompute_elastica_spaces(L=1.0, N=100, grading=0.0):
    """
    Build the beam mesh and function spaces, which are independent of the
    parameters (d, phi1, phi2) and reused for every solve.

    grading in [0, 1) refines the mesh towards both clamps, s = L g(u) with
    g(u) = (1 - grading) u + grading (1 - cos(pi u)) / 2: under high tension
    (d near 1, large clamp angles) the rod turns within boundary layers of
    width 1 / sqrt(|fx|), down to about 0.007 L on the production grid. The
    map is symmetric, g(1 - u) = 1 - g(u), as the mirror solutions require.

    Returns
    -------
    tuple (V, R, beam_x, beam_order)
        V: function space for theta (Lagrange, degree 1)
        R: real space for the force multipliers
        beam_x: sorted dof coordinates along the beam
        beam_order: permutation taking theta dof values to beam_x order
    """
    domain = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, L])
    u = domain.geometry.x[:, 0] / L
    domain.geometry.x[:, 0] = L * ((1.0 - grading) * u
                                   + grading * 0.5 * (1.0 - np.cos(np.pi * u)))
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1))
    R = fem.functionspace(domain, basix.ufl.real_element(domain.basix_cell()))
    dof_x = V.tabulate_dof_coordinates()[:, 0]
    beam_order = np.argsort(dof_x)
    return V, R, dof_x[beam_order], beam_order


def end_moments(theta_sorted, beam_x, fx, fy):
    """Return theta'(0) and theta'(L), the clamp moments over EI.

    The slope of the end element is theta' at its midpoint; the half-element
    Taylor correction with theta'' = -fx sin(theta) + fy cos(theta) makes the
    end values second-order accurate."""
    slopes = np.diff(theta_sorted) / np.diff(beam_x)
    h0, h1 = beam_x[1] - beam_x[0], beam_x[-1] - beam_x[-2]

    def curvature_rate(theta):
        return -fx * np.sin(theta) + fy * np.cos(theta)

    return (float(slopes[0] - 0.5 * h0 * curvature_rate(theta_sorted[0])),
            float(slopes[-1] + 0.5 * h1 * curvature_rate(theta_sorted[-1])))


def choose_start_node(coords, solve_mask):
    """The solve node closest to the angular origin, at the smallest d."""
    candidates = np.flatnonzero(solve_mask)
    key = np.lexsort((
        candidates,
        coords[candidates, 0],
        np.hypot(coords[candidates, 1], coords[candidates, 2]),
    ))
    return int(candidates[key[0]])


def grid_neighbors(node_idx, shape):
    """Yield the grid neighbours of node_idx, nearest first."""
    plane = shape[1] * shape[2]
    i, rest = divmod(node_idx, plane)
    j, k = divmod(rest, shape[2])
    for di, dj, dk in NEIGHBOR_OFFSETS:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
            yield ni * plane + nj * shape[2] + nk


def save_results_npz(output_npz, coords, tets, shape, axes, fields, beam_x,
                     compress=True):
    """Write the archive next to its target, then rename it into place, so an
    interrupted write never leaves a broken file behind."""
    temporary = f"{output_npz}.tmp"
    with open(temporary, "wb") as file:
        (np.savez_compressed if compress else np.savez)(
            file, coords=coords, tets=tets, shape=np.asarray(shape),
            d_vals=axes[0], phi1_vals=axes[1], phi2_vals=axes[2], beam_x=beam_x,
            **fields,
        )
    os.replace(temporary, output_npz)


def main():
    if MPI.COMM_WORLD.size > 1:
        raise SystemExit("main.py runs serially; do not launch it with mpirun")

    # Step 0: Read config. Every output path in it is relative to the
    # config's folder, which becomes the run directory.
    config_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    os.chdir(os.path.dirname(config_path))
    print(f"Run directory: {os.getcwd()}")

    # Step 1: Generate or load the solution-space mesh
    npz_file = config.get("output_npz", "solution_space.npz")
    settings = mesh_settings(config)
    if not settings["symmetric_mesh"]:
        raise ValueError("The solver solves half the space; set solve_phi_symmetry to true")
    if mesh_matches(npz_file, settings):
        print(f"Loading existing solution space mesh: {npz_file}")
    else:
        print("Generating solution space mesh...")
        build_rectilinear_mesh(**settings, out_npz=npz_file)

    with np.load(npz_file) as data:
        coords = data["coords"]          # (N, 3) array of (d, phi1, phi2)
        tets = data["tets"]              # (M, 4) tetrahedron connectivity
        shape = tuple(int(n) for n in data["shape"])
        mirror_indices = data["mirror_indices"].astype(np.int64)
        solve_mask = data["solve_mask"].astype(bool)
    num_nodes = coords.shape[0]
    axes = (coords[::shape[1] * shape[2], 0],
            coords[:shape[1] * shape[2]:shape[2], 1],
            coords[:shape[2], 2])
    print(f"Mesh: {shape[0]} x {shape[1]} x {shape[2]} = {num_nodes:,} nodes, "
          f"{len(tets):,} tetrahedra; steps "
          f"{', '.join(f'{np.diff(a).mean():.4g}' for a in axes)}; "
          f"{np.count_nonzero(solve_mask):,} solved directly, the rest mirrored")

    # Step 2: Beam mesh, function spaces and the reusable nonlinear solver
    beam_elements = int(config.get("beam_elements", 100))
    beam_grading = float(config.get("beam_grading", 0.0))
    V, R, beam_x, beam_order = precompute_elastica_spaces(
        L=1.0, N=beam_elements, grading=beam_grading)
    print(f"Beam: {beam_elements} elements, grading {beam_grading}, element "
          f"length {np.diff(beam_x).min():.4f} to {np.diff(beam_x).max():.4f}")
    num_beam_dofs = len(beam_x)
    elastica_solver = ElasticaSolver(
        V, R,
        petsc_options={
            "snes_max_it": 500,
            "snes_rtol": float(config.get("snes_rtol", 1e-8)),
            "snes_atol": float(config.get("snes_atol", 1e-10)),
        },
    )

    # A warm start from an adjacent node converges in a few Newton iterations;
    # one that needs more is a failed attempt, so stop early.
    warm_start_max_it = int(config.get("warm_start_max_it", 50))
    # Jumps are judged on q = p / a(d), a(d) = |p| of the exact first mode at
    # (d, 0, 0), so the tolerance means the same at every d (see branch.py).
    jump_tol = float(config.get("jump_tol", 0.17))
    amplitude = first_mode_amplitude(axes[0])[
        np.arange(num_nodes) // (shape[1] * shape[2])]
    # After this many failed warm starts from neighbours on one branch (sign
    # of p), a node stops trying that branch and waits for the other one: a
    # node past a fold would otherwise fail from every neighbour in turn.
    max_failures_per_branch = int(config.get("max_failures_per_branch", 3))
    checkpoint_interval = int(config.get("visualization_interval", 0))
    progress_interval = max(1, int(config.get("progress_interval", 10000)))
    verbose_solves = bool(config.get("verbose_solves", False))
    save_theta = bool(config.get("save_theta", False))
    live_results_npz = config.get("live_results_npz", "solution_space_live.npz")
    live_output_vtu = config.get("live_output_vtu", "solution_space_live.vtu")

    # Step 3: Result arrays, in the order of the coords array. theta is kept
    # for every node because neighbours warm-start from it; float32 is ample
    # for a starting guess and halves the largest array (~2.4 GB at 3 M nodes
    # and 201 beam dofs).
    fx_arr = np.full(num_nodes, np.nan)
    fy_arr = np.full(num_nodes, np.nan)
    moment_arr = np.full(num_nodes, np.nan)
    energy_arr = np.full(num_nodes, np.nan)
    p_arr = np.full(num_nodes, np.nan)
    order_arr = np.full(num_nodes, -1, dtype=np.int64)   # when each node was solved
    waited_arr = np.zeros(num_nodes, dtype=bool)         # solved only after waiting
    solved_arr = np.zeros(num_nodes, dtype=bool)
    theta_arr = np.full((num_nodes, num_beam_dofs), np.nan, dtype=np.float32)

    if os.path.exists(live_results_npz):
        print(f"Loading checkpoint: {live_results_npz}")
        with np.load(live_results_npz) as checkpoint:
            if (checkpoint["coords"].shape != coords.shape
                    or not np.allclose(checkpoint["coords"], coords)
                    or checkpoint["theta"].shape != theta_arr.shape):
                raise ValueError(
                    f"Checkpoint {live_results_npz} does not match this mesh "
                    f"and {beam_elements} beam elements; delete it to start over"
                )
            for name, array in (("fx", fx_arr), ("fy", fy_arr),
                                ("moment", moment_arr), ("energy", energy_arr),
                                ("p", p_arr), ("solve_order", order_arr),
                                ("waited", waited_arr), ("theta", theta_arr)):
                if name in checkpoint.files:
                    array[:] = checkpoint[name]
            solved_arr[:] = checkpoint["solved"].astype(bool)
            if "p" not in checkpoint.files:
                p_arr[solved_arr] = first_mode_projection(theta_arr[solved_arr], beam_x)
        print(f"Resuming with {np.count_nonzero(solved_arr):,}/{num_nodes:,} "
              "points already solved")

    def output_fields(include_theta):
        # The mirror node's left moment is this node's right one.
        return {"fx": fx_arr, "fy": fy_arr, "moment": moment_arr,
                "moment_right": moment_arr[mirror_indices],
                "energy": energy_arr, "p": p_arr, "q": p_arr / amplitude,
                "branch": np.sign(p_arr),
                "solve_order": order_arr, "waited": waited_arr.astype(float),
                "solved": solved_arr.astype(float),
                **({"theta": theta_arr} if include_theta else {})}

    def save_checkpoint():
        # Uncompressed: theta alone is ~2.4 GB on the production grid.
        save_results_npz(live_results_npz, coords, tets, shape, axes,
                         output_fields(True), beam_x, compress=False)
        if live_output_vtu:
            write_vtu(live_output_vtu, coords, tets, output_fields(False),
                      solved=solved_arr)

    next_order = [int(order_arr.max()) + 1]

    def store(node_idx, fx, fy, moment, energy, p, theta_sorted):
        fx_arr[node_idx] = fx
        fy_arr[node_idx] = fy
        moment_arr[node_idx] = moment
        energy_arr[node_idx] = energy
        p_arr[node_idx] = p
        theta_arr[node_idx] = theta_sorted
        order_arr[node_idx] = next_order[0]
        next_order[0] += 1
        solved_arr[node_idx] = True

    # Step 4: The front. Priority is squared grid distance from the start node,
    # so the solved region grows outwards from the cold start.
    start_idx = choose_start_node(coords, solve_mask)
    grid_index = np.column_stack(np.unravel_index(np.arange(num_nodes), shape))
    priority = np.sum((grid_index - grid_index[start_idx]) ** 2, axis=1)
    del grid_index
    print(f"Start node {start_idx}, (d, phi1, phi2) = "
          f"{coords[start_idx].round(6).tolist()}")

    def canonical(node_idx):
        return node_idx if solve_mask[node_idx] else int(mirror_indices[node_idx])

    heap = []
    queued = np.zeros(num_nodes, dtype=bool)

    def push(node_idx):
        if not solved_arr[node_idx] and not queued[node_idx]:
            queued[node_idx] = True
            heapq.heappush(heap, (int(priority[node_idx]), node_idx))

    def push_unsolved_neighbors(node_idx):
        for neighbor in grid_neighbors(node_idx, shape):
            if not solved_arr[neighbor]:
                push(canonical(neighbor))

    tried = {}          # deferred node -> neighbours already tried as warm starts
    failures = {}       # deferred node -> {branch sign: failed warm starts}
    for node_idx in np.flatnonzero(solved_arr):
        push_unsolved_neighbors(int(node_idx))

    solve_time_total = 0.0
    solve_attempts = 0
    stats = {"diverged": 0, "jumps rejected": 0, "unstable rejected": 0,
             "deferrals": 0, "extra cold starts": 0, "cold starts failed": 0}

    def timed(attempt, *args, **kwargs):
        nonlocal solve_time_total, solve_attempts
        solve_start = time.perf_counter()
        try:
            return attempt(*args, **kwargs)
        except RuntimeError as error:
            if verbose_solves:
                print(colored_status(f"  {error}", ANSI_RED))
            return None
        finally:
            solve_time_total += time.perf_counter() - solve_start
            solve_attempts += 1

    def accept(node_idx, solution):
        theta_sorted = solution.theta[beam_order]
        p = float(first_mode_projection(theta_sorted, beam_x))
        left_moment, right_moment = end_moments(theta_sorted, beam_x,
                                                solution.fx, solution.fy)
        store(node_idx, solution.fx, solution.fy, left_moment, solution.energy,
              p, theta_sorted)
        # Mirror: theta~(s) = -theta(L - s) solves (d, -phi2, -phi1) with
        # fx~ = fx, fy~ = -fy, theta~'(0) = theta'(L) and the same p.
        mirror_idx = int(mirror_indices[node_idx])
        if mirror_idx != node_idx:
            store(mirror_idx, solution.fx, -solution.fy, right_moment,
                  solution.energy, p, -theta_sorted[::-1])
        tried.pop(node_idx, None)
        failures.pop(node_idx, None)
        push_unsolved_neighbors(node_idx)
        if mirror_idx != node_idx:
            push_unsolved_neighbors(mirror_idx)

    def cold_start(node_idx):
        solution = timed(elastica_solver.cold_solve, *coords[node_idx])
        if solution is not None and unstable_directions(
                solution.theta[beam_order], beam_x, solution.fx, solution.fy):
            solution = None
        if solution is None:
            stats["cold starts failed"] += 1
            print(colored_status(
                f"  Cold start failed at node {node_idx}, (d, phi1, phi2) = "
                f"{coords[node_idx].round(6).tolist()}", ANSI_RED))
            return False
        accept(node_idx, solution)
        return True

    # Ctrl-C only sets a flag: an exception raised inside PETSc's callbacks
    # would be swallowed as a failed solve. A second Ctrl-C aborts at once.
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        signal.signal(signal.SIGINT, signal.default_int_handler)
        print("Interrupt received; stopping after the current node "
              "(Ctrl-C again to abort without saving)...")

    signal.signal(signal.SIGINT, request_stop)

    if not solved_arr.any() and not cold_start(start_idx):
        raise RuntimeError("The cold start at the angular origin failed")

    # Step 5: Grow the front; when it stops, cold-start the nearest deferred
    # node (all its solved neighbours have failed) and carry on.
    last_checkpoint_count = last_progress_count = int(np.count_nonzero(solved_arr))
    while not stop_requested:
        while heap and not stop_requested:
            _, node_idx = heapq.heappop(heap)
            queued[node_idx] = False
            if solved_arr[node_idx]:
                continue
            node_tried = tried.setdefault(node_idx, set())
            node_failures = failures.setdefault(node_idx, {1.0: 0, -1.0: 0, 0.0: 0})
            solution = None
            for neighbor in grid_neighbors(node_idx, shape):
                if not solved_arr[neighbor] or neighbor in node_tried:
                    continue
                branch = float(np.sign(p_arr[neighbor]))
                if node_failures[branch] >= max_failures_per_branch:
                    continue
                node_tried.add(neighbor)
                theta0 = np.empty(num_beam_dofs)
                theta0[beam_order] = theta_arr[neighbor]
                candidate = timed(
                    elastica_solver.solve, *coords[node_idx], theta0,
                    fx_arr[neighbor], fy_arr[neighbor], max_it=warm_start_max_it,
                )
                if candidate is None:
                    stats["diverged"] += 1
                    node_failures[branch] += 1
                    continue
                p = float(first_mode_projection(candidate.theta[beam_order], beam_x))
                if is_jump(p_arr[neighbor] / amplitude[neighbor],
                           p / amplitude[node_idx], jump_tol):
                    stats["jumps rejected"] += 1
                    node_failures[branch] += 1
                    if verbose_solves:
                        print(colored_status(
                            f"  Node {node_idx}: p jumps from neighbour {neighbor} "
                            f"(p {p_arr[neighbor]:+.4f} -> {p:+.4f}); deferred",
                            ANSI_YELLOW))
                    continue
                if unstable_directions(candidate.theta[beam_order], beam_x,
                                       candidate.fx, candidate.fy):
                    stats["unstable rejected"] += 1
                    node_failures[branch] += 1
                    if verbose_solves:
                        print(colored_status(
                            f"  Node {node_idx}: unstable from neighbour {neighbor} "
                            f"(p {p:+.4f}); deferred", ANSI_YELLOW))
                    continue
                solution = candidate
                break
            if solution is None:
                stats["deferrals"] += 1
                waited_arr[[node_idx, mirror_indices[node_idx]]] = True
                continue
            accept(node_idx, solution)

            solved_count = int(np.count_nonzero(solved_arr))
            if solved_count - last_progress_count >= progress_interval:
                last_progress_count = solved_count
                print(f"  Progress: {solved_count:,}/{num_nodes:,} solved, "
                      f"{solve_time_total / max(solve_attempts, 1) * 1e3:.2f} ms "
                      f"per solve, {len(tried):,} nodes waiting")
            if (checkpoint_interval > 0
                    and solved_count - last_checkpoint_count >= checkpoint_interval):
                save_checkpoint()
                last_checkpoint_count = solved_count

        waiting = [node for node in tried if not solved_arr[node]]
        if stop_requested or not waiting:
            break
        node_idx = min(waiting, key=lambda node: (priority[node], node))
        stats["extra cold starts"] += 1
        print(colored_status(
            f"  Front stopped with {len(waiting):,} nodes waiting; cold start at "
            f"node {node_idx}, (d, phi1, phi2) = "
            f"{coords[node_idx].round(6).tolist()}", ANSI_YELLOW))
        if not cold_start(node_idx):
            tried.pop(node_idx)     # give up on this node, try the next one

    solved_count = int(np.count_nonzero(solved_arr))
    print(f"Solved {solved_count:,}/{num_nodes:,} points "
          f"({100 * solved_count / num_nodes:.1f}%); "
          + ", ".join(f"{value:,} {name}" for name, value in stats.items())
          + f"; bent up {np.mean(p_arr[solved_arr] > 0):.1%} of solved points")
    save_checkpoint()
    if stop_requested:
        print(f"Checkpoint saved to {live_results_npz}; rerun to resume")
        return

    # Step 6: Save NPZ and ParaView-readable VTU output.
    output_npz = config.get("results_npz", "solution_space_results.npz")
    fields = output_fields(save_theta)
    save_results_npz(output_npz, coords, tets, shape, axes, fields, beam_x)
    print(f"Results saved to {output_npz}")
    output_vtu = config.get("output_vtu", "solution_space_results.vtu")
    write_vtu(output_vtu, coords, tets, output_fields(False))
    print(f"VTU results saved to {output_vtu}")


if __name__ == "__main__":
    main()
