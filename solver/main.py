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

Where both branches are stable a node stores the bent-up state: its solved
up neighbours (p > 0) are tried before its down ones. A node reached before
any up neighbour was solved can still take the down state, so after the
front a promotion pass revisits every down node beside an up one: the
reflection theta -> -theta of the state at (-phi1, -phi2), when that is a
second, down state, gives an up state with no solve; otherwise each up
neighbour warm-starts it, with the same tests. Promoted nodes pass the
pass on to their down neighbours, until nothing changes. The snap surface
is then the fold of the bent-up branch, whatever the visit order.
"""

import os

# One Newton solve has ~100 DOFs, so threaded BLAS/MUMPS only adds overhead
# (about 40x slower per solve). Must be set before PETSc is loaded.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import heapq
import itertools
from collections import deque
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
    promoted_arr = np.zeros(num_nodes, dtype=bool)       # down state replaced by up
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
                                ("waited", waited_arr), ("promoted", promoted_arr),
                                ("theta", theta_arr)):
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
                "promoted": promoted_arr.astype(float),
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
            # Up neighbours first (the sort is stable, so nearest first within
            # each branch): where both branches exist the node takes the up one.
            solved_neighbors = sorted(
                (nb for nb in grid_neighbors(node_idx, shape) if solved_arr[nb]),
                key=lambda nb: p_arr[nb] <= 0)
            for neighbor in solved_neighbors:
                if neighbor in node_tried:
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

    # Step 6: Promotion. A node reached before any up neighbour was solved can
    # hold the down state where the up one is stable too; give it the up one.
    def store_reflection(node_idx, fx, fy, left_moment, right_moment, energy, p,
                         theta_sorted):
        # theta -> -theta: the same fx and energy, opposite fy, moments and p.
        store(node_idx, fx, -fy, -left_moment, energy, -p, -theta_sorted)
        mirror_idx = int(mirror_indices[node_idx])
        if mirror_idx != node_idx:
            store(mirror_idx, fx, fy, -right_moment, energy, -p, theta_sorted[::-1])
        promoted_arr[[node_idx, mirror_idx]] = True

    def promote():
        counts = {"by reflection": 0, "by warm start": 0, "diverged": 0,
                  "jumped": 0, "unstable": 0}
        i_idx, rest = np.divmod(np.arange(num_nodes), shape[1] * shape[2])
        j_idx, k_idx = np.divmod(rest, shape[2])
        if np.allclose(axes[1], -axes[1][::-1]) and np.allclose(axes[2], -axes[2][::-1]):
            # The state at -x, reflected, is an up state at x when it is down
            # there; it is a second state unless it is the stored one itself
            # (the p = 0 S-shape on phi1 = phi2), told apart by a jump in q.
            reflect = (i_idx * shape[1] + shape[1] - 1 - j_idx) * shape[2] + shape[2] - 1 - k_idx
            q_arr = p_arr / amplitude
            down = solved_arr & (p_arr < 0)
            nodes = np.flatnonzero(solve_mask & down & down[reflect])
            nodes = nodes[is_jump(q_arr[nodes], -q_arr[reflect[nodes]], jump_tol)]
            source = reflect[nodes]
            # Copied first: storing a node's mirror can overwrite another's source.
            values = (fx_arr[source], fy_arr[source], moment_arr[source],
                      moment_arr[mirror_indices[source]], energy_arr[source],
                      p_arr[source], theta_arr[source])
            for row, node_idx in enumerate(nodes):
                store_reflection(int(node_idx), *(v[row] for v in values))
            counts["by reflection"] = len(nodes)
            del reflect, q_arr, down, values
        del i_idx, rest, j_idx, k_idx

        # Then a front from the up nodes into the down nodes beside them, with
        # the front's tests, until every attempt fails: past the fold.
        up = (solved_arr & (p_arr > 0)).reshape(shape)
        padded = np.pad(up, 1)
        near = np.zeros(shape, dtype=bool)
        for di, dj, dk in NEIGHBOR_OFFSETS:
            near |= padded[1 + di:1 + di + shape[0], 1 + dj:1 + dj + shape[1],
                           1 + dk:1 + dk + shape[2]]
        seeds = np.flatnonzero(near.ravel() & solve_mask & solved_arr & (p_arr < 0))
        del up, padded, near
        print(f"Promotion: {counts['by reflection']:,} nodes by reflection; "
              f"front from {len(seeds):,} down nodes beside an up node")
        failures_up = np.zeros(num_nodes, dtype=np.int16)
        tried_up = {}
        queue, in_queue = deque(), np.zeros(num_nodes, dtype=bool)

        def push(node_idx):
            node_idx = canonical(node_idx)
            if (p_arr[node_idx] < 0 and failures_up[node_idx] < max_failures_per_branch
                    and not in_queue[node_idx]):
                in_queue[node_idx] = True
                queue.append(node_idx)

        for node_idx in seeds:
            push(int(node_idx))
        attempts, last_report = 0, 0
        while queue and not stop_requested:
            node_idx = queue.popleft()
            in_queue[node_idx] = False
            if not p_arr[node_idx] < 0 or failures_up[node_idx] >= max_failures_per_branch:
                continue
            node_tried = tried_up.setdefault(node_idx, set())
            for neighbor in grid_neighbors(node_idx, shape):
                if not p_arr[neighbor] > 0 or neighbor in node_tried:
                    continue
                node_tried.add(neighbor)
                theta0 = np.empty(num_beam_dofs)
                theta0[beam_order] = theta_arr[neighbor]
                candidate = timed(
                    elastica_solver.solve, *coords[node_idx], theta0,
                    fx_arr[neighbor], fy_arr[neighbor], max_it=warm_start_max_it,
                )
                attempts += 1
                if candidate is None:
                    counts["diverged"] += 1
                else:
                    p = float(first_mode_projection(candidate.theta[beam_order], beam_x))
                    if p <= 0 or is_jump(p_arr[neighbor] / amplitude[neighbor],
                                         p / amplitude[node_idx], jump_tol):
                        counts["jumped"] += 1
                    elif unstable_directions(candidate.theta[beam_order], beam_x,
                                             candidate.fx, candidate.fy):
                        counts["unstable"] += 1
                    else:
                        accept(node_idx, candidate)
                        mirror_idx = int(mirror_indices[node_idx])
                        promoted_arr[[node_idx, mirror_idx]] = True
                        counts["by warm start"] += 1
                        for nb in itertools.chain(grid_neighbors(node_idx, shape),
                                                  grid_neighbors(mirror_idx, shape)):
                            push(nb)
                        break
                failures_up[node_idx] += 1
                if failures_up[node_idx] >= max_failures_per_branch:
                    break
            if attempts - last_report >= progress_interval:
                last_report = attempts
                print(f"  Promotion: {attempts:,} warm starts, queue {len(queue):,}: "
                      + ", ".join(f"{v:,} {k}" for k, v in counts.items()), flush=True)
        print(f"Promotion {'interrupted' if stop_requested else 'done'}: {attempts:,} "
              "warm starts; " + ", ".join(f"{v:,} {k}" for k, v in counts.items())
              + f"; {np.count_nonzero(promoted_arr):,} nodes promoted in all")

    if not stop_requested:
        promote()

    solved_count = int(np.count_nonzero(solved_arr))
    print(f"Solved {solved_count:,}/{num_nodes:,} points "
          f"({100 * solved_count / num_nodes:.1f}%); "
          + ", ".join(f"{value:,} {name}" for name, value in stats.items())
          + f"; bent up {np.mean(p_arr[solved_arr] > 0):.1%} of solved points")
    save_checkpoint()
    if stop_requested:
        print(f"Checkpoint saved to {live_results_npz}; rerun to resume")
        return

    # Step 7: Save NPZ and ParaView-readable VTU output.
    output_npz = config.get("results_npz", "solution_space_results.npz")
    fields = output_fields(save_theta)
    save_results_npz(output_npz, coords, tets, shape, axes, fields, beam_x)
    print(f"Results saved to {output_npz}")
    output_vtu = config.get("output_vtu", "solution_space_results.vtu")
    write_vtu(output_vtu, coords, tets, output_fields(False))
    print(f"VTU results saved to {output_vtu}")


if __name__ == "__main__":
    main()
