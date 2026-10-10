"""
main.py
Main script to run the elastica solver over the solution space.
Nodes are visited in order of grid distance from the cold-start node, and
each one is warm-started from its nearest already-solved grid neighbour.
"""

import os

# One Newton solve has ~100 DOFs, so threaded BLAS/MUMPS only adds overhead
# (about 40x slower per solve). Must be set before PETSc is loaded.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools
import json
import signal
import time
from typing import Optional

import basix.ufl
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from mesh_generator import build_rectilinear_mesh, mesh_matches, mesh_settings
from solver import ElasticaSolver
from vtu import write_vtu


ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"

# 26-neighbourhood in grid-index space, face neighbours first.
NEIGHBOR_OFFSETS = sorted(
    (offset for offset in itertools.product((-1, 0, 1), repeat=3) if any(offset)),
    key=lambda offset: sum(c * c for c in offset),
)


def colored_status(message, color):
    return f"{color}{message}{ANSI_RESET}"


def precompute_elastica_spaces(L=1.0, N=100):
    """
    Build the beam mesh and function spaces, which are independent of the
    parameters (d, phi1, phi2) and reused for every solve.

    Returns
    -------
    tuple (V, R, beam_x, beam_order)
        V: function space for theta (Lagrange, degree 1)
        R: real space for the force multipliers
        beam_x: sorted dof coordinates along the beam
        beam_order: permutation taking theta dof values to beam_x order
    """
    domain = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, L])
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1))
    R = fem.functionspace(domain, basix.ufl.real_element(domain.basix_cell()))
    dof_x = V.tabulate_dof_coordinates()[:, 0]
    beam_order = np.argsort(dof_x)
    return V, R, dof_x[beam_order], beam_order


def end_moments(theta_sorted, beam_x):
    """Return theta'(0) and theta'(L); exact for P1 theta."""
    slopes = np.diff(theta_sorted) / np.diff(beam_x)
    return float(slopes[0]), float(slopes[-1])


def modal_assurance_criterion(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return the normalized squared dot product between two theta fields."""
    norm_a = np.linalg.norm(values_a)
    norm_b = np.linalg.norm(values_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(values_a, values_b) ** 2 / (norm_a ** 2 * norm_b ** 2))


def choose_start_node(coords: np.ndarray) -> int:
    """Choose the node closest to the angular origin, preferring smaller d on ties."""
    indices = np.arange(coords.shape[0])
    angular_distance = np.hypot(coords[:, 1], coords[:, 2])
    order = np.lexsort((
        indices,
        np.abs(coords[:, 2]),
        np.abs(coords[:, 1]),
        coords[:, 0],
        angular_distance,
    ))
    return int(order[0])


def resolve_cold_start_node(coords: np.ndarray, config: dict) -> Optional[int]:
    """Return the user-selected cold-start node, if provided in the config."""
    cold_start = config.get("cold_start")
    if cold_start is None:
        return None

    if isinstance(cold_start, dict):
        try:
            target = np.array([
                float(cold_start["d"]),
                float(cold_start["phi1"]),
                float(cold_start["phi2"]),
            ], dtype=float)
        except KeyError as exc:
            raise ValueError(
                "cold_start must include d, phi1, and phi2 keys"
            ) from exc
    elif isinstance(cold_start, (list, tuple)) and len(cold_start) == 3:
        target = np.asarray(cold_start, dtype=float)
    else:
        raise ValueError(
            "cold_start must be a dict with d, phi1, phi2 or a 3-element sequence"
        )

    distances = np.linalg.norm(coords - target[None, :], axis=1)
    return int(np.argmin(distances))


def nearest_solved_neighbor(node_idx, shape, solved):
    """Return the closest solved grid neighbour of node_idx, or None."""
    plane = shape[1] * shape[2]
    i, rest = divmod(node_idx, plane)
    j, k = divmod(rest, shape[2])
    for di, dj, dk in NEIGHBOR_OFFSETS:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
            neighbor = ni * plane + nj * shape[2] + nk
            if solved[neighbor]:
                return neighbor
    return None


def save_results_npz(output_npz, coords, tets, fx_arr, fy_arr, moment_arr,
                     energy_arr, solved_arr, theta_arr, beam_x):
    np.savez_compressed(
        output_npz,
        coords=coords,
        tets=tets,
        fx=fx_arr,
        fy=fy_arr,
        moment=moment_arr,
        energy=energy_arr,
        solved=solved_arr,
        theta=(theta_arr if theta_arr is not None else np.empty((0, 0))),
        beam_x=beam_x,
    )


def main():
    if MPI.COMM_WORLD.size > 1:
        raise SystemExit("main.py runs serially; do not launch it with mpirun")

    # Step 0: Read config
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Step 1: Generate or load the solution space mesh
    npz_file = config.get("output_npz", "solution_space.npz")
    settings = mesh_settings(config)
    if mesh_matches(npz_file, settings):
        print(f"Loading existing solution space mesh: {npz_file}")
    else:
        print("Generating solution space mesh...")
        build_rectilinear_mesh(**settings, out_npz=npz_file)

    with np.load(npz_file) as data:
        coords = data["coords"]          # (N, 3) array of (d, phi1, phi2)
        tets = data["tets"]              # (M, 4) array of tetrahedron connectivity
        shape = tuple(int(n) for n in data["shape"])
        mirror_indices = data["mirror_indices"].astype(np.int64)
        solve_mask = data["solve_mask"].astype(bool)
    num_nodes = coords.shape[0]

    # Step 2: Beam mesh, function spaces and the reusable nonlinear solver
    print("Precomputing elastica beam mesh and function spaces...")
    beam_elements = int(config.get("beam_elements", 100))
    V, R, beam_x, beam_order = precompute_elastica_spaces(L=1.0, N=beam_elements)
    num_beam_dofs = len(beam_x)
    elastica_solver = ElasticaSolver(
        V,
        R,
        petsc_options={
            "snes_max_it": 500,
            "snes_rtol": float(config.get("snes_rtol", 1e-8)),
            "snes_atol": float(config.get("snes_atol", 1e-10)),
        },
    )

    # Step 3: Visit solve nodes in order of grid distance from the start node,
    # so every node after the first has a solved neighbour to warm-start from.
    cold_start_idx = resolve_cold_start_node(coords, config)
    start_idx = (cold_start_idx if cold_start_idx is not None
                 else choose_start_node(coords))
    if not solve_mask[start_idx]:
        start_idx = int(mirror_indices[start_idx])
    grid_index = np.column_stack(np.unravel_index(np.arange(num_nodes), shape))
    order = np.flatnonzero(solve_mask)
    order = order[np.argsort(
        np.sum((grid_index[order] - grid_index[start_idx]) ** 2, axis=1),
        kind="stable",
    )]
    del grid_index
    print(f"Traversal order has {len(order)} solve nodes, starting at node "
          f"{start_idx}, coords={coords[start_idx].tolist()}")

    # Step 4: Result arrays (in the order of the coords array). theta is kept
    # for every node because neighbours warm-start from it.
    fx_arr = np.full(num_nodes, np.nan)
    fy_arr = np.full(num_nodes, np.nan)
    moment_arr = np.full(num_nodes, np.nan)
    energy_arr = np.full(num_nodes, np.nan)
    solved_arr = np.zeros(num_nodes, dtype=bool)
    theta_arr = np.full((num_nodes, num_beam_dofs), np.nan)

    mac_threshold = float(config.get("mac_threshold", 0.9))
    checkpoint_interval = int(config.get("visualization_interval", 0))
    progress_interval = max(1, int(config.get("progress_interval", 10000)))
    verbose_solves = bool(config.get("verbose_solves", False))
    save_theta = bool(config.get("save_theta", False))
    live_results_npz = config.get("live_results_npz", "solution_space_live.npz")
    live_output_vtu = config.get("live_output_vtu", "solution_space_live.vtu")

    if os.path.exists(live_results_npz):
        print(f"Loading checkpoint: {live_results_npz}")
        with np.load(live_results_npz) as checkpoint:
            checkpoint_coords = checkpoint["coords"]
            if checkpoint_coords.shape != coords.shape or not np.allclose(
                    checkpoint_coords, coords
            ):
                raise ValueError(
                    f"Checkpoint {live_results_npz} does not match the solution mesh"
                )
            if checkpoint["theta"].shape != theta_arr.shape:
                raise ValueError(
                    f"Checkpoint {live_results_npz} has no theta data for "
                    f"{beam_elements} beam elements; delete it to start over"
                )
            fx_arr[:] = checkpoint["fx"]
            fy_arr[:] = checkpoint["fy"]
            moment_arr[:] = checkpoint["moment"]
            energy_arr[:] = checkpoint["energy"]
            solved_arr[:] = checkpoint["solved"].astype(bool)
            theta_arr[:] = checkpoint["theta"]
        print(f"Resuming with {np.count_nonzero(solved_arr)}/{num_nodes} "
              "points already solved")

    def output_fields(include_theta):
        return {"fx": fx_arr, "fy": fy_arr, "moment": moment_arr,
                "energy": energy_arr, "solved": solved_arr.astype(float),
                **({"theta": theta_arr} if include_theta else {})}

    def save_checkpoint():
        save_results_npz(
            live_results_npz, coords, tets, fx_arr, fy_arr, moment_arr,
            energy_arr, solved_arr, theta_arr, beam_x,
        )
        write_vtu(live_output_vtu, coords, tets, output_fields(save_theta),
                  solved=solved_arr)

    solve_time_total = 0.0
    solve_attempts = 0
    failed_nodes = 0

    def try_solve(node_idx, label, attempt):
        nonlocal solve_time_total, solve_attempts
        d_val, phi1_val, phi2_val = coords[node_idx]
        solve_start = time.perf_counter()
        try:
            return attempt(d_val, phi1_val, phi2_val)
        except RuntimeError as error:
            print(colored_status(
                f"  Failed solve at node {node_idx} "
                f"(d={d_val:.6g}, phi1={phi1_val:.6g}, "
                f"phi2={phi2_val:.6g}) using {label}: {error}", ANSI_RED
            ))
            return None
        finally:
            solve_time_total += time.perf_counter() - solve_start
            solve_attempts += 1

    def store(node_idx, fx, fy, moment, energy, theta_sorted):
        fx_arr[node_idx] = fx
        fy_arr[node_idx] = fy
        moment_arr[node_idx] = moment
        energy_arr[node_idx] = energy
        theta_arr[node_idx] = theta_sorted
        solved_arr[node_idx] = True

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

    # Step 5: Iterate over the traversal order
    solved_count = int(np.count_nonzero(solved_arr))
    last_checkpoint_count = solved_count
    print(f"Solving elastica problem for {num_nodes} parameter sets...")
    for idx, node_idx in enumerate(
            node_idx for node_idx in order if not solved_arr[node_idx]
    ):
        if stop_requested:
            break
        if idx % progress_interval == 0:
            average_solve_time = (solve_time_total / solve_attempts
                                  if solve_attempts else 0.0)
            print(f"  Progress: {solved_count}/{num_nodes} solved, "
                  f"average solver time {average_solve_time:.4f} s "
                  f"({solve_attempts} attempts)")

        d_val, phi1_val, phi2_val = coords[node_idx]
        neighbor = nearest_solved_neighbor(node_idx, shape, solved_arr)
        solution = None
        if neighbor is not None:
            # Warm-start theta and the force multipliers from the neighbour.
            theta0 = np.empty(num_beam_dofs)
            theta0[beam_order] = theta_arr[neighbor]
            label = f"neighbor {neighbor}"
            solution = try_solve(
                node_idx, label,
                lambda d, phi1, phi2: elastica_solver.solve(
                    d, phi1, phi2, theta0, fx_arr[neighbor], fy_arr[neighbor]),
            )
            if solution is not None:
                mac = modal_assurance_criterion(
                    solution.theta[beam_order], theta_arr[neighbor])
                if mac < mac_threshold:
                    print(colored_status(
                        f"  Rejected node {node_idx} from neighbor "
                        f"{neighbor}: MAC={mac:.4f} < {mac_threshold:.4f}",
                        ANSI_YELLOW,
                    ))
                    solution = None
        if solution is None:
            # Cold start: continue from the exact first clamped mode.
            label = "cold start"
            solution = try_solve(node_idx, label, elastica_solver.cold_solve)
            if solution is None:
                failed_nodes += 1
                continue

        theta_sorted = solution.theta[beam_order]
        left_moment, right_moment = end_moments(theta_sorted, beam_x)
        store(node_idx, solution.fx, solution.fy, left_moment, solution.energy,
              theta_sorted)
        solved_count += 1
        # Mirror: theta~(s) = -theta(L - s) solves (d, -phi2, -phi1) with
        # fx~ = fx, fy~ = -fy and theta~'(0) = theta'(L).
        mirror_idx = int(mirror_indices[node_idx])
        if mirror_idx != node_idx:
            store(mirror_idx, solution.fx, -solution.fy, right_moment,
                  solution.energy, -theta_sorted[::-1])
            solved_count += 1
        if verbose_solves:
            print(colored_status(
                f"  Solved node {node_idx} "
                f"(d={d_val:.6g}, phi1={phi1_val:.6g}, "
                f"phi2={phi2_val:.6g}) using {label}", ANSI_GREEN
            ))

        if (checkpoint_interval > 0
                and solved_count - last_checkpoint_count >= checkpoint_interval):
            save_checkpoint()
            last_checkpoint_count = solved_count

    print(f"Solved {solved_count}/{num_nodes} points "
          f"({100 * solved_count / num_nodes:.1f}%), "
          f"{failed_nodes} failed this run")
    save_checkpoint()
    if stop_requested:
        print(f"Checkpoint saved to {live_results_npz}; rerun to resume")
        return

    # Step 6: Save NPZ backup and Paraview-readable VTU output.
    output_npz = config.get("results_npz", "solution_space_results.npz")
    save_results_npz(
        output_npz, coords, tets, fx_arr, fy_arr, moment_arr, energy_arr,
        solved_arr, theta_arr if save_theta else None, beam_x,
    )
    print(f"Results saved to {output_npz}")
    output_vtu = config.get("output_vtu", "solution_space_results.vtu")
    write_vtu(output_vtu, coords, tets, output_fields(save_theta))
    print(f"VTU results saved to {output_vtu}")


if __name__ == "__main__":
    main()
