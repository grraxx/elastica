"""
main.py
Main script to run the elastica solver over the solution space.
Uses tetrahedral mesh traversal for continuation.
"""

import base64
import json
import numpy as np
import os
import time
from typing import Optional, List
import itertools
from mesh_generator import generate_solution_space_mesh
from solver import ElasticaSolution, ElasticaSolver
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem.petsc import NonlinearProblem
import basix.ufl
import gmsh  # for writing node data to .msh file


ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"


def colored_status(message, color):
    return f"{color}{message}{ANSI_RESET}"


def precompute_elastica_spaces(L=1.0, N=100):
    """
    Precompute the mesh and function spaces for the elastica problem.
    These are independent of the parameters (d, phi1, phi2) and can be reused.

    Returns
    -------
    tuple (domain, V, R, x_coords, dx)
        domain: the mesh object
        V: function space for theta (Lagrange, degree 1)
        R: function space for the Lagrange multipliers (real)
        x_coords: coordinates of the mesh nodes (sorted)
        dx: the measure for integration
    """
    domain = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, L])
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1))
    R = fem.functionspace(domain, basix.ufl.real_element(domain.basix_cell()))
    # Get coordinates and sort them (for computing derivative at left end)
    x_g = domain.geometry.x[:, 0]
    # Sort coordinates and get the permutation
    order = np.argsort(x_g)
    x_sorted = x_g[order]
    return domain, V, R, x_sorted, ufl.dx


def compute_moment_and_energy(theta_sol: fem.Function, x_sorted: np.ndarray, dx) -> tuple[float, float]:
    """
    Compute the bending moment at the left end (s=0) and the bending strain energy.

    Parameters
    ----------
    theta_sol : fem.Function
        The solved theta function.
    x_sorted : np.ndarray
        Sorted coordinates of the elastica mesh (used to compute derivative at left end).
    dx : ufl.Measure
        The measure for integration.

    Returns
    -------
    moment : float
        Bending moment at s=0, approximated by theta'(0) using first two points.
    energy : float
        Bending strain energy = 0.5 * int((theta')^2) dx.
    """
    # Compute derivative at left end using first two points
    theta_vals = theta_sol.x.array
    # We need the values at the sorted coordinates to match x_sorted
    domain = theta_sol.function_space.mesh
    x_g = domain.geometry.x[:, 0]
    order = np.argsort(x_g)
    theta_sorted = theta_vals[order]
    # Approximate derivative at left end (x=0) using first two points
    if len(x_sorted) >= 2:
        moment = (theta_sorted[1] - theta_sorted[0]) / (x_sorted[1] - x_sorted[0])
    else:
        moment = 0.0  # fallback

    # Compute energy: 0.5 * int((theta')^2) dx
    theta_ufl = theta_sol
    energy_form = 0.5 * ufl.inner(ufl.grad(theta_ufl), ufl.grad(theta_ufl)) * dx
    energy = fem.assemble_scalar(fem.form(energy_form))
    return float(moment), float(energy)


def modal_assurance_criterion(theta_a: fem.Function, theta_b: fem.Function) -> float:
    """Return the normalized squared dot product between two theta fields."""
    values_a = np.asarray(theta_a.x.array if hasattr(theta_a, "x") else theta_a)
    values_b = np.asarray(theta_b.x.array if hasattr(theta_b, "x") else theta_b)
    norm_a = np.linalg.norm(values_a)
    norm_b = np.linalg.norm(values_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.abs(np.dot(values_a, values_b)) ** 2 / (norm_a ** 2 * norm_b ** 2))


def choose_start_node(coords: np.ndarray) -> int:
    """
    Choose the node closest to the angular origin, preferring smaller d on ties.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (num_nodes, 3) with columns (d, phi1, phi2).

    Returns
    -------
    start_idx : int
        Index of the chosen start node.
    """
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
    elif isinstance(cold_start, (list, tuple, np.ndarray)) and len(cold_start) == 3:
        target = np.asarray(cold_start, dtype=float)
    else:
        raise ValueError(
            "cold_start must be a dict with d, phi1, phi2 or a 3-element sequence"
        )

    distances = np.linalg.norm(coords - target[None, :], axis=1)
    return int(np.argmin(distances))


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


def save_results_vtu(output_vtu, coords, tets, fields, solved_only=False):
    """Write tetrahedral point data in binary VTK XML format for Paraview."""

    def encode_data(values, dtype):
        values = np.asarray(values, dtype=dtype)
        payload = np.asarray([values.nbytes], dtype="<u4").tobytes()
        payload += values.tobytes(order="C")
        return base64.b64encode(payload).decode("ascii")

    coords = np.asarray(coords, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    if solved_only:
        solved = np.asarray(fields["solved"], dtype=bool)
        cell_mask = np.all(solved[tets], axis=1)
        selected_cells = tets[cell_mask]
        point_mask = solved.copy()
        point_indices = np.flatnonzero(point_mask)
        index_map = np.full(len(coords), -1, dtype=np.int64)
        index_map[point_indices] = np.arange(len(point_indices))
        coords = coords[point_indices]
        tets = index_map[selected_cells]
        fields = {
            name: np.asarray(values)[point_indices]
            for name, values in fields.items()
        }
    connectivity = encode_data(tets.ravel(), "<i8")
    offsets = encode_data(4 * np.arange(1, len(tets) + 1, dtype="<i8"), "<i8")
    types = encode_data(np.full(len(tets), 10, dtype="u1"), "u1")
    points = encode_data(coords, "<f8")
    point_data_arrays = []
    for name, values in fields.items():
        values = np.asarray(values)
        if values.ndim > 2:
            raise ValueError(f"VTU field {name} must be scalar or vector-valued")
        components = values.shape[1] if values.ndim == 2 else 1
        component_attribute = (f' NumberOfComponents="{components}"'
                               if components > 1 else "")
        serialized_values = encode_data(
            np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
            "<f8",
        )
        point_data_arrays.append(
            f'        <DataArray type="Float64" Name="{name}"'
            f'{component_attribute} format="binary">{serialized_values}</DataArray>'
        )
    point_data = "\n".join(point_data_arrays)
    xml = f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid><Piece NumberOfPoints="{len(coords)}" NumberOfCells="{len(tets)}">
      <PointData>
{point_data}
      </PointData>
            <Points><DataArray type="Float64" NumberOfComponents="3" format="binary">{points}</DataArray></Points>
      <Cells>
                <DataArray type="Int64" Name="connectivity" format="binary">{connectivity}</DataArray>
                <DataArray type="Int64" Name="offsets" format="binary">{offsets}</DataArray>
                <DataArray type="UInt8" Name="types" format="binary">{types}</DataArray>
      </Cells>
  </Piece></UnstructuredGrid>
</VTKFile>
'''
    temporary_vtu = f"{output_vtu}.tmp"
    with open(temporary_vtu, "w", encoding="ascii") as file:
        file.write(xml)
    os.replace(temporary_vtu, output_vtu)



def main():
    # Step 0: Read config
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Preserve the JIT cache between runs; rebuilding it is expensive.
    cache_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "fenics",
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Generate or load the solution space mesh
    npz_file = config.get("output_npz", "solution_space.npz")
    msh_file = config.get("output_mesh", "solution_space.msh")
    mesh_has_scaling = False
    symmetry_requested = bool(config.get("solve_phi_symmetry", False))
    if os.path.exists(npz_file):
        with np.load(npz_file) as mesh_data:
            mesh_has_scaling = (
                "mesh_coords" in mesh_data.files
                and "parameters_scaled" in mesh_data.files
                and bool(mesh_data["parameters_scaled"])
                and "grid_steps" in mesh_data.files
                and np.allclose(
                    mesh_data["grid_steps"],
                    np.array([
                        float(config.get("grid_steps", {}).get("d", 0.01)),
                        float(config.get("grid_steps", {}).get("phi1", 0.02)),
                        float(config.get("grid_steps", {}).get("phi2", 0.02)),
                    ]),
                )
                and (not symmetry_requested or (
                    "symmetric_mesh" in mesh_data.files
                    and bool(mesh_data["symmetric_mesh"])
                ))
            )
    if not (os.path.exists(npz_file) and os.path.exists(msh_file)
            and mesh_has_scaling):
        print(f"Generating solution space mesh...")
        npz_file = generate_solution_space_mesh(config_path)
        print(f"Generated mesh: {npz_file}")
    else:
        print(f"Loading existing solution space mesh: {npz_file}")

    # Load the mesh data
    data = np.load(npz_file)
    coords = data["coords"]          # (N, 3) array of (d, phi1, phi2)
    tets = data["tets"]              # (M, 4) array of tetrahedron connectivity
    num_nodes = coords.shape[0]
    if symmetry_requested:
        if "mirror_indices" not in data.files or "solve_mask" not in data.files:
            raise ValueError("Symmetric mesh is missing mirror metadata")
        mirror_indices = data["mirror_indices"].astype(np.int64)
        solve_mask = data["solve_mask"].astype(bool)
        if (mirror_indices.shape != (num_nodes,)
                or solve_mask.shape != (num_nodes,)):
            raise ValueError("Invalid symmetric mesh metadata")
    else:
        mirror_indices = np.arange(num_nodes, dtype=np.int64)
        solve_mask = np.ones(num_nodes, dtype=bool)

    # Step 2: Precompute elastica spaces (mesh and function spaces)
    print("Precomputing elastica beam mesh and function spaces...")
    beam_elements = int(config.get("beam_elements", 100))
    domain, V, R, x_sorted, dx = precompute_elastica_spaces(
        L=1.0, N=beam_elements
    )
    elastica_solver = ElasticaSolver(
        V,
        R,
        petsc_options={
            "snes_max_it": 500,
            "snes_rtol": float(config.get("snes_rtol", 1e-8)),
            "snes_atol": float(config.get("snes_atol", 1e-10)),
        },
    )

    # Step 3: Traverse representative nodes without materializing adjacency.
    cold_start_idx = resolve_cold_start_node(coords, config)
    start_idx = (cold_start_idx if cold_start_idx is not None
                 else choose_start_node(coords))
    order = np.flatnonzero(solve_mask)
    order = order[np.argsort(
        np.sum((coords[order] - coords[start_idx]) ** 2, axis=1)
    )].tolist()
    print(f"Coordinate traversal order has {len(order)} solve nodes")
    if cold_start_idx is not None:
        print(f"Cold start node prescribed in config: index {cold_start_idx}, "
              f"coords={coords[cold_start_idx].tolist()}")

    # Step 4: Initialize result arrays (in the order of coords array)
    fx_arr = np.full(num_nodes, np.nan)
    fy_arr = np.full(num_nodes, np.nan)
    moment_arr = np.full(num_nodes, np.nan)
    energy_arr = np.full(num_nodes, np.nan)
    solved_arr = np.zeros(num_nodes, dtype=bool)  # False means not solved yet
    theta_arr = None

    # Step 5: Iterate over the traversal order
    mac_threshold = config.get("mac_threshold", 0.9)
    visualization_interval = int(config.get("visualization_interval", 0))
    progress_interval = max(1, int(config.get("progress_interval", 10000)))
    verbose_solves = bool(config.get("verbose_solves", False))
    save_theta = bool(config.get("save_theta", False))
    live_results_npz = config.get("live_results_npz", "solution_space_live.npz")
    live_output_vtu = config.get("live_output_vtu", "solution_space_live.vtu")
    theta_solutions = [None] * num_nodes
    theta_arr = (np.full((num_nodes, len(x_sorted)), np.nan)
                 if save_theta else None)
    beam_order = np.argsort(domain.geometry.x[:, 0])
    solved_count = 0
    solve_time_total = 0.0
    solve_attempts = 0
    previous_solved_node = None

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
            checkpoint_theta = checkpoint["theta"]
            if checkpoint_theta.shape != (num_nodes, len(x_sorted)):
                raise ValueError(
                    f"Checkpoint {live_results_npz} has no complete theta data; "
                    "set save_theta to true before creating checkpoints"
                )
            if theta_arr is None:
                raise ValueError(
                    "Resuming requires save_theta to be true in config.json"
                )
            checkpoint_theta_arr = theta_arr
            fx_arr[:] = checkpoint["fx"]
            fy_arr[:] = checkpoint["fy"]
            moment_arr[:] = checkpoint["moment"]
            energy_arr[:] = checkpoint["energy"]
            solved_arr[:] = checkpoint["solved"].astype(bool)
            if theta_arr is not None:
                theta_arr[:] = checkpoint_theta

        for node_idx in np.flatnonzero(solved_arr):
            theta_solution = fem.Function(V)
            theta_values = np.empty_like(checkpoint_theta_arr[node_idx])
            theta_values[beam_order] = checkpoint_theta_arr[node_idx]
            theta_solution.x.array[:] = theta_values
            theta_solution.x.scatter_forward()
            theta_solutions[node_idx] = theta_solution
        solved_count = int(np.count_nonzero(solved_arr))
        print(f"Resuming with {solved_count}/{num_nodes} points already solved")

    print(f"Solving elastica problem for {num_nodes} parameter sets in tetrahedral traversal order...")
    for idx, node_idx in enumerate(
            node_idx for node_idx in order if not solved_arr[node_idx]
    ):
            if idx % progress_interval == 0:
                average_solve_time = (
                    solve_time_total / solve_attempts
                    if solve_attempts else 0.0
                )
                print(f"  Progress: {solved_count}/{num_nodes} solved")
                print(f"  Average solver time: {average_solve_time:.3f} s "
                      f"({solve_attempts} attempts)")

            d_val, phi1_val, phi2_val = coords[node_idx]
            warm_start_nodes = ([previous_solved_node]
                                if previous_solved_node is not None else [None])

            accepted_solution = None
            accepted_warm_start = None
            for warm_start_node in warm_start_nodes:
                theta0 = (None if warm_start_node is None
                          else theta_solutions[warm_start_node])
                # Warm-start the force multipliers from the same neighbor
                # theta came from, instead of always starting them at 0.
                # They're part of the equilibrium state too, so a cold
                # multiplier guess next to a warm theta guess wastes half
                # of the continuation's benefit and can still trip Newton
                # up on well-loaded points.
                if warm_start_node is None:
                    fx0, fy0 = 0.0, 0.0
                else:
                    fx0 = fx_arr[warm_start_node]
                    fy0 = fy_arr[warm_start_node]
                    if not np.isfinite(fx0):
                        fx0 = 0.0
                    if not np.isfinite(fy0):
                        fy0 = 0.0
                solve_start = time.perf_counter()
                try:
                    candidate = elastica_solver.solve(
                        d_val, phi1_val, phi2_val, theta0=theta0,
                        fx0=fx0, fy0=fy0,
                    )
                    solve_time_total += time.perf_counter() - solve_start
                    solve_attempts += 1
                except RuntimeError as error:
                    solve_time_total += time.perf_counter() - solve_start
                    solve_attempts += 1
                    warm_start_label = ("default initial field" if warm_start_node is None
                                        else f"neighbor {warm_start_node}")
                    print(colored_status(
                        f"  Failed solve at node {node_idx} "
                        f"(d={d_val:.6g}, phi1={phi1_val:.6g}, "
                        f"phi2={phi2_val:.6g}) using {warm_start_label}: "
                        f"{error}", ANSI_RED
                    ))
                    continue

                if warm_start_node is not None:
                    mac = modal_assurance_criterion(
                        candidate.theta, theta_solutions[warm_start_node])
                    if mac < mac_threshold:
                        print(colored_status(
                            f"  Rejected node {node_idx} from neighbor "
                            f"{warm_start_node}: MAC={mac:.4f} "
                            f"< {mac_threshold:.4f}", ANSI_YELLOW
                        ))
                        continue
                accepted_solution = candidate
                accepted_warm_start = warm_start_node
                break

            if accepted_solution is None:
                solve_start = time.perf_counter()
                try:
                    accepted_solution = elastica_solver.solve(
                        d_val, phi1_val, phi2_val, theta0=None,
                        fx0=0.0, fy0=0.0,
                    )
                    solve_time_total += time.perf_counter() - solve_start
                    solve_attempts += 1
                    accepted_warm_start = None
                except RuntimeError as error:
                    solve_time_total += time.perf_counter() - solve_start
                    solve_attempts += 1
                    print(colored_status(
                        f"  Failed cold retry at node {node_idx}: {error}",
                        ANSI_RED,
                    ))
                    continue

            fx_arr[node_idx] = accepted_solution.fx
            fy_arr[node_idx] = accepted_solution.fy
            moment, energy = compute_moment_and_energy(accepted_solution.theta,
                                                       x_sorted, dx)
            moment_arr[node_idx] = moment
            energy_arr[node_idx] = energy
            solved_arr[node_idx] = True
            theta_solutions[node_idx] = accepted_solution.theta
            theta_values = np.asarray(accepted_solution.theta.x.array)
            theta_sorted = theta_values[beam_order]
            right_moment = (
                (theta_sorted[-1] - theta_sorted[-2])
                / (x_sorted[-1] - x_sorted[-2])
                if len(x_sorted) >= 2 else 0.0
            )
            if theta_arr is not None:
                theta_arr[node_idx, :] = theta_sorted
            mirror_idx = int(mirror_indices[node_idx])
            if mirror_idx != node_idx:
                fx_arr[mirror_idx] = accepted_solution.fx
                fy_arr[mirror_idx] = -accepted_solution.fy
                moment_arr[mirror_idx] = right_moment
                energy_arr[mirror_idx] = energy
                solved_arr[mirror_idx] = True
                if theta_arr is not None:
                    theta_arr[mirror_idx, :] = -theta_arr[node_idx, ::-1]
            solved_count += 1 if mirror_idx == node_idx else 2
            previous_solved_node = node_idx
            if accepted_warm_start is None:
                warm_start_label = "default initial field"
            else:
                warm_start_label = f"neighbor {accepted_warm_start}"
            if verbose_solves:
                print(colored_status(
                    f"  Solved node {node_idx} "
                    f"(d={d_val:.6g}, phi1={phi1_val:.6g}, "
                    f"phi2={phi2_val:.6g}) using {warm_start_label}", ANSI_GREEN
                ))

            if (visualization_interval > 0
                    and solved_count % visualization_interval == 0):
                save_results_npz(
                    live_results_npz, coords, tets, fx_arr, fy_arr, moment_arr,
                    energy_arr, solved_arr, theta_arr, x_sorted,
                )
                save_results_vtu(
                    live_output_vtu, coords, tets,
                    {"fx": fx_arr, "fy": fy_arr, "moment": moment_arr,
                     "energy": energy_arr,
                     "solved": solved_arr.astype(float),
                     **({"theta": theta_arr} if theta_arr is not None else {})},
                    solved_only=True,
                )

    print(f"Solved {solved_count}/{num_nodes} points ({100*solved_count/num_nodes:.1f}%)")

    # Step 6: Save NPZ backup and Paraview-readable VTU output.
    output_npz = config.get("results_npz", "solution_space_results.npz")
    save_results_npz(
        output_npz, coords, tets, fx_arr, fy_arr, moment_arr, energy_arr,
        solved_arr, theta_arr, x_sorted,
    )
    print(f"Results saved to {output_npz}")
    output_vtu = config.get("output_vtu", "solution_space_results.vtu")
    save_results_vtu(
        output_vtu, coords, tets,
        {"fx": fx_arr, "fy": fy_arr, "moment": moment_arr,
            "energy": energy_arr, "solved": solved_arr.astype(float),
            **({"theta": theta_arr} if theta_arr is not None else {})},
    )
    print(f"VTU results saved to {output_vtu}")

if __name__ == "__main__":
    main()