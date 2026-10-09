"""
main.py
Main script to run the elastica solver over the solution space.
Uses tetrahedral mesh traversal for continuation.
"""

import json
import numpy as np
import os
import shutil
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


def build_adjacency(tets: np.ndarray, num_nodes: int) -> List[List[int]]:
    """
    Build an adjacency list for the nodes from the tetrahedron connectivity.

    Parameters
    ----------
    tets : np.ndarray
        Array of shape (num_tets, 4) containing node indices for each tetrahedron.
    num_nodes : int
        Total number of nodes.

    Returns
    -------
    adj : List of lists, where adj[i] is a list of neighbor node indices.
    """
    adj = [[] for _ in range(num_nodes)]
    for tet in tets:
        # For each pair of nodes in the tet, add each other as neighbors
        for i in range(4):
            for j in range(i+1, 4):
                a = tet[i]
                b = tet[j]
                adj[a].append(b)
                adj[b].append(a)
    # Remove duplicates (optional, but we can leave duplicates for simplicity)
    for i in range(num_nodes):
        adj[i] = list(set(adj[i]))
    return adj


def traverse_mesh(start_idx: int, adj: List[List[int]]) -> List[int]:
    """
    Perform a depth-first traversal of the mesh starting from start_idx.
    Returns a list of node indices in the order of traversal.
    Uses an iterative stack to avoid recursion depth issues.

    Parameters
    ----------
    start_idx : int
        Index of the starting node.
    adj : List[List[int]]
        Adjacency list.

    Returns
    -------
    order : List[int]
        List of node indices in DFS order.
    """
    num_nodes = len(adj)
    visited = [False] * num_nodes
    order = []
    stack = [start_idx]

    while stack:
        u = stack.pop()
        if not visited[u]:
            visited[u] = True
            order.append(u)
            # Add neighbors in reverse order so that we pop them in the original order
            # (since we use a stack, LIFO). To get the same order as recursive DFS,
            # we can push neighbors in reverse order.
            for v in reversed(adj[u]):
                if not visited[v]:
                    stack.append(v)
    return order


def modal_assurance_criterion(theta_a: fem.Function, theta_b: fem.Function) -> float:
    """Return the normalized squared dot product between two theta fields."""
    values_a = np.asarray(theta_a.x.array)
    values_b = np.asarray(theta_b.x.array)
    norm_a = np.linalg.norm(values_a)
    norm_b = np.linalg.norm(values_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.abs(np.dot(values_a, values_b)) ** 2 / (norm_a ** 2 * norm_b ** 2))


def choose_start_node(coords: np.ndarray) -> int:
    """
    Choose the node with the smallest d, then smallest |phi1|, then |phi2|.

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
    order = np.lexsort((
        indices,
        np.abs(coords[:, 2]),
        np.abs(coords[:, 1]),
        coords[:, 0],
    ))
    return int(order[0])


def save_results_npz(output_npz, coords, ijk, tets, grid_shape, d_vals,
                     p1_vals, p2_vals, fx_arr, fy_arr, moment_arr,
                     energy_arr, solved_arr):
    np.savez_compressed(
        output_npz,
        coords=coords,
        ijk=ijk,
        tets=tets,
        grid_shape=grid_shape,
        d_vals=d_vals,
        p1_vals=p1_vals,
        p2_vals=p2_vals,
        fx=fx_arr,
        fy=fy_arr,
        moment=moment_arr,
        energy=energy_arr,
        solved=solved_arr,
    )



def main():
    # Step 0: Read config
    config_path = "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Clear JIT cache to avoid compilation issues
    cache_dir = "/home/goraksh/.cache/fenics"
    if os.path.exists(cache_dir):
        print(f"Clearing JIT cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Generate or load the solution space mesh
    npz_file = config.get("output_npz", "solution_space.npz")
    msh_file = config.get("output_mesh", "solution_space.msh")
    if not (os.path.exists(npz_file) and os.path.exists(msh_file)):
        print(f"Generating solution space mesh...")
        npz_file = generate_solution_space_mesh(config_path)
        print(f"Generated mesh: {npz_file}")
    else:
        print(f"Loading existing solution space mesh: {npz_file}")

    # Load the mesh data
    data = np.load(npz_file)
    coords = data["coords"]          # (N, 3) array of (d, phi1, phi2)
    ijk = data["ijk"]                # (N, 3) array of grid indices for each node
    tets = data["tets"]              # (M, 4) array of tetrahedron connectivity
    grid_shape = data["grid_shape"]  # (nd, n1, n2)
    d_vals = data["d_vals"]
    p1_vals = data["p1_vals"]
    p2_vals = data["p2_vals"]
    nd, n1, n2 = grid_shape
    num_nodes = coords.shape[0]

    # Step 2: Precompute elastica spaces (mesh and function spaces)
    print("Precomputing elastica beam mesh and function spaces...")
    domain, V, R, x_sorted, dx = precompute_elastica_spaces(L=1.0, N=100)
    elastica_solver = ElasticaSolver(V, R)

    # Step 3: Build adjacency and choose traversal order
    print("Building adjacency from tetrahedral mesh...")
    adj = build_adjacency(tets, num_nodes)
    start_idx = choose_start_node(coords)
    cold_start_candidates = set(adj[start_idx]) | {start_idx}
    print(f"Starting traversal from node {start_idx} with coordinates {coords[start_idx]}")
    order = traverse_mesh(start_idx, adj)
    print(f"Traversal order has {len(order)} nodes")

    # Step 4: Initialize result arrays (in the order of coords array)
    fx_arr = np.full(num_nodes, np.nan)
    fy_arr = np.full(num_nodes, np.nan)
    moment_arr = np.full(num_nodes, np.nan)
    energy_arr = np.full(num_nodes, np.nan)
    solved_arr = np.zeros(num_nodes, dtype=bool)  # False means not solved yet

    # Step 5: Iterate over the traversal order
    mac_threshold = config.get("mac_threshold", 0.9)
    visualization_interval = int(config.get("visualization_interval", 0))
    live_results_npz = config.get("live_results_npz", "solution_space_live.npz")
    live_visualization = config.get(
        "live_visualization", "solution_space_live.png"
    )
    theta_solutions = [None] * num_nodes
    solved_count = 0
    print(f"Solving elastica problem for {num_nodes} parameter sets in tetrahedral traversal order...")
    pending = list(order)
    pass_number = 0
    while pending:
        pass_number += 1
        next_pending = []
        progress = False
        for idx, node_idx in enumerate(pending):
            if idx % 1000 == 0:
                print(f"  Progress: {solved_count}/{num_nodes} solved (pass {pass_number})")

            d_val, phi1_val, phi2_val = coords[node_idx]
            warm_start_nodes = [neighbor for neighbor in adj[node_idx]
                                if solved_arr[neighbor]]
            if not warm_start_nodes and solved_count == 0 and node_idx in cold_start_candidates:
                warm_start_nodes = [None]
            if not warm_start_nodes:
                next_pending.append(node_idx)
                continue

            accepted_solution = None
            accepted_warm_start = None
            for warm_start_node in warm_start_nodes:
                theta0 = (None if warm_start_node is None
                          else theta_solutions[warm_start_node])
                try:
                    candidate = elastica_solver.solve(
                        d_val, phi1_val, phi2_val, theta0=theta0
                    )
                except RuntimeError as error:
                    warm_start_label = ("linear ramp" if warm_start_node is None
                                        else f"neighbor {warm_start_node}")
                    print(f"  Failed solve at node {node_idx} "
                          f"(d={d_val:.6g}, phi1={phi1_val:.6g}, "
                          f"phi2={phi2_val:.6g}) using {warm_start_label}: "
                          f"{error}")
                    continue

                if warm_start_node is not None:
                    mac = modal_assurance_criterion(
                        candidate.theta, theta_solutions[warm_start_node])
                    if mac < mac_threshold:
                        print(f"  Rejected node {node_idx} from neighbor "
                              f"{warm_start_node}: MAC={mac:.4f} "
                              f"< {mac_threshold:.4f}")
                        continue
                accepted_solution = candidate
                accepted_warm_start = warm_start_node
                break

            if accepted_solution is None:
                next_pending.append(node_idx)
                continue

            fx_arr[node_idx] = accepted_solution.fx
            fy_arr[node_idx] = accepted_solution.fy
            moment, energy = compute_moment_and_energy(accepted_solution.theta,
                                                       x_sorted, dx)
            moment_arr[node_idx] = moment
            energy_arr[node_idx] = energy
            solved_arr[node_idx] = True
            theta_solutions[node_idx] = accepted_solution.theta
            solved_count += 1
            progress = True
            warm_start_label = ("cold start" if accepted_warm_start is None
                          else f"neighbor {accepted_warm_start}")
            print(f"  Solved node {node_idx} "
                f"(d={d_val:.6g}, phi1={phi1_val:.6g}, "
                f"phi2={phi2_val:.6g}) using {warm_start_label}")

            if (visualization_interval > 0
                    and solved_count % visualization_interval == 0):
                save_results_npz(
                    live_results_npz, coords, ijk, tets, grid_shape, d_vals,
                    p1_vals, p2_vals, fx_arr, fy_arr, moment_arr,
                    energy_arr, solved_arr,
                )
                try:
                    from visualization import plot_results
                    plot_results(live_results_npz,
                                 output_file=live_visualization)
                except ImportError:
                    print("Visualization module not found; skipping checkpoint plot.")
                except Exception as error:
                    print(f"Error during checkpoint visualization: {error}")

        pending = next_pending
        if not progress:
            print(f"Could not find an acceptable adjacent warm start for "
                  f"{len(pending)} node(s); leaving them unsolved.")
            for node_idx in pending:
                d_val, phi1_val, phi2_val = coords[node_idx]
                print(f"  Unsolved node {node_idx} "
                    f"(d={d_val:.6g}, phi1={phi1_val:.6g}, "
                    f"phi2={phi2_val:.6g})")
            break

    print(f"Solved {solved_count}/{num_nodes} points ({100*solved_count/num_nodes:.1f}%)")

    # Step 6: Save the results to a .npz file (for backup and visualization)
    output_npz = "solution_space_results.npz"
    save_results_npz(
        output_npz, coords, ijk, tets, grid_shape, d_vals, p1_vals, p2_vals,
        fx_arr, fy_arr, moment_arr, energy_arr, solved_arr,
    )
    print(f"Results saved to {output_npz}")

    # Step 8: Visualize the results (optional)
    try:
        from visualization import plot_results
        plot_results(output_npz)
        print("Visualization generated.")
    except ImportError:
        print("Visualization module not found. Skipping visualization.")
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()