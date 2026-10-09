"""Per-layer elastica solution-space study with half-diagonal symmetry."""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
from scipy.spatial import cKDTree
import basix.ufl

from mesh import load_or_create_uniform_space
from solver import ElasticaSolution, ElasticaSolver
from timing import NodeSolveTimer


def precompute_elastica_spaces(length=1.0, elements=100):
    """Create the beam space and sorted coordinates once for the study."""
    domain = mesh.create_interval(MPI.COMM_WORLD, elements, [0.0, length])
    theta_space = fem.functionspace(
        domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    )
    multiplier_space = fem.functionspace(
        domain, basix.ufl.real_element(domain.basix_cell())
    )
    order = np.argsort(domain.geometry.x[:, 0])
    beam_x = domain.geometry.x[order, 0].copy()
    return domain, theta_space, multiplier_space, beam_x


def moment_and_energy(theta, beam_x):
    """Compute the left moment and bending energy from ordered beam values."""
    spacing = np.diff(beam_x)
    slopes = np.diff(theta) / spacing
    moment = float(slopes[0])
    energy = float(0.5 * np.sum(0.5 * (slopes[:-1] ** 2 + slopes[1:] ** 2) * spacing[1:]))
    return moment, energy


def modal_assurance_criterion(theta_a, theta_b):
    """Return the normalized squared dot product between two theta fields."""
    norm_a = np.linalg.norm(theta_a)
    norm_b = np.linalg.norm(theta_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(abs(np.dot(theta_a, theta_b)) ** 2 / (norm_a ** 2 * norm_b ** 2))


def solve_with_adaptive_mac(
    solver,
    target_params,
    source_params,
    source_theta,
    source_fx,
    source_fy,
    mac_threshold,
    max_refinements,
    cold_start_mode=1,
):
    """Continue to one target with bounded MAC-guided subdivision."""
    if source_theta is None:
        return solver.solve(
            *target_params,
            theta0=None,
            fx0=0.0,
            fy0=0.0,
            cold_start_mode=cold_start_mode,
        ), 0
    current_params = np.asarray(source_params, dtype=float).copy()
    current_theta = np.asarray(source_theta, dtype=float).copy()
    current_fx = float(source_fx) if np.isfinite(source_fx) else 0.0
    current_fy = float(source_fy) if np.isfinite(source_fy) else 0.0
    total_refinements = 0
    delta = np.asarray(target_params, dtype=float) - current_params

    for parameter_axis in np.argsort(-np.abs(delta)):
        axis_delta = float(delta[parameter_axis])
        if np.isclose(axis_delta, 0.0):
            continue
        axis_start = (
            current_params.copy(), current_theta.copy(), current_fx, current_fy
        )
        subdivisions = 1
        refinements = 0
        completed = 0
        while completed < subdivisions:
            fraction = (completed + 1) / subdivisions
            trial_params = current_params.copy()
            trial_params[parameter_axis] = (
                axis_start[0][parameter_axis] + axis_delta * fraction
            )
            try:
                candidate = solver.solve(
                    *trial_params,
                    theta0=current_theta,
                    fx0=current_fx,
                    fy0=current_fy,
                )
            except RuntimeError:
                candidate = None
            mac = (
                0.0
                if candidate is None
                else modal_assurance_criterion(candidate.theta, current_theta)
            )
            if candidate is None or mac < mac_threshold:
                if candidate is not None and refinements >= max_refinements:
                    return ElasticaSolution(
                        candidate.theta, candidate.fx, candidate.fy, 0, 0
                    ), total_refinements
                if refinements >= max_refinements:
                    return None, total_refinements
                refinements += 1
                total_refinements += 1
                subdivisions *= 2
                completed = 0
                current_params, current_theta, current_fx, current_fy = (
                    axis_start[0].copy(), axis_start[1].copy(), axis_start[2], axis_start[3]
                )
                continue
            current_params = trial_params
            current_theta = candidate.theta
            current_fx = candidate.fx
            current_fy = candidate.fy
            completed += 1

    return (
        ElasticaSolution(current_theta, current_fx, current_fy, 0, 0),
        total_refinements,
    )


def nearest_propagation_order(targets, start):
    """Walk the current d layer from the nearest unsolved angular neighbor."""
    if len(targets) == 0:
        return np.empty(0, dtype=int)
    if start is None:
        start = targets[0]
    start = np.asarray(start, dtype=float)
    start_index = int(np.argmin(np.sum((targets - start[None, :]) ** 2, axis=1)))
    tree = cKDTree(targets[:, 1:])
    remaining = np.ones(len(targets), dtype=bool)
    current = start_index
    remaining[current] = False
    order = [current]
    neighbor_count = min(32, len(targets))
    while remaining.any():
        _, neighbors = tree.query(
            targets[current, None, 1:], k=neighbor_count, workers=-1
        )
        neighbors = np.atleast_1d(neighbors)
        available = neighbors[remaining[neighbors]]
        if len(available) == 0:
            available = np.flatnonzero(remaining)
        current = int(available[0])
        remaining[current] = False
        order.append(current)
    return np.asarray(order, dtype=int)


def nearest_indices(target, source_params, count):
    """Return the closest solved angular coordinates."""
    if source_params.size == 0:
        return np.empty(0, dtype=int)
    count = min(count, len(source_params))
    angular = source_params[:, 1:]
    distance = np.sum((angular - target[None, 1:]) ** 2, axis=1)
    return np.argpartition(distance, count - 1)[:count]


def solve_target_batch(
    solver,
    targets,
    existing_params,
    existing_theta,
    existing_fx,
    existing_fy,
    previous_params,
    previous_theta,
    previous_fx,
    previous_fy,
    mac_threshold,
    max_refinements,
    continuation_candidates,
    cold_first,
    cold_start=None,
    propagation_start=None,
):
    """Solve a batch in nearest-neighbor continuation order."""
    targets = np.asarray(targets, dtype=float)
    if targets.size == 0:
        return empty_layer()
    existing_count = len(existing_params)
    order = nearest_propagation_order(targets, propagation_start)
    if cold_first:
        if cold_start is None:
            cold_point = np.array([targets[0, 0], 0.0, 0.0])
        elif isinstance(cold_start, dict):
            cold_point = np.array(
                [cold_start["d"], cold_start["phi1"], cold_start["phi2"]],
                dtype=float,
            )
        else:
            cold_point = np.asarray(cold_start, dtype=float)
        first = np.argsort(np.linalg.norm(targets - cold_point[None, :], axis=1))[:1]
        order = np.concatenate((first, order[order != first[0]]))

    capacity = existing_count + len(targets)
    accepted_params = np.empty((capacity, 3), dtype=float)
    accepted_theta = np.empty((capacity, solver.theta.x.array.size), dtype=float)
    accepted_fx = np.empty(capacity, dtype=float)
    accepted_fy = np.empty(capacity, dtype=float)
    refinement_counts = np.zeros(capacity, dtype=np.int32)
    accepted_params[:existing_count] = existing_params
    if existing_count:
        accepted_theta[:existing_count] = existing_theta
        accepted_fx[:existing_count] = existing_fx
        accepted_fy[:existing_count] = existing_fy
    accepted_count = existing_count

    for target_index in order:
        target = targets[target_index]
        if accepted_count:
            candidate_indices = [accepted_count - 1]
        elif previous_params is not None and len(previous_params):
            candidate_indices = [len(previous_params) - 1]
        else:
            candidate_indices = []
        using_previous = not accepted_count and bool(len(previous_params))
        if using_previous:
            for candidate_index in candidate_indices:
                try:
                    solution, refinements = solve_with_adaptive_mac(
                        solver, target, previous_params[candidate_index],
                        previous_theta[candidate_index], previous_fx[candidate_index],
                        previous_fy[candidate_index], mac_threshold, max_refinements,
                    )
                except RuntimeError:
                    solution = None
                if solution is not None:
                    break
        else:
            solution = None
            refinements = 0
            for candidate_index in candidate_indices:
                try:
                    solution, refinements = solve_with_adaptive_mac(
                        solver, target, accepted_params[candidate_index],
                        accepted_theta[candidate_index], accepted_fx[candidate_index],
                        accepted_fy[candidate_index], mac_threshold, max_refinements,
                    )
                except RuntimeError:
                    solution = None
                if solution is not None:
                    break
        if solution is None and cold_first and accepted_count == 0:
            try:
                solution, refinements = solve_with_adaptive_mac(
                    solver, target, None, None, 0.0, 0.0,
                    mac_threshold, max_refinements, cold_start_mode=1,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"Cold start failed at d={target[0]:.8g}, "
                    f"phi1={target[1]:.8g}, phi2={target[2]:.8g}: {error}"
                ) from error
        if solution is None:
            raise RuntimeError(
                f"Warm-started solve failed at d={target[0]:.8g}, "
                f"phi1={target[1]:.8g}, phi2={target[2]:.8g}; "
                "no additional cold start is permitted"
            )
        accepted_params[accepted_count] = target
        accepted_theta[accepted_count] = solution.theta
        accepted_fx[accepted_count] = solution.fx
        accepted_fy[accepted_count] = solution.fy
        refinement_counts[accepted_count] = refinements
        accepted_count += 1
        if accepted_count % 1000 == 0 or accepted_count == capacity:
            print(f"    solved {accepted_count - existing_count:,}/{len(targets):,}", flush=True)

    return {
        "params": accepted_params[:accepted_count].copy(),
        "theta": accepted_theta[:accepted_count].copy(),
        "fx": accepted_fx[:accepted_count].copy(),
        "fy": accepted_fy[:accepted_count].copy(),
        "refinements": refinement_counts[:accepted_count].copy(),
        "new_nodes": [],
    }


def empty_layer():
    return {
        "params": np.empty((0, 3), dtype=float),
        "theta": np.empty((0, 0), dtype=float),
        "fx": np.empty(0, dtype=float),
        "fy": np.empty(0, dtype=float),
        "refinements": np.empty(0, dtype=np.int32),
        "new_nodes": [],
    }


def theta_parameter_gradients(params, theta):
    """Estimate d theta(s) / d phi1 and / d phi2 at every parameter point."""
    count, beam_size = theta.shape
    if count < 3:
        return np.empty((count, 2, beam_size), dtype=float)
    neighbor_count = min(7, count)
    _, neighbors = cKDTree(params[:, 1:]).query(
        params[:, 1:], k=neighbor_count, workers=-1
    )
    neighbors = np.atleast_2d(neighbors)
    if neighbor_count == 1:
        return np.empty((count, 2, beam_size), dtype=float)
    displacements = (params[neighbors] - params[:, None, :])[:, :, 1:]
    pair_a, pair_b = np.triu_indices(neighbor_count, k=1)
    cross = np.abs(
        displacements[:, pair_a, 0] * displacements[:, pair_b, 1]
        - displacements[:, pair_a, 1] * displacements[:, pair_b, 0]
    )
    best_pair = np.argmax(cross, axis=1)
    first = pair_a[best_pair]
    second = pair_b[best_pair]
    matrix = np.empty((count, 2, 2), dtype=float)
    matrix[:, 0, :] = displacements[np.arange(count), first]
    matrix[:, 1, :] = displacements[np.arange(count), second]
    delta_theta = np.stack(
        (
            theta[neighbors[np.arange(count), first]] - theta,
            theta[neighbors[np.arange(count), second]] - theta,
        ),
        axis=1,
    )
    gradient = np.linalg.solve(matrix, delta_theta)
    valid = cross[np.arange(count), best_pair] > 1.0e-20
    gradient[~valid] = np.nan
    return gradient


def gradient_refinement_points(params, theta, options, level):
    """Insert high-gradient edge midpoints in both phi directions."""
    gradients = theta_parameter_gradients(params, theta)
    tree = cKDTree(params[:, 1:])
    candidate_rows = []
    candidate_scores = []
    initial_spacing = float(options.get("initial_angle_spacing", np.pi / 144.0))
    min_spacing = float(options.get("min_spacing", 1.0e-5)) * (level + 1)
    min_spacing = max(min_spacing, initial_spacing / (2.0 ** (level + 1)) * 0.05)
    neighbor_count = min(8, len(params))
    _, neighbors = tree.query(params[:, 1:], k=neighbor_count, workers=-1)
    neighbors = np.atleast_2d(neighbors)
    quantile = float(options.get("quantile", 0.9))
    per_axis_limit = max(1, int(options.get("max_new_points_per_level", 512)) // 2)

    for parameter_axis in (0, 1):
        component = gradients[:, parameter_axis]
        scores = np.sqrt(np.nanmean(component ** 2, axis=1))
        axis_pool = []
        for point_index, point in enumerate(params):
            for neighbor_index in neighbors[point_index]:
                displacement = params[neighbor_index, 1:] - point[1:]
                if abs(displacement[parameter_axis]) < 0.25 * np.linalg.norm(displacement):
                    continue
                midpoint = 0.5 * (point[1:] + params[neighbor_index, 1:])
                if midpoint.sum() < -1.0e-12 or np.linalg.norm(displacement) < min_spacing:
                    continue
                axis_pool.append((midpoint, scores[point_index]))
        if not axis_pool:
            continue
        midpoint_array = np.asarray([row[0] for row in axis_pool])
        separation = tree.query(midpoint_array, k=1)[0]
        unique = np.flatnonzero(separation >= min_spacing)
        unique = unique[np.argsort(-np.asarray([axis_pool[index][1] for index in unique]))]
        count = min(per_axis_limit, max(1, int(np.ceil((1.0 - quantile) * len(unique)))))
        for index in unique[:count]:
            candidate_rows.append(axis_pool[index][0])
            candidate_scores.append(axis_pool[index][1])

    if not candidate_rows:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    candidates = np.column_stack(
        (
            np.full(len(candidate_rows), params[0, 0]),
            np.asarray(candidate_rows),
        )
    )
    order = np.lexsort((candidates[:, 2], candidates[:, 1], -np.asarray(candidate_scores)))
    unique_rows = []
    unique_scores = []
    for index in order:
        if any(np.linalg.norm(candidates[index, 1:] - row[1:]) < min_spacing for row in unique_rows):
            continue
        unique_rows.append(candidates[index])
        unique_scores.append(candidate_scores[index])
    return np.asarray(unique_rows, dtype=float), np.asarray(unique_scores, dtype=float)


def layer_gradient_score(params, theta):
    """Return one RMS theta-gradient score per layer point."""
    gradients = theta_parameter_gradients(params, theta)
    score = np.sqrt(np.nanmean(gradients[:, 0] ** 2 + gradients[:, 1] ** 2, axis=1))
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def build_result_arrays(seed_data, layer_outputs, phi1_values, phi2_values, beam_x):
    """Assemble representative and mirrored full-study results in chunks."""
    uniform_count = len(seed_data["coords"])
    spacing = np.diff(beam_x)
    base_fx = np.full(uniform_count, np.nan)
    base_fy = np.full(uniform_count, np.nan)
    base_moment = np.full(uniform_count, np.nan)
    base_energy = np.full(uniform_count, np.nan)
    base_score = np.full(uniform_count, np.nan)
    base_refinement = np.full(uniform_count, -1, dtype=np.int32)
    base_theta = np.full((uniform_count, len(beam_x)), np.nan, dtype=np.float32)
    base_solved = np.zeros(uniform_count, dtype=bool)
    base_adaptive = np.zeros(uniform_count, dtype=bool)
    base_mirrored = np.zeros(uniform_count, dtype=bool)
    base_representative = np.zeros(uniform_count, dtype=bool)
    chunks = {name: [seed_data["coords"]] for name in ("coords",)}
    chunks["ijk"] = [seed_data["ijk"]]
    for name in ("fx", "fy", "moment", "energy", "theta_gradient_score"):
        chunks[name] = []
    chunks["refinement_count"] = []
    chunks["theta"] = []
    for name in ("solved", "adaptive", "mirrored", "representative"):
        chunks[name] = []

    def append_chunk(params, theta, fx, fy, moment, energy, score, refinement,
                     adaptive, mirrored, representative):
        chunks["coords"].append(params)
        chunks["ijk"].append(np.column_stack((
            np.searchsorted(seed_data["d_vals"], params[:, 0]),
            np.abs(phi1_values[:, None] - params[None, :, 1]).argmin(axis=0),
            np.abs(phi2_values[:, None] - params[None, :, 2]).argmin(axis=0),
        )).astype(np.int32))
        chunks["fx"].append(fx)
        chunks["fy"].append(fy)
        chunks["moment"].append(moment)
        chunks["energy"].append(energy)
        chunks["theta_gradient_score"].append(score)
        chunks["refinement_count"].append(refinement.astype(np.int32))
        chunks["theta"].append(theta.astype(np.float32, copy=False))
        chunks["solved"].append(np.ones(len(params), dtype=bool))
        chunks["adaptive"].append(adaptive)
        chunks["mirrored"].append(mirrored)
        chunks["representative"].append(representative)

    for output in layer_outputs:
        params = np.asarray(output["params"], dtype=float)
        theta = np.asarray(output["theta"], dtype=float)
        slopes = np.diff(theta, axis=1) / spacing[None, :]
        source_moment = slopes[:, 0]
        source_energy = 0.5 * np.sum(
            0.5 * (slopes[:, :-1] ** 2 + slopes[:, 1:] ** 2) * spacing[None, 1:],
            axis=1,
        )
        representative_count = output["representative_count"]
        d_indices = np.searchsorted(seed_data["d_vals"], params[:, 0])
        j_indices = np.abs(phi1_values[:, None] - params[None, :, 1]).argmin(axis=0)
        k_indices = np.abs(phi2_values[:, None] - params[None, :, 2]).argmin(axis=0)
        base_nodes = (d_indices * len(phi1_values) + j_indices) * len(phi2_values) + k_indices
        source_seed_mask = np.arange(len(params)) < representative_count
        base_fx[base_nodes[source_seed_mask]] = output["fx"][:representative_count]
        base_fy[base_nodes[source_seed_mask]] = output["fy"][:representative_count]
        base_moment[base_nodes[source_seed_mask]] = source_moment[:representative_count]
        base_energy[base_nodes[source_seed_mask]] = source_energy[:representative_count]
        base_score[base_nodes[source_seed_mask]] = output["gradient_score"][:representative_count]
        base_refinement[base_nodes[source_seed_mask]] = output["refinements"][:representative_count]
        base_theta[base_nodes[source_seed_mask]] = theta[:representative_count]
        base_solved[base_nodes[source_seed_mask]] = True
        base_representative[base_nodes[source_seed_mask]] = True

        append_chunk(
            params, theta, output["fx"], output["fy"], source_moment,
            source_energy, output["gradient_score"], output["refinements"],
            np.arange(len(params)) >= representative_count,
            np.zeros(len(params), dtype=bool), np.ones(len(params), dtype=bool),
        )
        non_diagonal = ~np.isclose(params[:, 1] + params[:, 2], 0.0, atol=1.0e-12)
        if np.any(non_diagonal):
            mirror_params = params[non_diagonal].copy()
            mirror_params[:, 1:] = np.column_stack(
                (-params[non_diagonal, 2], -params[non_diagonal, 1])
            )
            mirror_theta = -theta[non_diagonal, ::-1]
            mirror_j = np.abs(phi1_values[:, None] - mirror_params[None, :, 1]).argmin(axis=0)
            mirror_k = np.abs(phi2_values[:, None] - mirror_params[None, :, 2]).argmin(axis=0)
            mirror_nodes = (d_indices[non_diagonal] * len(phi1_values) + mirror_j) * len(phi2_values) + mirror_k
            mirror_source_indices = np.flatnonzero(non_diagonal)
            mirror_seed_mask = mirror_source_indices < representative_count
            base_fx[mirror_nodes[mirror_seed_mask]] = output["fx"][mirror_source_indices[mirror_seed_mask]]
            base_fy[mirror_nodes[mirror_seed_mask]] = -output["fy"][mirror_source_indices[mirror_seed_mask]]
            base_moment[mirror_nodes[mirror_seed_mask]] = -slopes[mirror_source_indices[mirror_seed_mask], -1]
            base_energy[mirror_nodes[mirror_seed_mask]] = source_energy[mirror_source_indices[mirror_seed_mask]]
            base_score[mirror_nodes[mirror_seed_mask]] = output["gradient_score"][mirror_source_indices[mirror_seed_mask]]
            base_refinement[mirror_nodes[mirror_seed_mask]] = output["refinements"][mirror_source_indices[mirror_seed_mask]]
            base_theta[mirror_nodes[mirror_seed_mask]] = mirror_theta[mirror_seed_mask]
            base_solved[mirror_nodes[mirror_seed_mask]] = True
            base_mirrored[mirror_nodes[mirror_seed_mask]] = True
            append_chunk(
                mirror_params, mirror_theta, output["fx"][non_diagonal],
                -output["fy"][non_diagonal], -slopes[non_diagonal, -1],
                source_energy[non_diagonal], output["gradient_score"][non_diagonal],
                output["refinements"][non_diagonal],
                mirror_source_indices >= representative_count,
                np.ones(np.count_nonzero(non_diagonal), dtype=bool),
                np.zeros(np.count_nonzero(non_diagonal), dtype=bool),
            )

    for name, values in (
        ("fx", base_fx), ("fy", base_fy), ("moment", base_moment),
        ("energy", base_energy), ("theta_gradient_score", base_score),
        ("refinement_count", base_refinement), ("theta", base_theta),
        ("solved", base_solved), ("adaptive", base_adaptive),
        ("mirrored", base_mirrored), ("representative", base_representative),
    ):
        chunks[name].insert(0, values)
    return {
        "coords": np.concatenate(chunks["coords"]),
        "ijk": np.concatenate(chunks["ijk"]),
        "tets": np.empty((0, 4), dtype=np.int64),
        "grid_shape": seed_data["grid_shape"],
        "d_vals": seed_data["d_vals"],
        "p1_vals": phi1_values,
        "p2_vals": phi2_values,
        "uniform_seed_size": np.array(uniform_count, dtype=np.int64),
        "fx": np.concatenate(chunks["fx"]),
        "fy": np.concatenate(chunks["fy"]),
        "moment": np.concatenate(chunks["moment"]),
        "energy": np.concatenate(chunks["energy"]),
        "refinement_count": np.concatenate(chunks["refinement_count"]),
        "theta_gradient_score": np.concatenate(chunks["theta_gradient_score"]),
        "solved": np.concatenate(chunks["solved"]),
        "theta": np.concatenate(chunks["theta"]),
        "beam_x": beam_x,
        "adaptive": np.concatenate(chunks["adaptive"]),
        "mirrored": np.concatenate(chunks["mirrored"]),
        "representative": np.concatenate(chunks["representative"]),
        "study_mode": "per_d_half_diagonal_theta_gradient",
    }


def save_results(path, results):
    """Write a complete results archive without expensive compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **results)
    os.replace(temporary, path)


def configured_path(config_path, value):
    """Resolve an output path relative to its configuration file."""
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.parent / path


def main():
    """Run the per-d structured study."""
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "config.json").resolve()
    with config_path.open() as config_file:
        config = json.load(config_file)
    initial_path = configured_path(
        config_path, config.get("output_npz", "solution_space.npz")
    )
    config_path, initial_path = load_or_create_uniform_space(config_path, initial_path)
    with np.load(initial_path, allow_pickle=False) as seed_data:
        seed = {name: seed_data[name].copy() for name in seed_data.files}

    d_values = np.asarray(seed["d_vals"], dtype=float)
    phi1_values = np.asarray(seed["p1_vals"], dtype=float)
    phi2_values = np.asarray(seed["p2_vals"], dtype=float)
    grid_shape = tuple(int(value) for value in seed["grid_shape"])
    beam_elements = int(config.get("beam_elements", 100))
    _, theta_space, multiplier_space, beam_x = precompute_elastica_spaces(
        elements=beam_elements
    )
    solver = ElasticaSolver(
        theta_space,
        multiplier_space,
        petsc_options={
            "snes_max_it": int(config.get("snes_max_it", 200)),
            "snes_rtol": float(config.get("snes_rtol", 1.0e-10)),
            "snes_atol": float(config.get("snes_atol", 1.0e-12)),
        },
    )

    representative_mask = seed["coords"][:, 1] + seed["coords"][:, 2] >= -1.0e-12
    layer_outputs = []
    previous = empty_layer()
    total_seed = int(representative_mask.sum())
    solved_count = 0
    total_new = 0
    timer = NodeSolveTimer(int(config.get("timing_interval", 1000)))
    gradient_options = dict(config.get("gradient_refinement", {}))
    gradient_options.setdefault(
        "initial_angle_spacing",
        float(min(np.min(np.diff(phi1_values)), np.min(np.diff(phi2_values)))),
    )
    levels = int(gradient_options.get("levels", 3))
    mac_threshold = float(config.get("mac_threshold", 0.9))
    max_refinements = int(config.get("max_mac_refinements", 5))
    continuation_candidates = int(config.get("continuation_candidates", 8))
    results_path = configured_path(
        config_path, config.get("results_npz", "solution_space_results.npz")
    )
    checkpoint_path = config.get("live_results_npz")
    checkpoint_path = configured_path(config_path, checkpoint_path) if checkpoint_path else None
    checkpoint_layers = int(config.get("checkpoint_d_layers", 5))
    first_plane = True
    cold_point = np.array(
        [
            float(config["cold_start"]["d"]),
            float(config["cold_start"]["phi1"]),
            float(config["cold_start"]["phi2"]),
        ],
        dtype=float,
    )

    print(
        f"Structured study: {len(d_values)} d layers, "
        f"{total_seed:,} representative seeds, {levels} gradient levels"
    )
    for d_index, d_value in enumerate(d_values):
        timer.start_node()
        representative_indices = np.flatnonzero(
            (seed["ijk"][:, 0] == d_index) & representative_mask
        )
        representative_params = seed["coords"][representative_indices].copy()
        layer = solve_target_batch(
            solver,
            representative_params,
            previous["params"][:0],
            previous["theta"][:0],
            previous["fx"][:0],
            previous["fy"][:0],
            previous["params"],
            previous["theta"],
            previous["fx"],
            previous["fy"],
            mac_threshold,
            max_refinements,
            continuation_candidates,
            first_plane,
            config.get("cold_start"),
            cold_point if first_plane else previous["params"][-1],
        )
        representative_count = len(representative_params)
        all_params = [representative_params]
        all_theta = [layer["theta"]]
        all_fx = [layer["fx"]]
        all_fy = [layer["fy"]]
        all_refinements = [layer["refinements"]]

        for level in range(levels):
            points, scores = gradient_refinement_points(
                np.concatenate(all_params),
                np.concatenate(all_theta),
                gradient_options,
                level,
            )
            if len(points) == 0:
                print(f"  gradient level {level + 1}: no admissible points")
                break
            added = solve_target_batch(
                solver,
                points,
                np.concatenate(all_params),
                np.concatenate(all_theta),
                np.concatenate(all_fx),
                np.concatenate(all_fy),
                empty_layer()["params"],
                empty_layer()["theta"],
                empty_layer()["fx"],
                empty_layer()["fy"],
                mac_threshold,
                max_refinements,
                continuation_candidates,
                False,
                propagation_start=np.concatenate(all_params)[-1],
            )
            existing_count = len(np.concatenate(all_params))
            added_params = added["params"][existing_count:]
            added_theta = added["theta"][existing_count:]
            added_fx = added["fx"][existing_count:]
            added_fy = added["fy"][existing_count:]
            added_refinements = added["refinements"][existing_count:]
            all_params.append(added_params)
            all_theta.append(added_theta)
            all_fx.append(added_fx)
            all_fy.append(added_fy)
            all_refinements.append(added_refinements)
            total_new += len(added_params)
            print(
                f"  gradient level {level + 1}: added {len(added_params):,} points "
                f"(peak theta-gradient {scores.max():.6g})"
            )

        params = np.concatenate(all_params)
        theta = np.concatenate(all_theta)
        fx = np.concatenate(all_fx)
        fy = np.concatenate(all_fy)
        refinements = np.concatenate(all_refinements)
        gradient_score = layer_gradient_score(params, theta)
        layer_output = {
            "params": params,
            "theta": theta.astype(np.float32),
            "fx": fx,
            "fy": fy,
            "refinements": refinements,
            "gradient_score": gradient_score,
            "representative_count": representative_count,
        }
        layer_outputs.append(layer_output)
        solved_count += len(params) * 2 - int(np.count_nonzero(np.isclose(params[:, 1] + params[:, 2], 0.0, atol=1.0e-12)))
        previous = {
            "params": params[-1:],
            "theta": theta[-1:],
            "fx": fx[-1:],
            "fy": fy[-1:],
        }
        solved_count = min(solved_count, len(d_values) * len(phi1_values) * len(phi2_values))
        timer.finish_node(d_index, solved_count)
        print(
            f"d={d_value:.6f}: representative={len(params):,}, "
            f"adaptive={len(params) - representative_count:,}, "
            f"cumulative adaptive={total_new:,}"
        )
        if checkpoint_path is not None and (d_index + 1) % checkpoint_layers == 0:
            save_results(checkpoint_path, build_result_arrays(
                seed, layer_outputs, phi1_values, phi2_values, beam_x
            ))
        first_plane = False

    results = build_result_arrays(seed, layer_outputs, phi1_values, phi2_values, beam_x)
    expected = int(np.prod(grid_shape))
    base_solved = results["solved"][: len(seed["coords"])]
    if len(results["coords"]) < expected or not base_solved.all() or not results["solved"].all():
        raise RuntimeError(
            f"Incomplete mirrored study: {results['solved'].sum()}/{expected} base points"
        )
    save_results(results_path, results)
    print(
        f"Saved {len(results['coords']):,}/{expected:,} points to {results_path}; "
        f"independent solves={total_seed + total_new:,}, "
        f"mirror fraction={results['mirrored'].mean():.1%}"
    )
    if config.get("final_visualization", False):
        from vis import plot_results
        plot_results(str(results_path))


if __name__ == "__main__":
    main()
