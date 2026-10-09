"""
mesh_generator.py
Generate the solution space mesh for the elastica solver.
Uses irregular_mesh_gen.py to build an irregular tet mesh over the
(d, phi1, phi2) box.
"""

import json
import os
import numpy as np


def _axis_values(bounds, step, name):
    if step <= 0:
        raise ValueError(f"{name}_step must be positive")
    values = np.arange(bounds[0], bounds[1], step, dtype=float)
    if len(values) == 0 or not np.isclose(values[-1], bounds[1]):
        values = np.append(values, bounds[1])
    else:
        values[-1] = bounds[1]
    return values


def build_rectilinear_mesh(d_bounds, phi1_bounds, phi2_bounds, steps,
                           out_msh, out_npz, scale_parameters=True,
                           use_symmetry=False):
    d_values = _axis_values(d_bounds, steps[0], "d")
    phi1_values = _axis_values(phi1_bounds, steps[1], "phi1")
    phi2_values = _axis_values(phi2_bounds, steps[2], "phi2")
    if use_symmetry and not (
            np.isclose(phi1_bounds[0], -phi1_bounds[1])
            and np.isclose(phi2_bounds[0], -phi2_bounds[1])
            and np.isclose(steps[1], steps[2])
            and len(phi1_values) == len(phi2_values)):
        raise ValueError(
            "Diagonal symmetry requires matching symmetric phi bounds and steps"
        )

    shape = (len(d_values), len(phi1_values), len(phi2_values))
    coords = np.stack(np.meshgrid(
        d_values, phi1_values, phi2_values, indexing="ij"
    ), axis=-1).reshape(-1, 3)
    d_count, phi1_count, phi2_count = shape

    cell_i, cell_j, cell_k = np.meshgrid(
        np.arange(d_count - 1, dtype=np.int64),
        np.arange(phi1_count - 1, dtype=np.int64),
        np.arange(phi2_count - 1, dtype=np.int64),
        indexing="ij",
    )
    cell_base = (cell_i * phi1_count + cell_j) * phi2_count + cell_k
    v000 = cell_base
    v100 = v000 + phi1_count * phi2_count
    v010 = v000 + phi2_count
    v110 = v100 + phi2_count
    v001 = v000 + 1
    v101 = v100 + 1
    v011 = v010 + 1
    v111 = v110 + 1
    tets = np.stack([
        np.stack([v000, v100, v110, v111], axis=-1),
        np.stack([v000, v110, v010, v111], axis=-1),
        np.stack([v000, v010, v011, v111], axis=-1),
        np.stack([v000, v011, v001, v111], axis=-1),
        np.stack([v000, v001, v101, v111], axis=-1),
        np.stack([v000, v101, v100, v111], axis=-1),
    ], axis=-2).reshape(-1, 4)
    parameter_offsets = np.array([d_bounds[0], phi1_bounds[0], phi2_bounds[0]])
    parameter_spans = np.array([
        d_bounds[1] - d_bounds[0],
        phi1_bounds[1] - phi1_bounds[0],
        phi2_bounds[1] - phi2_bounds[0],
    ])
    mesh_coords = ((coords - parameter_offsets) / parameter_spans
                   if scale_parameters else coords.copy())

    solve_mask = (coords[:, 1] + coords[:, 2] >= -1e-12)
    node_indices = np.arange(len(coords), dtype=np.int64)
    plane_size = phi1_count * phi2_count
    node_i = node_indices // plane_size
    node_plane = node_indices % plane_size
    node_j = node_plane // phi2_count
    node_k = node_plane % phi2_count
    mirror_indices = (
        node_i * plane_size
        + (phi1_count - 1 - node_k) * phi2_count
        + (phi2_count - 1 - node_j)
    )
    if not use_symmetry:
        solve_mask[:] = True
        mirror_indices[:] = np.arange(len(coords))

    np.savez_compressed(
        out_npz,
        coords=coords,
        mesh_coords=mesh_coords,
        tets=tets,
        d_bounds=np.asarray(d_bounds),
        phi1_bounds=np.asarray(phi1_bounds),
        phi2_bounds=np.asarray(phi2_bounds),
        grid_steps=np.asarray(steps),
        parameter_offsets=parameter_offsets,
        parameter_spans=parameter_spans,
        parameters_scaled=np.array(bool(scale_parameters)),
        mirror_indices=mirror_indices,
        solve_mask=solve_mask,
        symmetric_mesh=np.array(bool(use_symmetry)),
    )
    # The solver consumes the NPZ; retain a marker for the existing cache check.
    with open(out_msh, "w", encoding="ascii") as file:
        file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    print(f"[mesh_gen] wrote rectilinear mesh {out_npz} "
          f"({len(coords):,} nodes, {len(tets):,} tets)")
    return out_npz


def validate_bounds(bounds, name):
    """
    Ensure bounds are valid as [min, max].

    Parameters
    ----------
    bounds : tuple or list of two numbers (min, max)
    name : str
        Name of the bounds for warning message.

    Returns
    -------
    tuple : (min, max)
    """
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError(f"{name} must be [min, max] with min < max")
    return tuple(float(value) for value in bounds)


def generate_solution_space_mesh(config_path="config.json"):
    """
    Read config.json and build the solution space mesh.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.

    Returns
    -------
    str
        Path to the generated .npz file.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    d_bounds = validate_bounds(tuple(config["d_bounds"]), "d_bounds")
    phi1_bounds = validate_bounds(tuple(config["phi1_bounds"]), "phi1_bounds")
    phi2_bounds = validate_bounds(tuple(config["phi2_bounds"]), "phi2_bounds")
    out_msh = config.get("output_mesh", "solution_space.msh")
    out_npz = config.get("output_npz", "solution_space.npz")

    steps = (
        float(config.get("grid_steps", {}).get("d", 0.01)),
        float(config.get("grid_steps", {}).get("phi1", 0.02)),
        float(config.get("grid_steps", {}).get("phi2", 0.02)),
    )
    scale_parameters = bool(config.get("scale_parameters_for_mesh", True))
    use_symmetry = bool(config.get("solve_phi_symmetry", False))
    return build_rectilinear_mesh(
        d_bounds, phi1_bounds, phi2_bounds, steps,
        out_msh=out_msh, out_npz=out_npz,
        scale_parameters=scale_parameters, use_symmetry=use_symmetry,
    )


if __name__ == "__main__":
    # Generate the mesh when this script is run directly
    npz_file = generate_solution_space_mesh()
    print(f"Generated solution space mesh: {npz_file}")