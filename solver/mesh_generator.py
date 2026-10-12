"""
mesh_generator.py
Generate the uniform tetrahedral solution-space mesh over the (d, phi1, phi2)
box: a uniformly spaced grid with every cell split into 6 tetrahedra
(Kuhn/Freudenthal split along the cell's main diagonal).
"""

import json
import os
import numpy as np


def _axis_values(bounds, step, name):
    """Uniformly spaced values from bounds[0] to bounds[1], spacing as close
    to step as divides the range exactly."""
    if step <= 0:
        raise ValueError(f"{name} step must be positive")
    intervals = max(1, int(round((bounds[1] - bounds[0]) / step)))
    return np.linspace(bounds[0], bounds[1], intervals + 1)


def validate_bounds(bounds, name):
    """Return bounds as a (min, max) float tuple, raising if they are invalid."""
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError(f"{name} must be [min, max] with min < max")
    return tuple(float(value) for value in bounds)


def mesh_settings(config):
    """Extract the config values that determine the mesh."""
    steps = config.get("grid_steps", {})
    return {
        "d_bounds": validate_bounds(config["d_bounds"], "d_bounds"),
        "phi1_bounds": validate_bounds(config["phi1_bounds"], "phi1_bounds"),
        "phi2_bounds": validate_bounds(config["phi2_bounds"], "phi2_bounds"),
        "grid_steps": (float(steps.get("d", 0.01)),
                       float(steps.get("phi1", 0.02)),
                       float(steps.get("phi2", 0.02))),
        "symmetric_mesh": bool(config.get("solve_phi_symmetry", False)),
    }


def build_rectilinear_mesh(d_bounds, phi1_bounds, phi2_bounds, grid_steps,
                           symmetric_mesh, out_npz):
    d_values = _axis_values(d_bounds, grid_steps[0], "d")
    phi1_values = _axis_values(phi1_bounds, grid_steps[1], "phi1")
    phi2_values = _axis_values(phi2_bounds, grid_steps[2], "phi2")
    # The mirror map (phi1, phi2) -> (-phi2, -phi1) must send grid nodes to
    # grid nodes, i.e. the phi1 grid reversed must equal the negated phi2 grid.
    if symmetric_mesh and not (
            len(phi1_values) == len(phi2_values)
            and np.allclose(phi1_values[::-1], -phi2_values)):
        raise ValueError(
            "Mirror symmetry requires phi1 and phi2 grids that are mirror "
            "images: matching symmetric bounds and a step dividing the range"
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

    node_i, node_j, node_k = np.unravel_index(np.arange(len(coords)), shape)
    if symmetric_mesh:
        # Solve the half phi1 + phi2 >= 0 (by grid index, so the diagonal is
        # exact); the other half is filled by the mirror (i, j, k) -> (i, n-1-k, n-1-j).
        solve_mask = node_j + node_k >= phi1_count - 1
        mirror_indices = np.ravel_multi_index(
            (node_i, phi1_count - 1 - node_k, phi2_count - 1 - node_j), shape
        )
    else:
        solve_mask = np.ones(len(coords), dtype=bool)
        mirror_indices = np.arange(len(coords), dtype=np.int64)

    np.savez_compressed(
        out_npz,
        coords=coords,
        tets=tets,
        shape=np.asarray(shape),
        d_bounds=np.asarray(d_bounds),
        phi1_bounds=np.asarray(phi1_bounds),
        phi2_bounds=np.asarray(phi2_bounds),
        grid_steps=np.asarray(grid_steps),
        mirror_indices=mirror_indices,
        solve_mask=solve_mask,
        symmetric_mesh=np.array(symmetric_mesh),
    )
    print(f"[mesh_gen] wrote rectilinear mesh {out_npz} "
          f"({len(coords):,} nodes, {len(tets):,} tets)")
    return out_npz


def mesh_matches(npz_file, settings):
    """True if npz_file holds a mesh built with exactly these settings."""
    if not os.path.exists(npz_file):
        return False
    with np.load(npz_file) as mesh_data:
        required = {"coords", "tets", "shape", "mirror_indices", "solve_mask",
                    *settings}
        return required.issubset(mesh_data.files) and all(
            np.allclose(mesh_data[key], value)
            for key, value in settings.items()
        )


def generate_solution_space_mesh(config_path="config.json"):
    """Read the config and build the solution space mesh; return the .npz path."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return build_rectilinear_mesh(
        **mesh_settings(config),
        out_npz=config.get("output_npz", "solution_space.npz"),
    )


if __name__ == "__main__":
    npz_file = generate_solution_space_mesh()
    print(f"Generated solution space mesh: {npz_file}")
