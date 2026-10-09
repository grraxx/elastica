"""Uniform structured solution-space initialization."""

import json
from pathlib import Path

import numpy as np


def validate_axis(bounds, name):
    """Validate and normalize an inclusive (lower, upper, spacing) axis."""
    values = np.asarray(bounds, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite [lower, upper, spacing] values")
    lower, upper, spacing = values
    if lower > upper:
        lower, upper = upper, lower
    if spacing <= 0.0:
        raise ValueError(f"{name} spacing must be positive")
    size = int(np.floor((upper - lower) / spacing + 1.0e-12)) + 1
    axis = np.linspace(lower, upper, size)
    axis[0], axis[-1] = lower, upper
    return axis


def structured_coordinates(d_values, phi1_values, phi2_values):
    """Return Cartesian product coordinates and integer grid indices."""
    d_grid, phi1_grid, phi2_grid = np.meshgrid(
        d_values,
        phi1_values,
        phi2_values,
        indexing="ij",
    )
    coordinates = np.column_stack((d_grid.ravel(), phi1_grid.ravel(), phi2_grid.ravel()))
    indices = np.indices((len(d_values), len(phi1_values), len(phi2_values)))
    return coordinates, indices.reshape(3, -1).T.astype(np.int32)


def build_uniform_space(config, output_npz):
    """Build the initial orthogonal, uniformly populated solution space."""
    d_values = validate_axis(config["d_bounds"], "d_bounds")
    phi1_values = validate_axis(config["phi1_bounds"], "phi1_bounds")
    phi2_values = validate_axis(config["phi2_bounds"], "phi2_bounds")
    if not (np.isclose(phi1_values[0], -phi1_values[-1])
            and np.isclose(phi2_values[0], -phi2_values[-1])):
        raise ValueError("phi1_bounds and phi2_bounds must be symmetric about zero")
    coordinates, ijk = structured_coordinates(d_values, phi1_values, phi2_values)
    np.savez(
        output_npz,
        coords=coordinates,
        ijk=ijk,
        tets=np.empty((0, 4), dtype=np.int64),
        grid_shape=np.array([len(d_values), len(phi1_values), len(phi2_values)]),
        d_vals=d_values,
        p1_vals=phi1_values,
        p2_vals=phi2_values,
        study_mode="uniform_half_diagonal",
    )
    return output_npz


def configured_axes(config):
    """Return validated axes from a study configuration."""
    return (
        validate_axis(config["d_bounds"], "d_bounds"),
        validate_axis(config["phi1_bounds"], "phi1_bounds"),
        validate_axis(config["phi2_bounds"], "phi2_bounds"),
    )


def load_or_create_uniform_space(config_path, output_npz):
    """Load a matching uniform space or replace a stale one."""
    config_path = Path(config_path).resolve()
    with config_path.open() as config_file:
        config = json.load(config_file)
    output_npz = Path(output_npz).resolve()
    expected = configured_axes(config)
    if output_npz.exists():
        with np.load(output_npz, allow_pickle=False) as data:
            required = {"coords", "ijk", "grid_shape", "d_vals", "p1_vals", "p2_vals"}
            if required.issubset(data.files):
                actual = tuple(
                    np.asarray(data[name], dtype=float)
                    for name in ("d_vals", "p1_vals", "p2_vals")
                )
                if all(np.array_equal(first, second) for first, second in zip(expected, actual)):
                    return config_path, output_npz
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    return config_path, build_uniform_space(config, output_npz)
