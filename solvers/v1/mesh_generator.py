"""
mesh_generator.py
Generate the solution space mesh for the elastica solver.
Uses mesh_gen.py to build a transfinite tet mesh over the (d, phi1, phi2) box.
"""

import json
from mesh_gen import build_mesh


def validate_and_fix_bounds(bounds, name):
    """
    Ensure bounds are valid (min <= max). If not, swap and warn.

    Parameters
    ----------
    bounds : tuple or list of three numbers (min, max, step)
    name : str
        Name of the bounds for warning message.

    Returns
    -------
    tuple : (min, max, step) with min <= max
    """
    lo, hi, step = bounds
    if lo > hi:
        print(f"Warning: {bounds} for {name} has min > max. Swapping to [{hi}, {lo}, {step}].")
        lo, hi = hi, lo
    return (lo, hi, step)


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

    d_bounds = validate_and_fix_bounds(tuple(config["d_bounds"]), "d_bounds")
    phi1_bounds = validate_and_fix_bounds(tuple(config["phi1_bounds"]), "phi1_bounds")
    phi2_bounds = validate_and_fix_bounds(tuple(config["phi2_bounds"]), "phi2_bounds")
    out_msh = config.get("output_mesh", "solution_space.msh")
    out_npz = config.get("output_npz", "solution_space.npz")

    # Build the mesh
    npz_file = build_mesh(d_bounds, phi1_bounds, phi2_bounds,
                          out_msh=out_msh, out_npz=out_npz, verbose=False)
    return npz_file


if __name__ == "__main__":
    # Generate the mesh when this script is run directly
    npz_file = generate_solution_space_mesh()
    print(f"Generated solution space mesh: {npz_file}")