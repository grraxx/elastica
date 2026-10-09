"""Write solution-space NPZ results as a ParaView-readable legacy VTK file.

Usage:
    python npz_to_vtk.py solution_space_results.npz solution_space_results.vtk
"""

import argparse

import numpy as np


def _vtk_name(name):
    return "".join(character if character.isalnum() or character == "_" else "_"
                   for character in name)


def write_vtk(npz_file, vtk_file):
    """Export parameter-space tetrahedra and all per-node result fields."""
    data = np.load(npz_file)
    required = {"coords", "tets", "solved", "fx", "fy", "moment", "energy"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{npz_file} is missing: {', '.join(sorted(missing))}")

    coords = np.asarray(data["coords"], dtype=float)
    tets = np.asarray(data["tets"], dtype=np.int64)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must have shape (number_of_tetrahedra, 4)")

    fields = {
        "solved": np.asarray(data["solved"], dtype=np.int32),
        "fx": np.asarray(data["fx"], dtype=float),
        "fy": np.asarray(data["fy"], dtype=float),
        "moment": np.asarray(data["moment"], dtype=float),
        "energy": np.asarray(data["energy"], dtype=float),
    }
    if "ijk" in data:
        ijk = np.asarray(data["ijk"], dtype=np.int32)
        fields.update({f"grid_{axis}": ijk[:, index]
                       for index, axis in enumerate(("d", "phi1", "phi2"))})

    if "theta" in data:
        theta = np.asarray(data["theta"], dtype=float)
        for index in range(theta.shape[1]):
            fields[f"theta_{index:04d}"] = theta[:, index]

    point_count = len(coords)
    if any(len(values) != point_count for values in fields.values()):
        raise ValueError("Every exported field must contain one value per point")

    with open(vtk_file, "w", encoding="ascii", newline="\n") as output:
        output.write("# vtk DataFile Version 3.0\n")
        output.write("Elastica solution-space results\n")
        output.write("ASCII\n")
        output.write("DATASET UNSTRUCTURED_GRID\n")
        output.write(f"POINTS {point_count} double\n")
        for point in coords:
            output.write(f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n")

        output.write(f"CELLS {len(tets)} {len(tets) * 5}\n")
        for tet in tets:
            output.write(f"4 {int(tet[0])} {int(tet[1])} "
                         f"{int(tet[2])} {int(tet[3])}\n")
        output.write(f"CELL_TYPES {len(tets)}\n")
        output.write(("10\n" * len(tets)))

        output.write(f"POINT_DATA {point_count}\n")
        for name, values in fields.items():
            output.write(f"SCALARS {_vtk_name(name)} double 1\n")
            output.write("LOOKUP_TABLE default\n")
            for value in values:
                output.write(f"{float(value):.17g}\n")

    print(f"Wrote {vtk_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_file", nargs="?", default="solution_space_results.npz")
    parser.add_argument("vtk_file", nargs="?", default="solution_space_results.vtk")
    args = parser.parse_args()
    write_vtk(args.npz_file, args.vtk_file)


if __name__ == "__main__":
    main()
