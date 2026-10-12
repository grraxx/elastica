"""Write only the solved points of a solver VTU or NPZ to a compact VTU for ParaView.

Examples:
    python repair_vtu.py solution_space_results.vtu
    python repair_vtu.py solution_space_live.npz --output solved.vtu
"""

import argparse
import os
import sys

import numpy as np

from vtu import write_vtu


RESULT_FIELDS = ("fx", "fy", "moment", "energy")


def infer_solved(fields):
    solved = fields.get("solved")
    if solved is not None:
        return np.isfinite(solved) & (solved > 0.5)
    masks = [np.isfinite(fields[name]) for name in RESULT_FIELDS
             if name in fields]
    if not masks:
        raise ValueError("No solved field or finite result fields were found")
    return np.logical_and.reduce(masks)


def load_npz(path):
    with np.load(path) as data:
        coords = np.asarray(data["coords"], dtype=float)
        tets = np.asarray(data["tets"], dtype=np.int64)
        fields = {
            name: np.asarray(data[name])
            for name in (*RESULT_FIELDS, "solved", "theta")
            if name in data.files and len(data[name]) == len(coords)
        }
    return coords, tets, fields


def load_vtu(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    grid = reader.GetOutput()
    if grid is None or grid.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not read any points from {path}")
    cell_types = vtk_to_numpy(grid.GetCellTypes())
    if not np.all(cell_types == 10):
        raise ValueError("Only VTK tetrahedral cells (type 10) are supported")
    coords = vtk_to_numpy(grid.GetPoints().GetData()).astype(float)
    tets = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(-1, 4)
    point_data = grid.GetPointData()
    fields = {
        point_data.GetArrayName(i): vtk_to_numpy(point_data.GetArray(i)).copy()
        for i in range(point_data.GetNumberOfArrays())
    }
    return coords, tets.astype(np.int64), fields


def repair(input_path, output_path):
    if input_path.lower().endswith(".npz"):
        coords, tets, fields = load_npz(input_path)
    else:
        coords, tets, fields = load_vtu(input_path)
    if any(len(values) != len(coords) for values in fields.values()):
        raise ValueError("A point-data field has a different length than points")

    solved = infer_solved(fields)
    write_vtu(output_path, coords, tets, fields, solved=solved)
    print(f"Wrote {output_path}: {np.count_nonzero(solved):,} points, "
          f"{np.count_nonzero(np.all(solved[tets], axis=1)):,} tetrahedra")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="solver .vtu or .npz file")
    parser.add_argument("-o", "--output", help="output VTU path")
    args = parser.parse_args()
    output = args.output or os.path.splitext(args.input)[0] + "_repaired.vtu"
    try:
        repair(args.input, output)
    except (OSError, KeyError, ValueError) as error:
        print(f"repair_vtu: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
