"""Repair half-solved ASCII VTU or solver NPZ files for ParaView.

Examples:
    python repair_vtu.py solution_space_live.vtu
    python repair_vtu.py solution_space_live.npz --output solved.vtu
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np


RESULT_FIELDS = ("fx", "fy", "moment", "energy")


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def data_arrays(parent):
    return {
        element.attrib.get("Name", ""): element
        for element in parent.iter()
        if local_name(element.tag) == "DataArray"
        and element.attrib.get("Name")
    }


def ascii_values(element, dtype=float):
    text = element.text or ""
    return np.fromstring(text, sep=" ", dtype=dtype)


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
    data = np.load(path)
    coords = np.asarray(data["coords"], dtype=float)
    tets = np.asarray(data["tets"], dtype=np.int64)
    fields = {
        name: np.asarray(data[name])
        for name in (*RESULT_FIELDS, "solved")
        if name in data.files and np.asarray(data[name]).ndim == 1
    }
    return coords, tets, fields


def load_vtu(path):
    root = ET.parse(path).getroot()
    piece = next((element for element in root.iter()
                  if local_name(element.tag) == "Piece"), None)
    if piece is None:
        raise ValueError("VTU has no Piece element")
    point_data = next((element for element in piece.iter()
                       if local_name(element.tag) == "PointData"), None)
    points = next((element for element in piece.iter()
                   if local_name(element.tag) == "Points"), None)
    cells = next((element for element in piece.iter()
                  if local_name(element.tag) == "Cells"), None)
    if point_data is None or points is None or cells is None:
        raise ValueError("VTU is missing PointData, Points, or Cells")

    point_arrays = data_arrays(point_data)
    point_coordinate = next((element for element in points.iter()
                             if local_name(element.tag) == "DataArray"), None)
    cell_arrays = data_arrays(cells)
    if point_coordinate is None:
        raise ValueError("VTU Points has no DataArray")
    coordinates = ascii_values(point_coordinate)
    coords = coordinates.reshape(-1, 3)
    connectivity = ascii_values(cell_arrays["connectivity"], np.int64)
    offsets = ascii_values(cell_arrays["offsets"], np.int64)
    cell_types = ascii_values(cell_arrays["types"], np.uint8)
    if not offsets.size or offsets[-1] != connectivity.size:
        raise ValueError("VTU cell offsets are inconsistent with connectivity")

    starts = np.r_[0, offsets[:-1]]
    sizes = offsets - starts
    if not np.all(sizes == 4):
        raise ValueError("Only tetrahedral cells with four nodes are supported")
    tets = connectivity.reshape(-1, 4)
    fields = {name: ascii_values(element)
              for name, element in point_arrays.items()}
    if len(cell_types) != len(tets) or not np.all(cell_types == 10):
        raise ValueError("Only VTK tetrahedral cells (type 10) are supported")
    return coords, tets, fields


def write_vtu(path, coords, tets, fields):
    coords = np.asarray(coords, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    point_text = " ".join("%.17g %.17g %.17g" % tuple(point)
                          for point in coords)
    point_data = "\n".join(
        f'        <DataArray type="Float64" Name="{name}" format="ascii">'
        f'{" ".join(str(value) for value in np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))}'
        "</DataArray>"
        for name, values in fields.items()
    )
    connectivity = " ".join(str(value) for value in tets.ravel())
    offsets = " ".join(str(value) for value in 4 * np.arange(1, len(tets) + 1))
    types = " ".join(["10"] * len(tets))
    xml = f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
    <UnstructuredGrid>
        <Piece NumberOfPoints="{len(coords)}" NumberOfCells="{len(tets)}">
            <PointData>
{point_data}
            </PointData>
            <Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">{point_text}</DataArray></Points>
            <Cells>
                <DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>
                <DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>
                <DataArray type="UInt8" Name="types" format="ascii">{types}</DataArray>
            </Cells>
        </Piece>
    </UnstructuredGrid>
</VTKFile>
'''
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="ascii") as file:
        file.write(xml)
    os.replace(temporary, path)


def repair(input_path, output_path):
    if input_path.lower().endswith(".npz"):
        coords, tets, fields = load_npz(input_path)
    else:
        coords, tets, fields = load_vtu(input_path)
    if any(len(values) != len(coords) for values in fields.values()):
        raise ValueError("A point-data field has a different length than points")

    solved = infer_solved(fields)
    cell_mask = np.all(solved[tets], axis=1)
    point_indices = np.flatnonzero(solved)
    index_map = np.full(len(coords), -1, dtype=np.int64)
    index_map[point_indices] = np.arange(len(point_indices))
    compact_fields = {
        name: np.asarray(values)[point_indices]
        for name, values in fields.items()
    }
    write_vtu(output_path, coords[point_indices],
              index_map[tets[cell_mask]], compact_fields)
    print(f"Wrote {output_path}: {len(point_indices):,} points, "
          f"{int(np.count_nonzero(cell_mask)):,} tetrahedra")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="ASCII .vtu or solver .npz file")
    parser.add_argument("-o", "--output", help="output VTU path")
    args = parser.parse_args()
    output = args.output or os.path.splitext(args.input)[0] + "_repaired.vtu"
    try:
        repair(args.input, output)
    except (OSError, ET.ParseError, KeyError, ValueError) as error:
        print(f"repair_vtu: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
