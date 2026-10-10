"""Write tetrahedral point data as binary VTK XML (.vtu) for ParaView."""

import base64
import os

import numpy as np


CHUNK_BYTES = 3 << 22   # 12 MiB; a multiple of 3, so chunks base64-encode independently


def _write_binary(file, values, dtype, rows=None, finite=False):
    """Stream one inline binary DataArray body: a UInt64 byte count, then the
    data, base64-encoded as one stream. The UInt64 header keeps arrays over
    4 GiB (theta on fine grids) valid; chunking keeps memory use flat."""
    values = np.asarray(values)
    count = len(values) if rows is None else len(rows)
    row_bytes = int(np.prod(values.shape[1:], dtype=np.int64)) * np.dtype(dtype).itemsize
    pending = np.array([count * row_bytes], dtype="<u8").tobytes()
    step = max(1, CHUNK_BYTES // max(1, row_bytes))
    for start in range(0, count, step):
        block = (values[start:start + step] if rows is None
                 else values[rows[start:start + step]])
        block = np.ascontiguousarray(block, dtype=dtype)
        if finite:
            block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        pending += block.tobytes()
        cut = len(pending) - len(pending) % 3
        file.write(base64.b64encode(pending[:cut]))
        pending = pending[cut:]
    file.write(base64.b64encode(pending))


def write_vtu(path, coords, tets, fields, solved=None):
    """Write the mesh and point fields; if a solved mask is given, keep only
    solved points and the tets whose four corners are all solved."""
    tets = np.asarray(tets, dtype=np.int64)
    rows = None
    if solved is not None:
        solved = np.asarray(solved, dtype=bool)
        rows = np.flatnonzero(solved)
        index_map = np.full(len(coords), -1, dtype=np.int64)
        index_map[rows] = np.arange(len(rows))
        tets = index_map[tets[np.all(solved[tets], axis=1)]]
    num_points = len(coords) if rows is None else len(rows)

    temporary_vtu = f"{path}.tmp"
    with open(temporary_vtu, "wb") as file:
        def text(value):
            file.write(value.encode("ascii"))

        text('<?xml version="1.0"?>\n'
             '<VTKFile type="UnstructuredGrid" version="1.0" '
             'byte_order="LittleEndian" header_type="UInt64">\n'
             f'  <UnstructuredGrid><Piece NumberOfPoints="{num_points}" '
             f'NumberOfCells="{len(tets)}">\n'
             '      <PointData>\n')
        for name, values in fields.items():
            values = np.asarray(values)
            if values.ndim > 2:
                raise ValueError(f"VTU field {name} must be scalar or vector-valued")
            components = (f' NumberOfComponents="{values.shape[1]}"'
                          if values.ndim == 2 and values.shape[1] > 1 else "")
            text(f'        <DataArray type="Float64" Name="{name}"'
                 f'{components} format="binary">')
            _write_binary(file, values, "<f8", rows, finite=True)
            text('</DataArray>\n')
        text('      </PointData>\n'
             '      <Points><DataArray type="Float64" NumberOfComponents="3" format="binary">')
        _write_binary(file, coords, "<f8", rows)
        text('</DataArray></Points>\n'
             '      <Cells>\n'
             '        <DataArray type="Int64" Name="connectivity" format="binary">')
        _write_binary(file, tets, "<i8")
        text('</DataArray>\n'
             '        <DataArray type="Int64" Name="offsets" format="binary">')
        _write_binary(file, 4 * np.arange(1, len(tets) + 1), "<i8")
        text('</DataArray>\n'
             '        <DataArray type="UInt8" Name="types" format="binary">')
        _write_binary(file, np.full(len(tets), 10), "u1")
        text('</DataArray>\n'
             '      </Cells>\n'
             '  </Piece></UnstructuredGrid>\n'
             '</VTKFile>\n')
    os.replace(temporary_vtu, path)
