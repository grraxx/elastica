"""Generate an irregular tetrahedral mesh over (d, phi1, phi2)."""

import numpy as np
import gmsh


def _bounds(values, name):
    if len(values) != 2:
        raise ValueError(f"{name} must be [min, max]")
    lo, hi = map(float, values)
    if lo >= hi:
        raise ValueError(f"{name} must have min < max")
    return lo, hi


def build_mesh(d_bounds, phi1_bounds, phi2_bounds, out_msh="param_mesh.msh",
               out_npz="param_mesh.npz", mesh_size=0.05, algorithm=1,
               scale_parameters=True, verbose=False, diagonal_half=False):
    """Build an unstructured tetrahedral mesh inside the parameter box."""
    d_lo, d_hi = _bounds(d_bounds, "d_bounds")
    p1_lo, p1_hi = _bounds(phi1_bounds, "phi1_bounds")
    p2_lo, p2_hi = _bounds(phi2_bounds, "phi2_bounds")
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")

    physical_bounds = np.array([
        [d_lo, d_hi], [p1_lo, p1_hi], [p2_lo, p2_hi]
    ], dtype=float)
    parameter_offsets = physical_bounds[:, 0]
    parameter_spans = physical_bounds[:, 1] - parameter_offsets
    mesh_bounds = (np.zeros(3), np.ones(3)) if scale_parameters else (
        physical_bounds[:, 0], physical_bounds[:, 1]
    )
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm))
        gmsh.model.add("param_space")
        if diagonal_half:
            if not (np.isclose(p1_lo, -p1_hi)
                    and np.isclose(p2_lo, -p2_hi)
                    and np.isclose(p1_hi, p2_hi)):
                raise ValueError(
                    "diagonal_half requires matching symmetric phi bounds"
                )
            x0, x1 = mesh_bounds[0][0], mesh_bounds[1][0]
            y0, y1 = mesh_bounds[0][1], mesh_bounds[1][1]
            z0, z1 = mesh_bounds[0][2], mesh_bounds[1][2]
            points = [
                gmsh.model.occ.addPoint(x0, y1, z0, mesh_size),
                gmsh.model.occ.addPoint(x0, y1, z1, mesh_size),
                gmsh.model.occ.addPoint(x0, y0, z1, mesh_size),
            ]
            lines = [
                gmsh.model.occ.addLine(points[0], points[1]),
                gmsh.model.occ.addLine(points[1], points[2]),
                gmsh.model.occ.addLine(points[2], points[0]),
            ]
            wire = gmsh.model.occ.addCurveLoop(lines)
            surface = gmsh.model.occ.addPlaneSurface([wire])
            gmsh.model.occ.extrude([(2, surface)], x1 - x0, 0, 0)
        else:
            gmsh.model.occ.addBox(
                *mesh_bounds[0], *(mesh_bounds[1] - mesh_bounds[0])
            )
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), float(mesh_size))
        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        node_tags = np.asarray(node_tags, dtype=np.int64)[order]
        mesh_coords = np.asarray(node_coords, dtype=float).reshape(-1, 3)[order]

        tets = []
        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
        for elem_type, nodes in zip(elem_types, elem_node_tags):
            if elem_type == 4:
                raw = np.asarray(nodes, dtype=np.int64).reshape(-1, 4)
                tets.append(np.searchsorted(node_tags, raw))
        if not tets:
            raise RuntimeError("No linear tetrahedra found in generated mesh")
        tets = np.concatenate(tets).astype(np.int64, copy=False)
        gmsh.write(out_msh)
    finally:
        gmsh.finalize()

    coords = mesh_coords.copy()
    if scale_parameters:
        coords = parameter_offsets + mesh_coords * parameter_spans

    np.savez_compressed(
        out_npz,
        coords=coords,
        mesh_coords=mesh_coords,
        tets=tets,
        d_bounds=np.array([d_lo, d_hi]),
        phi1_bounds=np.array([p1_lo, p1_hi]),
        phi2_bounds=np.array([p2_lo, p2_hi]),
        mesh_bounds=np.asarray(mesh_bounds),
        parameter_offsets=parameter_offsets,
        parameter_spans=parameter_spans,
        parameters_scaled=np.array(bool(scale_parameters)),
        mesh_size=np.array(float(mesh_size)),
    )
    print(f"[mesh_gen] wrote {out_msh} and {out_npz} "
          f"({coords.shape[0]:,} nodes, {tets.shape[0]:,} tets)")
    return out_npz


if __name__ == "__main__":
    build_mesh((0.94, 0.95), (-0.8, 0.8), (-0.8, 0.8),
               out_msh="demo_mesh.msh", out_npz="demo_mesh.npz",
               mesh_size=0.1)
