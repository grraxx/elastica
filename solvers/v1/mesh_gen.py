"""
mesh_gen.py

Builds a structured tetrahedral mesh over the 3D parameter box
    d    in [d_min, d_max]
    phi1 in [phi1_min, phi1_max]
    phi2 in [phi2_min, phi2_max]
using GMSH's transfinite meshing (exact node placement at every grid
step -- no Delaunay drift), with each hexahedral cell split into 6
tetrahedra (Kuhn triangulation), which is what you get automatically
from a transfinite volume when it is NOT recombined into hexes.

Why transfinite instead of a free Delaunay mesh: the user's bounds are
given as (min, max, step) triples, i.e. this is really a full-factorial
grid. Transfinite meshing is the GMSH-native way to force the mesher to
land exactly on that grid (one node per (i, j, k) grid index) rather than
approximating a target element size, which a free 3D Delaunay pass would
only do approximately and non-reproducibly across axes with very
different physical scales (d spans 0.29, phi1/phi2 each span pi).

Output: an .msh file, plus a compact .npz with:
    coords   (N, 3) array of (d, phi1, phi2) at each node
    ijk      (N, 3) array of the structured grid indices for each node
    tets     (M, 4) array of node indices forming each tetrahedron
    adjacency information is NOT stored here -- build it from `tets`
    (see continuation_solver.py: it's cheap and avoids duplicating data).

NOTE ON SCALE: the resolution requested in the prompt
    d:    (0.70, 0.99, 0.01)  -> 30 points
    phi1: (-pi/2, pi/2, 0.004) -> ~786 points
    phi2: (-pi/2, pi/2, 0.004) -> ~786 points
is a ~30 x 786 x 786 grid, i.e. ~18.5 million nodes. That's a legitimate
target for a production run, but it is too large to build/mesh/traverse
inside this interactive session (GMSH transfinite generation, node
extraction, and the tet->adjacency graph would all be multi-GB and
multi-minute-to-hour operations here). The code below is written for
that full resolution directly -- nothing about it is "demo-only" -- but
`build_mesh()` takes the same (min, max, step) triples as parameters, so
you can run it at a coarser resolution first to validate the pipeline
before committing to the full run. A coarse call is used in
`if __name__ == "__main__"` purely to keep this session responsive.
"""

import numpy as np
import gmsh


def grid_sizes(bounds):
    """(min, max, step) -> number of points along that axis (inclusive)."""
    lo, hi, step = bounds
    n = int(round((hi - lo) / step)) + 1
    return n


def build_mesh(d_bounds, phi1_bounds, phi2_bounds, out_msh="param_mesh.msh",
               out_npz="param_mesh.npz", verbose=False):
    """
    Build the transfinite tet mesh over the (d, phi1, phi2) box.

    d_bounds, phi1_bounds, phi2_bounds: (min, max, step) triples, exactly
    as specified in the problem statement.
    """
    nd = grid_sizes(d_bounds)
    n1 = grid_sizes(phi1_bounds)
    n2 = grid_sizes(phi2_bounds)
    total_nodes = nd * n1 * n2
    total_hex_cells = (nd - 1) * (n1 - 1) * (n2 - 1)
    print(f"[mesh_gen] grid: {nd} x {n1} x {n2} = {total_nodes:,} nodes, "
          f"{total_hex_cells:,} hex cells -> {6*total_hex_cells:,} tets")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("param_space")

    d_lo, d_hi, _ = d_bounds
    p1_lo, p1_hi, _ = phi1_bounds
    p2_lo, p2_hi, _ = phi2_bounds

    # Build the box with the OpenCASCADE kernel, then make it transfinite.
    vol_tag = gmsh.model.occ.addBox(
        d_lo, p1_lo, p2_lo,
        d_hi - d_lo, p1_hi - p1_lo, p2_hi - p2_lo,
    )
    gmsh.model.occ.synchronize()

    # Identify the 12 edges of the box and assign the transfinite point
    # count according to which axis (d, phi1, or phi2) each edge runs
    # along, by comparing endpoint coordinates.
    edges = gmsh.model.getEntities(dim=1)

    def axis_of(tag):
        (x0, y0, z0, x1, y1, z1) = gmsh.model.getBoundingBox(1, tag)
        dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
        return int(np.argmax([dx, dy, dz]))  # 0=d, 1=phi1, 2=phi2

    npts_by_axis = {0: nd, 1: n1, 2: n2}
    for (dim, tag) in edges:
        ax = axis_of(tag)
        gmsh.model.mesh.setTransfiniteCurve(tag, npts_by_axis[ax])

    for (dim, tag) in gmsh.model.getEntities(dim=2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        # deliberately do NOT call setRecombine here -- leaving surfaces
        # un-recombined is what makes the volume mesh come out as
        # tetrahedra (6 per hex cell) instead of hexahedra.

    gmsh.model.mesh.setTransfiniteVolume(vol_tag)

    gmsh.model.mesh.generate(3)

    # --- extract nodes ---
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    coords = node_coords.reshape(-1, 3)[order]  # (N,3) in (d, phi1, phi2)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # --- extract tetrahedra (element type 4 = 4-node tet) ---
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tets = None
    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype == 4:  # linear tetrahedron
            tets_raw = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)
            tets = np.vectorize(tag_to_idx.get)(tets_raw)
    if tets is None:
        raise RuntimeError("No linear tetrahedra found -- check that surfaces "
                            "were left un-recombined.")

    # Recover structured (i, j, k) indices for each node from its
    # coordinates (useful for picking a start node and for sanity checks).
    d_vals = np.round(np.linspace(d_lo, d_hi, nd), 10)
    p1_vals = np.round(np.linspace(p1_lo, p1_hi, n1), 10)
    p2_vals = np.round(np.linspace(p2_lo, p2_hi, n2), 10)
    d_idx = np.searchsorted(d_vals, np.round(coords[:, 0], 10))
    p1_idx = np.searchsorted(p1_vals, np.round(coords[:, 1], 10))
    p2_idx = np.searchsorted(p2_vals, np.round(coords[:, 2], 10))
    ijk = np.stack([d_idx, p1_idx, p2_idx], axis=1).astype(np.int32)

    gmsh.write(out_msh)
    gmsh.finalize()

    np.savez_compressed(
        out_npz,
        coords=coords.astype(np.float64),
        ijk=ijk,
        tets=tets.astype(np.int64),
        grid_shape=np.array([nd, n1, n2]),
        d_vals=d_vals, p1_vals=p1_vals, p2_vals=p2_vals,
    )
    print(f"[mesh_gen] wrote {out_msh} and {out_npz} "
          f"({coords.shape[0]:,} nodes, {tets.shape[0]:,} tets)")
    return out_npz


if __name__ == "__main__":
    # Small resolution here ONLY to keep this interactive demo fast.
    # For the actual target resolution from the prompt, call:
    #
    #   build_mesh((0.70, 0.99, 0.01), (-np.pi/2, np.pi/2, 0.004),
    #              (-np.pi/2, np.pi/2, 0.004))
    #
    # Same step sizes as the production target (0.01 in d, 0.004 in phi),
    # just over a small sub-window, so this demo is a fair test of how
    # continuation behaves at the ACTUAL requested resolution rather than
    # an artificially coarsened one.
    build_mesh(
        d_bounds=(0.97, 0.99, 0.01),           # 3 points
        phi1_bounds=(-0.04, 0.04, 0.004),      # 21 points
        phi2_bounds=(-0.04, 0.04, 0.004),      # 21 points
        out_msh="demo_mesh.msh",
        out_npz="demo_mesh.npz",
        verbose=False,
    )
