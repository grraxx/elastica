import gmsh, math

Lx, Ly, Lz = 200.0, 200.0, 210.0   # mm
TARGET_NODES = 5_000_000
TOL = 0.02          # accept within 2% of target
MAX_ITERS = 6

V = Lx * Ly * Lz
lc = (V / TARGET_NODES) ** (1.0 / 3.0)   # ~1.189 mm

gmsh.initialize()
gmsh.model.add("cuboid")

gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)
gmsh.model.occ.synchronize()

# uniform sizing, no curvature/boundary refinement
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)    # 2D: Frontal-Delaunay
gmsh.option.setNumber("Mesh.Algorithm3D", 10) # 3D: HXT (fast, parallel)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("General.NumThreads", 0)  # 0 = all cores

for it in range(MAX_ITERS):
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.model.mesh.clear()
    gmsh.model.mesh.generate(3)

    n = len(gmsh.model.mesh.getNodes()[0])
    print(f"iter {it}: lc = {lc:.4f} mm -> {n:,} nodes")

    if abs(n - TARGET_NODES) / TARGET_NODES < TOL:
        break
    # nodes scale as lc^-3
    lc *= (n / TARGET_NODES) ** (1.0 / 3.0)

_, tets, _ = gmsh.model.mesh.getElements(dim=3)
print(f"final: {n:,} nodes, {len(tets[0]):,} tetrahedra, lc = {lc:.4f} mm")

gmsh.write("cuboid_5M.msh")
gmsh.finalize()