"""Read the CURRENTv4 0.02-run VTU, compute the neighbour-MAC snap indicator, write it back out.

For every tet edge (a, b):  MAC = (θa·θb)² / (|θa|²|θb|²).  Nodal indicator = max over incident
edges of (1 - MAC); the isosurface 1 - MAC = 0.1 is the deck's MAC = 0.90 threshold.
Points are re-ordered to (phi1, phi2, d) so d is vertical in ParaView.
"""
import argparse
from pathlib import Path

import numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("src", nargs="?", default=REPO / "data/v4/sol2_0.02_snes_relaxed/solution_space_results.vtu",
                help="results VTU that stores the theta field")
ap.add_argument("-o", "--output", default=REPO / "data/snap_surface_3d/snap_mac_indicator.vtu")
args = ap.parse_args()
OUT = Path(args.output); OUT.parent.mkdir(parents=True, exist_ok=True)

r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(str(args.src)); r.Update(); g = r.GetOutput()
pd = g.GetPointData()
c = vtk_to_numpy(g.GetPoints().GetData()).copy()                 # (d, phi1, phi2)
th = vtk_to_numpy(pd.GetArray("theta")); U = vtk_to_numpy(pd.GetArray("energy"))
solved = vtk_to_numpy(pd.GetArray("solved")) > 0.5
tets = vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1, 4)

e = np.concatenate([tets[:, [i, j]] for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
e.sort(1); n = len(c); k = np.unique(e[:, 0] * n + e[:, 1]); E = np.column_stack((k // n, k % n))
E = E[solved[E[:, 0]] & solved[E[:, 1]]]
a, b = th[E[:, 0]], th[E[:, 1]]
mac = np.sum(a * b, 1) ** 2 / (np.sum(a * a, 1) * np.sum(b * b, 1))
jump = np.zeros(n); np.maximum.at(jump, E[:, 0], 1 - mac); np.maximum.at(jump, E[:, 1], 1 - mac)
dU = np.abs(U[E[:, 0]] - U[E[:, 1]])
print(f"{len(E):,} edges; MAC<0.9: {(mac < 0.9).sum():,}; of those with |dU|>0.5: {((mac < 0.9) & (dU > 0.5)).mean() / (mac < 0.9).mean():.1%}")

g.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(c[:, [1, 2, 0]]), deep=True))
for name, v in (("mac_jump", jump), ("d", c[:, 0])):
    arr = numpy_to_vtk(np.ascontiguousarray(v), deep=True); arr.SetName(name); pd.AddArray(arr)
pd.RemoveArray("theta")
w = vtk.vtkXMLUnstructuredGridWriter(); w.SetFileName(str(OUT)); w.SetInputData(g)
w.SetDataModeToAppended(); w.SetCompressorTypeToZLib(); w.Write()
print("wrote", OUT, n, "points")
