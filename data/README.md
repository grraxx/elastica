# data/

Simulation output lives here but is **not tracked by git** (about 1.9 GB; several files exceed
GitHub's 100 MB limit). Everything in this folder can be regenerated with the solvers in
`solvers/` and the scripts in `postprocessing/`.

| Folder | Produced by | Parameter box | Points | θ(s) stored | Notes |
|---|---|---|---|---|---|
| `v1/sol1/` | `solvers/v1` | d 0.81–0.99, φ ±π/2 | 40,960 (10 × 64 × 64) | no | 94 % of points converged |
| `v1/root/` | `solvers/v1` | from `solvers/v1/config.json` | – | – | parameter mesh only |
| `v2/sol1/` | copy of `v1/sol1` | d 0.81–0.99, φ ±π/2 | 40,960 | no | identical files to `v1/sol1` |
| `v2/root/` | `solvers/v2` | d 0.80–0.90, φ 0.01–1.5 | 11,250 (2 × 75 × 75) | no | |
| `v3/sol2/`, `v3/sol3/` | early `solvers/v3` | d 0.90–0.95, φ ±0.8 | 13,122 (2 × 81 × 81) | yes (101) | |
| `v3/sol4/` | early `solvers/v3` | d 0.90–0.95, φ ±0.8, Δφ 0.01 | 155,526 (6 × 161 × 161) | yes (101) | used for the 2D figures in the deck |
| `v3/run_d0.80-0.99_partial/` | `solvers/v3` | d 0.80–0.99, φ ±0.8 | 518,420 (20 × 161 × 161) | yes (101) | live checkpoint, ~4 % solved |
| `v3/script_grid/` | `solvers/v3/mesh.py` | current `solvers/v3/config.json` | – | – | uniform seed grid |
| `v4/sol1_0.01/` | `solvers/v4` | d 0.80–0.99, φ ±0.8 | 740,741 (irregular tets) | no | NPZ + VTU |
| `v4/sol2_0.02_snes_relaxed/` | `solvers/v4` | d 0.80–0.99, φ ±0.8 | 98,120 (irregular tets) | yes (51) | input to the 3D snap surface |
| `snap_surface_3d/` | `postprocessing/snap_surface_3d` | – | – | – | MAC-indicator VTU and ParaView state |
| `presentation_backups/` | – | – | – | – | earlier versions of the deck |

Each run folder typically holds `solution_space.msh`/`.npz` (the parameter mesh),
`solution_space_results.npz` (final results), `solution_space_live.*` (checkpoints) and,
for v4, `.vtu` files for ParaView.
