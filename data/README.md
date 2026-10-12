# data/

Simulation output lives here but is **not tracked by git** (the production run is about 7.8 GB;
several files exceed GitHub's 100 MB limit). Each run is a folder holding its `config.json`;
`python solver/main.py data/<run>/config.json` writes the outputs next to it.

## `production/` (2026-10-12)

d 0.70–0.99, φ₁, φ₂ ∈ [−π/2, π/2], step 0.01: 30 × 315 × 315 = 2,976,750 nodes, all solved.

| File | Contents |
|---|---|
| `config.json` | the run's configuration |
| `solution_space.npz` | the parameter mesh (coordinates, tetrahedra, mirror map) |
| `solution_space_results.npz` | final results, with θ(s) at 201 beam nodes (float32) |
| `solution_space_live.npz` | last checkpoint (uncompressed, same content); the solver resumes from it |
| `solution_space_results.vtu` | the solver's VTU of the results |
| `post/` | `postprocessing/postprocess.py` output: ParaView VTU, snap surface, level sets, PNGs, `summary.json` |
| `checkpoint_reports/` | validation reports written during the run |
| `run_part1.log`, `run.log` | solver logs: the run was stopped once at 1.77 M nodes and resumed |
| `validation_part1.log`, `validation.log`, `validation_final.log` | `validate_results.py` reports |
| `stability_full.log` | stability re-test of every node |
| `unsolve_unstable.py`, `watch_checkpoints.sh` | one-off tools used during the run |
