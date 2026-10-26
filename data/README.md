# data/

Simulation output lives here but is **not tracked by git** (the production run is about 7.8 GB;
several files exceed GitHub's 100 MB limit). Each run is a folder holding its `config.json`;
`python solver/main.py data/<run>/config.json` writes the outputs next to it.

## `production/` (2026-10-12)

d 0.70–0.99, φ₁, φ₂ ∈ [−π/2, π/2], step 0.01: 30 × 315 × 315 = 2,976,750 nodes, all solved.
On 2026-10-19 a promotion pass gave 1,571 bistable nodes the bent-up state (see the main README).

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
| `stability_full.log` | stability re-test of every node (of the promoted map) |
| `promotion.log`, `validation_promoted.log` | the promotion pass (2026-10-19: `main.py` resumed on the finished run) and its validation |
| `pre_promotion/` | the results, VTU, `post/` and stability log from before the promotion pass |
| `unsolve_unstable.py`, `watch_checkpoints.sh` | one-off tools used during the run |

## `production_nn/`, `production_iso/`, `production_shell/`: training data for the surrogate

The three refined-sampling datasets of `solver/refine.py` on `production/` (see the main README):
the cells scheme (2026-10-14), the iso-energy scheme (2026-10-18) and the energy-shell scheme
(2026-10-21). The first two were rebuilt on 2026-10-19 on the promoted map; their earlier versions
are in `pre_promotion/`. `production_iso` and `production_shell` reuse the branches of
`production_nn` (`"branches_npz"` in their configs).

| File | Contents |
|---|---|
| `config.json` | the refinement's configuration (`"scheme"`: cells, iso or shell) |
| `branches_grid.npz`, `branches.log` | both branches on the grid, and the front past the folds (`production_nn` only) |
| `iso_plan.npz`, `shell_plan.npz`, `plan.log` | the planned points of the iso and shell schemes |
| `sample_parts/` | one solved chunk per file; a rerun of `sample` resumes from them |
| `refined_samples.npz`, `sample.log` | the merged refined solves (with θ(s)) |
| `nn_dataset.npz`, `export.log` | the training dataset: one row per (point, branch), with its split |
| `nn_dataset.density.npz` | the explorer's density-map cache |
| `check.log`, `post/` | `check` report and the energy histograms |
| `pre_promotion/` | the same, from before the promotion pass (`production_nn`, `production_iso`) |

## `production_shell/mlp/`, `production_shell/mlp_no_fx/` (2026-10-23)

The two trained surrogates of both branches (`surrogate/train_mlp.py`, see the main README) on
`production_shell/nn_dataset.npz`: `mlp` with fx in the loss (the one to use), `mlp_no_fx` without.

| File | Contents |
|---|---|
| `model.pt` | configuration and weights of the field and existence networks (2.7 MB); load with `train_mlp.load_model` |
| `metrics.json` | test and training errors per target, branch and energy band (median and p99 relative error, ‖e‖ / ‖y‖, median absolute error); existence accuracy; symmetry of the learned branches |
| `train.log` | the run's arguments, loss terms every 5 epochs, final test errors |
| `parity.png`, `error_vs_energy.png` | `surrogate/plot_mlp.py` output (in `mlp/` only) |

## `nn_test/`, `nn_test_iso/`, `nn_test_shell/`

Small trial runs of the three refinement schemes on a corner of `production/` (d 0.95–0.99,
|φ| ≤ 0.3), the same files as above.

## `runs/`

The explorer's run console: one folder per started program, `<date>_<time>_<program>[_<label>]`,
holding `config.json` (for solve and refine), `job.json` (the parameters and the exact commands),
`run.log` and `status.json`.
