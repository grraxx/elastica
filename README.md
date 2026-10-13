# Clamped elastica: mapping the snap surface

Numerical continuation of the planar, inextensible, clamped–clamped elastica over the
three-parameter space **(d, φ₁, φ₂)**: end-to-end distance and the two clamp angles. Every
grid point is one FEniCSx/PETSc Newton solve. The solver grows the solution from a single
exact cold start by warm starts. Each solution is classified as **bent up** or **bent down**
by its projection on the first buckling mode, and the **snap surface** is where that sign
flips with a jump.

![Snap surface over (φ₁, φ₂, d), coloured by d](presentation/figures/snap3d.png)

## Repository layout

```
.
├── solver/                    the continuation solver, its config and the result checks
├── postprocessing/            postprocess.py (ParaView VTU, surfaces, figures); web/ (solution explorer)
├── presentation/
│   ├── elastica_snap_surface.pptx
│   ├── figures/               the deck's plots, written by scripts/make_figures.py
│   └── scripts/make_figures.py
├── notebooks/                 single_point_solve.ipynb: one solve, step by step
├── data/production/           the production run (not in git, see data/README.md)
└── environment.yml            conda environment (FEniCSx 0.11, PETSc, PyVista, ParaView)
```

## Quick start

```bash
conda env create -f environment.yml          # creates "elastica-env"
conda activate elastica-env

mkdir -p data/my_run && cp solver/config.json data/my_run/
python solver/main.py data/my_run/config.json                     # solve; Ctrl-C saves, rerun resumes
python solver/validate_results.py data/my_run/solution_space_results.npz
python postprocessing/postprocess.py data/my_run/solution_space_results.npz   # writes data/my_run/post/
python presentation/scripts/make_figures.py data/my_run                       # deck figures
```

All outputs go next to the config file. For a test that runs in about 20 s, shrink the box in
the copied config, for example d 0.88–0.92, φ ±0.8, steps 0.02 / 0.04 / 0.04. On a grid that
coarse the `branches` check in `validate_results.py` can fail: its bound on same-sign jumps is
set for Δφ = 0.01.

## Governing equations

Minimise the bending energy subject to the end position and clamp angles (L = 1, EI = 1):

$$
\Pi[\theta]=\tfrac12\int_0^L \theta'(s)^2\,ds,\qquad
\int_0^L\cos\theta\,ds=d,\quad \int_0^L\sin\theta\,ds=0,\quad
\theta(0)=\varphi_1,\quad \theta(L)=\varphi_2 .
$$

With Lagrange multipliers $f_x, f_y$ (the end reaction forces), stationarity gives
$\theta'' + f_x\sin\theta - f_y\cos\theta = 0$. The solver assembles the mixed weak form

$$
\int_0^L\!\left(\theta' v' - f_x\sin\theta\,v + f_y\cos\theta\,v\right)ds = 0,\qquad
\int_0^L(\cos\theta-d)\,q_x\,ds=0,\qquad \int_0^L\sin\theta\,q_y\,ds=0,
$$

with θ in P1 Lagrange and $f_x, f_y$ in a one-DOF "Real" space. The beam has 200 elements graded
towards both clamps (0.001 L there, 0.007 L mid-span): under high tension the rod turns within
boundary layers of width $1/\sqrt{|f_x|}$. Each point is a Newton solve (SNES, cubic
backtracking, MUMPS LU) on 203 unknowns. Outputs per point: θ(s), $f_x$, $f_y$, the clamp
moments $M_1=\theta'(0)$ and $M_2=\theta'(L)$ (end slope plus a half-element Taylor correction,
second-order accurate) and the energy $U$.

## Branch tracking

**Bent up or bent down.** The transverse shape y(s) = ∫₀ˢ sin θ is projected onto the first
clamped buckling mode w₁(s) = 1 − cos(2πs/L) (`solver/branch.py`):

$$
p=\frac{\int_0^L y\,w_1\,ds}{\lVert w_1\rVert},\qquad p>0:\ \text{bent up},\quad p<0:\ \text{bent down}.
$$

The second mode is antisymmetric and carries no up/down information. p is unchanged by the
mirror (φ₁, φ₂) → (−φ₂, −φ₁) and flips under (φ₁, φ₂) → (−φ₁, −φ₂).

**Jumps.** p shrinks like √(1 − d) as the rod straightens, so jumps are judged on q = p / a(d),
where a(d) is |p| of the exact first mode at (d, 0, 0). Between neighbours q jumps when
|Δq| > `jump_tol` (0.17, which is |Δp| ≈ 0.02 at d = 0.9) and |Δq| > ½ max |q|. A sign change is
one of two things:
- **Snap:** a sign change with a jump.
- **Smooth crossing:** a sign change without a jump, where the rod passes through an S-shape
  near φ₁ = φ₂.

Past a fold, Newton can also jump without a sign change, onto the unstable S-shaped family
(p ≈ 0, high energy). The solver rejects that jump too.

**Stability.** A jump does not catch every unstable result. On the anti-diagonal φ₁ = −φ₂ the
rod is symmetric (f_y = 0), and the bent-up branch loses stability at a symmetry-breaking
bifurcation rather than a fold, so Newton follows it with no jump in q. Every candidate is
therefore also tested (`solver/stability.py`): the constrained second variation
∫ v′² − (f_x cos θ + f_y sin θ) v², with v(0) = v(L) = 0 and ∫ v cos θ = ∫ v sin θ = 0, must have
no negative direction. The count comes from the inertia of the bordered tridiagonal matrix
(Haynsworth): O(N), about 0.2 ms per node.

## The solver (`solver/`)

- **Mesh.** A uniform grid over (d, φ₁, φ₂), each cell split into 6 tetrahedra. Only the half
  φ₁ + φ₂ ≥ 0 is solved; the mirror θ̃(s) = −θ(L − s) fills the other half, with the same f_x
  and p, the opposite f_y, and the clamp moments swapped.
- **One cold start.** The exact first clamped mode at φ₁ = φ₂ = 0 and the smallest d:
  θ = 2 asin(k sn(4Ks/L | m)), m = k², with d/L = 2E(m)/K(m) − 1. Ties go to the bent-up hump.
- **The front.** Nodes are solved outward in order of grid distance. Each node is warm-started
  from its solved grid neighbours, nearest first, with at most 50 Newton iterations per attempt.
  A result is rejected if q jumps or if it is unstable.
- **Going around a fold.** A node that cannot be reached without a jump is deferred, and retried
  when a new neighbour is solved. After `max_failures_per_branch` failures from one branch it
  waits for the other branch. A region cut off by a fold is therefore never entered by jumping
  across the snap surface: the front reaches it the long way round, through the smooth
  crossings, and solves it from the far side.
- **Last resort.** Only when the front stops altogether is the nearest deferred node
  cold-started: the exact mode at (d, 0, 0), continued in the clamp angles for both hump signs,
  keeping the lower energy.
- **Output.** Live NPZ checkpoints (also on Ctrl-C; written to a temporary file and renamed)
  and resume. The results NPZ and VTU hold f_x, f_y, `moment` (left clamp), `moment_right`,
  `energy`, `p`, `q`, `branch` = sign(p), `solve_order` and `waited`; with `save_theta`, the NPZ
  also holds θ (float32).

Files:

| File | Purpose |
|---|---|
| `main.py` | the continuation front; `python main.py [config.json]` |
| `solver.py` | FEniCSx Newton solver, exact first mode, cold start |
| `branch.py` | first-mode projection p, amplitude a(d), jump rule |
| `stability.py` | O(N) stability test of an equilibrium |
| `mesh_generator.py` | the (d, φ₁, φ₂) grid and its mirror map |
| `vtu.py`, `repair_vtu.py` | VTU writer; compact VTU of the solved points of a checkpoint |
| `validate_results.py` | checks a checkpoint or result against independent references |

`validate_results.py` checks:
- coverage, and that mirror pairs agree;
- φ₁ = φ₂ = 0 against the exact first mode (f_x = 16K², the energy and |p|);
- a random sample re-solved by scipy's collocation BVP solver (`solve_bvp`): f_x, f_y, both
  moments and the energy must agree, on the same branch;
- the branch structure: snaps, smooth crossings, rare same-sign jumps, no node on the unstable
  S-shaped family;
- stability of every node on a snap edge and of a random sample of the rest.

Key `config.json` fields:
- `d_bounds`, `phi1_bounds`, `phi2_bounds` and `grid_steps`: the box and its spacing. φ₁ and
  φ₂ must be symmetric, for the mirror.
- `beam_elements`, `beam_grading` (0 uniform, 0.8 production) and the SNES tolerances.
- `warm_start_max_it`, `jump_tol` (0.17) and `max_failures_per_branch` (3).
- `save_theta`, checkpoint intervals and the output file names.

The default box is the production one: d 0.70–0.99 and φ ±π/2, step 0.01, so
30 × 315 × 315 = 2.98 M nodes and 1.49 M solves.

## Post-processing

`postprocessing/postprocess.py results.npz` writes `post/` next to the results (about a minute):
- `elastica_solution_space.vtu`: every node and tetrahedron with p, q, branch, energy, f_x,
  f_y, |f|, both moments, the snap indicator, solve order and `waited`; open it in ParaView and
  scale z by about 10;
- `snap_surface.{vtp,stl}`, `snap_surface_jump.stl`: the p = 0 surface and its snap part;
- `iso_energy.vtp`, `iso_fx.vtp`, `iso_fy.vtp`, `iso_moment_{left,right}.vtp`: level sets over
  the jump-free tetrahedra, so they end at the snap surface;
- the PNGs used in the deck, and `summary.json` (field ranges, areas).

## Solution explorer (web)

```bash
python postprocessing/web/server.py            # data/production by default; another run: give its folder
```

Open http://localhost:8765. From Windows this works too, because WSL forwards localhost, as long as
no Windows program already uses the port. If the page says "not found", start the server with
another port, for example `--port 8790`.
- **Map:** one d layer coloured by q (branch), energy, f_x, f_y, M₁ or M₂, with the snap edges
  marked. Hover to read values.
- **Probe:** click a point to see the rod shape, θ(s), curvature θ′(s), the forces, energy and
  moments, and profiles of the field along φ₁, φ₂ and d through that point.
- **Compare:** Shift+click (or P) pins up to three points for comparison.
- **Keys:** arrow keys move the probe one grid step; PgUp/PgDn change d.
- **Export and share:** "Download θ(s) CSV" exports the rod. The URL keeps the view, so a link
  reopens the same point.
- **Iso-surfaces in 3D:** level sets of U, f_x, f_y, M₁ or M₂ over (φ₁, φ₂, d), computed by the
  server on demand (marching tetrahedra over the jump-free cells, as in `postprocess.py`, so
  they stop at the snap surface, drawn in red). Drag the level slider or type a level. "Keep"
  holds a surface so you can overlay several levels or fields. Colour the surfaces by d, by
  surface, or by another field; cut away d above or below the current layer, or half the box
  along the mirror plane φ₁ + φ₂ = 0. Hover a surface to read it; click to probe the nearest
  node, and the map, rod and profiles follow. The viewer loads three.js from cdn.jsdelivr.net,
  so the browser needs internet access.

θ is read from `solution_space_live.npz` without loading it. If that file is missing, the
first start writes a 2.4 GB `.theta.npy` cache next to the results.

## Production run (`data/production`, 2026-10-12)

- **Run:** every node solved (2,976,750) in about 4 h on one core (about 11 ms per solve).
  One extra cold start, at (0.99, 0.20, 0.20). 465 jumps and 17 unstable results rejected;
  1,155 nodes waited, all at d ≥ 0.97.
- **Validation (`validation_final.log`):** all checks pass. BVP on 300 samples: maximum
  relative error 0.27 %, median 0.014 %. Exact first mode: f_x within 1.3e-4. Mirror pairs
  within 3e-8. A stability re-test of all 2,976,750 nodes finds none unstable
  (`stability_full.log`).
- **Ranges:** f_x from −13,785 (tension, d = 0.99, |φ| = π/2) to +92; |f_y| ≤ 340; clamp
  moments within ±168; energy 0.10–142.
- **Branch share:** bent up on 70.0 % of the layer at d = 0.70, falling almost linearly to
  50.5 % at d = 0.99.
- **Snap surface:** 29,756 snap edges, area 2.62 in (φ₁, φ₂, d) units. In each d layer it is an
  L-shaped curve with its corner on the anti-diagonal φ₁ = −φ₂ and its ends on the diagonal
  φ₁ = φ₂; it closes in on the origin as d → 1. At φ₂ = 0 it sits at φ₁ ≈ −0.98 (d = 0.70),
  −0.57 (0.90) and −0.18 (0.99).

## Presentation

`presentation/elastica_snap_surface.pptx` (26 slides): the governing math, the snap surface and
the buckling-mode criterion, the continuation strategy (deferral, stability test, exact cold
start, mirror symmetry) and the production results. Every data plot comes from
`data/production`; `presentation/scripts/make_figures.py` regenerates them in
`presentation/figures/`. Equations are PowerPoint equations.

## Known issues

- **Near d → 1.** At d = 0.99 and φ = ±π/2 the tension reaches f_x ≈ −13,800. A uniform
  100-element beam is then off by up to 40 %; the graded 200-element beam agrees with the BVP
  check to 0.3 %. Check this again before raising `d_bounds` past 0.99.
- **Visit order decides the bistable region.** Where both branches exist, a node takes the
  branch whose front reaches it first. Starting bent up at the origin, the surface found is
  where the bent-up branch ends.
