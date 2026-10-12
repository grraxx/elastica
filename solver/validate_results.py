"""
validate_results.py
Check a checkpoint (solution_space_live.npz) or results file against
results that are known independently of the front solver.

1. coverage     every solved node's mirror is solved, fields are finite
2. mirror       fx, energy and p equal on mirror pairs, fy opposite, theta
                mirrored: theta~(s) = -theta(L - s)
3. analytic     on phi1 = phi2 = 0 the solution is the exact first clamped
                mode (solver.first_clamped_mode): fx = 16 K(m)^2 and the
                energy, |p| of the elliptic-function solution
4. bvp          an independent collocation solve (scipy solve_bvp, 4 states
                and the 2 forces as unknown parameters, tol 1e-8) started from
                a random sample of stored solutions: it must stay on the same
                branch, and fx, fy, both clamp moments and the energy must
                agree to discretisation error (P1)
5. branches     grid edges between solved nodes, with the solver's jump rule
                on q = p / a(d): snaps (p changes sign with a jump), smooth
                zero crossings, and same-sign jumps. In a d layer these must
                be rare (a few occur where p passes zero steeply beside the
                diagonal phi1 = phi2); across d layers they are only reported,
                since at large clamp angles p does not scale like a(d) and a
                coarse d step can exceed tol on one branch. No node may sit
                on the unstable S-shaped family: |q| < tol with an energy over
                twice that of its lowest in-layer neighbour
6. stability    every node on a snap edge and a random sample of the rest
                must be stable: the constrained second variation has no
                negative direction (stability.py)

usage: python validate_results.py [results.npz] [--samples 200] [--json report.json]
theta is read row by row from the archive, so a 2.4 GB checkpoint can be
checked while the solver that writes it is still running.
Exits with status 1 if a check fails.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from scipy.special import ellipe, ellipj, ellipk

from branch import first_mode_amplitude, first_mode_projection, is_jump
from stability import unstable_directions

# Pass thresholds. The mirror is exact except on the diagonal phi1 + phi2 = 0,
# whose nodes are their own mirror and symmetric to solver tolerance. P1 with h = 0.01 has relative errors of order fx h^2 / 12,
# about 1e-3 for the forces met here; the bounds leave room for the stiff
# corners of the domain (large fx, sharp boundary layers).
TOL = {"mirror": 1e-6, "analytic_fx": 2e-3, "analytic_energy": 2e-3,
       "analytic_p": 2e-3, "bvp_rel": 2e-2, "bvp_converged": 0.98,
       "same_sign_jump_edges": 1e-3}


def exact_first_mode(d, n=200001):
    """theta, fx, energy and |p| of the exact first clamped mode at (d, 0, 0)."""
    m = brentq(lambda m: 2.0 * ellipe(m) / ellipk(m) - 1.0 - d, 0.0, 1.0 - 1e-15, xtol=1e-15)
    K = ellipk(m)
    s = np.linspace(0.0, 1.0, n)
    sn, cn, dn, _ = ellipj(4.0 * K * s, m)
    theta = 2.0 * np.arcsin(np.sqrt(m) * sn)
    # d theta / du = 2 sqrt(m) cn dn / sqrt(1 - m sn^2) = 2 sqrt(m) cn, u = 4 K s
    dtheta = 8.0 * K * np.sqrt(m) * cn
    energy = 0.5 * np.trapezoid(dtheta ** 2, s)
    return 16.0 * K ** 2, energy, abs(float(first_mode_projection(theta, s)))


def bvp_check(d, phi1, phi2, theta, s, fx, fy):
    """Solve the elastica BVP by collocation from the stored solution.
    Returns (fx, fy, M(0), M(L), energy, p) or None if it fails."""
    kappa = np.gradient(theta, s)
    x = np.concatenate(([0.0], np.cumsum(0.5 * (np.cos(theta[1:]) + np.cos(theta[:-1])) * np.diff(s))))
    y = np.concatenate(([0.0], np.cumsum(0.5 * (np.sin(theta[1:]) + np.sin(theta[:-1])) * np.diff(s))))

    def fun(_, u, q):
        return np.vstack((u[1], -q[0] * np.sin(u[0]) + q[1] * np.cos(u[0]),
                          np.cos(u[0]), np.sin(u[0])))

    def bc(a, b, q):
        return np.array((a[0] - phi1, b[0] - phi2, a[2], a[3], b[2] - d, b[3]))

    sol = solve_bvp(fun, bc, s, np.vstack((theta, kappa, x, y)), p=[fx, fy],
                    tol=1e-8, max_nodes=200000)
    if not sol.success:
        return None
    fine = np.linspace(0.0, 1.0, 20001)
    u = sol.sol(fine)
    return (sol.p[0], sol.p[1], u[1, 0], u[1, -1], 0.5 * np.trapezoid(u[1] ** 2, fine),
            float(first_mode_projection(u[0], fine)))


def read_rows(path, name, rows):
    """Rows of one array in an .npz without loading the whole array."""
    rows = np.asarray(rows)
    order = np.argsort(rows)
    out = None
    with zipfile.ZipFile(path) as archive, archive.open(f"{name}.npy") as file:
        version = np.lib.format.read_magic(file)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, _, dtype = read_header(file)
        start = file.tell()
        row_bytes = dtype.itemsize * int(np.prod(shape[1:]))
        out = np.empty((len(rows),) + tuple(shape[1:]), dtype=float)
        for position in order:          # ascending: one forward pass
            file.seek(start + int(rows[position]) * row_bytes)
            out[position] = np.frombuffer(file.read(row_bytes), dtype=dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("results", nargs="?", default="solution_space_live.npz")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--tol", type=float, default=0.17, help="the solver's jump_tol, on q")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stability-samples", type=int, default=20000,
                    help="random nodes tested for stability, besides all snap-edge nodes")
    ap.add_argument("--json", help="write the report here")
    args = ap.parse_args()

    with np.load(args.results) as r:
        shape = tuple(int(n) for n in r["shape"])
        d_vals, p1_vals, p2_vals = (r[k] for k in ("d_vals", "phi1_vals", "phi2_vals"))
        s = r["beam_x"]
        fields = {k: r[k].reshape(shape) for k in ("fx", "fy", "moment", "energy", "p")}
        solved = r["solved"].reshape(shape) > 0.5
        has_theta = "theta" in r.files
    report, failed = {"file": str(args.results)}, []

    def check(name, ok, **values):
        report[name] = {"pass": bool(ok), **values}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
              + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                          for k, v in values.items()))
        if not ok:
            failed.append(name)

    n = shape[1]
    mirror = (slice(None), slice(None, None, -1), slice(None, None, -1))   # (i, j, k) -> (i, n-1-k, n-1-j)

    def mirrored(a):
        return np.swapaxes(a, 1, 2)[mirror]

    moment_right = mirrored(fields["moment"])

    # 1. Coverage
    finite = np.all([np.isfinite(f[solved]).all() for f in fields.values()])
    check("coverage", finite and np.array_equal(solved, mirrored(solved)),
          solved=f"{solved.sum():,}/{solved.size:,} ({solved.mean():.2%})",
          layers_complete=int(solved.all(axis=(1, 2)).sum()))

    # 2. Mirror consistency
    both = solved & mirrored(solved)
    err = max(np.abs(fields["fx"] - mirrored(fields["fx"]))[both].max(initial=0),
              np.abs(fields["fy"] + mirrored(fields["fy"]))[both].max(initial=0),
              np.abs(fields["energy"] - mirrored(fields["energy"]))[both].max(initial=0),
              np.abs(fields["p"] - mirrored(fields["p"]))[both].max(initial=0))
    if has_theta:
        idx = np.argwhere(both)[:: max(1, both.sum() // 2000)]
        mi = np.column_stack((idx[:, 0], n - 1 - idx[:, 2], n - 1 - idx[:, 1]))
        flat = np.ravel_multi_index(idx.T, shape), np.ravel_multi_index(mi.T, shape)
        rows = read_rows(args.results, "theta", np.concatenate(flat))
        half = len(idx)
        err = max(err, np.abs(rows[:half] + rows[half:][:, ::-1]).max(initial=0))
    check("mirror", err <= TOL["mirror"], max_error=float(err))

    # 3. Exact first mode on phi1 = phi2 = 0
    j0 = np.flatnonzero(np.isclose(p1_vals, 0.0, atol=1e-12))
    k0 = np.flatnonzero(np.isclose(p2_vals, 0.0, atol=1e-12))
    if len(j0) and len(k0):
        worst = {"fx": 0.0, "energy": 0.0, "p": 0.0}
        rows = 0
        for i, d in enumerate(d_vals):
            if not solved[i, j0[0], k0[0]]:
                continue
            fx, energy, p = exact_first_mode(d)
            got = {k: fields[k][i, j0[0], k0[0]] for k in ("fx", "energy", "p")}
            worst["fx"] = max(worst["fx"], abs(got["fx"] / fx - 1))
            worst["energy"] = max(worst["energy"], abs(got["energy"] / energy - 1))
            worst["p"] = max(worst["p"], abs(abs(got["p"]) / p - 1))
            rows += 1
        check("analytic", rows > 0 and worst["fx"] <= TOL["analytic_fx"]
              and worst["energy"] <= TOL["analytic_energy"] and worst["p"] <= TOL["analytic_p"],
              layers=rows, fx_rel=worst["fx"], energy_rel=worst["energy"], p_rel=worst["p"])

    # 4. Independent BVP solves on a random sample
    if has_theta and args.samples > 0:
        rng = np.random.default_rng(args.seed)
        candidates = np.argwhere(solved)
        picks = candidates[rng.choice(len(candidates), min(args.samples, len(candidates)), replace=False)]
        sample_theta = read_rows(args.results, "theta", np.ravel_multi_index(picks.T, shape))
        errors, converged, branch_kept, worst_at = [], 0, 0, None
        for (i, j, k), theta_ijk in zip(picks, sample_theta):
            stored = {k_: fields[k_][i, j, k] for k_ in ("fx", "fy", "moment", "energy", "p")}
            out = bvp_check(d_vals[i], p1_vals[j], p2_vals[k], theta_ijk, s,
                            stored["fx"], stored["fy"])
            if out is None:
                continue
            converged += 1
            fx, fy, m0, m1, energy, p = out
            branch_kept += (np.sign(p) == np.sign(stored["p"])
                            or abs(p - stored["p"]) < args.tol * first_mode_amplitude(d_vals[i]))
            scale_f = max(abs(fx), abs(fy), 1.0)
            scale_m = max(abs(m0), abs(m1), 1.0)
            e = max(abs(stored["fx"] - fx) / scale_f, abs(stored["fy"] - fy) / scale_f,
                    abs(stored["moment"] - m0) / scale_m,
                    abs(moment_right[i, j, k] - m1) / scale_m,
                    abs(stored["energy"] / energy - 1))
            errors.append(e)
            if worst_at is None or e >= max(errors):
                worst_at = (round(float(d_vals[i]), 4), round(float(p1_vals[j]), 4),
                            round(float(p2_vals[k]), 4), round(float(fx), 2))
        errors = np.array(errors) if errors else np.array([np.inf])
        check("bvp", converged >= TOL["bvp_converged"] * len(picks)
              and branch_kept == converged and errors.max() <= TOL["bvp_rel"],
              sampled=len(picks), converged=converged, same_branch=int(branch_kept),
              median_rel=float(np.median(errors)), max_rel=float(errors.max()),
              worst_at_d_phi1_phi2_fx=str(worst_at))

    # 5. Snaps, smooth crossings and same-sign jumps on grid edges
    P = fields["p"]
    Q = P / first_mode_amplitude(d_vals)[:, None, None]
    snaps = smooth = same_sign = edges = 0
    in_layer_same_sign = in_layer_edges = 0
    near_snap = np.zeros(shape, dtype=bool)
    for axis in range(3):
        a = np.take(P, range(shape[axis] - 1), axis=axis)
        b = np.take(P, range(1, shape[axis]), axis=axis)
        ok = np.take(solved, range(shape[axis] - 1), axis=axis) & np.take(solved, range(1, shape[axis]), axis=axis)
        change = (np.sign(a) != np.sign(b)) & ok
        jump = is_jump(np.take(Q, range(shape[axis] - 1), axis=axis),
                       np.take(Q, range(1, shape[axis]), axis=axis), args.tol) & ok
        edges += int(ok.sum()); snaps += int((change & jump).sum())
        lower = [slice(None)] * 3; lower[axis] = slice(0, shape[axis] - 1)
        upper = [slice(None)] * 3; upper[axis] = slice(1, shape[axis])
        near_snap[tuple(lower)] |= change & jump
        near_snap[tuple(upper)] |= change & jump
        smooth += int((change & ~jump).sum()); same_sign += int((jump & ~change).sum())
        if axis > 0:
            in_layer_edges += int(ok.sum()); in_layer_same_sign += int((jump & ~change).sum())
    # A landing on the unstable S-shaped family has p near 0 and an energy far
    # above every neighbour's in its d layer (9.6 against 1.2 when it
    # happened). Across layers the energy itself grows steeply as d -> 1.
    E = np.where(solved, fields["energy"], np.inf)
    lowest_neighbour = np.full(shape, np.inf)
    for axis in (1, 2):
        for shift in (1, -1):
            rolled = np.roll(E, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            rolled[tuple(edge)] = np.inf
            lowest_neighbour = np.minimum(lowest_neighbour, rolled)
    s_family = int((solved & (np.abs(Q) < args.tol) & (E > 2.0 * lowest_neighbour)).sum())
    check("branches", in_layer_same_sign <= TOL["same_sign_jump_edges"] * max(in_layer_edges, 1)
          and s_family == 0,
          edges=edges, snap_edges=snaps, smooth_crossings=smooth,
          same_sign_jumps_in_layer=in_layer_same_sign,
          same_sign_jumps_across_d=same_sign - in_layer_same_sign, s_family_suspects=s_family)

    # 6. Stability where it can be lost (both sides of every snap), and on a
    # random sample elsewhere
    if has_theta:
        rng = np.random.default_rng(args.seed + 1)
        flat_solved = np.flatnonzero(solved)
        sample = rng.choice(flat_solved, min(args.stability_samples, len(flat_solved)),
                            replace=False)
        nodes = np.union1d(np.flatnonzero(near_snap), sample)
        fx_flat, fy_flat = fields["fx"].ravel(), fields["fy"].ravel()
        unstable = []
        for start in range(0, len(nodes), 20000):
            chunk = nodes[start:start + 20000]
            for node, theta_n in zip(chunk, read_rows(args.results, "theta", chunk)):
                if unstable_directions(theta_n, s, fx_flat[node], fy_flat[node]):
                    unstable.append(node)
        where = [tuple(round(float(a[i]), 4) for a, i in
                       zip((d_vals, p1_vals, p2_vals), np.unravel_index(node, shape)))
                 for node in unstable[:5]]
        check("stability", not unstable, checked=len(nodes),
              snap_edge_nodes=int(near_snap.sum()), unstable=len(unstable),
              first_unstable_d_phi1_phi2=str(where))

    # Per-layer summary
    up = [(float(d), float((P[i][solved[i]] > 0).mean()) if solved[i].any() else None,
           float(solved[i].mean())) for i, d in enumerate(d_vals)]
    report["layers"] = up
    report["failed"] = failed
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    print("FAILED: " + ", ".join(failed) if failed else "All checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
