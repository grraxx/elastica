"""
train_mlp.py
An MLP surrogate of both branches on the energy-shell dataset: inputs
(d, phi1, phi2, branch), outputs the energy U, the clamp moments M1, M2 and the
probability that the branch exists there; the end forces come from U by
automatic differentiation.

Forces. The envelope theorem gives dU = -fx dd - M1 dphi1 + M2 dphi2, so

    fx = -dU/dd,

and rotating the whole rod by a leaves U unchanged while the free end moves by
d da across the chord, so -M1 + M2 = fy d:

    fy = (dU/dphi1 + dU/dphi2) / d.

(The dataset satisfies fy = (M2 - M1) / d to 0.07, and finite differences of U
on the grid match -fx, -M1, M2 to 1e-4 relative.)

Rows. Every row of the dataset: branch = +1 (bent up) or -1 (bent down) is an
input, so the network holds both states wherever both exist. Rows where the
branch exists (9.0 M) train U and the moments; all rows (11.4 M, the 2.3 M
negatives past the folds included) train the existence output. Nothing imposes
the reflection down(d, phi1, phi2) = S up(d, -phi1, -phi2), S = diag(1, -1, -1)
on (U, M1, M2): the network learns it from the data, and the report measures
how well. The split is the dataset's own, by symmetry orbit of cells within
each energy band, so no row has an image on the other side: train is "train"
(80 %), test is "validation" + "test" (20 %), used only for the report.

Network. Two MLPs on the features (d, phi1, phi2, log(1 - d), branch) scaled to
O(1), each `depth` SiLU layers of `width` (smooth, so the derivatives are). The
field network gives z with U = exp(z_U), M_i = sinh(z_Mi): log U and asinh M
make the loss relative where the targets are large (U up to 142, |M| up to 168)
and absolute near M = 0. The existence network gives a logit. Two networks, so
the sharp edge of existence at a fold costs the smooth fields nothing.

Loss, per batch, on the transformed scale, the field terms over existing rows:

    (z_U - log U)^2 + sum_i (z_Mi - asinh M_i)^2
      + w_moment sum_i (asinh(-+dU/dphi_i) - asinh M_i)^2       energy gradient vs moments
      + w_force (asinh(-dU/dd / f_scale) - asinh(fx / f_scale))^2
      + w_exist BCE(logit, exists)

w_force = 1 by default (Sobolev training on fx); with --w-force 0 fx is not seen
in training, and its test error measures the autodiff forces alone.

Usage:
    python surrogate/train_mlp.py data/production_shell/nn_dataset.npz [--out DIR]
Writes DIR/model.pt, DIR/metrics.json, DIR/train.log (default DIR: mlp/ next to the dataset).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

BANDS = (0.0, 1.0, 3.0, 10.0, math.inf)          # solver/refine.py's energy bands
TARGETS = ("energy", "moment", "moment_right", "fx", "fy")
FIELD_TERMS = ("energy", "moments", "grad_moments", "grad_force")


def mlp(n_in, width, depth, n_out):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(n_in, width), nn.SiLU()]
        n_in = width
    return nn.Sequential(*layers, nn.Linear(n_in, n_out))


class Surrogate(nn.Module):
    def __init__(self, width=256, depth=6, d_range=(0.70, 0.99)):
        super().__init__()
        lo, hi = d_range
        log_gap = np.log(1.0 - np.array([hi, lo]))
        # Affine maps of (d, phi1, phi2, log(1 - d), branch) onto [-1, 1]
        center = [(lo + hi) / 2, 0.0, 0.0, log_gap.mean(), 0.0]
        span = [(hi - lo) / 2, math.pi / 2, math.pi / 2, (log_gap[1] - log_gap[0]) / 2, 1.0]
        self.register_buffer("center", torch.tensor(np.array(center, dtype=np.float32)))
        self.register_buffer("span", torch.tensor(np.array(span, dtype=np.float32)))
        self.field = mlp(5, width, depth, 3)
        self.exist = mlp(5, width, depth, 1)
        self.config = {"width": width, "depth": depth, "d_range": list(d_range)}

    def features(self, x):
        return (torch.cat([x[:, :3], torch.log1p(-x[:, :1]), x[:, 3:]], dim=1) - self.center) / self.span

    def forward(self, x):
        """x = (d, phi1, phi2, branch), (n, 4) -> z = (log U, asinh M1, asinh M2), (n, 3),
        and the existence logit, (n,)."""
        f = self.features(x)
        return self.field(f), self.exist(f)[:, 0]


def evaluate(model, x, create_graph=False):
    """z, U, M1, M2 from the field network, the energy gradient dU/d(d, phi1, phi2)
    and the existence logit, from x (n, 4) that requires grad."""
    z, logit = model(x)
    log_u = z[:, 0]
    (grad_log_u,) = torch.autograd.grad(log_u.sum(), x, create_graph=create_graph)
    u = torch.exp(log_u)
    return z, u, torch.sinh(z[:, 1]), torch.sinh(z[:, 2]), u[:, None] * grad_log_u[:, :3], logit


def forces(x, grad_u):
    return -grad_u[:, 0], (grad_u[:, 1] + grad_u[:, 2]) / x[:, 0]


def predict(model, d, phi1, phi2, branch=1, batch=1 << 18):
    """Surrogate states at the points (d, phi1, phi2) on branch +1 (bent up) or -1
    (bent down), scalars or arrays: dict of numpy arrays energy, moment,
    moment_right (outputs), fx, fy, moment_ad, moment_right_ad (autodiff: -dU/dd,
    (dU/dphi1 + dU/dphi2) / d, -dU/dphi1, dU/dphi2) and exists_prob, the
    probability that the branch exists there. U and the rest are only meaningful
    where exists_prob > 0.5."""
    device = next(model.parameters()).device
    x_all = np.column_stack(np.broadcast_arrays(d, phi1, phi2, branch)).astype(np.float32)
    out = {k: [] for k in ("energy", "moment", "moment_right", "fx", "fy",
                           "moment_ad", "moment_right_ad", "exists_prob")}
    for start in range(0, len(x_all), batch):
        x = torch.tensor(x_all[start:start + batch], device=device, requires_grad=True)
        _, u, m1, m2, grad_u, logit = evaluate(model, x)
        fx, fy = forces(x, grad_u)
        for k, v in zip(out, (u, m1, m2, fx, fy, -grad_u[:, 1], grad_u[:, 2], torch.sigmoid(logit))):
            out[k].append(v.detach().cpu().numpy())
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


def load_model(path, device="cuda"):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = Surrogate(**saved["config"]).to(device)
    model.load_state_dict(saved["state"])
    return model.eval()


# ---------------------------------------------------------------- training

def load_rows(path):
    """Every row: x = (d, phi1, phi2, branch), targets y (NaN on negatives),
    exists, is_test, fold_cell."""
    with np.load(path) as z:
        x = np.column_stack([z[k] for k in ("d", "phi1", "phi2", "branch")]).astype(np.float64)
        y = np.column_stack([z[k] for k in TARGETS])
        return x, y, z["exists"], z["split"] != 0, z["fold_cell"]


def loss_terms(model, x, y, exists, f_scale):
    """The field terms over the rows where the branch exists, the existence term over
    all. The field network never sees the other rows: its output there is
    unconstrained, and an overflow of exp(z_U) would turn a masked loss into NaN."""
    xe = x[exists].detach().requires_grad_(True)
    ye = y[exists]
    z = model.field(model.features(xe))
    (grad_log_u,) = torch.autograd.grad(z[:, 0].sum(), xe, create_graph=True)
    grad_u = torch.exp(z[:, :1]) * grad_log_u[:, :3]
    logit = model.exist(model.features(x))[:, 0]
    a1, a2 = torch.asinh(ye[:, 1]), torch.asinh(ye[:, 2])
    return {
        "energy": ((z[:, 0] - torch.log(ye[:, 0])) ** 2).mean(),
        "moments": ((z[:, 1] - a1) ** 2 + (z[:, 2] - a2) ** 2).mean(),
        "grad_moments": ((torch.asinh(-grad_u[:, 1]) - a1) ** 2
                         + (torch.asinh(grad_u[:, 2]) - a2) ** 2).mean(),
        "grad_force": ((torch.asinh(-grad_u[:, 0] / f_scale)
                        - torch.asinh(ye[:, 3] / f_scale)) ** 2).mean(),
        "exists": nn.functional.binary_cross_entropy_with_logits(logit, exists.float()),
    }


def field_metrics(p, x, y):
    """Errors per target, branch and energy band: median and p99 relative error,
    relative L2 error ||e|| / ||y||, and median absolute error."""
    pairs = {"energy": (p["energy"], y[:, 0]),
             "moment (output)": (p["moment"], y[:, 1]),
             "moment_right (output)": (p["moment_right"], y[:, 2]),
             "moment (autodiff)": (p["moment_ad"], y[:, 1]),
             "moment_right (autodiff)": (p["moment_right_ad"], y[:, 2]),
             "fx (autodiff)": (p["fx"], y[:, 3]),
             "fy (autodiff)": (p["fy"], y[:, 4]),
             "fy (outputs, (M2 - M1) / d)": ((p["moment_right"] - p["moment"]) / x[:, 0], y[:, 4])}
    report = {}
    for name, (pred, true) in pairs.items():
        report[name] = {}
        for branch, rows in (("both", x[:, 3] != 0), ("up", x[:, 3] == 1), ("down", x[:, 3] == -1)):
            for lo, hi in [(0.0, math.inf), *zip(BANDS[:-1], BANDS[1:])]:
                band = rows & (y[:, 0] >= lo) & (y[:, 0] < hi)
                e, t = pred[band] - true[band], true[band]
                # Pointwise relative error, with a floor where the target crosses zero
                rel = np.abs(e) / np.maximum(np.abs(t), 1e-2 * np.sqrt(np.mean(t ** 2)))
                key = f"{branch} [{lo:g}, {hi:g})"
                report[name][key] = {
                    "rows": int(band.sum()), "median_rel": float(np.median(rel)),
                    "p99_rel": float(np.percentile(rel, 99)),
                    "rel_l2": float(np.linalg.norm(e) / np.linalg.norm(t)),
                    "median_abs": float(np.median(np.abs(e)))}
    return report


def existence_metrics(prob, exists, fold):
    """Share of rows classified right, overall, per class and in fold cells."""
    right = (prob > 0.5) == exists
    return {"rows": int(len(exists)), "accuracy": float(right.mean()),
            "exists: predicted absent": float(1 - right[exists].mean()),
            "absent: predicted to exist": float(1 - right[~exists].mean()),
            "fold-cell rows": int(fold.sum()), "fold-cell accuracy": float(right[fold].mean())}


def symmetry_metrics(model, x):
    """How far the learned bent-down branch is from the reflection of the learned
    bent-up one, at the points x (bent-down states, so up exists at their images):
    median and p99 of |down - S up| / |down| for U, M1 and fx, and the share of
    points where the two existence answers differ."""
    down = predict(model, x[:, 0], x[:, 1], x[:, 2], -1)
    up = predict(model, x[:, 0], -x[:, 1], -x[:, 2], 1)
    report = {}
    for key, sign in (("energy", 1), ("moment", -1), ("fx", 1)):
        rel = np.abs(down[key] - sign * up[key]) / np.maximum(np.abs(down[key]), 1e-2)
        report[key] = {"median_rel": float(np.median(rel)), "p99_rel": float(np.percentile(rel, 99))}
    report["existence differs"] = float(np.mean((down["exists_prob"] > 0.5) != (up["exists_prob"] > 0.5)))
    return report


def report_for(model, x, y, exists, fold):
    p = predict(model, x[:, 0], x[:, 1], x[:, 2], x[:, 3])
    return {"fields": field_metrics({k: v[exists] for k, v in p.items()}, x[exists], y[exists]),
            "existence": existence_metrics(p["exists_prob"], exists, fold)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-moment", type=float, default=1.0)
    ap.add_argument("--w-force", type=float, default=1.0)
    ap.add_argument("--w-exist", type=float, default=1.0)
    ap.add_argument("--f-scale", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out or Path(args.dataset).parent / "mlp")
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False     # derivatives want full fp32
    device = "cuda"
    x_np, y_np, exists_np, is_test, fold_np = load_rows(args.dataset)
    log(f"{args.dataset}: {len(x_np):,} rows ({exists_np.sum():,} states, both branches; "
        f"{(~exists_np).sum():,} negatives), train {np.count_nonzero(~is_test):,} "
        f"({np.mean(~is_test):.1%}), test {np.count_nonzero(is_test):,}")
    log(f"args: {vars(args)}")

    def tensors(rows):
        return (torch.tensor(x_np[rows], dtype=torch.float32, device=device),
                torch.tensor(np.nan_to_num(y_np[rows], nan=1.0), dtype=torch.float32, device=device),
                torch.tensor(exists_np[rows], device=device))

    x_train, y_train, e_train = tensors(~is_test)
    x_test, y_test, e_test = tensors(np.flatnonzero(is_test)[::20])

    model = Surrogate(args.width, args.depth,
                      (float(x_np[:, 0].min()), float(x_np[:, 0].max()))).to(device)
    with torch.no_grad():                           # start at the mean of the targets
        yy = y_train[e_train]
        model.field[-1].bias.copy_(torch.stack([torch.log(yy[:, 0]).mean(),
                                                torch.asinh(yy[:, 1]).mean(),
                                                torch.asinh(yy[:, 2]).mean()]))
        model.exist[-1].bias.fill_(math.log(e_train.float().mean() / (1 - e_train.float().mean())))
    n_train = len(x_train)
    steps = args.epochs * (n_train // args.batch)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.05, final_div_factor=1e3)
    weights = {"energy": 1.0, "moments": 1.0, "grad_moments": args.w_moment,
               "grad_force": args.w_force, "exists": args.w_exist}
    log(f"{sum(p.numel() for p in model.parameters()):,} parameters, {steps:,} steps")

    t0 = time.time()
    skipped = 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n_train, device=device)
        sums = {k: 0.0 for k in weights}
        n_batches = n_train // args.batch
        for b in range(n_batches):
            idx = order[b * args.batch:(b + 1) * args.batch]
            terms = loss_terms(model, x_train[idx], y_train[idx], e_train[idx], args.f_scale)
            loss = sum(weights[k] * v for k, v in terms.items() if weights[k] > 0)
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                opt.step()
            else:                                   # a safeguard: skip the step, keep the weights
                skipped += 1
            sched.step()
            for k, v in terms.items():
                sums[k] += v.item() if b % 20 == 0 else 0.0
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            test_terms = loss_terms(model, x_test, y_test, e_test, args.f_scale)
            n_logged = len(range(0, n_batches, 20))
            log(f"epoch {epoch:4d}  lr {sched.get_last_lr()[0]:.2e}  train "
                + " ".join(f"{k} {sums[k] / n_logged:.2e}" for k in weights)
                + "  | test(5%) " + " ".join(f"{k} {v.item():.2e}" for k, v in test_terms.items())
                + f"  [{time.time() - t0:.0f} s, {skipped} steps skipped]")

    torch.save({"config": model.config, "state": model.state_dict(), "args": vars(args)},
               out / "model.pt")
    model.eval()
    train_rows = np.flatnonzero(~is_test)[::4]
    down = is_test & exists_np & (x_np[:, 3] == -1)
    report = {"dataset": str(args.dataset), "args": vars(args),
              "rows": {"train": int(n_train), "test": int(is_test.sum())},
              "train_seconds": time.time() - t0,
              "test": report_for(model, x_np[is_test], y_np[is_test], exists_np[is_test],
                                 fold_np[is_test]),
              "train": report_for(model, x_np[train_rows], y_np[train_rows], exists_np[train_rows],
                                  fold_np[train_rows]),
              "symmetry (test)": symmetry_metrics(model, x_np[down][::10])}
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    log("Test errors (both branches, all energies):")
    for name, bands in report["test"]["fields"].items():
        a = bands["both [0, inf)"]
        log(f"  {name:28s} median rel {a['median_rel']:.2e}  p99 rel {a['p99_rel']:.2e}  "
            f"rel L2 {a['rel_l2']:.2e}  median abs {a['median_abs']:.2e}")
    log(f"Existence (test): {report['test']['existence']}")
    log(f"Symmetry (test): {report['symmetry (test)']}")
    log(f"Wrote {out / 'model.pt'}, {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
