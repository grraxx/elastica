"""Solution-space explorer: a local web app for probing the solver results.

usage: python postprocessing/web/server.py [data/production] [--dataset data/production_nn/nn_dataset.npz ...]
                                          [--port 8765] [--host 127.0.0.1] [--allow-remote-runs]

Then open the printed address in a browser (from Windows too: WSL forwards localhost,
unless a Windows program already uses that port; then pick another with --port).
The argument is a run folder or its results NPZ. theta(s) is memory-mapped from
solution_space_live.npz (stored uncompressed); without it, the first start writes a
2.4 GB <results>.theta.npy cache next to the results.
With the training datasets of solver/refine.py, the density fields (density*) count their
points per grid box, one set per dataset: by default the cells scheme (data/production_nn),
the iso-energy scheme (data/production_iso) and the energy-shell scheme
(data/production_shell), told apart by the "scheme" of the config.json beside each. The cells scheme keeps the plain names
(density, density_up, ...);
any other scheme gets its name as a suffix (density_iso, density_up_shell, ...). Each point
goes to its nearest node. A box cut by the edge of the domain is scaled to a whole one for the points inside cells; a point on the node itself
counts once. The counts are cached in <dataset>.density.npz.

API (all GET):
    /api/meta                         grid axes, fields and their ranges
    /api/layer?i=&field=              one d layer of a field (or density), float32, [phi1][phi2]
    /api/snap?i=                      snap and smooth-crossing edge midpoints of a layer
    /api/node?i=&j=&k=                every field at a node, plus theta, x, y, curvature
    /api/column?j=&k=&field=          a field along d at fixed (phi1, phi2)
    /api/iso?field=&level=&color=&stride=
                                      level set of a field over the jump-free tetrahedra
    /api/snapsurface?stride=          the p = 0 surface, snap triangles first
        Both return binary little-endian: uint32 nv, nt, n_snap, 0; float32 xyz[nv][3]
        (phi1, phi2, d); float32 color[nv] (the color field, or d); uint32 tri[nt][3].

Run console (console.py; each run in a new folder under data/runs):
    GET  /api/console                 the programs, their parameters and defaults, source runs
    GET  /api/runs                    every run with its state
    GET  /api/runs/job?id=            a run's parameters and commands
    GET  /api/runs/log?id=&offset=    run.log from a byte offset, and the state
    POST /api/runs                    {program, label, params}: start a run
    POST /api/runs/stop, /resume      {id}
    POSTs need the header X-Elastica-Console: 1 and a JSON body (so another web page cannot
    start runs), and are refused unless the server listens on localhost (or --allow-remote-runs).
"""
import argparse
import json
import struct
import sys
import zipfile
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "solver"))
from branch import is_jump  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import console  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
JUMP_TOL = 0.17
FIELDS = {          # name: (label, colour scale)
    "q": ("q = p / a(d), branch", "branch"),
    "energy": ("Bending energy U", "log"),
    "fx": ("Axial force fx (> 0 compression)", "symlog"),
    "fy": ("Transverse force fy", "symlog"),
    "moment": ("Clamp moment M1 = θ′(0)", "symlog"),
    "moment_right": ("Clamp moment M2 = θ′(L)", "symlog"),
    "p": ("First-mode projection p", "branch"),
}
DENSITY = {         # name: label; points per 0.01-cube around a node, from the training dataset
    "density": "Dataset states per grid box, both branches",
    "density_up": "Dataset states per grid box, bent-up branch",
    "density_down": "Dataset states per grid box, bent-down branch",
    "density_grid": "Grid states per box, before refinement",
    "density_neg": "Negatives per grid box (no state there)",
}
SCHEMES = {"cells": "cells scheme", "iso": "iso-energy scheme",     # refine.py's "scheme"
           "shell": "energy-shell scheme"}
ISO_FIELDS = ("energy", "fx", "fy", "moment", "moment_right")
# The six tetrahedra of a grid cell, as (di, dj, dk) corner offsets; the same split
# of each cube along its 000-111 diagonal as solver/mesh_generator.py.
_C = {"000": (0, 0, 0), "100": (1, 0, 0), "010": (0, 1, 0), "001": (0, 0, 1),
      "110": (1, 1, 0), "101": (1, 0, 1), "011": (0, 1, 1), "111": (1, 1, 1)}
CELL_TETS = np.array([[_C[c] for c in tet] for tet in (
    ("000", "100", "110", "111"), ("000", "110", "010", "111"), ("000", "010", "011", "111"),
    ("000", "011", "001", "111"), ("000", "001", "101", "111"), ("000", "101", "100", "111"))])
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def theta_source(results):
    """theta (nodes x beam nodes) as a read-only memory map."""
    for path in (results, results.with_name("solution_space_live.npz")):
        if not path.exists():
            continue
        with zipfile.ZipFile(path) as archive:
            if "theta.npy" not in archive.namelist():
                continue
            info = archive.getinfo("theta.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            continue
        with open(path, "rb") as file:
            file.seek(info.header_offset)
            name_len, extra_len = struct.unpack("<HH", file.read(30)[26:30])
            file.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(file)
            shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                     else np.lib.format.read_array_header_2_0)(file)
            offset = file.tell()
        return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape,
                         order="F" if fortran else "C")
    cache = results.with_suffix(".theta.npy")
    if not cache.exists():
        print(f"Writing {cache} (one time, about a minute)...")
        with np.load(results) as r:
            np.save(cache, r["theta"])
    return np.load(cache, mmap_mode="r")


def box_volume(shape):
    """Share of a whole grid box around each node that lies in the domain."""
    volume = np.ones(shape)
    for axis, n in enumerate(shape):
        index = [slice(None)] * len(shape)
        for end in (0, n - 1):
            index[axis] = end
            volume[tuple(index)] *= 0.5
    return volume


def dataset_density(path, axes):
    """Points of the dataset per grid box (DENSITY), as (n0, n1, n2) float32 arrays."""
    cache = path.with_name(path.name.removesuffix(".npz") + ".density.npz")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        with np.load(cache) as c:
            if all(k in c.files for k in DENSITY):
                return {k: c[k] for k in DENSITY}
    print(f"Counting the points of {path} per grid box (one time)...", flush=True)
    shape = tuple(len(a) for a in axes)
    with np.load(path) as ds:
        steps = [(ds[c] - a[0]) / np.diff(a).mean() for c, a in zip(("d", "phi1", "phi2"), axes)]
        exists, branch, source = ds["exists"], ds["branch"], ds["source"]
    index = [np.clip(np.rint(x), 0, len(a) - 1).astype(np.int64) for x, a in zip(steps, axes)]
    node = np.ravel_multi_index(index, shape)
    on_node = np.all([np.abs(x - i) < 1e-6 for x, i in zip(steps, index)], axis=0)
    del steps, index
    volume = box_volume(shape).ravel()
    masks = {"density": exists, "density_up": exists & (branch > 0),
             "density_down": exists & (branch < 0), "density_grid": exists & (source <= 3),
             "density_neg": ~exists}
    # A node's own point stands for its (cut) box, so it counts in full there.
    weight = np.where(on_node, volume[node], 1.0)
    out = {k: (np.bincount(node[m], weight[m], minlength=volume.size) / volume)
           .reshape(shape).astype(np.float32) for k, m in masks.items()}
    np.savez(cache, **out)
    return out


def dataset_scheme(path):
    """The sampling scheme of a refine.py dataset: "scheme" in the config.json beside it."""
    try:
        return json.loads((path.parent / "config.json").read_text()).get("scheme", "cells")
    except (OSError, ValueError):
        return "cells"


class Data:
    def __init__(self, results, datasets=()):
        print(f"Loading {results} ...", flush=True)
        with np.load(results) as r:
            self.shape = tuple(int(n) for n in r["shape"])
            self.d, self.p1, self.p2 = r["d_vals"], r["phi1_vals"], r["phi2_vals"]
            self.s = r["beam_x"].astype(float)
            self.fields = {k: r[k].reshape(self.shape).astype(np.float32) for k in FIELDS}
            self.solve_order = r["solve_order"].reshape(self.shape)
            self.waited = r["waited"].reshape(self.shape)
        self.theta = theta_source(results)
        self.run = results.parent.name
        density, labels, self.datasets = {}, {}, []
        for path in datasets:
            if not path.exists():
                continue
            scheme = dataset_scheme(path)
            suffix = "" if scheme == "cells" else f"_{scheme}"
            if any(ds["suffix"] == suffix for ds in self.datasets):      # two of one scheme
                suffix += f"_{len(self.datasets)}"
            self.datasets.append({"name": path.parent.name, "scheme": scheme, "suffix": suffix,
                                  "label": SCHEMES.get(scheme, f"{scheme} scheme")})
            for k, values in dataset_density(path, (self.d, self.p1, self.p2)).items():
                density[k + suffix] = values
                labels[k + suffix] = f"{DENSITY[k]}, {SCHEMES.get(scheme, scheme + ' scheme')}"
        self.fields.update(density)
        self.meta = {
            "run": self.run, "shape": self.shape, "d": self.d.tolist(),
            "phi1": self.p1.tolist(), "phi2": self.p2.tolist(), "jump_tol": JUMP_TOL,
            "iso_fields": list(ISO_FIELDS),
            "fields": {k: {"label": label, "scale": scale,
                           "min": float(self.fields[k].min()), "max": float(self.fields[k].max())}
                       for k, (label, scale) in FIELDS.items()},
            "datasets": self.datasets,
        }
        for k, values in density.items():
            positive = values[values > 0]
            self.meta["fields"][k] = {"label": labels[k], "scale": "count",
                                      "min": float(positive.min()) if positive.size else 0.0,
                                      "max": float(values.max())}

    @lru_cache(maxsize=64)
    def snap(self, i):
        Q = self.fields["q"][i].astype(float)
        out = {"snap": [], "smooth": []}
        for a, b, axis in ((Q[:-1], Q[1:], 0), (Q[:, :-1], Q[:, 1:], 1)):
            change = np.sign(a) != np.sign(b)
            snap = change & is_jump(a, b, JUMP_TOL)
            for name, mask in (("snap", snap), ("smooth", change & ~snap)):
                j, k = np.nonzero(mask)
                x = 0.5 * (self.p1[j] + self.p1[j + 1]) if axis == 0 else self.p1[j]
                y = self.p2[k] if axis == 0 else 0.5 * (self.p2[k] + self.p2[k + 1])
                out[name] += np.round(np.column_stack((x, y)), 5).tolist()
        return json.dumps(out).encode()

    @lru_cache(maxsize=24)
    def iso(self, field, level, color, stride, snap_split=False):
        """Marching tetrahedra for field = level, on every stride-th phi node.

        Without snap_split, tetrahedra with a jump of q on an edge are dropped, so the
        level set ends at the snap surface (as in postprocess.py). With it, all are kept
        and the triangles from jump tetrahedra come first; their count is returned.
        """
        F = self.fields[field][:, ::stride, ::stride]
        Q = self.fields["q"][:, ::stride, ::stride]
        C = None if color == "d" else self.fields[color][:, ::stride, ::stride]
        p1, p2 = self.p1[::stride], self.p2[::stride]
        n0, n1, n2 = F.shape
        # Cells whose corners straddle the level.
        corners = [F[a:n0 - 1 + a, b:n1 - 1 + b, c:n2 - 1 + c] for a, b, c in _C.values()]
        lo, hi = np.minimum.reduce(corners), np.maximum.reduce(corners)
        ci, cj, ck = np.nonzero((hi > level) & (lo <= level))
        off = CELL_TETS.reshape(-1, 3)
        V = (((ci[:, None] + off[:, 0]) * n1 + cj[:, None] + off[:, 1]) * n2
             + ck[:, None] + off[:, 2]).reshape(-1, 4)              # (6 cells, 4) node ids
        Ff, Qf = F.ravel(), Q.ravel()
        above = Ff[V] > level
        count = above.sum(axis=1)
        keep = (count > 0) & (count < 4)
        V, above, count = V[keep], above[keep], count[keep]
        jump = np.zeros(len(V), dtype=bool)
        for a, b in TET_EDGES:
            jump |= is_jump(Qf[V[:, a]], Qf[V[:, b]], JUMP_TOL)
        if snap_split:
            order = np.argsort(~jump, kind="stable")
            V, above, count, jump = V[order], above[order], count[order], jump[order]
        else:
            V, above, count = V[~jump], above[~jump], count[~jump]
        # Above-level corners first; then each case is a fixed list of edges.
        order = np.argsort(~above, axis=1, kind="stable")
        V = np.take_along_axis(V, order, axis=1)
        tris, src = [], []
        for n, edge_lists in ((1, [((0, 1), (0, 2), (0, 3))]),
                              (3, [((3, 0), (3, 1), (3, 2))]),
                              (2, [((0, 2), (0, 3), (1, 3)), ((0, 2), (1, 3), (1, 2))])):
            rows = np.flatnonzero(count == n)
            for edges in edge_lists:
                tris.append(np.stack([V[rows][:, list(e)] for e in edges], axis=1))   # (t, 3, 2)
                src.append(rows)
        tris, src = np.concatenate(tris), np.concatenate(src)
        order = np.argsort(src, kind="stable")            # keep snap triangles first
        tris, src = tris[order], src[order]
        n_snap = int(jump[src].sum()) if snap_split else 0
        # Shared vertices: one per crossed grid edge.
        a, b = tris[..., 0].ravel(), tris[..., 1].ravel()
        key = np.minimum(a, b) * Ff.size + np.maximum(a, b)
        key, first, index = np.unique(key, return_index=True, return_inverse=True)
        a, b = a[first], b[first]
        fa, fb = Ff[a].astype(float), Ff[b].astype(float)
        t = (level - fa) / (fb - fa)

        def lerp(values):
            return values[a] + t * (values[b] - values[a])

        ia, ja, ka = np.unravel_index(a, F.shape)
        ib, jb, kb = np.unravel_index(b, F.shape)
        xyz = np.column_stack((p1[ja] + t * (p1[jb] - p1[ja]), p2[ka] + t * (p2[kb] - p2[ka]),
                               self.d[ia] + t * (self.d[ib] - self.d[ia])))
        col = xyz[:, 2] if C is None else lerp(C.ravel().astype(float))
        tri = index.reshape(-1, 3)
        # Orient every triangle towards increasing field (inside its tetrahedron), so
        # that averaged vertex normals are smooth.
        P = xyz[tri]
        normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        flip = np.einsum("ij,ij->i", normal, P.mean(axis=1) - self._lowside(V[src], Ff, level, F.shape, p1, p2)) < 0
        tri[flip] = tri[flip][:, ::-1]
        header = np.array([len(xyz), len(tri), n_snap, 0], dtype="<u4")
        return b"".join((header.tobytes(), xyz.astype("<f4").tobytes(), col.astype("<f4").tobytes(),
                         tri.astype("<u4").tobytes()))

    def _lowside(self, V, Ff, level, shape, p1, p2):
        """Mean position of each tetrahedron's corners at or below the level."""
        i, j, k = np.unravel_index(V, shape)
        P = np.stack((p1[j], p2[k], self.d[i]), axis=-1)
        below = (Ff[V] <= level)[..., None]
        return (P * below).sum(axis=1) / below.sum(axis=1)

    def node(self, i, j, k):
        index = int(np.ravel_multi_index((i, j, k), self.shape))
        th = np.asarray(self.theta[index], dtype=float)
        s = self.s
        ds = np.diff(s)
        x = np.concatenate(([0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)))
        y = np.concatenate(([0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)))
        values = {key: float(self.fields[key][i, j, k]) for key in FIELDS}
        return {"i": i, "j": j, "k": k, "node": index, "d": float(self.d[i]),
                "phi1": float(self.p1[j]), "phi2": float(self.p2[k]), **values,
                "solve_order": int(self.solve_order[i, j, k]), "waited": bool(self.waited[i, j, k] > 0.5),
                "s": np.round(s, 6).tolist(), "theta": np.round(th, 7).tolist(),
                "x": np.round(x, 7).tolist(), "y": np.round(y, 7).tolist(),
                "curvature": np.round(np.gradient(th, s), 6).tolist()}


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def make_handler(data, allow_runs=False, local_only=True):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC), **kwargs)

        def log_message(self, fmt, *args):         # quiet: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def end_headers(self):
            # Static files: revalidate on every load (a cheap 304), so an edited script is never
            # mixed with stale cached ones. The API sets its own no-store in send().
            if not self.path.startswith("/api/"):
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def send(self, body, content_type, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def error(self, message, status=HTTPStatus.BAD_REQUEST):
            self.send(json.dumps({"error": message}).encode(), "application/json", status)

        def do_GET(self):
            url = urlparse(self.path)
            if not url.path.startswith("/api/"):
                return super().do_GET()
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                n0, n1, n2 = data.shape

                def index(name, n):
                    value = int(q[name])
                    if not 0 <= value < n:
                        raise ValueError(f"{name} out of range")
                    return value

                if url.path == "/api/meta":
                    return self.send(json.dumps(data.meta).encode(), "application/json")
                if url.path == "/api/layer":
                    field = q.get("field", "q")
                    if field not in data.fields:
                        raise ValueError("unknown field")
                    body = np.ascontiguousarray(data.fields[field][index("i", n0)]).tobytes()
                    return self.send(body, "application/octet-stream")
                if url.path == "/api/snap":
                    return self.send(data.snap(index("i", n0)), "application/json")
                if url.path == "/api/node":
                    body = data.node(index("i", n0), index("j", n1), index("k", n2))
                    return self.send(json.dumps(body).encode(), "application/json")
                if url.path == "/api/column":
                    field = q.get("field", "q")
                    if field not in data.fields:
                        raise ValueError("unknown field")
                    values = data.fields[field][:, index("j", n1), index("k", n2)]
                    return self.send(json.dumps(values.astype(float).tolist()).encode(), "application/json")
                if url.path == "/api/console":
                    body = {"programs": console.schema(), "source_runs": console.source_runs(),
                            "runs_enabled": allow_runs, "runs_dir": "data/runs"}
                    return self.send(json.dumps(body).encode(), "application/json")
                if url.path == "/api/runs":
                    return self.send(json.dumps(console.runs()).encode(), "application/json")
                if url.path == "/api/runs/job":
                    return self.send(json.dumps(console.job(q.get("id", ""))).encode(), "application/json")
                if url.path == "/api/runs/log":
                    body = console.log(q.get("id", ""), int(q.get("offset", 0)))
                    return self.send(json.dumps(body).encode(), "application/json")
                if url.path in ("/api/iso", "/api/snapsurface"):
                    stride = int(q.get("stride", 1))
                    if stride not in (1, 2, 3, 4):
                        raise ValueError("stride must be 1 to 4")
                    if url.path == "/api/snapsurface":
                        body = data.iso("q", 0.0, "d", stride, snap_split=True)
                    else:
                        field, color, level = q.get("field", "energy"), q.get("color", "d"), float(q["level"])
                        if field not in ISO_FIELDS or (color != "d" and color not in FIELDS):
                            raise ValueError("unknown field")
                        if not np.isfinite(level):
                            raise ValueError("level must be finite")
                        body = data.iso(field, level, color, stride)
                    return self.send(body, "application/octet-stream")
                return self.error("unknown endpoint", HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as err:
                return self.error(str(err))

        def do_POST(self):
            url = urlparse(self.path)
            if not url.path.startswith("/api/runs"):
                return self.error("unknown endpoint", HTTPStatus.NOT_FOUND)
            if not allow_runs:
                return self.error("runs are disabled: the server is not on localhost "
                                  "(start it with --allow-remote-runs to allow them)", HTTPStatus.FORBIDDEN)
            host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
            if local_only and host not in LOOPBACK:          # another site behind a localhost name
                return self.error("runs must be started from localhost", HTTPStatus.FORBIDDEN)
            if self.headers.get("X-Elastica-Console") != "1" or \
                    not self.headers.get("Content-Type", "").startswith("application/json"):
                return self.error("missing console header", HTTPStatus.FORBIDDEN)
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 65536:
                return self.error("bad body size")
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("a JSON object is needed")
                if url.path == "/api/runs":
                    run_id = console.start(body.get("program"), body.get("params", {}), body.get("label", ""))
                    return self.send(json.dumps({"id": run_id}).encode(), "application/json")
                if url.path == "/api/runs/stop":
                    return self.send(json.dumps({"state": console.stop(body.get("id"))}).encode(),
                                     "application/json")
                if url.path == "/api/runs/resume":
                    console.resume(body.get("id"))
                    return self.send(b'{"ok": true}', "application/json")
                return self.error("unknown endpoint", HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, OSError) as err:      # ParamError is a ValueError
                return self.error(str(err))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", type=Path, default=REPO / "data" / "production",
                    help="run folder or results NPZ (default: data/production)")
    ap.add_argument("--dataset", type=Path, action="append",
                    help="training dataset of solver/refine.py, for the density maps; repeat for several "
                         "(default: data/production_nn, production_iso and production_shell; absent ones are "
                         "skipped)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines too")
    ap.add_argument("--allow-remote-runs", action="store_true",
                    help="let the run console start programs although the server is not on localhost")
    args = ap.parse_args()
    results = args.run / "solution_space_results.npz" if args.run.is_dir() else args.run
    datasets = args.dataset or [REPO / "data" / name / "nn_dataset.npz"
                                for name in ("production_nn", "production_iso", "production_shell")]
    data = Data(results, datasets)
    try:
        allow_runs = args.host in LOOPBACK or args.allow_remote_runs
        server = ThreadingHTTPServer((args.host, args.port),
                                     make_handler(data, allow_runs, local_only=args.host in LOOPBACK))
    except OSError as error:
        raise SystemExit(f"Port {args.port} is in use ({error.strerror}); try --port {args.port + 1}")
    print(f"Explorer for {results}\nOpen http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}"
          f":{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
