"""Solution-space explorer: a local web app for probing the solver results.

usage: python postprocessing/web/server.py [data/production] [--port 8765] [--host 127.0.0.1]

Then open the printed address in a browser (from Windows too: WSL forwards localhost,
unless a Windows program already uses that port; then pick another with --port).
The argument is a run folder or its results NPZ. theta(s) is memory-mapped from
solution_space_live.npz (stored uncompressed); without it, the first start writes a
2.4 GB <results>.theta.npy cache next to the results.

API (all GET):
    /api/meta                         grid axes, fields and their ranges
    /api/layer?i=&field=              one d layer of a field, float32, [phi1][phi2]
    /api/snap?i=                      snap and smooth-crossing edge midpoints of a layer
    /api/node?i=&j=&k=                every field at a node, plus theta, x, y, curvature
    /api/column?j=&k=&field=          a field along d at fixed (phi1, phi2)
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


class Data:
    def __init__(self, results):
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
        self.meta = {
            "run": self.run, "shape": self.shape, "d": self.d.tolist(),
            "phi1": self.p1.tolist(), "phi2": self.p2.tolist(), "jump_tol": JUMP_TOL,
            "fields": {k: {"label": label, "scale": scale,
                           "min": float(self.fields[k].min()), "max": float(self.fields[k].max())}
                       for k, (label, scale) in FIELDS.items()},
        }

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


def make_handler(data):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC), **kwargs)

        def log_message(self, fmt, *args):         # quiet: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

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
                    if field not in FIELDS:
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
                    if field not in FIELDS:
                        raise ValueError("unknown field")
                    values = data.fields[field][:, index("j", n1), index("k", n2)]
                    return self.send(json.dumps(values.astype(float).tolist()).encode(), "application/json")
                return self.error("unknown endpoint", HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError) as err:
                return self.error(str(err))

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", type=Path, default=REPO / "data" / "production",
                    help="run folder or results NPZ (default: data/production)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines too")
    args = ap.parse_args()
    results = args.run / "solution_space_results.npz" if args.run.is_dir() else args.run
    data = Data(results)
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(data))
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
