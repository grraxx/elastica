"""Run console for the explorer: the programs' parameters, and runs in their own folders.

Each run gets a new folder data/runs/<date>_<time>_<program>[_<label>], created with
exist_ok=False (a numbered suffix if the name is taken), so no run ever writes into another's
folder. It holds config.json (the program's config, for solve and refine), job.json (the
parameters and the exact commands) and, from run_job.py, run.log and status.json. The
runner is started in its own session, so a run outlives the server; Stop sends it SIGINT,
which it passes on (main.py then saves a checkpoint), and a second Stop kills the step.
Validation is a soft step: if a check fails, the run goes on and ends "finished" with a
warning, since a failed check is a finding about the results, not a broken run.
"""
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "runs"
RUNNER = Path(__file__).resolve().with_name("run_job.py")
SOLVER_DEFAULTS = REPO / "solver" / "config.json"
REFINE_DEFAULTS = REPO / "data" / "production_nn" / "config.json"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _p(key, label, kind, default, help="", **limits):
    return {"key": key, "label": label, "type": kind, "default": default, "help": help, **limits}


def _solver_defaults():
    with open(SOLVER_DEFAULTS) as f:
        return json.load(f)


def _refine_defaults():
    base = {"max_failures": 3, "density": {"u_ref": 3.0, "power": 2.0, "max_points": 64,
                                           "fold_min_points": 16},
            "seed": 0, "workers": max(1, (os.cpu_count() or 2) - 4), "chunk_points": 5000,
            "seconds_per_solve": 0.011, "split": {"validation": 0.1, "test": 0.1},
            "scheme": "cells", "iso": {"alpha": 1.0, "scale": 1.0, "level_step": 0.05,
                                       "min_spacing": 0.05, "project_steps": 4},
            "shell": {"alpha": 1.0, "scale": 1.0, "level_step": 0.05, "min_spacing": 0.05,
                      "oversample": 12, "project_steps": 4}}
    if REFINE_DEFAULTS.exists():
        with open(REFINE_DEFAULTS) as f:
            for k, v in json.load(f).items():
                base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def _get(tree, key):
    for part in key.split("."):
        tree = tree[part]
    return tree


def _set(tree, key, value):
    *parents, last = key.split(".")
    for part in parents:
        tree = tree.setdefault(part, {})
    tree[last] = value


def schema():
    """The programs, their parameter groups and defaults."""
    s, r = _solver_defaults(), _refine_defaults()
    half_pi = math.pi / 2
    solve = {
        "label": "Solve", "script": "solver/main.py",
        "about": "The continuation solve over the (d, φ₁, φ₂) grid, then optionally validation and "
                 "post-processing of its results. The production box takes about 4 h on one core.",
        "groups": [
            {"title": "Domain", "params": [
                _p("d_bounds", "d range", "range", s["d_bounds"], "end-to-end distance, 0 < d < 1",
                   min=0.01, max=0.999),
                _p("phi1_bounds", "φ₁ range", "range", s["phi1_bounds"], "left clamp angle (rad; π allowed, e.g. −π/2)",
                   min=-half_pi, max=half_pi, angle=True),
                _p("phi2_bounds", "φ₂ range", "range", s["phi2_bounds"], "right clamp angle (rad; π allowed, e.g. −π/2)",
                   min=-half_pi, max=half_pi, angle=True),
                _p("grid_steps.d", "Δd", "float", s["grid_steps"]["d"], "grid step in d", gt=0, max=0.5),
                _p("grid_steps.phi1", "Δφ₁", "float", s["grid_steps"]["phi1"], "grid step in φ₁",
                   gt=0, max=1.0),
                _p("grid_steps.phi2", "Δφ₂", "float", s["grid_steps"]["phi2"], "grid step in φ₂",
                   gt=0, max=1.0),
                _p("solve_phi_symmetry", "mirror symmetry", "bool", s["solve_phi_symmetry"],
                   "solve φ₁ + φ₂ ≥ 0 only; needs φ₁ and φ₂ ranges equal and symmetric"),
            ]},
            {"title": "Beam and Newton", "params": [
                _p("beam_elements", "beam elements", "int", s["beam_elements"], "P1 elements along the rod",
                   min=10, max=5000),
                _p("beam_grading", "beam grading", "float", s["beam_grading"],
                   "0 uniform; 0.8 production (finer at the clamps)", min=0.0, max=0.99),
                _p("snes_rtol", "SNES rtol", "float", s["snes_rtol"], "relative Newton tolerance",
                   gt=0, max=1e-2),
                _p("snes_atol", "SNES atol", "float", s["snes_atol"], "absolute Newton tolerance",
                   gt=0, max=1e-2),
                _p("warm_start_max_it", "max Newton its", "int", s["warm_start_max_it"],
                   "iterations per warm-start attempt", min=1, max=1000),
            ]},
            {"title": "Continuation", "params": [
                _p("jump_tol", "jump tolerance", "float", s["jump_tol"],
                   "largest |Δq| between neighbours that is not a snap", gt=0, max=2.0),
                _p("max_failures_per_branch", "failures per branch", "int", s["max_failures_per_branch"],
                   "attempts from one branch before a node waits for the other", min=1, max=100),
            ]},
            {"title": "Output", "params": [
                _p("save_theta", "save θ(s)", "bool", s["save_theta"], "store the rod shapes (needed by "
                   "refine and the explorer's rod view)"),
                _p("visualization_interval", "checkpoint every", "int", s["visualization_interval"],
                   "solves between live checkpoints; 0 only at the end", min=0, max=10**9),
                _p("progress_interval", "progress every", "int", s["progress_interval"],
                   "solves between progress lines", min=1, max=10**9),
                _p("verbose_solves", "verbose solves", "bool", s["verbose_solves"], "print every solve"),
                _p("output_npz", "mesh NPZ", "file", s["output_npz"]),
                _p("live_results_npz", "checkpoint NPZ", "file", s["live_results_npz"]),
                _p("results_npz", "results NPZ", "file", s.get("results_npz", "solution_space_results.npz")),
                _p("output_vtu", "results VTU", "file", s["output_vtu"]),
                _p("live_output_vtu", "checkpoint VTU", "file", s["live_output_vtu"], "empty: none",
                   optional=True),
            ]},
            {"title": "After the solve", "params": [
                _p("run_validate", "validate", "bool", True, "validate_results.py on the results",
                   cli=True),
                _p("validate_samples", "BVP samples", "int", 200, "nodes re-solved with scipy solve_bvp",
                   min=0, max=100000, cli=True),
                _p("validate_stability_samples", "stability samples", "int", 20000,
                   "random nodes tested for stability, besides all snap-edge nodes", min=0, max=10**8,
                   cli=True),
                _p("validate_seed", "seed", "int", 0, min=0, max=2**31 - 1, cli=True),
                _p("run_postprocess", "post-process", "bool", True,
                   "postprocess.py: VTU, surfaces, figures in post/", cli=True),
            ]},
        ],
    }
    refine = {
        "label": "Refine", "script": "solver/refine.py",
        "about": "Training data from a finished solve: both branches at every node, then extra points, "
                 "more where the energy is low: inside each grid cell (cells), or regularly spaced on "
                 "iso-energy surfaces so that the energy histogram follows C U^-alpha (iso). On the "
                 "production run, with 14 workers: branches about 15 min, then sampling about 25 min "
                 "(cells) or 30 min (iso).",
        "groups": [
            {"title": "Source and stages", "params": [
                _p("source_run", "solve run", "run", "data/production", "a finished solve (with θ saved)"),
                _p("stages", "stages", "stages", ["branches", "sample", "export", "check"],
                   "run in this order; each needs the ones before it", choices=["branches", "sample",
                                                                                "export", "check"], cli=True),
                _p("plan_only", "plan only", "bool", False, "branches, then sample --dry-run: counts, "
                   "time and the predicted histogram, no sampling solves", cli=True),
                _p("scheme", "scheme", "choice", r["scheme"], "cells: Sobol points per grid cell; "
                   "iso: points on iso-energy surfaces; shell: points through the shells between "
                   "energy levels", choices=["cells", "iso", "shell"]),
            ]},
            {"title": "Iso-energy surfaces (scheme iso)", "params": [
                _p("iso.alpha", "alpha", "float", r["iso"]["alpha"], "target rows per unit U: "
                   "C U^-alpha up to the grid's peak; 1: as many per decade of U", min=0, max=4),
                _p("iso.scale", "scale", "float", r["iso"]["scale"], "C over its least value, where "
                   "the power law meets the grid's peak", min=1, max=100),
                _p("iso.level_step", "level step", "float", r["iso"]["level_step"],
                   "relative gap between levels", gt=0, max=1),
                _p("iso.min_spacing", "min spacing", "float", r["iso"]["min_spacing"],
                   "closest points on a surface, in grid steps", gt=0, max=10),
                _p("iso.project_steps", "projections", "int", r["iso"]["project_steps"],
                   "re-solves along grad U onto the level; 0: none", min=0, max=20),
            ]},
            {"title": "Energy shells (scheme shell)", "params": [
                _p("shell.alpha", "alpha", "float", r["shell"]["alpha"], "target rows per unit U: "
                   "C U^-alpha up to the grid's peak; 1: as many per decade of U", min=0, max=4),
                _p("shell.scale", "scale", "float", r["shell"]["scale"], "C over its least value, where "
                   "the power law meets the grid's peak", min=1, max=100),
                _p("shell.level_step", "bin width", "float", r["shell"]["level_step"],
                   "relative width of the energy bins", gt=0, max=1),
                _p("shell.min_spacing", "min spacing", "float", r["shell"]["min_spacing"],
                   "closest points in a shell, in grid steps", gt=0, max=10),
                _p("shell.oversample", "oversample", "int", r["shell"]["oversample"],
                   "random candidates per kept point, before the Poisson-disk selection", min=2, max=100),
                _p("shell.project_steps", "projections", "int", r["shell"]["project_steps"],
                   "re-solves along grad U back into the bin; 0: none", min=0, max=20),
            ]},
            {"title": "Points per cell", "params": [
                _p("density.u_ref", "u_ref", "float", r["density"]["u_ref"], "n = (u_ref / U_min)^power",
                   gt=0, max=1000),
                _p("density.power", "power", "float", r["density"]["power"], gt=0, max=10),
                _p("density.max_points", "max points", "int", r["density"]["max_points"],
                   "per cell, the node included", min=1, max=4096),
                _p("density.fold_min_points", "fold-cell minimum", "int", r["density"]["fold_min_points"],
                   "cells where the branch holds at some corners only", min=1, max=4096),
                _p("max_failures", "tries per point", "int", r["max_failures"],
                   "corners tried, nearest first", min=1, max=8),
                _p("seed", "seed", "int", r["seed"], "Sobol scramble, shifts and split", min=0,
                   max=2**31 - 1),
            ]},
            {"title": "Workers and split", "params": [
                _p("workers", "workers", "int", r["workers"], f"processes; this machine has "
                   f"{os.cpu_count()} cores", min=1, max=max(1, os.cpu_count() or 1)),
                _p("chunk_points", "chunk size", "int", r["chunk_points"], "points per saved chunk",
                   min=100, max=10**6),
                _p("seconds_per_solve", "s per solve", "float", r["seconds_per_solve"],
                   "for the time estimate only", gt=0, max=10),
                _p("split.validation", "validation share", "float", r["split"]["validation"], min=0,
                   max=0.5),
                _p("split.test", "test share", "float", r["split"]["test"], min=0, max=0.5),
            ]},
        ],
    }
    validate = {
        "label": "Validate", "script": "solver/validate_results.py",
        "about": "Check a finished solve against independent references (exact first mode, scipy BVP, "
                 "branch structure, stability). The report goes to the new run folder.",
        "groups": [{"title": "Validation", "params": [
            _p("source_run", "solve run", "run", "data/production"),
            _p("samples", "BVP samples", "int", 200, min=0, max=100000, cli=True),
            _p("stability_samples", "stability samples", "int", 20000, min=0, max=10**8, cli=True),
            _p("seed", "seed", "int", 0, min=0, max=2**31 - 1, cli=True),
            _p("tol", "jump tolerance", "float", 0.17, "the solver's jump_tol, on q", gt=0, max=2, cli=True),
        ]}],
    }
    post = {
        "label": "Post-process", "script": "postprocessing/postprocess.py",
        "about": "ParaView VTU, snap and iso-surfaces, figures and summary.json of a finished solve, "
                 "written to post/ in the new run folder.",
        "groups": [{"title": "Post-processing", "params": [
            _p("source_run", "solve run", "run", "data/production"),
            _p("tol", "jump tolerance", "float", 0.17, "the solver's jump_tol, on q", gt=0, max=2, cli=True),
        ]}],
    }
    return {"solve": solve, "refine": refine, "validate": validate, "postprocess": post}


# ------------------------------------------------------------------ source runs
def _results_file(folder):
    cfg = folder / "config.json"
    name = "solution_space_results.npz"
    if cfg.exists():
        try:
            with open(cfg) as f:
                name = json.load(f).get("results_npz", name)
        except (OSError, ValueError):
            pass
    path = folder / name
    return path if path.exists() else None


def source_runs():
    """Folders holding a finished solve, as repo-relative paths, newest first."""
    out = []
    for parent in (REPO / "data", RUNS):
        if not parent.is_dir():
            continue
        for folder in parent.iterdir():
            if folder.is_dir() and folder != RUNS:
                results = _results_file(folder)
                if results is not None:
                    out.append((results.stat().st_mtime, str(folder.relative_to(REPO))))
    return [name for _, name in sorted(out, reverse=True)]


# ------------------------------------------------------------------ validation
class ParamError(ValueError):
    pass


def _check(p, value, runs):
    kind, label = p["type"], p["label"]

    def number(v, integer=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ParamError(f"{label}: a number is needed")
        if integer and v != int(v):
            raise ParamError(f"{label}: a whole number is needed")
        v = int(v) if integer else float(v)
        if "min" in p and v < p["min"]:
            raise ParamError(f"{label}: at least {p['min']}")
        if "gt" in p and v <= p["gt"]:
            raise ParamError(f"{label}: must be above {p['gt']}")
        if "max" in p and v > p["max"]:
            raise ParamError(f"{label}: at most {p['max']}")
        return v

    if kind == "float":
        return number(value)
    if kind == "int":
        return number(value, integer=True)
    if kind == "bool":
        if not isinstance(value, bool):
            raise ParamError(f"{label}: true or false")
        return value
    if kind == "range":
        if not isinstance(value, list) or len(value) != 2:
            raise ParamError(f"{label}: two numbers")
        lo, hi = number(value[0]), number(value[1])
        if not lo < hi:
            raise ParamError(f"{label}: the lower end must be below the upper")
        return [lo, hi]
    if kind == "file":
        if value == "" and p.get("optional"):
            return value
        if not isinstance(value, str) or not FILE_NAME.match(value):
            raise ParamError(f"{label}: a plain file name (letters, digits, _ . -)")
        return value
    if kind == "run":
        if value not in runs:
            raise ParamError(f"{label}: not a finished solve run")
        return value
    if kind == "stages":
        if not isinstance(value, list) or not value or any(v not in p["choices"] for v in value):
            raise ParamError(f"{label}: pick one or more of {', '.join(p['choices'])}")
        return [c for c in p["choices"] if c in value]          # always in pipeline order
    if kind == "choice":
        if value not in p["choices"]:
            raise ParamError(f"{label}: one of {', '.join(p['choices'])}")
        return value
    raise ParamError(f"{label}: unknown type")


def validate(program, params):
    """Checked parameters (defaults filled in) for a program, or ParamError."""
    programs = schema()
    if program not in programs:
        raise ParamError("unknown program")
    if not isinstance(params, dict):
        raise ParamError("parameters must be an object")
    known = {p["key"]: p for g in programs[program]["groups"] for p in g["params"]}
    unknown = set(params) - set(known)
    if unknown:
        raise ParamError(f"unknown parameters: {', '.join(sorted(unknown))}")
    runs = source_runs()
    out = {k: _check(p, params.get(k, p["default"]), runs) for k, p in known.items()}
    if program == "solve":
        lo, hi = out["d_bounds"]
        if not 0 < lo < hi < 1:
            raise ParamError("d range: 0 < d < 1")
        if out["solve_phi_symmetry"]:
            a, b = out["phi1_bounds"], out["phi2_bounds"]
            if not (math.isclose(a[0], -a[1]) and math.isclose(b[0], -b[1]) and math.isclose(a[1], b[1])
                    and math.isclose(out["grid_steps.phi1"], out["grid_steps.phi2"])):
                raise ParamError("mirror symmetry needs φ₁ and φ₂ ranges equal and symmetric about 0, "
                                 "with equal steps")
        nodes = grid_nodes(out)
        if nodes > 50_000_000:
            raise ParamError(f"{nodes:,} grid nodes: too many (at most 50 M)")
    if program == "refine" and out["split.validation"] + out["split.test"] >= 1:
        raise ParamError("validation and test shares must leave some training data")
    return out


def grid_nodes(params):
    n = 1
    for axis in ("d", "phi1", "phi2"):
        lo, hi = params[f"{axis}_bounds"]
        n *= int(round((hi - lo) / params[f"grid_steps.{axis}"])) + 1
    return n


# ------------------------------------------------------------------ runs
def _new_folder(program, label):
    RUNS.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label.strip())[:40].strip("-")
    base = f"{time.strftime('%Y-%m-%d_%H%M%S')}_{program}" + (f"_{slug}" if slug else "")
    for n in range(1, 1000):
        folder = RUNS / (base if n == 1 else f"{base}-{n}")
        try:
            folder.mkdir()                                  # never an existing folder
            return folder
        except FileExistsError:
            continue
    raise RuntimeError("no free run folder name")


def _commands(program, params, folder):
    py = sys.executable
    script = lambda rel: str(REPO / rel)                     # noqa: E731
    if program == "solve":
        config = _solver_defaults()
        for k, v in params.items():
            if not k.startswith(("run_", "validate_")):
                _set(config, k, v)
        steps, soft = [[py, "-u", script("solver/main.py"), "config.json"]], []
        results = params["results_npz"]
        if params["run_validate"]:
            soft.append(len(steps))
            steps.append([py, "-u", script("solver/validate_results.py"), results,
                          "--samples", str(params["validate_samples"]),
                          "--stability-samples", str(params["validate_stability_samples"]),
                          "--seed", str(params["validate_seed"]), "--tol", str(params["jump_tol"]),
                          "--json", "validation.json"])
        if params["run_postprocess"]:
            steps.append([py, "-u", script("postprocessing/postprocess.py"), results,
                          "--tol", str(params["jump_tol"])])
        return config, steps, soft
    if program == "refine":
        config = {}
        for k, v in params.items():
            if k not in ("stages", "plan_only", "source_run"):
                _set(config, k, v)
        config["source_run"] = str(REPO / params["source_run"])
        if params["plan_only"]:
            stages = [["branches"], ["sample", "--dry-run"]]
        else:
            stages = [[s] for s in params["stages"]]
        return config, [[py, "-u", script("solver/refine.py"), "config.json", *s] for s in stages], []
    results = str(_results_file(REPO / params["source_run"]))
    if program == "validate":
        return None, [[py, "-u", script("solver/validate_results.py"), results,
                       "--samples", str(params["samples"]), "--stability-samples",
                       str(params["stability_samples"]), "--seed", str(params["seed"]),
                       "--tol", str(params["tol"]), "--json", "validation.json"]], [0]
    return None, [[py, "-u", script("postprocessing/postprocess.py"), results, "-o", "post",
                   "--tol", str(params["tol"])]], []


def start(program, params, label=""):
    """Validate, make the run folder, write config and job, start the runner. Returns the id."""
    if not isinstance(label, str) or len(label) > 80:
        raise ParamError("label: up to 80 characters")
    params = validate(program, params)
    folder = _new_folder(program, label)
    config, steps, soft = _commands(program, params, folder)
    if config is not None:
        with open(folder / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    job = {"id": folder.name, "program": program, "label": label, "params": params, "steps": steps,
           "soft_steps": soft,
           "created": time.time(), "python": sys.executable}
    with open(folder / "job.json", "w") as f:
        json.dump(job, f, indent=2)
    launch(folder)
    return folder.name


def launch(folder, from_step=0):
    subprocess.Popen([sys.executable, str(RUNNER), str(folder), str(from_step)], cwd=folder,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def _folder(run_id):
    if not isinstance(run_id, str) or not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        raise ParamError("bad run id")
    folder = RUNS / run_id
    if not (folder / "job.json").exists():
        raise ParamError("no such run")
    return folder


def _alive(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"run_job.py" in f.read()
    except OSError:
        return False


def status(folder):
    try:
        with open(folder / "status.json") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {"state": "starting"}
    if st.get("state") in ("running", "stopping") and not _alive(st.get("pid", -1)):
        st["state"] = "lost"                          # the runner died (crash, reboot)
    if st.get("state") == "starting" and time.time() - (folder / "job.json").stat().st_mtime > 30:
        st["state"] = "lost"
    return st


def runs():
    out = []
    if RUNS.is_dir():
        for folder in sorted(RUNS.iterdir(), reverse=True):
            if not (folder / "job.json").exists():
                continue
            with open(folder / "job.json") as f:
                job = json.load(f)
            st = status(folder)
            outputs = [name for name in ("solution_space_results.npz", "validation.json", "post",
                                         "nn_dataset.npz", "branches_grid.npz") if (folder / name).exists()]
            out.append({"id": job["id"], "program": job["program"], "label": job["label"],
                        "created": job["created"], "steps": len(job["steps"]), **st,
                        "outputs": outputs, "path": str(folder.relative_to(REPO))})
    return out


def job(run_id):
    with open(_folder(run_id) / "job.json") as f:
        return json.load(f)


def log(run_id, offset=0, limit=256 * 1024):
    path = _folder(run_id) / "run.log"
    size = path.stat().st_size if path.exists() else 0
    if offset < 0 or offset > size:
        offset = 0
    if size - offset > limit:                         # far behind: skip to the last part
        offset = size - limit
    data = b""
    if path.exists():
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(limit)
    return {"text": ANSI.sub("", data.decode("utf-8", "replace")), "offset": offset + len(data),
            "size": size, **status(path.parent)}


def stop(run_id):
    st = status(_folder(run_id))
    if st.get("state") not in ("running", "stopping"):
        raise ParamError("the run is not running")
    os.kill(st["pid"], signal.SIGINT)                 # the runner forwards it; a second one kills
    return st["state"]


def resume(run_id):
    """Restart a stopped, failed or lost run in its own folder, from the step that did not finish.
    The solver and refine pick up their checkpoints there."""
    folder = _folder(run_id)
    st = status(folder)
    if st.get("state") not in ("stopped", "failed", "lost"):
        raise ParamError("only a stopped, failed or lost run can be resumed")
    launch(folder, int(st.get("step", 0)))
