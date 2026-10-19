"""Run the steps of a console job in its folder, one after another.

usage: python run_job.py <run folder> [first step]

Reads job.json, appends every step's output to run.log and keeps status.json current:
state (running, stopping, finished, failed, stopped), the step and the return codes.
A failing soft step (job["soft_steps"], the validation) does not stop the run; it ends
"finished" with a warning.
SIGINT or SIGTERM is passed on to the running step as SIGINT, so main.py saves a
checkpoint, and no later step starts; a second one kills the step and its children.
"""
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    folder = Path(sys.argv[1]).resolve()
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with open(folder / "job.json") as f:
        job = json.load(f)
    status_path = folder / "status.json"
    try:
        with open(status_path) as f:
            previous = json.load(f)
    except (OSError, ValueError):
        previous = {}
    status = {"state": "running", "pid": os.getpid(), "started": previous.get("started", time.time()),
              "step": first, "steps": len(job["steps"]), "returncodes": previous.get("returncodes", []),
              "resumed": previous.get("resumed", 0) + (1 if first or previous else 0)}
    if previous.get("warning") and any(n < first for n in job.get("soft_steps", [])):
        status["warning"] = previous["warning"]          # the failed check is not rerun

    def save():
        status["updated"] = time.time()
        tmp = status_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, status_path)

    child, requests = None, 0

    def on_signal(signum, frame):
        nonlocal requests
        requests += 1
        status["state"] = "stopping"
        save()
        if child is not None and child.poll() is None:
            if requests == 1:
                child.send_signal(signal.SIGINT)
            else:
                os.killpg(child.pid, signal.SIGKILL)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    save()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(folder / "run.log", "ab", buffering=0) as log:
        if status["resumed"]:
            log.write(f"\n=== resumed {time.strftime('%Y-%m-%d %H:%M:%S')} at step {first + 1} ===\n".encode())
        state = "finished"
        for n in range(first, len(job["steps"])):
            if requests:
                state = "stopped"
                break
            argv = job["steps"][n]
            status["step"] = n
            save()
            log.write(f"\n$ {shlex.join(argv)}\n".encode())
            started = time.time()
            child = subprocess.Popen(argv, cwd=folder, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
            while True:
                try:
                    code = child.wait()
                    break
                except InterruptedError:
                    continue
            status["returncodes"] = status["returncodes"][:n] + [code]
            log.write(f"[step {n + 1}/{len(job['steps'])} exited with {code} after "
                      f"{time.time() - started:.0f} s]\n".encode())
            if requests:
                state = "stopped"
                break
            if code != 0:
                if n not in job.get("soft_steps", []):
                    state = "failed"
                    break
                status["warning"] = "checks failed, see the log"
        else:
            status["step"] = len(job["steps"])
    status["state"] = state
    status["finished"] = time.time()
    save()


if __name__ == "__main__":
    main()
