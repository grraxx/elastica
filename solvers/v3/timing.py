"""Timing utilities for per-node solution runs."""

import time


class NodeSolveTimer:
    """Measure solve durations and report progress at a fixed interval."""

    def __init__(self, report_interval=1000):
        if report_interval <= 0:
            raise ValueError("report_interval must be positive")
        self.report_interval = int(report_interval)
        self.started_at = time.perf_counter()
        self.node_started_at = None
        self.solve_times = []

    def start_node(self):
        self.node_started_at = time.perf_counter()

    def finish_node(self, node_index, solved_count):
        if self.node_started_at is None:
            raise RuntimeError("start_node must be called before finish_node")
        duration = time.perf_counter() - self.node_started_at
        self.solve_times.append(duration)
        self.node_started_at = None
        if solved_count % self.report_interval == 0:
            elapsed = time.perf_counter() - self.started_at
            average = sum(self.solve_times) / len(self.solve_times)
            print(
                f"  Timing after {solved_count} nodes: "
                f"last={duration:.3f} s, average={average:.3f} s/node, "
                f"elapsed={elapsed:.1f} s"
            )
        return duration
