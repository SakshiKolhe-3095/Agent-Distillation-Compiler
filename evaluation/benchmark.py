"""
Benchmark harness skeleton for evaluating teacher/student pipeline runs.
Tracks pass@1, latency, API cost estimate, and VRAM usage per run.
Designed to be wired to the final trajectory dataset schema in a later step.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class BenchmarkResult:
    task_id: str
    passed: bool
    latency_seconds: float
    estimated_cost_usd: float = 0.0
    vram_mb: float = 0.0
    retries: int = 0
    error: str = ""


@dataclass
class BenchmarkReport:
    results: list = field(default_factory=list)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def pass_at_1(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_seconds for r in self.results) / len(self.results)

    def total_cost(self) -> float:
        return sum(r.estimated_cost_usd for r in self.results)

    def avg_vram(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.vram_mb for r in self.results) / len(self.results)

    def summary(self) -> dict:
        return {
            "num_tasks": len(self.results),
            "pass_at_1": round(self.pass_at_1(), 4),
            "avg_latency_seconds": round(self.avg_latency(), 3),
            "total_estimated_cost_usd": round(self.total_cost(), 4),
            "avg_vram_mb": round(self.avg_vram(), 1),
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "summary": self.summary(),
                "results": [vars(r) for r in self.results],
            }, f, indent=2)


def get_vram_usage_mb() -> float:
    """
    Returns current GPU VRAM usage in MB, if a CUDA device is available.
    Returns 0.0 if torch/CUDA isn't available (e.g. CPU-only or API-only run).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return 0.0


def run_benchmark(
    tasks: list,
    pipeline_fn: Callable[[dict], dict],
    cost_per_call_usd: float = 0.0,
) -> BenchmarkReport:
    """
    Runs `pipeline_fn` over a list of tasks, timing each call and recording
    pass/fail + latency + rough VRAM/cost.

    tasks: list of dicts, each with at least "id", "problem", "test"
    pipeline_fn: callable that takes a task dict and returns a dict with
                 at least "passed": bool, "retries": int (e.g. wraps
                 agents.graph.build_graph().invoke(...))
    cost_per_call_usd: flat per-task cost estimate (refine later once real
                       token counts / API pricing are wired in)
    """
    report = BenchmarkReport()

    for task in tasks:
        start = time.time()
        try:
            result = pipeline_fn(task)
            passed = result.get("passed", False)
            retries = result.get("retries", 0)
            error = result.get("error", "")
        except Exception as e:
            passed = False
            retries = 0
            error = str(e)
        elapsed = time.time() - start

        report.add(BenchmarkResult(
            task_id=task["id"],
            passed=passed,
            latency_seconds=elapsed,
            estimated_cost_usd=cost_per_call_usd,
            vram_mb=get_vram_usage_mb(),
            retries=retries,
            error=error,
        ))

    return report


if __name__ == "__main__":
    # quick manual smoke test with a trivial fake pipeline
    def fake_pipeline(task):
        time.sleep(0.05)
        return {"passed": task["id"] != "fail_me", "retries": 0}

    sample_tasks = [
        {"id": "task_1", "problem": "p1", "test": "t1"},
        {"id": "fail_me", "problem": "p2", "test": "t2"},
        {"id": "task_3", "problem": "p3", "test": "t3"},
    ]
    report = run_benchmark(sample_tasks, fake_pipeline)
    print(report.summary())