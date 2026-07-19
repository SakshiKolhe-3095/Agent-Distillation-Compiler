"""
Wires the benchmark harness to the real pipeline and dataset.
Runs a sample of tasks through the full agent pipeline (planner->coder->
tester->debugger) and produces a BenchmarkReport with pass@1, latency,
cost, and VRAM stats.
"""

import argparse
import json
import os

from agents.graph import build_graph
from datasets.schema import validate_dataset
from evaluation.benchmark import run_benchmark


def load_tasks(path: str, limit: int = None) -> list:
    """
    Loads tasks from a raw task file (list of {id, problem, test, ...})
    used by the collect_trajectories_*.py scripts.
    """
    with open(path) as f:
        tasks = json.load(f)
    if limit:
        tasks = tasks[:limit]
    return tasks


def make_pipeline_fn(teacher_model: str = None):
    """
    Returns a callable(task) -> {"passed": bool, "retries": int, "error": str}
    that runs a single task through the full agent graph.
    """
    if teacher_model:
        os.environ["TEACHER_MODEL"] = teacher_model

    pipeline = build_graph()

    def pipeline_fn(task: dict) -> dict:
        state = {
            "problem": task["problem"],
            "test_code": task["test"],
            "plan": "", "code": "", "passed": False, "error": "", "retries": 0,
        }
        final_state = pipeline.invoke(state)
        return {
            "passed": final_state["passed"],
            "retries": final_state.get("retries", 0),
            "error": final_state.get("error", ""),
        }

    return pipeline_fn


def main():
    parser = argparse.ArgumentParser(description="Run the benchmark harness against a task file.")
    parser.add_argument("--tasks", default="datasets/raw/tasks_sakshi_api.json",
                         help="Path to a raw task JSON file (list of {id, problem, test}).")
    parser.add_argument("--limit", type=int, default=5,
                         help="Number of tasks to run (default 5, for a quick smoke test).")
    parser.add_argument("--output", default="evaluation/results/benchmark_report.json",
                         help="Where to save the benchmark report.")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks, limit=args.limit)
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    pipeline_fn = make_pipeline_fn()
    report = run_benchmark(tasks, pipeline_fn)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report.save(args.output)

    print("\n=== Benchmark summary ===")
    print(json.dumps(report.summary(), indent=2))
    print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()