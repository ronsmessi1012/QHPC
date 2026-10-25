"""
MPI strong-scaling benchmark for Step 8.

    python -m backend.tests.benchmark_mpi_scaling
    python -m backend.tests.benchmark_mpi_scaling --quick
    python -m backend.tests.benchmark_mpi_scaling --json mpi_scaling.json

WHAT THIS MEASURES
------------------
Strong scaling: the problem size is held FIXED while the rank count grows.
Speedup is T(1) / T(n); parallel efficiency is speedup / n.

Weak scaling (growing the problem with the ranks) is a different experiment
and is not what this does. Saying which one a number came from matters —
they answer different questions and are not interchangeable.

THE BASELINE
------------
T(1) is measured through `mpiexec -n 1`, not by calling the workload
in-process. Both ends then carry the same subprocess and MPI_Init overhead,
so the ratio isolates parallelism rather than measuring process startup.

WHAT TO EXPECT, AND WHY IT IS WORTH REPORTING
---------------------------------------------
The two workloads should scale very differently, and the contrast is a
result in itself:

  Monte Carlo        Embarrassingly parallel — ranks sample independently
                     and communicate once at the end. Should scale close to
                     linearly up to the physical core count.

  Genetic Algorithm  Island model with an allreduce EVERY generation.
                     Communication-bound, so efficiency should fall off
                     noticeably. A GA that scaled perfectly here would be
                     evidence the migration step was not actually running.

Expect the curve to flatten past the PHYSICAL core count. Hyperthreads
share execution units, so ranks beyond that compete rather than add.

CORRECTNESS AT EVERY RANK COUNT
-------------------------------
Each run records its correctness metric, not just its time. A parallel
implementation that got faster by computing the wrong answer would show up
here as a metric that drifts with rank count — which is exactly the bug
that silent speedup numbers hide.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from typing import Any, Dict, List, Optional

from backend.engine.workloads import WORKLOAD_REGISTRY
from backend.models.decision_schema import ExecutionDecision
from backend.models.hardware_schema import HardwareMapping
from backend.models.problem_schema import ProblemSpecification
from backend.models.resource_schema import ResourceEstimate
from backend.agents.resource_estimator import estimate_resources
from backend.engine.performance_engine import execute
from backend.utils.hardware_detect import detect_hardware

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# Problem sizes chosen so a single rank takes a few seconds: long enough
# that process startup is not a significant fraction, short enough that a
# full sweep finishes in minutes.
# Sized so the 1-rank baseline runs for tens of seconds. Each rank launches
# a fresh interpreter and imports numpy, costing roughly half a second that
# no amount of parallelism removes. Against a 1.6s baseline that fixed cost
# caps the achievable speedup near 3x however many ranks are used; against
# a 30s baseline it is noise.
# Calibrated so a 1-rank baseline runs ~15-20s: long enough that per-rank
# startup (~0.5s) is a small fraction, short enough that a full sweep of
# 5 rank counts x 3 repeats x 2 workloads finishes in roughly half an hour.
WORKLOAD_SIZES = {
    "Monte Carlo Simulation": {"dimensionality": 10, "dataset_size": 400_000},
    # dataset_size is the GENERATION COUNT for the GA (see workloads.py)
    "Genetic Algorithm": {"dimensionality": 200, "dataset_size": 8_000},
}
QUICK_SIZES = {
    "Monte Carlo Simulation": {"dimensionality": 10, "dataset_size": 60_000},
    "Genetic Algorithm": {"dimensionality": 100, "dataset_size": 2_000},
}

# A baseline shorter than this cannot support a credible speedup claim once
# per-rank startup is accounted for.
MIN_MEANINGFUL_RUNTIME_S = 10.0


def _host_tag() -> str:
    cpu = platform.processor() or platform.machine() or "unknown-cpu"
    logical = psutil.cpu_count(logical=True) if psutil else "?"
    physical = psutil.cpu_count(logical=False) if psutil else "?"
    return f"{platform.system()} {cpu} ({logical} logical / {physical} physical)"


def measure_launch_overhead(ranks: int, repeats: int = 3) -> float:
    """
    Time an MPI launch that does no work.

    Every rank starts an interpreter, imports mpi4py and calls MPI_Init.
    That cost is paid whatever the workload, so reporting it separately
    turns an unexplained efficiency shortfall into a quantified one.
    """
    import shutil, subprocess, sys as _sys
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        return float("nan")

    from backend.engine.performance_engine import _is_openmpi
    cmd = [launcher, "-n", str(ranks)]
    if _is_openmpi(launcher):
        physical = psutil.cpu_count(logical=False) if psutil else 1
        if ranks > (physical or 1):
            cmd.append("--oversubscribe")
        if hasattr(__import__("os"), "geteuid") and __import__("os").geteuid() == 0:
            cmd.append("--allow-run-as-root")
    cmd += [_sys.executable, "-c", "from mpi4py import MPI; MPI.COMM_WORLD.Barrier()"]

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        done = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        if done.returncode != 0:
            return float("nan")
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times))


def _rank_counts(max_slots: int) -> List[int]:
    """Powers of two up to the slot count, always including 1 and the max."""
    counts, n = [], 1
    while n <= max_slots:
        counts.append(n)
        n *= 2
    if max_slots not in counts:
        counts.append(max_slots)
    return counts


def run_one(algorithm: str, ranks: int, sizes: Dict[str, int],
            repeats: int) -> Dict[str, Any]:
    """Run `algorithm` on `ranks` ranks, `repeats` times; return the median."""
    spec = ProblemSpecification(
        user_query=f"mpi scaling probe, {ranks} ranks",
        problem_class="simulation", confidence=0.8,
        dimensionality=sizes["dimensionality"],
        dataset_size=sizes["dataset_size"],
        constraint_structure="constrained",
    )
    decision = ExecutionDecision(
        selected_family="classical", selected_algorithm=algorithm,
        routing_mode="classical_only", decision_confidence=0.80,
        borderline_score=0.20, explanation="mpi scaling benchmark",
    )
    estimate = estimate_resources(decision, spec)
    mapping = HardwareMapping(
        backend_type="mpi_cluster",
        device_spec=f"MPI ({ranks} ranks)",
        justification="scaling benchmark",
        reasoning_trace={"mpi_slots": ranks},
    )

    times: List[float] = []
    metrics: List[float] = []
    device_detail = ""
    error: Optional[str] = None
    behaviour, target, tolerance, note = "invariant", None, None, ""

    for _ in range(repeats):
        telemetry = execute(mapping, estimate, decision, spec, mpi_ranks=ranks)
        if not telemetry.success:
            error = telemetry.error_message
            break
        times.append(telemetry.actual_runtime_seconds)
        if telemetry.result_metric_value is not None:
            metrics.append(telemetry.result_metric_value)
        device_detail = telemetry.device_detail
        behaviour = telemetry.raw_metrics.get("metric_rank_behaviour", behaviour)
        target = telemetry.raw_metrics.get("metric_target", target)
        tolerance = telemetry.raw_metrics.get("metric_tolerance", tolerance)
        note = telemetry.raw_metrics.get("metric_note", note)

    if error or not times:
        return {"ranks": ranks, "success": False, "error": error}

    return {
        "ranks": ranks,
        "success": True,
        "runtime_seconds": float(statistics.median(times)),
        "all_runtimes": times,
        "metric_value": float(statistics.median(metrics)) if metrics else None,
        "device_detail": device_detail,
        "metric_behaviour": behaviour,
        "metric_target": target,
        "metric_tolerance": tolerance,
        "metric_note": note,
    }


def sweep(algorithm: str, max_slots: int, sizes: Dict[str, int],
          repeats: int) -> Dict[str, Any]:
    print(f"\n{algorithm}")
    print("=" * len(algorithm))
    print(f"  fixed problem size: dimensionality={sizes['dimensionality']}, "
          f"dataset_size={sizes['dataset_size']}  (strong scaling)")
    print()
    print(f"  {'ranks':>6}  {'time (s)':>10}  {'speedup':>8}  "
          f"{'efficiency':>11}  {'metric':>12}")
    print("  " + "-" * 56)

    runs: List[Dict[str, Any]] = []
    baseline: Optional[float] = None

    for ranks in _rank_counts(max_slots):
        result = run_one(algorithm, ranks, sizes, repeats)
        runs.append(result)

        if not result["success"]:
            print(f"  {ranks:>6}  FAILED: {str(result['error'])[:44]}")
            continue

        runtime = result["runtime_seconds"]
        if baseline is None:
            baseline = runtime
        speedup = baseline / runtime if runtime > 0 else float("nan")
        efficiency = speedup / ranks
        result["speedup"] = round(speedup, 4)
        result["efficiency"] = round(efficiency, 4)

        metric = result["metric_value"]
        metric_str = f"{metric:.6f}" if metric is not None else "n/a"
        print(f"  {ranks:>6}  {runtime:>10.3f}  {speedup:>7.2f}x  "
              f"{efficiency * 100:>10.1f}%  {metric_str:>12}")

    ok = [r for r in runs if r.get("success")]

    if ok and ok[0]["runtime_seconds"] < MIN_MEANINGFUL_RUNTIME_S:
        print(f"\n  [warn] the 1-rank baseline ran in "
              f"{ok[0]['runtime_seconds']:.3f}s, under "
              f"{MIN_MEANINGFUL_RUNTIME_S}s. Process startup is a large "
              f"fraction of that, so these speedups understate the real "
              f"parallelism. Increase the problem size.")

    # ---- Correctness across rank counts -------------------------------
    # Interpreted according to what the workload says its metric does, so
    # that legitimate variation is not reported as corruption and genuine
    # corruption is not excused as "expected".
    metrics = [r["metric_value"] for r in ok if r.get("metric_value") is not None]
    verdict = None
    if len(metrics) > 1:
        behaviour = ok[0].get("metric_behaviour", "invariant")
        target = ok[0].get("metric_target")
        tolerance = ok[0].get("metric_tolerance")
        note = ok[0].get("metric_note", "")
        spread = max(metrics) - min(metrics)

        print(f"\n  correctness metric across rank counts: "
              f"{min(metrics):.6g} to {max(metrics):.6g} (spread {spread:.6g})")
        if note:
            print(f"  expected behaviour: {note}")

        if behaviour == "invariant":
            if spread > 1e-9:
                verdict = "FAIL"
                print(f"  -> PROBLEM: this metric must not change with rank "
                      f"count. A drift of {spread:.6g} means the parallel "
                      f"version is computing something different.")
            else:
                verdict = "OK"
                print("  -> OK: identical at every rank count, as required.")

        elif behaviour == "stochastic" and target is not None and tolerance:
            worst = max(abs(m - target) for m in metrics)
            if worst <= tolerance:
                verdict = "OK"
                print(f"  -> OK: every rank count stayed within {tolerance:.6g} "
                      f"of {target:g} (worst {worst:.6g}). Variation is "
                      f"sampling noise, not corruption.")
            else:
                verdict = "FAIL"
                print(f"  -> PROBLEM: worst deviation {worst:.6g} exceeds the "
                      f"tolerance {tolerance:.6g}. That is more than sampling "
                      f"noise can explain.")

        elif behaviour == "rank_dependent":
            verdict = "EXPECTED_TREND"
            if spread <= 1e-9:
                direction = "did not change"
            elif metrics[-1] > metrics[0]:
                direction = "worsens"
            else:
                direction = "improves"
            print(f"  -> EXPECTED: the metric {direction} as ranks increase, "
                  f"by design. Report the trend; it is a property of the "
                  f"algorithm under strong scaling, not an error.")

    return {"algorithm": algorithm, "sizes": sizes, "runs": runs,
            "correctness_verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="smaller problems; faster but noisier")
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per rank count; median is used (default 3)")
    parser.add_argument("--max-ranks", type=int, default=None,
                        help="cap the rank count (default: detected MPI slots)")
    parser.add_argument("--json", type=str, default=None,
                        help="write raw measurements to this path")
    args = parser.parse_args()

    inventory = detect_hardware()
    print(f"Host     : {_host_tag()}")
    print(f"MPI      : ", end="")
    if not inventory.mpi_available:
        print("NOT AVAILABLE")
        for note in inventory.detection_notes:
            if note.startswith("MPI"):
                print(f"           {note}")
        print("\nNothing to measure. Install an MPI runtime first "
              "(OpenMPI/MPICH on Linux, Microsoft MPI on Windows).")
        return
    print(f"available, {inventory.mpi_slots} slots")

    logical = psutil.cpu_count(logical=True) if psutil else None
    physical = psutil.cpu_count(logical=False) if psutil else None
    if logical and physical and logical > physical:
        print(f"\nNOTE: {logical} logical cores but only {physical} physical. "
              f"Speedup past\n      {physical} ranks is limited by "
              f"hyperthreading — ranks share execution\n      units rather "
              f"than adding throughput. Report the physical count.")

    max_slots = args.max_ranks or inventory.mpi_slots
    sizes_table = QUICK_SIZES if args.quick else WORKLOAD_SIZES

    results = []
    for algorithm in WORKLOAD_SIZES:
        workload = WORKLOAD_REGISTRY.get(algorithm)
        if workload is None or "mpi_cluster" not in workload.supported_backends:
            print(f"\n[skip] {algorithm}: no MPI implementation")
            continue
        results.append(sweep(algorithm, max_slots, sizes_table[algorithm],
                             args.repeats))

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    for entry in results:
        ok = [r for r in entry["runs"] if r.get("success") and "speedup" in r]
        if not ok:
            print(f"  {entry['algorithm']:<26} no successful runs")
            continue
        best = max(ok, key=lambda r: r["speedup"])
        print(f"  {entry['algorithm']:<26} best {best['speedup']:.2f}x "
              f"at {best['ranks']} ranks "
              f"({best['efficiency'] * 100:.0f}% efficiency)")

    if args.json:
        payload = {
            "host": platform.platform(),
            "host_tag": _host_tag(),
            "mpi_slots": inventory.mpi_slots,
            "logical_cores": logical,
            "physical_cores": physical,
            "scaling_type": "strong",
            "repeats": args.repeats,
            "results": results,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nRaw measurements written to {args.json}")

    print("\nThese numbers describe THIS machine. Runtimes do not transfer "
          "between\nhosts, so report the host tag alongside them.")


if __name__ == "__main__":
    main()
