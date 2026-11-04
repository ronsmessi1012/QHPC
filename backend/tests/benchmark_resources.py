"""
Calibration harness for Step 6 resource profiles.

Runs each available algorithm at several problem sizes, measures wall-clock
time and peak resident memory, fits a power law to the measurements, and
prints ready-to-paste profile constants.

    python -m backend.tests.benchmark_resources
    python -m backend.tests.benchmark_resources --quick
    python -m backend.tests.benchmark_resources --only XGBoost,"A* Search"

Fitting model, matching backend/utils/resource_constants.py:

    cost(size) = base * (size / ref_size) ** exponent

Taking logs makes this linear, so a least-squares fit over log(scale) gives
the exponent directly, and the intercept gives `base` at ref_size:

    log(cost) = log(base) + exponent * log(scale)

After running, copy the emitted values into the matching profile and set
`"calibrated": True`. Until then those profiles are reasoned defaults from
each algorithm's complexity class and must not be described as measured.

Measurement caveats worth stating in any write-up:
  - Peak RSS is sampled from a background thread, so very short runs may
    under-report memory. Sizes are chosen large enough to mitigate this.
  - RSS includes interpreter and library overhead; the baseline is
    subtracted, but allocator behaviour makes this approximate.
  - Timings are single-run per size by default. Use --repeats for a median.
  - Results are machine-specific. Record the host spec alongside them.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# ---------------------------------------------------------------------------
# Host identity
# ---------------------------------------------------------------------------

def _host_tag() -> str:
    """
    Short label identifying the machine a measurement came from.

    base_runtime_s does not transfer between hosts, so every calibrated
    profile must record where its numbers were produced.
    """
    cpu = platform.processor() or platform.machine() or "unknown-cpu"
    cores = psutil.cpu_count(logical=True) if psutil else "?"
    gpu = ""
    try:
        import subprocess as _sp
        out = _sp.run(["nvidia-smi", "--query-gpu=name",
                       "--format=csv,noheader"],
                      capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0 and out.stdout.strip():
            gpu = " + " + out.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return f"{platform.system()} {cpu} ({cores} cores){gpu}"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class _PeakRSS:
    """Sample process RSS in a background thread and record the peak."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak = 0
        self.baseline = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process() if psutil else None

    def _rss(self) -> int:
        return self._proc.memory_info().rss if self._proc else 0

    def __enter__(self):
        gc.collect()
        self.baseline = self._rss()
        self.peak = self.baseline
        if self._proc:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._rss())

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._rss())
        return False

    @property
    def delta_gb(self) -> float:
        return max(self.peak - self.baseline, 0) / (1024 ** 3)


def measure(fn: Callable[[], None], repeats: int = 1) -> Tuple[float, float]:
    """Return (median runtime seconds, max peak memory GB) over `repeats`."""
    times: List[float] = []
    mems: List[float] = []
    for _ in range(repeats):
        with _PeakRSS() as rss:
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        mems.append(rss.delta_gb)
    return float(np.median(times)), max(mems)


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def fit_power_law(
    sizes: Sequence[int], costs: Sequence[float], ref_size: int
) -> Tuple[float, float, float]:
    """
    Fit cost = base * (size/ref_size) ** exponent.

    Returns (base, exponent, r_squared). Non-positive costs are dropped,
    since the fit is done in log space.
    """
    pairs = [(s, c) for s, c in zip(sizes, costs) if s > 0 and c > 0]
    if len(pairs) < 2:
        return float("nan"), float("nan"), float("nan")

    log_scale = np.log([s / ref_size for s, _ in pairs])
    log_cost = np.log([c for _, c in pairs])

    exponent, intercept = np.polyfit(log_scale, log_cost, 1)
    base = float(np.exp(intercept))

    predicted = intercept + exponent * log_scale
    ss_res = float(np.sum((log_cost - predicted) ** 2))
    ss_tot = float(np.sum((log_cost - np.mean(log_cost)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return base, float(exponent), r2


# ---------------------------------------------------------------------------
# Workload definitions
# ---------------------------------------------------------------------------
# Each entry: (algorithm name as it appears in utils/constants.py,
#              size_driver, ref_size, sizes to test, builder returning a
#              zero-arg callable that performs the real work)

def _sklearn_workloads(quick: bool) -> Dict[str, dict]:
    """Supervised-learning workloads. Returns {} if scikit-learn is unusable."""
    try:
        from sklearn.ensemble import (
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.svm import SVC
    except Exception as exc:  # noqa: BLE001
        # Report the REAL cause. An earlier version printed "not installed"
        # for every failure, which hid a Windows Application Control policy
        # blocking sklearn's compiled extensions and cost real debugging time.
        print(f"  [skip] scikit-learn unusable: {type(exc).__name__}: {exc}")
        return {}

    ml_sizes = [2_000, 5_000, 10_000] if quick else [2_000, 5_000, 10_000, 20_000]
    svm_sizes = [1_000, 2_000, 4_000] if quick else [1_000, 2_000, 4_000, 8_000]
    n_features = 20

    def clf_data(n):
        rng = np.random.default_rng(0)
        return rng.standard_normal((n, n_features)), rng.integers(0, 2, n)

    def reg_data(n):
        rng = np.random.default_rng(0)
        return rng.standard_normal((n, n_features)), rng.standard_normal(n)

    def fit(model_factory, data_factory):
        def builder(n):
            X, y = data_factory(n)
            model = model_factory()
            return lambda: model.fit(X, y)
        return builder

    return {
        "Random Forest": {
            "size_driver": "dataset_size", "ref_size": 10_000, "sizes": ml_sizes,
            "builder": fit(lambda: RandomForestClassifier(n_estimators=100, n_jobs=1),
                           clf_data),
        },
        "Random Forest Regressor": {
            "size_driver": "dataset_size", "ref_size": 10_000, "sizes": ml_sizes,
            "builder": fit(lambda: RandomForestRegressor(n_estimators=100, n_jobs=1),
                           reg_data),
        },
        "SVM": {
            "size_driver": "dataset_size", "ref_size": 10_000, "sizes": svm_sizes,
            "builder": fit(lambda: SVC(), clf_data),
        },
        "Logistic Regression": {
            "size_driver": "dataset_size", "ref_size": 10_000, "sizes": ml_sizes,
            "builder": fit(lambda: LogisticRegression(max_iter=1000), clf_data),
        },
        "Linear Regression": {
            "size_driver": "dataset_size", "ref_size": 10_000, "sizes": ml_sizes,
            "builder": fit(lambda: LinearRegression(), reg_data),
        },
    }


def _xgboost_workloads(quick: bool) -> Dict[str, dict]:
    """XGBoost is NOT in requirements.txt — skipped unless installed."""
    try:
        import xgboost as xgb
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] xgboost unusable: {type(exc).__name__}: {exc}")
        print("         (not in requirements.txt; pip install xgboost to "
              "calibrate these two profiles)")
        return {}

    sizes = [2_000, 5_000, 10_000] if quick else [2_000, 5_000, 10_000, 20_000]
    n_features = 20

    def builder_clf(n):
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((n, n_features)), rng.integers(0, 2, n)
        model = xgb.XGBClassifier(n_estimators=100, tree_method="hist", n_jobs=1)
        return lambda: model.fit(X, y)

    def builder_reg(n):
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((n, n_features)), rng.standard_normal(n)
        model = xgb.XGBRegressor(n_estimators=100, tree_method="hist", n_jobs=1)
        return lambda: model.fit(X, y)

    return {
        "XGBoost": {"size_driver": "dataset_size", "ref_size": 10_000,
                    "sizes": sizes, "builder": builder_clf},
        "XGBoost Regressor": {"size_driver": "dataset_size", "ref_size": 10_000,
                              "sizes": sizes, "builder": builder_reg},
    }


def _graph_workloads(quick: bool) -> Dict[str, dict]:
    """Dijkstra via scipy on a sparse random graph."""
    try:
        from scipy.sparse import random as sparse_random
        from scipy.sparse.csgraph import dijkstra
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] scipy unusable: {type(exc).__name__}: {exc}")
        return {}

    # Earlier sizes (500-4,000 nodes, single source) completed in under a
    # millisecond and the "measurements" were pure timer noise: identical
    # times across a 4x size range, fitting a nonsensical memory exponent
    # of -1.02. Fixed two ways: much larger graphs, and many source nodes
    # per run so each measurement lasts long enough to mean something.
    sizes = [5_000, 10_000, 20_000] if quick else [5_000, 10_000, 20_000, 40_000]
    n_sources = 32

    def builder(n):
        rng = np.random.default_rng(0)
        graph = sparse_random(n, n, density=min(10.0 / n, 0.05),
                              format="csr", random_state=rng)
        graph.data = np.abs(graph.data) + 0.01
        sources = np.linspace(0, n - 1, n_sources, dtype=int)
        return lambda: dijkstra(graph, indices=sources)

    return {
        "Dijkstra": {"size_driver": "dataset_size", "ref_size": 10_000,
                     "sizes": sizes, "builder": builder},
    }


def _quantum_workloads(quick: bool) -> Dict[str, dict]:
    """
    Statevector simulation timing via qiskit-aer.

    This calibrates SIM_FLOPS_PER_SECOND rather than a per-algorithm profile:
    memory for statevector simulation is exactly 2^n * 16 bytes and needs no
    fitting, but the throughput constant does.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] qiskit / qiskit-aer unusable: "
              f"{type(exc).__name__}: {exc}")
        return {}

    sizes = [10, 14, 18] if quick else [10, 14, 18, 22]

    def builder(n):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for layer in range(3):
            for q in range(0, n - 1, 2):
                qc.cx(q, q + 1)
            for q in range(1, n - 1, 2):
                qc.cx(q, q + 1)
        qc.save_statevector()
        sim = AerSimulator(method="statevector")
        compiled = transpile(qc, sim)
        return lambda: sim.run(compiled, shots=1).result()

    return {
        "Quantum Statevector Simulation": {
            "size_driver": "qubits", "ref_size": 1, "sizes": sizes,
            "builder": builder, "quantum": True,
        },
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(only: Optional[List[str]], quick: bool, repeats: int) -> dict:
    print(f"Host: {platform.platform()} | Python {platform.python_version()}")
    if psutil:
        print(f"CPUs: {psutil.cpu_count(logical=True)} logical | "
              f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    else:
        print("psutil not installed — memory will report 0.0; "
              "install it for memory calibration")
    print(f"Host tag: {_host_tag()}")
    print(f"Mode: {'quick' if quick else 'full'} | repeats={repeats}\n")

    workloads: Dict[str, dict] = {}
    for group in (_xgboost_workloads, _sklearn_workloads,
                  _graph_workloads, _quantum_workloads):
        workloads.update(group(quick))

    if only:
        workloads = {k: v for k, v in workloads.items() if k in set(only)}
        if not workloads:
            print(f"No workloads matched {only}")
            return {}

    results = {}
    for name, cfg in workloads.items():
        print(f"\n{name}")
        print("-" * len(name))
        sizes, times, mems = [], [], []
        for n in cfg["sizes"]:
            try:
                work = cfg["builder"](n)
                t, m = measure(work, repeats=repeats)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  size {n:>7,}: FAILED ({type(exc).__name__}: {exc})")
                continue
            sizes.append(n)
            times.append(t)
            mems.append(m)
            unit = "qubits" if cfg.get("quantum") else "size"
            print(f"  {unit} {n:>7,}: {t:>9.3f} s   {m:>7.3f} GB peak")

        if len(sizes) < 2:
            print("  not enough successful runs to fit")
            continue

        # A measurement shorter than this is dominated by timer resolution
        # and process noise; fitting to it produces confident nonsense.
        MIN_MEANINGFUL_RUNTIME_S = 0.05
        too_fast = [t for t in times if t < MIN_MEANINGFUL_RUNTIME_S]
        if too_fast:
            print(f"  [warn] {len(too_fast)}/{len(times)} runs completed in "
                  f"under {MIN_MEANINGFUL_RUNTIME_S}s — too fast to measure "
                  f"reliably. Increase the sizes for this workload before "
                  f"trusting the fit below.")

        if cfg.get("quantum"):
            # Throughput: gates * 2^n / seconds, using the same gate model
            # as the estimator (depth*n/2 two-qubit gates, 3 layers here).
            flops = []
            for n, t in zip(sizes, times):
                depth = 3 * 2  # 3 layers, two sweeps each
                gates = max(1, (depth * n) // 2)
                flops.append(gates * (2 ** n) / t)
            suggested = float(np.median(flops))
            print(f"\n  SIM_FLOPS_PER_SECOND = {suggested:.3g}   "
                  f"(current: see resource_constants.py)")
            results[name] = {"sim_flops_per_second": suggested,
                             "sizes": sizes, "times": times}
            continue

        ref = cfg["ref_size"]
        base_m, exp_m, r2_m = fit_power_law(sizes, mems, ref)
        base_t, exp_t, r2_t = fit_power_law(sizes, times, ref)

        print(f'\n    "{name}": {{')
        print(f'        "size_driver": "{cfg["size_driver"]}", "ref_size": {ref:_},')
        print(f'        "base_memory_gb": {base_m:.4f}, "memory_exp": {exp_m:.2f},'
              f'   # R^2={r2_m:.3f}')
        print(f'        "base_runtime_s": {base_t:.4f}, "runtime_exp": {exp_t:.2f},'
              f'   # R^2={r2_t:.3f}')
        print('        "gpu_capable": ?, "mpi_capable": ?,')
        print(f'        "calibrated": True, "calibration_host": "{_host_tag()}",')
        print('    },')

        if not math.isnan(r2_m) and r2_m < 0.90:
            print(f"  [warn] memory fit is poor (R^2={r2_m:.3f}) — peak RSS is "
                  f"noisy at these sizes; try --repeats 3 or larger sizes")

        results[name] = {
            "size_driver": cfg["size_driver"], "ref_size": ref,
            "base_memory_gb": base_m, "memory_exp": exp_m, "memory_r2": r2_m,
            "base_runtime_s": base_t, "runtime_exp": exp_t, "runtime_r2": r2_t,
            "sizes": sizes, "times": times, "memories": mems,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="fewer and smaller sizes (faster, less accurate)")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated algorithm names to benchmark")
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per size; median is used (default 1)")
    parser.add_argument("--json", type=str, default=None,
                        help="write raw measurements to this path")
    args = parser.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    results = run(only, args.quick, args.repeats)

    if args.json and results:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"host": platform.platform(),
                       "host_tag": _host_tag(),
                       "results": results}, fh, indent=2)
        print(f"\nRaw measurements written to {args.json}")

    print("\nPaste the emitted blocks into backend/utils/resource_constants.py, "
          "fill in gpu_capable/mpi_capable from what your runtime actually "
          "supports, and set calibrated=True only for profiles you measured.")


if __name__ == "__main__":
    main()
