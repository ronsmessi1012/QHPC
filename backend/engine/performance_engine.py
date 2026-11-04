"""
Step 8 — Runtime Execution & Telemetry Engine

Input:  HardwareMapping (Step 7), ResourceEstimate (Step 6),
        ExecutionDecision (Step 5.5), ProblemSpecification (Step 1)
Output: ExecutionTelemetry

This is the first module in the pipeline that does work rather than
deciding about work. Steps 2 through 7 produce a plan; this executes it on
the backend Step 7 assigned, measures what it actually cost, and reports
the gap against what Step 6 predicted.

BACKEND DISPATCH
----------------
    cpu            in-process, single-threaded
    gpu_cuda       in-process, library's own CUDA path (device="cuda")
    mpi_cluster    out-of-process via mpiexec; ranks run a worker module
    qpu_simulator  in-process, qiskit-aer (CPU or GPU statevector/MPS)

MPI is the odd one out and has to be. `mpiexec -n N` launches N separate
interpreters; it cannot be entered from inside an already-running process.
So the MPI path shells out to backend/engine/mpi_worker.py and reads a JSON
result back, rather than pretending to parallelise in-process.

HONESTY CONSTRAINTS
-------------------
A failed run produces telemetry with success=False and the real exception,
never a zero-cost "success". An algorithm with no implementation is
reported as unavailable, not as a run of nothing. Error ratios are None
when the prediction was zero, because a ratio against zero is undefined and
inventing one would fabricate a result.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

from backend.models.decision_schema import ExecutionDecision
from backend.models.hardware_schema import HardwareMapping
from backend.models.problem_schema import ProblemSpecification
from backend.models.resource_schema import ResourceEstimate
from backend.models.telemetry_schema import ExecutionTelemetry
from backend.engine.workloads import (
    WorkloadResult,
    WorkloadUnavailable,
    get_workload,
)

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

# How long an MPI subprocess may run before being abandoned.
MPI_TIMEOUT_S = 900


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

class _PeakMemory:
    """
    Sample process RSS on a background thread and keep the peak.

    Caveats worth stating in any write-up: RSS includes interpreter and
    library overhead (the baseline is subtracted, but allocator behaviour
    makes that approximate), and a very short run may be missed between
    samples entirely.
    """

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak = 0
        self.baseline = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process() if psutil else None

    def _rss(self) -> int:
        return self._proc.memory_info().rss if self._proc else 0

    def __enter__(self) -> "_PeakMemory":
        gc.collect()
        self.baseline = self.peak = self._rss()
        if self._proc:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._rss())

    def __exit__(self, *exc: Any) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._rss())
        return False

    @property
    def delta_gb(self) -> float:
        return max(self.peak - self.baseline, 0) / (1024 ** 3)


def _host_tag() -> str:
    """Machine identity. Runtimes are meaningless without it."""
    cpu = platform.processor() or platform.machine() or "unknown-cpu"
    cores = psutil.cpu_count(logical=True) if psutil else "?"
    return f"{platform.system()} {cpu} ({cores} cores)"


def _ratio(actual: float, predicted: float) -> Optional[float]:
    """actual/predicted, or None when predicted is zero (undefined)."""
    if predicted <= 0:
        return None
    return round(actual / predicted, 4)


def reportable_memory_gb(backend: str, measured_gb: float) -> float:
    """
    What may honestly be reported as the run's peak memory.

    For an MPI run the engine measures the RSS of the LAUNCHING process,
    which excludes every child rank — the processes that did the work. That
    figure would understate the job badly, so it is not reported at all.

    Extracted as a function so the guarantee is testable on a host with no
    MPI runtime. Asserting it through execute() alone passes vacuously
    there, because the run fails before reaching the measurement.
    """
    return 0.0 if backend == "mpi_cluster" else measured_gb


# ---------------------------------------------------------------------------
# Backend executors
# ---------------------------------------------------------------------------

def _run_in_process(algorithm: str, spec: ProblemSpecification,
                    backend: str, **kwargs: Any) -> WorkloadResult:
    """CPU, CUDA and simulator all run here; only the kwargs differ."""
    workload = get_workload(algorithm)
    if backend not in workload.supported_backends:
        raise WorkloadUnavailable(
            f"'{algorithm}' does not support backend '{backend}' "
            f"(supports: {workload.supported_backends})"
        )
    return workload.run(
        dataset_size=spec.dataset_size,
        dimensionality=spec.dimensionality,
        backend=backend,
        **kwargs,
    )


def _launcher_error_detail(completed: Any) -> str:
    """
    Pull the informative lines out of a launcher's error output.

    MPI launchers pad failures with banner rules and blank lines, so taking
    the last few lines naively yields something like ": /" that says nothing
    about what actually went wrong.
    """
    noise = ("---", "===", "***", "____")
    blob = ((completed.stderr or "") + "\n" + (completed.stdout or ""))
    lines = [
        ln.strip() for ln in blob.splitlines()
        if ln.strip() and not ln.strip().startswith(noise)
    ]
    return " | ".join(lines[-6:]) if lines else f"exit {completed.returncode}, no output"


def _launcher_family(launcher: str) -> str:
    """
    Identify the MPI launcher family, because they take incompatible flags.

    Returns "openmpi4", "openmpi5", "mpich", "msmpi" or "unknown".

    Version strings differ more than the binary names suggest:
        OpenMPI 4  "mpiexec (OpenRTE) 4.x"
        OpenMPI 5  "mpiexec (PRRTE) 3.x"     PRRTE replaced ORTE
        MPICH      "HYDRA build details"
        MS-MPI     "Microsoft MPI Startup Program"

    Matching only "open mpi"/"openrte" misses OpenMPI 5 entirely, which is
    how an --oversubscribe fix silently failed to apply on a PRRTE host.
    """
    try:
        out = subprocess.run([launcher, "--version"], capture_output=True,
                             text=True, timeout=10, check=False)
        blob = (out.stdout + out.stderr).lower()
    except (subprocess.TimeoutExpired, OSError):
        return "unknown"

    if "microsoft mpi" in blob:
        return "msmpi"
    if "prrte" in blob or "prte" in blob:
        return "openmpi5"
    if "open mpi" in blob or "openrte" in blob or "open-rte" in blob:
        return "openmpi4"
    if "hydra" in blob or "mpich" in blob:
        return "mpich"
    return "unknown"


def _oversubscribe_variants(family: str) -> List[List[str]]:
    """
    Flag variants permitting more ranks than slots, best candidate first.

    OpenMPI 5 dropped --oversubscribe for --map-by :OVERSUBSCRIBE, but which
    one a given build accepts is not reliably predictable from its version
    string, so the caller tries them in order rather than guessing once.
    MPICH and MS-MPI oversubscribe by default and reject both flags.
    """
    if family == "openmpi5":
        return [["--map-by", ":OVERSUBSCRIBE"], ["--oversubscribe"]]
    if family == "openmpi4":
        return [["--oversubscribe"], ["--map-by", ":OVERSUBSCRIBE"]]
    return [[]]


def _is_openmpi(launcher: str) -> bool:
    """Yes/no form, for callers that do not need the exact family."""
    return _launcher_family(launcher).startswith("openmpi")


def _run_via_mpi(algorithm: str, spec: ProblemSpecification,
                 n_ranks: int) -> WorkloadResult:
    """
    Launch `mpiexec -n N` on the worker module and read back its JSON.

    This cannot be done in-process: MPI_Init must happen in each of N
    freshly-launched interpreters.
    """
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        raise WorkloadUnavailable(
            "no MPI launcher found (mpiexec/mpirun not on PATH). On Windows "
            "install Microsoft MPI; on Linux install OpenMPI or MPICH."
        )

    family = _launcher_family(launcher)

    base_flags: List[str] = []
    # OpenMPI refuses to run as root without this; normal inside Colab and
    # many containers. MPICH and MS-MPI reject the flag.
    if family.startswith("openmpi") and hasattr(os, "geteuid") and os.geteuid() == 0:
        base_flags.append("--allow-run-as-root")

    # OpenMPI counts slots as PHYSICAL cores, so more ranks than that is
    # refused unless oversubscription is permitted. A 16-logical/8-physical
    # machine hits this at -n 16, precisely where a scaling study looks.
    physical = (psutil.cpu_count(logical=False) or 1) if psutil else 1
    need_oversubscribe = n_ranks > physical

    worker = [sys.executable, "-m", "backend.engine.mpi_worker",
              "--algorithm", algorithm,
              "--dataset-size", str(spec.dataset_size),
              "--dimensionality", str(spec.dimensionality)]

    variants = _oversubscribe_variants(family) if need_oversubscribe else [[]]

    attempts: List[str] = []
    completed = None
    for extra in variants:
        cmd = [launcher, "-n", str(n_ranks)] + base_flags + extra + worker
        completed = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=MPI_TIMEOUT_S, check=False, cwd=os.getcwd(),
        )
        if completed.returncode == 0:
            break
        attempts.append(
            f"[{' '.join(extra) if extra else 'no extra flags'}] "
            + _launcher_error_detail(completed)
        )

    if completed is None or completed.returncode != 0:
        hint = ""
        if need_oversubscribe:
            hint = (f"; {n_ranks} ranks exceeds {physical} physical cores, so "
                    f"oversubscription was required and "
                    f"{len(variants)} flag variant(s) were tried")
        raise WorkloadUnavailable(
            f"{os.path.basename(launcher)} ({family}) failed with "
            f"{n_ranks} ranks{hint}: " + " ;; ".join(attempts)
        )

    # The worker prints one JSON line prefixed so it survives MPI's own
    # chatter on stdout.
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith("QHPC_RESULT "):
            payload = json.loads(line[len("QHPC_RESULT "):])
            break
    if payload is None:
        raise WorkloadUnavailable(
            "MPI worker produced no result line; stdout tail: "
            + " | ".join(completed.stdout.strip().splitlines()[-5:])
        )

    fallback_reason = payload.get("mpi_fallback_reason")
    comm_size = payload.get("comm_size", payload.get("detail", {}).get("comm_size", 1))

    if fallback_reason or (n_ranks > 1 and comm_size <= 1):
        # The launcher ran, but the work happened in ONE process. Reporting
        # this as a successful mpi_cluster run would put a serial timing in
        # the telemetry under a distributed backend — the exact confusion
        # this engine exists to prevent.
        raise WorkloadUnavailable(
            f"MPI launch produced no parallelism: {n_ranks} ranks were "
            f"requested but the worker ran with comm_size={comm_size}"
            + (f" because mpi4py is unavailable ({fallback_reason})"
               if fallback_reason else "")
            + ". Install mpi4py in the same interpreter the launcher starts."
        )

    return WorkloadResult(
        metric_name=payload["metric_name"],
        metric_value=payload["metric_value"],
        device_detail=payload["device_detail"],
        detail=payload.get("detail", {}),
        metric_rank_behaviour=payload.get("metric_rank_behaviour", "invariant"),
        metric_target=payload.get("metric_target"),
        metric_tolerance=payload.get("metric_tolerance"),
        metric_note=payload.get("metric_note", ""),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def execute(
    mapping: HardwareMapping,
    estimate: ResourceEstimate,
    decision: ExecutionDecision,
    spec: ProblemSpecification,
    *,
    mpi_ranks: Optional[int] = None,
    use_gpu_simulator: bool = False,
) -> ExecutionTelemetry:
    """
    Run the workload on the backend Step 7 assigned and measure the cost.

    `mpi_ranks` defaults to the slot count Step 7 recorded in its trace.
    `use_gpu_simulator` routes qiskit-aer to CUDA; it is ignored unless the
    backend is qpu_simulator, and will raise inside qiskit-aer if the GPU
    build is not installed — which is the correct, visible failure.

    A mapping flagged `is_hypothetical` describes hardware that was never
    verified, so executing against it is refused rather than producing
    telemetry that looks measured.
    """
    algorithm = decision.selected_algorithm
    backend = mapping.backend_type

    base: Dict[str, Any] = {
        "algorithm": algorithm,
        "backend_type": backend,
        "telemetry_host": _host_tag(),
        "predicted_runtime_seconds": estimate.estimated_runtime_seconds,
        "predicted_memory_gb": estimate.estimated_memory_gb,
        "compared_against_uncalibrated_estimate": not estimate.calibrated,
    }

    if mapping.is_hypothetical:
        return ExecutionTelemetry(
            **base,
            device_detail="none — mapping was hypothetical",
            success=False,
            error_message=(
                "Refusing to execute against a hypothetical mapping. Step 7 "
                "ran in planning mode 'target', so the assigned backend was "
                "declared rather than verified present. Re-run with "
                "QHPC_PLANNING_MODE=detected on hardware that actually has it."
            ),
            actual_runtime_seconds=0.0,
            actual_peak_memory_gb=0.0,
            raw_metrics={"refused": "hypothetical_mapping"},
        )

    if mpi_ranks is None:
        mpi_ranks = int(mapping.reasoning_trace.get("mpi_slots", 1) or 1)

    kwargs: Dict[str, Any] = {}
    if backend == "qpu_simulator":
        kwargs["use_gpu"] = use_gpu_simulator
        bond = estimate.reasoning_trace.get("bond_dimension")
        if bond:
            kwargs["bond_dimension"] = int(bond)

    start = time.perf_counter()
    try:
        with _PeakMemory() as mem:
            if backend == "mpi_cluster":
                result = _run_via_mpi(algorithm, spec, max(mpi_ranks, 2))
            else:
                result = _run_in_process(algorithm, spec, backend, **kwargs)
        runtime = time.perf_counter() - start

        peak_gb = reportable_memory_gb(backend, mem.delta_gb)

        return ExecutionTelemetry(
            **base,
            device_detail=result.device_detail,
            success=True,
            actual_runtime_seconds=round(runtime, 6),
            actual_peak_memory_gb=round(peak_gb, 6),
            runtime_error_ratio=_ratio(runtime, estimate.estimated_runtime_seconds),
            memory_error_ratio=(
                None if backend == "mpi_cluster"
                else _ratio(peak_gb, estimate.estimated_memory_gb)
            ),
            result_metric_name=result.metric_name,
            result_metric_value=result.metric_value,
            raw_metrics={
                **result.detail,
                # How a scaling study should read this metric across rank
                # counts — see WorkloadResult for what each value means.
                "metric_rank_behaviour": result.metric_rank_behaviour,
                "metric_target": result.metric_target,
                "metric_tolerance": result.metric_tolerance,
                "metric_note": result.metric_note,
                "mpi_ranks": mpi_ranks if backend == "mpi_cluster" else None,
                "memory_note": (
                    "not measured: peak RSS of the launching process excludes "
                    "MPI child ranks" if backend == "mpi_cluster" else None
                ),
            },
        )

    except (WorkloadUnavailable, subprocess.TimeoutExpired) as exc:
        return ExecutionTelemetry(
            **base,
            device_detail="none — workload unavailable",
            success=False,
            error_message=f"{type(exc).__name__}: {exc}",
            actual_runtime_seconds=round(time.perf_counter() - start, 6),
            actual_peak_memory_gb=0.0,
            raw_metrics={"unavailable": True},
        )

    except Exception as exc:  # noqa: BLE001 - record, never swallow
        return ExecutionTelemetry(
            **base,
            device_detail="none — execution failed",
            success=False,
            error_message=f"{type(exc).__name__}: {exc}",
            actual_runtime_seconds=round(time.perf_counter() - start, 6),
            actual_peak_memory_gb=0.0,
            raw_metrics={"failed": True},
        )
