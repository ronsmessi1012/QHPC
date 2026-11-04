"""
Step 8 — MPI rank worker

Launched by backend/engine/performance_engine.py as:

    mpiexec -n <N> python -m backend.engine.mpi_worker --algorithm ... \
        --dataset-size ... --dimensionality ...

Why a separate module: `mpiexec -n N` starts N independent interpreters and
MPI_Init runs inside each. That cannot be entered from a process that is
already running, so the engine shells out and this module is what each rank
executes.

Rank 0 prints a single line to stdout:

    QHPC_RESULT {"metric_name": ..., "metric_value": ..., ...}

The prefix matters. MPI launchers write their own diagnostics to stdout, so
the engine scans for this marker rather than trying to parse the whole
stream. Other ranks print nothing, or N copies would appear.

Run directly (without mpiexec) and it still works as a single rank, which
makes debugging the workload itself straightforward.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from backend.engine.workloads import WorkloadUnavailable, get_workload

RESULT_PREFIX = "QHPC_RESULT "


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument("--dimensionality", type=int, required=True)
    args = parser.parse_args()

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception as exc:  # noqa: BLE001
        # No MPI available. The workload can still be exercised serially,
        # which is useful when debugging the workload itself, but the
        # caller MUST be able to tell that apart from a real parallel run.
        # Reporting it as a plain success would let a serial run
        # masquerade as a distributed one in the telemetry.
        print(f"[mpi_worker] mpi4py unavailable ({type(exc).__name__}: {exc}); "
              f"running as a single process", file=sys.stderr)
        comm, rank = None, 0
        mpi_fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        mpi_fallback_reason = None

    try:
        workload = get_workload(args.algorithm)
    except WorkloadUnavailable as exc:
        if rank == 0:
            print(f"[mpi_worker] {exc}", file=sys.stderr)
        return 2

    if "mpi_cluster" not in workload.supported_backends:
        if rank == 0:
            print(f"[mpi_worker] '{args.algorithm}' has no MPI implementation "
                  f"(supports: {workload.supported_backends})", file=sys.stderr)
        return 3

    start = time.perf_counter()
    try:
        result = workload.run(
            dataset_size=args.dataset_size,
            dimensionality=args.dimensionality,
            backend="mpi_cluster",
            mpi_comm=comm,
        )
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            print(f"[mpi_worker] workload raised {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        return 4
    elapsed = time.perf_counter() - start

    # Only rank 0 reports, or the engine would read N duplicate results.
    if rank == 0:
        payload = {
            "metric_name": result.metric_name,
            "metric_value": result.metric_value,
            "device_detail": result.device_detail,
            "detail": {
                **result.detail,
                "rank0_elapsed_seconds": round(elapsed, 6),
                "comm_size": comm.Get_size() if comm is not None else 1,
            },
            # Must cross the subprocess boundary, or a scaling study loses
            # the information it needs to interpret the metric.
            "metric_rank_behaviour": result.metric_rank_behaviour,
            "metric_target": result.metric_target,
            "metric_tolerance": result.metric_tolerance,
            "metric_note": result.metric_note,
            # None on a real MPI run; a reason string when mpi4py was
            # unavailable and this ran serially despite being launched
            # under mpiexec.
            "mpi_fallback_reason": mpi_fallback_reason,
            "comm_size": comm.Get_size() if comm is not None else 1,
        }
        print(RESULT_PREFIX + json.dumps(payload))
        sys.stdout.flush()

    if comm is not None:
        comm.Barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
