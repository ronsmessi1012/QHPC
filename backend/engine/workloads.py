"""
Step 8 — Executable Workloads

The registry in backend/utils/constants.py holds algorithm NAMES and expert
prior scores. Nothing in the framework ever ran one. This module supplies
the actual runnable implementations the Performance Engine dispatches to.

SCOPE — deliberately a subset, not all 22
-----------------------------------------
Implementing all 22 registry algorithms would be a library, not a thesis
contribution. This module implements a representative subset chosen so that
every execution backend has at least one workload that genuinely exercises
it:

    XGBoost                 cpu, gpu_cuda       device= is a real switch
    Monte Carlo Simulation  cpu, mpi_cluster    embarrassingly parallel
    Genetic Algorithm       cpu, mpi_cluster    population splits per rank
    Grover Search           qpu_simulator       statevector, CPU or GPU
    QAOA                    qpu_simulator       MPS (shallow variational)

An algorithm the registry names but this module does not implement is
reported as unsupported rather than silently skipped — a missing
implementation must not look like a successful run of nothing.

Every workload returns a correctness metric. Timing alone only proves
something consumed CPU; the metric proves it solved the problem.

Optional dependencies (xgboost, qiskit, qiskit-aer) are imported lazily
inside each workload, so a host missing one can still run the others. The
real exception is always reported, never flattened to "not installed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Fixed seed everywhere: a benchmark whose input changes between runs
# cannot be compared against itself.
RANDOM_SEED = 0

# Synthetic feature count for supervised workloads.
N_FEATURES = 50

# Monte Carlo draws samples in batches of this size. Peak memory is then
# fixed at ~16 bytes x SAMPLE_CHUNK regardless of the total sample count,
# instead of growing until the process is killed.
SAMPLE_CHUNK = 2_000_000


class WorkloadUnavailable(RuntimeError):
    """Raised when a workload cannot run here, carrying the real reason."""


@dataclass
class WorkloadResult:
    """What a workload produced, beyond the timing the engine measures."""

    metric_name: str
    metric_value: float
    device_detail: str
    detail: Dict[str, Any] = field(default_factory=dict)

    # --- How to read the metric across rank counts ------------------------
    # A scaling study compares the same workload at 1, 2, 4 ... ranks. If
    # the metric moves, that may be a broken parallel implementation OR an
    # expected property of the algorithm. Conflating the two makes the
    # check useless: it either cries wolf or misses real corruption.
    #
    #   "invariant"  the metric MUST NOT move. Any drift is a bug.
    #   "stochastic" the metric varies with the random streams, which
    #                differ per rank by design. Judge it against
    #                metric_target/metric_tolerance, not against itself.
    #   "rank_dependent"
    #                the metric legitimately changes with rank count,
    #                because the algorithm's behaviour does. Report the
    #                trend; do not treat it as corruption.
    metric_rank_behaviour: str = "invariant"

    # For "stochastic": the value a correct run converges toward, and how
    # far from it is still acceptable. Lets a scaling study verify
    # correctness at every rank count without demanding bit-identity.
    metric_target: Optional[float] = None
    metric_tolerance: Optional[float] = None

    # Plain-language note explaining the expected behaviour, printed by the
    # scaling benchmark so a reader is not left guessing.
    metric_note: str = ""


@dataclass
class Workload:
    name: str
    supported_backends: List[str]
    run: Callable[..., WorkloadResult]
    notes: str = ""


# ---------------------------------------------------------------------------
# Classical — supervised learning
# ---------------------------------------------------------------------------

def _xgboost(dataset_size: int, dimensionality: int, backend: str,
             **_: Any) -> WorkloadResult:
    """
    Gradient-boosted trees. The CPU/GPU switch is `device=`, which is a real
    dispatch inside XGBoost rather than a wrapper, so the same call measures
    both backends fairly.
    """
    try:
        import xgboost as xgb
    except Exception as exc:  # noqa: BLE001
        raise WorkloadUnavailable(
            f"xgboost unusable: {type(exc).__name__}: {exc}"
        ) from exc

    device = "cuda" if backend == "gpu_cuda" else "cpu"

    rng = np.random.default_rng(RANDOM_SEED)
    n_features = max(dimensionality, 2) if dimensionality <= 200 else N_FEATURES
    X = rng.standard_normal((dataset_size, n_features), dtype=np.float32)
    # A learnable signal, so accuracy means something rather than hovering
    # at chance: label depends on a linear combination of the features.
    weights = rng.standard_normal(n_features)
    y = (X @ weights > 0).astype(np.int32)

    split = int(dataset_size * 0.8)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, tree_method="hist",
        device=device, verbosity=0,
    )
    model.fit(X[:split], y[:split])
    preds = model.predict(X[split:])
    accuracy = float((preds == y[split:]).mean())

    return WorkloadResult(
        metric_name="accuracy",
        metric_value=accuracy,
        device_detail=f"xgboost device={device}",
        detail={"n_estimators": 200, "max_depth": 6,
                "n_features": n_features, "train_rows": split,
                "test_rows": dataset_size - split},
    )


# ---------------------------------------------------------------------------
# Classical — simulation
# ---------------------------------------------------------------------------

def _monte_carlo(dataset_size: int, dimensionality: int, backend: str,
                 mpi_comm: Optional[Any] = None, **_: Any) -> WorkloadResult:
    """
    Monte Carlo estimation of pi.

    Chosen because it is embarrassingly parallel with a known exact answer,
    which makes it the cleanest possible MPI demonstration: the error
    against the true value confirms the ranks' partial results were
    combined correctly, not merely that they all finished.

    Under MPI each rank samples a disjoint slice with its own seed, then
    the counts are reduced to rank 0.
    """
    total_samples = max(dataset_size, 1) * 1000

    def count_inside(rng, n: int) -> int:
        """
        Count samples inside the unit circle, in chunks.

        Allocating all n points at once costs 16n bytes — at 4e8 samples
        that is 6.4 GB, enough to exhaust a WSL instance before the
        benchmark produces a number. Chunking keeps peak memory flat and
        independent of the sample count.
        """
        inside = 0
        remaining = n
        while remaining > 0:
            batch = min(remaining, SAMPLE_CHUNK)
            pts = rng.random((batch, 2))
            inside += int(((pts ** 2).sum(axis=1) <= 1.0).sum())
            remaining -= batch
        return inside

    if mpi_comm is not None:
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()
        # Distinct seed per rank, or every rank draws identical samples and
        # the "parallel" result is just one rank's answer repeated.
        rng = np.random.default_rng(RANDOM_SEED + rank)
        local_n = total_samples // size + (1 if rank < total_samples % size else 0)
        local_inside = count_inside(rng, local_n)

        total_inside = mpi_comm.allreduce(local_inside)
        counted = mpi_comm.allreduce(local_n)
        device_detail = f"mpi4py, {size} ranks"
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        total_inside = count_inside(rng, total_samples)
        counted = total_samples
        device_detail = "numpy, single process"

    pi_estimate = 4.0 * total_inside / counted
    abs_error = abs(pi_estimate - np.pi)

    # Statistical error of a Monte Carlo estimate falls as 1/sqrt(N). Three
    # standard errors is a generous but still meaningful bound: a correct
    # run lands inside it, a broken reduction does not.
    tolerance = 3.0 * (np.pi * (4 - np.pi) / counted) ** 0.5

    return WorkloadResult(
        metric_name="pi_absolute_error",
        metric_value=float(abs_error),
        device_detail=device_detail,
        detail={"samples": counted, "pi_estimate": pi_estimate,
                "inside": total_inside, "tolerance_3sigma": tolerance},
        # Each rank seeds differently, so more ranks means a different
        # mixture of random streams and a different estimate. The sample
        # COUNT is identical, so the error must stay within sampling noise.
        metric_rank_behaviour="stochastic",
        metric_target=0.0,
        metric_tolerance=float(tolerance),
        metric_note=(
            "pi error varies with rank count because each rank uses its own "
            "seed; total samples are unchanged, so it must stay within "
            "3-sigma sampling noise"
        ),
    )


# ---------------------------------------------------------------------------
# Classical — optimization
# ---------------------------------------------------------------------------

def _genetic_algorithm(dataset_size: int, dimensionality: int, backend: str,
                       mpi_comm: Optional[Any] = None,
                       **_: Any) -> WorkloadResult:
    """
    Genetic algorithm minimising the Rastrigin function.

    `dataset_size` sets the GENERATION COUNT here. A GA has no dataset; its
    work scales with generations and population, and generations is the
    knob that makes the run long enough to measure without changing the
    search's character.

    Rastrigin is a standard multimodal benchmark with a known global minimum
    of 0 at the origin, so the final best fitness is a meaningful quality
    metric rather than an arbitrary number.

    Under MPI each rank evolves a sub-population and the best genome is
    shared each generation — island-model parallelism, which is the normal
    way GAs distribute.
    """
    n_dims = max(dimensionality, 2)
    population_size = 200
    generations = max(int(dataset_size), 10)

    min_op = None
    if mpi_comm is not None:
        # mpi4py.MPI is a submodule: `import mpi4py` alone does not expose it.
        from mpi4py import MPI
        min_op = MPI.MIN
        rank, size = mpi_comm.Get_rank(), mpi_comm.Get_size()
        local_pop = max(population_size // size, 8)
        rng = np.random.default_rng(RANDOM_SEED + rank)
        device_detail = f"mpi4py island model, {size} ranks"
    else:
        rank, size = 0, 1
        local_pop = population_size
        rng = np.random.default_rng(RANDOM_SEED)
        device_detail = "numpy, single process"

    def rastrigin(pop: np.ndarray) -> np.ndarray:
        return 10 * n_dims + np.sum(
            pop ** 2 - 10 * np.cos(2 * np.pi * pop), axis=1)

    pop = rng.uniform(-5.12, 5.12, (local_pop, n_dims))
    best = float("inf")

    for _ in range(generations):
        fitness = rastrigin(pop)
        order = np.argsort(fitness)
        pop = pop[order]
        best = min(best, float(fitness[order[0]]))

        # Elitist truncation selection + Gaussian mutation.
        elite = pop[: max(local_pop // 4, 2)]
        children = np.repeat(elite, int(np.ceil(local_pop / len(elite))),
                             axis=0)[:local_pop]
        children += rng.normal(0, 0.3, children.shape)
        pop = np.clip(children, -5.12, 5.12)
        pop[0] = elite[0]  # keep the best genome unmutated

        if mpi_comm is not None:
            # Migrate: every rank adopts the global best found so far.
            best = mpi_comm.allreduce(best, op=min_op)

    return WorkloadResult(
        metric_name="best_fitness",  # global optimum is 0.0
        metric_value=best,
        device_detail=device_detail,
        detail={"dimensions": n_dims, "generations": generations,
                "population_per_rank": local_pop, "ranks": size,
                "total_population": local_pop * size},
        # Strong scaling divides a FIXED total population across ranks, so
        # each island shrinks as ranks grow (200 individuals at 1 rank, 25
        # at 8) and search quality degrades. That is a real property of
        # island-model GAs under strong scaling, not a corrupted result.
        metric_rank_behaviour="rank_dependent",
        metric_target=0.0,
        metric_note=(
            "fitness worsens as ranks increase because the fixed total "
            "population is split into smaller islands; this is expected "
            "under strong scaling and is itself a finding"
        ),
    )


# ---------------------------------------------------------------------------
# Quantum — simulator workloads
# ---------------------------------------------------------------------------

def _grover(dataset_size: int, dimensionality: int, backend: str,
            use_gpu: bool = False, max_iterations: Optional[int] = None,
            **_: Any) -> WorkloadResult:
    """
    Grover search over a log2(N)-qubit index register.

    Correctness metric is the probability mass on the marked state, which
    should approach 1.0 after the optimal number of iterations. That is a
    real check: a broken oracle or diffuser leaves it near uniform.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:  # noqa: BLE001
        raise WorkloadUnavailable(
            f"qiskit / qiskit-aer unusable: {type(exc).__name__}: {exc}"
        ) from exc

    n_qubits = max(2, min(int(np.ceil(np.log2(max(dataset_size, 2)))), 24))
    marked = (1 << n_qubits) - 1  # mark the all-ones state

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    # Grover needs floor(pi/4 * sqrt(N)) iterations. This was previously
    # capped at 64 "to keep large registers runnable", which silently broke
    # the algorithm: P(marked) = sin^2((2k+1)*theta), so at 22 qubits 64 of
    # the required 1608 iterations leaves the success probability at 0.004 —
    # barely above the 1/N floor of an unsearched register. The circuit
    # still ran and still timed nicely, so the only thing that exposed it
    # was the correctness metric.
    #
    # The cost is now explicit instead of hidden: iterations grow as
    # sqrt(N), so a caller wanting a large register must accept the runtime
    # or choose fewer qubits. `max_iterations` exists for deliberate
    # truncation experiments and records that the result is not a valid
    # search.
    optimal_iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(2 ** n_qubits))))
    iterations = optimal_iterations
    truncated = False
    if max_iterations is not None and max_iterations < optimal_iterations:
        iterations = max(1, int(max_iterations))
        truncated = True

    for _ in range(iterations):
        # Oracle: phase-flip the all-ones state.
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        # Diffuser.
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    qc.save_statevector()
    device = "GPU" if use_gpu else "CPU"
    sim = AerSimulator(method="statevector", device=device)
    result = sim.run(transpile(qc, sim), shots=1).result()
    amplitudes = np.asarray(result.get_statevector())
    success_probability = float(abs(amplitudes[marked]) ** 2)

    # Theoretical success probability after k iterations, for comparison.
    theta = float(np.arcsin(1.0 / np.sqrt(2 ** n_qubits)))
    expected_probability = float(np.sin((2 * iterations + 1) * theta) ** 2)

    return WorkloadResult(
        metric_name="marked_state_probability",
        metric_value=success_probability,
        device_detail=f"qiskit-aer statevector device={device}",
        detail={"qubits": n_qubits, "grover_iterations": iterations,
                "optimal_iterations": optimal_iterations,
                "truncated": truncated, "marked_state": marked,
                "amplitudes": int(amplitudes.size),
                "expected_probability": expected_probability},
        # A correct Grover run concentrates nearly all amplitude on the
        # marked state regardless of rank or device.
        metric_rank_behaviour="invariant",
        metric_target=1.0,
        metric_tolerance=0.05,
        metric_note=(
            f"P(marked) should approach 1.0 after the optimal "
            f"{optimal_iterations} iterations"
            + (f"; THIS RUN WAS TRUNCATED to {iterations}, so it is a timing "
               f"measurement and NOT a valid search "
               f"(expected P={expected_probability:.4f})"
               if truncated else "")
        ),
    )


def _statevector_sim(dataset_size: int, dimensionality: int, backend: str,
                     use_gpu: bool = False, layers: int = 6,
                     **_: Any) -> WorkloadResult:
    """
    Fixed-depth entangling circuit, simulated exactly.

    Exists so that GPU-vs-CPU statevector scaling can be measured WITHOUT
    borrowing an algorithm's name. Grover was previously used for this and
    had to be truncated to stay runnable, which produced valid timings
    attached to a search that no longer worked — a timing benchmark
    masquerading as an algorithm.

    The circuit depth here is fixed and independent of qubit count, so
    runtime reflects statevector size (2^n amplitudes) rather than a
    changing amount of work. That is what makes the speedup across qubit
    counts comparable.

    Correctness metric is norm preservation: unitary evolution must leave
    sum|amplitude|^2 at exactly 1.0. A GPU kernel that corrupted amplitudes
    would show up here immediately.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:  # noqa: BLE001
        raise WorkloadUnavailable(
            f"qiskit / qiskit-aer unusable: {type(exc).__name__}: {exc}"
        ) from exc

    n_qubits = max(2, min(dimensionality, 32))

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(layers):
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.rz(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(1, n_qubits - 1, 2):
            qc.cx(q, q + 1)
        for q in range(n_qubits):
            qc.rx(float(rng.uniform(0, 2 * np.pi)), q)
    qc.save_statevector()

    device = "GPU" if use_gpu else "CPU"
    sim = AerSimulator(method="statevector", device=device)
    result = sim.run(transpile(qc, sim), shots=1).result()
    amplitudes = np.asarray(result.get_statevector())
    norm = float(np.sum(np.abs(amplitudes) ** 2))

    return WorkloadResult(
        metric_name="statevector_norm",
        metric_value=norm,
        device_detail=f"qiskit-aer statevector device={device}",
        detail={"qubits": n_qubits, "layers": layers,
                "amplitudes": int(amplitudes.size),
                "memory_gb": amplitudes.nbytes / (1024 ** 3)},
        metric_rank_behaviour="invariant",
        metric_target=1.0,
        metric_tolerance=1e-6,
        metric_note="unitary evolution preserves the norm; must stay 1.0",
    )


def _qaoa(dataset_size: int, dimensionality: int, backend: str,
          use_gpu: bool = False, bond_dimension: int = 64,
          **_: Any) -> WorkloadResult:
    """
    Shallow QAOA-style variational circuit on a ring MaxCut instance,
    simulated with matrix product states.

    MPS is the point: this is the workload whose statevector simulation
    would need 2^n amplitudes, and which Step 6 therefore models with the
    polynomial MPS cost instead. Running it here is what validates that
    modelling choice.

    Metric is the expected cut value from sampling, compared against the
    ring's known optimum.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except Exception as exc:  # noqa: BLE001
        raise WorkloadUnavailable(
            f"qiskit / qiskit-aer unusable: {type(exc).__name__}: {exc}"
        ) from exc

    # MPS keeps memory polynomial, but circuit construction and sampling
    # still cost time, so hold the demo to a tractable width.
    n_qubits = max(4, min(dimensionality, 64))
    layers = 2
    gamma, beta = 0.7, 0.4  # fixed angles: this measures simulation cost,
                            # not the quality of a classical optimiser

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(layers):
        for i in range(n_qubits):                 # ring cost Hamiltonian
            j = (i + 1) % n_qubits
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        for i in range(n_qubits):                 # mixer
            qc.rx(2 * beta, i)
    qc.measure_all()

    device = "GPU" if use_gpu else "CPU"
    sim = AerSimulator(method="matrix_product_state", device=device,
                       matrix_product_state_max_bond_dimension=bond_dimension)
    shots = 256
    counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()

    def cut_value(bitstring: str) -> int:
        bits = bitstring.replace(" ", "")[::-1]
        return sum(1 for i in range(n_qubits)
                   if bits[i] != bits[(i + 1) % n_qubits])

    expected_cut = sum(cut_value(b) * c for b, c in counts.items()) / shots
    optimum = n_qubits if n_qubits % 2 == 0 else n_qubits - 1

    return WorkloadResult(
        metric_name="approximation_ratio",
        metric_value=float(expected_cut / optimum),
        device_detail=(f"qiskit-aer matrix_product_state device={device}, "
                       f"bond_dimension={bond_dimension}"),
        detail={"qubits": n_qubits, "layers": layers, "shots": shots,
                "expected_cut": expected_cut, "ring_optimum": optimum,
                "bond_dimension": bond_dimension},
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Keys MUST match backend/utils/constants.py exactly. A mismatch here would
# silently report an algorithm as unimplemented.

WORKLOAD_REGISTRY: Dict[str, Workload] = {
    "XGBoost": Workload(
        name="XGBoost",
        supported_backends=["cpu", "gpu_cuda"],
        run=_xgboost,
        notes="device= switches between CPU and CUDA inside XGBoost itself",
    ),
    "Monte Carlo Simulation": Workload(
        name="Monte Carlo Simulation",
        supported_backends=["cpu", "mpi_cluster"],
        run=_monte_carlo,
        notes="embarrassingly parallel; known exact answer validates reduction",
    ),
    "Genetic Algorithm": Workload(
        name="Genetic Algorithm",
        supported_backends=["cpu", "mpi_cluster"],
        run=_genetic_algorithm,
        notes="island model across ranks; Rastrigin optimum is 0.0",
    ),
    "Grover Search": Workload(
        name="Grover Search",
        supported_backends=["qpu_simulator"],
        run=_grover,
        notes=("statevector; marked-state probability validates the oracle. "
               "Iterations grow as sqrt(N), so large registers are slow — "
               "use Quantum Statevector Simulation for pure timing work"),
    ),
    "Quantum Statevector Simulation": Workload(
        name="Quantum Statevector Simulation",
        supported_backends=["qpu_simulator"],
        run=_statevector_sim,
        notes=("fixed-depth circuit for GPU-vs-CPU statevector scaling; "
               "norm preservation is the correctness check"),
    ),
    "Quantum Approximate Optimization Algorithm (QAOA)": Workload(
        name="Quantum Approximate Optimization Algorithm (QAOA)",
        supported_backends=["qpu_simulator"],
        run=_qaoa,
        notes="MPS simulation; validates Step 6's polynomial memory model",
    ),
}


def get_workload(algorithm: str) -> Workload:
    """Look up a runnable implementation, or say plainly that there isn't one."""
    workload = WORKLOAD_REGISTRY.get(algorithm)
    if workload is None:
        raise WorkloadUnavailable(
            f"No executable implementation for '{algorithm}'. Implemented: "
            f"{sorted(WORKLOAD_REGISTRY)}"
        )
    return workload
