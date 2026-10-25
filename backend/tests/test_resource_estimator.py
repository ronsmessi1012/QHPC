"""
Step 6 — Resource Estimator tests.

Run:  pytest backend/tests/test_resource_estimator.py -v

These tests pin down the behaviour Step 7 depends on. They also act as the
contract check against backend/models/: if ProblemSpecification or
ExecutionDecision field names differ from what the estimator reads, these
fail loudly at import or construction rather than producing wrong numbers.
"""

import math

import pytest

from backend.agents.resource_estimator import estimate_resources
from backend.models.decision_schema import ExecutionDecision
from backend.models.problem_schema import ProblemSpecification
from backend.utils.hardware_detect import (
    GPUDevice,
    HardwareInventory,
    PlanningMode,
)
from backend.utils.resource_constants import (
    CLASSICAL_RESOURCE_PROFILES,
    GPU_SPEEDUP_FACTOR,
    MAX_MPI_SPEEDUP,
    MAX_REPORTED_MEMORY_GB,
    MPI_PARALLEL_EFFICIENCY,
    QUANTUM_RESOURCE_PROFILES,
    RESOURCE_TIER_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Fixed inventories.
#
# Every test injects one of these rather than probing the host, so results
# do not change between a laptop with no GPU and a CI box with four.
# ---------------------------------------------------------------------------

FULL_HW = HardwareInventory(
    planning_mode=PlanningMode.DETECTED,
    cpu_logical_cores=16, cpu_physical_cores=8, total_ram_gb=16.0,
    cuda_available=True,
    cuda_devices=[GPUDevice(name="NVIDIA RTX 4090", vram_gb=24.0)],
    mpi_available=True, mpi_slots=8,
)

CPU_ONLY_HW = HardwareInventory(
    planning_mode=PlanningMode.DETECTED,
    cpu_logical_cores=16, cpu_physical_cores=8, total_ram_gb=16.0,
    cuda_available=False, cuda_devices=[],
    mpi_available=False, mpi_slots=1,
)

TARGET_HW = HardwareInventory(
    planning_mode=PlanningMode.TARGET,
    cpu_logical_cores=16, cpu_physical_cores=8, total_ram_gb=16.0,
    cuda_available=True,
    cuda_devices=[GPUDevice(name="NVIDIA A100", vram_gb=40.0)],
    mpi_available=True, mpi_slots=8,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def make_spec(dimensionality=100, dataset_size=1000, problem_class="optimization",
              constraint_structure="constrained"):
    return ProblemSpecification(
        user_query="test query",
        problem_class=problem_class,
        confidence=0.8,
        dimensionality=dimensionality,
        dataset_size=dataset_size,
        constraint_structure=constraint_structure,
        quantum_candidate=True,
    )


def estimate(decision, spec, inventory=FULL_HW):
    """All tests go through here so hardware is always explicit."""
    return estimate_resources(decision, spec, inventory=inventory)


def make_decision(family, algorithm, routing_mode="borderline_review"):
    return ExecutionDecision(
        selected_family=family,
        selected_algorithm=algorithm,
        routing_mode=routing_mode,
        decision_confidence=0.70,
        borderline_score=0.90,
        explanation="test decision",
    )


# ---------------------------------------------------------------------------
# Registry coverage — every algorithm must have a profile
# ---------------------------------------------------------------------------

def test_every_registry_algorithm_has_a_profile():
    """
    Guards the silent-fallback failure mode: a name mismatch between
    utils/constants.py and utils/resource_constants.py would quietly route
    an algorithm to the default profile and produce meaningless estimates.
    """
    from backend.utils.constants import CLASSICAL_ALGORITHMS, QUANTUM_ALGORITHMS

    classical_names = {
        entry["name"]
        for algos in CLASSICAL_ALGORITHMS.values()
        for entry in algos
    }
    quantum_names = {
        entry["name"]
        for algos in QUANTUM_ALGORITHMS.values()
        for entry in algos
    }

    missing_classical = classical_names - set(CLASSICAL_RESOURCE_PROFILES)
    missing_quantum = quantum_names - set(QUANTUM_RESOURCE_PROFILES)

    assert not missing_classical, f"classical algorithms with no profile: {missing_classical}"
    assert not missing_quantum, f"quantum algorithms with no profile: {missing_quantum}"


def test_no_orphan_profiles():
    """A profile for an algorithm that no longer exists is dead weight."""
    from backend.utils.constants import CLASSICAL_ALGORITHMS, QUANTUM_ALGORITHMS

    classical_names = {
        e["name"] for a in CLASSICAL_ALGORITHMS.values() for e in a
    }
    quantum_names = {
        e["name"] for a in QUANTUM_ALGORITHMS.values() for e in a
    }

    assert not set(CLASSICAL_RESOURCE_PROFILES) - classical_names
    assert not set(QUANTUM_RESOURCE_PROFILES) - quantum_names


@pytest.mark.parametrize("name,profile", CLASSICAL_RESOURCE_PROFILES.items())
def test_classical_profile_shape(name, profile):
    required = {
        "size_driver", "ref_size", "base_memory_gb", "memory_exp",
        "base_runtime_s", "runtime_exp", "gpu_capable", "mpi_capable",
        "calibrated", "calibration_host",
    }
    assert set(profile) == required, f"{name} has wrong fields"
    assert profile["size_driver"] in ("dataset_size", "dimensionality")
    assert profile["ref_size"] > 0
    assert profile["base_memory_gb"] > 0
    assert profile["base_runtime_s"] > 0
    assert profile["memory_exp"] > 0
    assert profile["runtime_exp"] > 0


@pytest.mark.parametrize("name,profile", QUANTUM_RESOURCE_PROFILES.items())
def test_quantum_profile_shape(name, profile):
    required = {
        "qubit_mode", "qubit_coeff", "depth_layers", "depth_multiplier",
        "sim_strategy", "bond_dimension", "calibrated", "calibration_host",
    }
    assert set(profile) == required, f"{name} has wrong fields"
    assert profile["qubit_mode"] in ("linear", "log2")
    assert profile["sim_strategy"] in ("statevector", "mps")
    if profile["sim_strategy"] == "mps":
        assert profile["bond_dimension"] and profile["bond_dimension"] > 0
    else:
        assert profile["bond_dimension"] is None


# ---------------------------------------------------------------------------
# Classical estimation
# ---------------------------------------------------------------------------

def test_classical_scales_with_its_own_size_driver():
    """
    XGBoost is driven by dataset_size, so changing dimensionality alone must
    not move the estimate. This is the bug that made optimization problems
    look trivial and ML problems look enormous under a single global driver.
    """
    dec = make_decision("classical", "XGBoost")
    a = estimate(dec, make_spec(dimensionality=10, dataset_size=10_000))
    b = estimate(dec, make_spec(dimensionality=5_000, dataset_size=10_000))
    assert a.estimated_memory_gb == b.estimated_memory_gb


def test_optimization_algorithm_is_driven_by_dimensionality():
    dec = make_decision("classical", "Branch and Bound")
    small = estimate(dec, make_spec(dimensionality=100, dataset_size=100))
    large = estimate(dec, make_spec(dimensionality=500, dataset_size=100))
    assert large.estimated_memory_gb > small.estimated_memory_gb


def test_classical_estimate_is_monotonic_in_size():
    dec = make_decision("classical", "Random Forest")
    prev_mem = prev_rt = 0.0
    for n in (1_000, 5_000, 10_000, 50_000):
        est = estimate(dec, make_spec(dataset_size=n))
        assert est.estimated_memory_gb > prev_mem
        assert est.estimated_runtime_seconds > prev_rt
        prev_mem, prev_rt = est.estimated_memory_gb, est.estimated_runtime_seconds


def test_classical_matches_closed_form():
    """Estimator output must equal the documented formula exactly."""
    profile = CLASSICAL_RESOURCE_PROFILES["XGBoost"]
    dataset_size = 20_000
    est = estimate(
        make_decision("classical", "XGBoost"),
        make_spec(dataset_size=dataset_size),
    )
    scale = dataset_size / profile["ref_size"]
    expected_mem = profile["base_memory_gb"] * scale ** profile["memory_exp"]
    expected_rt = (
        profile["base_runtime_s"] * scale ** profile["runtime_exp"]
    ) / GPU_SPEEDUP_FACTOR  # XGBoost is gpu_capable; FULL_HW has a CUDA device

    # The estimator rounds its outputs to 6 decimal places, so compare at
    # that granularity rather than with a tighter relative tolerance.
    assert est.estimated_memory_gb == pytest.approx(expected_mem, abs=1e-6)
    assert est.estimated_runtime_seconds == pytest.approx(expected_rt, abs=1e-6)


def test_classical_has_no_quantum_fields():
    est = estimate(make_decision("classical", "XGBoost"), make_spec())
    assert est.qubit_count is None
    assert est.circuit_depth is None
    assert est.simulation_strategy is None


def test_parallelization_factor_reflects_capability():
    expected_mpi = min(
        1.0 + (FULL_HW.mpi_slots - 1) * MPI_PARALLEL_EFFICIENCY, MAX_MPI_SPEEDUP)

    serial = estimate(
        make_decision("classical", "Logistic Regression"), make_spec())
    assert serial.parallelization_factor == 1.0

    gpu = estimate(make_decision("classical", "XGBoost"), make_spec())
    assert gpu.parallelization_factor == GPU_SPEEDUP_FACTOR

    mpi = estimate(
        make_decision("classical", "Genetic Algorithm"), make_spec())
    assert mpi.parallelization_factor == pytest.approx(expected_mpi)


# ---------------------------------------------------------------------------
# Hardware awareness — speedup is only claimed for hardware that exists
# ---------------------------------------------------------------------------

def test_no_gpu_speedup_claimed_without_a_gpu():
    """
    The bug this guards: a GPU-capable algorithm on a machine with no CUDA
    device previously still had its runtime divided by GPU_SPEEDUP_FACTOR,
    reporting a speedup that could not happen.
    """
    with_gpu = estimate(make_decision("classical", "XGBoost"),
                        make_spec(dataset_size=20_000), inventory=FULL_HW)
    without = estimate(make_decision("classical", "XGBoost"),
                       make_spec(dataset_size=20_000), inventory=CPU_ONLY_HW)

    assert with_gpu.parallelization_factor == GPU_SPEEDUP_FACTOR
    assert without.parallelization_factor == 1.0
    assert without.estimated_runtime_seconds > with_gpu.estimated_runtime_seconds


def test_no_mpi_speedup_claimed_without_mpi():
    with_mpi = estimate(make_decision("classical", "Genetic Algorithm"),
                        make_spec(dimensionality=500), inventory=FULL_HW)
    without = estimate(make_decision("classical", "Genetic Algorithm"),
                       make_spec(dimensionality=500), inventory=CPU_ONLY_HW)

    assert with_mpi.parallelization_factor > 1.0
    assert without.parallelization_factor == 1.0


def test_mpi_speedup_scales_with_actual_slot_count():
    """Slot count comes from the inventory, not a hardcoded node count."""
    few = HardwareInventory(
        planning_mode=PlanningMode.DETECTED,
        cpu_logical_cores=4, cpu_physical_cores=4, total_ram_gb=8.0,
        cuda_available=False, mpi_available=True, mpi_slots=4)
    many = HardwareInventory(
        planning_mode=PlanningMode.DETECTED,
        cpu_logical_cores=16, cpu_physical_cores=16, total_ram_gb=32.0,
        cuda_available=False, mpi_available=True, mpi_slots=16)

    a = estimate(make_decision("classical", "Simulated Annealing"),
                 make_spec(dimensionality=500), inventory=few)
    b = estimate(make_decision("classical", "Simulated Annealing"),
                 make_spec(dimensionality=500), inventory=many)
    assert b.parallelization_factor > a.parallelization_factor


def test_mpi_speedup_is_capped():
    huge = HardwareInventory(
        planning_mode=PlanningMode.TARGET,
        cpu_logical_cores=16, cpu_physical_cores=16, total_ram_gb=32.0,
        cuda_available=False, mpi_available=True, mpi_slots=10_000)
    est = estimate(make_decision("classical", "Genetic Algorithm"),
                   make_spec(dimensionality=500), inventory=huge)
    assert est.parallelization_factor <= MAX_MPI_SPEEDUP


def test_target_mode_marks_the_estimate_hypothetical():
    detected = estimate(make_decision("classical", "XGBoost"), make_spec(),
                        inventory=FULL_HW)
    target = estimate(make_decision("classical", "XGBoost"), make_spec(),
                      inventory=TARGET_HW)

    assert detected.hardware_is_hypothetical is False
    assert target.hardware_is_hypothetical is True
    assert target.reasoning_trace["planning_mode"] == "target"


def test_trace_records_which_parallelism_was_applied():
    est = estimate(make_decision("classical", "XGBoost"), make_spec(),
                   inventory=CPU_ONLY_HW)
    assert est.reasoning_trace["parallelism_applied"] == ["none"]
    assert est.reasoning_trace["mpi_slots_used"] == 0

    est = estimate(make_decision("classical", "Genetic Algorithm"),
                   make_spec(), inventory=FULL_HW)
    assert "mpi" in est.reasoning_trace["parallelism_applied"]
    assert est.reasoning_trace["mpi_slots_used"] == FULL_HW.mpi_slots


def test_quantum_estimates_are_hardware_independent():
    """Simulation cost is set by the circuit, not by the host's GPU."""
    a = estimate(make_decision("quantum", "Grover Search"),
                 make_spec(dataset_size=1_000), inventory=FULL_HW)
    b = estimate(make_decision("quantum", "Grover Search"),
                 make_spec(dataset_size=1_000), inventory=CPU_ONLY_HW)
    assert a.estimated_memory_gb == b.estimated_memory_gb
    assert a.estimated_runtime_seconds == b.estimated_runtime_seconds


def test_unknown_algorithm_uses_default_profile():
    est = estimate(
        make_decision("classical", "Nonexistent Algorithm"), make_spec())
    assert est.estimated_memory_gb > 0
    assert est.calibrated is False


# ---------------------------------------------------------------------------
# Quantum estimation
# ---------------------------------------------------------------------------

def test_linear_qubit_mode_tracks_dimensionality():
    est = estimate(
        make_decision("quantum", "Quantum Approximate Optimization Algorithm (QAOA)"),
        make_spec(dimensionality=500),
    )
    assert est.qubit_count == 500


def test_log2_qubit_mode_tracks_dataset_size():
    """Grover indexes a search space, so qubits grow as log2(N), not N."""
    est = estimate(
        make_decision("quantum", "Grover Search"),
        make_spec(dimensionality=1_000, dataset_size=1_000),
    )
    assert est.qubit_count == math.ceil(math.log2(1_000))  # 10


def test_variational_algorithms_use_mps_not_statevector():
    """
    Regression test for the 1e142 GB bug: applying the exact-statevector
    model to a 500-variable QAOA instance produced a physically meaningless
    number. Variational circuits must use the polynomial MPS model.
    """
    est = estimate(
        make_decision("quantum", "Quantum Approximate Optimization Algorithm (QAOA)"),
        make_spec(dimensionality=500),
    )
    assert est.simulation_strategy == "mps"
    assert est.estimated_memory_gb < 1.0, (
        f"500-qubit QAOA reported {est.estimated_memory_gb} GB; MPS memory is "
        f"polynomial in qubit count and must stay small"
    )


def test_exact_simulation_algorithms_use_statevector():
    for algorithm in ("Quantum Statevector Simulation", "Hamiltonian Simulation",
                      "Grover Search"):
        est = estimate(
            make_decision("quantum", algorithm), make_spec(dimensionality=12))
        assert est.simulation_strategy == "statevector", algorithm


def test_statevector_memory_is_exponential():
    """2^n * 16 bytes, exactly."""
    n = 20
    est = estimate(
        make_decision("quantum", "Quantum Statevector Simulation"),
        make_spec(dimensionality=n),
    )
    expected_gb = (2 ** n) * 16 / (1024 ** 3)
    assert est.qubit_count == n
    assert est.estimated_memory_gb == pytest.approx(expected_gb, rel=1e-6)


def test_statevector_memory_doubles_per_qubit():
    def mem(n):
        return estimate(
            make_decision("quantum", "Quantum Statevector Simulation"),
            make_spec(dimensionality=n),
        ).estimated_memory_gb

    assert mem(21) == pytest.approx(2 * mem(20), rel=1e-6)


def test_mps_memory_is_polynomial_not_exponential():
    def mem(n):
        return estimate(
            make_decision("quantum", "Quantum Approximate Optimization Algorithm (QAOA)"),
            make_spec(dimensionality=n),
        ).estimated_memory_gb

    # Doubling qubits should roughly double MPS memory (linear in n),
    # nowhere near squaring it.
    assert mem(200) == pytest.approx(2 * mem(100), rel=1e-6)


def test_absurd_qubit_count_is_clamped_not_overflowed():
    """A 5000-qubit request must not raise OverflowError or print 1e1500."""
    est = estimate(
        make_decision("quantum", "Hamiltonian Simulation"),
        make_spec(dimensionality=5_000),
    )
    assert est.estimated_memory_gb <= MAX_REPORTED_MEMORY_GB
    assert math.isfinite(est.estimated_memory_gb)
    assert math.isfinite(est.estimated_runtime_seconds)


def test_circuit_depth_is_positive_for_all_quantum_algorithms():
    for algorithm in QUANTUM_RESOURCE_PROFILES:
        est = estimate(
            make_decision("quantum", algorithm), make_spec(dimensionality=8))
        assert est.circuit_depth is not None and est.circuit_depth >= 1, algorithm
        assert est.qubit_count is not None and est.qubit_count >= 1, algorithm


# ---------------------------------------------------------------------------
# Tiering, calibration flag, audit trail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("memory,expected", [
    (0.5, "light"),
    (RESOURCE_TIER_THRESHOLDS["light"], "light"),
    (RESOURCE_TIER_THRESHOLDS["light"] + 0.01, "moderate"),
    (RESOURCE_TIER_THRESHOLDS["moderate"], "moderate"),
    (RESOURCE_TIER_THRESHOLDS["moderate"] + 0.01, "heavy"),
    (RESOURCE_TIER_THRESHOLDS["heavy"], "heavy"),
    (RESOURCE_TIER_THRESHOLDS["heavy"] + 0.01, "extreme"),
])
def test_resource_tier_boundaries(memory, expected):
    from backend.agents.resource_estimator import _resource_tier
    assert _resource_tier(memory) == expected


def test_calibrated_flag_is_false_until_benchmarked():
    """
    Every shipped profile is a reasoned default, not a measurement. If this
    starts failing, someone calibrated a profile — update the test and say so
    in the write-up.
    """
    est = estimate(make_decision("classical", "XGBoost"), make_spec())
    assert est.calibrated is False
    assert est.reasoning_trace["calibrated"] is False


def test_reasoning_trace_carries_the_full_derivation():
    est = estimate(
        make_decision("classical", "Genetic Algorithm"),
        make_spec(dimensionality=500),
    )
    trace = est.reasoning_trace
    for key in ("selected_family", "selected_algorithm", "dimensionality",
                "dataset_size", "profile_used", "raw_memory_gb", "raw_runtime_s",
                "parallelization_factor", "final_memory_gb", "final_runtime_s",
                "size_driver", "driver_value", "ref_size", "scale", "calibrated"):
        assert key in trace, f"reasoning_trace missing {key}"


def test_quantum_trace_carries_simulation_details():
    est = estimate(
        make_decision("quantum", "Variational Quantum Eigensolver (VQE)"),
        make_spec(dimensionality=10),
    )
    for key in ("gate_count", "simulation_strategy", "bond_dimension"):
        assert key in est.reasoning_trace, f"reasoning_trace missing {key}"


def test_runtime_is_never_negative_or_nan():
    for family, algorithms in (
        ("classical", CLASSICAL_RESOURCE_PROFILES),
        ("quantum", QUANTUM_RESOURCE_PROFILES),
    ):
        for algorithm in algorithms:
            est = estimate(
                make_decision(family, algorithm),
                make_spec(dimensionality=16, dataset_size=1_000),
            )
            assert est.estimated_memory_gb >= 0
            assert est.estimated_runtime_seconds >= 0
            assert math.isfinite(est.estimated_memory_gb)
            assert math.isfinite(est.estimated_runtime_seconds)


def test_minimum_problem_size_does_not_divide_by_zero():
    est = estimate(
        make_decision("classical", "Dijkstra"),
        make_spec(dimensionality=1, dataset_size=1),
    )
    assert est.estimated_memory_gb > 0


def test_calibrated_profiles_must_name_their_host():
    """
    A profile cannot claim to be measured without saying where. base_runtime_s
    does not transfer between machines, so an unattributed runtime figure is
    not reportable.
    """
    for name, profile in list(CLASSICAL_RESOURCE_PROFILES.items()) + \
            list(QUANTUM_RESOURCE_PROFILES.items()):
        if profile["calibrated"]:
            assert profile["calibration_host"], (
                f"{name} is marked calibrated but names no calibration_host"
            )
