"""
Step 7 — Hardware Mapper tests.

Run:  pytest backend/tests/test_hardware_mapper.py -v

Covers every routing branch: quantum -> simulator (within and beyond the
per-strategy qubit ceiling), and classical -> CUDA / MPI / CPU.
"""

import pytest

from backend.engine.hardware_mapper import map_hardware
from backend.models.decision_schema import ExecutionDecision
from backend.models.hardware_schema import HardwareMapping
from backend.models.resource_schema import ResourceEstimate
from backend.utils.hardware_detect import (
    GPUDevice,
    HardwareInventory,
    PlanningMode,
)
from backend.utils.resource_constants import (
    GPU_MEMORY_THRESHOLD_GB,
    MPI_RUNTIME_THRESHOLD_S,
    MPS_QUBIT_LIMIT,
    STATEVECTOR_QUBIT_LIMIT,
)


# ---------------------------------------------------------------------------
# Fixed inventories, injected so tests do not depend on the host machine.
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

MPI_ONLY_HW = HardwareInventory(
    planning_mode=PlanningMode.DETECTED,
    cpu_logical_cores=16, cpu_physical_cores=8, total_ram_gb=16.0,
    cuda_available=False, cuda_devices=[],
    mpi_available=True, mpi_slots=16,
)

TARGET_HW = HardwareInventory(
    planning_mode=PlanningMode.TARGET,
    cpu_logical_cores=16, cpu_physical_cores=8, total_ram_gb=16.0,
    cuda_available=True,
    cuda_devices=[GPUDevice(name="NVIDIA A100", vram_gb=40.0)],
    mpi_available=True, mpi_slots=8,
)


def route(estimate, decision, inventory=FULL_HW):
    """All tests go through here so hardware is always explicit."""
    return map_hardware(estimate, decision, inventory=inventory)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_decision(family, algorithm):
    return ExecutionDecision(
        selected_family=family,
        selected_algorithm=algorithm,
        routing_mode="borderline_review",
        decision_confidence=0.70,
        borderline_score=0.90,
        explanation="test decision",
    )


def make_estimate(memory_gb=1.0, runtime_s=1.0, qubit_count=None,
                  circuit_depth=None, simulation_strategy=None,
                  tier="light", calibrated=False, trace=None):
    return ResourceEstimate(
        estimated_memory_gb=memory_gb,
        estimated_runtime_seconds=runtime_s,
        qubit_count=qubit_count,
        circuit_depth=circuit_depth,
        simulation_strategy=simulation_strategy,
        parallelization_factor=1.0,
        resource_tier=tier,
        calibrated=calibrated,
        reasoning_trace=trace or {},
    )


# ---------------------------------------------------------------------------
# Quantum routing
# ---------------------------------------------------------------------------

def test_quantum_always_routes_to_simulator():
    """No physical QPU exists in this deployment."""
    mapping = route(
        make_estimate(qubit_count=10, simulation_strategy="statevector"),
        make_decision("quantum", "Grover Search"),
    )
    assert mapping.backend_type == "qpu_simulator"
    assert mapping.fallback_backend == "cpu"


def test_statevector_within_limit_is_accepted():
    mapping = route(
        make_estimate(qubit_count=STATEVECTOR_QUBIT_LIMIT,
                      simulation_strategy="statevector"),
        make_decision("quantum", "Quantum Statevector Simulation"),
    )
    assert "EXCEEDS" not in mapping.device_spec


def test_statevector_beyond_limit_is_flagged():
    mapping = route(
        make_estimate(qubit_count=STATEVECTOR_QUBIT_LIMIT + 1,
                      simulation_strategy="statevector"),
        make_decision("quantum", "Quantum Statevector Simulation"),
    )
    assert mapping.backend_type == "qpu_simulator"
    assert "EXCEEDS" in mapping.device_spec
    assert "not classically simulable" in mapping.justification


def test_mps_gets_a_higher_ceiling_than_statevector():
    """
    A 500-qubit MPS circuit is fine; a 500-qubit statevector circuit is not.
    Same qubit count, different verdict — this is the whole point of
    tracking simulation strategy.
    """
    qubits = 500
    assert STATEVECTOR_QUBIT_LIMIT < qubits < MPS_QUBIT_LIMIT

    mps = route(
        make_estimate(qubit_count=qubits, simulation_strategy="mps",
                      trace={"bond_dimension": 64}),
        make_decision("quantum", "Quantum Approximate Optimization Algorithm (QAOA)"),
    )
    statevector = route(
        make_estimate(qubit_count=qubits, simulation_strategy="statevector"),
        make_decision("quantum", "Quantum Statevector Simulation"),
    )

    assert "EXCEEDS" not in mps.device_spec
    assert "EXCEEDS" in statevector.device_spec


def test_mps_beyond_its_own_limit_is_flagged():
    mapping = route(
        make_estimate(qubit_count=MPS_QUBIT_LIMIT + 1, simulation_strategy="mps",
                      trace={"bond_dimension": 64}),
        make_decision("quantum", "Quantum Approximate Optimization Algorithm (QAOA)"),
    )
    assert "EXCEEDS" in mapping.device_spec


def test_mps_device_spec_reports_bond_dimension():
    mapping = route(
        make_estimate(qubit_count=100, simulation_strategy="mps",
                      trace={"bond_dimension": 64}),
        make_decision("quantum", "Quantum Approximate Optimization Algorithm (QAOA)"),
    )
    assert "bond dimension" in mapping.device_spec


def test_quantum_trace_records_the_strategy_and_its_limit():
    mapping = route(
        make_estimate(qubit_count=12, simulation_strategy="statevector"),
        make_decision("quantum", "Grover Search"),
    )
    assert mapping.reasoning_trace["simulation_strategy"] == "statevector"
    assert mapping.reasoning_trace["qubit_limit_for_strategy"] == STATEVECTOR_QUBIT_LIMIT


# ---------------------------------------------------------------------------
# Classical routing
# ---------------------------------------------------------------------------

def test_gpu_capable_above_memory_threshold_routes_to_cuda():
    mapping = route(
        make_estimate(memory_gb=GPU_MEMORY_THRESHOLD_GB, runtime_s=1.0),
        make_decision("classical", "XGBoost"),  # gpu_capable
    )
    assert mapping.backend_type == "gpu_cuda"
    assert mapping.fallback_backend == "cpu"


def test_gpu_capable_below_memory_threshold_does_not_route_to_cuda():
    mapping = route(
        make_estimate(memory_gb=GPU_MEMORY_THRESHOLD_GB - 0.01, runtime_s=1.0),
        make_decision("classical", "XGBoost"),
    )
    assert mapping.backend_type != "gpu_cuda"


def test_mpi_capable_above_runtime_threshold_routes_to_cluster():
    mapping = route(
        make_estimate(memory_gb=0.1, runtime_s=MPI_RUNTIME_THRESHOLD_S),
        make_decision("classical", "Genetic Algorithm"),  # mpi_capable
    )
    assert mapping.backend_type == "mpi_cluster"


def test_mpi_capable_below_runtime_threshold_stays_on_cpu():
    mapping = route(
        make_estimate(memory_gb=0.1, runtime_s=MPI_RUNTIME_THRESHOLD_S - 0.01),
        make_decision("classical", "Genetic Algorithm"),
    )
    assert mapping.backend_type == "cpu"


def test_small_serial_job_stays_on_cpu():
    mapping = route(
        make_estimate(memory_gb=0.05, runtime_s=0.5),
        make_decision("classical", "Logistic Regression"),
    )
    assert mapping.backend_type == "cpu"
    assert mapping.fallback_backend is None


def test_non_gpu_algorithm_never_routes_to_cuda():
    """Random Forest is mpi_capable but not gpu_capable."""
    mapping = route(
        make_estimate(memory_gb=100.0, runtime_s=1.0),
        make_decision("classical", "Random Forest"),
    )
    assert mapping.backend_type != "gpu_cuda"


def test_gpu_takes_precedence_over_mpi_when_both_apply():
    """Monte Carlo Simulation is both gpu_capable and mpi_capable."""
    mapping = route(
        make_estimate(memory_gb=GPU_MEMORY_THRESHOLD_GB + 1,
                      runtime_s=MPI_RUNTIME_THRESHOLD_S + 1),
        make_decision("classical", "Monte Carlo Simulation"),
    )
    assert mapping.backend_type == "gpu_cuda"


def test_dual_family_is_routed_as_classical():
    """`dual` and `borderline` are not quantum, so they take the classical path."""
    for family in ("classical", "dual", "borderline"):
        mapping = route(
            make_estimate(memory_gb=0.1, runtime_s=0.1),
            make_decision(family, "XGBoost"),
        )
        assert mapping.backend_type in ("cpu", "gpu_cuda", "mpi_cluster"), family


def test_unknown_algorithm_falls_back_to_default_capabilities():
    mapping = route(
        make_estimate(memory_gb=1000.0, runtime_s=1000.0),
        make_decision("classical", "Nonexistent Algorithm"),
    )
    # Default profile is neither gpu_capable nor mpi_capable.
    assert mapping.backend_type == "cpu"


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_mapping_is_always_a_valid_schema_instance():
    cases = [
        (make_estimate(qubit_count=10, simulation_strategy="statevector"),
         make_decision("quantum", "Grover Search")),
        (make_estimate(memory_gb=10.0), make_decision("classical", "XGBoost")),
        (make_estimate(runtime_s=100.0),
         make_decision("classical", "Simulated Annealing")),
        (make_estimate(), make_decision("classical", "Linear Regression")),
    ]
    for estimate, decision in cases:
        mapping = route(estimate, decision)
        assert isinstance(mapping, HardwareMapping)
        assert mapping.justification.strip()
        assert mapping.device_spec.strip()


def test_trace_propagates_the_calibration_flag():
    """A downstream report must be able to see the estimate was not measured."""
    mapping = route(
        make_estimate(calibrated=False), make_decision("classical", "XGBoost"))
    assert mapping.reasoning_trace["estimate_calibrated"] is False


def test_classical_trace_records_thresholds_used():
    mapping = route(
        make_estimate(memory_gb=1.0, runtime_s=1.0),
        make_decision("classical", "XGBoost"),
    )
    for key in ("gpu_capable", "mpi_capable", "gpu_memory_threshold_gb",
                "mpi_runtime_threshold_s", "estimated_memory_gb",
                "estimated_runtime_seconds"):
        assert key in mapping.reasoning_trace, f"trace missing {key}"


def test_backend_type_is_always_in_the_allowed_set():
    allowed = {"cpu", "gpu_cuda", "mpi_cluster", "qpu_simulator"}
    mapping = route(
        make_estimate(qubit_count=5, simulation_strategy="mps",
                      trace={"bond_dimension": 32}),
        make_decision("quantum", "Quantum Kernel Classifier"),
    )
    assert mapping.backend_type in allowed


# ---------------------------------------------------------------------------
# Hardware availability gating
# ---------------------------------------------------------------------------

def test_never_assigns_cuda_when_no_gpu_is_present():
    """
    The core fix: a GPU-capable algorithm with a GPU-sized footprint must
    NOT be mapped to CUDA on a machine that has no CUDA device.
    """
    estimate = make_estimate(memory_gb=GPU_MEMORY_THRESHOLD_GB + 10, runtime_s=1.0)
    decision = make_decision("classical", "XGBoost")

    assert route(estimate, decision, FULL_HW).backend_type == "gpu_cuda"
    assert route(estimate, decision, CPU_ONLY_HW).backend_type == "cpu"


def test_never_assigns_mpi_when_no_mpi_runtime_is_present():
    estimate = make_estimate(memory_gb=0.1, runtime_s=MPI_RUNTIME_THRESHOLD_S + 10)
    decision = make_decision("classical", "Genetic Algorithm")

    assert route(estimate, decision, FULL_HW).backend_type == "mpi_cluster"
    assert route(estimate, decision, CPU_ONLY_HW).backend_type == "cpu"


def test_cpu_justification_names_the_missing_hardware():
    """
    "no GPU on this host" and "too small to need one" are different
    conclusions and the justification must distinguish them.
    """
    big = route(
        make_estimate(memory_gb=GPU_MEMORY_THRESHOLD_GB + 10),
        make_decision("classical", "XGBoost"), CPU_ONLY_HW)
    assert "not available on this deployment" in big.justification
    assert "CUDA" in big.justification

    small = route(
        make_estimate(memory_gb=0.01, runtime_s=0.01),
        make_decision("classical", "XGBoost"), FULL_HW)
    assert "not available on this deployment" not in small.justification


def test_unavailable_backends_are_recorded_in_the_trace():
    mapping = route(
        make_estimate(memory_gb=100.0, runtime_s=100.0),
        make_decision("classical", "Monte Carlo Simulation"),  # gpu + mpi capable
        CPU_ONLY_HW)
    unavailable = mapping.reasoning_trace["unavailable_backends"]
    assert any("CUDA" in u for u in unavailable)
    assert any("MPI" in u for u in unavailable)


def test_device_spec_reports_the_real_device():
    mapping = route(
        make_estimate(memory_gb=GPU_MEMORY_THRESHOLD_GB + 1),
        make_decision("classical", "XGBoost"), FULL_HW)
    assert "RTX 4090" in mapping.device_spec


def test_mpi_device_spec_reports_actual_slot_count():
    mapping = route(
        make_estimate(memory_gb=0.1, runtime_s=MPI_RUNTIME_THRESHOLD_S + 1),
        make_decision("classical", "Simulated Annealing"), MPI_ONLY_HW)
    assert mapping.backend_type == "mpi_cluster"
    assert "16 slots" in mapping.device_spec


def test_cpu_device_spec_reports_real_cores_and_ram():
    mapping = route(
        make_estimate(memory_gb=0.01, runtime_s=0.01),
        make_decision("classical", "Linear Regression"), CPU_ONLY_HW)
    assert "16 logical cores" in mapping.device_spec


def test_target_mode_marks_the_mapping_hypothetical():
    estimate = make_estimate(memory_gb=GPU_MEMORY_THRESHOLD_GB + 1)
    decision = make_decision("classical", "XGBoost")

    assert route(estimate, decision, FULL_HW).is_hypothetical is False

    hypothetical = route(estimate, decision, TARGET_HW)
    assert hypothetical.is_hypothetical is True
    assert hypothetical.backend_type == "gpu_cuda"
    assert "declared target" in hypothetical.device_spec or \
        hypothetical.reasoning_trace["planning_mode"] == "target"


def test_quantum_simulator_assignment_is_hardware_independent():
    """A simulator runs on whatever CPU exists; no GPU is required."""
    estimate = make_estimate(qubit_count=10, simulation_strategy="statevector")
    decision = make_decision("quantum", "Grover Search")
    for inventory in (FULL_HW, CPU_ONLY_HW, TARGET_HW):
        assert route(estimate, decision, inventory).backend_type == "qpu_simulator"


def test_trace_carries_detection_notes_for_audit():
    mapping = route(
        make_estimate(), make_decision("classical", "XGBoost"), CPU_ONLY_HW)
    assert "detection_notes" in mapping.reasoning_trace
    assert "planning_mode" in mapping.reasoning_trace
