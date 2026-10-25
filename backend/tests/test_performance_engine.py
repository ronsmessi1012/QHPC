"""
Step 8 — Performance Engine tests.

Run:  pytest backend/tests/test_performance_engine.py -v

Several tests exercise real computation rather than mocks, because the
point of Step 8 is that it actually runs things. Those are marked in their
docstrings and kept small.

Tests for optional-dependency backends (xgboost/CUDA, qiskit/simulator)
assert on BEHAVIOUR THAT HOLDS EITHER WAY: when the library is present the
run must succeed with a sane correctness metric, and when it is absent the
telemetry must report the real reason with success=False. That way the
suite is meaningful on a laptop without xgboost and on Colab with a GPU,
without being skipped into vacuousness on either.
"""

import pytest

from backend.engine.performance_engine import execute, reportable_memory_gb
from backend.engine.workloads import (
    WORKLOAD_REGISTRY,
    WorkloadUnavailable,
    get_workload,
)
from backend.models.decision_schema import ExecutionDecision
from backend.models.hardware_schema import HardwareMapping
from backend.models.problem_schema import ProblemSpecification
from backend.models.resource_schema import ResourceEstimate
from backend.models.telemetry_schema import ExecutionTelemetry


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_spec(dimensionality=10, dataset_size=200):
    return ProblemSpecification(
        user_query="test", problem_class="simulation", confidence=0.8,
        dimensionality=dimensionality, dataset_size=dataset_size,
        constraint_structure="constrained",
    )


def make_decision(family, algorithm):
    return ExecutionDecision(
        selected_family=family, selected_algorithm=algorithm,
        routing_mode="borderline_review", decision_confidence=0.70,
        borderline_score=0.90, explanation="test",
    )


def make_estimate(memory_gb=0.05, runtime_s=1.0, calibrated=False, trace=None):
    return ResourceEstimate(
        estimated_memory_gb=memory_gb, estimated_runtime_seconds=runtime_s,
        resource_tier="light", calibrated=calibrated,
        reasoning_trace=trace or {},
    )


def make_mapping(backend, hypothetical=False, trace=None):
    return HardwareMapping(
        backend_type=backend, device_spec=f"test {backend}",
        justification="test", is_hypothetical=hypothetical,
        reasoning_trace=trace or {},
    )


def run(backend, algorithm, family="classical", *, estimate=None,
        mapping=None, spec=None, **kwargs):
    return execute(
        mapping or make_mapping(backend),
        estimate or make_estimate(),
        make_decision(family, algorithm),
        spec or make_spec(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Workload registry integrity
# ---------------------------------------------------------------------------

def test_workload_names_match_the_algorithm_registry():
    """
    A key typo here would report an implemented algorithm as unimplemented.
    Same silent-fallback class of bug as the Step 6 profile keys.
    """
    from backend.utils.constants import CLASSICAL_ALGORITHMS, QUANTUM_ALGORITHMS

    known = {e["name"] for a in CLASSICAL_ALGORITHMS.values() for e in a}
    known |= {e["name"] for a in QUANTUM_ALGORITHMS.values() for e in a}

    unknown = set(WORKLOAD_REGISTRY) - known
    assert not unknown, f"workloads naming algorithms not in the registry: {unknown}"


@pytest.mark.parametrize("name,workload", WORKLOAD_REGISTRY.items())
def test_every_workload_declares_valid_backends(name, workload):
    allowed = {"cpu", "gpu_cuda", "mpi_cluster", "qpu_simulator"}
    assert workload.supported_backends, f"{name} declares no backends"
    assert set(workload.supported_backends) <= allowed, name
    assert callable(workload.run)


def test_unimplemented_algorithm_raises_with_a_useful_message():
    with pytest.raises(WorkloadUnavailable) as exc:
        get_workload("SVM")
    assert "SVM" in str(exc.value)
    assert "Implemented" in str(exc.value)


# ---------------------------------------------------------------------------
# Real execution — these actually compute
# ---------------------------------------------------------------------------

def test_monte_carlo_runs_and_estimates_pi():
    """Real run. Validates the reduction: a broken sum gives a wild pi."""
    telemetry = run("cpu", "Monte Carlo Simulation")

    assert telemetry.success, telemetry.error_message
    assert telemetry.result_metric_name == "pi_absolute_error"
    # Monte Carlo with this many samples should land close to pi; a value
    # far off means the sampling or counting is wrong, not just noisy.
    assert telemetry.result_metric_value < 0.05
    assert telemetry.actual_runtime_seconds > 0


def test_genetic_algorithm_runs_and_improves_fitness():
    """Real run. Rastrigin's optimum is 0; a random start scores far higher."""
    telemetry = run("cpu", "Genetic Algorithm", spec=make_spec(dimensionality=5))

    assert telemetry.success, telemetry.error_message
    assert telemetry.result_metric_name == "best_fitness"
    # A 5-dimensional Rastrigin starts around 50-80 for random genomes;
    # anything under 40 shows the search actually descended.
    assert 0.0 <= telemetry.result_metric_value < 40.0


def test_telemetry_is_always_a_valid_schema_instance():
    for backend, algorithm in [
        ("cpu", "Monte Carlo Simulation"),
        ("cpu", "SVM"),                      # unimplemented
        ("gpu_cuda", "XGBoost"),             # may lack xgboost
        ("qpu_simulator", "Grover Search"),  # may lack qiskit
        ("mpi_cluster", "Monte Carlo Simulation"),  # may lack mpiexec
    ]:
        telemetry = run(backend, algorithm)
        assert isinstance(telemetry, ExecutionTelemetry)
        assert telemetry.backend_type == backend
        assert telemetry.telemetry_host
        if not telemetry.success:
            assert telemetry.error_message, f"{backend}/{algorithm} failed silently"


# ---------------------------------------------------------------------------
# Predicted vs actual — the scientific payload
# ---------------------------------------------------------------------------

def test_error_ratio_is_actual_over_predicted():
    telemetry = run("cpu", "Monte Carlo Simulation",
                    estimate=make_estimate(runtime_s=10.0, memory_gb=1.0))
    assert telemetry.success
    expected = telemetry.actual_runtime_seconds / 10.0
    assert telemetry.runtime_error_ratio == pytest.approx(expected, abs=1e-4)


def test_error_ratio_is_none_when_prediction_is_zero():
    """
    A ratio against zero is undefined. Reporting 0.0 or inf there would be a
    fabricated number in a results table.
    """
    telemetry = run("cpu", "Monte Carlo Simulation",
                    estimate=make_estimate(runtime_s=0.0, memory_gb=0.0))
    assert telemetry.runtime_error_ratio is None
    assert telemetry.memory_error_ratio is None


def test_uncalibrated_estimates_are_flagged_in_telemetry():
    """
    A large error ratio against an uncalibrated estimate says the model was
    never measured, not that the framework mis-predicted. Step 9 and the
    report generator both need to tell those apart.
    """
    uncal = run("cpu", "Monte Carlo Simulation",
                estimate=make_estimate(calibrated=False))
    cal = run("cpu", "Monte Carlo Simulation",
              estimate=make_estimate(calibrated=True))

    assert uncal.compared_against_uncalibrated_estimate is True
    assert cal.compared_against_uncalibrated_estimate is False


def test_runtime_within_helper():
    telemetry = run("cpu", "Monte Carlo Simulation",
                    estimate=make_estimate(runtime_s=1.0))
    assert telemetry.success
    # Whatever the ratio is, the helper must agree with the bound.
    ratio = telemetry.runtime_error_ratio
    assert telemetry.runtime_within(1000.0) is (0.001 <= ratio <= 1000.0)


# ---------------------------------------------------------------------------
# Refusals and failure reporting
# ---------------------------------------------------------------------------

def test_hypothetical_mapping_is_refused():
    """
    Step 7 in 'target' mode assigns hardware it never verified. Executing
    against that would produce telemetry indistinguishable from a real
    measurement, so it must be refused rather than run.
    """
    telemetry = run("gpu_cuda", "XGBoost",
                    mapping=make_mapping("gpu_cuda", hypothetical=True))

    assert telemetry.success is False
    assert "hypothetical" in telemetry.error_message.lower()
    assert telemetry.actual_runtime_seconds == 0.0


def test_unsupported_backend_for_an_implemented_algorithm_is_reported():
    """Monte Carlo has no CUDA path; asking for one must say so."""
    telemetry = run("gpu_cuda", "Monte Carlo Simulation")
    assert telemetry.success is False
    assert "does not support backend" in telemetry.error_message


def test_missing_implementation_is_not_reported_as_success():
    telemetry = run("cpu", "Dijkstra")  # in the registry, not implemented
    assert telemetry.success is False
    assert "No executable implementation" in telemetry.error_message
    assert telemetry.result_metric_value is None


def test_failure_carries_the_real_exception_type():
    """
    An earlier bug in the benchmark harness printed "not installed" for every
    failure and hid a Windows policy error for an hour. Failures here must
    name the actual exception.
    """
    telemetry = run("cpu", "SVM")
    assert telemetry.success is False
    assert "WorkloadUnavailable" in telemetry.error_message


# ---------------------------------------------------------------------------
# Optional-dependency backends — assert correct behaviour either way
# ---------------------------------------------------------------------------

def test_xgboost_cuda_path_succeeds_or_explains_itself():
    telemetry = run("gpu_cuda", "XGBoost", spec=make_spec(50, 2000))

    if telemetry.success:
        assert "device=cuda" in telemetry.device_detail
        assert telemetry.result_metric_name == "accuracy"
        # The labels are a learnable linear function, so a fitted model
        # must beat chance by a wide margin.
        assert telemetry.result_metric_value > 0.75
    else:
        assert telemetry.error_message
        assert "xgboost" in telemetry.error_message.lower()


def test_xgboost_cpu_path_succeeds_or_explains_itself():
    telemetry = run("cpu", "XGBoost", spec=make_spec(50, 2000))

    if telemetry.success:
        assert "device=cpu" in telemetry.device_detail
        assert telemetry.result_metric_value > 0.75
    else:
        assert "xgboost" in telemetry.error_message.lower()


def test_grover_succeeds_or_explains_itself():
    telemetry = run("qpu_simulator", "Grover Search", family="quantum",
                    spec=make_spec(dataset_size=64))

    if telemetry.success:
        assert telemetry.result_metric_name == "marked_state_probability"
        # Grover must concentrate amplitude on the marked state. Uniform
        # would be 1/64; a working oracle and diffuser beat that decisively.
        assert telemetry.result_metric_value > 0.5
    else:
        assert "qiskit" in telemetry.error_message.lower()


def test_qaoa_uses_mps_and_succeeds_or_explains_itself():
    telemetry = run(
        "qpu_simulator",
        "Quantum Approximate Optimization Algorithm (QAOA)",
        family="quantum", spec=make_spec(dimensionality=8),
        estimate=make_estimate(trace={"bond_dimension": 32}),
    )

    if telemetry.success:
        assert "matrix_product_state" in telemetry.device_detail
        assert telemetry.result_metric_name == "approximation_ratio"
        assert 0.0 <= telemetry.result_metric_value <= 1.0
    else:
        assert "qiskit" in telemetry.error_message.lower()


def test_mpi_path_succeeds_or_explains_itself():
    """
    Either a genuine parallel run, or an honest failure. What must never
    happen is the third case this originally allowed: a launcher present
    but mpi4py missing, so the worker ran serially and the engine reported
    success under backend_type="mpi_cluster" with comm_size=1. That put a
    serial timing in the telemetry under a distributed backend.
    """
    telemetry = run("mpi_cluster", "Monte Carlo Simulation",
                    mapping=make_mapping("mpi_cluster", trace={"mpi_slots": 2}))

    if telemetry.success:
        assert "mpi4py" in telemetry.device_detail
        assert telemetry.raw_metrics.get("comm_size", 1) > 1, (
            "an mpi_cluster success must have run on more than one rank"
        )
        assert telemetry.result_metric_value < 0.05  # pi error
        # Peak RSS of the launcher excludes child ranks, so memory is not
        # claimed for MPI runs rather than being understated.
        assert telemetry.actual_peak_memory_gb == 0.0
        assert telemetry.memory_error_ratio is None
    else:
        msg = telemetry.error_message.lower()
        assert "mpi" in msg or "mpiexec" in msg


def test_mpi_without_mpi4py_is_not_reported_as_success():
    """
    Regression test for the silent-serial bug. On a host with a launcher
    but no mpi4py (Colab, as it happens), the run must fail with a message
    naming the cause, not succeed with comm_size=1.
    """
    telemetry = run("mpi_cluster", "Monte Carlo Simulation",
                    mapping=make_mapping("mpi_cluster", trace={"mpi_slots": 4}))
    if telemetry.success:
        # Real MPI present: the run must actually be parallel.
        assert telemetry.raw_metrics.get("comm_size", 1) > 1
    else:
        # No real MPI: the error must say why, not merely that something failed.
        msg = telemetry.error_message.lower()
        assert ("no mpi launcher" in msg or "no parallelism" in msg
                or "failed with" in msg), telemetry.error_message


@pytest.mark.parametrize("backend,measured,expected", [
    ("mpi_cluster", 1.25, 0.0),   # launcher RSS excludes the child ranks
    ("mpi_cluster", 0.0, 0.0),
    ("cpu", 1.25, 1.25),
    ("gpu_cuda", 0.5, 0.5),
    ("qpu_simulator", 2.0, 2.0),
])
def test_mpi_memory_is_never_claimed_from_launcher_rss(backend, measured, expected):
    """
    An MPI run measures the RSS of the LAUNCHING process, which excludes
    every rank that did the work. Reporting it would understate the job.

    Tested directly rather than through execute(), because on a host with no
    MPI runtime the run fails before reaching the measurement and an
    end-to-end assertion would pass vacuously.
    """
    assert reportable_memory_gb(backend, measured) == expected


def test_mpi_memory_is_not_falsely_claimed_end_to_end():
    telemetry = run("mpi_cluster", "Monte Carlo Simulation",
                    mapping=make_mapping("mpi_cluster", trace={"mpi_slots": 2}))
    assert telemetry.actual_peak_memory_gb == 0.0
    assert telemetry.memory_error_ratio is None
