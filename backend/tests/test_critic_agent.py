"""
Step 9 — Critic Agent tests.

Run:  pytest backend/tests/test_critic_agent.py -v

The critic judges the framework's own claims, so these tests are mostly
about judgement quality: does it flag what matters, stay quiet about what
does not, and never overstate what the evidence supports.
"""

import pytest

from backend.agents.critic_agent import (
    MIN_MEANINGFUL_RUNTIME_S,
    MIN_RUNS_FOR_CALIBRATION,
    critique,
    format_report,
)
from backend.models.critic_schema import CriticReport
from backend.models.hardware_schema import HardwareMapping
from backend.models.telemetry_schema import ExecutionTelemetry


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_telemetry(
    algorithm="Monte Carlo Simulation", backend="cpu", success=True,
    predicted=1.0, actual=1.0, memory_predicted=0.1, memory_actual=0.1,
    metric_name="pi_absolute_error", metric_value=0.001,
    metric_target=None, metric_tolerance=None,
    metric_behaviour="invariant", uncalibrated=True,
    error=None, host="test-host", extra=None,
):
    raw = {"metric_rank_behaviour": metric_behaviour}
    if metric_target is not None:
        raw["metric_target"] = metric_target
    if metric_tolerance is not None:
        raw["metric_tolerance"] = metric_tolerance
    raw.update(extra or {})

    return ExecutionTelemetry(
        algorithm=algorithm, backend_type=backend,
        device_detail="test device", telemetry_host=host,
        success=success, error_message=error,
        actual_runtime_seconds=actual, actual_peak_memory_gb=memory_actual,
        predicted_runtime_seconds=predicted, predicted_memory_gb=memory_predicted,
        runtime_error_ratio=(round(actual / predicted, 4) if predicted > 0 else None),
        memory_error_ratio=(round(memory_actual / memory_predicted, 4)
                            if memory_predicted > 0 else None),
        result_metric_name=metric_name if success else None,
        result_metric_value=metric_value if success else None,
        compared_against_uncalibrated_estimate=uncalibrated,
        raw_metrics=raw,
    )


def make_mapping(backend="cpu", trace=None):
    return HardwareMapping(
        backend_type=backend, device_spec="test", justification="test",
        reasoning_trace=trace or {},
    )


def categories(report):
    return {f.category for f in report.findings}


# ---------------------------------------------------------------------------
# Correctness — the highest-severity check
# ---------------------------------------------------------------------------

def test_incorrect_result_is_critical():
    """
    The Grover case: the run completed, timed well, and returned the wrong
    answer. That must outrank every performance consideration.
    """
    report = critique([make_telemetry(
        algorithm="Grover Search", backend="qpu_simulator",
        metric_name="marked_state_probability", metric_value=0.0040,
        metric_target=1.0, metric_tolerance=0.05,
    )])

    failures = report.findings_by_severity("critical")
    assert len(failures) == 1
    assert failures[0].category == "correctness_failure"
    assert "not be reported as a result" in failures[0].recommendation
    assert report.has_blocking_findings


def test_correct_result_within_tolerance_is_not_flagged():
    report = critique([make_telemetry(
        metric_value=0.0004, metric_target=0.0, metric_tolerance=0.01)])
    assert "correctness_failure" not in categories(report)


def test_rank_dependent_metric_is_not_judged_against_a_target():
    """
    The GA's fitness worsens with rank count by design. Flagging that as a
    correctness failure would make the check useless through false alarms.
    """
    report = critique([make_telemetry(
        algorithm="Genetic Algorithm", metric_name="best_fitness",
        metric_value=2441.0, metric_target=0.0, metric_tolerance=1.0,
        metric_behaviour="rank_dependent")])
    assert "correctness_failure" not in categories(report)


def test_correctness_failure_appears_in_the_verdict():
    report = critique([make_telemetry(
        algorithm="Grover Search", metric_value=0.004,
        metric_target=1.0, metric_tolerance=0.05)])
    assert "CORRECTNESS FAILURE" in report.verdict
    assert "Grover Search" in report.verdict


# ---------------------------------------------------------------------------
# Prediction accuracy
# ---------------------------------------------------------------------------

def test_accurate_prediction_produces_no_error_finding():
    report = critique([make_telemetry(predicted=7.063, actual=6.953)])
    assert "prediction_error" not in categories(report)


@pytest.mark.parametrize("predicted,actual,expected_severity", [
    (1.0, 3.0, "low"),       # 3x
    (1.0, 7.0, "medium"),    # 7x
    (1.0, 20.0, "high"),     # 20x
    (20.0, 1.0, "high"),     # 20x the other way
])
def test_prediction_error_severity_scales_with_deviation(
        predicted, actual, expected_severity):
    report = critique([make_telemetry(predicted=predicted, actual=actual)])
    errors = [f for f in report.findings if f.category == "prediction_error"]
    assert len(errors) == 1
    assert errors[0].severity == expected_severity


def test_prediction_error_records_the_numbers():
    report = critique([make_telemetry(predicted=10.0, actual=1.0)])
    finding = next(f for f in report.findings if f.category == "prediction_error")
    assert finding.predicted == 10.0
    assert finding.actual == 1.0
    assert finding.ratio == pytest.approx(0.1)


def test_very_short_runs_are_excluded_from_judgement():
    """
    A sub-50ms run is dominated by interpreter startup. Its ratio describes
    overhead, not the cost model, and judging it would produce noise.
    """
    report = critique([make_telemetry(
        predicted=10.0, actual=MIN_MEANINGFUL_RUNTIME_S / 2)])
    assert "prediction_error" not in categories(report)
    assert report.runs_with_usable_ratio == 0
    assert report.median_runtime_ratio is None


def test_direction_of_error_is_stated():
    optimistic = critique([make_telemetry(predicted=1.0, actual=10.0)])
    pessimistic = critique([make_telemetry(predicted=10.0, actual=1.0)])
    assert "optimistic" in optimistic.findings[0].summary
    assert "pessimistic" in pessimistic.findings[0].summary


def test_calibrated_estimate_error_is_described_as_a_regression():
    """
    A wrong prediction from a MEASURED model is a different problem from a
    wrong prediction from a reasoned one, and the recommendation says so.
    """
    uncal = critique([make_telemetry(predicted=1.0, actual=20.0,
                                     uncalibrated=True)])
    cal = critique([make_telemetry(predicted=1.0, actual=20.0,
                                   uncalibrated=False)])

    assert "unmeasured model" in uncal.findings[0].recommendation
    assert "regression" in cal.findings[0].recommendation


# ---------------------------------------------------------------------------
# Failures, distinguished by cause
# ---------------------------------------------------------------------------

def test_missing_implementation_is_distinct_from_execution_failure():
    missing = critique([make_telemetry(
        algorithm="SVM", success=False,
        error="WorkloadUnavailable: No executable implementation for 'SVM'.")])
    broken = critique([make_telemetry(
        algorithm="XGBoost", success=False,
        error="WorkloadUnavailable: xgboost unusable: ModuleNotFoundError")])

    assert "missing_implementation" in categories(missing)
    assert "execution_failure" in categories(broken)
    # The two call for different fixes, so they must not collapse together.
    assert categories(missing) != categories(broken)


def test_hypothetical_refusal_is_informational_not_a_defect():
    """
    Refusing to execute against unverified hardware is the engine working
    correctly. Reporting it as a failure would train the reader to ignore
    failures.
    """
    report = critique([make_telemetry(
        backend="gpu_cuda", success=False,
        error=("Refusing to execute against a hypothetical mapping. Step 7 "
               "ran in planning mode 'target'."))])

    finding = next(f for f in report.findings
                   if f.category == "unverified_hardware")
    assert finding.severity == "info"
    assert not report.has_blocking_findings


def test_failed_runs_are_counted_separately():
    report = critique([
        make_telemetry(success=True),
        make_telemetry(success=False, error="WorkloadUnavailable: boom"),
        make_telemetry(success=False, error="WorkloadUnavailable: boom"),
    ])
    assert report.runs_analysed == 3
    assert report.runs_succeeded == 1


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_cpu_routing_is_questioned_when_mpi_was_available_and_needed():
    report = critique(
        [make_telemetry(algorithm="Genetic Algorithm", backend="cpu",
                        predicted=5.0, actual=45.0)],
        [make_mapping("cpu", trace={"mpi_available": True})],
    )
    assert "suboptimal_routing" in categories(report)


def test_cpu_routing_is_not_questioned_when_mpi_was_unavailable():
    """Choosing CPU on a machine with no MPI is correct, not suboptimal."""
    report = critique(
        [make_telemetry(algorithm="Genetic Algorithm", backend="cpu",
                        predicted=5.0, actual=45.0)],
        [make_mapping("cpu", trace={"mpi_available": False})],
    )
    assert "suboptimal_routing" not in categories(report)


def test_routing_checks_are_skipped_without_mappings():
    """Telemetry alone must still produce the other findings."""
    report = critique([make_telemetry(predicted=1.0, actual=20.0)])
    assert "prediction_error" in categories(report)
    assert "suboptimal_routing" not in categories(report)


def test_short_companion_lists_do_not_raise():
    """Archived telemetry may arrive without its mappings."""
    report = critique(
        [make_telemetry(), make_telemetry(), make_telemetry()],
        [make_mapping()],
    )
    assert report.runs_analysed == 3


# ---------------------------------------------------------------------------
# Calibration feedback — the loop back to Step 6
# ---------------------------------------------------------------------------

def test_calibration_suggestion_computes_the_correction():
    """
    The suggested value multiplied into base_runtime_s should make the
    median prediction exact.
    """
    runs = [make_telemetry(algorithm="Monte Carlo Simulation",
                           predicted=10.0, actual=2.0) for _ in range(3)]
    report = critique(runs)

    suggestion = next(s for s in report.calibration_suggestions
                      if s.algorithm == "Monte Carlo Simulation")
    assert suggestion.median_ratio == pytest.approx(0.2)
    assert suggestion.suggested_base_runtime_s == pytest.approx(
        suggestion.current_base_runtime_s * 0.2, rel=1e-3)


def test_single_run_suggestion_is_low_confidence():
    report = critique([make_telemetry(predicted=10.0, actual=1.0)])
    suggestion = report.calibration_suggestions[0]
    assert suggestion.confidence == "low"
    assert "provisional" in suggestion.note


def test_consistent_repeated_runs_give_high_confidence():
    runs = [make_telemetry(predicted=10.0, actual=2.0)
            for _ in range(MIN_RUNS_FOR_CALIBRATION)]
    suggestion = critique(runs).calibration_suggestions[0]
    assert suggestion.confidence == "high"


def test_scattered_ratios_reduce_confidence():
    runs = [make_telemetry(predicted=10.0, actual=a) for a in (1.0, 5.0, 9.0)]
    suggestion = critique(runs).calibration_suggestions[0]
    assert suggestion.confidence == "moderate"


def test_mixed_hosts_make_a_suggestion_unsafe():
    """
    base_runtime_s does not transfer between machines, so a suggestion
    averaging two hosts would be meaningless. It must say so.
    """
    runs = [
        make_telemetry(predicted=10.0, actual=2.0, host="laptop"),
        make_telemetry(predicted=10.0, actual=5.0, host="colab"),
        make_telemetry(predicted=10.0, actual=3.0, host="laptop"),
    ]
    suggestion = critique(runs).calibration_suggestions[0]
    assert "MIXED HOSTS" in suggestion.note
    assert "unsafe" in suggestion.note


def test_mixed_hosts_are_flagged_in_the_verdict():
    runs = [make_telemetry(host="laptop"), make_telemetry(host="colab")]
    assert "hosts" in critique(runs).verdict


# ---------------------------------------------------------------------------
# Aggregates and reporting
# ---------------------------------------------------------------------------

def test_median_ratio_is_none_when_nothing_is_measurable():
    """
    Reporting 0.0 would read as "perfectly wrong" rather than "not
    measured" — the same fabrication the telemetry schema avoids.
    """
    report = critique([make_telemetry(success=False, error="WorkloadUnavailable: x")])
    assert report.median_runtime_ratio is None
    assert report.runs_with_usable_ratio == 0


def test_within_2x_counts_only_usable_runs():
    report = critique([
        make_telemetry(predicted=1.0, actual=1.5),    # within
        make_telemetry(predicted=1.0, actual=20.0),   # outside
        make_telemetry(predicted=1.0, actual=0.9),    # within
    ])
    assert report.runs_with_usable_ratio == 3
    assert report.runs_within_2x == 2


def test_findings_are_ordered_by_severity():
    report = critique([
        make_telemetry(algorithm="A", predicted=1.0, actual=3.0),   # low
        make_telemetry(algorithm="B", success=False,
                       error="WorkloadUnavailable: boom"),           # high
        make_telemetry(algorithm="C", metric_value=0.004,
                       metric_target=1.0, metric_tolerance=0.05),    # critical
    ])
    severities = [f.severity for f in report.findings]
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    assert severities == sorted(severities, key=lambda s: order[s])
    assert severities[0] == "critical"


def test_empty_input_does_not_crash():
    report = critique([])
    assert report.runs_analysed == 0
    assert report.findings == []
    assert "No executions" in report.verdict


def test_report_is_a_valid_schema_instance():
    report = critique([make_telemetry()])
    assert isinstance(report, CriticReport)
    assert report.verdict.strip()


def test_format_report_is_readable_without_an_llm():
    """
    The deterministic report must stand alone — Ollama is frequently
    unavailable and nothing here may depend on it.
    """
    text = format_report(critique([make_telemetry(predicted=1.0, actual=20.0)]))
    assert "CRITIC AGENT REPORT" in text
    assert "FINDINGS" in text
    assert "No LLM narrative" in text


def test_uncalibrated_runs_are_called_out_in_the_verdict():
    report = critique([make_telemetry(uncalibrated=True)])
    assert report.all_estimates_uncalibrated
    assert "uncalibrated" in report.verdict


def test_every_finding_carries_an_actionable_recommendation():
    report = critique([
        make_telemetry(predicted=1.0, actual=20.0),
        make_telemetry(algorithm="SVM", success=False,
                       error="WorkloadUnavailable: No executable implementation"),
        make_telemetry(algorithm="Grover Search", metric_value=0.004,
                       metric_target=1.0, metric_tolerance=0.05),
    ])
    for finding in report.findings:
        assert finding.summary.strip()
        assert finding.evidence.strip()
        assert finding.recommendation.strip()


def test_zero_ratio_does_not_crash():
    """
    A ratio of exactly 0.0 is reachable, because it is rounded to four
    decimals: a 0.06s run against a 1e6s prediction rounds to zero. Step 6's
    uncalibrated constants with a steep exponent do predict runtimes that
    large, so `1.0 / ratio` would crash on real data.
    """
    telemetry = make_telemetry(predicted=1e6, actual=0.06)
    assert telemetry.runtime_error_ratio == 0.0

    report = critique([telemetry])          # must not raise
    errors = [f for f in report.findings if f.category == "prediction_error"]
    assert len(errors) == 1
    assert errors[0].severity == "high"
    assert "pessimistic" in errors[0].summary
    assert "more than" in errors[0].summary


def test_extremely_pessimistic_prediction_is_high_severity():
    report = critique([make_telemetry(predicted=1e9, actual=0.1)])
    assert report.findings[0].severity == "high"
