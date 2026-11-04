"""
Step 10 — Report Generator tests.

Run:  pytest backend/tests/test_report_generator.py -v

The report is the document a reader cites from, so most of these tests are
about what it is allowed to claim. A report that presented a prediction as
a measurement, or a wrong answer as a performance result, would be worse
than no report at all.
"""

import json

import pytest

from backend.engine.report_generator import (
    generate_report,
    render_json,
    render_markdown,
)
from backend.models.critic_schema import CriticFinding, CriticReport
from backend.models.decision_schema import ExecutionDecision
from backend.models.hardware_schema import HardwareMapping
from backend.models.problem_schema import ProblemSpecification
from backend.models.report_schema import ExecutionReport
from backend.models.resource_schema import ResourceEstimate
from backend.models.telemetry_schema import ExecutionTelemetry

QUERY = "Optimize delivery routes for 500 cities under travel constraints."


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_spec():
    return ProblemSpecification(
        user_query=QUERY, problem_class="optimization", confidence=0.80,
        dimensionality=500, dataset_size=500,
        constraint_structure="constrained", quantum_candidate=True,
        reasoning="Constrained combinatorial optimization.")


def make_decision(algorithm="Genetic Algorithm", family="classical"):
    return ExecutionDecision(
        selected_family=family, selected_algorithm=algorithm,
        routing_mode="borderline_review", decision_confidence=0.70,
        borderline_score=0.90, explanation="test")


def make_estimate(calibrated=False, hypothetical=False):
    return ResourceEstimate(
        estimated_memory_gb=0.2, estimated_runtime_seconds=5.0,
        resource_tier="light", calibrated=calibrated,
        hardware_is_hypothetical=hypothetical, reasoning_trace={})


def make_mapping(backend="cpu", hypothetical=False):
    return HardwareMapping(
        backend_type=backend, device_spec="test device",
        justification="test", is_hypothetical=hypothetical,
        reasoning_trace={})


def make_telemetry(success=True, actual=4.5, metric_value=0.001, error=None):
    return ExecutionTelemetry(
        algorithm="Genetic Algorithm", backend_type="cpu",
        device_detail="numpy", telemetry_host="test-host",
        success=success, error_message=error,
        actual_runtime_seconds=actual, actual_peak_memory_gb=0.18,
        predicted_runtime_seconds=5.0, predicted_memory_gb=0.2,
        runtime_error_ratio=round(actual / 5.0, 4),
        memory_error_ratio=0.9,
        result_metric_name="best_fitness" if success else None,
        result_metric_value=metric_value if success else None,
        raw_metrics={})


def make_critic(with_critical=False):
    findings = []
    if with_critical:
        findings.append(CriticFinding(
            category="correctness_failure", severity="critical",
            algorithm="Grover Search", backend_type="qpu_simulator",
            summary="Produced an incorrect result.",
            evidence="P(marked) = 0.004, expected 1.0 +/- 0.05",
            recommendation="Fix the workload before using the measurement."))
    return CriticReport(
        runs_analysed=1, runs_succeeded=1, findings=findings,
        verdict="test verdict")


def full_report(**overrides):
    kwargs = dict(
        user_query=QUERY, spec=make_spec(), decision=make_decision(),
        estimate=make_estimate(), mapping=make_mapping(),
        telemetry=make_telemetry(), critic=make_critic(),
    )
    kwargs.update(overrides)
    return generate_report(**kwargs)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

def test_successful_run_is_measured():
    report = full_report()
    assert report.outcome == "measured"
    assert report.is_reportable_as_measurement


def test_plan_without_execution_is_planned_only():
    report = full_report(telemetry=None, critic=None)
    assert report.outcome == "planned_only"
    assert not report.is_reportable_as_measurement


def test_failed_execution_is_failed():
    report = full_report(
        telemetry=make_telemetry(success=False,
                                 error="WorkloadUnavailable: boom"))
    assert report.outcome == "failed"
    assert not report.is_reportable_as_measurement


def test_no_decision_is_incomplete():
    report = generate_report(user_query=QUERY, spec=make_spec())
    assert report.outcome == "incomplete"


# ---------------------------------------------------------------------------
# What may be claimed — the central question
# ---------------------------------------------------------------------------

def test_incorrect_result_makes_a_report_uncitable():
    """
    The single most important rule here. A run that completed quickly while
    returning the wrong answer must not appear as a performance result, or
    the Grover case becomes publishable by accident.
    """
    report = full_report(critic=make_critic(with_critical=True))

    assert report.outcome == "failed"
    assert not report.is_reportable_as_measurement
    assert any(c.kind == "incorrect_result" for c in report.caveats)
    assert "INCORRECT" in report.headline


def test_hypothetical_mapping_makes_a_report_uncitable():
    report = full_report(mapping=make_mapping(hypothetical=True))
    assert not report.is_reportable_as_measurement
    caveat = next(c for c in report.caveats if c.kind == "hypothetical")
    assert caveat.severity == "blocking"
    assert "DECLARED, not" in caveat.message


def test_unexecuted_plan_states_its_numbers_are_predictions():
    report = full_report(telemetry=None, critic=None)
    caveat = next(c for c in report.caveats if c.kind == "not_executed")
    assert caveat.severity == "blocking"
    assert "PREDICTION" in caveat.message
    assert "not a measurement" in caveat.message


def test_uncalibrated_estimate_is_important_but_not_blocking():
    """
    An uncalibrated model still produces a real measurement; it only limits
    what the error ratio means. Treating it as blocking would make every
    current report uncitable.
    """
    report = full_report(estimate=make_estimate(calibrated=False))
    caveat = next(c for c in report.caveats if c.kind == "uncalibrated")
    assert caveat.severity == "important"
    assert report.is_reportable_as_measurement


def test_calibrated_estimate_produces_no_calibration_caveat():
    report = full_report(estimate=make_estimate(calibrated=True))
    assert not any(c.kind == "uncalibrated" for c in report.caveats)


def test_measured_runtime_is_marked_host_specific():
    report = full_report()
    caveat = next(c for c in report.caveats if c.kind == "host_specific")
    assert "test-host" in caveat.message
    assert "do not transfer" in caveat.message


# ---------------------------------------------------------------------------
# Partial pipelines
# ---------------------------------------------------------------------------

def test_missing_stages_are_recorded_not_reconstructed():
    """
    A reconstructed decision in a results document is indistinguishable
    from a real one, so absent stages are named rather than inferred.
    """
    report = generate_report(
        user_query=QUERY, decision=make_decision(),
        estimate=make_estimate(), mapping=make_mapping(),
        telemetry=make_telemetry())

    assert report.problem_summary is None
    assert report.candidates_summary is None
    assert "Step 1-2: Problem Analysis" in report.stages_missing
    assert "Step 8: Execution" in report.stages_completed


def test_partial_pipeline_is_noted_as_a_caveat():
    report = generate_report(user_query=QUERY, decision=make_decision(),
                             estimate=make_estimate())
    caveat = next(c for c in report.caveats if c.kind == "partial_pipeline")
    assert "not reconstructed" in caveat.message


def test_telemetry_only_report_is_valid():
    """Archived telemetry without upstream context must still render."""
    report = generate_report(user_query=QUERY, telemetry=make_telemetry())
    assert isinstance(report, ExecutionReport)
    assert report.execution_summary is not None
    assert render_markdown(report)


def test_every_stage_is_accounted_for():
    report = full_report()
    total = len(report.stages_completed) + len(report.stages_missing)
    assert total == 9


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_markdown_contains_the_essentials():
    text = render_markdown(full_report())
    for expected in ("Q-HPC Execution Report", QUERY, "Resource Estimate",
                     "Hardware Mapping", "Execution & Measurement",
                     "Critique", "Pipeline Coverage"):
        assert expected in text


def test_markdown_shows_predicted_beside_measured():
    """The comparison is the point; both must appear together."""
    text = render_markdown(full_report())
    assert "| Quantity | Predicted | Measured | Ratio |" in text
    assert "runtime (s)" in text


def test_markdown_states_citability_explicitly():
    good = render_markdown(full_report())
    bad = render_markdown(full_report(critic=make_critic(with_critical=True)))
    assert "**Citable as a measured result:** yes" in good
    assert "**Citable as a measured result:** NO" in bad


def test_blocking_caveats_are_marked_in_the_markdown():
    text = render_markdown(full_report(mapping=make_mapping(hypothetical=True)))
    assert "[BLOCKING]" in text


def test_failed_run_markdown_shows_the_error():
    text = render_markdown(full_report(
        telemetry=make_telemetry(success=False,
                                 error="WorkloadUnavailable: xgboost missing")))
    assert "Execution failed" in text
    assert "xgboost missing" in text


def test_json_is_valid_and_round_trips():
    report = full_report()
    data = json.loads(render_json(report))
    assert data["user_query"] == QUERY
    assert data["outcome"] == "measured"
    assert len(data["caveats"]) == len(report.caveats)


def test_report_without_caveats_still_renders():
    report = generate_report(user_query=QUERY, decision=make_decision(),
                             estimate=make_estimate(calibrated=True),
                             mapping=make_mapping(),
                             telemetry=make_telemetry())
    assert render_markdown(report)


# ---------------------------------------------------------------------------
# Headline
# ---------------------------------------------------------------------------

def test_headline_reports_the_measurement():
    headline = full_report().headline
    assert "Genetic Algorithm" in headline
    assert "cpu" in headline
    assert "4.500s" in headline


def test_headline_for_an_unexecuted_plan_says_so():
    headline = full_report(telemetry=None, critic=None).headline
    assert "Planned" in headline
    assert "no cost was measured" in headline


def test_headline_never_claims_a_measurement_for_a_wrong_answer():
    headline = full_report(critic=make_critic(with_critical=True)).headline
    assert "INCORRECT" in headline
    assert "not a performance measurement" in headline
