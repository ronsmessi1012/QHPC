"""
Step 9 — Critic Agent Feedback Loop

Input:  ExecutionTelemetry (one or many), plus the ResourceEstimate and
        HardwareMapping that produced each run
Output: CriticReport

Steps 6 and 7 make two claims about every run: this will cost X, and this
backend is the right one. Step 8 measures what actually happened. Step 9
checks the claims against the measurement and says, with evidence, where
the framework was wrong.

HYBRID BY DESIGN
----------------
Every judgement here is deterministic: ratios, thresholds, comparisons
against stated tolerances. The LLM is optional and produces narrative only,
over findings that already exist. It cannot add, remove or re-rank them.

Two reasons. First, a critic that invented its own numbers would be exactly
the uncalibrated-confidence problem Step 5.5 was built to prevent. Second,
a judgement about measurement should be reproducible: run the critic twice
on the same telemetry and the findings must be identical.

WHAT COUNTS AS A FINDING
------------------------
A finding is about the FRAMEWORK, never the workload. A slow algorithm is
not a finding; a slow algorithm the framework predicted would be fast is.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.models.critic_schema import (
    CalibrationSuggestion,
    CriticFinding,
    CriticReport,
)
from backend.models.hardware_schema import HardwareMapping
from backend.models.resource_schema import ResourceEstimate
from backend.models.telemetry_schema import ExecutionTelemetry
from backend.utils.resource_constants import (
    CLASSICAL_RESOURCE_PROFILES,
    GPU_MEMORY_THRESHOLD_GB,
    MPI_RUNTIME_THRESHOLD_S,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
# A cost model is useful for routing if it gets the ORDER OF MAGNITUDE
# right; routing decisions turn on thresholds like "is this over 4 GB",
# not on exact seconds. These bands reflect that: within 2x is good enough
# to route correctly, beyond 10x is wrong enough to route incorrectly.
RATIO_GOOD = 2.0        # within 2x either way
RATIO_POOR = 5.0        # beyond this the model is misleading
RATIO_SEVERE = 10.0     # beyond this it is actively wrong

# Below this, timing is dominated by process startup and interpreter
# overhead, so a large ratio says nothing about the model.
MIN_MEANINGFUL_RUNTIME_S = 0.05

# A calibration suggestion from a single run is an anecdote.
MIN_RUNS_FOR_CALIBRATION = 3


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_execution(
    telemetry: ExecutionTelemetry,
) -> Optional[CriticFinding]:
    """A run that did not complete, categorised by why."""
    if telemetry.success:
        return None

    message = telemetry.error_message or ""

    if "hypothetical" in message.lower():
        return CriticFinding(
            category="unverified_hardware", severity="info",
            algorithm=telemetry.algorithm, backend_type=telemetry.backend_type,
            summary="Execution refused because the mapping was hypothetical.",
            evidence=message,
            recommendation=(
                "Expected behaviour, not a defect: Step 7 ran in planning "
                "mode 'target', so the backend was declared rather than "
                "verified. Re-run with QHPC_PLANNING_MODE=detected on "
                "hardware that has it."
            ),
        )

    if "No executable implementation" in message:
        return CriticFinding(
            category="missing_implementation", severity="medium",
            algorithm=telemetry.algorithm, backend_type=telemetry.backend_type,
            summary=f"'{telemetry.algorithm}' has no executable implementation.",
            evidence=message,
            recommendation=(
                "The framework can route to this algorithm but cannot run "
                "it. Either implement it in backend/engine/workloads.py or "
                "exclude it from candidate generation, so routing cannot "
                "select something unexecutable."
            ),
        )

    return CriticFinding(
        category="execution_failure", severity="high",
        algorithm=telemetry.algorithm, backend_type=telemetry.backend_type,
        summary=f"Execution failed on {telemetry.backend_type}.",
        evidence=message,
        recommendation=(
            "Step 7 assigned a backend the workload could not run on. "
            "Check that the backend's dependencies are installed, or that "
            "the mapper's capability flags match reality."
        ),
    )


def _check_prediction(
    telemetry: ExecutionTelemetry,
) -> Optional[CriticFinding]:
    """How far Step 6's runtime prediction was from the measurement."""
    ratio = telemetry.runtime_error_ratio
    if not telemetry.success or ratio is None:
        return None

    # A sub-50ms run is mostly interpreter startup; its ratio describes
    # overhead, not the cost model.
    if telemetry.actual_runtime_seconds < MIN_MEANINGFUL_RUNTIME_S:
        return None

    # A ratio of exactly 0.0 is reachable: it is rounded to 4 decimals, so
    # a 0.06s run against a 1e6s prediction rounds to zero, and 1.0/ratio
    # would crash. Uncalibrated constants with a steep exponent do predict
    # runtimes that large, so this is not hypothetical. Such a case is the
    # most pessimistic possible error, not an absent one.
    if ratio <= 0.0:
        deviation = float("inf")
    else:
        deviation = max(ratio, 1.0 / ratio)

    if deviation <= RATIO_GOOD:
        return None

    if deviation > RATIO_SEVERE or deviation == float("inf"):
        severity = "high"
    elif deviation > RATIO_POOR:
        severity = "medium"
    else:
        severity = "low"

    direction = "optimistic" if ratio > 1.0 else "pessimistic"
    deviation_text = (
        "more than 10000x" if deviation == float("inf")
        else f"{deviation:.1f}x"
    )

    return CriticFinding(
        category="prediction_error", severity=severity,
        algorithm=telemetry.algorithm, backend_type=telemetry.backend_type,
        summary=(
            f"Runtime prediction was {deviation_text} {direction} "
            f"for {telemetry.algorithm} on {telemetry.backend_type}."
        ),
        evidence=(
            f"predicted {telemetry.predicted_runtime_seconds:.4f}s, "
            f"measured {telemetry.actual_runtime_seconds:.4f}s, "
            f"ratio {ratio:.3f} (host: {telemetry.telemetry_host})"
        ),
        recommendation=(
            f"Scale base_runtime_s for '{telemetry.algorithm}' by "
            f"{ratio:.3f} and set calibration_host. "
            + ("Note the estimate was uncalibrated, so this measures an "
               "unmeasured model rather than a faulty one."
               if telemetry.compared_against_uncalibrated_estimate else
               "The profile is marked calibrated, so this is a genuine "
               "regression in a measured model.")
        ),
        predicted=telemetry.predicted_runtime_seconds,
        actual=telemetry.actual_runtime_seconds,
        ratio=ratio,
    )


def _check_correctness(
    telemetry: ExecutionTelemetry,
) -> Optional[CriticFinding]:
    """
    Whether the workload produced a right answer, judged against the
    target and tolerance the workload itself declared.

    This is the check that caught a Grover implementation returning noise
    while posting an excellent speedup curve. Timing alone could not have.
    """
    if not telemetry.success or telemetry.result_metric_value is None:
        return None

    target = telemetry.raw_metrics.get("metric_target")
    tolerance = telemetry.raw_metrics.get("metric_tolerance")
    behaviour = telemetry.raw_metrics.get("metric_rank_behaviour", "invariant")

    # rank_dependent metrics change by design; without a target there is
    # nothing to judge against.
    if target is None or tolerance is None or behaviour == "rank_dependent":
        return None

    deviation = abs(telemetry.result_metric_value - float(target))
    if deviation <= float(tolerance):
        return None

    return CriticFinding(
        category="correctness_failure", severity="critical",
        algorithm=telemetry.algorithm, backend_type=telemetry.backend_type,
        summary=(
            f"{telemetry.algorithm} completed but produced an incorrect "
            f"result on {telemetry.backend_type}."
        ),
        evidence=(
            f"{telemetry.result_metric_name} = "
            f"{telemetry.result_metric_value:.6g}, expected "
            f"{float(target):.6g} +/- {float(tolerance):.6g} "
            f"(deviation {deviation:.6g})"
            + (f". {telemetry.raw_metrics['metric_note']}"
               if telemetry.raw_metrics.get("metric_note") else "")
        ),
        recommendation=(
            "Timing from this run must not be reported as a result for "
            "this algorithm. The implementation computed the wrong answer, "
            "so any speedup it shows is a speedup at doing something "
            "incorrect. Fix the workload before using the measurement."
        ),
    )


def _check_routing(
    telemetry: ExecutionTelemetry,
    mapping: Optional[HardwareMapping],
) -> Optional[CriticFinding]:
    """
    Whether Step 7's backend choice is contradicted by the measurement.

    Only flags cases where the measurement itself is the evidence: the
    chosen backend's threshold was never actually crossed. Comparing
    against a backend that was never run would be speculation.
    """
    if not telemetry.success or mapping is None:
        return None

    profile = CLASSICAL_RESOURCE_PROFILES.get(telemetry.algorithm)

    if (telemetry.backend_type == "gpu_cuda"
            and telemetry.actual_peak_memory_gb > 0
            and telemetry.actual_peak_memory_gb < GPU_MEMORY_THRESHOLD_GB):
        return CriticFinding(
            category="suboptimal_routing", severity="low",
            algorithm=telemetry.algorithm, backend_type=telemetry.backend_type,
            summary=(
                "Routed to CUDA on a predicted memory footprint the "
                "measurement did not support."
            ),
            evidence=(
                f"predicted {telemetry.predicted_memory_gb:.4f} GB crossed "
                f"the {GPU_MEMORY_THRESHOLD_GB} GB CUDA threshold, but "
                f"measured peak was {telemetry.actual_peak_memory_gb:.4f} GB"
            ),
            recommendation=(
                "The routing decision rested on an overestimate. Calibrate "
                "the memory model for this algorithm; the GPU may still be "
                "the right choice on throughput grounds, but not for the "
                "reason given."
            ),
        )

    if (telemetry.backend_type == "cpu" and profile is not None
            and profile.get("mpi_capable")
            and telemetry.actual_runtime_seconds >= MPI_RUNTIME_THRESHOLD_S):
        available = mapping.reasoning_trace.get("mpi_available")
        if available:
            return CriticFinding(
                category="suboptimal_routing", severity="medium",
                algorithm=telemetry.algorithm,
                backend_type=telemetry.backend_type,
                summary=(
                    "Ran on CPU despite MPI being available and the "
                    "measured runtime exceeding the MPI threshold."
                ),
                evidence=(
                    f"predicted {telemetry.predicted_runtime_seconds:.3f}s "
                    f"was below the {MPI_RUNTIME_THRESHOLD_S}s threshold, "
                    f"but the run took "
                    f"{telemetry.actual_runtime_seconds:.3f}s"
                ),
                recommendation=(
                    "Step 6 underestimated, so Step 7 never considered MPI. "
                    "Calibrating the runtime model would correct the "
                    "routing decision as well as the estimate."
                ),
            )

    return None


def _check_calibration(
    telemetry: ExecutionTelemetry,
) -> Optional[CriticFinding]:
    """Flag reliance on constants that were never measured."""
    if not telemetry.compared_against_uncalibrated_estimate:
        return None
    if not telemetry.success:
        return None

    return CriticFinding(
        category="uncalibrated_estimate", severity="info",
        algorithm=telemetry.algorithm, backend_type=telemetry.backend_type,
        summary=(
            f"The prediction for {telemetry.algorithm} used uncalibrated "
            f"constants."
        ),
        evidence=(
            f"calibration_host="
            f"{telemetry.raw_metrics.get('calibration_host') or 'none'}; "
            f"constants derived from complexity-class reasoning, not "
            f"measurement"
        ),
        recommendation=(
            "Run backend/tests/benchmark_resources.py on this host and set "
            "calibrated=True with calibration_host. Until then, any error "
            "ratio measures an unmeasured model, not a faulty one, and "
            "must be described that way in reporting."
        ),
    )


# ---------------------------------------------------------------------------
# Calibration feedback
# ---------------------------------------------------------------------------

def _suggest_calibration(
    runs: Sequence[ExecutionTelemetry],
) -> List[CalibrationSuggestion]:
    """
    Turn repeated observations into concrete constant corrections.

    This closes the loop: Step 6 predicts, Step 8 measures, Step 9 computes
    the multiplier that would have made the median prediction exact.
    """
    grouped: Dict[str, List[ExecutionTelemetry]] = {}
    for telemetry in runs:
        if (telemetry.success and telemetry.runtime_error_ratio
                and telemetry.actual_runtime_seconds >= MIN_MEANINGFUL_RUNTIME_S):
            grouped.setdefault(telemetry.algorithm, []).append(telemetry)

    suggestions: List[CalibrationSuggestion] = []
    for algorithm, group in sorted(grouped.items()):
        ratios = [t.runtime_error_ratio for t in group]
        median_ratio = float(statistics.median(ratios))

        profile = CLASSICAL_RESOURCE_PROFILES.get(algorithm)
        current = profile.get("base_runtime_s") if profile else None
        suggested = round(current * median_ratio, 6) if current else None

        # One run is an anecdote; agreement across runs is evidence.
        if len(group) >= MIN_RUNS_FOR_CALIBRATION:
            spread = max(ratios) / min(ratios) if min(ratios) > 0 else float("inf")
            confidence = "high" if spread < 1.5 else "moderate"
        elif len(group) > 1:
            confidence = "moderate"
        else:
            confidence = "low"

        hosts = {t.telemetry_host for t in group}
        note = (
            f"median of {len(group)} run(s)"
            + ("; MIXED HOSTS, runtimes do not transfer between machines "
               "and this suggestion is unsafe to apply"
               if len(hosts) > 1 else "")
            + ("" if len(group) >= MIN_RUNS_FOR_CALIBRATION else
               f"; fewer than {MIN_RUNS_FOR_CALIBRATION} runs, treat as "
               f"provisional")
        )

        suggestions.append(CalibrationSuggestion(
            algorithm=algorithm,
            observed_runs=len(group),
            median_ratio=round(median_ratio, 4),
            current_base_runtime_s=current,
            suggested_base_runtime_s=suggested,
            calibration_host=" + ".join(sorted(hosts)),
            confidence=confidence,
            note=note,
        ))

    return suggestions


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def critique(
    runs: Sequence[ExecutionTelemetry],
    mappings: Optional[Sequence[Optional[HardwareMapping]]] = None,
    estimates: Optional[Sequence[Optional[ResourceEstimate]]] = None,
) -> CriticReport:
    """
    Judge the framework's predictions and routing against measurement.

    `mappings` and `estimates` are positional companions to `runs`; pass
    them to enable routing checks. Telemetry alone still produces
    prediction, correctness and calibration findings.
    """
    mappings = list(mappings or [None] * len(runs))
    estimates = list(estimates or [None] * len(runs))
    # Tolerate short companion lists rather than failing on an index error:
    # a caller analysing archived telemetry may not have kept them.
    mappings += [None] * (len(runs) - len(mappings))
    estimates += [None] * (len(runs) - len(estimates))

    findings: List[CriticFinding] = []
    for telemetry, mapping in zip(runs, mappings):
        for check in (
            _check_execution(telemetry),
            _check_correctness(telemetry),
            _check_prediction(telemetry),
            _check_routing(telemetry, mapping),
            _check_calibration(telemetry),
        ):
            if check is not None:
                findings.append(check)

    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.algorithm))

    succeeded = [t for t in runs if t.success]
    usable = [
        t for t in succeeded
        if t.runtime_error_ratio is not None
        and t.actual_runtime_seconds >= MIN_MEANINGFUL_RUNTIME_S
    ]
    ratios = [t.runtime_error_ratio for t in usable]
    within_2x = sum(1 for r in ratios if 1 / RATIO_GOOD <= r <= RATIO_GOOD)

    median_ratio = float(statistics.median(ratios)) if ratios else None
    hosts = sorted({t.telemetry_host for t in runs})
    all_uncalibrated = all(
        t.compared_against_uncalibrated_estimate for t in runs
    ) if runs else True

    report = CriticReport(
        runs_analysed=len(runs),
        runs_succeeded=len(succeeded),
        findings=findings,
        calibration_suggestions=_suggest_calibration(runs),
        median_runtime_ratio=round(median_ratio, 4) if median_ratio else None,
        runs_within_2x=within_2x,
        runs_with_usable_ratio=len(usable),
        all_estimates_uncalibrated=all_uncalibrated,
        telemetry_hosts=hosts,
        verdict=_build_verdict(
            runs, succeeded, usable, within_2x, median_ratio,
            findings, all_uncalibrated, hosts,
        ),
        reasoning_trace={
            "ratio_good": RATIO_GOOD,
            "ratio_poor": RATIO_POOR,
            "ratio_severe": RATIO_SEVERE,
            "min_meaningful_runtime_s": MIN_MEANINGFUL_RUNTIME_S,
            "min_runs_for_calibration": MIN_RUNS_FOR_CALIBRATION,
            "findings_by_severity": {
                sev: sum(1 for f in findings if f.severity == sev)
                for sev in ("critical", "high", "medium", "low", "info")
            },
            "runs_excluded_as_too_short": len(succeeded) - len(usable),
        },
    )
    return report


def _build_verdict(
    runs: Sequence[ExecutionTelemetry],
    succeeded: Sequence[ExecutionTelemetry],
    usable: Sequence[ExecutionTelemetry],
    within_2x: int,
    median_ratio: Optional[float],
    findings: Sequence[CriticFinding],
    all_uncalibrated: bool,
    hosts: Sequence[str],
) -> str:
    """Deterministic prose summary. No LLM involved."""
    if not runs:
        return "No executions to analyse."

    parts = [
        f"{len(succeeded)}/{len(runs)} runs completed."
    ]

    critical = [f for f in findings if f.severity == "critical"]
    if critical:
        names = ", ".join(sorted({f.algorithm for f in critical}))
        parts.append(
            f"CORRECTNESS FAILURE in {names}: the result was wrong, so any "
            f"timing from those runs must not be reported as a result."
        )

    if median_ratio is not None:
        parts.append(
            f"Cost model: median runtime ratio {median_ratio:.3f} "
            f"({within_2x}/{len(usable)} within 2x)."
        )
    else:
        parts.append(
            "Cost model: no run produced a usable ratio (all predictions "
            "were zero, or all runs too short to measure)."
        )

    if all_uncalibrated:
        parts.append(
            "All estimates used uncalibrated constants, so these ratios "
            "measure an unmeasured model rather than a faulty one."
        )

    if len(hosts) > 1:
        parts.append(
            f"Runs span {len(hosts)} hosts; runtimes do not transfer "
            f"between machines, so aggregate ratios mix incomparable "
            f"measurements."
        )

    high = [f for f in findings if f.severity == "high"]
    if high:
        parts.append(f"{len(high)} high-severity finding(s) require attention.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Optional LLM narrative
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """
You are the Critic Agent inside the Q-HPC framework.

You receive a COMPLETED deterministic analysis of execution telemetry. Your
only job is to write a short narrative interpretation of it.

STRICT RULES

1. Every number has already been computed. Never invent, recompute or
   contradict one.
2. Never add findings. Never remove or re-rank them.
3. If the analysis says estimates were uncalibrated, say that any large
   error ratio reflects an unmeasured model, not a faulty one.
4. If a correctness failure is present, say plainly that timings from that
   run are not reportable as results.
5. Three to five sentences. No headings, no bullet points, no preamble.

Return plain prose only.
"""


def add_llm_narrative(report: CriticReport, model: Optional[str] = None) -> CriticReport:
    """
    Attach an LLM narrative over findings that already exist.

    Deliberately separate from `critique()`: the deterministic report is
    complete on its own, and nothing downstream may depend on this having
    run. A failure here — no Ollama, no model, a timeout — leaves the
    report fully usable and records that the narrative is absent.
    """
    try:
        import ollama
        from backend.config.settings import OLLAMA_MODEL
    except Exception:  # noqa: BLE001
        report.llm_available = False
        return report

    evidence = {
        "runs_analysed": report.runs_analysed,
        "runs_succeeded": report.runs_succeeded,
        "median_runtime_ratio": report.median_runtime_ratio,
        "runs_within_2x": report.runs_within_2x,
        "runs_with_usable_ratio": report.runs_with_usable_ratio,
        "all_estimates_uncalibrated": report.all_estimates_uncalibrated,
        "verdict": report.verdict,
        "findings": [
            {"category": f.category, "severity": f.severity,
             "algorithm": f.algorithm, "summary": f.summary,
             "evidence": f.evidence}
            for f in report.findings
        ],
    }

    try:
        response = ollama.chat(
            model=model or OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": str(evidence)},
            ],
            options={"temperature": 0},
        )
        report.llm_narrative = response["message"]["content"].strip()
        report.llm_available = True
    except Exception as exc:  # noqa: BLE001
        report.llm_available = False
        report.reasoning_trace["llm_error"] = f"{type(exc).__name__}: {exc}"

    return report


def format_report(report: CriticReport) -> str:
    """Human-readable rendering, for CLI output and Step 10."""
    lines = [
        "=" * 72,
        "STEP 9 — CRITIC AGENT REPORT",
        "=" * 72,
        "",
        report.verdict,
        "",
    ]

    if report.findings:
        lines.append(f"FINDINGS ({len(report.findings)})")
        lines.append("-" * 72)
        for i, f in enumerate(report.findings, 1):
            lines += [
                f"{i}. [{f.severity.upper()}] {f.category} — "
                f"{f.algorithm} on {f.backend_type}",
                f"   {f.summary}",
                f"   evidence: {f.evidence}",
                f"   action:   {f.recommendation}",
                "",
            ]
    else:
        lines += ["No findings.", ""]

    if report.calibration_suggestions:
        lines.append("CALIBRATION SUGGESTIONS")
        lines.append("-" * 72)
        for s in report.calibration_suggestions:
            line = (f"  {s.algorithm}: multiply base_runtime_s by "
                    f"{s.median_ratio:.3f}")
            if s.current_base_runtime_s and s.suggested_base_runtime_s:
                line += (f" ({s.current_base_runtime_s} -> "
                         f"{s.suggested_base_runtime_s})")
            lines.append(line)
            lines.append(f"    confidence: {s.confidence}; {s.note}")
            lines.append(f"    host: {s.calibration_host}")
        lines.append("")

    if report.llm_narrative:
        lines += ["NARRATIVE", "-" * 72, report.llm_narrative, ""]
    elif not report.llm_available:
        lines += ["(No LLM narrative: model unavailable. The analysis above "
                  "is complete without it.)", ""]

    return "\n".join(lines)
