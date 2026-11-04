"""
Step 10 — Final Synthesis & Report Generator

Input:  any subset of the pipeline's artifacts, from ProblemSpecification
        through CriticReport
Output: ExecutionReport, renderable as Markdown or JSON

The pipeline's artifacts are individually auditable but collectively
scattered: nine schemas, each telling part of one query's story. This
module assembles them into the single document a reader needs — what was
asked, what the framework decided and why, what it predicted, what it
actually cost, and what may legitimately be claimed from that.

TOLERANT OF PARTIAL RUNS
------------------------
Every stage is optional. A report may be built from routing alone, or from
archived telemetry whose upstream context is gone. A missing stage is
recorded as missing; it is never inferred, because a reconstructed
decision in a results document is indistinguishable from a real one.

CAVEATS ARE STRUCTURAL
----------------------
Every number invites a claim, and the caveats bound it. They are collected
as typed objects with severities rather than written as prose, so Step 10
cannot quietly omit one and a reader cannot skim past it.
`is_reportable_as_measurement` is the single question most readers
actually have.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.models.critic_schema import CriticReport
from backend.models.decision_schema import ExecutionDecision
from backend.models.hardware_schema import HardwareMapping
from backend.models.problem_schema import ProblemSpecification
from backend.models.report_schema import ExecutionReport, ReportCaveat
from backend.models.resource_schema import ResourceEstimate
from backend.models.telemetry_schema import ExecutionTelemetry

try:
    from backend.models.borderline_schema import BorderlineResult
    from backend.models.candidate_schema import CandidatePool
    from backend.models.compatibility_schema import CompatibilityResult
    from backend.models.feature_score_schema import FeatureScore
except ImportError:  # pragma: no cover - upstream schemas optional
    BorderlineResult = CandidatePool = CompatibilityResult = FeatureScore = None

ALL_STAGES = [
    "Step 1-2: Problem Analysis",
    "Step 3: Candidate Generation",
    "Step 4-5: Scoring & Compatibility",
    "Step 5.4: Borderline Routing",
    "Step 5.5: Arbitration",
    "Step 6: Resource Estimation",
    "Step 7: Hardware Mapping",
    "Step 8: Execution",
    "Step 9: Critique",
]


# ---------------------------------------------------------------------------
# Stage summarisers
# ---------------------------------------------------------------------------

def _summarise_problem(spec: ProblemSpecification) -> Dict[str, Any]:
    return {
        "problem_class": spec.problem_class,
        "confidence": spec.confidence,
        "dimensionality": spec.dimensionality,
        "dataset_size": spec.dataset_size,
        "constraint_structure": spec.constraint_structure,
        "linearity": spec.linearity,
        "sparsity": spec.sparsity,
        "quantum_candidate": spec.quantum_candidate,
        "reasoning": spec.reasoning,
    }


def _summarise_candidates(pool: Any) -> Dict[str, Any]:
    return {
        "classical_count": len(pool.classical_candidates),
        "quantum_count": len(pool.quantum_candidates),
        "top_classical": [
            {"name": c.name, "prior_score": c.prior_score}
            for c in pool.classical_candidates[:3]
        ],
        "top_quantum": [
            {"name": c.name, "prior_score": c.prior_score}
            for c in pool.quantum_candidates[:3]
        ],
    }


def _summarise_scoring(
    features: Any, compatibility: Any
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if features is not None:
        out.update({
            "dimension_score": features.dimension_score,
            "constraint_score": features.constraint_score,
            "complexity_score": features.complexity_score,
            "classical_affinity": features.classical_affinity,
            "quantum_affinity": features.quantum_affinity,
        })
    if compatibility is not None:
        out.update({
            "best_classical": compatibility.best_classical_algorithm,
            "best_classical_score": compatibility.best_classical_score,
            "best_quantum": compatibility.best_quantum_algorithm,
            "best_quantum_score": compatibility.best_quantum_score,
        })
    return out


def _summarise_routing(borderline: Any) -> Dict[str, Any]:
    return {
        "classical_score": borderline.classical_score,
        "quantum_score": borderline.quantum_score,
        "score_margin": borderline.score_margin,
        "borderline_score": borderline.borderline_score,
        "routing_recommendation": borderline.routing_recommendation,
        "winner_family": borderline.winner_family,
        "explanation": borderline.explanation,
    }


def _summarise_decision(decision: ExecutionDecision) -> Dict[str, Any]:
    return {
        "selected_family": decision.selected_family,
        "selected_algorithm": decision.selected_algorithm,
        "routing_mode": decision.routing_mode,
        "decision_confidence": decision.decision_confidence,
        "borderline_score": decision.borderline_score,
        "explanation": decision.explanation,
        "reasoning_trace": decision.reasoning_trace,
    }


def _summarise_estimate(estimate: ResourceEstimate) -> Dict[str, Any]:
    out = {
        "predicted_memory_gb": estimate.estimated_memory_gb,
        "predicted_runtime_seconds": estimate.estimated_runtime_seconds,
        "resource_tier": estimate.resource_tier,
        "parallelization_factor": estimate.parallelization_factor,
        "calibrated": estimate.calibrated,
        "hardware_is_hypothetical": estimate.hardware_is_hypothetical,
    }
    if estimate.qubit_count is not None:
        out.update({
            "qubit_count": estimate.qubit_count,
            "circuit_depth": estimate.circuit_depth,
            "simulation_strategy": estimate.simulation_strategy,
        })
    return out


def _summarise_mapping(mapping: HardwareMapping) -> Dict[str, Any]:
    out = {
        "backend_type": mapping.backend_type,
        "device_spec": mapping.device_spec,
        "justification": mapping.justification,
        "fallback_backend": mapping.fallback_backend,
        "is_hypothetical": mapping.is_hypothetical,
    }
    unavailable = mapping.reasoning_trace.get("unavailable_backends")
    if unavailable:
        out["unavailable_backends"] = unavailable
    return out


def _summarise_execution(telemetry: ExecutionTelemetry) -> Dict[str, Any]:
    return {
        "success": telemetry.success,
        "error_message": telemetry.error_message,
        "device_detail": telemetry.device_detail,
        "telemetry_host": telemetry.telemetry_host,
        "actual_runtime_seconds": telemetry.actual_runtime_seconds,
        "actual_peak_memory_gb": telemetry.actual_peak_memory_gb,
        "runtime_error_ratio": telemetry.runtime_error_ratio,
        "memory_error_ratio": telemetry.memory_error_ratio,
        "result_metric_name": telemetry.result_metric_name,
        "result_metric_value": telemetry.result_metric_value,
    }


def _summarise_critic(report: CriticReport) -> Dict[str, Any]:
    return {
        "verdict": report.verdict,
        "runs_analysed": report.runs_analysed,
        "runs_succeeded": report.runs_succeeded,
        "median_runtime_ratio": report.median_runtime_ratio,
        "findings_count": len(report.findings),
        "findings": [
            {"severity": f.severity, "category": f.category,
             "summary": f.summary, "evidence": f.evidence,
             "recommendation": f.recommendation}
            for f in report.findings
        ],
        "calibration_suggestions": [
            s.model_dump() for s in report.calibration_suggestions
        ],
        "llm_narrative": report.llm_narrative,
    }


# ---------------------------------------------------------------------------
# Caveat collection
# ---------------------------------------------------------------------------

def _collect_caveats(
    estimate: Optional[ResourceEstimate],
    mapping: Optional[HardwareMapping],
    telemetry: Optional[ExecutionTelemetry],
    critic: Optional[CriticReport],
    stages_missing: List[str],
) -> List[ReportCaveat]:
    caveats: List[ReportCaveat] = []

    if telemetry is None:
        caveats.append(ReportCaveat(
            kind="not_executed", severity="blocking",
            message=(
                "This run was planned but never executed. Every cost figure "
                "below is a PREDICTION from Step 6's model, not a "
                "measurement, and must not be cited as a performance result."
            ),
        ))

    if mapping is not None and mapping.is_hypothetical:
        caveats.append(ReportCaveat(
            kind="hypothetical", severity="blocking",
            message=(
                f"The backend '{mapping.backend_type}' was DECLARED, not "
                f"verified present. Step 7 ran in planning mode 'target', so "
                f"this placement describes a hypothetical deployment."
            ),
        ))

    if critic is not None:
        blocking = [f for f in critic.findings if f.severity == "critical"]
        for finding in blocking:
            caveats.append(ReportCaveat(
                kind="incorrect_result", severity="blocking",
                message=(
                    f"{finding.summary} {finding.evidence} "
                    f"Timings from this run are not a performance result, "
                    f"because the computation returned a wrong answer."
                ),
            ))

    if estimate is not None and not estimate.calibrated:
        caveats.append(ReportCaveat(
            kind="uncalibrated", severity="important",
            message=(
                "The cost prediction used constants derived from "
                "complexity-class reasoning, not measurement. A large "
                "prediction error therefore reflects an unmeasured model "
                "rather than a faulty one."
            ),
        ))

    if telemetry is not None and telemetry.success:
        caveats.append(ReportCaveat(
            kind="host_specific", severity="note",
            message=(
                f"Runtime was measured on {telemetry.telemetry_host}. "
                f"Runtimes do not transfer between machines; this figure "
                f"describes that host only."
            ),
        ))

    if stages_missing:
        caveats.append(ReportCaveat(
            kind="partial_pipeline", severity="note",
            message=(
                f"{len(stages_missing)} pipeline stage(s) absent from this "
                f"report: {', '.join(stages_missing)}. Missing stages are "
                f"not reconstructed."
            ),
        ))

    return caveats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(
    *,
    user_query: str,
    spec: Optional[ProblemSpecification] = None,
    candidates: Any = None,
    features: Any = None,
    compatibility: Any = None,
    borderline: Any = None,
    decision: Optional[ExecutionDecision] = None,
    estimate: Optional[ResourceEstimate] = None,
    mapping: Optional[HardwareMapping] = None,
    telemetry: Optional[ExecutionTelemetry] = None,
    critic: Optional[CriticReport] = None,
) -> ExecutionReport:
    """
    Assemble whatever pipeline artifacts are available into one report.

    Keyword-only, because eleven positional arguments of similar type is a
    recipe for silently passing an estimate where a mapping belongs.
    """
    completed: List[str] = []
    missing: List[str] = []

    def stage(name: str, present: bool) -> None:
        (completed if present else missing).append(name)

    stage(ALL_STAGES[0], spec is not None)
    stage(ALL_STAGES[1], candidates is not None)
    stage(ALL_STAGES[2], features is not None or compatibility is not None)
    stage(ALL_STAGES[3], borderline is not None)
    stage(ALL_STAGES[4], decision is not None)
    stage(ALL_STAGES[5], estimate is not None)
    stage(ALL_STAGES[6], mapping is not None)
    stage(ALL_STAGES[7], telemetry is not None)
    stage(ALL_STAGES[8], critic is not None)

    caveats = _collect_caveats(estimate, mapping, telemetry, critic, missing)

    # Outcome, in order of severity. A wrong answer outranks a successful
    # run: the computation finished, but the result is not usable.
    if telemetry is None:
        outcome = "planned_only" if decision is not None else "incomplete"
    elif not telemetry.success:
        outcome = "failed"
    elif any(c.kind == "incorrect_result" for c in caveats):
        outcome = "failed"
    else:
        outcome = "measured"

    headline = _build_headline(decision, mapping, telemetry, outcome, caveats)

    return ExecutionReport(
        user_query=user_query,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report_host=f"{platform.system()} {platform.machine()}",
        problem_summary=_summarise_problem(spec) if spec else None,
        candidates_summary=_summarise_candidates(candidates) if candidates else None,
        scoring_summary=(_summarise_scoring(features, compatibility)
                         if (features is not None or compatibility is not None)
                         else None),
        routing_summary=_summarise_routing(borderline) if borderline else None,
        decision_summary=_summarise_decision(decision) if decision else None,
        estimate_summary=_summarise_estimate(estimate) if estimate else None,
        mapping_summary=_summarise_mapping(mapping) if mapping else None,
        execution_summary=_summarise_execution(telemetry) if telemetry else None,
        critic_summary=_summarise_critic(critic) if critic else None,
        outcome=outcome,
        headline=headline,
        caveats=caveats,
        stages_completed=completed,
        stages_missing=missing,
    )


def _build_headline(
    decision: Optional[ExecutionDecision],
    mapping: Optional[HardwareMapping],
    telemetry: Optional[ExecutionTelemetry],
    outcome: str,
    caveats: List[ReportCaveat],
) -> str:
    algorithm = decision.selected_algorithm if decision else "an algorithm"
    backend = mapping.backend_type if mapping else "an unassigned backend"

    if outcome == "incomplete":
        return "Pipeline did not reach a decision."

    if outcome == "planned_only":
        return (f"Planned: {algorithm} on {backend}. Not executed, so no "
                f"cost was measured.")

    if outcome == "failed":
        wrong = [c for c in caveats if c.kind == "incorrect_result"]
        if wrong:
            return (f"{algorithm} ran on {backend} but produced an "
                    f"INCORRECT result; its timing is not a performance "
                    f"measurement.")
        error = (telemetry.error_message or "")[:100] if telemetry else ""
        return f"{algorithm} failed on {backend}: {error}"

    ratio = telemetry.runtime_error_ratio if telemetry else None
    ratio_text = (f", {ratio:.2f}x the predicted time" if ratio else "")
    return (
        f"{algorithm} ran on {backend} in "
        f"{telemetry.actual_runtime_seconds:.3f}s{ratio_text}."
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_SEVERITY_MARK = {"blocking": "[BLOCKING]", "important": "[IMPORTANT]",
                  "note": "[note]"}


def render_markdown(report: ExecutionReport) -> str:
    """Render as a Markdown document suitable for a thesis appendix."""
    L: List[str] = [
        "# Q-HPC Execution Report",
        "",
        f"**Query:** {report.user_query}",
        "",
        f"**Outcome:** {report.outcome}  ",
        f"**Generated:** {report.generated_at} on {report.report_host}",
        "",
        "---",
        "",
        "## Summary",
        "",
        report.headline,
        "",
    ]

    if report.caveats:
        L += ["### How these numbers may be read", ""]
        for c in report.caveats:
            L.append(f"- {_SEVERITY_MARK[c.severity]} {c.message}")
        L.append("")
        L.append(
            f"**Citable as a measured result:** "
            f"{'yes' if report.is_reportable_as_measurement else 'NO'}"
        )
        L.append("")

    def section(title: str, data: Optional[Dict[str, Any]],
                order: Optional[List[str]] = None) -> None:
        if not data:
            return
        L.extend([f"## {title}", "", "| Field | Value |", "|---|---|"])
        keys = order or list(data)
        for k in keys:
            if k not in data or data[k] is None:
                continue
            v = data[k]
            if isinstance(v, (dict, list)):
                continue
            L.append(f"| `{k}` | {v} |")
        L.append("")

    section("Problem Analysis (Steps 1–2)", report.problem_summary)

    if report.candidates_summary:
        c = report.candidates_summary
        L += [
            "## Candidate Generation (Step 3)", "",
            f"{c['classical_count']} classical and {c['quantum_count']} "
            f"quantum candidates generated.", "",
            "| Family | Algorithm | Prior |", "|---|---|---|",
        ]
        for cand in c.get("top_classical", []):
            L.append(f"| classical | {cand['name']} | {cand['prior_score']} |")
        for cand in c.get("top_quantum", []):
            L.append(f"| quantum | {cand['name']} | {cand['prior_score']} |")
        L.append("")

    section("Scoring & Compatibility (Steps 4–5)", report.scoring_summary)
    section("Borderline Routing (Step 5.4)", report.routing_summary)
    section("Arbitration (Step 5.5)", report.decision_summary,
            order=["selected_family", "selected_algorithm", "routing_mode",
                   "decision_confidence", "borderline_score", "explanation"])
    section("Resource Estimate (Step 6)", report.estimate_summary)
    section("Hardware Mapping (Step 7)", report.mapping_summary)

    if report.execution_summary:
        e = report.execution_summary
        L += ["## Execution & Measurement (Step 8)", ""]
        if e["success"]:
            L += [
                "| Quantity | Predicted | Measured | Ratio |",
                "|---|---|---|---|",
            ]
            est = report.estimate_summary or {}
            L.append(
                f"| runtime (s) | {est.get('predicted_runtime_seconds', '—')} "
                f"| {e['actual_runtime_seconds']} "
                f"| {e['runtime_error_ratio'] or '—'} |"
            )
            L.append(
                f"| peak memory (GB) | {est.get('predicted_memory_gb', '—')} "
                f"| {e['actual_peak_memory_gb']} "
                f"| {e['memory_error_ratio'] or 'not measured'} |"
            )
            L += [
                "",
                f"**Device:** {e['device_detail']}  ",
                f"**Host:** {e['telemetry_host']}  ",
                f"**Correctness:** {e['result_metric_name']} = "
                f"{e['result_metric_value']}",
                "",
            ]
        else:
            L += [f"Execution failed: `{e['error_message']}`", ""]

    if report.critic_summary:
        c = report.critic_summary
        L += ["## Critique (Step 9)", "", c["verdict"], ""]
        if c["findings"]:
            L += ["| Severity | Category | Finding |", "|---|---|---|"]
            for f in c["findings"]:
                L.append(f"| {f['severity']} | {f['category']} | {f['summary']} |")
            L.append("")
        for s in c.get("calibration_suggestions", []):
            L.append(
                f"- Calibration: multiply `base_runtime_s` for "
                f"**{s['algorithm']}** by {s['median_ratio']} "
                f"(confidence: {s['confidence']})"
            )
        if c.get("calibration_suggestions"):
            L.append("")
        if c.get("llm_narrative"):
            L += ["### Narrative", "", c["llm_narrative"], ""]

    L += [
        "---", "",
        "## Pipeline Coverage", "",
        f"**Completed ({len(report.stages_completed)}):** "
        + ", ".join(report.stages_completed),
        "",
    ]
    if report.stages_missing:
        L += [
            f"**Not run ({len(report.stages_missing)}):** "
            + ", ".join(report.stages_missing),
            "",
        ]

    return "\n".join(L)


def render_json(report: ExecutionReport, indent: int = 2) -> str:
    return json.dumps(report.model_dump(), indent=indent, default=str)


def save_report(
    report: ExecutionReport, path_base: str
) -> Dict[str, str]:
    """
    Write both renderings. Markdown for reading, JSON for further analysis.

    Returns the paths written, so a caller can report them rather than
    reconstructing the names.
    """
    md_path, json_path = f"{path_base}.md", f"{path_base}.json"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    with open(json_path, "w", encoding="utf-8") as fh:
        fh.write(render_json(report))
    return {"markdown": md_path, "json": json_path}
