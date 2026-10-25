"""
Step 10 — ExecutionReport Schema

Produced by: backend/engine/report_generator.py

The final artifact: one query's complete journey through the pipeline,
from natural language to measured execution, with every intermediate
decision and its justification.

The schema is deliberately tolerant of missing stages. A report may be
generated from a partial run — routing without execution, or telemetry
recovered from an archive without its upstream context — and a stage that
did not happen is recorded as absent rather than guessed at.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ReportCaveat(BaseModel):
    """
    A limitation on how this report's numbers may be read.

    Caveats are first-class rather than footnotes. Every number in a
    results table invites a claim, and the caveats are what bound it.
    """

    kind: Literal[
        "uncalibrated",       # the prediction rested on reasoned constants
        "hypothetical",       # hardware was declared, not verified
        "incorrect_result",   # the workload produced a wrong answer
        "not_executed",       # the plan was never run
        "host_specific",      # the figure does not transfer between machines
        "partial_pipeline",   # upstream stages are missing
    ]
    severity: Literal["blocking", "important", "note"]
    message: str


class ExecutionReport(BaseModel):
    # --- Identity ---------------------------------------------------------
    user_query: str
    generated_at: str
    report_host: str

    # --- Stage summaries. None means the stage did not run. --------------
    problem_summary: Optional[Dict[str, Any]] = None       # Steps 1-2
    candidates_summary: Optional[Dict[str, Any]] = None    # Step 3
    scoring_summary: Optional[Dict[str, Any]] = None       # Steps 4-5
    routing_summary: Optional[Dict[str, Any]] = None       # Step 5.4
    decision_summary: Optional[Dict[str, Any]] = None      # Step 5.5
    estimate_summary: Optional[Dict[str, Any]] = None      # Step 6
    mapping_summary: Optional[Dict[str, Any]] = None       # Step 7
    execution_summary: Optional[Dict[str, Any]] = None     # Step 8
    critic_summary: Optional[Dict[str, Any]] = None        # Step 9

    # --- Headline --------------------------------------------------------
    outcome: Literal["measured", "planned_only", "failed", "incomplete"]
    headline: str = Field(..., description="One sentence stating what happened")

    caveats: List[ReportCaveat] = Field(default_factory=list)

    stages_completed: List[str] = Field(default_factory=list)
    stages_missing: List[str] = Field(default_factory=list)

    @property
    def is_reportable_as_measurement(self) -> bool:
        """
        Whether this report's timings may be cited as a measured result.

        False when execution did not happen, or happened but produced a
        wrong answer — a timing for a computation that returned noise is
        not a performance result.
        """
        return self.outcome == "measured" and not any(
            c.severity == "blocking" for c in self.caveats
        )
