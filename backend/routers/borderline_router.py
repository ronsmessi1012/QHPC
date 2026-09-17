from backend.models.compatibility_schema import CompatibilityResult
from backend.models.borderline_schema import BorderlineResult


class BorderlineRouter:
    """
    Determines whether execution should be classical,
    quantum, dual, or escalated for LLM arbitration.
    """

    # Routing thresholds
    CLEAR_MARGIN = 0.25        # Strong winner
    DUAL_MARGIN = 0.15         # Close enough to execute both
    REVIEW_MARGIN = 0.05       # Extremely close -> LLM arbitration

    def route(self, compatibility: CompatibilityResult) -> BorderlineResult:

        classical = compatibility.best_classical_score
        quantum = compatibility.best_quantum_score

        margin = round(abs(classical - quantum), 3)
        borderline = round(1.0 - margin, 3)

        # Determine deterministic winner
        if classical > quantum:
            winner = "classical"
        elif quantum > classical:
            winner = "quantum"
        else:
            winner = "tie"

        # -------------------------------
        # Routing Logic
        # -------------------------------

        if margin >= self.CLEAR_MARGIN:
            routing = "classical_only" if winner == "classical" else "quantum_only"

            explanation = (
                f"{winner.capitalize()} candidates clearly outperform the other family "
                f"(margin = {margin})."
            )

        elif margin >= self.DUAL_MARGIN:
            routing = "dual_execution"

            explanation = (
                "Classical and quantum candidates are competitive. "
                "Execute both pipelines and compare results."
            )

        else:
            routing = "borderline_review"

            explanation = (
                "Compatibility scores are extremely close. "
                "Escalating decision to the LLM Agent Selector."
            )

        return BorderlineResult(
            classical_score=classical,
            quantum_score=quantum,
            score_margin=margin,
            borderline_score=borderline,
            routing_recommendation=routing,
            winner_family=winner,
            explanation=explanation,
        )