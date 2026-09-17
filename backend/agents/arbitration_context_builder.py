from backend.models.problem_schema import ProblemSpecification
from backend.models.feature_score_schema import FeatureScore
from backend.models.compatibility_schema import CompatibilityResult
from backend.models.borderline_schema import BorderlineResult
from backend.models.arbitration_context_schema import ArbitrationContext


class ArbitrationContextBuilder:
    """
    Combines deterministic outputs into a single LLM-ready context.
    """

    def build(
        self,
        problem: ProblemSpecification,
        features: FeatureScore,
        compatibility: CompatibilityResult,
        borderline: BorderlineResult,
    ) -> ArbitrationContext:

        summary = (
            f"Detected a {problem.problem_class} problem with "
            f"{problem.dataset_size} samples and dimensionality "
            f"{problem.dimensionality}. "
            f"Best classical candidate is "
            f"{compatibility.best_classical_algorithm} "
            f"({compatibility.best_classical_score}). "
            f"Best quantum candidate is "
            f"{compatibility.best_quantum_algorithm} "
            f"({compatibility.best_quantum_score}). "
            f"Borderline score is {borderline.borderline_score}."
        )

        return ArbitrationContext(
            user_query=problem.user_query,
            problem_class=problem.problem_class,
            confidence=problem.confidence,

            dimensionality=problem.dimensionality,
            dataset_size=problem.dataset_size,
            constraint_structure=problem.constraint_structure,

            dimension_score=features.dimension_score,
            constraint_score=features.constraint_score,
            complexity_score=features.complexity_score,

            best_classical_algorithm=compatibility.best_classical_algorithm,
            best_classical_score=compatibility.best_classical_score,

            best_quantum_algorithm=compatibility.best_quantum_algorithm,
            best_quantum_score=compatibility.best_quantum_score,

            score_margin=borderline.score_margin,
            borderline_score=borderline.borderline_score,
            routing_recommendation=borderline.routing_recommendation,

            summary=summary,
        )