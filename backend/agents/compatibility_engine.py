from backend.models.problem_schema import ProblemSpecification
from backend.models.candidate_schema import CandidatePool
from backend.models.compatibility_schema import CompatibilityResult
from backend.utils.compatibility_library import ALGORITHM_COMPATIBILITY


class CompatibilityEngine:
    """
    Computes algorithm-level compatibility scores.
    """

    def _dimension_category(self, dimensionality: int) -> str:
        if dimensionality <= 50:
            return "small"
        elif dimensionality <= 200:
            return "medium"
        elif dimensionality <= 1000:
            return "large"
        else:
            return "very_large"

    def _dimension_bonus(self, preferred: str, actual: str) -> float:
        if preferred == actual:
            return 1.00

        neighbours = {
            "small": ["medium"],
            "medium": ["small", "large"],
            "large": ["medium", "very_large"],
            "very_large": ["large"],
        }

        if actual in neighbours.get(preferred, []):
            return 0.75

        return 0.40

    def _compatibility_score(
        self,
        prior_score: float,
        algorithm_name: str,
        actual_dimension: str,
    ) -> float:

        knowledge = ALGORITHM_COMPATIBILITY.get(algorithm_name)

        if knowledge is None:
            return prior_score

        dimension_bonus = self._dimension_bonus(
            knowledge["preferred_dimension"],
            actual_dimension,
        )

        score = (
            0.45 * prior_score
            + 0.25 * knowledge["constraint_fit"]
            + 0.20 * knowledge["complexity_fit"]
            + 0.10 * dimension_bonus
        )

        return round(min(score, 1.0), 3)

    def evaluate(
        self,
        problem: ProblemSpecification,
        candidates: CandidatePool,
    ) -> CompatibilityResult:

        dimension_category = self._dimension_category(problem.dimensionality)

        algorithm_scores = {}

        best_classical_algorithm = ""
        best_quantum_algorithm = ""

        best_classical_score = -1.0
        best_quantum_score = -1.0

        # Classical candidates
        for candidate in candidates.classical_candidates:

            score = self._compatibility_score(
                candidate.prior_score,
                candidate.name,
                dimension_category,
            )

            algorithm_scores[candidate.name] = score

            if score > best_classical_score:
                best_classical_score = score
                best_classical_algorithm = candidate.name

        # Quantum candidates
        for candidate in candidates.quantum_candidates:

            score = self._compatibility_score(
                candidate.prior_score,
                candidate.name,
                dimension_category,
            )

            algorithm_scores[candidate.name] = score

            if score > best_quantum_score:
                best_quantum_score = score
                best_quantum_algorithm = candidate.name

        return CompatibilityResult(
            algorithm_scores=dict(
    sorted(
        algorithm_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
),
            best_classical_algorithm=best_classical_algorithm,
            best_quantum_algorithm=best_quantum_algorithm,
            best_classical_score=round(best_classical_score, 3),
            best_quantum_score=round(best_quantum_score, 3),
        )