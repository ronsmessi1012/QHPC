from backend.models.problem_schema import ProblemSpecification
from backend.models.candidate_schema import CandidatePool
from backend.models.feature_score_schema import FeatureScore


class FeatureScoringEngine:
    """
    Computes deterministic feature scores from ProblemSpecification.
    """

    def score(
        self,
        problem: ProblemSpecification,
        candidates: CandidatePool,
    ) -> FeatureScore:

        # -----------------------------
        # 1. Dimension Score
        # -----------------------------
        if problem.dimensionality <= 50:
            dimension_score = 0.30
        elif problem.dimensionality <= 200:
            dimension_score = 0.55
        elif problem.dimensionality <= 1000:
            dimension_score = 0.80
        else:
            dimension_score = 0.95

        # -----------------------------
        # 2. Constraint Score
        # -----------------------------
        constraint_lookup = {
            "unconstrained": 0.30,
            "mixed": 0.60,
            "constrained": 0.95,
            "unknown": 0.50,
        }

        constraint_score = constraint_lookup.get(
            problem.constraint_structure,
            0.50,
        )

        # -----------------------------
        # 3. Complexity Score
        # -----------------------------
        complexity_lookup = {
            "classification": 0.55,
            "regression": 0.45,
            "search": 0.65,
            "simulation": 0.90,
            "optimization": 0.95,
            "unknown": 0.50,
        }

        complexity_score = complexity_lookup.get(
            problem.problem_class,
            0.50,
        )

        # -----------------------------
        # 4. Affinity Scores
        # -----------------------------
        if candidates.classical_candidates:
            classical_affinity = max(
                candidate.prior_score
                for candidate in candidates.classical_candidates
            )
        else:
            classical_affinity = 0.0

        if candidates.quantum_candidates:
            quantum_affinity = max(
                candidate.prior_score
                for candidate in candidates.quantum_candidates
            )
        else:
            quantum_affinity = 0.0

        return FeatureScore(
            dimension_score=dimension_score,
            constraint_score=constraint_score,
            complexity_score=complexity_score,
            classical_affinity=classical_affinity,
            quantum_affinity=quantum_affinity,
        )