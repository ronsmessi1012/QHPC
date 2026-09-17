from backend.models.candidate_schema import AlgorithmCandidate
from backend.utils.constants import CLASSICAL_ALGORITHMS


def get_classical_candidates(problem_class: str):
    """
    Returns AlgorithmCandidate objects for a given problem class.
    """

    algorithms = CLASSICAL_ALGORITHMS.get(problem_class, [])

    candidates = []

    for algo in algorithms:
        candidates.append(
            AlgorithmCandidate(
                name=algo["name"],
                family="classical",
                algorithm_type=problem_class,
                prior_score=algo["score"],
                complexity=algo["complexity"],
                justification=algo["justification"],
            )
        )

    candidates.sort(
    key=lambda candidate: candidate.prior_score,
    reverse=True,
)

    return candidates