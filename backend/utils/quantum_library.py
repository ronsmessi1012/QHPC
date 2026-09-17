from backend.models.candidate_schema import AlgorithmCandidate
from backend.utils.constants import QUANTUM_ALGORITHMS


def get_quantum_candidates(problem_class: str):
    """
    Returns AlgorithmCandidate objects for a given problem class.
    """

    algorithms = QUANTUM_ALGORITHMS.get(problem_class, [])

    candidates = []

    for algo in algorithms:
        candidates.append(
            AlgorithmCandidate(
                name=algo["name"],
                family="quantum",
                algorithm_type=problem_class,
                prior_score=algo["score"],
                complexity=algo["complexity"],
                justification=algo["justification"],
            )
        )

    return candidates