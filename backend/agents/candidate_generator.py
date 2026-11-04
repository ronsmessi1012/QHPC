from backend.models.problem_schema import ProblemSpecification
from backend.models.candidate_schema import CandidatePool

from backend.utils.classical_library import get_classical_candidates
from backend.utils.quantum_library import get_quantum_candidates


class CandidateGenerator:
    """
    Generates parallel classical and quantum candidate pools
    for a given computational problem.
    """

    def generate(self, problem: ProblemSpecification) -> CandidatePool:

        # Retrieve deterministic candidate libraries.
        classical_candidates = get_classical_candidates(problem.problem_class)

        # Always generate quantum candidates if they exist for this problem class.
        quantum_candidates = get_quantum_candidates(problem.problem_class)

        return CandidatePool(
            classical_candidates=classical_candidates,
            quantum_candidates=quantum_candidates,
        )