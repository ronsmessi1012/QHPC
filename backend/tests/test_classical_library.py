from backend.utils.classical_library import get_classical_candidates


problem_types = [
    "classification",
    "regression",
    "optimization",
    "simulation",
    "search",
]

for problem in problem_types:

    print("=" * 60)
    print(problem.upper())
    print("=" * 60)

    candidates = get_classical_candidates(problem)

    for candidate in candidates:
        print(candidate.model_dump())

    print()