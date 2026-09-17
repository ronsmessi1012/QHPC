from backend.models.compatibility_schema import CompatibilityResult

result = CompatibilityResult(
    algorithm_scores={
        "Branch and Bound": 0.88,
        "Genetic Algorithm": 0.91,
        "QAOA": 0.95,
        "VQE": 0.74,
    },
    best_classical_algorithm="Genetic Algorithm",
    best_quantum_algorithm="QAOA",
    best_classical_score=0.91,
    best_quantum_score=0.95,
)

print(result.model_dump_json(indent=4))