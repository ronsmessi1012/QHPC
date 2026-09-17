from backend.models.candidate_schema import (
    AlgorithmCandidate,
    CandidatePool,
)

# One classical candidate
xgboost = AlgorithmCandidate(
    name="XGBoost",
    family="classical",
    algorithm_type="classification",
    suitability_score=0.91,
    complexity="O(n log n)",
    justification="Strong baseline for tabular classification datasets.",
)

# One quantum candidate
qaoa = AlgorithmCandidate(
    name="QAOA",
    family="quantum",
    algorithm_type="optimization",
    suitability_score=0.88,
    complexity="Depends on circuit depth (p) and qubit count.",
    justification="Designed for constrained combinatorial optimization problems.",
)

pool = CandidatePool(
    classical_candidates=[xgboost],
    quantum_candidates=[qaoa],
)

print(pool.model_dump_json(indent=4))