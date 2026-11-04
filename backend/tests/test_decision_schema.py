from backend.models.decision_schema import ExecutionDecision


decision = ExecutionDecision(
    selected_family="dual",
    selected_algorithm="Quantum Approximate Optimization Algorithm (QAOA)",
    routing_mode="dual_execution",
    decision_confidence=0.71,
    borderline_score=0.58,
    explanation="Both classical and quantum optimization algorithms appear competitive for this constrained optimization problem.",
    reasoning_trace={
        "prior_score": 0.94,
        "problem_confidence": 0.80,
        "constraint_score": 0.92,
        "dimension_score": 0.76,
    },
)

print(decision.model_dump_json(indent=4))