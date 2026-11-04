from backend.models.borderline_schema import BorderlineResult

result = BorderlineResult(
    classical_score=0.902,
    quantum_score=0.958,
    score_margin=0.056,
    borderline_score=0.944,
    routing_recommendation="borderline_review",
    winner_family="quantum",
    explanation="Quantum and classical candidates have similar compatibility scores; LLM arbitration recommended.",
)

print(result.model_dump_json(indent=4))