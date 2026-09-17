import json

import ollama

from backend.config.settings import OLLAMA_MODEL
from backend.models.arbitration_context_schema import ArbitrationContext
from backend.models.decision_schema import ExecutionDecision


class AgentSelector:
    """
    LLM-powered Borderline Arbitration Agent.
    """

    SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are the arbitration agent inside the Q-HPC framework.

You receive deterministic evidence from previous modules.

Your responsibility is ONLY to arbitrate borderline execution decisions.

IMPORTANT DECISION POLICY

1. Treat the deterministic scores as ground truth.
2. Never invent new compatibility scores.
3. Use routing_recommendation as the default routing strategy.
4. If routing_recommendation is "borderline_review":
      - selected_family may be "classical", "quantum", or "dual".
      - routing_mode MUST remain "borderline_review".
5. Use "dual_execution" ONLY when both algorithms should actually execute.
6. decision_confidence must reflect uncertainty:
      confidence = 1 - (borderline_score × 0.4)

Interpretation:

borderline_score ≥ 0.90
→ Very uncertain.
→ Confidence around 0.60–0.70.

0.70 ≤ borderline_score < 0.90
→ Moderate uncertainty.
→ Confidence around 0.70–0.80.

borderline_score < 0.70
→ Clear evidence.
→ Confidence above 0.80.

Return STRICT JSON.

Schema:

{
  "selected_family":"classical | quantum | dual | borderline",
  "selected_algorithm":"string",
  "routing_mode":"classical_only | quantum_only | dual_execution | borderline_review",
  "decision_confidence":0.0,
  "borderline_score":0.0,
  "explanation":"short explanation"
}
"""

    def select(self, context: ArbitrationContext) -> ExecutionDecision:

        response = ollama.chat(
            model=OLLAMA_MODEL,
            format="json",
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": context.model_dump_json(indent=2),
                },
            ],
        )

        raw = response["message"]["content"]
        data = json.loads(raw)

        # Clamp confidence using deterministic uncertainty.
        expected_confidence = round(
            max(0.55, min(0.95, 1 - context.borderline_score * 0.4)),
            3,
        )

        data["decision_confidence"] = expected_confidence
        data["routing_mode"] = context.routing_recommendation
        data["borderline_score"] = context.borderline_score

        # Resolve family consistently.
        if context.best_quantum_score > context.best_classical_score:
            data["selected_family"] = "quantum"
            data["selected_algorithm"] = context.best_quantum_algorithm
        elif context.best_classical_score > context.best_quantum_score:
            data["selected_family"] = "classical"
            data["selected_algorithm"] = context.best_classical_algorithm
        else:
            data["selected_family"] = "dual"

        # Clamp confidence from deterministic uncertainty.
        data["decision_confidence"] = expected_confidence

        # Preserve deterministic evidence.
        data["reasoning_trace"] = {
            "problem_confidence": context.confidence,
            "dimension_score": context.dimension_score,
            "constraint_score": context.constraint_score,
            "complexity_score": context.complexity_score,
            "classical_score": context.best_classical_score,
            "quantum_score": context.best_quantum_score,
            "score_margin": context.score_margin,
        }

        return ExecutionDecision(**data)