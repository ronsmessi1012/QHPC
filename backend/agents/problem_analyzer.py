import json
import ollama

from backend.models.problem_schema import ProblemSpecification
from backend.config.settings import OLLAMA_MODEL


class ProblemAnalyzer:
    """
    LLM-powered Problem Analyzer for the Q-HPC framework.

    Responsibilities
    ----------------
    1. Accept a natural-language computational problem.
    2. Extract structured computational metadata.
    3. Return a validated ProblemSpecification object.
    4. Do NOT recommend algorithms or hardware.
    """

    SYSTEM_PROMPT = """
You are the Problem Analyzer Agent inside the Q-HPC framework.

ROLE
Your ONLY responsibility is to extract computational metadata from a user's problem statement.

DO NOT:
- Recommend any classical algorithm.
- Recommend any quantum algorithm.
- Mention QAOA, VQE, QSVM, quantum annealing, Grover, etc.
- Mention CPU, GPU, CUDA, MPI, HPC, hardware, execution plans, or performance.
- Explain whether quantum is better than classical.

Return ONLY valid JSON.
No markdown.
No extra text.

Schema:

{
  "user_query": "...",
  "problem_class": "classification | regression | optimization | simulation | search | unknown",
  "dimensionality": integer (must be >= 1, e.g., variables, features, or qubits),
  "dataset_size": integer (must be >= 1, e.g., records, samples, cities, nodes, or qubits; never null),
  "linearity": "linear | nonlinear | unknown",
  "sparsity": "dense | sparse | unknown",
  "constraint_structure": "constrained | unconstrained | mixed | unknown",
  "quantum_candidate": boolean,
  "confidence": float,
  "reasoning": "one sentence describing only the computational characteristics"
}

GOOD reasoning examples:
- "Detected a constrained combinatorial optimization problem involving 500 decision variables."
- "Detected a nonlinear simulation problem involving quantum state evolution."
- "Detected a binary classification problem with tabular input features."

BAD reasoning examples:
- "QAOA is suitable for this problem."
- "Quantum annealing could improve performance."
- "Use CUDA for execution."
- "A hybrid quantum-classical method is recommended."
"""

    def analyze(self, query: str) -> ProblemSpecification:
        """
        Convert a natural-language computational problem into a
        validated ProblemSpecification object.
        """

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                format="json",
                options={
                    "temperature": 0
                }
            )

            raw_json = response["message"]["content"]

            data = json.loads(raw_json)

            import re

            # Remove algorithm or hardware recommendations from reasoning.
            FORBIDDEN_TERMS = [
                "qaoa", "vqe", "qsvm", "grover", "quantum annealing",
                "cuda", "gpu", "cpu", "mpi", "xgboost",
                "random forest", "svm", "hybrid quantum-classical"
            ]

            reasoning = data.get("reasoning", "")

            for term in FORBIDDEN_TERMS:
                reasoning = re.sub(term, "", reasoning, flags=re.IGNORECASE)

            # Clean repeated spaces.
            reasoning = " ".join(reasoning.split())

            data["reasoning"] = reasoning.strip(" ,.-")

            # Normalize confidence value
            if "confidence" in data and data["confidence"] is not None:
                data["confidence"] = float(data["confidence"])

            # Defensive fallback and integer casting for dimensionality and dataset_size
            dim = data.get("dimensionality")
            dset = data.get("dataset_size")

            if dim is None or dim < 1:
                dim = dset if (dset is not None and dset >= 1) else 1
            if dset is None or dset < 1:
                dset = dim if (dim is not None and dim >= 1) else 1

            data["dimensionality"] = int(dim)
            data["dataset_size"] = int(dset)

            required_fields = set(ProblemSpecification.model_fields.keys())
            received_fields = set(data.keys())

            missing_fields = required_fields - received_fields

            if missing_fields:
                raise ValueError(
                    f"Missing fields returned by LLM: {sorted(missing_fields)}"
                )

            filtered_data = {
                key: value
                for key, value in data.items()
                if key in required_fields
            }

            return ProblemSpecification(**filtered_data)
            
            # Normalize simulation metadata
            if (
                data["problem_class"] == "simulation"
                and "qubit" in query.lower()
            ):
                data["dataset_size"] = data["dimensionality"]

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Problem Analyzer returned invalid JSON: {e}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Problem Analyzer failed: {e}"
            )