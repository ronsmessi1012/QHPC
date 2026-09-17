import sys
from pathlib import Path

# Add backend directory to sys.path so modules can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.problem_schema import ProblemSpecification


sample_problem = ProblemSpecification(
    user_query="Optimize delivery routes for 500 cities.",
    problem_class="computer_vision",
    dimensionality=500,
    dataset_size=500,
    linearity="nonlinear",
    sparsity="dense",
    constraint_structure="constrained",
    quantum_candidate=True,
    reasoning="Travelling Salesman style optimization problem.",
)

print(sample_problem.model_dump_json(indent=4))