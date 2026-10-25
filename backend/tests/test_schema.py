"""
ProblemSpecification schema tests.

REPLACES the previous backend/tests/test_schema.py, which was a stale demo
script rather than a test:

  - it used problem_class="computer_vision", which is not one of the six
    documented classes and does not appear in problem_analyzer.py's prompt;
  - it omitted `confidence` entirely;
  - it imported `from models.problem_schema import ...`, which only works
    if backend/ happens to be on sys.path, unlike every other test here;
  - it had no assertions, so pytest ran the constructor at import time and
    a failure showed up as a COLLECTION error rather than a test failure.

The field set below matches the JSON schema in problem_analyzer.py's own
system prompt, which is the authoritative source for what the LLM is asked
to produce and what ProblemSpecification(**filtered_data) must accept.
"""

import pytest
from pydantic import ValidationError

from backend.models.problem_schema import ProblemSpecification


VALID = dict(
    user_query="Optimize delivery routes for 500 cities.",
    problem_class="optimization",
    confidence=0.80,
    dimensionality=500,
    dataset_size=500,
    linearity="nonlinear",
    sparsity="dense",
    constraint_structure="constrained",
    quantum_candidate=True,
    reasoning="Travelling-salesman style combinatorial optimization problem.",
)


def test_valid_specification_round_trips():
    spec = ProblemSpecification(**VALID)
    assert spec.problem_class == "optimization"
    assert spec.dimensionality == 500
    assert spec.dataset_size == 500
    assert spec.quantum_candidate is True
    # The analyzer serializes this straight into downstream prompts.
    assert "optimization" in spec.model_dump_json()


@pytest.mark.parametrize("problem_class", [
    "classification", "regression", "optimization",
    "simulation", "search", "unknown",
])
def test_every_documented_problem_class_is_accepted(problem_class):
    spec = ProblemSpecification(**{**VALID, "problem_class": problem_class})
    assert spec.problem_class == problem_class


def test_undocumented_problem_class_is_rejected():
    """
    The six classes are a closed set: candidate_generator.py looks the class
    up in CLASSICAL_ALGORITHMS/QUANTUM_ALGORITHMS, so an unknown value would
    silently yield an empty candidate pool rather than an error.
    """
    with pytest.raises(ValidationError):
        ProblemSpecification(**{**VALID, "problem_class": "computer_vision"})


def test_confidence_must_be_a_probability():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            ProblemSpecification(**{**VALID, "confidence": bad})


@pytest.mark.parametrize("field", ["dimensionality", "dataset_size"])
def test_size_fields_must_be_at_least_one(field):
    """
    problem_analyzer.py coerces these to >= 1 before constructing the model.
    The schema enforces the same floor, because a zero would divide by zero
    in the Step 6 scale calculation.
    """
    with pytest.raises(ValidationError):
        ProblemSpecification(**{**VALID, field: 0})


def test_fields_match_what_the_analyzer_sends():
    """
    problem_analyzer.py builds `required_fields` from
    ProblemSpecification.model_fields and raises if the LLM omits any of
    them. A field added here without updating that prompt would make every
    analysis fail at runtime, so the two must agree.
    """
    expected = {
        "user_query", "problem_class", "confidence", "dimensionality",
        "dataset_size", "linearity", "sparsity", "constraint_structure",
        "quantum_candidate", "reasoning",
    }
    assert set(ProblemSpecification.model_fields) == expected
