"""
AlgorithmCandidate / CandidatePool schema tests.

REPLACES the previous backend/tests/test_candidate_schema.py, which was a
stale demo script rather than a test:

  - it passed `suitability_score=`, a field name that no longer exists. The
    field is `prior_score` in docs/sampling.md, and Step 4 defines
    classical_affinity as "highest classical candidate prior score", so the
    rename happened and the test was never updated.
  - it had no assertions, so pytest ran the constructor at import time and
    the mismatch surfaced as a COLLECTION error rather than a test failure.

Note that `prior_score` is also distinct from the `score` key used in
utils/constants.py; candidate_generator.py maps one to the other.
"""

import pytest
from pydantic import ValidationError

from backend.models.candidate_schema import AlgorithmCandidate, CandidatePool


def make_candidate(**overrides):
    base = dict(
        name="XGBoost",
        family="classical",
        algorithm_type="classification",
        prior_score=0.95,
        complexity="O(n log n)",
        justification="Strong baseline for tabular classification datasets.",
    )
    return AlgorithmCandidate(**{**base, **overrides})


def test_classical_candidate_round_trips():
    candidate = make_candidate()
    assert candidate.name == "XGBoost"
    assert candidate.family == "classical"
    assert candidate.prior_score == 0.95


def test_quantum_candidate_round_trips():
    candidate = make_candidate(
        name="Quantum Approximate Optimization Algorithm (QAOA)",
        family="quantum", algorithm_type="optimization", prior_score=0.94,
        complexity="Depends on circuit depth p and qubit count",
        justification="Designed for constrained combinatorial optimization.")
    assert candidate.family == "quantum"
    assert candidate.prior_score == 0.94


def test_family_is_restricted_to_the_two_paradigms():
    """
    The whole framework splits on this field — Step 5 compares the best
    classical against the best quantum. A third value would be silently
    excluded from both pools.
    """
    with pytest.raises(ValidationError):
        make_candidate(family="hybrid")


@pytest.mark.parametrize("bad_score", [-0.01, 1.01])
def test_prior_score_must_be_a_probability(bad_score):
    """
    S_compat = 0.45*P + 0.25*C + 0.20*K + 0.10*D_bonus assumes every input
    is in [0, 1]; a prior outside that range would let a compatibility
    score exceed 1 and break the borderline margin arithmetic.
    """
    with pytest.raises(ValidationError):
        make_candidate(prior_score=bad_score)


def test_candidate_pool_holds_both_families():
    pool = CandidatePool(
        classical_candidates=[make_candidate()],
        quantum_candidates=[make_candidate(
            name="Grover Search", family="quantum",
            algorithm_type="search", prior_score=0.94,
            complexity="O(sqrt(N))",
            justification="Quadratic speedup for unstructured search.")],
    )
    assert len(pool.classical_candidates) == 1
    assert len(pool.quantum_candidates) == 1
    assert pool.classical_candidates[0].family == "classical"
    assert pool.quantum_candidates[0].family == "quantum"


def test_empty_pool_is_valid():
    """
    A problem class with no registered quantum algorithm yields an empty
    quantum pool; that is a legitimate state, not an error.
    """
    pool = CandidatePool()
    assert pool.classical_candidates == []
    assert pool.quantum_candidates == []


def test_field_names_match_the_documented_schema():
    """
    Guards the rename that broke the old test: if `prior_score` becomes
    something else again, this fails loudly instead of leaving a stale
    test to error at collection time.
    """
    expected = {"name", "family", "algorithm_type", "prior_score",
                "complexity", "justification"}
    assert set(AlgorithmCandidate.model_fields) == expected
    assert set(CandidatePool.model_fields) == {
        "classical_candidates", "quantum_candidates"}
