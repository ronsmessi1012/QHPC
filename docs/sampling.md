# Q-HPC Backend Architecture Evolution (Step 0 → Step 5.4)

> **Project:** Q-HPC — Quantum-Classical High Performance Computing Execution Framework
>
> **Module Documentation:** Agent Selector Architecture Evolution
>
> **Version:** v0.5 (Completed up to Borderline Router)

---

# Table of Contents

- Overall Architecture Evolution
- Step 0 — Backend Infrastructure
- Step 1 — Problem Specification Schema
- Step 2 — LLM Problem Analyzer
- Step 3 — Dual Candidate Generation
- Step 4 — Feature Scoring Engine
- Step 5 — Compatibility Engine
- Step 5.4 — Borderline Router
- Mathematical Formulations
- Constants & Thresholds
- Important Schemas
- Information Flow
- Backend Folder Structure

---

# Overall Q-HPC Architecture Evolution

## Initial Architecture (Concept Stage)

The first conceptual workflow was a direct pipeline where the LLM analyzed the problem and immediately selected an execution path.

```text
User Query
     │
     ▼
Problem Analyzer
     │
     ▼
Candidate Generator
     │
     ▼
Agent Selector
     │
     ▼
Hardware Mapper
```

### Limitations

- Candidate selection depended entirely on LLM reasoning.
- No deterministic scoring mechanism.
- Difficult to explain routing decisions.
- No intermediate evidence generation.
- Impossible to debug why one algorithm was preferred.

---

## Final Architecture After Step 5.4

The architecture was redesigned into a **hybrid deterministic + LLM arbitration framework**.

```text
User Query
      │
      ▼
LLM Problem Analyzer (Qwen2.5)
      │
      ▼
ProblemSpecification
      │
      ▼
Candidate Generator
      │
      ▼
CandidatePool
      │
      ├──────────────┐
      ▼              ▼
Feature Scoring   Compatibility Engine
      │              │
      └──────┬───────┘
             ▼
      CompatibilityResult
             │
             ▼
      Borderline Router
             │
             ▼
      BorderlineResult
             │
             ▼
      LLM Agent Selector (Upcoming)
             │
             ▼
      ExecutionDecision
             │
             ▼
Hardware Mapper (Upcoming)
```

### Architectural Improvements

| Version | Improvement |
|----------|-------------|
| Initial | LLM-only routing. |
| Step 3 | Parallel classical & quantum candidate generation. |
| Step 4 | Deterministic feature extraction before LLM reasoning. |
| Step 5 | Algorithm-level compatibility reasoning. |
| Step 5.4 | Mathematical borderline routing before LLM arbitration. |

---

# Step 0 — Backend Infrastructure

## Backend Directory Layout

```text
backend/
│
├── agents/
├── routers/
├── engine/
├── models/
├── config/
├── utils/
├── outputs/
├── tests/
└── main.py
```

## Purpose of Each Module

| Folder | Responsibility |
|--------|----------------|
| `agents/` | AI reasoning modules. |
| `routers/` | Routing and arbitration logic. |
| `engine/` | Execution and HPC runtime modules. |
| `models/` | Pydantic schemas. |
| `config/` | Configuration settings. |
| `utils/` | Knowledge bases and helper libraries. |
| `tests/` | Unit and integration tests. |
| `outputs/` | Generated execution reports. |

---

# Step 1 — Problem Specification Schema

## Objective

Transform natural language computational problems into a standardized machine-readable schema.

## Architecture Update

```text
Natural Language Query
        │
        ▼
ProblemSpecification
```

## Schema Fields

| Field | Description |
|-------|-------------|
| `user_query` | Original user problem. |
| `problem_class` | classification / regression / optimization / simulation / search. |
| `confidence` | Confidence of problem classification. |
| `dimensionality` | Estimated feature or problem dimension. |
| `dataset_size` | Estimated dataset size. |
| `linearity` | Linear / Nonlinear / Unknown. |
| `sparsity` | Sparse / Dense / Unknown. |
| `constraint_structure` | Constrained / Mixed / Unconstrained / Unknown. |
| `quantum_candidate` | Whether quantum execution is plausible. |
| `reasoning` | Short reasoning from the analyzer. |

---

## Example Output

```json
{
  "problem_class":"optimization",
  "confidence":0.80,
  "dimensionality":500,
  "constraint_structure":"constrained",
  "quantum_candidate":true
}
```

---

# Step 2 — LLM Problem Analyzer

## Objective

Use **Qwen2.5 running locally through Ollama** to infer computational characteristics from natural language.

## Architecture Update

```text
User Query
      │
      ▼
Qwen2.5 Problem Analyzer
      │
      ▼
ProblemSpecification
```

## Important Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| Ollama | Local inference server. |
| Qwen2.5:7B | Problem reasoning model. |
| JSON Mode | Forces strict structured output. |
| Pydantic Validation | Rejects malformed outputs. |

---

## Design Decisions

- Output **only JSON**.
- No markdown.
- No explanations outside JSON.
- Every field validated before entering the pipeline.

---

# Step 3 — Dual Candidate Generation

## Objective

Generate **parallel classical and quantum algorithm pools** instead of immediately selecting one execution path.

## Architecture Update

```text
ProblemSpecification
        │
        ▼
Candidate Generator
        │
  ┌──────────────┬──────────────┐
  ▼              ▼
Classical Pool  Quantum Pool
```

---

## CandidatePool Schema

| Field | Description |
|-------|-------------|
| `classical_candidates` | Classical candidate algorithms. |
| `quantum_candidates` | Quantum candidate algorithms. |

---

## Classical Knowledge Base

| Problem Type | Algorithms |
|--------------|------------|
| Classification | XGBoost, Random Forest, SVM, Logistic Regression |
| Regression | Linear Regression, Random Forest Regressor, XGBoost Regressor |
| Optimization | Branch & Bound, Genetic Algorithm, Simulated Annealing |
| Simulation | Monte Carlo Simulation, Finite Difference Solver |
| Search | A*, Dijkstra |

---

## Quantum Knowledge Base

| Problem Type | Algorithms |
|--------------|------------|
| Classification | QSVM, Quantum Kernel Classifier |
| Regression | Variational Quantum Regressor |
| Optimization | QAOA, VQE |
| Simulation | Quantum Statevector Simulation, Hamiltonian Simulation |
| Search | Grover Search |

---

## Prior Score Definition

Each algorithm receives an expert prior score.

### Constraint

```math
0 \le PriorScore \le 1
```

### Candidate Sorting Formula

```math
CandidatePool = Sort(PriorScore,\ descending)
```

Highest prior score always appears first.

---

## Important Prior Scores

| Algorithm | Prior Score |
|-----------|------------|
| XGBoost | **0.95** |
| QAOA | **0.94** |
| Grover Search | **0.94** |
| Statevector Simulation | **0.95** |
| Branch & Bound | **0.90** |
| Genetic Algorithm | **0.87** |
| QSVM | **0.89** |

---

# Step 4 — Feature Scoring Engine

## Objective

Compute deterministic **problem-level evidence** before LLM arbitration.

## Architecture Update

```text
ProblemSpecification
        │
        ▼
Feature Scoring Engine
        │
        ▼
FeatureScore
```

---

## FeatureScore Schema

| Feature | Description |
|---------|-------------|
| `dimension_score` | Problem size score. |
| `constraint_score` | Constraint difficulty score. |
| `complexity_score` | Computational complexity score. |
| `classical_affinity` | Highest classical prior score. |
| `quantum_affinity` | Highest quantum prior score. |

---

## Dimension Score

| Problem Dimension | Score |
|-------------------|------|
| ≤ 50 | **0.30** |
| 51–200 | **0.55** |
| 201–1000 | **0.80** |
| >1000 | **0.95** |

### Reason

Higher-dimensional problems are more likely to benefit from advanced optimization strategies.

---

## Constraint Score

| Constraint Structure | Score |
|----------------------|------|
| Unconstrained | **0.30** |
| Mixed | **0.60** |
| Constrained | **0.95** |
| Unknown | **0.50** |

### Reason

Constrained optimization problems have stronger potential quantum applicability.

---

## Complexity Score

| Problem Class | Score |
|---------------|------|
| Regression | **0.45** |
| Classification | **0.55** |
| Search | **0.65** |
| Simulation | **0.90** |
| Optimization | **0.95** |

### Reason

Represents intrinsic computational difficulty.

---

## Affinity Formulae

### Classical Affinity

```math
Affinity_C=\max(PriorScore_C)
```

### Quantum Affinity

```math
Affinity_Q=\max(PriorScore_Q)
```

These are family-level scores.

---

# Step 5 — Compatibility Engine

## Why This Module Was Added

Feature scores operate only at the **problem family level**.

The compatibility engine introduces **algorithm-level reasoning**.

---

## Architecture Evolution

### Before

```text
Quantum Affinity = 0.94
Classical Affinity = 0.90
```

### After

```text
QAOA                  → 0.958
VQE                   → 0.816
Genetic Algorithm     → 0.902
Branch & Bound        → 0.882
```

---

## Compatibility Knowledge Base

Every algorithm stores deterministic behavioral priors.

| Field | Description |
|-------|-------------|
| `constraint_fit` | Suitability for constrained problems. |
| `complexity_fit` | Suitability for computational complexity. |
| `preferred_dimension` | Preferred problem scale. |

---

## Preferred Dimension Categories

| Category | Range |
|----------|------|
| Small | ≤50 |
| Medium | 51–200 |
| Large | 201–1000 |
| Very Large | >1000 |

---

## Dimension Bonus Mechanism

| Actual Problem vs Preferred | Bonus |
|-----------------------------|------|
| Exact Match | **1.00** |
| Adjacent Category | **0.75** |
| Otherwise | **0.40** |

### Purpose

Reward algorithms naturally designed for problems of similar scale.

---

## Compatibility Formula

```math
Compatibility =
0.45 \times PriorScore +
0.25 \times ConstraintFit +
0.20 \times ComplexityFit +
0.10 \times DimensionBonus
```

---

## Example — QAOA

```math
0.45(0.94)+0.25(0.98)+0.20(0.95)+0.10(1.00)=0.958
```

---

## Example — Branch & Bound

```math
0.45(0.90)+0.25(0.95)+0.20(0.82)+0.10(0.75)=0.882
```

---

## CompatibilityResult Schema

| Field | Description |
|-------|-------------|
| `algorithm_scores` | Score for every algorithm. |
| `best_classical_algorithm` | Highest scoring classical algorithm. |
| `best_quantum_algorithm` | Highest scoring quantum algorithm. |
| `best_classical_score` | Highest classical compatibility score. |
| `best_quantum_score` | Highest quantum compatibility score. |

---

## Example Compatibility Output

```json
{
  "algorithm_scores":{
    "QAOA":0.958,
    "Genetic Algorithm":0.902,
    "Branch and Bound":0.882
  },
  "best_classical_algorithm":"Genetic Algorithm",
  "best_quantum_algorithm":"QAOA"
}
```

---

# Step 5.4 — Borderline Router

## Objective

Perform **deterministic arbitration** before invoking the LLM Agent Selector.

Only uncertain cases are escalated.

---

## Architecture Update

```text
CompatibilityResult
        │
        ▼
Borderline Router
        │
        ▼
BorderlineResult
```

---

## BorderlineResult Schema

| Field | Description |
|-------|-------------|
| `classical_score` | Best classical compatibility score. |
| `quantum_score` | Best quantum compatibility score. |
| `score_margin` | Difference between families. |
| `borderline_score` | Similarity between families. |
| `routing_recommendation` | Deterministic routing recommendation. |
| `winner_family` | Family with higher score. |
| `explanation` | Human-readable reasoning. |

---

## Mathematical Formulae

### Score Margin

```math
Margin=|S_C-S_Q|
```

---

### Borderline Similarity Score

```math
BorderlineScore=1-Margin
```

**Interpretation**

- Larger margin → clearer winner.
- Smaller margin → more uncertainty.

---

### Winner Function

```math
Winner=\arg\max(S_C,S_Q)
```

---

## Routing Thresholds

| Score Margin | Routing Recommendation | Meaning |
|--------------|-----------------------|---------|
| Margin ≥ **0.25** | `classical_only` / `quantum_only` | Clear deterministic winner. |
| **0.15 ≤ Margin < 0.25** | `dual_execution` | Execute both pipelines and benchmark. |
| Margin < **0.15** | `borderline_review` | Escalate to LLM Agent Selector. |

---

## Example Outputs

### Optimization (500 Cities)

| Metric | Value |
|--------|------|
| Classical Score | **0.902** |
| Quantum Score | **0.958** |
| Margin | **0.056** |
| Borderline Score | **0.944** |
| Winner | Quantum |
| Recommendation | Borderline Review |

---

### Classification (20,000 Samples)

| Metric | Value |
|--------|------|
| Classical Score | **0.922** |
| Quantum Score | **0.827** |
| Margin | **0.095** |
| Borderline Score | **0.905** |
| Winner | Classical |
| Recommendation | Borderline Review |

---

### Search (1000 Nodes)

| Metric | Value |
|--------|------|
| Classical Score | **0.907** |
| Quantum Score | **0.945** |
| Margin | **0.038** |
| Borderline Score | **0.962** |
| Winner | Quantum |
| Recommendation | Borderline Review |

---

# Complete Mathematical Formulations Used So Far

## 1. Confidence Constraint

```math
0 \le Confidence \le 1
```

---

## 2. Prior Score Constraint

```math
0 \le PriorScore \le 1
```

---

## 3. Classical Affinity

```math
Affinity_C=\max(PriorScore_C)
```

---

## 4. Quantum Affinity

```math
Affinity_Q=\max(PriorScore_Q)
```

---

## 5. Compatibility Score

```math
Compatibility=
0.45P+
0.25C+
0.20K+
0.10D
```

Where

| Symbol | Meaning |
|--------|---------|
| P | Prior Score |
| C | Constraint Fit |
| K | Complexity Fit |
| D | Dimension Bonus |

---

## 6. Margin

```math
Margin=|S_C-S_Q|
```

---

## 7. Borderline Similarity

```math
BorderlineScore=1-Margin
```

---

## 8. Winner Selection

```math
Winner=\arg\max(S_C,S_Q)
```

---

# Constants & Thresholds Used in Q-HPC

## Feature Scoring Constants

| Parameter | Value | Purpose |
|-----------|------|---------|
| Small Dimension | ≤50 | Low-dimensional problems. |
| Medium Dimension | 51–200 | Medium search space. |
| Large Dimension | 201–1000 | Large search space. |
| Very Large Dimension | >1000 | Massive computational problems. |

---

## Compatibility Formula Weights

| Parameter | Weight |
|-----------|-------|
| Prior Score | **0.45** |
| Constraint Fit | **0.25** |
| Complexity Fit | **0.20** |
| Dimension Bonus | **0.10** |

**Total Weight = 1.00**

---

## Borderline Router Thresholds

| Constant | Value | Meaning |
|----------|------|---------|
| `CLEAR_MARGIN` | **0.25** | Clear winner threshold. |
| `DUAL_MARGIN` | **0.15** | Dual execution threshold. |
| `REVIEW_MARGIN` | **0.05** | Extremely close candidates. |

---

## Dimension Bonus Constants

| Condition | Bonus |
|-----------|------|
| Exact Match | **1.00** |
| Adjacent Match | **0.75** |
| Non-matching | **0.40** |

---

# Important Schemas Created So Far

| Schema | Purpose |
|--------|---------|
| `ProblemSpecification` | Standardized computational problem representation. |
| `AlgorithmCandidate` | Individual algorithm metadata. |
| `CandidatePool` | Parallel classical and quantum candidate pools. |
| `FeatureScore` | Deterministic problem-level scores. |
| `CompatibilityResult` | Algorithm-level compatibility scores. |
| `BorderlineResult` | Deterministic routing recommendation. |
| `ExecutionDecision` | Final routing decision (upcoming LLM output). |

---

# Information Flow Across Modules

| Stage | Input | Output |
|-------|-------|--------|
| Problem Analyzer | User Query | `ProblemSpecification` |
| Candidate Generator | `ProblemSpecification` | `CandidatePool` |
| Feature Scoring Engine | `ProblemSpecification`, `CandidatePool` | `FeatureScore` |
| Compatibility Engine | `ProblemSpecification`, `CandidatePool` | `CompatibilityResult` |
| Borderline Router | `CompatibilityResult` | `BorderlineResult` |
| LLM Agent Selector *(Next Step)* | `ProblemSpecification`, `FeatureScore`, `CompatibilityResult`, `BorderlineResult` | `ExecutionDecision` |

---

# Current Backend Folder Structure (After Step 5.4)

```text
backend/
│
├── agents/
│   ├── problem_analyzer.py
│   ├── candidate_generator.py
│   ├── feature_scoring_engine.py
│   └── compatibility_engine.py
│
├── routers/
│   └── borderline_router.py
│
├── models/
│   ├── problem_schema.py
│   ├── candidate_schema.py
│   ├── feature_score_schema.py
│   ├── compatibility_schema.py
│   ├── borderline_schema.py
│   └── decision_schema.py
│
├── utils/
│   ├── constants.py
│   ├── classical_library.py
│   ├── quantum_library.py
│   └── compatibility_library.py
│
├── config/
│   └── settings.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_problem_analyzer.py
│   ├── test_problem_analyzer_llm.py
│   ├── test_candidate_generator.py
│   ├── test_feature_scoring_engine.py
│   ├── test_compatibility_engine.py
│   ├── test_borderline_router.py
│   └── ...
│
├── outputs/
└── main.py
```

---

# Current Completion Status

| Module | Status |
|--------|--------|
| Backend Infrastructure | ✅ Completed |
| Problem Specification Schema | ✅ Completed |
| LLM Problem Analyzer | ✅ Completed |
| Dual Candidate Generation | ✅ Completed |
| Feature Scoring Engine | ✅ Completed |
| Compatibility Engine | ✅ Completed |
| Borderline Router | ✅ Completed |
| LLM Agent Selector | ⏳ Next Step |
| Resource Estimator | ⏳ Pending |
| Hardware Mapper | ⏳ Pending |
| Performance Engine | ⏳ Pending |
| Critic Agent | ⏳ Pending |
| Report Generator | ⏳ Pending |

---

## Key Design Principles Introduced Up to Step 5.4

1. **Hybrid Decision Pipeline:** Deterministic scoring precedes LLM reasoning.
2. **Parallel Candidate Generation:** Classical and quantum candidates are generated simultaneously.
3. **Algorithm-Level Compatibility:** Every candidate is evaluated independently using expert priors.
4. **Explainable Routing:** Mathematical compatibility and borderline scores justify routing decisions.
5. **Borderline Arbitration:** The LLM is invoked only when deterministic evidence indicates uncertainty, making the routing process transparent and reproducible.