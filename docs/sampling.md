# Q-HPC Backend Architecture Evolution (Step 0 → Step 5.5)

> **Project:** Q-HPC — Quantum-Classical High Performance Computing Execution Framework
>
> **Module Documentation:** Architecture Evolution & Agent Selector Pipeline
>
> **Version:** v0.55 (Completed through Step 5.5: LLM Agent Selector Arbitration)

---

# Table of Contents

- [Overall Q-HPC Architecture Evolution](#overall-q-hpc-architecture-evolution)
  - [Initial Architecture (Concept Stage)](#initial-architecture-concept-stage)
  - [Evolution at Step 5.4 (Deterministic Routing)](#evolution-at-step-54-deterministic-routing)
  - [Final Architecture at Step 5.5 (Hybrid Arbitration)](#final-architecture-at-step-55-hybrid-arbitration)
  - [Architectural Improvements Across Versions](#architectural-improvements-across-versions)
- [Step 0 — Backend Infrastructure](#step-0--backend-infrastructure)
- [Step 1 — Problem Specification Schema](#step-1--problem-specification-schema)
- [Step 2 — LLM Problem Analyzer](#step-2--llm-problem-analyzer)
- [Step 3 — Dual Candidate Generation](#step-3--dual-candidate-generation)
- [Step 4 — Feature Scoring Engine](#step-4--feature-scoring-engine)
- [Step 5 — Compatibility Engine](#step-5--compatibility-engine)
- [Step 5.4 — Borderline Router](#step-54--borderline-router)
- [Step 5.5 — LLM Agent Selector Arbitration](#step-55--llm-agent-selector-arbitration)
  - [Step 5.5.1 — Arbitration Context Builder](#step-551--arbitration-context-builder)
  - [Step 5.5.2 — LLM Agent Selector](#step-552--llm-agent-selector)
  - [Deterministic Confidence Calibration](#deterministic-confidence-calibration)
  - [Reasoning Trace Mechanism](#reasoning-trace-mechanism)
  - [End-to-End Optimization Example](#end-to-end-optimization-example)
- [Complete Mathematical Formulations](#complete-mathematical-formulations)
- [Constants & Thresholds Registry](#constants--thresholds-registry)
- [Important Schemas Reference](#important-schemas-reference)
- [Information Flow Across Modules](#information-flow-across-modules)
- [Visualizations Suite (Steps V1 → V4)](#visualizations-suite-steps-v1--v4)
- [Backend Folder Structure](#backend-folder-structure)
- [Current Completion Status](#current-completion-status)
- [Key Design Principles](#key-design-principles)

---

# Overall Q-HPC Architecture Evolution

## Initial Architecture (Concept Stage)

The first conceptual workflow was a direct pipeline where an LLM analyzed the problem statement and immediately selected an execution path without intermediate quantitative validation.

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

### Limitations of Initial Concept
- Candidate selection depended entirely on unconstrained LLM heuristics.
- No deterministic scoring mechanism existed to quantify suitability.
- Difficult to explain or mathematically justify routing decisions.
- No intermediate evidence generation for debugging or validation.
- Impossible to audit why one algorithm was chosen over another.

---

## Evolution at Step 5.4 (Deterministic Routing)

At Step 5.4, the architecture introduced deterministic feature scoring, algorithm compatibility evaluation, and mathematical borderline detection before LLM engagement.

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
```

---

## Final Architecture at Step 5.5 (Hybrid Arbitration)

Step 5.5 completed the arbitration pipeline by introducing a unified **Arbitration Context Builder** and the **LLM Agent Selector**, creating a robust **hybrid deterministic–LLM arbitration framework**.

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
      ├────────────────────────────┐
      ▼                            ▼
Feature Scoring Engine      Compatibility Engine
      │                            │
FeatureScore                CompatibilityResult
      │                            │
      └────────────┬───────────────┘
                   ▼
            Borderline Router
                   │
            BorderlineResult
                   │
                   ▼
      Arbitration Context Builder
                   │
            ArbitrationContext
                   │
                   ▼
       LLM Agent Selector (Qwen2.5)
                   │
            ExecutionDecision
                   │
                   ▼
      Resource Estimator (Step 6)
                   │
                   ▼
        Hardware Mapper (Step 7)
```

---

## Architectural Improvements Across Versions

| Version / Stage | Architectural Improvement |
|-----------------|---------------------------|
| **Initial Concept** | Unconstrained LLM-only routing without intermediate validation. |
| **Step 3** | Parallel classical and quantum candidate generation (no premature pruning). |
| **Step 4** | Deterministic problem feature extraction (dimension, constraint, complexity). |
| **Step 5** | Fine-grained algorithm-level compatibility scoring using empirical priors. |
| **Step 5.4** | Mathematical borderline routing and thresholding before LLM escalation. |
| **Step 5.5** | Unified immutable arbitration context, prompt-constrained LLM arbitration, deterministic confidence calibration, and full reasoning traces. |

---

# Step 0 — Backend Infrastructure

## Backend Directory Layout

```text
backend/
│
├── agents/      # AI reasoning and scoring agents
├── routers/     # Routing and arbitration logic
├── engine/      # Execution engine and HPC runtime modules
├── models/      # Pydantic data schemas
├── config/      # Configuration and environment settings
├── utils/       # Knowledge bases, constants, and helper libraries
├── outputs/     # Generated execution reports, logs, and telemetry
├── tests/       # Unit and integration test suites
└── main.py      # Framework entry point
```

## Purpose of Each Module

| Folder | Responsibility |
|--------|----------------|
| `agents/` | LLM-powered agents and deterministic scoring engines. |
| `routers/` | Mathematical routing, thresholding, and arbitration logic. |
| `engine/` | CUDA, MPI, OpenMP runtime execution, telemetry, and monitoring. |
| `models/` | Strictly typed Pydantic models (data contracts only; no business logic). |
| `config/` | Environment variables, model names, and system thresholds. |
| `utils/` | Central algorithm libraries, compatibility knowledge bases, and constants. |
| `tests/` | Comprehensive test cases validating schemas and modules. |
| `outputs/` | JSON execution traces, telemetry metrics, and generated reports. |

---

# Step 1 — Problem Specification Schema

## Objective

Transform natural language computational problem statements into a standardized, machine-readable JSON contract used across all downstream agents.

```text
Natural Language Query
        │
        ▼
ProblemSpecification Schema
```

## Schema Fields (`ProblemSpecification`)

| Field | Type | Description |
|-------|------|-------------|
| `user_query` | `str` | Original natural-language problem statement. |
| `problem_class` | `Literal[...]` | Problem class: `classification`, `regression`, `optimization`, `simulation`, `search`, `unknown`. |
| `confidence` | `float` | Model confidence in classification ($0.0 \le c \le 1.0$). |
| `dimensionality` | `int` | Number of variables, features, or qubits ($\ge 1$). |
| `dataset_size` | `int` | Number of samples, records, nodes, or states ($\ge 1$). |
| `linearity` | `Literal[...]` | Linearity characteristics: `linear`, `nonlinear`, `unknown`. |
| `sparsity` | `Literal[...]` | Matrix/data sparsity: `dense`, `sparse`, `unknown`. |
| `constraint_structure` | `Literal[...]` | Constraint type: `unconstrained`, `constrained`, `mixed`, `unknown`. |
| `quantum_candidate` | `bool` | Flag indicating whether quantum methods are theoretically applicable. |
| `reasoning` | `Optional[str]` | Concise description of computational characteristics only. |

## Example Output

```json
{
  "user_query": "Optimize delivery routes for 500 cities while minimizing fuel cost under travel constraints.",
  "problem_class": "optimization",
  "confidence": 0.80,
  "dimensionality": 500,
  "dataset_size": 500,
  "linearity": "unknown",
  "sparsity": "unknown",
  "constraint_structure": "constrained",
  "quantum_candidate": true,
  "reasoning": "Detected a constrained combinatorial optimization problem involving 500 decision variables."
}
```

---

# Step 2 — LLM Problem Analyzer

## Objective

Use **Qwen2.5-7B running locally through Ollama** to parse natural language queries into validated `ProblemSpecification` objects.

```text
User Query ──► Qwen2.5 (Ollama, JSON Mode) ──► Pydantic Validator ──► ProblemSpecification
```

## Key Mechanisms & Design Decisions

| Mechanism | Implementation & Purpose |
|-----------|--------------------------|
| **Local Inference** | Ollama running `qwen2.5:7b-instruct` with `temperature=0` for reproducibility. |
| **Strict JSON Mode** | `format="json"` ensures valid JSON output without Markdown formatting. |
| **Negative Constraints** | Explicit prompt forbidding algorithm names (QAOA, XGBoost, etc.) or hardware mentions (GPU, CUDA, MPI) in reasoning. |
| **Sanitization Filter** | Regex-based post-processor stripping any leaked algorithm or hardware keywords. |
| **Defensive Fallbacks** | Integer coercion and automatic dimension/dataset size defaulting to prevent null values in simulation/search tasks. |

---

# Step 3 — Dual Candidate Generation

## Objective

Construct **parallel classical and quantum candidate pools** for every problem class, ensuring neither paradigm is prematurely eliminated before quantitative evaluation.

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

## Schema (`CandidatePool` & `AlgorithmCandidate`)

| Schema / Field | Type | Description |
|----------------|------|-------------|
| `AlgorithmCandidate.name` | `str` | Formal algorithm name. |
| `AlgorithmCandidate.family` | `Literal["classical", "quantum"]` | Candidate paradigm. |
| `AlgorithmCandidate.algorithm_type` | `str` | Problem class addressed. |
| `AlgorithmCandidate.prior_score` | `float` | Expert prior suitability score ($0.0 \le p \le 1.0$). |
| `AlgorithmCandidate.complexity` | `str` | Asymptotic time or circuit scaling. |
| `AlgorithmCandidate.justification` | `str` | Reason for candidate inclusion. |
| `CandidatePool.classical_candidates` | `List[AlgorithmCandidate]` | Sorted classical candidates. |
| `CandidatePool.quantum_candidates` | `List[AlgorithmCandidate]` | Sorted quantum candidates. |

---

## Classical & Quantum Knowledge Bases

### Classical Algorithm Registry

| Problem Class | Registered Algorithms | Key Prior Scores |
|---------------|-----------------------|------------------|
| **Classification** | XGBoost (0.95), Random Forest (0.90), SVM (0.86), Logistic Regression (0.80) | Strong tabular ML baselines. |
| **Regression** | XGBoost Regressor (0.93), Random Forest Regressor (0.88), Linear Regression (0.82) | Continuous value estimators. |
| **Optimization** | Branch and Bound (0.90), Genetic Algorithm (0.87), Simulated Annealing (0.84) | Exact & heuristic solvers. |
| **Simulation** | Monte Carlo Simulation (0.91), Finite Difference Solver (0.85) | Probabilistic & PDE solvers. |
| **Search** | A* Search (0.92), Dijkstra (0.88) | Informed graph traversals. |

### Quantum Algorithm Registry

| Problem Class | Registered Algorithms | Key Prior Scores |
|---------------|-----------------------|------------------|
| **Classification** | Quantum Support Vector Machine / QSVM (0.89), Quantum Kernel Classifier (0.86) | High-dimensional feature maps. |
| **Regression** | Variational Quantum Regressor / VQR (0.84) | Parameterized circuit regression. |
| **Optimization** | Quantum Approximate Optimization Algorithm / QAOA (0.94), VQE (0.83) | Variational combinatorial solvers. |
| **Simulation** | Quantum Statevector Simulation (0.95), Hamiltonian Simulation (0.90) | Exact state evolution & dynamics. |
| **Search** | Grover Search (0.94) | Quadratic search speedup. |

### Candidate Ordering Rule

All candidate lists are strictly sorted in descending order of prior score:

$$CandidatePool = \operatorname{Sort}(\text{prior-score}, \text{descending})$$

---

# Step 4 — Feature Scoring Engine

## Objective

Compute deterministic, problem-level mathematical scores from the `ProblemSpecification` before algorithm compatibility evaluation or LLM arbitration.

```text
ProblemSpecification + CandidatePool ──► Feature Scoring Engine ──► FeatureScore
```

## Schema (`FeatureScore`)

| Field | Type | Description |
|-------|------|-------------|
| `dimension_score` | `float` | Difficulty score derived from dimensionality. |
| `constraint_score` | `float` | Difficulty score derived from constraint structure. |
| `complexity_score` | `float` | Difficulty score derived from problem class complexity. |
| `classical_affinity` | `float` | Highest classical candidate prior score ($\max P_C$). |
| `quantum_affinity` | `float` | Highest quantum candidate prior score ($\max P_Q$). |

---

## Deterministic Scoring Lookups

### 1. Dimension Score ($S_{dim}$)

| Dimensionality ($D$) | Score | Rationale |
|---------------------|-------|-----------|
| $D \le 50$ | **0.30** | Low dimensional; well handled by simple classical methods. |
| $51 \le D \le 200$ | **0.55** | Medium search space. |
| $201 \le D \le 1000$ | **0.80** | Large search space; classical methods scale exponentially/polynomial. |
| $D > 1000$ | **0.95** | Massive computational problem requiring high HPC / quantum scaling. |

### 2. Constraint Score ($S_{con}$)

| Constraint Structure | Score | Rationale |
|----------------------|-------|-----------|
| `unconstrained` | **0.30** | Simple boundary conditions. |
| `mixed` | **0.60** | Partial equality / inequality constraints. |
| `constrained` | **0.95** | Heavy combinatorial constraints; high optimization complexity. |
| `unknown` | **0.50** | Neutral default. |

### 3. Complexity Score ($S_{comp}$)

| Problem Class | Score | Rationale |
|---------------|-------|-----------|
| `regression` | **0.45** | Standard continuous fitting. |
| `classification` | **0.55** | Decision boundary partitioning. |
| `search` | **0.65** | Graph / unstructured search complexity. |
| `simulation` | **0.90** | Exponential state space scaling ($O(2^n)$). |
| `optimization` | **0.95** | NP-hard / NP-complete combinatorial spaces. |
| `unknown` | **0.50** | Neutral default. |

### 4. Family Affinity Formulas

$$Affinity_C = \max_{c \in C_{classical}} (c.prior\_score)$$

$$Affinity_Q = \max_{q \in C_{quantum}} (q.prior\_score)$$

---

# Step 5 — Compatibility Engine

## Objective

Calculate algorithm-level compatibility scores by fusing algorithm prior suitability, constraint handling capability, complexity alignment, and dimension bonus.

```text
ProblemSpecification + CandidatePool ──► Compatibility Engine ──► CompatibilityResult
```

---

## Compatibility Knowledge Base Fields

Each algorithm entry in `ALGORITHM_COMPATIBILITY` contains:
- `constraint_fit` ($0.0 \le C \le 1.0$): How effectively the algorithm handles constraints.
- `complexity_fit` ($0.0 \le K \le 1.0$): Algorithm resilience against high complexity.
- `preferred_dimension`: Ideal problem scale (`small`, `medium`, `large`, `very_large`).

---

## Dimension Matching & Bonus Matrix

| Actual Problem Dimension ($D$) | Dimension Category |
|-------------------------------|--------------------|
| $D \le 50$ | `small` |
| $51 \le D \le 200$ | `medium` |
| $201 \le D \le 1000$ | `large` |
| $D > 1000$ | `very_large` |

### Dimension Bonus Multiplier ($D_{bonus}$)

| Relationship between Preferred & Actual Category | Multiplier ($D_{bonus}$) |
|---------------------------------------------------|--------------------------|
| **Exact Category Match** | **1.00** |
| **Adjacent Category** (e.g. `medium` vs `small` or `large`) | **0.75** |
| **Distant Category** | **0.40** |

---

## Compatibility Formula

For each candidate algorithm $a$, its final compatibility score $S_{compat}(a)$ is defined as:

$$S_{compat}(a) = 0.45 \cdot P_a + 0.25 \cdot C_a + 0.20 \cdot K_a + 0.10 \cdot D_{bonus}(a)$$

Where:
- $P_a$ = Candidate prior score
- $C_a$ = `constraint_fit`
- $K_a$ = `complexity_fit`
- $D_{bonus}(a)$ = Dimension category match bonus

### Calculation Examples

**QAOA on 500-city Constrained Optimization (Large Dimension):**
$$S_{compat}(\text{QAOA}) = 0.45(0.94) + 0.25(0.98) + 0.20(0.95) + 0.10(1.00) = 0.423 + 0.245 + 0.190 + 0.100 = \mathbf{0.958}$$

**Branch and Bound on 500-city Constrained Optimization (Medium Preferred vs Large Actual):**
$$S_{compat}(\text{B\&B}) = 0.45(0.90) + 0.25(0.95) + 0.20(0.82) + 0.10(0.75) = 0.405 + 0.2375 + 0.164 + 0.075 = \mathbf{0.882}$$

**Genetic Algorithm on 500-city Constrained Optimization (Large Dimension):**
$$S_{compat}(\text{GA}) = 0.45(0.87) + 0.25(0.92) + 0.20(0.90) + 0.10(1.00) = 0.3915 + 0.230 + 0.180 + 0.100 = \mathbf{0.902}$$

---

## Schema (`CompatibilityResult`)

| Field | Type | Description |
|-------|------|-------------|
| `algorithm_scores` | `Dict[str, float]` | Sorted mapping of algorithm names to compatibility scores. |
| `best_classical_algorithm` | `str` | Name of the highest-scoring classical algorithm. |
| `best_quantum_algorithm` | `str` | Name of the highest-scoring quantum algorithm. |
| `best_classical_score` | `float` | Highest classical compatibility score ($S_C$). |
| `best_quantum_score` | `float` | Highest quantum compatibility score ($S_Q$). |

---

# Step 5.4 — Borderline Router

## Objective

Evaluate the competitive margin between the top classical and quantum candidates. If one paradigm decisively wins, route immediately; if the margin is narrow, flag for dual execution or escalate to LLM arbitration.

```text
CompatibilityResult ──► Borderline Router ──► BorderlineResult
```

---

## Mathematical Formulation

### 1. Score Margin ($\Delta S$)
$$\Delta S = |S_C - S_Q|$$

### 2. Borderline Similarity Score ($B$)
$$B = 1.0 - \Delta S$$

- Higher $B$ ($B \ge 0.85$) indicates close competition and high uncertainty.
- Lower $B$ indicates a decisive winner.

### 3. Winner Determination
$$\text{Winner} = \begin{cases} 
\text{classical}, & \text{if } S_C > S_Q \\ 
\text{quantum}, & \text{if } S_Q > S_C \\ 
\text{tie}, & \text{if } S_C = S_Q 
\end{cases}$$

---

## Routing Thresholds

| Margin ($\Delta S$) | Routing Recommendation (`routing_recommendation`) | Meaning |
|---------------------|---------------------------------------------------|---------|
| $\Delta S \ge \mathbf{0.25}$ | `classical_only` or `quantum_only` | Decisive single-paradigm winner. |
| $\mathbf{0.15} \le \Delta S < \mathbf{0.25}$ | `dual_execution` | Both candidates viable; benchmark both in parallel. |
| $\Delta S < \mathbf{0.15}$ | `borderline_review` | Extremely close margin; escalate to LLM Agent Selector. |

---

## Schema (`BorderlineResult`)

| Field | Type | Description |
|-------|------|-------------|
| `classical_score` | `float` | Best classical score ($S_C$). |
| `quantum_score` | `float` | Best quantum score ($S_Q$). |
| `score_margin` | `float` | Absolute difference ($\Delta S$). |
| `borderline_score` | `float` | Similarity score ($B = 1 - \Delta S$). |
| `routing_recommendation` | `Literal[...]` | `classical_only`, `quantum_only`, `dual_execution`, `borderline_review`. |
| `winner_family` | `Literal[...]` | `classical`, `quantum`, `tie`. |
| `explanation` | `str` | Human-readable explanation. |

---

## Benchmark Routing Case Studies

| Problem Query | $S_C$ (Best Classical) | $S_Q$ (Best Quantum) | Margin ($\Delta S$) | Borderline ($B$) | Winner | Recommendation |
|---------------|------------------------|----------------------|--------------------|------------------|--------|----------------|
| **500-City Delivery Optimization** | 0.902 (GA) | 0.958 (QAOA) | 0.056 | 0.944 | Quantum | `borderline_review` |
| **20,000-Record Spam Classification** | 0.922 (XGBoost) | 0.827 (QSVM) | 0.095 | 0.905 | Classical | `borderline_review` |
| **1,000-Node Shortest Path Search** | 0.907 (A* Search) | 0.945 (Grover) | 0.038 | 0.962 | Quantum | `borderline_review` |

---

# Step 5.5 — LLM Agent Selector Arbitration

## Objective

The **LLM Agent Selector** provides explainable arbitration for borderline execution decisions. Rather than making unconstrained choices, the agent receives an immutable summary of all upstream deterministic evidence and produces an auditable, calibrated decision.

---

## Reasoning Architecture (3 Stages)

| Stage | Module | Responsibility |
|-------|--------|----------------|
| **1. Numerical Scoring** | Feature Engine & Compatibility Engine | Generate deterministic numerical scores. |
| **2. Borderline Detection** | Borderline Router | Compute margins and identify borderline cases mathematically. |
| **3. LLM Arbitration** | Context Builder & Agent Selector | Review evidence, assign calibrated confidence, and output explainable JSON decision. |

---

## Step 5.5.1 — Arbitration Context Builder

Consolidates all upstream Pydantic schemas into a single immutable context:

```text
ProblemSpecification + FeatureScore + CompatibilityResult + BorderlineResult
                                 │
                                 ▼
                     ArbitrationContextBuilder
                                 │
                                 ▼
                         ArbitrationContext
```

### Schema (`ArbitrationContext`)

| Field | Type | Description |
|-------|------|-------------|
| `user_query` | `str` | Original problem query. |
| `problem_class` | `str` | Problem class. |
| `confidence` | `float` | Problem Analyzer confidence. |
| `dimensionality` | `int` | Dimensionality of the problem. |
| `dataset_size` | `int` | Dataset size. |
| `constraint_structure` | `str` | Constraint category. |
| `dimension_score` | `float` | Deterministic dimension score. |
| `constraint_score` | `float` | Deterministic constraint score. |
| `complexity_score` | `float` | Deterministic complexity score. |
| `best_classical_algorithm` | `str` | Top classical algorithm name. |
| `best_classical_score` | `float` | Top classical score. |
| `best_quantum_algorithm` | `str` | Top quantum algorithm name. |
| `best_quantum_score` | `float` | Top quantum score. |
| `score_margin` | `float` | Mathematical score margin ($\Delta S$). |
| `borderline_score` | `float` | Borderline similarity score ($B$). |
| `routing_recommendation` | `str` | Recommendation from Borderline Router. |
| `summary` | `str` | Deterministic text summary of evidence. |

---

## Step 5.5.2 — LLM Agent Selector

Invokes Qwen2.5 with a strict decision prompt and deterministic constraints.

```text
ArbitrationContext ──► Qwen2.5 (JSON Mode) ──► Post-Processor & Calibrator ──► ExecutionDecision
```

### Decision Policy & System Rules
1. Treat deterministic scores as immutable ground truth.
2. Never hallucinate or invent new score values.
3. Preserve the deterministic `routing_mode` from the Borderline Router.
4. Set `selected_family` and `selected_algorithm` aligned with the highest compatibility score.
5. Calibrate decision confidence deterministically based on mathematical uncertainty.

---

## Deterministic Confidence Calibration

To prevent the LLM from outputting uncalibrated confidence (e.g. claiming 0.95 confidence when the borderline score is 0.944), the framework computes confidence mathematically:

$$C_{decision} = 1.0 - (B \times 0.40)$$

Subject to clamping boundaries:

$$0.55 \le C_{decision} \le 0.95$$

### Confidence Interpretation Scale

| Borderline Score ($B$) | Uncertainty Level | Calibrated Decision Confidence ($C_{decision}$) |
|-----------------------|-------------------|------------------------------------------------|
| $B \ge 0.90$ | Very High Uncertainty | $0.550 - 0.640$ |
| $0.70 \le B < 0.90$ | Moderate Uncertainty | $0.640 - 0.720$ |
| $0.50 \le B < 0.70$ | Low Uncertainty | $0.720 - 0.800$ |
| $B < 0.50$ | Very Clear Margin | $0.800 - 0.950$ |

---

## Reasoning Trace Mechanism

Every `ExecutionDecision` embeds a complete `reasoning_trace` dictionary containing all quantitative values for complete reproducibility and auditing.

### Schema (`ExecutionDecision`)

| Field | Type | Description |
|-------|------|-------------|
| `selected_family` | `Literal["classical", "quantum", "dual", "borderline"]` | Selected execution family. |
| `selected_algorithm` | `str` | Specific algorithm selected. |
| `routing_mode` | `Literal["classical_only", "quantum_only", "dual_execution", "borderline_review"]` | Framework routing mode. |
| `decision_confidence` | `float` | Calibrated confidence score ($0.55 \le c \le 0.95$). |
| `borderline_score` | `float` | Borderline similarity score ($B$). |
| `explanation` | `str` | Concise justification synthesized by the LLM. |
| `reasoning_trace` | `Dict[str, Any]` | Full audit trail of quantitative metrics. |

---

## End-to-End Optimization Example

**Query:** *"Optimize delivery routes for 500 cities under travel constraints."*

```json
{
    "selected_family": "quantum",
    "selected_algorithm": "Quantum Approximate Optimization Algorithm (QAOA)",
    "routing_mode": "borderline_review",
    "decision_confidence": 0.622,
    "borderline_score": 0.944,
    "explanation": "The borderline score is high (0.944), indicating competitive viability between classical heuristics and QAOA. Given the heavy combinatorial constraints and 500-variable space, quantum optimization holds a slight edge while retaining borderline review status.",
    "reasoning_trace": {
        "problem_confidence": 0.80,
        "dimension_score": 0.80,
        "constraint_score": 0.95,
        "complexity_score": 0.95,
        "classical_score": 0.902,
        "quantum_score": 0.958,
        "score_margin": 0.056
    }
}
```

---

# Complete Mathematical Formulations

| Metric | Formulation | Description |
|--------|-------------|-------------|
| **Classification Confidence** | $0.0 \le Confidence \le 1.0$ | Problem Analyzer output confidence constraint. |
| **Prior Score** | $0.0 \le PriorScore \le 1.0$ | Expert prior score in algorithm knowledge bases. |
| **Classical Affinity** | $Affinity_C = \max_{c \in C_{classical}} (c.prior\_score)$ | Peak classical prior for problem class. |
| **Quantum Affinity** | $Affinity_Q = \max_{q \in C_{quantum}} (q.prior\_score)$ | Peak quantum prior for problem class. |
| **Algorithm Compatibility** | $S_{compat} = 0.45 P + 0.25 C + 0.20 K + 0.10 D_{bonus}$ | Weighted multi-attribute compatibility score. |
| **Score Margin** | $\Delta S = \|S_C - S_Q\|$ | Difference between peak classical and quantum scores. |
| **Borderline Similarity** | $B = 1.0 - \Delta S$ | Mathematical degree of competition / uncertainty. |
| **Winner Determination** | $Winner = \arg\max(S_C, S_Q)$ | Mathematical maximum score selector. |
| **Calibrated Confidence** | $C_{decision} = \text{clamp}(1.0 - 0.40 B,\ 0.55,\ 0.95)$ | Deterministic uncertainty-to-confidence transformation. |

---

# Constants & Thresholds Registry

### 1. Compatibility Formula Weights ($\sum w_i = 1.00$)
- **Prior Score ($P$):** $w_1 = \mathbf{0.45}$
- **Constraint Fit ($C$):** $w_2 = \mathbf{0.25}$
- **Complexity Fit ($K$):** $w_3 = \mathbf{0.20}$
- **Dimension Bonus ($D_{bonus}$):** $w_4 = \mathbf{0.10}$

### 2. Dimension Bonus Multipliers
- **Exact Match:** $\mathbf{1.00}$
- **Adjacent Match:** $\mathbf{0.75}$
- **Distant Match:** $\mathbf{0.40}$

### 3. Dimension Categories
- **Small:** $D \le 50$
- **Medium:** $51 \le D \le 200$
- **Large:** $201 \le D \le 1000$
- **Very Large:** $D > 1000$

### 4. Borderline Router Thresholds
- **Clear Margin (`CLEAR_MARGIN`):** $\Delta S \ge \mathbf{0.25}$ (`classical_only` / `quantum_only`)
- **Dual Margin (`DUAL_MARGIN`):** $\mathbf{0.15} \le \Delta S < \mathbf{0.25}$ (`dual_execution`)
- **Review Margin (`REVIEW_MARGIN`):** $\Delta S < \mathbf{0.15}$ (`borderline_review`)

### 5. Confidence Calibration Constants
- **Scaling Factor:** $\mathbf{0.40}$
- **Minimum Clamped Confidence:** $\mathbf{0.55}$
- **Maximum Clamped Confidence:** $\mathbf{0.95}$

---

# Important Schemas Reference

| Schema Name | File Location | Purpose |
|-------------|---------------|---------|
| `ProblemSpecification` | `backend/models/problem_schema.py` | Validated natural language extraction contract. |
| `AlgorithmCandidate` | `backend/models/candidate_schema.py` | Individual classical or quantum candidate representation. |
| `CandidatePool` | `backend/models/candidate_schema.py` | Parallel candidate pools for a given problem. |
| `FeatureScore` | `backend/models/feature_score_schema.py` | Deterministic problem feature scores ($S_{dim}, S_{con}, S_{comp}$). |
| `CompatibilityResult` | `backend/models/compatibility_schema.py` | Ranked algorithm compatibility scores ($S_C, S_Q$). |
| `BorderlineResult` | `backend/models/borderline_schema.py` | Margin analysis and routing recommendations. |
| `ArbitrationContext` | `backend/models/arbitration_context_schema.py` | Consolidated immutable input for LLM arbitration. |
| `ExecutionDecision` | `backend/models/decision_schema.py` | Final arbitration decision with reasoning trace. |

---

# Information Flow Across Modules

| Stage | Input | Primary Processing | Output |
|-------|-------|--------------------|--------|
| **1. Problem Analyzer** | Natural Language Query | Qwen2.5 LLM with JSON mode & regex sanitization | `ProblemSpecification` |
| **2. Candidate Generator** | `ProblemSpecification` | Deterministic library lookup & sorting | `CandidatePool` |
| **3. Feature Scoring Engine** | `ProblemSpecification`, `CandidatePool` | Deterministic mathematical scoring lookup | `FeatureScore` |
| **4. Compatibility Engine** | `ProblemSpecification`, `CandidatePool` | Multi-attribute formula with dimension bonus | `CompatibilityResult` |
| **5. Borderline Router** | `CompatibilityResult` | Margin calculation & threshold comparison | `BorderlineResult` |
| **6. Context Builder** | All upstream schemas | Merges and summarizes evidence | `ArbitrationContext` |
| **7. LLM Agent Selector** | `ArbitrationContext` | Qwen2.5 arbitration + confidence calibration | `ExecutionDecision` |
| **8. Resource Estimator** *(Upcoming)* | `ExecutionDecision` | Predicts memory, execution time, and circuit depth | `ResourceEstimate` |
| **9. Hardware Mapper** *(Upcoming)* | `ResourceEstimate`, `ExecutionDecision` | Assigns CPU, GPU (CUDA), MPI, or QPU backend | `HardwareMapping` |

---

# Visualizations Suite (Steps V1 → V4)

To provide deep analytical transparency and publication-grade presentation, Q-HPC includes a dedicated analytical visualization suite in `backend/visualizations/`:

| Visualization Module | Artifact Files | Purpose & Features |
|----------------------|----------------|--------------------|
| **Step V1: Dynamic Boundary & Fluctuation Curve** | `v1_borderline_switching_curve.png`<br>`v1_borderline_switching_curve.html` | Visualizes fine-grained classical vs. quantum compatibility trajectories, NISQ decoherence oscillations, and algorithm switching pivots (e.g. $N=335$ vs $N=350$) in the borderline area ($B \ge 0.85$). |
| **Step V2: 3D Borderline Score Decision Surface** | `v2_borderline_decision_surface_3d.png`<br>`v2_borderline_decision_surface_3d.html` | Interactive 3D WebGL decision landscape with 2D bottom contour projections across Problem Dimensionality ($D$) and Constraint/Complexity Fit ($S_{con} \times S_{comp}$). |
| **Step V3: AI Agent Compatibility Heatmap** | `v3_agent_compatibility_heatmap.png`<br>`v3_agent_compatibility_heatmap.pdf`<br>`v3_agent_compatibility_heatmap.html` | Conference publication-grade (IEEE/Nature style) compatibility matrix spanning all 22 algorithms across 12 diverse computational archetypes. |
| **Step V4: AI Agent Impact Network** | `v4_agent_impact_network.png`<br>`v4_agent_impact_network.html` | Flagship multi-stage directed agentic impact network and interactive Sankey flow diagram representing end-to-end evidence propagation and decision flow. |

---

# Backend Folder Structure

```text
backend/
│
├── agents/
│   ├── problem_analyzer.py             # Step 2: LLM Problem Analyzer
│   ├── candidate_generator.py          # Step 3: Dual Candidate Generator
│   ├── feature_scoring_engine.py       # Step 4: Deterministic Feature Scorer
│   ├── compatibility_engine.py         # Step 5: Algorithm Compatibility Scorer
│   ├── arbitration_context_builder.py  # Step 5.5.1: Arbitration Context Builder
│   └── agent_selector.py               # Step 5.5.2: LLM Agent Selector
│
├── routers/
│   └── borderline_router.py            # Step 5.4: Mathematical Borderline Router
│
├── visualizations/                     # Visualizations Suite (Steps V1 → V4)
│   ├── __init__.py
│   ├── v1_borderline_switching_curve.py # Step V1: Dynamic Boundary & Fluctuation Curve
│   ├── v2_decision_surface_3d.py       # Step V2: 3D Decision Surface & Contours
│   ├── v3_compatibility_heatmap.py     # Step V3: Conference Publication Heatmap
│   ├── v4_agent_impact_network.py      # Step V4: Flagship AI Agent Impact Network
│   └── run_all_visualizations.py       # Master Pipeline Generator
│
├── models/
│   ├── problem_schema.py               # ProblemSpecification model
│   ├── candidate_schema.py             # AlgorithmCandidate & CandidatePool models
│   ├── feature_score_schema.py         # FeatureScore model
│   ├── compatibility_schema.py         # CompatibilityResult model
│   ├── borderline_schema.py            # BorderlineResult model
│   ├── arbitration_context_schema.py   # ArbitrationContext model
│   └── decision_schema.py              # ExecutionDecision model
│
├── utils/
│   ├── constants.py                    # Classical & Quantum algorithms registry
│   ├── classical_library.py            # Classical candidates retrieval
│   ├── quantum_library.py              # Quantum candidates retrieval
│   └── compatibility_library.py        # Algorithm compatibility priors & bounds
│
├── config/
│   └── settings.py                     # Environment variables & system configuration
│
├── tests/
│   ├── test_schema.py
│   ├── test_problem_analyzer.py
│   ├── test_problem_analyzer_llm.py
│   ├── test_candidate_schema.py
│   ├── test_classical_library.py
│   ├── test_quantum_library.py
│   ├── test_candidate_generator.py
│   ├── test_feature_scoring_engine.py
│   ├── test_compatibility_schema.py
│   ├── test_compatibility_library.py
│   ├── test_compatibility_engine.py
│   ├── test_borderline_schema.py
│   ├── test_borderline_router.py
│   ├── test_arbitration_context_builder.py
│   ├── test_decision_schema.py
│   └── test_agent_selector.py
│
├── outputs/
│   └── visualizations/                 # Generated PNG, PDF, and interactive HTML visual artifacts
└── main.py                             # Framework entry point
```

---

# Current Completion Status

| Module / Milestone | Stage in Workflow | Status |
|--------------------|-------------------|--------|
| **Backend Infrastructure** | Step 0 | ✅ Completed |
| **Problem Specification Schema** | Step 1 | ✅ Completed |
| **LLM Problem Analyzer** | Step 2 | ✅ Completed |
| **Dual Candidate Generation** | Step 3 | ✅ Completed |
| **Feature Scoring Engine** | Step 4 | ✅ Completed |
| **Compatibility Engine** | Step 5 | ✅ Completed |
| **Borderline Router** | Step 5.4 | ✅ Completed |
| **Arbitration Context Builder** | Step 5.5.1 | ✅ Completed |
| **LLM Agent Selector** | Step 5.5.2 | ✅ Completed |
| **Resource Estimator** | Step 6 | ⏳ Next Step |
| **HPC Hardware Mapper** | Step 7 | ⏳ Pending |
| **Runtime Execution & Telemetry Engine** | Step 8 | ⏳ Pending |
| **Critic Agent Feedback Loop** | Step 9 | ⏳ Pending |
| **Final Synthesis & Report Generator** | Step 10 | ⏳ Pending |

---

# Key Design Principles

1. **Hybrid Deterministic–LLM Architecture:** Mathematical scoring and empirical knowledge bases precede language-model reasoning.
2. **Parallel Candidate Generation:** Classical and quantum candidate pools are generated simultaneously; no early pruning.
3. **Algorithm-Level Explainability:** Every algorithm is evaluated against mathematical weights and empirical priors.
4. **Targeted LLM Escalation:** The LLM is invoked only when mathematical evidence shows genuine uncertainty ($\Delta S < 0.15$).
5. **Deterministic Confidence Calibration:** Decision confidence is strictly derived from mathematical uncertainty, preventing uncalibrated LLM confidence estimates.
6. **Immutable Reasoning Trace:** Every decision retains the full numerical evidence dictionary for verification, auditability, and scientific reproducibility.
7. **Modular Immutable Context:** A single unified Pydantic context isolates deterministic engine calculations from LLM prompt construction.