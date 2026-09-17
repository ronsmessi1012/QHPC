# Q-HPC Development Log (Steps 0–3)

**Project:** Q-HPC — Intelligent Quantum-Classical Execution Framework via AI Agents and Distributed HPC

**Objective:** Build the complete backend pipeline of Q-HPC exactly according to the proposed architecture, one validated module at a time. Every step is implemented, tested, validated, and documented before moving to the next stage.

---

# Overall Workflow Being Implemented

1. User Ingestion & Problem Feature Extraction
2. Dual Candidate Generation
3. Resource Estimation
4. Primary Algorithm Selection & Borderline Routing
5. LLM Agent Selector Arbitration
6. HPC Hardware Mapping
7. Runtime Execution & Telemetry
8. Critic Agent Feedback Loop
9. Final Report Synthesis

Current Progress:

- [x] Step 0 — Project Infrastructure
- [x] Step 1 — Problem Specification Schema
- [x] Step 2 — Deterministic Problem Analyzer
- [x] Step 3 — LLM-Powered Problem Analyzer (Qwen2.5 via Ollama)
- [ ] Step 4 — Dual Candidate Generator
- [ ] Remaining workflow.

---

# STEP 0 — Project Infrastructure

## Objective

Create a clean modular backend for Q-HPC that mirrors the architecture shown in the project synopsis.

---

## Folder Structure

```text
QHPC/
│
├── backend/
│   ├── agents/
│   ├── config/
│   ├── engine/
│   ├── models/
│   ├── outputs/
│   ├── routers/
│   ├── tests/
│   ├── utils/
│   ├── __init__.py
│   └── main.py
│
├── requirements.txt
├── .gitignore
├── .env
└── venv/
```

Every subdirectory also contains its own `__init__.py`.

---

## Folder Responsibilities

| Folder | Responsibility |
|--------|----------------|
| agents | LLM-powered reasoning agents. |
| models | Pydantic schemas only. No business logic. |
| routers | Decision routing and threshold logic. |
| engine | CUDA, MPI, telemetry and execution layer. |
| utils | Logger and helper utilities. |
| config | Settings and environment configuration. |
| tests | Unit tests for every module. |
| outputs | Generated JSON reports, telemetry, logs. |

---

## Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Initial Dependencies

Installed:

- pydantic
- rich
- python-dotenv
- ollama
- numpy
- pandas
- scikit-learn
- qiskit
- qiskit-aer
- mpi4py
- pytest
- PyYAML

---

## requirements.txt

Created a centralized dependency file for the project.

---

## .gitignore

Configured to ignore:

- venv
- logs
- generated outputs
- checkpoints
- cache
- Ollama model cache
- IDE files
- pytest cache

---

## Initial Entry Point

`backend/main.py`

```python
from rich import print

print("[bold cyan]Q-HPC Backend Initialized[/bold cyan]")
```

---

## Validation

Command:

```bash
python3.10 backend/main.py
```

Output:

```text
Q-HPC Backend Initialized
```

Step 0 Status: **Completed**

---

# Errors Encountered During Step 0

## Error 1 — Wrong Tree Command

Command executed:

```bash
tree backend
```

while already inside the backend directory.

Output:

```text
backend [error opening dir]
```

### Cause

Attempted to view a folder that did not exist relative to the current directory.

### Fix

Move to project root.

```bash
cd ..
tree backend
```

---

## Error 2 — Running From Wrong Working Directory

Several scripts were executed from `backend/tests`.

### Cause

Python resolves imports relative to the working directory.

### Fix

Adopt project-wide convention:

Always execute from the project root using:

```bash
python -m backend.tests.test_file
```

---

# STEP 1 — Problem Specification Schema

## Objective

Create the standardized JSON contract exchanged between every Q-HPC agent.

This schema represents the output of the Problem Analyzer and input for every downstream module.

---

## File Created

`backend/models/problem_schema.py`

---

## Pydantic Schema

Fields implemented:

| Field | Description |
|-------|-------------|
| user_query | Original natural language problem. |
| problem_class | classification / regression / optimization / simulation / search / unknown |
| dimensionality | Number of variables/features/cities/qubits etc. |
| dataset_size | Number of records/samples/cities/qubits. |
| linearity | linear / nonlinear / unknown |
| sparsity | dense / sparse / unknown |
| constraint_structure | constrained / unconstrained / mixed / unknown |
| quantum_candidate | Boolean indicating possible quantum applicability. |
| confidence | LLM confidence score between 0 and 1. |
| reasoning | One sentence describing computational characteristics only. |

---

## Why Pydantic?

Advantages:

- Type safety.
- Automatic validation.
- JSON serialization.
- Downstream contract enforcement.
- Detects malformed LLM outputs.

---

## Validation Test

Created:

`backend/tests/test_schema.py`

Constructed a sample optimization problem.

Output:

```json
{
  "problem_class":"optimization",
  "dataset_size":500,
  "quantum_candidate":true
}
```

Validation succeeded.

---

## Error Validation Test

Changed:

```python
problem_class="computer_vision"
```

Expected Output:

```text
ValidationError

Input should be:
classification
regression
optimization
simulation
search
unknown
```

---

## Lesson Learned

Literal validation prevents invalid problem categories from propagating into the execution pipeline.

Step 1 Status: **Completed**

---

# Errors Encountered During Step 1

## Error 1 — Invalid Literal Value

Problem:

```python
problem_class="computer_vision"
```

### Cause

Not included in allowed Literal types.

### Fix

Pydantic correctly rejected it.

No code changes required.

---

# STEP 2 — Deterministic Problem Analyzer

## Objective

Implement a rule-based analyzer before integrating an LLM.

Purpose:

- Easy debugging.
- Offline fallback.
- Verify pipeline independently of Ollama.

---

## File Created

`backend/agents/problem_analyzer.py`

---

## Logic Implemented

Input:

Natural language query.

Output:

`ProblemSpecification`

### Detection Rules

#### Problem Class

Keyword mapping:

| Keywords | Problem Class |
|----------|---------------|
| optimize, routing | optimization |
| classify, spam | classification |
| predict, regression | regression |
| simulate, molecule | simulation |
| search | search |

---

### Dataset Size Extraction

Extract first integer appearing inside the query.

Example:

```text
Optimize routes for 500 cities
```

Dataset size:

500

---

### Dimensionality Estimation

Initial heuristic:

```python
dimensionality = min(dataset_size,1000)
```

Later improved because dataset size and feature count are different concepts.

Final heuristic:

- Classification → 64
- Regression → 32
- Optimization → dataset_size
- Simulation → dataset_size
- Unknown → 16

---

### Other Metadata

- linearity
- sparsity
- constraint_structure
- quantum_candidate
- reasoning

---

## Unit Test

Created:

`backend/tests/test_problem_analyzer.py`

Tested four queries.

### Optimization

Output:

Optimization detected.

### Classification

Output:

Classification detected.

### Regression

Output:

Regression detected.

### Simulation

Output:

Simulation detected.

---

## Validation Results

All four cases produced valid ProblemSpecification objects.

Step 2 Status: **Completed**

---

# Errors Encountered During Step 2

## Error 1 — Import Errors

Problem:

Needed:

```python
import sys
from pathlib import Path

sys.path.insert(...)
```

### Cause

Running scripts directly from backend/tests.

### Fix

Converted backend into a package.

Created `__init__.py` inside:

- backend
- agents
- models
- routers
- engine
- utils
- config
- tests

---

## Error 2 — Absolute Imports

Changed imports everywhere.

Old:

```python
from models.problem_schema import ...
```

New:

```python
from backend.models.problem_schema import ...
```

---

## Standard Execution Convention

Run tests only from project root.

Correct:

```bash
python -m backend.tests.test_problem_analyzer
```

Never run:

```bash
python backend/tests/test_problem_analyzer.py
```

---

# STEP 3 — LLM Powered Problem Analyzer

## Objective

Replace heuristic analysis with a local LLM (Qwen2.5-7B) running through Ollama.

Pipeline:

Natural Language

↓

Qwen2.5-7B

↓

Strict JSON

↓

Pydantic Validation

↓

ProblemSpecification

---

## Prerequisite

Verified installed models.

Command:

```bash
ollama list
```

Output included:

```text
qwen2.5:7b-instruct
```

---

## Configuration Layer

Created:

`backend/config/settings.py`

Contents:

- OLLAMA_MODEL
- OLLAMA_HOST
- BORDERLINE_THRESHOLD

---

## Environment File

Created `.env`

```text
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_HOST=http://localhost:11434
```

---

## Settings Test

Created:

`backend/tests/test_settings.py`

Output:

```text
Model : qwen2.5:7b-instruct

Host : http://localhost:11434
```

---

## LLM Integration

`backend/agents/problem_analyzer.py`

Uses:

- ollama.chat()
- format="json"
- temperature=0
- strict JSON parsing
- ProblemSpecification validation

---

## System Prompt Design

Role:

Problem Analyzer Agent.

Responsibilities:

- Extract metadata only.
- Never recommend algorithms.
- Never recommend hardware.
- Never recommend execution plans.

Returns JSON matching the schema exactly.

---

## Why format="json"?

Without it:

Qwen may return Markdown.

With it:

Guaranteed JSON response.

---

## Added Deterministic Behaviour

Temperature set to zero.

Purpose:

Identical query should generate identical metadata.

Important for experiments and benchmarking.

---

## Added Missing Field Validation

After parsing JSON:

Compare returned keys with required schema keys.

Raise ValueError if anything is missing.

---

## Added Confidence Normalization

Force:

```python
confidence=float(confidence)
```

Ensures numeric confidence regardless of model formatting.

---

## Added Error Handling

Possible failures:

- Invalid JSON.
- Missing fields.
- Ollama unavailable.
- Pydantic validation.

Everything wrapped into RuntimeError.

---

## Initial LLM Output

Output:

```json
{
 "problem_class":"optimization",
 "quantum_candidate":true,
 "reasoning":"Quantum annealing..."
}
```

---

## Architectural Problem Found

Reasoning leaked algorithm recommendations.

Problem Analyzer should not discuss:

- QAOA
- QSVM
- VQE
- CUDA
- GPU
- MPI
- Quantum Annealing

Those belong to later agents.

---

## Prompt Refinement

Added explicit forbidden instructions.

Added good examples.

Added bad examples.

---

## Post Processing Safety Filter

Implemented forbidden-term cleanup inside reasoning.

Removed mentions of:

- qaoa
- quantum annealing
- cuda
- gpu
- mpi
- svm
- xgboost
- random forest

---

## Final Validated Output

```json
{
  "user_query":"Optimize delivery routes for 500 cities while minimizing fuel cost under travel constraints.",
  "problem_class":"optimization",
  "confidence":0.8,
  "dimensionality":500,
  "dataset_size":500,
  "linearity":"unknown",
  "sparsity":"unknown",
  "constraint_structure":"constrained",
  "quantum_candidate":true,
  "reasoning":"Detected a constrained combinatorial optimization problem involving 500 decision variables."
}
```

---

## Validation Checklist

- Strict JSON.
- Valid schema.
- Deterministic output.
- Confidence score.
- No algorithm recommendations.
- Ready for Candidate Generator.

Step 3 Status: **Completed**

---

# Errors Encountered During Step 3

## Error 1 — ModuleNotFoundError

Output:

```text
ModuleNotFoundError: No module named backend
```

### Cause

Running script inside backend directory.

### Fix

Always execute from project root.

```bash
python -m backend.tests.test_problem_analyzer_llm
```

---

## Error 2 — pytest Not Found

Output:

```text
pytest: command not found
```

### Cause

pytest missing inside virtual environment.

### Fix

```bash
pip install pytest
```

---

## Error 3 — Running Tests Inside tests Folder

Command:

```bash
python tests/test_problem_analyzer_llm.py
```

### Cause

Incorrect relative path.

### Fix

Execute from root.

---

## Error 4 — LLM Mentioned Quantum Annealing

Output reasoning:

```text
could benefit from quantum annealing
```

### Cause

Prompt not restrictive enough.

### Fix

- Stronger prompt.
- Explicit forbidden examples.
- Post-processing filter.

---

# Coding Standards Established

## Import Convention

Always use absolute imports.

Correct:

```python
from backend.models.problem_schema import ProblemSpecification
```

Never modify sys.path.

---

## Test Convention

Always execute from project root.

```bash
python -m backend.tests.test_file
```

---

## Agent Responsibility Rules

| Module | Responsibility |
|--------|----------------|
| Problem Analyzer | Metadata extraction only. |
| Candidate Generator | Classical & quantum candidate generation. |
| Resource Estimator | Resource prediction. |
| Borderline Router | Threshold computation. |
| Agent Selector | Arbitration for borderline cases. |
| Hardware Mapper | CPU/GPU/MPI mapping. |
| Performance Engine | Runtime telemetry. |
| Critic Agent | Feedback and retry decisions. |

Each module returns a Pydantic schema instead of raw dictionaries.

---

# Current Project Status

## Completed

- Project structure.
- Virtual environment.
- requirements.txt.
- .gitignore.
- Package initialization.
- ProblemSpecification schema.
- Deterministic Problem Analyzer.
- Ollama configuration.
- LLM Problem Analyzer.
- JSON validation.
- Confidence score.
- Robust error handling.
- Prompt refinement.
- Reasoning sanitization.

---

# Step 4 — Dual Candidate Generation

> **Module Status:** ✅ Completed & Validated
>
> **Workflow Stage:** Step 2 of the Q-HPC Architecture

The **Dual Candidate Generation** module is responsible for constructing **parallel Classical and Quantum algorithm pools** for every computational problem identified by the **Problem Analyzer**. This stage **does not select** an execution strategy; instead, it generates all feasible candidate algorithms that will later be evaluated by the **Resource Estimator**, **Borderline Router**, and **Agent Selector**.

---

## Workflow Position

```text
User Query
      │
      ▼
Problem Analyzer (Step 3)
      │
ProblemSpecification
      │
      ▼
Candidate Generator (Step 4)
      │
 ┌──────────────┬───────────────┐
 │ Classical     │ Quantum        │
 │ Candidate Pool│ Candidate Pool │
 └──────────────┴───────────────┘
      │
      ▼
CandidatePool
      │
      ▼
Resource Estimator (Step 5)
```

---

# Step 4.1 — Candidate Schema

## Objective

Create strongly typed Pydantic schemas representing every algorithm candidate.

### Files Added

```text
backend/models/candidate_schema.py
backend/tests/test_candidate_schema.py
```

---

## Schema Design

### `AlgorithmCandidate`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable algorithm name. |
| `family` | `Literal["classical","quantum"]` | Candidate family. |
| `algorithm_type` | `Literal[...]` | Supported problem category. |
| `prior_score` | `float` | Initial heuristic suitability score. |
| `complexity` | `str` | Classical asymptotic complexity or quantum circuit scaling. |
| `justification` | `str` | One-line explanation for candidate inclusion. |

### `CandidatePool`

| Field | Type | Description |
|-------|------|-------------|
| `classical_candidates` | `List[AlgorithmCandidate]` | Classical candidate list. |
| `quantum_candidates` | `List[AlgorithmCandidate]` | Quantum candidate list. |

---

## Why Separate Schemas?

- Strong Pydantic validation.
- Common interface for downstream agents.
- Easy JSON serialization.
- Eliminates malformed candidate objects.

---

## Validation Test

**Command**

```bash
python3.10 -m backend.tests.test_candidate_schema
```

**Result**

- One classical candidate created successfully.
- One quantum candidate created successfully.
- CandidatePool serialized into JSON.

**Status:** ✅ Passed

---

# Step 4.2 — Classical Algorithm Knowledge Base

## Objective

Build a deterministic registry of supported classical algorithms grouped by problem type.

### Files Added

```text
backend/utils/constants.py
backend/utils/classical_library.py
backend/tests/test_classical_library.py
```

---

## Classical Algorithm Registry

| Problem Class | Registered Algorithms |
|---------------|-----------------------|
| Classification | XGBoost, Random Forest, SVM, Logistic Regression |
| Regression | Linear Regression, Random Forest Regressor, XGBoost Regressor |
| Optimization | Branch and Bound, Genetic Algorithm, Simulated Annealing |
| Simulation | Monte Carlo Simulation, Finite Difference Solver |
| Search | A* Search, Dijkstra |

---

## Classical Prior Scores

| Algorithm | Prior Score | Reason |
|-----------|------------|--------|
| XGBoost | **0.95** | Strong tabular ML baseline. |
| Random Forest | **0.90** | Robust ensemble learning. |
| SVM | **0.86** | Effective for high-dimensional classification. |
| Logistic Regression | **0.80** | Simple interpretable baseline. |
| Branch and Bound | **0.90** | Exact combinatorial optimization solver. |
| Genetic Algorithm | **0.87** | Population-based optimization heuristic. |
| Simulated Annealing | **0.84** | Handles nonlinear constrained optimization. |
| Monte Carlo Simulation | **0.91** | Standard probabilistic simulation technique. |
| A* Search | **0.92** | Efficient informed graph search. |

---

## Candidate Ordering

Candidates are sorted in descending order using:

```python
candidates.sort(
    key=lambda candidate: candidate.prior_score,
    reverse=True
)
```

This guarantees that the strongest heuristic candidate appears first.

---

## Validation Test

**Command**

```bash
python3.10 -m backend.tests.test_classical_library
```

**Result**

- All five problem classes returned valid candidate lists.
- Candidate ordering followed descending `prior_score`.

**Status:** ✅ Passed

---

# Step 4.3 — Quantum Algorithm Knowledge Base

## Objective

Create the deterministic registry of supported quantum algorithms.

### Files Added

```text
backend/utils/quantum_library.py
backend/tests/test_quantum_library.py
```

---

## Quantum Algorithm Registry

| Problem Class | Registered Algorithms |
|---------------|-----------------------|
| Classification | QSVM, Quantum Kernel Classifier |
| Regression | Variational Quantum Regressor (VQR) |
| Optimization | QAOA, VQE |
| Simulation | Quantum Statevector Simulation, Hamiltonian Simulation |
| Search | Grover Search |

---

## Quantum Prior Scores

| Algorithm | Prior Score | Reason |
|-----------|------------|--------|
| QAOA | **0.94** | Designed for constrained combinatorial optimization. |
| QSVM | **0.89** | Quantum kernel classification algorithm. |
| Hamiltonian Simulation | **0.90** | Quantum molecular simulation. |
| Quantum Statevector Simulation | **0.95** | Exact quantum state evolution simulation. |
| Grover Search | **0.94** | Quadratic speedup for unstructured search. |
| VQE | **0.83** | Variational optimization through parameterized circuits. |
| VQR | **0.84** | Regression using variational quantum circuits. |

---

## Validation Test

**Command**

```bash
python3.10 -m backend.tests.test_quantum_library
```

**Result**

- All supported quantum algorithms loaded successfully.
- Candidate objects created correctly.
- Candidate ordering sorted by `prior_score`.

**Status:** ✅ Passed

---

# Step 4.4 — Candidate Generator Agent

## Objective

Generate parallel classical and quantum candidate pools from the output of the Problem Analyzer.

### Files Added

```text
backend/agents/candidate_generator.py
backend/tests/test_candidate_generator.py
```

---

## Candidate Generator Logic

```python
classical_candidates = get_classical_candidates(problem.problem_class)
quantum_candidates = get_quantum_candidates(problem.problem_class)

return CandidatePool(
    classical_candidates=classical_candidates,
    quantum_candidates=quantum_candidates
)
```

---

## Input

| Object | Produced By |
|--------|-------------|
| `ProblemSpecification` | Problem Analyzer |

## Output

| Object | Consumed By |
|--------|-------------|
| `CandidatePool` | Resource Estimator |

---

## Example CandidatePool (Optimization)

### Classical Candidates

| Algorithm | Prior Score |
|-----------|------------|
| Branch and Bound | **0.90** |
| Genetic Algorithm | **0.87** |
| Simulated Annealing | **0.84** |

### Quantum Candidates

| Algorithm | Prior Score |
|-----------|------------|
| QAOA | **0.94** |
| VQE | **0.83** |

---

## Integration Test Results

### Classification Problem

| Classical | Quantum |
|-----------|---------|
| XGBoost | QSVM |
| Random Forest | Quantum Kernel Classifier |
| SVM | — |
| Logistic Regression | — |

### Optimization Problem

| Classical | Quantum |
|-----------|---------|
| Branch and Bound | QAOA |
| Genetic Algorithm | VQE |
| Simulated Annealing | — |

### Simulation Problem

| Classical | Quantum |
|-----------|---------|
| Monte Carlo Simulation | Quantum Statevector Simulation |
| Finite Difference Solver | Hamiltonian Simulation |

### Search Problem

| Classical | Quantum |
|-----------|---------|
| A* Search | Grover Search |
| Dijkstra | — |

**Status:** ✅ Passed

---

# Errors Encountered During Step 4

## Error 1 — Ambiguous Score Name

### Initial Design

```python
suitability_score = 0.95
```

### Problem

Later stages introduce multiple scoring mechanisms.

### Fix

Renamed to:

```python
prior_score
```

### Reason

| Score | Generated By |
|-------|--------------|
| `confidence` | Problem Analyzer |
| `prior_score` | Candidate Generator |
| `resource_score` | Resource Estimator |
| `final_score` | Borderline Router |

This prevents ambiguity throughout the pipeline.

---

## Error 2 — Quantum Candidates Were Missing

### Initial Output

Classification and Search returned:

```text
Quantum Candidates
None
```

### Root Cause

The generator filtered candidates using:

```python
if problem.quantum_candidate:
    quantum_candidates = ...
else:
    quantum_candidates = []
```

### Why Incorrect?

Dual Candidate Generation must always construct both candidate pools.

Selection happens later.

### Fix

Removed the conditional filter.

```python
quantum_candidates = get_quantum_candidates(problem.problem_class)
```

### Result

| Problem | Quantum Candidates Generated |
|---------|------------------------------|
| Classification | QSVM, Quantum Kernel Classifier |
| Search | Grover Search |
| Optimization | QAOA, VQE |
| Simulation | Statevector Simulation, Hamiltonian Simulation |

---

## Error 3 — Candidate Ordering

### Issue

Algorithms appeared in insertion order instead of importance order.

### Fix

Sort by descending `prior_score`.

### Result

Highest-priority candidates appear first in every candidate pool.

---

## Error 4 — Simulation Metadata Inconsistency

### Initial Output

```json
{
  "dimensionality": 32,
  "dataset_size": 1
}
```

### Root Cause

The LLM interpreted "32 qubits" as a single simulation instance.

### Fix (Problem Analyzer)

Normalize qubit-based simulations:

```python
if problem_class == "simulation" and "qubit" in query.lower():
    dataset_size = dimensionality
```

### Expected Output

```json
{
  "dimensionality": 32,
  "dataset_size": 32
}
```

This normalization is required for quantum resource estimation.

---

# Constants Introduced in Step 4

| Constant | Value / Purpose |
|----------|------------------|
| `CLASSICAL_ALGORITHMS` | Registry of classical algorithms grouped by problem class. |
| `QUANTUM_ALGORITHMS` | Registry of quantum algorithms grouped by problem class. |
| `prior_score` | Initial heuristic suitability estimate. |
| `family` | Classical or Quantum algorithm identifier. |
| `algorithm_type` | Problem category identifier. |

---

# Mathematical Definitions Introduced

## Prior Score Range

$$
0 \leq prior\_score \leq 1
$$

Represents an expert prior before hardware-aware estimation.

---

## Candidate Ordering

$$
CandidatePool = Sort(prior\_score,\ descending)
$$

The highest prior score appears first.

---

## Candidate Generation Mapping

$$
ProblemSpecification
\;\xrightarrow{problem\_class}\;
(C_{classical}, C_{quantum})
$$

Where:

- $C_{classical}$ = Classical candidate set.
- $C_{quantum}$ = Quantum candidate set.

---

## CandidatePool Output Contract

$$
CandidatePool =
(C_{classical}, C_{quantum})
$$

Both candidate sets are forwarded unchanged into the Resource Estimator.

---

# Validation Checklist

| Validation | Status |
|------------|--------|
| Candidate Schema | ✅ Passed |
| Classical Knowledge Base | ✅ Passed |
| Quantum Knowledge Base | ✅ Passed |
| Candidate Generator Integration | ✅ Passed |
| Candidate Ordering | ✅ Passed |
| Classical + Quantum Pools Generated | ✅ Passed |

---

# Deliverables Created in Step 4

```text
backend/
│
├── agents/
│   └── candidate_generator.py          ✅
│
├── models/
│   └── candidate_schema.py             ✅
│
├── utils/
│   ├── constants.py                    ✅
│   ├── classical_library.py            ✅
│   └── quantum_library.py              ✅
│
└── tests/
    ├── test_candidate_schema.py        ✅
    ├── test_classical_library.py       ✅
    ├── test_quantum_library.py         ✅
    └── test_candidate_generator.py     ✅
```

---

## Step 4 Completion Status

- [x] 4.1 Candidate Schema
- [x] 4.2 Classical Algorithm Knowledge Base
- [x] 4.3 Quantum Algorithm Knowledge Base
- [x] 4.4 Candidate Generator Agent
- [x] Integration Testing Completed

**Output Artifact:** `CandidatePool` — the complete parallel classical and quantum candidate space that becomes the input to **Step 5: Resource Estimator**.