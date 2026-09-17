# Q-HPC: Intelligent Quantum-Classical Execution Framework via AI Agents and Distributed HPC

> **A hybrid deterministic–LLM agentic framework for multi-paradigm computational problem analysis, parallel classical/quantum candidate generation, explainable compatibility scoring, mathematical borderline arbitration, and high-performance computing hardware mapping.**

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
- [Repository Structure](#-repository-structure)
- [Core Pipeline Architecture (Steps 0 → 5.5)](#-core-pipeline-architecture-steps-0--55)
- [Mathematical Formulations](#-mathematical-formulations)
- [Comprehensive Visualizations Suite](#-comprehensive-visualizations-suite)
- [End-to-End Decision Trace Example](#-end-to-end-decision-trace-example)
- [Setup & Installation](#-setup--installation)
- [Running Tests & Visualizations](#-running-tests--visualizations)
- [Management Platform (NotionLocal)](#-management-platform-notionlocal)

---

## 🌌 Overview & Architecture

Modern scientific and computational workloads frequently span a spectrum where neither classical high-performance computing (CPU/GPU/MPI) nor quantum computing (QPU/variational circuits) is unilaterally superior across all problem scales. 

**Q-HPC** resolves this challenge through a **Hybrid Deterministic–LLM Architecture**:
1. **Deterministic Pre-Processing**: Rigorous mathematical scoring engines calculate problem difficulty, dimension scaling, and algorithm compatibility using empirical priors.
2. **Mathematical Borderline Detection**: The framework quantitatively detects whether one paradigm clearly dominates ($\Delta S \ge 0.25$), both are competitive ($\Delta S \ge 0.15$), or if the margin is narrow ($\Delta S < 0.15$).
3. **Prompt-Constrained LLM Arbitration**: A localized LLM agent (Qwen2.5-7B) is escalated **only** for genuine borderline review, enforcing grounded decisions with deterministically calibrated confidence and full auditability.

```text
User Natural Language Query
             │
             ▼
┌──────────────────────────────────────────────┐
│  Step 2: LLM Problem Analyzer (Qwen2.5)      │
└──────────────────────┬───────────────────────┘
                       │ ProblemSpecification Schema
                       ▼
┌──────────────────────────────────────────────┐
│  Step 3: Dual Candidate Generator            │
│  ├── Classical Candidate Pool (14 Algorithms)│
│  └── Quantum Candidate Pool (8 Algorithms)   │
└──────────────────────┬───────────────────────┘
                       │ CandidatePool Schema
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ Step 4: Feature Scorer    │ │ Step 5: Compatibility     │
│ (Dim, Con, Comp Scores)   │ │ Engine (Multi-Attribute)  │
└─────────────┬─────────────┘ └─────────────┬─────────────┘
              │ FeatureScore                │ CompatibilityResult
              └──────────────┬──────────────┘
                             ▼
┌──────────────────────────────────────────────┐
│  Step 5.4: Mathematical Borderline Router    │
│  (Margin ΔS, Borderline Similarity Score B)  │
└──────────────────────┬───────────────────────┘
                       │ BorderlineResult Schema
                       ▼
┌──────────────────────────────────────────────┐
│  Step 5.5.1: Arbitration Context Builder     │
└──────────────────────┬───────────────────────┘
                       │ ArbitrationContext Schema
                       ▼
┌──────────────────────────────────────────────┐
│  Step 5.5.2: LLM Agent Selector (Arbitration)│
│  (Confidence Calibration & Reasoning Trace)  │
└──────────────────────┬───────────────────────┘
                       │ ExecutionDecision Schema
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ Step 6: Resource Estimator│ │ Step 7: Hardware Mapper   │
│ (Time, Memory, Depth)     │ │ (CPU, CUDA, MPI, QPU)     │
└───────────────────────────┘ └───────────────────────────┘
```

---

## 📂 Repository Structure

```text
QHPC/
├── backend/
│   ├── agents/                     # AI Reasoning & Deterministic Scoring Engines
│   │   ├── problem_analyzer.py             # LLM metadata extraction (Qwen2.5)
│   │   ├── candidate_generator.py          # Parallel classical & quantum generator
│   │   ├── feature_scoring_engine.py       # Deterministic feature difficulty scorer
│   │   ├── compatibility_engine.py         # Multi-attribute algorithm compatibility
│   │   ├── arbitration_context_builder.py  # Unified immutable context synthesizer
│   │   └── agent_selector.py               # LLM arbitration & confidence calibrator
│   │
│   ├── routers/                    # Routing & Borderline Threshold Logic
│   │   └── borderline_router.py            # Mathematical margin & routing classifier
│   │
│   ├── models/                     # Strongly Typed Pydantic Contracts
│   │   ├── problem_schema.py               # ProblemSpecification schema
│   │   ├── candidate_schema.py             # AlgorithmCandidate & CandidatePool
│   │   ├── feature_score_schema.py         # FeatureScore schema
│   │   ├── compatibility_schema.py         # CompatibilityResult schema
│   │   ├── borderline_schema.py            # BorderlineResult schema
│   │   ├── arbitration_context_schema.py   # ArbitrationContext schema
│   │   └── decision_schema.py              # ExecutionDecision schema
│   │
│   ├── utils/                      # Algorithm Registries & Knowledge Bases
│   │   ├── constants.py                    # Classical & Quantum algorithms registry
│   │   ├── classical_library.py            # Classical algorithm retriever & sorter
│   │   ├── quantum_library.py              # Quantum algorithm retriever & sorter
│   │   └── compatibility_library.py        # Constraint & complexity priors
│   │
│   ├── visualizations/             # High-Impact Visualizations & Analytical Suite
│   │   ├── v1_borderline_switching_curve.py        # Data-point switching & fluctuation curve
│   │   ├── v2_decision_surface_3d.py               # Interactive 3D WebGL Decision Surface
│   │   ├── v3_compatibility_heatmap.py             # IEEE/Nature style matrix heatmap
│   │   ├── v4_agent_impact_network.py              # Flagship agent impact network & Sankey
│   │   ├── viz01_borderline_landscape.py           # Multi-problem log-scale size landscape
│   │   ├── viz02_algorithm_decision_boundary.py    # Continuous boundary transition graph
│   │   ├── viz03_problem_scaling_quantum_vs_classical.py # Asymptotic complexity curves
│   │   ├── viz04_ai_agent_routing_heatmap.py       # Matrix compatibility heatmap
│   │   ├── viz05_algorithm_transition_timeline.py  # Log-scale algorithm transition timeline
│   │   ├── viz_agent_selector_output.html          # Interactive Live Web GUI selector
│   │   └── run_all_visualizations.py               # Master CLI generator
│   │
│   ├── config/                     # System Settings & Environment Configuration
│   │   └── settings.py
│   │
│   ├── tests/                      # Full Unit & Integration Test Suite
│   ├── outputs/                    # Output traces, telemetry, and visualizations
│   │   └── visualizations/                 # Generated PNG, PDF, and interactive HTMLs
│   └── main.py                     # Backend initialization entry point
│
├── Management/                     # NotionLocal: Private Collaborative Workspace
│   ├── server/                     # Node.js backend & file-based storage
│   └── src/                        # React (Vite) block editor & Kanban portal
│
├── docs/                           # Technical Specifications & Evolution Logs
│   └── sampling.md                         # Complete mathematical & architectural evolution
│
├── requirements.txt                # Unified Python dependencies
└── README.md
```

---

## ⚙️ Core Pipeline Architecture (Steps 0 → 5.5)

### Step 1: Standardized Problem Specification
Converts unstructured queries into strict mathematical metadata:
- `dimensionality` ($\ge 1$), `dataset_size` ($\ge 1$), `linearity`, `sparsity`, `constraint_structure`, and `quantum_candidate`.

### Step 2: LLM Problem Analyzer
Runs local `qwen2.5:7b-instruct` under `temperature=0` and `format="json"`. Enforces negative prompt constraints (preventing leaked algorithm/hardware names) followed by regex-based sanitization and defensive integer fallbacks.

### Step 3: Dual Candidate Generation
Simultaneously populates:
- **Classical Pool (14 Algorithms)**: XGBoost, Random Forest, SVM, Logistic Regression, Linear Regression, Random Forest Regressor, XGBoost Regressor, Branch & Bound, Genetic Algorithm, Simulated Annealing, Monte Carlo Simulation, Finite Difference Solver, A* Search, Dijkstra.
- **Quantum Pool (8 Algorithms)**: QSVM, Quantum Kernel Classifier, Variational Quantum Regressor (VQR), QAOA, VQE, Quantum Statevector Simulation, Hamiltonian Simulation, Grover Search.

### Step 4: Deterministic Feature Scoring
Computes problem-level metrics:
- $S_{dim}$ (Dimension Difficulty) $\in [0.30, 0.95]$
- $S_{con}$ (Constraint Tightness) $\in [0.30, 0.95]$
- $S_{comp}$ (Computational Complexity) $\in [0.45, 0.95]$
- $Affinity_C = \max(P_C)$ and $Affinity_Q = \max(P_Q)$

### Step 5: Multi-Attribute Compatibility Engine
Computes algorithm-level compatibility scores ($S_{compat}$) using weighted empirical priors and dimension scale bonuses.

### Step 5.4: Mathematical Borderline Router
Calculates the competitive score margin $\Delta S = |S_C - S_Q|$ and borderline score $B = 1 - \Delta S$, classifying the execution state into `classical_only`, `quantum_only`, `dual_execution`, or `borderline_review`.

### Step 5.5: LLM Agent Selector Arbitration
Synthesizes all evidence into an immutable `ArbitrationContext`. For borderline cases ($B \ge 0.85$), Qwen2.5 performs arbitration with deterministically calibrated confidence and comprehensive reasoning traces.

---

## 📐 Mathematical Formulations

All decision logic in Q-HPC is governed by strict mathematical formulas:

### 1. Classification & Prior Score Bounds
$$0.0 \le \text{Confidence} \le 1.0, \quad 0.0 \le \text{PriorScore} \le 1.0$$

### 2. Paradigm Family Affinities
$$\text{Affinity}_C = \max_{c \in C_{\text{classical}}} (c.\text{prior\_score})$$
$$\text{Affinity}_Q = \max_{q \in C_{\text{quantum}}} (q.\text{prior\_score})$$

### 3. Dimension Category Matching & Bonus Multiplier
$$\text{Category}(D) = \begin{cases} \text{small}, & D \le 50 \\ \text{medium}, & 51 \le D \le 200 \\ \text{large}, & 201 \le D \le 1000 \\ \text{very\_large}, & D > 1000 \end{cases}$$

$$D_{\text{bonus}}(a) = \begin{cases} 1.00, & \text{if Preferred Category} = \text{Actual Category} \\ 0.75, & \text{if Preferred Category is Adjacent} \\ 0.40, & \text{otherwise} \end{cases}$$

### 4. Multi-Attribute Algorithm Compatibility Formulation
For any algorithm candidate $a$:
$$S_{\text{compat}}(a) = 0.45 \cdot P_a + 0.25 \cdot C_a + 0.20 \cdot K_a + 0.10 \cdot D_{\text{bonus}}(a)$$
- $P_a$ = Expert prior suitability score
- $C_a$ = Constraint handling capability fit
- $K_a$ = Computational complexity resilience
- $D_{\text{bonus}}(a)$ = Dimension scale bonus

### 5. Competitive Score Margin & Borderline Similarity Score
Let $S_C = \max_{a \in C} S_{\text{compat}}(a)$ and $S_Q = \max_{a \in Q} S_{\text{compat}}(a)$:
$$\Delta S = |S_C - S_Q|$$
$$B = 1.0 - \Delta S$$

### 6. Deterministic Routing Decision Boundaries
$$\text{RoutingMode} = \begin{cases} \text{classical\_only}, & \Delta S \ge 0.25 \text{ and } S_C > S_Q \\ \text{quantum\_only}, & \Delta S \ge 0.25 \text{ and } S_Q > S_C \\ \text{dual\_execution}, & 0.15 \le \Delta S < 0.25 \\ \text{borderline\_review}, & \Delta S < 0.15 \quad (\text{Escalate to LLM Agent Selector}) \end{cases}$$

### 7. Deterministic Confidence Calibration
To guarantee mathematical consistency between uncertainty and decision confidence:
$$C_{\text{decision}} = \text{clamp}\Big(1.0 - (B \times 0.40),\ 0.55,\ 0.95\Big)$$

| Borderline Score ($B$) | Uncertainty Level | Calibrated Confidence ($C_{\text{decision}}$) |
|:---:|:---:|:---:|
| $B \ge 0.90$ | Very High Uncertainty | $0.550 - 0.640$ |
| $0.70 \le B < 0.90$ | Moderate Uncertainty | $0.640 - 0.720$ |
| $0.50 \le B < 0.70$ | Low Uncertainty | $0.720 - 0.800$ |
| $B < 0.50$ | Decisive Winner | $0.800 - 0.950$ |

---

## 📊 Comprehensive Visualizations Suite

Q-HPC includes a publication-grade visualization suite in `backend/visualizations/`:

| Module / Script | Generated Artifacts | Description & Key Analytical Capabilities |
| :--- | :--- | :--- |
| **Step V1: Dynamic Switching Curve**<br>`v1_borderline_switching_curve.py` | `v1_borderline_switching_curve.png`<br>`v1_borderline_switching_curve.html` | Sweeps identical routing problem sizes ($N=100 \rightarrow 500$) showing point-by-point algorithm switching (e.g. Classical GA wins at $N=335$, Quantum QAOA wins at $N=350$) in the borderline zone ($B \ge 0.85$). |
| **Step V2: 3D Decision Surface**<br>`v2_decision_surface_3d.py` | `v2_borderline_decision_surface_3d.png`<br>`v2_borderline_decision_surface_3d.html` | Interactive 3D WebGL Decision Surface + 2D base contour projections mapping Dimensionality ($D$) and Constraint/Complexity Fit to Borderline Score ($B$). Supports interactive mouse rotation via `--gui`. |
| **Step V3: Compatibility Matrix**<br>`v3_compatibility_heatmap.py` | `v3_agent_compatibility_heatmap.png`<br>`v3_agent_compatibility_heatmap.pdf`<br>`v3_agent_compatibility_heatmap.html` | Conference publication-grade (IEEE/Nature style) clustered matrix evaluating all 22 algorithms across 12 diverse computational problem archetypes. |
| **Step V4: Agent Impact Network**<br>`v4_agent_impact_network.py` | `v4_agent_impact_network.png`<br>`v4_agent_impact_network.html` | Flagship multi-layer directed agentic impact network and interactive Sankey flow diagram representing end-to-end evidence propagation and decision flow. |
| **Problem Size Landscape**<br>`viz01_borderline_landscape.py` | Interactive Plotly GUI | Multi-problem logarithmic scale visualization charting Classical, Borderline, and Quantum transition thresholds across 7 problem classes. |
| **Decision Boundary Step Graph**<br>`viz02_algorithm_decision_boundary.py` | Interactive Plotly GUI | Step-function boundary chart mapping dataset scaling directly to execution backend modes with highlighted arbitration bands. |
| **Asymptotic Scaling Curves**<br>`viz03_problem_scaling_quantum_vs_classical.py` | Interactive Plotly GUI | Asymptotic complexity curves comparing classical $O(n \log n)$ scaling vs quantum $O(\sqrt{n})$ speedups against threshold limits. |
| **Archetype Routing Heatmap**<br>`viz04_ai_agent_routing_heatmap.py` | Interactive Plotly GUI | Direct side-by-side paradigm score matrix across real-world problem benchmarks. |
| **Algorithm Transition Timeline**<br>`viz05_algorithm_transition_timeline.py` | Interactive Plotly GUI | Logarithmic transition timeline displaying Classical $\rightarrow$ Borderline $\rightarrow$ Quantum crossover points. |
| **Live Web GUI Selector Dashboard**<br>`viz_agent_selector_output.html` | Standalone Web App | Responsive HTML5/Tailwind dashboard featuring real-time problem size sliders, dynamic compatibility scoring, and live execution decision card rendering. |

---

## 🔬 End-to-End Decision Trace Example

**User Query**: *"Optimize delivery routes for 500 cities under travel constraints."*

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

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) with `qwen2.5:7b-instruct` installed
- Node.js 18+ (for NotionLocal management portal)

### 1. Clone & Environment Setup
```bash
git clone https://github.com/your-username/QHPC.git
cd QHPC

# Create & activate virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create `.env` in the root directory:
```env
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_HOST=http://localhost:11434
BORDERLINE_THRESHOLD=0.15
```

### 3. Ensure Local LLM is Running
```bash
ollama pull qwen2.5:7b-instruct
ollama serve
```

---

## 🧪 Running Tests & Visualizations

### Run Full Test Suite
```bash
source venv/bin/activate

# Run all unit and integration tests
pytest backend/tests/

# Or run individual test modules
python -m backend.tests.test_problem_analyzer_llm
python -m backend.tests.test_candidate_generator
python -m backend.tests.test_compatibility_engine
python -m backend.tests.test_borderline_router
python -m backend.tests.test_agent_selector
```

### Run Visualizations Suite
```bash
# Generate all static and interactive visualizations (Steps V1 → V4)
python -m backend.visualizations.run_all_visualizations

# Run interactive 3D Decision Surface with mouse rotation
python -m backend.visualizations.v2_decision_surface_3d --gui

# Open the live interactive Web GUI selector dashboard
open backend/visualizations/viz_agent_selector_output.html
```

---

## 💼 Management Platform (NotionLocal)

The `Management/` directory contains **NotionLocal**, an isolated private workspace for team task tracking and documentation.

```bash
cd Management
npm install
npm run dev
```
- **Client URL**: `http://localhost:5173`
- **Server API**: `http://localhost:5001`
