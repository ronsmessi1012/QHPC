# Q-HPC Agent Backend: LLM Agents & Decision Logic

This module implements the **LLM Agents & Decision Logic** subsystem of the **Q-HPC Agent** platform (Classical–Quantum Algorithm Selection + Resource-Aware HPC Execution). 

Its primary responsibility is to analyze incoming computational problems, generate candidate classical/quantum pipelines, and execute closed-loop decision critiquing based on hardware execution feedback.

---

## 🏗️ Subsystem Architecture

Following the blueprint in [`Q_HPC_Agent_Literature_Grounded_Blueprint.docx`](file:///Users/ronnie/Documents/QHPC/Agent/docs/Q_HPC_Agent_Literature_Grounded_Blueprint.docx), this backend governs three core agentic layers:

```
                  +-----------------------------------------+
                  |               User Input                |
                  |           (Problem + Dataset)           |
                  +--------------------+--------------------+
                                       |
                                       v
                  +--------------------+--------------------+
                  |       Layer 1: LLM Problem Analyzer      |
                  |   - Characterizes problem type & size   |
                  +--------------------+--------------------+
                                       |
                                       v
                  +--------------------+--------------------+
                  |     Layer 2: Algorithm Selection Agent   |
                  |   - Classical: SVM / RF / XGBoost       |
                  |   - Quantum: QAOA / VQE / QSVM          |
                  +--------------------+--------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v (To Resource Estimator)               v (From Performance Engine)
            [Candidate Spec]                       [Execution Metrics]
                   |                                       |
                   |                                       v
                   |                      +----------------+----------------+
                   |                      |    Layer 3: Agent Critic /      |
                   |                      |         Decision Logic          |
                   |                      |  - Compares classical vs quantum|
                   |                      |  - Loops back to improve spec   |
                   |                      |  - Outputs Final Report         |
                   |                      +----------------+----------------+
                   |                                       |
                   v                                       v
           (Out to Execution)                       (To User Interface)
```

### 1. LLM Problem Analyzer
- Parses user natural language queries and raw dataset metadata.
- Identifies the problem domain (e.g., Combinatorial Optimization, Supervised Classification, Quantum Chemistry).
- Extracts critical constraints (dataset size, feature dimensions, target precision).

### 2. Algorithm Selection Agent
- **Classical Pool**: Generates hyperparameters and configurations for Support Vector Machines (SVM), Random Forests (RF), and XGBoost.
- **Quantum Pool**: Configures Quantum Approximate Optimization Algorithm (QAOA), Variational Quantum Eigensolver (VQE), or Quantum Support Vector Machines (QSVM).
- Synthesizes execution specifications to pass down to the **Resource Estimator** and **HPC Execution Agent**.

### 3. Agent Critic & Decision Logic
- Reviews performance telemetry (accuracy, execution runtime, memory footprint, GPU utilization, MPI scaling overhead) from the simulator or hardware run.
- Runs a critiquing logic to determine if circuit optimization, batch adjustments, or execution restructuring is required.
- Handles the high-level logic to **reject quantum execution** if the classical alternative offers superior performance/efficiency.

---

## 📂 Recommended Directory Structure

```directory
backend/
├── agents/
│   ├── __init__.py
│   ├── analyzer.py       # LLM Problem Analyzer Agent
│   ├── selector.py       # Algorithm Selection Agent (Classical & Quantum specs)
│   └── critic.py         # Decision critic & performance feedback loop logic
├── models/
│   ├── __init__.py
│   ├── schemas.py        # Pydantic models for structured agent outputs & metrics
│   └── classical.py      # Baseline estimators (scikit-learn / XGBoost training)
├── utils/
│   ├── __init__.py
│   ├── llm_client.py     # Unified wrapper for OpenAI / Gemini API calls
│   └── prompts.py        # System and agent prompts
├── tests/
│   ├── test_analyzer.py
│   ├── test_selector.py
│   └── test_critic.py
├── .env.example          # Environment variables template
├── .gitignore
├── requirements.txt
├── README.md
└── main.py               # Core orchestrator pipeline entry point
```

---

## ⚡ Getting Started (Local Setup)

### 1. Set Up Virtual Environment
Initialize a clean Python virtual environment inside the backend directory:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API Credentials
Create a `.env` file in the root of the `backend/` directory:
```bash
cp .env.example .env
```
Populate the file with your keys:
```env
# Gemini API Key (Recommended)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (Optional)
OPENAI_API_KEY=your_openai_api_key_here

# Pipeline Settings
DEBUG_MODE=true
```
