# QHPC: Quantum-HPC Agent Workspace

Welcome to the **QHPC** repository. This workspace is organized into two primary sub-projects: a research and execution engine (**Agent**), and a private collaborative project management portal (**Management**).

---

## 📂 Repository Overview

```
QHPC/
├── Agent/                      # Q-HPC-Agent Research & Pipeline
│   ├── agents/                 # LLM Agents (problem_analyzer, algorithm_selector, critic)
│   ├── core/                   # Schemas, types, and configurations
│   ├── llm/                    # Local/Cloud LLM and vLLM clients
│   ├── hpc/                    # HPC compilation and execution modules
│   ├── algorithms/             # Classical and Quantum algorithm solver pools
│   │   ├── classical/
│   │   └── quantum/
│   ├── experiments/            # Benchmark runs and experimental logs
│   ├── tests/                  # Unit and integration test suites
│   ├── scripts/                # Execution and helper scripts
│   ├── docs/                   # Architecture blueprints and references
│   ├── .env.example
│   ├── .gitignore
│   ├── requirements.txt
│   └── main.py
│
└── Management/                 # NotionLocal: Collaborative Project Board
    ├── server/                 # Node.js backend & local file-based database
    └── src/                    # React (Vite) frontend with Block Editor & Kanban
```

| Component | Technology Stack | Role in Repository |
| :--- | :--- | :--- |
| [**Agent**](file:///Users/ronnie/Documents/QHPC/Agent) | Python, Gemini/OpenAI, Scikit-Learn, Qiskit | The core scientific workflow. Selects between classical and quantum algorithms, estimates resources, executes on HPC backends, and refines execution specs via closed-loop feedback. |
| [**Management**](file:///Users/ronnie/Documents/QHPC/Management) | React (Vite), Express, Node.js, Vanilla CSS | The team collaboration space. A local Notion clone (NotionLocal) featuring section isolation, a block-based editor, and interactive Kanban boards. |

---

## 🔬 1. Q-HPC Agent (`Agent/`)

The **Q-HPC Agent** is an agentic framework designed to solve computational tasks by intelligently choosing between classical and quantum solvers and executing them efficiently on CPU, GPU, or MPI-distributed resources.

### Key Components governed by the LLM & Decision Backend:
*   **LLM Problem Analyzer**: Parses user input and datasets to classify the compute problem.
*   **Algorithm Selection Agent**: Synthesizes classical (SVM, Random Forest, XGBoost) and quantum (QAOA, VQE, QSVM) candidate pipelines.
*   **Agent Critic & Decision Logic**: Evaluates performance telemetry (accuracy, runtime, memory, MPI overhead) and loops back to optimize the circuit parameters or selects classical execution if quantum provides no advantage.

### Quick Setup:
For configuration guidelines, dependencies, and environment variable templates, refer to the [**Agent Backend README**](file:///Users/ronnie/Documents/QHPC/Agent/backend/README.md).

---

## 💼 2. NotionLocal Collaboration Platform (`Management/`)

**NotionLocal** is a self-hosted, local-first platform built so you and your project mates can collaborate on task management and documentation securely.

### Core Features:
*   **Section-Level Access Control**: Admins can register team members and assign project access. Teammates can only view projects they are assigned to.
*   **Notion-like Block Editor**: Custom page editor with `/` slash commands supporting Headings, Bullets, Checklists, Code, and Callouts.
*   **Interactive Kanban Board**: Visual task lists supporting card prioritization, due dates, assignees, and native HTML5 drag-and-drop movement.

### Running NotionLocal:
Navigate to the directory and start the concurrent servers:
```bash
cd Management
npm run dev
```
*   **Client URL**: `http://localhost:5173` (also shared with mates on the local network).
*   **API URL**: `http://localhost:5001` (avoids macOS Control Center port conflicts).
