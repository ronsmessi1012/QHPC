# README.md — additions for Steps 6, 7 and 8

Targeted edits rather than a rewrite, so nothing already in the README is
lost. Four changes, in the order they appear in the file.

---

## EDIT 1 — the repository description (GitHub "About" + opening blurb)

**Current text:**

> Q-HPC Agent is an agentic AI framework that intelligently selects between
> classical and quantum algorithms and dynamically orchestrates their
> execution across CPU, CUDA-enabled GPUs, and distributed OpenMPI + CUDA
> environments using performance-aware feedback.

**Problem:** "distributed OpenMPI + CUDA environments" claims multi-node
distributed execution. Nothing in the project has been tested across more
than one machine, and no cluster was available. Single-host MPI and single
-GPU CUDA are both now genuinely measured — the overclaim is specifically
the word *distributed*.

**Replace with:**

> Q-HPC Agent is an agentic AI framework that selects between classical and
> quantum algorithms through deterministic scoring and constrained LLM
> arbitration, predicts the resource cost of the chosen algorithm, maps it
> to the execution backend the host actually provides, and runs it —
> recording measured cost against predicted cost. Execution is validated on
> CPU, single-host OpenMPI, CUDA GPU and quantum simulators; multi-node
> distributed execution is future work.

---

## EDIT 2 — extend the pipeline diagram

In the `Overview & Architecture` code block, **replace** the final two
boxes:

```text
┌───────────────────────────┐ ┌───────────────────────────┐
│ Step 6: Resource Estimator│ │ Step 7: Hardware Mapper   │
│ (Time, Memory, Depth)     │ │ (CPU, CUDA, MPI, QPU)     │
└───────────────────────────┘ └───────────────────────────┘
```

**with:**

```text
                       │ ExecutionDecision Schema
                       ▼
┌──────────────────────────────────────────────┐
│  Step 6: Resource Estimator                  │
│  (memory, runtime, qubits, circuit depth)    │
└──────────────────────┬───────────────────────┘
                       │ ResourceEstimate Schema
                       ▼
┌──────────────────────────────────────────────┐      ┌──────────────────┐
│  Step 7: Hardware Mapper                     │◄─────┤ hardware_detect  │
│  (CPU / CUDA / MPI / simulator)              │      │ probes the host  │
└──────────────────────┬───────────────────────┘      └──────────────────┘
                       │ HardwareMapping Schema
                       ▼
┌──────────────────────────────────────────────┐
│  Step 8: Runtime Execution & Telemetry       │
│  (executes, measures, compares to prediction)│
└──────────────────────┬───────────────────────┘
                       │ ExecutionTelemetry Schema
                       ▼
            predicted vs measured  ──►  Step 9 (Critic Agent)
```

---

## EDIT 3 — repository structure

In the `Repository Structure` block, **add** to `backend/agents/`:

```text
│   │   └── resource_estimator.py          # Step 6: deterministic cost model
```

**Add** a new `engine/` entry after `routers/`:

```text
│   ├── engine/                     # Execution backends & runtime
│   │   ├── hardware_mapper.py              # Step 7: backend assignment
│   │   ├── performance_engine.py           # Step 8: dispatch & telemetry
│   │   ├── workloads.py                    # Step 8: executable algorithms
│   │   └── mpi_worker.py                   # Step 8: MPI rank entry point
```

**Add** to `backend/models/`:

```text
│   │   ├── resource_schema.py              # ResourceEstimate schema
│   │   ├── hardware_schema.py              # HardwareMapping schema
│   │   └── telemetry_schema.py             # ExecutionTelemetry schema
```

**Add** to `backend/utils/`:

```text
│   │   ├── hardware_detect.py              # CPU/CUDA/MPI probing
│   │   └── resource_constants.py           # Steps 6-7 constants registry
```

**Add** to `docs/`:

```text
│   └── steps_6_7_8.md                      # Steps 6-8 architecture & results
```

---

## EDIT 4 — new sections after "Core Pipeline Architecture (Steps 0 → 5.5)"

Insert the following. Full detail lives in `docs/steps_6_7_8.md`; this is
the summary a README should carry.

---

### ⚙️ Execution Pipeline (Steps 6 → 8)

#### Step 6: Resource Estimator

Predicts the cost of the selected algorithm before any hardware is
committed. Deterministic — table lookups and closed-form formulas, no LLM.

Classical algorithms scale from a per-algorithm reference point:

$$\text{cost} = \text{base} \times \left(\frac{\text{size driver}}{\text{ref size}}\right)^{e}$$

Each algorithm declares **which** quantity drives its cost: supervised
learning scales with samples, combinatorial optimization with decision
variables. Exponents approximate the documented complexity class.

Quantum algorithms are modelled by **simulation strategy**, which matters
more than the algorithm's name:

| Strategy | Memory | Used for |
|----------|--------|----------|
| statevector | $2^n \times 16$ bytes | Exact state evolution, small index registers |
| MPS | $n \times 2\chi^2 \times 16$ bytes | Shallow variational circuits (QAOA, VQE, kernels) |

A 500-variable QAOA instance needs $4.9 \times 10^{142}$ GB under the
statevector model and **0.061 GB** under MPS — which is how it would
actually be simulated.

#### Step 7: Hardware Mapper

Assigns a backend the deployment actually has. `hardware_detect.py` probes
for CUDA (`nvidia-smi`), MPI (`mpiexec` + `mpi4py`) and CPU/RAM
(`psutil`) first, so the mapper never recommends hardware that is not
present. A GPU-capable algorithm on a machine with no GPU maps to CPU, and
the justification says why.

Two planning modes: `detected` assigns only verified hardware; `target`
plans against a declared deployment and marks every result
`is_hypothetical`, which Step 8 then refuses to execute.

#### Step 8: Runtime Execution & Telemetry

Runs the workload and measures what it cost. Each run returns a
**correctness metric** alongside its timing, because timing alone proves
only that something consumed CPU.

Five algorithms have executable implementations, chosen so that every
backend is genuinely exercised: XGBoost (CPU/CUDA), Monte Carlo and
Genetic Algorithm (CPU/MPI), Grover and QAOA (simulator). An algorithm in
the registry without an implementation is reported as unavailable, never
as a successful run of nothing.

---

### 📊 Measured Results

Every figure names its host. Runtimes do not transfer between machines.

#### MPI strong scaling — 16 logical / 8 physical cores, OpenMPI 5

| Workload | 2 ranks | 4 ranks | 8 ranks | efficiency @8 |
|----------|--------:|--------:|--------:|--------------:|
| Monte Carlo Simulation | 1.06× | 1.99× | **3.57×** | 44.6% |
| Genetic Algorithm | 0.98× | 1.75× | **2.47×** | 30.9% |

The gap between them is the result. Monte Carlo communicates once at the
end; the GA performs an `allreduce` every generation and is
communication-bound. Speedup past 8 ranks is limited by hyperthreading.

#### CUDA — NVIDIA Tesla T4

XGBoost, `device="cpu"` vs `device="cuda"`, accuracy identical on both:

| n | CPU | GPU | speedup |
|--:|----:|----:|--------:|
| 10,000 | 2.61 s | 0.90 s | 2.90× |
| 100,000 | 6.92 s | 1.67 s | 4.14× |
| 500,000 | 22.17 s | 5.00 s | **4.43×** |

Statevector simulation, CPU vs GPU:

| qubits | CPU | GPU | speedup |
|-------:|----:|----:|--------:|
| 18 | 2.88 s | 0.56 s | 5.16× |
| 20 | 11.50 s | 0.86 s | 13.39× |
| 22 | 50.97 s | 1.93 s | **26.40×** |

#### Predicted vs measured

`ExecutionTelemetry` records actual ÷ predicted for every run. XGBoost on
CPU landed at **0.985** — within 1.5% — from constants reasoned purely
from complexity class, with no calibration.

Most Step 6 constants remain uncalibrated and are flagged as such in every
estimate, so a large error ratio is never mistaken for a mis-prediction by
a measured model.

---

### 🚫 What This Framework Has Not Demonstrated

| Capability | Status |
|------------|--------|
| CPU execution | Measured |
| Single-host MPI | Measured — 8 physical cores |
| CUDA | Measured — Tesla T4 |
| Quantum simulation | Measured — statevector and MPS |
| **Multi-node distributed MPI** | **Untested** — no cluster available |
| **Physical QPU** | **Out of scope** — simulator only |

---

## EDIT 5 — setup additions

In `Setup & Installation`, under Prerequisites, **add**:

```text
- An MPI runtime for Step 8 distributed execution:
  - Linux/WSL: `sudo apt install openmpi-bin libopenmpi-dev`
  - Windows: Microsoft MPI
- Optional: `xgboost` for the CPU/CUDA workload
- Optional: `qiskit-aer-gpu` instead of `qiskit-aer` for GPU simulation
```

And under `Running Tests`, **add**:

```bash
# Steps 6-8 and the schema contracts (no LLM required)
pytest backend/tests/test_resource_estimator.py \
       backend/tests/test_hardware_mapper.py \
       backend/tests/test_performance_engine.py \
       backend/tests/test_schema.py \
       backend/tests/test_candidate_schema.py

# What hardware does this host provide?
python -m backend.utils.hardware_detect

# MPI strong-scaling benchmark (~30 minutes)
python -m backend.tests.benchmark_mpi_scaling --repeats 3 --json mpi_scaling.json

# Step 6 calibration harness
python -m backend.tests.benchmark_resources --repeats 3 --json calibration.json
```

---

## Two things worth raising with Person 1

**1. `backend/models/` is excluded by `.gitignore`.** The bare `models/`
pattern under "Ollama / Model Cache" matches a directory of that name at
any depth, so it silently excluded `backend/models/` along with the model
cache it was meant for. Fix:

```gitignore
/models/        # leading slash = repository root only
```

then `git add -f backend/models/`.

**2. Eight test files call `analyzer.analyze(query)` at module level**, so
a live Ollama instance is required merely to *collect* them. One
unreachable service aborts the entire suite before any unrelated test
runs. Moving those calls inside test functions would let the suite report
eight failures instead of zero tests.

Also: `v4_agent_impact_network.py` is described in the README as the
flagship visualization but is not in the repository, and `plotly` and
`matplotlib` are absent from `requirements.txt`, so the visualization
suite cannot run as shipped.
