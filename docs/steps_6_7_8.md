# Q-HPC Backend Architecture Evolution (Step 6 → Step 8)

> **Project:** Q-HPC — Quantum-Classical High Performance Computing Execution Framework
>
> **Module Documentation:** Resource Estimation, Hardware Mapping & Runtime Execution
>
> **Version:** v0.80 (Completed through Step 8: Runtime Execution & Telemetry)
>
> *Continues `docs/sampling.md`, which covers Steps 0 → 5.5.*

---

## Table of Contents

- [Architecture at Step 8](#architecture-at-step-8)
- [Step 6 — Resource Estimator](#step-6--resource-estimator)
  - [Classical Cost Model](#classical-cost-model)
  - [Quantum Cost Model](#quantum-cost-model)
  - [Why Simulation Strategy Decides the Memory Model](#why-simulation-strategy-decides-the-memory-model)
  - [Parallel Speedup Model](#parallel-speedup-model)
  - [Resource Tiers](#resource-tiers)
- [Step 7 — Hardware Mapper](#step-7--hardware-mapper)
  - [Hardware Inventory Detection](#hardware-inventory-detection)
  - [Planning Modes](#planning-modes)
  - [Backend Assignment Rules](#backend-assignment-rules)
- [Step 8 — Runtime Execution & Telemetry](#step-8--runtime-execution--telemetry)
  - [Executable Workloads](#executable-workloads)
  - [Backend Dispatch](#backend-dispatch)
  - [Correctness Metrics](#correctness-metrics)
  - [Honesty Constraints](#honesty-constraints)
- [Measured Results](#measured-results)
  - [MPI Strong Scaling](#mpi-strong-scaling)
  - [CUDA Acceleration](#cuda-acceleration)
  - [Predicted vs Measured](#predicted-vs-measured)
- [Calibration Status](#calibration-status)
- [Constants & Thresholds Registry (Steps 6–8)](#constants--thresholds-registry-steps-68)
- [Schemas Reference (Steps 6–8)](#schemas-reference-steps-68)
- [Limitations](#limitations)
- [Completion Status](#completion-status)

---

## Architecture at Step 8

Steps 2 → 5.5 produce a *decision*. Steps 6 → 8 turn that decision into a
*measured execution*.

```text
            ExecutionDecision  (Step 5.5)
                   │
                   ▼
      ┌────────────────────────────┐
      │ Step 6: Resource Estimator │  predicts memory, runtime,
      │                            │  qubits, circuit depth
      └─────────────┬──────────────┘
                    │ ResourceEstimate
                    ▼
      ┌────────────────────────────┐      ┌─────────────────────┐
      │ Step 7: Hardware Mapper     │◄────┤ hardware_detect.py  │
      │ assigns a backend           │      │ probes the host     │
      └─────────────┬──────────────┘      └─────────────────────┘
                    │ HardwareMapping
                    ▼
      ┌────────────────────────────┐
      │ Step 8: Performance Engine │  RUNS it, measures cost
      └─────────────┬──────────────┘
                    │ ExecutionTelemetry
                    ▼
         predicted vs measured  ──►  Step 9 (Critic Agent)
                                ──►  Step 6 recalibration
```

The distinction between Steps 7 and 8 matters: **Step 7 decides where work
should run; Step 8 actually runs it.** Everything before Step 8 is
planning, and nothing before Step 8 produces a measurement.

---

## Step 6 — Resource Estimator

### Objective

Predict what executing the selected algorithm will cost, before any
hardware is committed.

```text
ExecutionDecision + ProblemSpecification ──► Resource Estimator ──► ResourceEstimate
```

Entirely deterministic — no LLM call. Table lookups plus closed-form
formulas, in the manner of Step 4's Feature Scoring Engine.

### Schema (`ResourceEstimate`)

| Field | Type | Description |
|-------|------|-------------|
| `estimated_memory_gb` | `float` | Predicted peak memory |
| `estimated_runtime_seconds` | `float` | Predicted wall-clock time |
| `circuit_depth` | `Optional[int]` | Quantum only |
| `qubit_count` | `Optional[int]` | Quantum only |
| `simulation_strategy` | `Optional[Literal["statevector","mps"]]` | How the circuit would be simulated |
| `parallelization_factor` | `float` | Speedup applied; 1.0 means serial |
| `resource_tier` | `Literal["light","moderate","heavy","extreme"]` | Bucketed classification |
| `calibrated` | `bool` | False when the profile's constants are reasoned, not measured |
| `hardware_is_hypothetical` | `bool` | True when the estimate assumed declared rather than verified hardware |
| `reasoning_trace` | `Dict[str, Any]` | Full audit trail |

### Classical Cost Model

Per algorithm:

$$\text{scale} = \frac{\text{size driver value}}{\text{ref size}}$$

$$\text{memory} = \text{base memory} \times \text{scale}^{\,m}, \qquad
\text{runtime} = \text{base runtime} \times \text{scale}^{\,r}$$

Two fields make this work across algorithm families that scale with
different quantities:

| Field | Purpose |
|-------|---------|
| `size_driver` | Which quantity drives cost — `dataset_size` or `dimensionality` |
| `ref_size` | The size at which `base_memory_gb` and `base_runtime_s` hold |

Supervised learning and graph algorithms scale with **samples or nodes**;
combinatorial optimization and PDE solvers scale with **decision variables
or grid size**. An earlier version used one global size measure and one
global reference point, which made optimization problems look trivial and
machine-learning problems look enormous — XGBoost on 20,000 records was
estimated at 288 GB.

The exponents approximate each algorithm's documented complexity class:
$O(n) \to 1.0$, $O(n \log n) \to 1.15$, $O(n^2) \to 2.0$, exponential
$\to 2.5+$.

| Algorithm | `size_driver` | `ref_size` | `memory_exp` | `runtime_exp` | GPU | MPI |
|-----------|---------------|-----------:|-------------:|--------------:|:---:|:---:|
| XGBoost | dataset_size | 10,000 | 1.05 | 1.15 | ✓ | |
| XGBoost Regressor | dataset_size | 10,000 | 1.05 | 1.15 | ✓ | |
| Random Forest | dataset_size | 10,000 | 1.10 | 1.15 | | ✓ |
| Random Forest Regressor | dataset_size | 10,000 | 1.10 | 1.15 | | ✓ |
| SVM | dataset_size | 10,000 | 2.00 | 2.00 | ✓ | |
| Logistic Regression | dataset_size | 10,000 | 1.00 | 1.00 | | |
| Linear Regression | dataset_size | 10,000 | 1.00 | 1.00 | | |
| Genetic Algorithm | dimensionality | 100 | 1.20 | 1.40 | | ✓ |
| Simulated Annealing | dimensionality | 100 | 1.10 | 1.30 | | ✓ |
| Branch and Bound | dimensionality | 100 | 2.20 | 2.60 | | ✓ |
| Monte Carlo Simulation | dataset_size | 1,000 | 1.00 | 1.10 | ✓ | ✓ |
| Finite Difference Solver | dimensionality | 1,000 | 1.60 | 1.80 | ✓ | ✓ |
| A* Search | dataset_size | 1,000 | 1.40 | 1.40 | | |
| Dijkstra | dataset_size | 1,000 | 1.30 | 1.30 | | |

### Quantum Cost Model

Qubit count, by mode:

$$n_{\text{linear}} = \lceil c \cdot D \rceil, \qquad
n_{\text{log2}} = \lceil c \cdot \log_2(N) \rceil$$

where $D$ is dimensionality (one qubit per decision variable or feature)
and $N$ is dataset size (an index register over a search space).

Circuit depth:

$$\text{depth} = \lceil \ell \cdot n \cdot \mu \rceil$$

for `depth_layers` $\ell$ and `depth_multiplier` $\mu$.

| Algorithm (exact registry key) | qubit mode | layers | multiplier | simulation strategy |
|-----------|-----------|-------:|-----------:|---------------------|
| Quantum Support Vector Machine (QSVM) | linear | 2 | 2.0 | MPS |
| Quantum Kernel Classifier | linear | 2 | 2.0 | MPS |
| Variational Quantum Regressor (VQR) | linear | 3 | 2.5 | MPS |
| Quantum Approximate Optimization Algorithm (QAOA) | linear | 3 | 2.0 | MPS |
| Variational Quantum Eigensolver (VQE) | linear | 4 | 2.5 | MPS |
| Quantum Statevector Simulation | linear | 1 | 1.0 | statevector |
| Hamiltonian Simulation | linear | 5 | 3.0 | statevector |
| Grover Search | log2 | 1 | 1.0 | statevector |

### Why Simulation Strategy Decides the Memory Model

Classical simulation cost depends far more on **how** a circuit is
simulated than on which algorithm it implements.

**Statevector** — exact, stores all $2^n$ complex amplitudes:

$$M_{\text{sv}} = 2^n \times 16 \text{ bytes}$$

Correct for algorithms whose purpose is exact state evolution, and for
small index registers. Memory is exponential, so it is viable to roughly
30 qubits.

**Matrix product state (MPS)** — polynomial in qubit count, quadratic in
bond dimension $\chi$:

$$M_{\text{mps}} = n \times 2\chi^2 \times 16 \text{ bytes}$$

Correct for shallow variational circuits (QAOA, VQE, quantum kernels),
which is how a large variational instance would actually be simulated.

The distinction is not cosmetic. Applying the statevector model to a
500-variable QAOA instance yields:

$$2^{500} \times 16 \text{ bytes} \approx 4.9 \times 10^{142}\ \text{GB}$$

— arithmetically correct and physically meaningless. Under MPS with
$\chi = 64$ the same instance needs **0.061 GB**. The `sim_strategy` field
exists so that the framework never reports the former.

Guards cap reported memory at $10^9$ GB and refuse to compute $2^n$ beyond
$n = 300$, so an extreme qubit count produces a clamped number rather than
an overflow.

### Parallel Speedup Model

$$\text{runtime}_{\text{final}} = \frac{\text{runtime}_{\text{raw}}}{f}$$

Speedup $f$ is claimed only when the algorithm supports a form of
parallelism **and the host actually provides it**:

$$f_{\text{MPI}} = \min\big(1 + (S - 1)\eta,\ f_{\max}\big), \qquad
\eta = 0.75,\ f_{\max} = 32$$

where $S$ is the detected MPI slot count — not a hardcoded node count. A
GPU-capable algorithm on a machine without CUDA runs serially, and the
estimate says so.

### Resource Tiers

| Tier | Bound |
|------|-------|
| `light` | ≤ 2 GB |
| `moderate` | ≤ 8 GB |
| `heavy` | ≤ 32 GB |
| `extreme` | > 32 GB |

---

## Step 7 — Hardware Mapper

### Objective

Assign an execution backend that the deployment actually has.

```text
ResourceEstimate + ExecutionDecision ──► Hardware Mapper ──► HardwareMapping
```

### Schema (`HardwareMapping`)

| Field | Type | Description |
|-------|------|-------------|
| `backend_type` | `Literal["cpu","gpu_cuda","mpi_cluster","qpu_simulator"]` | Selected backend |
| `device_spec` | `str` | Concrete device descriptor |
| `justification` | `str` | Why this backend |
| `fallback_backend` | `Optional[BackendType]` | Backup if the primary is unavailable |
| `is_hypothetical` | `bool` | True when targeting declared rather than verified hardware |
| `reasoning_trace` | `Dict[str, Any]` | Thresholds and detection notes |

### Hardware Inventory Detection

`backend/utils/hardware_detect.py` probes the host before anything is
assigned:

| Resource | Probe |
|----------|-------|
| CPU cores, RAM | `psutil` |
| CUDA devices | `nvidia-smi --query-gpu=name,memory.total` |
| MPI runtime | `mpiexec`/`mpirun` on PATH **and** `mpi4py` importable |

Without this the mapper recommends an 8-node cluster on a laptop that has
none, and an NVIDIA GPU on a machine with no NVIDIA hardware. Every probe
result is recorded in `detection_notes` and propagated into the mapping's
reasoning trace.

### Planning Modes

| Mode | Behaviour |
|------|-----------|
| `detected` (default) | Assign only backends present on this host |
| `target` | Assign against a declared target deployment |

Set via `QHPC_PLANNING_MODE`, with `QHPC_TARGET_MPI_SLOTS`,
`QHPC_TARGET_GPU_NAME` and `QHPC_TARGET_GPU_VRAM_GB` describing the
target.

Every estimate and mapping produced in `target` mode carries
`is_hypothetical = True`, and **Step 8 refuses to execute against a
hypothetical mapping**. A planned placement therefore cannot be reported
as an observed one.

### Backend Assignment Rules

**Quantum** — always the simulator; no physical QPU exists in this
deployment. The qubit ceiling depends on the simulation strategy:

| Strategy | Limit | Rationale |
|----------|------:|-----------|
| statevector | 30 | $2^{30} \times 16$ B = 16 GB, the edge of a workstation |
| MPS | 1,000 | Memory is polynomial; the ceiling is set by bond growth |

Beyond the limit the mapping is still produced but flagged as not
classically simulable with the configured method.

**Classical:**

$$\text{backend} = \begin{cases}
\texttt{gpu\_cuda}, & \text{GPU-capable} \wedge \text{CUDA present} \wedge M \ge 4\ \text{GB} \\
\texttt{mpi\_cluster}, & \text{MPI-capable} \wedge \text{MPI present} \wedge T \ge 20\ \text{s} \\
\texttt{cpu}, & \text{otherwise}
\end{cases}$$

When an algorithm supports a backend the host lacks, that gap is recorded
in `unavailable_backends` and named in the justification. "No GPU on this
host" and "too small to need one" are different conclusions and the
mapping distinguishes them.

---

## Step 8 — Runtime Execution & Telemetry

### Objective

Execute the mapped workload, measure what it cost, and compare that
against what Step 6 predicted.

```text
HardwareMapping + ResourceEstimate ──► Performance Engine ──► ExecutionTelemetry
```

This is the first module in the pipeline that performs work rather than
deciding about work.

### Schema (`ExecutionTelemetry`)

| Field | Type | Description |
|-------|------|-------------|
| `algorithm`, `backend_type`, `device_detail` | `str` | What ran, where |
| `telemetry_host` | `str` | Machine identity — runtimes do not transfer |
| `success` | `bool` | Whether it ran |
| `error_message` | `Optional[str]` | The real exception when it did not |
| `actual_runtime_seconds` | `float` | Measured |
| `actual_peak_memory_gb` | `float` | Measured |
| `predicted_runtime_seconds` | `float` | From Step 6 |
| `predicted_memory_gb` | `float` | From Step 6 |
| `runtime_error_ratio` | `Optional[float]` | actual ÷ predicted |
| `memory_error_ratio` | `Optional[float]` | actual ÷ predicted |
| `result_metric_name` | `Optional[str]` | Correctness metric |
| `result_metric_value` | `Optional[float]` | Its value |
| `compared_against_uncalibrated_estimate` | `bool` | Whether the prediction was measured |
| `raw_metrics` | `Dict[str, Any]` | Full detail |

The error ratio is the central quantity:

$$\rho = \frac{\text{actual}}{\text{predicted}}$$

$\rho = 1$ means the model was right; $\rho > 1$ that it was optimistic;
$\rho < 1$ that it was pessimistic. When the prediction is zero, $\rho$ is
reported as `None` rather than as a number, because a ratio against zero
is undefined and a placeholder would fabricate a result.

### Executable Workloads

The registries in Steps 3–5 hold algorithm **names and expert priors**;
nothing in the framework ever ran one. Step 8 supplies implementations for
a deliberately chosen subset — one per backend — rather than all 22, which
would be a library rather than a contribution.

| Workload | Backends | Why this one |
|----------|----------|--------------|
| XGBoost | cpu, gpu_cuda | `device=` is a real dispatch inside XGBoost |
| Monte Carlo Simulation | cpu, mpi_cluster | Embarrassingly parallel, known exact answer |
| Genetic Algorithm | cpu, mpi_cluster | Island model; communication-bound by contrast |
| Grover Search | qpu_simulator | Statevector; oracle correctness is checkable |
| Quantum Statevector Simulation | qpu_simulator | Fixed depth, for GPU-vs-CPU scaling |
| QAOA | qpu_simulator | MPS; validates the polynomial memory model |

An algorithm named in the registry but not implemented is reported as
unavailable, never as a successful run of nothing.

### Backend Dispatch

| Backend | Mechanism |
|---------|-----------|
| `cpu` | In-process, single-threaded |
| `gpu_cuda` | In-process, the library's own CUDA path |
| `mpi_cluster` | Out-of-process via `mpiexec` on `backend/engine/mpi_worker.py` |
| `qpu_simulator` | In-process, qiskit-aer (CPU or GPU) |

MPI is necessarily different: `mpiexec -n N` launches N independent
interpreters and `MPI_Init` runs in each, which cannot be entered from an
already-running process. The engine therefore spawns a worker module and
reads a JSON result line back.

Launcher families take incompatible flags. OpenMPI 4 reports
`OpenRTE`, OpenMPI 5 reports `PRRTE`, MPICH reports `HYDRA`, and Microsoft
MPI rejects OpenMPI's flags entirely. The engine classifies the launcher
from its `--version` output and tries the appropriate oversubscription
flag variants in order, because OpenMPI 5 replaced `--oversubscribe` with
`--map-by :OVERSUBSCRIBE`.

### Correctness Metrics

Every workload returns a correctness metric. Timing alone proves only that
something consumed CPU; the metric proves the algorithm solved its
problem.

| Workload | Metric | Correct value |
|----------|--------|---------------|
| XGBoost | test accuracy | well above chance |
| Monte Carlo | absolute error vs $\pi$ | within sampling noise |
| Genetic Algorithm | best Rastrigin fitness | global optimum is 0 |
| Grover Search | P(marked state) | → 1.0 |
| Statevector Simulation | $\sum\lvert a_i\rvert^2$ | exactly 1.0 |
| QAOA | approximation ratio vs ring optimum | in [0, 1] |

This is not decoration. During GPU validation, Grover produced an
excellent speedup curve (up to 26.4× at 22 qubits) while P(marked)
collapsed from 0.9995 to 0.0040. The implementation capped iterations at
64 where $\lfloor \pi/4 \cdot \sqrt{N} \rfloor = 1608$ were needed, so the
search was returning essentially noise. The measured probabilities matched
$\sin^2((2k+1)\theta)$ exactly at every qubit count, confirming the cause.

**Nothing but the correctness metric would have revealed it.** The timings
were clean and the speedup table looked like a result.

Because a scaling study compares the same workload across rank counts, each
metric declares how it should behave:

| Behaviour | Meaning |
|-----------|---------|
| `invariant` | Must not change; any drift is a bug |
| `stochastic` | Varies with random streams; judged against a target and tolerance |
| `rank_dependent` | Changes with rank count by design; report the trend |

Without this, a Monte Carlo estimate that legitimately varies per rank and
a corrupted reduction look identical.

### Honesty Constraints

| Constraint | Prevents |
|------------|----------|
| Hypothetical mappings are refused | Planned placements reported as measured |
| Error ratios are `None` against a zero prediction | Fabricated numbers in a results table |
| MPI runs report no memory figure | Launcher RSS, which excludes the ranks that did the work, standing in for the job's memory |
| `mpi_cluster` success requires `comm_size > 1` | A serial run masquerading as distributed when `mpi4py` is missing |
| Failures carry the real exception | A misleading "not installed" hiding the true cause |
| `compared_against_uncalibrated_estimate` | A large ratio read as mis-prediction rather than an unmeasured model |

Each of these was added in response to a defect found during development,
not anticipated in advance.

---

## Measured Results

All measurements name their host. Runtimes do not transfer between
machines, and a figure without a host is not reportable.

### MPI Strong Scaling

**Host:** Linux x86_64, 16 logical / 8 physical cores, WSL2, OpenMPI 5 +
mpi4py. Problem size held fixed; speedup is $T(1)/T(n)$, with $T(1)$
measured through `mpiexec -n 1` so both ends carry the same process
overhead. Median of 3 runs.

**Monte Carlo Simulation** — $4 \times 10^8$ samples:

| ranks | time (s) | speedup | efficiency | π error |
|------:|---------:|--------:|-----------:|--------:|
| 1 | 16.837 | 1.00× | 100.0% | 1.06e-4 |
| 2 | 15.917 | 1.06× | 52.9% | 1.06e-4 |
| 4 | 8.440 | 1.99× | 49.9% | 7.2e-5 |
| 8 | 4.722 | **3.57×** | 44.6% | 9.0e-5 |

**Genetic Algorithm** — 200 dimensions, 8,000 generations:

| ranks | time (s) | speedup | efficiency | best fitness |
|------:|---------:|--------:|-----------:|-------------:|
| 1 | 16.515 | 1.00× | 100.0% | 2209.5 |
| 2 | 16.895 | 0.98× | 48.9% | 2209.5 |
| 4 | 9.415 | 1.75× | 43.9% | 2266.3 |
| 8 | 6.680 | **2.47×** | 30.9% | 2441.4 |

**The contrast is the finding.** Monte Carlo communicates once at the end
and reaches 45% efficiency; the GA performs an `allreduce` every
generation and reaches 31%. A GA that scaled as well as Monte Carlo would
be evidence its migration step was not running.

Two effects are visible and both are real:

- **Efficiency falls as ranks rise.** Communication and startup are fixed
  costs that more ranks cannot amortise.
- **GA fitness degrades with rank count** (2209 → 2441). Strong scaling
  divides a fixed total population across ranks, so each island shrinks
  from 200 individuals to 25 and searches less well. This is a property of
  island-model GAs under strong scaling, not a defect.

Speedup past 8 ranks is limited by hyperthreading: 16 logical cores share
8 physical execution units, so additional ranks compete rather than add.
The 8-rank figures are the reportable ones.

### CUDA Acceleration

**Host:** Google Colab, NVIDIA Tesla T4 (15 GB), against Colab's CPU.

**XGBoost**, `device="cpu"` vs `device="cuda"`, identical data and
hyperparameters:

| n | CPU (s) | GPU (s) | speedup | CPU accuracy | GPU accuracy |
|--:|--------:|--------:|--------:|-------------:|-------------:|
| 10,000 | 2.613 | 0.901 | 2.90× | 0.8960 | 0.8960 |
| 100,000 | 6.916 | 1.670 | 4.14× | 0.9266 | 0.9266 |
| 500,000 | 22.167 | 5.004 | **4.43×** | 0.9402 | 0.9404 |

Accuracy is identical on both devices at every size, so the speedup is not
bought by computing something different.

Speedup grows with input size because kernel launch and host-to-device
transfer are fixed costs a short run cannot amortise. A single scalar
`GPU_SPEEDUP_FACTOR` is therefore a simplification of a size-dependent
curve; the calibrated value (4.3) is the median of the large-input regime.

**Statevector simulation**, CPU vs GPU, fixed 64-iteration circuit:

| qubits | CPU (s) | GPU (s) | speedup |
|-------:|--------:|--------:|--------:|
| 10 | 0.363 | 0.250 | 1.45× |
| 14 | 0.429 | 0.348 | 1.23× |
| 18 | 2.876 | 0.557 | 5.16× |
| 20 | 11.495 | 0.858 | 13.39× |
| 22 | 50.974 | 1.931 | **26.40×** |

GPU advantage grows sharply with statevector size: at 22 qubits there are
$4.2 \times 10^6$ amplitudes to update per gate, which suits the GPU's
parallelism, while small registers are dominated by launch overhead.

*These timings were collected with a truncated Grover circuit (64 of 1608
iterations). As statevector-simulation measurements they are valid — the
identical circuit ran on both devices — but they do not describe a working
Grover search. The dedicated* `Quantum Statevector Simulation` *workload
now exists so that timing work does not borrow an algorithm's name.*

### Predicted vs Measured

**Host:** Colab T4. Every prediction below came from **uncalibrated**
Step 6 constants.

| algorithm | backend | predicted (s) | actual (s) | ratio | correctness |
|-----------|---------|--------------:|-----------:|------:|------------:|
| XGBoost | cpu | 7.063 | 6.953 | **0.985** | 0.9266 |
| XGBoost | gpu_cuda | 7.063 | 1.620 | 0.229 | 0.9266 |
| Monte Carlo | cpu | 1.958 | 0.264 | 0.135 | 5e-4 |
| Grover Search | qpu_simulator | 0.042 | 0.489 | 11.517 | 0.0621 |
| QAOA | qpu_simulator | 0.453 | 0.320 | 0.706 | 0.4489 |

Median ratio 0.706; 2 of 5 within 2×.

Two observations:

- **XGBoost on CPU landed at $\rho = 0.985$** — within 1.5% — from
  constants reasoned purely from complexity class. That the model can be
  that close without calibration is itself informative.
- **The Grover ratio of 11.5 is not meaningful**, because that run was the
  truncated circuit described above. It is listed for completeness and
  excluded from any conclusion.

---

## Calibration Status

Each profile and constant records whether it was measured, and on which
host.

| Constant | Status | Source |
|----------|--------|--------|
| `GPU_SPEEDUP_FACTOR` = 4.3 | **Measured** | Colab T4 vs Colab CPU, XGBoost, n ≥ 100k |
| `MPI_PARALLEL_EFFICIENCY` = 0.75 | Reasoned | Conventional figure for coarse-grained workloads |
| `SIM_FLOPS_PER_SECOND` = 1e9 | **Uncalibrated** | Not derived from the truncated Grover run |
| Classical profiles (14) | **Uncalibrated** | Complexity-class reasoning |
| Quantum profiles (8) | **Uncalibrated** | Circuit-structure reasoning |

`backend/tests/benchmark_resources.py` fits power laws to real timings and
emits paste-ready constants. A profile marked `calibrated: True` must also
name a `calibration_host`, enforced by test.

The three fitted quantities do not transfer between machines equally:

| Quantity | Transfers? | Why |
|----------|-----------|-----|
| `memory_exp`, `runtime_exp` | **Yes** | Complexity class is a property of the algorithm |
| `base_memory_gb` | Broadly | An algorithm allocates roughly what it allocates |
| `base_runtime_s` | **No** | A figure from one CPU says nothing about another |

---

## Constants & Thresholds Registry (Steps 6–8)

**Resource tiers:** light ≤ 2 GB, moderate ≤ 8 GB, heavy ≤ 32 GB, extreme > 32 GB

**Backend thresholds:** `GPU_MEMORY_THRESHOLD_GB` = 4.0, `MPI_RUNTIME_THRESHOLD_S` = 20.0

**Simulation ceilings:** `STATEVECTOR_QUBIT_LIMIT` = 30, `MPS_QUBIT_LIMIT` = 1,000

**Parallel model:** `MPI_PARALLEL_EFFICIENCY` = 0.75, `MAX_MPI_SPEEDUP` = 32.0, `GPU_SPEEDUP_FACTOR` = 4.3 *(measured)*

**Simulation cost:** `STATEVECTOR_BYTES_PER_AMPLITUDE` = 16, `MPS_TENSORS_PER_SITE` = 2, default bond dimension 64, `SIM_FLOPS_PER_SECOND` = 1e9 *(uncalibrated)*

**Overflow guards:** `MAX_STATEVECTOR_EXPONENT` = 300, `MAX_REPORTED_MEMORY_GB` = 1e9

---

## Schemas Reference (Steps 6–8)

| Schema | File | Purpose |
|--------|------|---------|
| `ResourceEstimate` | `backend/models/resource_schema.py` | Step 6 cost prediction |
| `HardwareMapping` | `backend/models/hardware_schema.py` | Step 7 backend assignment |
| `ExecutionTelemetry` | `backend/models/telemetry_schema.py` | Step 8 measured execution |

| Module | File | Role |
|--------|------|------|
| Resource Estimator | `backend/agents/resource_estimator.py` | Step 6 |
| Hardware Detection | `backend/utils/hardware_detect.py` | Step 7 support |
| Hardware Mapper | `backend/engine/hardware_mapper.py` | Step 7 |
| Performance Engine | `backend/engine/performance_engine.py` | Step 8 dispatcher |
| Workloads | `backend/engine/workloads.py` | Step 8 implementations |
| MPI Worker | `backend/engine/mpi_worker.py` | Step 8 rank entry point |
| Constants | `backend/utils/resource_constants.py` | Steps 6–7 registry |

---

## Limitations

Stated explicitly, because a framework that routes workloads invites
questions about what it has actually demonstrated.

**The framework plans execution and validates it partially.**

| Capability | Status |
|------------|--------|
| CPU execution | Measured |
| MPI, single host | Measured — 16 logical / 8 physical cores |
| CUDA | Measured — Colab T4 |
| Quantum simulation (statevector, MPS) | Measured |
| **Multi-node distributed MPI** | **Untested.** No cluster was available |
| **Physical QPU** | **Out of scope.** Quantum work runs on a simulator |

Multi-node MPI differs from single-host MPI in ways that matter —
inter-node network latency, process placement, fault handling — and no
claim is made about it.

**Further limitations:**

- Most Step 6 constants are uncalibrated and marked as such throughout.
- Five of 22 algorithms have executable implementations; the rest are
  reported as unavailable rather than silently skipped.
- Peak memory is sampled from a background thread, so very short runs may
  under-report it; MPI runs report no memory figure at all.
- `GPU_SPEEDUP_FACTOR` is a single scalar standing in for a
  size-dependent curve, measured on one GPU with one algorithm.

---

## Completion Status

| Module | Stage | Status |
|--------|-------|--------|
| Resource Estimator | Step 6 | ✅ Completed |
| HPC Hardware Mapper | Step 7 | ✅ Completed |
| Runtime Execution & Telemetry Engine | Step 8 | ✅ Completed |
| Critic Agent Feedback Loop | Step 9 | ⏳ Pending |
| Final Synthesis & Report Generator | Step 10 | ⏳ Pending |

**Test coverage:** 141 tests across Steps 6–8 and the schema contracts,
excluding the LLM-dependent suites which require a running Ollama
instance.

Step 9 has concrete evidence to consume: `ExecutionTelemetry` carries
predicted-versus-actual for every run, so the critic agent can evaluate
the framework's own cost model rather than reasoning in the abstract.
