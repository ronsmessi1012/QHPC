# Q-HPC Documentation

Technical documentation for the Q-HPC backend, covering the full pipeline
from natural-language query to measured execution.

---

## Architecture Documents

Read in order; each continues the last.

| Document | Covers | Contents |
|----------|--------|----------|
| **[sampling.md](sampling.md)** | Steps 0 → 5.5 | Infrastructure, LLM problem analyzer, dual candidate generation, feature scoring, compatibility engine, borderline router, LLM agent selector |
| **[steps_6_7_8.md](steps_6_7_8.md)** | Steps 6 → 8 | Resource estimation, hardware mapping, runtime execution — **plus the measured MPI and CUDA results** |
| **[steps_9_10.md](steps_9_10.md)** | Steps 9 → 10 | Critic agent feedback loop, final synthesis and what may be claimed |

---

## The Pipeline at a Glance

```text
  User query
      │
      ▼  Steps 2 → 5.5           DECIDE
  Problem analysis, candidate generation, scoring,
  borderline detection, LLM arbitration
      │
      │  ExecutionDecision
      ▼  Steps 6 → 7             PREDICT & PLACE
  Cost estimation, hardware-aware backend assignment
      │
      │  ResourceEstimate + HardwareMapping
      ▼  Step 8                  EXECUTE & MEASURE
  Runtime execution, telemetry, predicted vs actual
      │
      │  ExecutionTelemetry
      ▼  Steps 9 → 10            JUDGE & REPORT
  Critique of the framework's own claims, calibration
  feedback, final document stating what is citable
```

---

## Where to Find Things

**Measured results** — [steps_6_7_8.md § Measured Results](steps_6_7_8.md#measured-results)

| Measurement | Result | Raw data |
|-------------|--------|----------|
| MPI strong scaling, Monte Carlo | 3.57× at 8 ranks (44.6% efficiency) | [`mpi_scaling.json`](../mpi_scaling.json) |
| MPI strong scaling, Genetic Algorithm | 2.47× at 8 ranks (30.9% efficiency) | [`mpi_scaling.json`](../mpi_scaling.json) |
| CUDA, XGBoost @ 500k rows | 4.43× on a Tesla T4 | *written up only* |
| CUDA, statevector simulation @ 22 qubits | 26.40× | *written up only* |

**Mathematical formulations** — the classical scaling model and quantum
simulation cost models are in
[steps_6_7_8.md § Step 6](steps_6_7_8.md#step-6--resource-estimator); the
prediction accuracy bands and calibration formula are in
[steps_9_10.md § Step 9](steps_9_10.md#step-9--critic-agent-feedback-loop).

**Limitations** — stated explicitly in both
[steps_6_7_8.md § Limitations](steps_6_7_8.md#limitations) and
[steps_9_10.md § Limitations](steps_9_10.md#limitations). Multi-node
distributed MPI is untested and no physical QPU is used.

**Calibration status** — most Step 6 constants are derived from
complexity-class reasoning rather than measurement, and are flagged as
such in every estimate the framework produces. See
[steps_6_7_8.md § Calibration Status](steps_6_7_8.md#calibration-status).

---

## Reproducing the Results

```bash
# What hardware does this host provide?
python -m backend.utils.hardware_detect

# Steps 6-10 and the schema contracts (no LLM required)
pytest backend/tests/test_resource_estimator.py \
       backend/tests/test_hardware_mapper.py \
       backend/tests/test_performance_engine.py \
       backend/tests/test_critic_agent.py \
       backend/tests/test_report_generator.py \
       backend/tests/test_schema.py \
       backend/tests/test_candidate_schema.py

# MPI strong-scaling benchmark (~30 minutes)
python -m backend.tests.benchmark_mpi_scaling --repeats 3 --json mpi_scaling.json

# Step 6 calibration harness
python -m backend.tests.benchmark_resources --repeats 3 --json calibration.json
```

CUDA measurements require an NVIDIA GPU; the notebook
[`notebooks/qhpc_step8_cuda_colab.ipynb`](../notebooks/qhpc_step8_cuda_colab.ipynb)
runs them on Google Colab's free T4.

---

## Other Files

| File | Purpose |
|------|---------|
| [README_additions.md](README_additions.md) | Proposed edits to the top-level README for Steps 6–10 |
