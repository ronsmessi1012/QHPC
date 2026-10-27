# Q-HPC Backend Architecture Evolution (Step 9 → Step 10)

> **Project:** Q-HPC — Quantum-Classical High Performance Computing Execution Framework
>
> **Module Documentation:** Critic Agent Feedback Loop & Final Synthesis
>
> **Version:** v1.00 (Completed through Step 10: Final Synthesis & Report Generator)
>
> *Continues [sampling.md](sampling.md) (Steps 0 → 5.5) and [steps_6_7_8.md](steps_6_7_8.md) (Steps 6 → 8).*
>
> *Index: [docs/README.md](README.md)*

---

## Table of Contents

- [Architecture at Step 10](#architecture-at-step-10)
- [Step 9 — Critic Agent Feedback Loop](#step-9--critic-agent-feedback-loop)
  - [Why the Judgement Is Deterministic](#why-the-judgement-is-deterministic)
  - [Finding Categories](#finding-categories)
  - [Prediction Accuracy Bands](#prediction-accuracy-bands)
  - [Metric Behaviour and False Alarms](#metric-behaviour-and-false-alarms)
  - [The Calibration Feedback Loop](#the-calibration-feedback-loop)
  - [Optional LLM Narrative](#optional-llm-narrative)
- [Step 10 — Final Synthesis & Report Generator](#step-10--final-synthesis--report-generator)
  - [Caveats as Structure, Not Prose](#caveats-as-structure-not-prose)
  - [Outcome Classification](#outcome-classification)
  - [Partial Pipelines](#partial-pipelines)
  - [Rendering](#rendering)
- [Worked Example](#worked-example)
- [A Bug Found by Mutation Testing](#a-bug-found-by-mutation-testing)
- [Constants Registry (Steps 9–10)](#constants-registry-steps-910)
- [Schemas Reference (Steps 9–10)](#schemas-reference-steps-910)
- [Limitations](#limitations)
- [Completion Status](#completion-status)

---

## Architecture at Step 10

Steps 2 → 5.5 decide. Steps 6 → 8 predict, place and measure. Steps 9 → 10
close the loop: they judge whether the framework's own claims held, and
produce a document stating what may legitimately be concluded.

```text
      ExecutionTelemetry  (Step 8)
              │
              ▼
┌──────────────────────────────┐
│ Step 9: Critic Agent         │  compares predicted vs measured
│  deterministic analysis      │  judges correctness and routing
│  + optional LLM narrative    │
└──────────────┬───────────────┘
               │ CriticReport
               │
               ├────────────────────────►  CalibrationSuggestion
               │                             feeds back into Step 6
               ▼
┌──────────────────────────────┐
│ Step 10: Report Generator    │  assembles all nine stages
│  typed caveats               │  states what may be claimed
└──────────────┬───────────────┘
               │ ExecutionReport
               ▼
       Markdown  +  JSON
```

The critic judges the **framework**, never the workload. A slow algorithm
is not a finding; a slow algorithm the framework predicted would be fast
is.

---

## Step 9 — Critic Agent Feedback Loop

### Objective

Check Steps 6 and 7's two claims — *this will cost X* and *this backend is
the right one* — against what Step 8 actually measured, and report with
evidence where the framework was wrong.

```text
ExecutionTelemetry (+ HardwareMapping, ResourceEstimate) ──► Critic Agent ──► CriticReport
```

### Why the Judgement Is Deterministic

Steps 2 and 5.5 use an LLM because algorithm selection under genuine
uncertainty benefits from judgement. Step 9 deliberately does not, for two
reasons.

**A critic that invented numbers would reintroduce the exact problem Step
5.5 exists to prevent.** Step 5.5 computes confidence mathematically
precisely so the LLM cannot claim 0.95 certainty on a borderline score of
0.944. A critic free to assert its own error ratios would undo that at the
final stage.

**A judgement about measurement must be reproducible.** Run the critic
twice over the same telemetry and the findings must be identical, or the
report is not evidence.

The LLM is therefore confined to narrative, written *over* findings that
already exist, and it cannot add, remove or re-rank them. Practically this
also means Step 9 is fully functional with no model available — which
matters, since Ollama is frequently not running.

### Finding Categories

Seven categories, each with a severity, evidence and a concrete
recommendation:

| Category | Severity | Meaning |
|----------|----------|---------|
| `correctness_failure` | critical | The workload returned a wrong answer |
| `execution_failure` | high | The assigned backend could not run it |
| `prediction_error` | scaled | Step 6's cost model was wrong (see bands below) |
| `missing_implementation` | medium | Routable but not executable |
| `suboptimal_routing` | low–medium | The measurement contradicts Step 7's choice |
| `uncalibrated_estimate` | info | The prediction rested on reasoning, not measurement |
| `unverified_hardware` | info | Execution correctly refused as hypothetical |

Two of these deserve comment.

**`correctness_failure` outranks everything.** A run that completed in good
time while producing the wrong answer is worse than one that failed, because
a failure is visible and a wrong answer is not. Its recommendation says
plainly that timings from that run must not be reported as results.

**`unverified_hardware` is `info`, not an error.** Step 8 refusing to
execute against a hypothetical mapping is the engine working correctly.
Filing correct behaviour as a defect would train a reader to ignore
findings.

### Prediction Accuracy Bands

A cost model is useful for routing if it gets the order of magnitude right
— routing decisions turn on thresholds like *is this over 4 GB*, not on
exact seconds. The bands reflect that:

$$\text{deviation} = \max\left(\rho,\ \frac{1}{\rho}\right), \qquad
\rho = \frac{\text{actual}}{\text{predicted}}$$

| Deviation | Severity | Interpretation |
|-----------|----------|----------------|
| ≤ 2× | *no finding* | Good enough to route correctly |
| 2–5× | low | Noticeable but not misleading |
| 5–10× | medium | The model is misleading |
| > 10× | high | The model is actively wrong |

Runs shorter than **0.05 s** are excluded entirely. Below that, timing is
dominated by interpreter startup, so the ratio describes overhead rather
than the cost model, and judging it would generate noise.

The recommendation distinguishes two cases that look identical in the
numbers:

- **Uncalibrated estimate** → "this measures an unmeasured model rather
  than a faulty one"
- **Calibrated estimate** → "this is a genuine regression in a measured
  model"

### Metric Behaviour and False Alarms

A correctness check that cried wolf would be ignored. Each workload
declares how its metric behaves, and the critic judges accordingly:

| Behaviour | Judged how |
|-----------|-----------|
| `invariant` | Must not change; any drift is a bug |
| `stochastic` | Against a declared target and tolerance |
| `rank_dependent` | Not judged — it changes by design |

The Genetic Algorithm's fitness worsens as ranks increase, because strong
scaling divides a fixed population into smaller islands. Flagging that as
a correctness failure would make the whole check useless. Monte Carlo's π
error varies with rank count because each rank seeds differently, but the
sample count is unchanged — so it is judged against 3σ sampling noise
rather than against itself.

### The Calibration Feedback Loop

This is what makes Step 9 a *feedback loop* rather than a report. Step 6
predicts, Step 8 measures, Step 9 computes the correction:

$$\text{suggested base runtime} = \text{current} \times \operatorname{median}(\rho)$$

Applying it would have made the median prediction exact. Each suggestion
rates its own confidence:

| Evidence | Confidence |
|----------|-----------|
| ≥ 3 runs, ratios within 1.5× of each other | high |
| ≥ 3 runs, scattered ratios | moderate |
| 2 runs | moderate |
| 1 run | low, marked provisional |

**Runs from different hosts are flagged `MIXED HOSTS` and unsafe to
apply.** `base_runtime_s` does not transfer between machines, so a median
over a laptop and a Colab instance describes neither. The critic refuses
to quietly average them.

### Optional LLM Narrative

`add_llm_narrative()` is a separate call, deliberately. The deterministic
report is complete without it, and a failure — no Ollama, no model, a
timeout — leaves the report fully usable with `llm_available = False`
recorded.

The system prompt forbids the model from inventing, recomputing or
contradicting any number, from adding or removing findings, and from
exceeding five sentences.

---

## Step 10 — Final Synthesis & Report Generator

### Objective

Assemble one query's journey through all nine stages into a single
document, and state what may legitimately be claimed from it.

```text
any subset of the pipeline's artifacts ──► Report Generator ──► ExecutionReport
                                                                 (Markdown + JSON)
```

### Caveats as Structure, Not Prose

Every number invites a claim; the caveats bound it. They are typed objects
with severities rather than a paragraph at the end, so the generator
cannot quietly omit one and a reader cannot skim past it.

| Kind | Severity | What it means |
|------|----------|---------------|
| `not_executed` | **blocking** | The plan was never run; figures are predictions |
| `hypothetical` | **blocking** | Hardware was declared, not verified |
| `incorrect_result` | **blocking** | The computation returned a wrong answer |
| `uncalibrated` | important | The prediction rested on reasoning |
| `host_specific` | note | The runtime describes one machine |
| `partial_pipeline` | note | Upstream stages are missing |

`is_reportable_as_measurement` answers the question most readers actually
have, and the rendered document states it in words:

```
**Citable as a measured result:** yes / NO
```

**The rule that matters most:** a critical correctness finding from Step 9
forces `outcome = "failed"` and blocks citability, whatever the timings
say. Without it, the Grover case documented in
[steps_6_7_8.md](steps_6_7_8.md#correctness-metrics) becomes
publishable by accident — a 26.4× speedup table attached to a search that
found nothing.

Note that `uncalibrated` is *important*, not *blocking*. An uncalibrated
model still produces a real measurement; it only limits what the error
ratio means. Treating it as blocking would make every current report
uncitable and the distinction worthless.

### Outcome Classification

| Outcome | When |
|---------|------|
| `measured` | Executed successfully, no blocking caveat |
| `planned_only` | A decision was reached but never executed |
| `failed` | Execution failed, **or** succeeded with a wrong answer |
| `incomplete` | The pipeline never reached a decision |

A wrong answer is classified `failed` rather than `measured`, because the
computation finished but its result is unusable.

### Partial Pipelines

Every stage is optional. A report can be built from routing alone, or from
archived telemetry whose upstream context is gone. **A missing stage is
recorded as missing, never inferred** — a reconstructed decision in a
results document is indistinguishable from a real one.

The report always accounts for all nine stages, split between
`stages_completed` and `stages_missing`.

### Rendering

`render_markdown()` for reading or a thesis appendix; `render_json()` for
further analysis; `save_report(report, path_base)` writes both.

The Markdown places predicted beside measured in one table, because the
comparison is the point:

| Quantity | Predicted | Measured | Ratio |
|---|---|---|---|
| runtime (s) | 1.109 | 0.2687 | 0.242 |
| peak memory (GB) | 0.0574 | 0.0121 | 0.2108 |

---

## Worked Example

A real run through Steps 6 → 10 on a 40-dimensional Rastrigin
minimisation, 400 generations.

**Step 9 output:**

```text
1/1 runs completed. Cost model: median runtime ratio 0.242 (0/1 within 2x).
All estimates used uncalibrated constants, so these ratios measure an
unmeasured model rather than a faulty one.

FINDINGS (2)
1. [LOW] prediction_error — Genetic Algorithm on cpu
   Runtime prediction was 4.1x pessimistic.
   evidence: predicted 1.1090s, measured 0.2687s, ratio 0.242
             (host: Linux x86_64)
   action:   Scale base_runtime_s for 'Genetic Algorithm' by 0.242 and set
             calibration_host. Note the estimate was uncalibrated, so this
             measures an unmeasured model rather than a faulty one.

2. [INFO] uncalibrated_estimate — Genetic Algorithm on cpu

CALIBRATION SUGGESTIONS
  Genetic Algorithm: multiply base_runtime_s by 0.242 (4.0 -> 0.9692)
    confidence: low; median of 1 run(s); fewer than 3 runs, provisional
```

**Step 10 output (abridged):**

```markdown
# Q-HPC Execution Report

**Query:** Minimise a 40-dimensional Rastrigin landscape over 400 generations.
**Outcome:** measured

## Summary
Genetic Algorithm ran on cpu in 0.269s, 0.24x the predicted time.

### How these numbers may be read
- [IMPORTANT] The cost prediction used constants derived from
  complexity-class reasoning, not measurement.
- [note] Runtime was measured on Linux x86_64. Runtimes do not transfer
  between machines.

**Citable as a measured result:** yes
```

Note the 4.1× prediction error is `low` severity, and the report remains
citable. The framework was wrong about the cost but right about the
routing, the computation was correct, and the caveats state the limits.
That is the distinction the two steps exist to draw.

---

## A Bug Found by Mutation Testing

Worth recording, because it illustrates why a green test suite is not
proof.

The critic's deviation calculation is `max(ratio, 1.0 / ratio)`.
`runtime_error_ratio` is rounded to four decimal places, so a 0.06 s run
against a 10⁶ s prediction rounds to **exactly 0.0** — and the division
crashes.

That is reachable on real data: uncalibrated constants with a steep
exponent (Branch and Bound, runtime exponent 2.6) genuinely predict
runtimes that large. The crash was masked only by the 0.05 s minimum-runtime
guard, which exists for an entirely different reason.

It surfaced when a mutation test removed that guard and produced a
`ZeroDivisionError` instead of the expected assertion failure. A zero
ratio is now handled as the most pessimistic possible error, with a
regression test.

---

## Constants Registry (Steps 9–10)

**Prediction accuracy bands:** `RATIO_GOOD` = 2.0, `RATIO_POOR` = 5.0,
`RATIO_SEVERE` = 10.0

**Measurement floor:** `MIN_MEANINGFUL_RUNTIME_S` = 0.05

**Calibration evidence threshold:** `MIN_RUNS_FOR_CALIBRATION` = 3

---

## Schemas Reference (Steps 9–10)

| Schema | File | Purpose |
|--------|------|---------|
| `CriticFinding` | `backend/models/critic_schema.py` | One evidenced judgement |
| `CalibrationSuggestion` | `backend/models/critic_schema.py` | A constant correction with confidence |
| `CriticReport` | `backend/models/critic_schema.py` | Aggregate judgement |
| `ReportCaveat` | `backend/models/report_schema.py` | A typed limit on what may be claimed |
| `ExecutionReport` | `backend/models/report_schema.py` | The final document |

| Module | File | Role |
|--------|------|------|
| Critic Agent | `backend/agents/critic_agent.py` | Step 9 |
| Report Generator | `backend/engine/report_generator.py` | Step 10 |

---

## Limitations

- **The critic judges cost and correctness, not algorithm choice.** It
  does not second-guess whether Step 5.5 selected the right algorithm,
  only whether the cost of running it was predicted correctly and whether
  the result was right.

- **Routing findings are conservative.** `suboptimal_routing` fires only
  where the measurement itself is the evidence. Claiming a backend that
  was never run would have been faster is speculation, and the critic does
  not do it.

- **Calibration suggestions are per-algorithm and per-host.** They adjust
  `base_runtime_s` only; the scaling exponents are not fitted here —
  `backend/tests/benchmark_resources.py` does that.

- **The LLM narrative is untested against a live model.** The code path
  exists and degrades correctly when Ollama is absent, which is the state
  it has been exercised in.

---

## Completion Status

| Module | Stage | Status |
|--------|-------|--------|
| Resource Estimator | Step 6 | ✅ Completed |
| HPC Hardware Mapper | Step 7 | ✅ Completed |
| Runtime Execution & Telemetry Engine | Step 8 | ✅ Completed |
| Critic Agent Feedback Loop | Step 9 | ✅ Completed |
| Final Synthesis & Report Generator | Step 10 | ✅ Completed |

**Test coverage:** 201 tests across Steps 6–10 and the schema contracts —
60 for Step 9 and Step 10 specifically. Four of Step 9's judgement rules
and four of Step 10's honesty rules are mutation-verified: each is
confirmed to fail a named test when removed.

**Not yet demonstrated:** the full pipeline has not been run end to end
from a natural-language query, because Steps 2 and 5.5 require a running
Ollama instance. Steps 6 → 10 have been run together repeatedly.
