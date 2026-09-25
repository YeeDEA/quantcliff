# QuantCliff — Preregistered Evaluation Plan (v1)

**Committed before any experiment was run.** This document fixes the hypotheses, the
experiment matrix, the metrics, and the exclusion rules in advance, so that the results
published later in this repository can be checked against what we said we would measure.
Any deviation from this plan is documented in `DEVIATIONS.md`.

- Author: Yongjun Choi (@YeeDEA)
- Date: 2026-08-20
- Status: preregistration (no results existed at commit time — verify via git history)

## 1. Research questions

- **RQ1 (collapse).** As weight quantization deepens (FP16 → Q8_0 → Q4_K_M → Q3_K_M),
  when and in what order do (a) single-shot tool-call accuracy and (b) multi-step agent
  trajectory success degrade? *Hypothesis H1: trajectory success degrades earlier and
  more steeply than single-call accuracy — "agents break before benchmarks."*
- **RQ2 (anatomy).** What is the distribution of failure modes at each quantization
  level? Taxonomy (fixed in advance):
  1. `invalid-output` — output cannot be parsed into a tool call / violates the action schema
  2. `wrong-tool` — parseable call, but a tool that cannot serve the current step
  3. `bad-args` — right tool, wrong/hallucinated/malformed arguments
  4. `loop` — repeats an already-completed or already-failed identical call, or exhausts the step budget without progress
  5. `premature-stop` — produces a final answer / gives up while the task is objectively incomplete
  *Hypothesis H2: semantic failures (`wrong-tool`, `bad-args`) grow under quantization
  even where syntactic failures (`invalid-output`) remain flat or are absorbed elsewhere.*
- **RQ3 (intervention).** Does grammar-constrained decoding (JSON-schema-constrained
  sampling in llama.cpp) recover trajectory success lost to quantization, and at what
  latency cost? *Hypothesis H3: constrained decoding eliminates most `invalid-output`
  failures but does not recover `wrong-tool`/`bad-args`, i.e. it treats syntax, not
  semantics.* A null or negative result is a publishable outcome and will be reported as-is.

## 2. Experiment matrix (locked — no cell may be added mid-run)

| Axis | Values |
|---|---|
| Models | Qwen3-4B-Instruct-2507 · Qwen3-1.7B (thinking disabled) · Llama-3.1-8B-Instruct |
| Weight quant | F16/BF16* · Q8_0 · Q4_K_M · Q3_K_M (*F16 omitted for 8B: >8 GB VRAM, documented as a data point) |
| Decoding | unconstrained · JSON-schema-constrained (same schema, same prompts) |
| Tasks | T1: 100 single-shot items (BFCL v3 subset: 50 `simple` + 50 `multiple`) · T2: 20 multi-step scenarios × 2 repetitions |

≈ 11 model×quant combos × 2 decoding conditions. GGUF quantizations are taken from a
single provider (bartowski, imatrix K-quants) to avoid mixing quantization pipelines;
if a specific file is unavailable, the substitute is recorded in `DEVIATIONS.md`.

Runtime: llama.cpp `llama-server` (single release build, commit hash recorded), full GPU
offload where the model fits in 8 GB VRAM; when a model does not fit, the partial-offload
configuration is recorded and flagged — "does not fit an 8 GB consumer GPU" is itself a
reported result, not an excluded one.

Hardware (fixed): NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB VRAM, Windows 11.
Sampling: temperature 0, fixed seed. T1 runs once per config (greedy decoding);
T2 runs twice per config to expose any nondeterminism (both runs reported).

## 3. Tasks

### T1 — single-shot tool calls (BFCL v3 subset)
100 items from the Berkeley Function-Calling Leaderboard v3 dataset (Apache-2.0,
attributed): 50 from `simple`, 50 from `multiple` (tool selection among distractors).
Items are the *first 50 of each category in upstream file order* — chosen by position,
not by content, to prevent cherry-picking. Scoring: AST-style matching against the
upstream `possible_answer` files (function name + every required argument matches one
of the allowed values; type-aware; reimplementation of BFCL's checker, validated by
hand on 20 random items before the sweep).

### T2 — multi-step agent scenarios (authored for this repo)
20 deterministic scenarios in a mock-tool environment (no network, seeded), each
requiring 3–7 dependent tool calls (output of call *k* needed for call *k+1*), each with
**one injected tool error** (e.g. a transient "service unavailable" on the first attempt
of a specific step) to measure error recovery. Success = a scenario-specific goal
predicate over the final environment state (rule-based, no LLM judgment involved).
Per-step failure labels follow the RQ2 taxonomy; primary labels are rule-based, and an
LLM judge is used only as a secondary annotator whose agreement with rules and with
50 hand-labeled trajectories is itself reported.

All models see the *same* tool specifications and the same uniform action format
(a single JSON object per step), delivered through each model's own chat template.
We deliberately standardize the action format instead of using each model's native
tool-call wire format: the measured variable is quantization × decoding constraint,
not template plumbing. This choice and its limits are discussed in the README.

## 4. Metrics

Accuracy: T1 exact-match rate · T2 trajectory success rate · step-level correctness ·
recovery rate after the injected error · failure-type distribution.
Systems: time-to-first-token, decode tok/s (per-call, reported as median ± IQR across
all calls of a config; warmup call excluded), peak VRAM (nvidia-smi polled at 1 Hz),
model load time, and an analytic memory-budget table (weights + KV cache vs context).

## 5. Exclusion & retry rules (fixed in advance)

- Server crash / OOM: retry the affected call once; if the config cannot run at all,
  report the config as DNF with the reason (never silently dropped).
- Malformed *harness* behavior (bug in our code): fix, then rerun the **entire**
  affected config from scratch; the bug and rerun are logged in `DEVIATIONS.md`.
- A chat-template misconfiguration discovered mid-sweep counts as a harness bug (rerun),
  not as model failure.
- No prompt tuning after the sweep starts. Prompts are frozen at the commit that starts
  the sweep.

## 6. What would falsify our hypotheses

- H1 falsified if trajectory success tracks single-call accuracy within noise across the
  ladder, or degrades *less*. We will report "Q4 is safe for agents (on these tasks)" if so.
- H2 falsified if failure mix stays constant while volume scales.
- H3 falsified if constrained decoding recovers semantic failures too (interesting!),
  or recovers nothing, or costs prohibitive latency. All three outcomes get reported.
