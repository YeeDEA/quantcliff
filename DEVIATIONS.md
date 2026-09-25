# Deviations from the preregistered plan

Every departure from `EVAL_PLAN.md`, in the order it happened. Nothing here is
retroactive: entries are committed when the deviation occurs, before the affected
results are collected.

---

## D1 — BFCL v3 → v4 (dataset version)

**Planned:** "100 items from the Berkeley Function-Calling Leaderboard v3 dataset:
50 from `simple`, 50 from `multiple`."

**Actual:** Upstream (`ShishirPatil/gorilla`) has moved to BFCL **v4**; the v3 files
are no longer the maintained path. We use `BFCL_v4_simple_python.json` (first 50) and
`BFCL_v4_multiple.json` (first 50), with the matching `possible_answer/` keys.

**Why it does not affect the hypotheses:** the categories, item format, and
possible-answer matching semantics are unchanged between v3 and v4 for these two
categories; only the file naming and the item pool changed. Selection is still
by upstream file order (first 50), so no cherry-picking is possible.

**Date:** 2026-08-20, before any sweep data was collected.

---

## D2 — Harness bug: parameter schemas were emptied (fixed, smoke data discarded)

**What broke:** `bfcl_params_to_json_schema()` in `harness/common.py` recursed into
the `properties` object as if property *names* were schema keywords. Because names
like `base` / `height` matched none of the allowed keywords, every property was
dropped, yielding `{"type": "object", "properties": {}, "required": ["base","height"]}`.

**Impact:** both arms were affected, not just the intervention arm.
1. The tool specifications placed in the prompt lost all parameter types and
   descriptions, so models had to guess arguments from the `required` name list.
2. Under constrained decoding the grammar permitted `args: {}`, so the model emitted
   empty arguments and scored 0% on T1 — an artifact that would have been misread as
   "grammar constraints destroy tool calling."

**Resolution:** fixed to map each property name to its converted subschema (and to
handle tuple-typed `items` lists). Verified across all 100 T1 items that no
answer-key parameter is missing from the converted schema. Per `EVAL_PLAN.md` §5
("malformed harness behavior ... rerun the entire affected config from scratch"),
**all smoke-run outputs collected before the fix were deleted**, not patched.

**No results were reported from the buggy code.** The bug was caught during the
pre-sweep chain check, before the locked matrix was run.

**Date:** 2026-08-20, before the full sweep.

---

## D3 — Two models cannot be fully offloaded to 8 GB (anticipated, reported not excluded)

> (D3 below was anticipated in the plan; D4 was added after a literature check,
> **before any config in the locked matrix had finished**.)

`Meta-Llama-3.1-8B-Instruct-Q8_0` (8.54 GB) and `Qwen3-4B-Instruct-2507-bf16`
(8.05 GB) exceed the 8 GB VRAM budget before any KV cache is allocated. Per
`EVAL_PLAN.md` §2 these are **not excluded**: the runner was set to fall back down an
`n_gpu_layers` ladder (99 → 32 → 20) and record the effective value in each config's
manifest. **Correction (post-publication):** the fallback never triggered — llama.cpp
accepted `ngl = 99` for both models, and peak VRAM sat at the ceiling (7822 / 7818 MiB)
while decode fell to 5.9 / 8.1 tok/s, consistent with the Windows driver spilling into
shared system memory rather than a layer offload. The original aggregator derived
`fits_8gb` from `ngl == 99` alone and therefore marked both as fitting; it now also
requires weights ≤ 8 GB, and `summary.md` / `summary.csv` / the figures were regenerated
(no accuracy or throughput value changed, only the flag). Their latency numbers are
not comparable to configs that genuinely fit —
tables and plots flag them, and the README states this explicitly. "This model
does not fit a consumer 8 GB GPU" is a result, not a gap.

---

## D4 — Added analysis: error-volume and shrinking-budget diagnostics; H1 reframed

**Trigger.** A pre-sweep literature check surfaced Jang, Yang, Lim & Park,
*"Flat Score, Amplified Failures: How the Error Budget Masks Damage in Quantized
LLM Agents"* (arXiv:2607.27275, 2026-07-29). It tests weight-only quantization at
16/8/4-bit on multi-turn tool-calling agents (τ²-bench, two model families, two
domains) and reports that task scores stay flat under multiple-comparison
correction while the *volume* of the model's pre-existing failure mode grows up to
2.5× — the benchmark's ten-error budget absorbs the extra failures. Shrinking that
budget to two errors re-exposes a 17-point gap.

**Why this matters here.** That result partly anticipates our H1. We are therefore
restating what this repository does and does not claim:

- **Not claimed:** that we are the first to observe quantization damaging agentic
  process while aggregate scores stay flat. Flat Score has priority on that claim
  and is cited as the closest prior work.
- **Still open, and what we measure:** (a) the small-model, consumer-hardware
  regime — 1.7B–8B on a single 8 GB laptop GPU, with peak VRAM and decode
  throughput measured alongside accuracy, which Flat Score does not report;
  (b) an extra rung **below** 4-bit (Q3_K_M), where the ladder may actually break;
  (c) an explicit **single-shot vs multi-step contrast on the same models**
  (BFCL v4 items vs our trajectories) — Flat Score evaluates only multi-turn
  episodes, so the cross-task-type gap is not measured there;
  (d) a **decoding-level** intervention (JSON-schema-constrained decoding) rather
  than the prompt-level error-repair intervention they tested — a different
  mechanism, and one in direct tension with Tam et al., *"Let Me Speak Freely?"*
  (arXiv:2408.02442), which reports that format restrictions degrade reasoning.

**Analyses added** (both computable from data the locked matrix already collects;
no change to the matrix, prompts, or scoring):
1. **Error volume per task** — total step-level failures per trajectory by type,
   not just the binary success rate. Already recorded as `step_failure_counts`.
2. **Shrinking error budget** — trajectory success recomputed under a cap of
   *k* step-level failures for k = ∞ (as preregistered), 2, and 1, to test whether
   a flat success curve hides rising error volume in our regime too. Computed by
   replaying stored traces; it re-scores existing logs and re-runs nothing.

**H1 restated.** Original: "trajectory success degrades earlier and more steeply
than single-call accuracy." Retained as the primary hypothesis, with an added
secondary prediction from the masking account: *if* trajectory success is flat
across the ladder, step-level error volume should still rise, and the
shrinking-budget score should separate where volume rose. Both outcomes are
reportable; neither is a null result.

**Date:** 2026-08-20, while the first config was still running and **before any
completed config was inspected** (verify: `results/raw/` was empty at this commit).

---

## D5 — Added diagnostic: context-window sensitivity check (post-hoc, no headline number changes)

**Status.** Unlike D1–D4, this entry is **post-hoc**: the run was executed *after*
the locked matrix had completed. It is recorded here for completeness because the
raw output ships with the repository, and an unexplained run in a benchmark
repository is indistinguishable from an unreported one.

**What was run.** `llama-3.1-8b` / `Q8_0`, the config that sits closest to the 8 GB
ceiling, repeated with the context window halved: `ctx 8192 -> 4096`. Everything
else identical — same GGUF (`gguf_bytes: 8540775840`), same `llama_build`
(`b10502 win-cuda-13.3-x64`), same `seed: 42`, same `ngl: 99`, same items, both
decoding arms.

**Why.** Context length is a plausible confound for this config specifically: KV
cache competes with weights for the same 8 GB, so a reviewer can reasonably ask
whether the reported numbers are an artifact of the chosen window.

**Result — identical.** Not "close": bit-identical outcome counts in both arms.

| | ctx 8192 (locked matrix) | ctx 4096 (diagnostic) |
|---|---|---|
| T1 correct, unconstrained | 88 / 100 | 88 / 100 |
| T1 correct, constrained | 89 / 100 | 89 / 100 |
| T2 success, unconstrained | 18 / 40 | 18 / 40 |
| T2 success, constrained | 18 / 40 | 18 / 40 |
| Peak VRAM | 7818 MiB | 7827 MiB |
| Wall clock | 6731.8 s | 6228.0 s |

**Interpretation.** In this regime the context window is not a confound for the
reported accuracy or trajectory metrics. Peak VRAM is essentially unchanged, which
indicates weights rather than KV cache are the binding constraint at `ngl: 99`.
Wall-clock time drops about 7.5%, which is a throughput observation, not a quality
one.

**Not folded into the results.** `results/summary.csv` and every figure cover the
11-config locked matrix only. This diagnostic is deliberately excluded from those
aggregates so that the reported matrix stays exactly as preregistered; it is shipped
as raw data (`results/raw/llama-3.1-8b_Q8_0_ctx4096.jsonl`, log
`results/diag_ctx4096.log`) for anyone who wants to check it.

**Date:** 2026-08-20, after the locked matrix completed.

---

## D6 — Planned measurements that were not delivered (recorded post-publication)

`EVAL_PLAN.md` promised four things that do not appear in the results. They were not
produced, and this entry exists so that the gap is stated rather than discovered:

| Planned (EVAL_PLAN.md) | Status |
|---|---|
| §3 T1: BFCL checker "validated by hand on 20 random items before the sweep" | No record of the hand check survives in the repo |
| §3 T2: LLM judge as a secondary annotator, agreement with rules and with 50 hand-labeled trajectories reported | Not run — all failure labels in `FAILURES.md` / `failures.csv` are the rule-based primary labels only |
| §4: decode tok/s reported as **median ± IQR** | Only the median is reported (`summary.md`, "tok/s (median)") |
| §4: analytic memory-budget table (weights + KV cache vs context) | Not produced; the closest evidence is peak VRAM per config and the D5 context-halving diagnostic |

None of these changes a reported number. They narrow what the results can claim: failure
categories are rule-assigned without an inter-annotator check, and throughput has no
spread attached.
