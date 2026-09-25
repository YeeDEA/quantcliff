# Failure catalogue

Real traces from `results/failures.csv` (381 failed trajectories across all 11
configurations). Notation: `tool > tool!label` — a call and, if it failed, its label.

The point of this file is that the aggregate numbers in the README hide *why* agents
fail. Almost none of it is broken JSON.

---

## `premature-stop` — the dominant failure, and the least visible one

The model completes a plausible-looking sequence, announces success with specific
identifiers, and stops. Nothing in its output signals failure.

**Qwen3-1.7B / Q3_K_M / `s01_flight_booking`**

```
search_flights > book_flight > book_flight > reserve_shuttle
final answer: "Booking reference: BR-3310. Total flight price: $850.
               Shuttle confirmation: SH-220."
```

The task asked for the cheapest **direct** flight. UA892 at $850 is the cheapest flight;
it has a layover. KE023 at $890 is the cheapest direct one. The model optimised the
adjective it noticed and dropped the one it did not, then reported a confident,
internally consistent answer with a real booking reference and a real shuttle
confirmation. Every individual tool call succeeded. Only the goal predicate knows this
run failed.

This is the failure mode that a single-call benchmark cannot see and that a
schema-constrained grammar cannot prevent: the output was perfectly well-formed.

## `loop` — thrashing between lookups

**Qwen3-1.7B / bf16 / `s03_meeting_scheduling`**

```
find_slots > list_rooms > list_rooms!loop > find_slots!loop > list_rooms!loop
  > find_slots!loop > list_rooms!loop > list_rooms > book_meeting > book_meeting!bad-args
```

The scenario requires cross-referencing slots against room capacity. The model re-issues
the same two lookups with identical arguments five times — it has the information it
needs after step two but cannot combine it — then exhausts its step budget on a booking
with wrong arguments. Note this is at **bf16**: full precision does not prevent it.

## `bad-args` — the arithmetic wall

**Qwen3-1.7B / Q3_K_M / `s05_analytics_pipeline`**

```
find_dataset > fetch_rows > fetch_rows > validate_rows
  > store_report!bad-args ×6
```

The scenario requires the mean of four EMEA rows, which is 41250.25, and `store_report`
rejects any other value. The model retried the same rejected call six consecutive times
without revising the number. The tool's error message stated the requirement each time.

`bad-args` is the most common step-level error at nearly every precision on all three
models, and its volume is what grows as the ladder descends (14 → 25 events on
Qwen3-4B, 46 → 55 on Qwen3-1.7B). The *kind* of mistake does not change; the *amount* does.

## `invalid-output` — what the grammar actually binds on

This label covers two things: output that cannot be parsed as an action at all, and calls
to a tool that does not exist. It appears on Qwen3-1.7B at Q4_K_M (26 step-level events)
and Q3_K_M (15), and on Llama-3.1-8B at every precision (18 / 21 / 24). It **never**
appears on Qwen3-4B, at any precision.

Those are exactly the cells where constrained decoding has anything to bind on. It
removes all of them — 104 of 104 events, to zero in every cell. What it does *not* do is
turn them into successes; see the substitution table at the end of this file. It also
cannot touch the wrong-flight selection above, the thrashing, or the arithmetic wall,
because all three produce perfectly well-formed output.

---

## Scenario discrimination (a limitation, stated plainly)

Not every scenario carries signal for the models tested.

**Never failed, in any configuration** — no discriminative power here:
`s12_devops_incident`, `s13_repo_branch_pr`, `s14_db_migration`. All three enforce strict
state preconditions (drain before restart, branch before commit, backup before migrate),
and the tools' error messages appear to guide models onto the correct order reliably.

**Failed in every configuration** — floor effect, also no discriminative power:
`s04_cloud_drive`, `s06_order_placement`, `s11_invoice_creation`. Each combines a
selection trap with exact-arithmetic validation (discounted totals, ordered rounding),
and no model at any precision cleared them.

So 6 of 20 scenarios are at ceiling or floor, leaving roughly 14 that separate
configurations. Any future version of this benchmark should rebalance: the state-machine
scenarios are too easy and the compound arithmetic ones too hard for the 1–4B range.

## What the catalogue implies

If you are deploying a small quantized model as an agent, the failure you should
instrument for is not malformed JSON. It is a well-formed, confident, wrong completion —
`premature-stop` accounts for 15–20 of 40 failed runs on Qwen3-1.7B and 7–10 on
Qwen3-4B. Schema validation at the API boundary will not catch any of it. A goal
predicate, or a verification step, will.

---

## Error substitution under constrained decoding

The strongest evidence in this repository is a before/after on the same scenarios with
only the decoding constraint changed. `invalid-output` goes to zero in every cell where
it occurred; the freed error mass reappears elsewhere.

| model / quant | invalid-output | wrong-tool | bad-args | loop | total | success |
|---|---|---|---|---|---|---|
| Qwen3-1.7B Q4_K_M off | 26 | 0 | 25 | 8 | 59 | 45% |
| Qwen3-1.7B Q4_K_M **on** | **0** | 0 | **46** | 8 | 54 | 45% |
| Qwen3-1.7B Q3_K_M off | 15 | 0 | 55 | 7 | 77 | 20% |
| Qwen3-1.7B Q3_K_M **on** | **0** | 0 | 50 | 12 | 62 | **30%** |
| Llama-3.1-8B Q8_0 off | 18 | 18 | 32 | 40 | 108 | 45% |
| Llama-3.1-8B Q8_0 **on** | **0** | 20 | 36 | **62** | **118** | 45% |
| Llama-3.1-8B Q4_K_M off | 21 | 8 | 19 | 27 | 75 | 57% |
| Llama-3.1-8B Q4_K_M **on** | **0** | **20** | 23 | 38 | **81** | 57% |
| Llama-3.1-8B Q3_K_M off | 24 | 2 | 35 | 15 | 76 | 70% |
| Llama-3.1-8B Q3_K_M **on** | **0** | **16** | 32 | 23 | 71 | 70% |

Two cells end with *more* total step errors under the constraint than without it, and
four of the five end with identical task success. The tool names the unconstrained models
invented are worth reading: `filter`, `map`, `json_query`, `calculate`,
`get_shipping_fee`. The model was not producing noise — it was reaching for a
data-manipulation or arithmetic primitive that the environment does not provide. Blocking
the name does not supply the capability, so the constrained model picks a real tool that
cannot do the job instead. That is why `wrong-tool` absorbs the difference.

Qwen3-4B never emitted a malformed action at any precision and its error counts are
unchanged by the constraint (16→17, 20→20, 27→27, 35→35), which is the control this
comparison needs: the grammar does nothing when it does not bind.
