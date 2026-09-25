"""Aggregate raw sweep output into results/summary.csv (+ a markdown table).

Usage:  python harness/aggregate.py
Reads:  results/raw/*.jsonl   (skips *_smoke.jsonl unless --include-smoke)
Writes: results/summary.csv, results/summary.md, results/failures.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results"

# order matters for plots and tables: most→least precision
QUANT_ORDER = ["bf16", "Q8_0", "Q4_K_M", "Q3_K_M"]
MODEL_ORDER = ["qwen3-1.7b", "qwen3-4b", "llama-3.1-8b"]
FAIL_TYPES = ["invalid-output", "wrong-tool", "bad-args", "loop",
              "premature-stop", "server-error"]


def quant_rank(q: str) -> int:
    return QUANT_ORDER.index(q) if q in QUANT_ORDER else len(QUANT_ORDER)


def model_rank(m: str) -> int:
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)


def iqr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    qs = st.quantiles(values, n=4, method="inclusive")
    return qs[2] - qs[0]


def load_configs(include_smoke: bool) -> list[dict]:
    configs = []
    for path in sorted(RAW.glob("*.jsonl")):
        if "_ctx" in path.name:
            continue  # post-hoc diagnostics never enter the main summary
        if "_smoke" in path.name and not include_smoke:
            continue
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        manifest = next((r for r in rows if r.get("type") == "manifest"), None)
        if manifest is None:
            print(f"[warn] {path.name}: no manifest, skipped")
            continue
        if not any(r.get("type") == "config_done" for r in rows):
            print(f"[warn] {path.name}: incomplete (no config_done), skipped")
            continue
        summary = next((r for r in rows if r.get("type") == "config_summary"), {})
        configs.append({"path": path, "manifest": manifest, "summary": summary,
                        "rows": [r for r in rows if "task" in r]})
    return configs


def decode_speeds(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Collect per-call decode tok/s and TTFT(ms) from llama.cpp timings."""
    tps, ttft = [], []
    for r in rows:
        tl = r.get("timings")
        entries = tl if isinstance(tl, list) else [tl]
        for t in entries:
            if not isinstance(t, dict):
                continue
            if t.get("predicted_per_second"):
                tps.append(float(t["predicted_per_second"]))
            if t.get("prompt_ms") is not None:
                ttft.append(float(t["prompt_ms"]))
    return tps, ttft


def summarize(cfg: dict) -> list[dict]:
    man, summ, rows = cfg["manifest"], cfg["summary"], cfg["rows"]
    out = []
    for constrained in (False, True):
        t1 = [r for r in rows if r["task"] == "t1" and r["constrained"] is constrained]
        t2 = [r for r in rows if r["task"] == "t2" and r["constrained"] is constrained]
        if not t1 and not t2:
            continue

        t1_by_cat = defaultdict(list)
        for r in t1:
            t1_by_cat[r["category"]].append(r)

        fired = [r for r in t2 if r.get("injection_fired")]
        step_fails = Counter()
        for r in t2:
            step_fails.update(r.get("step_failure_counts", {}))
        primary = Counter(r["primary_failure"] for r in t2 if r["primary_failure"])

        # --- D4 diagnostics: error volume, and success under a shrinking budget.
        # Preregistered scoring caps step failures at infinity (goal predicate only).
        # Re-scoring the same stored traces under caps of 2 and 1 tests whether a
        # flat success curve is hiding rising step-level error volume.
        volumes = [sum(r.get("step_failure_counts", {}).values()) for r in t2]

        def success_under(cap: float) -> float | None:
            if not t2:
                return None
            n = sum(1 for r, v in zip(t2, volumes) if r["success"] and v <= cap)
            return round(n / len(t2), 4)

        tps, ttft = decode_speeds(t1 + t2)

        row = {
            "model": man["model"],
            "quant": man["quant"],
            "constrained": int(constrained),
            "ngl": man.get("ngl"),
            # Fits = every layer on the GPU AND the weights themselves within the 8 GB
            # budget (EVAL_PLAN §2, DEVIATIONS D3). ngl alone is not enough: on Windows
            # llama.cpp accepted ngl=99 for the two 8+ GB models and the driver spilled
            # to shared system memory, so ngl==99 does not imply the model fit.
            "fits_8gb": int(man.get("ngl") == 99 and man.get("gguf_bytes", 0) <= 8e9),
            "gguf_gb": round(man.get("gguf_bytes", 0) / 1e9, 2),
            "peak_vram_mib": summ.get("vram_peak_mib"),
            "load_s": man.get("load_seconds"),
            # --- accuracy
            "t1_n": len(t1),
            "t1_acc": round(sum(r["correct"] for r in t1) / len(t1), 4) if t1 else None,
            "t1_acc_simple": (round(sum(r["correct"] for r in t1_by_cat["simple"])
                                    / len(t1_by_cat["simple"]), 4)
                              if t1_by_cat["simple"] else None),
            "t1_acc_multiple": (round(sum(r["correct"] for r in t1_by_cat["multiple"])
                                      / len(t1_by_cat["multiple"]), 4)
                                if t1_by_cat["multiple"] else None),
            "t2_n": len(t2),
            "t2_success": round(sum(r["success"] for r in t2) / len(t2), 4) if t2 else None,
            "t2_answer_ok": round(sum(r["answer_ok"] for r in t2) / len(t2), 4) if t2 else None,
            "t2_recovery": (round(sum(r["recovered"] for r in fired) / len(fired), 4)
                            if fired else None),
            "t2_mean_steps": (round(st.mean(r["steps_used"] for r in t2), 2) if t2 else None),
            # D4: process damage, independent of the pass/fail verdict
            "t2_err_volume_mean": round(st.mean(volumes), 2) if volumes else None,
            "t2_err_volume_total": sum(volumes),
            "t2_success_budget2": success_under(2),
            "t2_success_budget1": success_under(1),
            "t2_success_budget0": success_under(0),
            # --- systems
            "decode_tps_median": round(st.median(tps), 2) if tps else None,
            "decode_tps_iqr": round(iqr(tps), 2) if tps else None,
            "ttft_ms_median": round(st.median(ttft), 1) if ttft else None,
            "schema_fallbacks": sum(1 for r in t1 + t2 if r.get("schema_fallback")),
        }
        for ft in FAIL_TYPES:
            row[f"t1_{ft}"] = sum(1 for r in t1 if r.get("fail_type") == ft)
            row[f"t2_step_{ft}"] = step_fails.get(ft, 0)
            row[f"t2_primary_{ft}"] = primary.get(ft, 0)
        out.append(row)
    return out


def write_failures(configs: list[dict]) -> int:
    """One row per failing T2 trajectory — the raw material for FAILURES.md."""
    path = OUT / "failures.csv"
    fields = ["model", "quant", "constrained", "scenario", "rep", "primary_failure",
              "steps_used", "oracle_steps", "injection_fired", "recovered",
              "ended_by_final", "final_answer", "trace"]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cfg in configs:
            man = cfg["manifest"]
            for r in cfg["rows"]:
                if r["task"] != "t2" or r["success"]:
                    continue
                w.writerow({
                    "model": man["model"], "quant": man["quant"],
                    "constrained": int(r["constrained"]), "scenario": r["id"],
                    "rep": r["rep"], "primary_failure": r["primary_failure"],
                    "steps_used": r["steps_used"], "oracle_steps": r["oracle_steps"],
                    "injection_fired": int(bool(r["injection_fired"])),
                    "recovered": int(bool(r["recovered"])),
                    "ended_by_final": int(bool(r["ended_by_final"])),
                    "final_answer": (r.get("final_answer") or "")[:200],
                    "trace": " > ".join(
                        f"{t['action']}{'!' + t['label'] if t['label'] else ''}"
                        for t in r.get("trace", [])),
                })
                n += 1
    return n


def write_markdown(rows: list[dict]) -> None:
    lines = ["# QuantCliff — summary", "",
             "`t1` = single-shot tool calls (BFCL v4 subset). "
             "`t2` = multi-step agent trajectories. "
             "`grammar` = JSON-schema-constrained decoding.", "",
             "| model | quant | grammar | fits 8GB | peak VRAM | t1 acc | t2 success | "
             "recovery | tok/s (median) |",
             "|---|---|---|---|---|---|---|---|---|"]

    def pct(v):
        return "—" if v is None else f"{v * 100:.0f}%"

    for r in rows:
        lines.append(
            f"| {r['model']} | {r['quant']} | {'on' if r['constrained'] else 'off'} | "
            f"{'yes' if r['fits_8gb'] else f'no ({r['gguf_gb']} GB weights, ngl={r['ngl']})'} | "
            f"{r['peak_vram_mib']} MiB | {pct(r['t1_acc'])} | {pct(r['t2_success'])} | "
            f"{pct(r['t2_recovery'])} | "
            f"{'—' if r['decode_tps_median'] is None else f'{r['decode_tps_median']:.1f}'} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true")
    args = ap.parse_args()

    configs = load_configs(args.include_smoke)
    if not configs:
        print("no complete config files in results/raw -- run the sweep first")
        return

    rows: list[dict] = []
    for cfg in configs:
        rows += summarize(cfg)
    rows.sort(key=lambda r: (model_rank(r["model"]), quant_rank(r["quant"]),
                             r["constrained"]))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_markdown(rows)
    n_fail = write_failures(configs)

    print(f"{len(configs)} configs -> results/summary.csv ({len(rows)} rows), "
          f"summary.md, failures.csv ({n_fail} failed trajectories)")
    for r in rows:
        g = "on " if r["constrained"] else "off"
        print(f"  {r['model']:<14} {r['quant']:<7} grammar={g} "
              f"t1={r['t1_acc']} t2={r['t2_success']} "
              f"recov={r['t2_recovery']} tok/s={r['decode_tps_median']}")


if __name__ == "__main__":
    main()
