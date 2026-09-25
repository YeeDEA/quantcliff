"""Sweep runner: model × quant × decoding-condition over T1 and T2.

Usage:
  python harness/runner.py --smoke              # tiny sanity run on one config
  python harness/runner.py                      # full locked matrix (resumable)
  python harness/runner.py --configs qwen3-1.7b:Q3_K_M,llama-3.1-8b:Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import ROOT, LlamaServer, ServerConfig, VramPoller  # noqa: E402
import bfcl_task  # noqa: E402
import t2_task  # noqa: E402
from engine import load_scenarios  # noqa: E402

RESULTS = ROOT / "results" / "raw"
LLAMA_BUILD = "b10502 (win-cuda-13.3-x64 release binary)"

MODELS = {
    "qwen3-1.7b": {
        "file": "Qwen_Qwen3-1.7B-{q}.gguf",
        "quants": ["bf16", "Q8_0", "Q4_K_M", "Q3_K_M"],
        "extra_args": ["--chat-template-kwargs", '{"enable_thinking": false}'],
    },
    "qwen3-4b": {
        "file": "Qwen_Qwen3-4B-Instruct-2507-{q}.gguf",
        "quants": ["bf16", "Q8_0", "Q4_K_M", "Q3_K_M"],
        "extra_args": [],
    },
    "llama-3.1-8b": {
        "file": "Meta-Llama-3.1-8B-Instruct-{q}.gguf",
        "quants": ["Q8_0", "Q4_K_M", "Q3_K_M"],
        "extra_args": [],
    },
}


def driver_info() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def config_list(only: str | None) -> list[tuple[str, str]]:
    combos = [(m, q) for m, info in MODELS.items() for q in info["quants"]]
    if only:
        want = set()
        for tok in only.split(","):
            m, q = tok.split(":")
            want.add((m, q))
        combos = [c for c in combos if c in want]
    return combos


def run_config(model: str, quant: str, *, smoke: bool = False,
               ctx: int | None = None) -> None:
    """Run one cell. `ctx` is for post-hoc diagnostics only: it writes to a
    separate `_ctx{N}` file that aggregate.py ignores, so a diagnostic can never
    contaminate the preregistered matrix."""
    info = MODELS[model]
    gguf = ROOT / "models" / info["file"].format(q=quant)
    if not gguf.exists():
        print(f"[skip] {gguf.name} not downloaded yet", flush=True)
        return
    suffix = "_smoke" if smoke else (f"_ctx{ctx}" if ctx else "")
    out_path = RESULTS / f"{model}_{quant}{suffix}.jsonl"
    if out_path.exists() and out_path.read_text(encoding="utf-8").strip().endswith('"config_done"}'):
        print(f"[skip] {out_path.name} already complete", flush=True)
        return
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"[start] {model} {quant}", flush=True)
    poller = VramPoller().start()
    scfg = ServerConfig(model_path=gguf, extra_args=info["extra_args"])
    if ctx:
        scfg.ctx = ctx
    server = LlamaServer(scfg)
    t_cfg = time.time()
    server.start()

    rows: list[dict] = []
    manifest = {
        "type": "manifest", "model": model, "quant": quant,
        "gguf": gguf.name, "gguf_bytes": gguf.stat().st_size,
        "llama_build": LLAMA_BUILD, "gpu": driver_info(),
        "ngl": server.effective_ngl, "load_seconds": server.load_seconds,
        "ctx": server.cfg.ctx, "seed": server.cfg.seed,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rows.append(manifest)

    try:
        # warmup (excluded from all timing stats)
        server.chat([{"role": "user", "content": "Say OK."}], max_tokens=8)

        n_t1 = 5 if smoke else 50
        items = bfcl_task.load_items("simple", n_t1) + bfcl_task.load_items("multiple", n_t1)
        scenarios = load_scenarios()
        if smoke:
            scenarios = scenarios[:2]
        reps = 1 if smoke else 2

        for constrained in (False, True):
            t0 = time.time()
            rows += bfcl_task.run_t1(server, items, constrained)
            print(f"  t1 constrained={constrained}: {time.time()-t0:.0f}s", flush=True)
            for rep in range(reps):
                t0 = time.time()
                for spec in scenarios:
                    rows.append(t2_task.run_scenario(server, spec, constrained, rep))
                print(f"  t2 constrained={constrained} rep={rep}: {time.time()-t0:.0f}s",
                      flush=True)
    finally:
        server.stop()
        vram = poller.stop()

    rows.append({"type": "config_summary", **vram,
                 "wall_seconds": round(time.time() - t_cfg, 1)})
    rows.append({"type": "config_done"})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[done] {model} {quant} -> {out_path.name} "
          f"(peak VRAM {vram['vram_peak_mib']} MiB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", help="comma list like qwen3-4b:Q4_K_M")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ctx", type=int,
                    help="post-hoc diagnostic: override context size; "
                         "writes a separate _ctx{N} file")
    args = ap.parse_args()

    combos = config_list(args.configs)
    if args.smoke and not args.configs:
        combos = combos[:1]
    for model, quant in combos:
        try:
            run_config(model, quant, smoke=args.smoke, ctx=args.ctx)
        except Exception as e:
            print(f"[DNF] {model} {quant}: {e}", flush=True)
    print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
