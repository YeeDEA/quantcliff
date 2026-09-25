"""Validate every scenario: schema sanity + the oracle must actually solve it.

Run:  python harness/validate_scenarios.py
Exit code 0 = all scenarios valid. This is the gate every scenario must pass
before it enters the benchmark.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from engine import ScenarioEnv, load_scenarios, validate_args  # noqa: E402


def check(spec: dict) -> list[str]:
    errs: list[str] = []
    sid = spec.get("id", "<no id>")

    for field in ("id", "task", "tools", "oracle", "goal", "max_steps"):
        if field not in spec:
            errs.append(f"missing field {field!r}")
            return errs

    tools = {t["name"]: t for t in spec["tools"]}
    distractors = {n for n, t in tools.items() if t.get("distractor")}
    real_tools = set(tools) - distractors

    if len(distractors) < 2:
        errs.append(f"needs >=2 distractor tools, has {len(distractors)}")
    if not (3 <= len(spec["oracle"]) <= 7):
        errs.append(f"oracle length {len(spec['oracle'])} outside 3..7")
    if spec["max_steps"] < len(spec["oracle"]) + 3:
        errs.append("max_steps too tight (< oracle + 3)")

    inj = spec.get("error_injection")
    if not inj:
        errs.append("missing error_injection")
    elif inj["tool"] not in {o["action"] for o in spec["oracle"]}:
        errs.append("error_injection tool never called by oracle (would never fire)")

    for o in spec["oracle"]:
        if o["action"] in distractors:
            errs.append(f"oracle uses distractor {o['action']}")
        if o["action"] not in tools:
            errs.append(f"oracle uses unknown tool {o['action']}")
        else:
            err = validate_args(o["args"], tools[o["action"]].get("parameters", {}))
            if err:
                errs.append(f"oracle args invalid for {o['action']}: {err}")
    if errs:
        return errs

    # --- replay the oracle: transient injected error gets one retry ---------
    env = ScenarioEnv(spec)
    for o in spec["oracle"]:
        payload = env.call(o["action"], o["args"])
        if "error" in payload:
            if env.injection_fired and not env.recovered:
                payload = env.call(o["action"], o["args"])  # the intended retry
            if "error" in payload:
                errs.append(f"oracle step {o['action']} failed: {payload['error']}")
                return errs

    if not env.injection_fired:
        errs.append("injected error never fired during oracle replay")
    if not env.recovered:
        errs.append("oracle retry did not recover from the injected error")
    if not env.goal_met():
        errs.append(f"goal not met after oracle replay; final state={env.state}")

    bad = [l for l in env.step_labels if l not in (None,)]
    if bad:
        errs.append(f"oracle replay produced failure labels: {bad}")

    # answer_contains must be evaluable
    try:
        env.answer_ok("probe")
    except Exception as e:
        errs.append(f"answer_contains not evaluable: {e!r}")

    _ = sid
    return errs


def main() -> int:
    specs = load_scenarios()
    ids = [s.get("id") for s in specs]
    dupes = {i for i in ids if ids.count(i) > 1}
    failed = False
    if dupes:
        print(f"FAIL duplicate ids: {dupes}")
        failed = True
    for spec in specs:
        errs = check(spec)
        if errs:
            failed = True
            print(f"FAIL {spec.get('id')}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {spec.get('id')} "
                  f"(oracle={len(spec['oracle'])} steps, "
                  f"tools={len(spec['tools'])}, "
                  f"distractors={sum(1 for t in spec['tools'] if t.get('distractor'))})")
    print(f"\n{len(specs)} scenarios, {'FAILURES PRESENT' if failed else 'all valid'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
