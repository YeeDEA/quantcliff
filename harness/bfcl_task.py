"""T1: BFCL v4 subset — prompt construction and AST-style scoring.

Scoring reimplements the BFCL possible-answer matching rules for the `simple`
and `multiple` categories: function name must match; every required parameter
must be present with a value in the allowed list; optional parameters (those
whose allowed list contains "") may be omitted but must match if present;
parameters not in the answer key are not allowed. Type-aware: int/float are
cross-comparable, strings are stripped, lists compare element-wise.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import ROOT, bfcl_params_to_json_schema, extract_json_object, action_schema

DATA = ROOT / "data" / "bfcl"

SYSTEM_PROMPT = """You are a precise function-calling assistant.
You are given one or more function specifications (JSON schema). Decide which single function answers the user's request and produce the call.

Reply with EXACTLY ONE JSON object and nothing else — no prose, no markdown fence:
{"thought": "<one short sentence>", "action": "<function name>", "args": {<arguments matching the function's schema>}}

Rules:
- "action" must be one of the provided function names.
- Include every required argument; use the schema's types exactly (numbers as numbers, not strings).
- Omit optional arguments unless the request specifies them."""


def load_items(category: str, n: int = 50) -> list[dict]:
    fn = {"simple": "BFCL_v4_simple_python.json", "multiple": "BFCL_v4_multiple.json"}[category]
    items = [json.loads(line) for line in open(DATA / fn, encoding="utf-8")][:n]
    answers = {json.loads(line)["id"]: json.loads(line)["ground_truth"]
               for line in open(DATA / f"possible_answer_{fn}", encoding="utf-8")}
    out = []
    for it in items:
        funcs = it["function"]
        tool_schemas = {f["name"]: bfcl_params_to_json_schema(f["parameters"]) for f in funcs}
        out.append({
            "id": it["id"],
            "category": category,
            "question": it["question"][0][-1]["content"],
            "functions": funcs,
            "tool_schemas": tool_schemas,
            "ground_truth": answers[it["id"]][0],
        })
    return out


def build_messages(item: dict) -> list[dict]:
    specs = [{"name": f["name"], "description": f.get("description", ""),
              "parameters": item["tool_schemas"][f["name"]]} for f in item["functions"]]
    sys = SYSTEM_PROMPT + "\n\nAvailable functions:\n" + json.dumps(specs, ensure_ascii=False)
    return [{"role": "system", "content": sys},
            {"role": "user", "content": item["question"]}]


def build_schema(item: dict) -> dict:
    names = [f["name"] for f in item["functions"]]
    return action_schema(names, item["tool_schemas"])


# ------------------------------------------------------------------ matching

def _norm(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _value_matches(got, allowed_list) -> bool:
    g = _norm(got)
    for allowed in allowed_list:
        if allowed == "":
            continue  # "" marks omittability, not a literal value
        if g == _norm(allowed):
            return True
    return False


def score_call(item: dict, obj: dict | None, parse_fail: str | None) -> dict:
    """Returns {correct, fail_type, detail}. fail_type ∈ None|invalid-output|wrong-tool|bad-args."""
    if obj is None:
        return {"correct": False, "fail_type": "invalid-output", "detail": parse_fail}
    if not isinstance(obj, dict) or "action" not in obj or not isinstance(obj.get("args"), dict):
        return {"correct": False, "fail_type": "invalid-output", "detail": "missing action/args"}

    gt = item["ground_truth"]
    gt_name = next(iter(gt))
    name, args = obj["action"], obj["args"]
    provided = {f["name"] for f in item["functions"]}
    if name != gt_name:
        ft = "wrong-tool" if name in provided else "invalid-output"
        return {"correct": False, "fail_type": ft, "detail": f"called {name!r}, expected {gt_name!r}"}

    spec = gt[gt_name]
    for p, allowed in spec.items():
        if p not in args:
            if "" in allowed:
                continue  # omittable
            return {"correct": False, "fail_type": "bad-args", "detail": f"missing required arg {p!r}"}
        if not _value_matches(args[p], allowed):
            return {"correct": False, "fail_type": "bad-args",
                    "detail": f"arg {p!r}={args[p]!r} not in allowed {allowed!r}"}
    extra = set(args) - set(spec)
    if extra:
        return {"correct": False, "fail_type": "bad-args", "detail": f"unexpected args {sorted(extra)}"}
    return {"correct": True, "fail_type": None, "detail": None}


def run_t1(server, items: list[dict], constrained: bool) -> list[dict]:
    results = []
    for it in items:
        msgs = build_messages(it)
        schema = build_schema(it) if constrained else None
        resp = server.chat(msgs, json_schema=schema, max_tokens=512)
        if resp["error"] and schema is not None:
            # oneOf-grammar may fail to compile for exotic schemas → structural fallback
            fallback = action_schema([f["name"] for f in it["functions"]])
            resp = server.chat(msgs, json_schema=fallback, max_tokens=512)
            resp["schema_fallback"] = True
        obj, why = extract_json_object(resp["text"])
        verdict = (score_call(it, obj, why) if not resp["error"]
                   else {"correct": False, "fail_type": "server-error", "detail": resp["error"]})
        results.append({
            "task": "t1", "id": it["id"], "category": it["category"],
            "constrained": constrained, **verdict,
            "raw": resp["text"][:2000], "timings": resp["timings"],
            "schema_fallback": resp.get("schema_fallback", False),
        })
    return results
