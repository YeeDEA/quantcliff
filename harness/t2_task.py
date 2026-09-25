"""T2: drive one model through the multi-step scenarios and label the outcome."""

from __future__ import annotations

import json

from common import extract_json_object, action_schema
from engine import ScenarioEnv

SYSTEM_PROMPT = """You are an AI agent that completes the user's task by calling tools.

Every turn, reply with EXACTLY ONE JSON object and nothing else — no prose, no markdown fence:
{"thought": "<one short sentence>", "action": "<tool name or 'final'>", "args": {<arguments>}}

Rules:
- To call a tool, set "action" to its name and give arguments matching its schema exactly.
- Tool results arrive in the next message. Use them; do not invent values.
- Tools may fail transiently — retrying the same call once is allowed.
- When the task is fully complete, reply {"thought": "...", "action": "final", "args": {"answer": "<short answer including the key results/ids/numbers>"}}.
- Never call "final" before the task is actually done."""

FINAL_SCHEMA = {"type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"], "additionalProperties": False}


def build_system(spec: dict) -> str:
    tools = [{"name": t["name"], "description": t.get("description", ""),
              "parameters": t.get("parameters", {"type": "object"})}
             for t in spec["tools"]]
    return (SYSTEM_PROMPT + "\n\nAvailable tools:\n"
            + json.dumps(tools, ensure_ascii=False))


def build_schema(spec: dict) -> dict:
    names = [t["name"] for t in spec["tools"]] + ["final"]
    schemas = {t["name"]: t.get("parameters", {"type": "object"}) for t in spec["tools"]}
    schemas["final"] = FINAL_SCHEMA
    return action_schema(names, schemas)


def run_scenario(server, spec: dict, constrained: bool, rep: int) -> dict:
    env = ScenarioEnv(spec)
    msgs = [{"role": "system", "content": build_system(spec)},
            {"role": "user", "content": "Task: " + spec["task"]}]
    schema = build_schema(spec) if constrained else None
    max_steps = spec.get("max_steps", 12)

    step_labels: list[str] = []
    timings: list[dict] = []
    ended_by_final = False
    final_answer = None
    server_error = None
    schema_fallback = False

    for _ in range(max_steps):
        resp = server.chat(msgs, json_schema=schema, max_tokens=512)
        if resp["error"] and schema is not None and not schema_fallback:
            schema = action_schema([t["name"] for t in spec["tools"]] + ["final"])
            schema_fallback = True
            resp = server.chat(msgs, json_schema=schema, max_tokens=512)
        if resp["error"]:
            server_error = resp["error"]
            break
        timings.append(resp["timings"])
        raw = resp["text"]
        obj, why = extract_json_object(raw)
        msgs.append({"role": "assistant", "content": raw})

        if (obj is None or not isinstance(obj, dict) or "action" not in obj
                or not isinstance(obj.get("args"), dict)):
            step_labels.append("invalid-output")
            msgs.append({"role": "user", "content":
                         "ERROR: your reply was not a single valid JSON action object "
                         f"({why or 'wrong structure'}). Reply with exactly one JSON object "
                         'like {"thought": "...", "action": "...", "args": {...}}.'})
            continue

        if obj["action"] == "final":
            ended_by_final = True
            final_answer = str(obj["args"].get("answer", ""))
            break

        payload = env.call(obj["action"], obj["args"])
        step_labels.append(env.step_labels[-1])
        msgs.append({"role": "user",
                     "content": f"TOOL RESULT ({obj['action']}): "
                                + json.dumps(payload, ensure_ascii=False, default=str)})

    success = env.goal_met()
    if success:
        primary = None
    elif server_error:
        primary = "server-error"
    elif ended_by_final:
        primary = "premature-stop"
    else:
        primary = "loop"  # budget exhausted without reaching the goal (per EVAL_PLAN)

    effective = [l for l in step_labels if l]
    counts: dict[str, int] = {}
    for l in effective:
        counts[l] = counts.get(l, 0) + 1

    return {
        "task": "t2", "id": spec["id"], "constrained": constrained, "rep": rep,
        "success": success,
        "answer_ok": env.answer_ok(final_answer) if (success and final_answer is not None) else False,
        "primary_failure": primary,
        "step_failure_counts": counts,
        "steps_used": len(step_labels) + (1 if ended_by_final else 0),
        "oracle_steps": len(spec["oracle"]),
        "injection_fired": env.injection_fired,
        "recovered": env.recovered,
        "ended_by_final": ended_by_final,
        "final_answer": (final_answer or "")[:500],
        "server_error": server_error,
        "schema_fallback": schema_fallback,
        "timings": timings,
        "trace": [{k: t[k] for k in ("action", "label")} for t in env.trace],
        "transcript": msgs if primary else None,  # keep full logs only for failures
    }
