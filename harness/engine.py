"""T2 scenario engine: deterministic mock-tool environments defined as data.

A scenario JSON declares tools (JSON-schema args + a data-driven behavior),
constant lookup tables (`data`), an injected transient error, a goal predicate
over the final environment state, and an oracle call sequence that must solve
the scenario (used by validate_scenarios.py, never shown to the model).

Behaviors and goals use a restricted expression language evaluated by
`safe_eval` (python ast with a whitelist — no attribute access, no imports),
so every scenario is auditable data, not code.

Step failure taxonomy (fixed in EVAL_PLAN.md):
  invalid-output   unparseable / wrong structure / hallucinated tool name
  wrong-tool       schema-parseable call to a tool irrelevant to the task (distractor)
  bad-args         right tool, but args fail schema or hit no data (lookup miss)
  loop             identical repeat of an already-succeeded call, or step budget
                   exhausted without reaching the goal
  premature-stop   'final' issued while the goal predicate is unsatisfied
"""

from __future__ import annotations

import ast
import copy
import json
import operator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "harness" / "scenarios"

# ------------------------------------------------------------------ safe eval

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
        ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
_FUNCS = {"len": len, "round": round, "str": str, "int": int, "float": float,
          "abs": abs, "min": min, "max": max, "sum": sum, "sorted": sorted,
          "any": any, "all": all, "get": lambda d, k, default=None: d.get(k, default)}


def safe_eval(expr: str, names: dict):
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            raise NameError(node.id)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -ev(node.operand)
            if isinstance(node.op, ast.Not):
                return not ev(node.operand)
        if isinstance(node, ast.BoolOp):
            vals = [ev(v) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if type(op) not in _CMP or not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Subscript):
            return ev(node.value)[ev(node.slice)]
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {ev(k): ev(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"disallowed expression node: {ast.dump(node)[:80]}")

    return ev(ast.parse(expr, mode="eval"))


# ------------------------------------------------------------------ arg validation

def validate_args(args: dict, schema: dict) -> str | None:
    """Minimal JSON-schema check. Returns an error string or None."""
    if not isinstance(args, dict):
        return "args must be an object"
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return f"missing required argument '{req}'"
    for k, v in args.items():
        if k not in props:
            return f"unexpected argument '{k}'"
        err = _check_type(v, props[k], k)
        if err:
            return err
    return None


def _check_type(v, spec: dict, name: str) -> str | None:
    t = spec.get("type")
    ok = {None: lambda x: True,
          "string": lambda x: isinstance(x, str),
          "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
          "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
          "boolean": lambda x: isinstance(x, bool),
          "array": lambda x: isinstance(x, list),
          "object": lambda x: isinstance(x, dict)}.get(t, lambda x: True)(v)
    if not ok:
        return f"argument '{name}' must be of type {t}"
    if "enum" in spec and v not in spec["enum"]:
        return f"argument '{name}' must be one of {spec['enum']}"
    if t == "array" and "items" in spec:
        for i, item in enumerate(v):
            err = _check_type(item, spec["items"], f"{name}[{i}]")
            if err:
                return err
    return None


# ------------------------------------------------------------------ environment

class ScenarioEnv:
    def __init__(self, spec: dict):
        self.spec = spec
        self.state = copy.deepcopy(spec.get("initial_state", {}))
        self.data = spec.get("data", {})
        self.tools = {t["name"]: t for t in spec["tools"]}
        self.call_counts: dict[str, int] = {}
        self.succeeded_calls: list[tuple] = []
        self.injection = spec.get("error_injection")
        self.injection_fired = False
        self.recovered = False
        self.step_labels: list[str | None] = []
        self.trace: list[dict] = []

    # main entry — returns dict for the model + records the label
    def call(self, action: str, args) -> dict:
        label, payload = self._call_inner(action, args)
        self.step_labels.append(label)
        self.trace.append({"action": action, "args": args,
                           "label": label, "result": payload})
        return payload

    def _call_inner(self, action: str, args):
        if action not in self.tools:
            return "invalid-output", {"error": f"Unknown tool '{action}'."}
        tool = self.tools[action]
        if not isinstance(args, dict):
            return "bad-args", {"error": "Arguments must be a JSON object."}

        err = validate_args(args, tool.get("parameters", {"type": "object"}))
        if err:
            return "bad-args", {"error": f"Invalid arguments: {err}"}

        if tool.get("distractor"):
            # schema-valid call to an irrelevant tool: plausible reply, wrong move
            return "wrong-tool", {"result": tool.get("distractor_reply",
                                                     "OK (no effect on your task).")}

        # transient injected error (first attempt of the designated call)
        self.call_counts[action] = self.call_counts.get(action, 0) + 1
        if (self.injection and action == self.injection["tool"]
                and self.call_counts[action] == self.injection.get("nth_call", 1)
                and not self.injection_fired):
            self.injection_fired = True
            return None, {"error": self.injection["message"]}

        sig = (action, json.dumps(args, sort_keys=True))
        if sig in self.succeeded_calls:
            prev = next(t for t in self.trace
                        if (t["action"], json.dumps(t["args"], sort_keys=True)) == sig
                        and t["label"] is None)
            return "loop", prev["result"]

        beh = tool.get("behavior", {})
        names = {"state": self.state, "args": args, "data": self.data,
                 "True": True, "False": False, "None": None}
        try:
            for req in beh.get("require", []):
                if not safe_eval(req["cond"], names):
                    return "bad-args", {"error": req["error"]}
            result = safe_eval(beh["result"], names) if "result" in beh else {"ok": True}
            for path, expr in beh.get("effect", {}).items():
                self._set_state(path, safe_eval(expr, names))
        except (KeyError, IndexError, TypeError):
            return "bad-args", {"error": "Not found: no data matches these arguments."}

        if self.injection_fired and self.injection and action == self.injection["tool"]:
            self.recovered = True
        self.succeeded_calls.append(sig)
        return None, {"result": result}

    def _set_state(self, path: str, value):
        keys = path.split(".")
        node = self.state
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    # ------------------------------------------------------------ goal check

    def goal_met(self) -> bool:
        names = {"state": self.state, "data": self.data,
                 "True": True, "False": False, "None": None}
        try:
            return all(safe_eval(c, names) for c in self.spec["goal"].get("conditions", []))
        except (KeyError, IndexError, TypeError, NameError, ValueError):
            return False

    def answer_ok(self, answer: str) -> bool:
        names = {"state": self.state, "data": self.data,
                 "True": True, "False": False, "None": None}
        for expr in self.spec["goal"].get("answer_contains", []):
            try:
                needle = safe_eval(expr, names)
            except Exception:
                return False
            s = str(needle)
            if isinstance(needle, float) and needle == int(needle):
                if str(int(needle)) in str(answer) or s in str(answer):
                    continue
                return False
            if s.lower() not in str(answer).lower():
                return False
        return True


def load_scenarios() -> list[dict]:
    specs = []
    for p in sorted(SCENARIO_DIR.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            specs.append(json.load(f))
    return specs
