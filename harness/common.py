"""Shared infrastructure: llama-server lifecycle, chat calls, VRAM polling, JSON parsing."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_EXE = ROOT / "tools" / "llama-cuda" / "llama-server.exe"
PORT = 8090


# ---------------------------------------------------------------- VRAM poller

class VramPoller:
    """Polls nvidia-smi total memory.used at ~1 Hz; records the peak."""

    def __init__(self):
        self.peak_mib = 0
        self.baseline_mib = self._read()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read() -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return -1

    def _loop(self):
        while not self._stop.is_set():
            v = self._read()
            if v > self.peak_mib:
                self.peak_mib = v
            self._stop.wait(1.0)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        return {"vram_baseline_mib": self.baseline_mib, "vram_peak_mib": self.peak_mib}


# ---------------------------------------------------------------- server

@dataclass
class ServerConfig:
    model_path: Path
    ctx: int = 8192
    ngl: int = 99
    port: int = PORT
    seed: int = 42
    extra_args: list[str] = field(default_factory=list)


class LlamaServer:
    """Starts/stops one llama-server process and reports what actually happened."""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.load_seconds: float | None = None
        self.effective_ngl: int | None = None
        self.log_path = ROOT / "server.log"

    def start(self, ngl_ladder=(99, 32, 20)) -> "LlamaServer":
        for ngl in ngl_ladder:
            if self._try_start(ngl):
                self.effective_ngl = ngl
                return self
        raise RuntimeError(f"server failed to start for {self.cfg.model_path.name} "
                           f"(tried ngl={ngl_ladder}); see {self.log_path}")

    def _try_start(self, ngl: int) -> bool:
        args = [
            str(SERVER_EXE), "-m", str(self.cfg.model_path),
            "-c", str(self.cfg.ctx), "-ngl", str(ngl),
            "--port", str(self.cfg.port), "--seed", str(self.cfg.seed),
            "--temp", "0", "--jinja", "--no-webui",
            "--parallel", "1",
        ] + self.cfg.extra_args
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        t0 = time.time()
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        deadline = t0 + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:  # crashed (likely OOM at this ngl)
                return False
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.cfg.port}/health", timeout=2) as r:
                    if r.status == 200:
                        self.load_seconds = round(time.time() - t0, 1)
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        self.stop()
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    # ------------------------------------------------------------ chat call

    def chat(self, messages: list[dict], *, json_schema: dict | None = None,
             max_tokens: int = 512, retries: int = 1) -> dict:
        """One /v1/chat/completions call. Returns {text, timings, error}."""
        body: dict = {
            "messages": messages,
            "temperature": 0,
            "seed": self.cfg.seed,
            "max_tokens": max_tokens,
            "timings_per_token": False,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "strict": True, "schema": json_schema},
            }
        data = json.dumps(body).encode()
        last_err = None
        for _ in range(retries + 1):
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.cfg.port}/v1/chat/completions",
                data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    resp = json.loads(r.read().decode("utf-8", errors="replace"))
                return {
                    "text": resp["choices"][0]["message"].get("content") or "",
                    "timings": resp.get("timings", {}),
                    "error": None,
                }
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read()[:300]!r}"
            except Exception as e:  # timeout, connection reset, …
                last_err = repr(e)
            time.sleep(2)
        return {"text": "", "timings": {}, "error": last_err}


# ---------------------------------------------------------------- JSON output parsing

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str):
    """Extract the first JSON object from model output.

    Returns (obj, None) on success, (None, reason) on failure. Tolerates code
    fences and <think> blocks, but NOT trailing commas / single quotes — those
    count as invalid output by design.
    """
    if not text or not text.strip():
        return None, "empty"
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    start = t.find("{")
    if start == -1:
        return None, "no-object"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1]), None
                    except json.JSONDecodeError as e:
                        return None, f"json-error: {e.msg}"
    return None, "unbalanced"


# ---------------------------------------------------------------- BFCL type mapping

_TYPE_MAP = {
    "dict": "object", "float": "number", "integer": "integer", "boolean": "boolean",
    "string": "string", "array": "array", "tuple": "array", "any": None,
}


def bfcl_params_to_json_schema(params: dict) -> dict:
    """Convert BFCL's python-flavored parameter spec to standard JSON schema."""
    def conv(node):
        """Convert one schema node. Property *names* are data, not keywords."""
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                mapped = _TYPE_MAP.get(v, v)
                if mapped is None:
                    continue  # 'any': drop the type constraint entirely
                out[k] = mapped
            elif k == "properties" and isinstance(v, dict):
                # keys here are property names — recurse into the values only
                out[k] = {name: conv(sub) for name, sub in v.items()}
            elif k == "items":
                # tuple-typed 'items' may be a list of per-position schemas
                out[k] = [conv(x) for x in v] if isinstance(v, list) else conv(v)
            elif k in ("required", "enum", "default", "description"):
                out[k] = v
            # drop non-schema keys like 'optional'
        return out

    schema = conv(params)
    schema.setdefault("type", "object")
    return schema


def action_schema(tool_names: list[str], tool_schemas: dict[str, dict] | None = None) -> dict:
    """JSON schema for the uniform action format, used for constrained decoding.

    Best effort: when per-tool arg schemas are provided, ties each action name to
    its args schema via oneOf; otherwise constrains structure + action enum only.
    """
    base = {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string", "enum": tool_names},
            "args": {"type": "object"},
        },
        "required": ["thought", "action", "args"],
        "additionalProperties": False,
    }
    if not tool_schemas:
        return base
    branches = []
    for name in tool_names:
        args_schema = tool_schemas.get(name, {"type": "object"})
        branches.append({
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"const": name},
                "args": args_schema,
            },
            "required": ["thought", "action", "args"],
            "additionalProperties": False,
        })
    return {"oneOf": branches}
