"""Stage runner: gates, caching, live status, and the final report.

Every stage returns a gate dict with at least ``passed`` (bool). A failed
gate stops the run (unless ``stop_on_failed_gate`` is false) and the report
names it. Nothing downstream ever runs on the output of a failed stage, and
there are no silent fallbacks: a stage either does its job or says why not.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.core.io import read_json, stable_hash, write_json


class GateFailure(RuntimeError):
    """Raised inside a stage for a precondition it cannot satisfy."""


@dataclass
class RunContext:
    run_dir: str
    cfg: Dict[str, Any]
    decisions: List[str] = field(default_factory=list)

    def stage_dir(self, name: str) -> str:
        d = os.path.join(self.run_dir, name)
        os.makedirs(d, exist_ok=True)
        return d

    def path(self, *parts: str) -> str:
        return os.path.join(self.run_dir, *parts)

    def gate(self, name: str) -> Dict[str, Any]:
        p = self.path(name, "gate.json")
        if not os.path.exists(p):
            raise GateFailure(f"upstream stage '{name}' has not run in {self.run_dir}")
        g = read_json(p)
        if not g.get("passed", False):
            raise GateFailure(f"upstream stage '{name}' failed its gate")
        return g

    def decide(self, text: str) -> None:
        print(f"[decision] {text}")
        self.decisions.append(text)


@dataclass
class Stage:
    name: str
    fn: Callable[[RunContext], Dict[str, Any]]
    config_keys: List[str]
    requires: List[str] = field(default_factory=list)
    optional: bool = False  # a disabled optional stage writes a passing "skipped" gate


def _cache_key(stage: Stage, ctx: RunContext, upstream_keys: Dict[str, str]) -> str:
    cfg_slice = {k: ctx.cfg.get(k) for k in stage.config_keys}
    return stable_hash({"stage": stage.name, "cfg": cfg_slice, "inputs": ctx.cfg.get("inputs"),
                        "up": {r: upstream_keys.get(r) for r in stage.requires}})


class Pipeline:
    def __init__(self, stages: List[Stage]):
        self.stages = stages
        self.names = [s.name for s in stages]

    def run(self, ctx: RunContext, until: Optional[str] = None, start: Optional[str] = None,
            force: bool = False) -> Dict[str, Any]:
        os.makedirs(ctx.run_dir, exist_ok=True)
        write_json(ctx.path("run_config.json"), ctx.cfg)
        stop_idx = self.names.index(until) if until else len(self.stages) - 1
        start_idx = self.names.index(start) if start else 0
        report: Dict[str, Any] = {"run_dir": ctx.run_dir, "status": "RUNNING", "stages": {},
                                  "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        upstream_keys: Dict[str, str] = {}
        t_total = time.perf_counter()
        failed_at = None

        for i, stage in enumerate(self.stages[: stop_idx + 1]):
            sdir = ctx.stage_dir(stage.name)
            key = _cache_key(stage, ctx, upstream_keys)
            upstream_keys[stage.name] = key
            done_path = os.path.join(sdir, "stage.done")
            gate_path = os.path.join(sdir, "gate.json")
            if start is not None:
                # --from X: everything before X is reused as-is, X and later always rerun
                cached = i < start_idx
            else:
                cached = (not force and os.path.exists(done_path) and os.path.exists(gate_path)
                          and open(done_path, encoding="utf-8").read().strip() == key)
            if cached and os.path.exists(gate_path):
                gate = read_json(gate_path)
                gate.setdefault("cached", True)
                report["stages"][stage.name] = gate
                print(f"[pipeline] {stage.name}: cached ({'pass' if gate.get('passed') else 'FAIL'})")
                if not gate.get("passed") and ctx.cfg.get("stop_on_failed_gate", True):
                    failed_at = stage.name
                    break
                continue

            self._status(ctx, stage.name, i, "running", t_total)
            print(f"\n[pipeline] ===== {stage.name} =====")
            t0 = time.perf_counter()
            try:
                gate = stage.fn(ctx) or {}
            except GateFailure as exc:
                gate = {"passed": False, "error": str(exc)}
            except Exception as exc:  # a crash is a failed gate with a traceback
                gate = {"passed": False, "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()}
            gate.setdefault("passed", False)
            gate["seconds"] = round(time.perf_counter() - t0, 2)
            write_json(gate_path, gate)
            report["stages"][stage.name] = gate
            verdict = "pass" if gate["passed"] else "FAIL"
            print(f"[pipeline] {stage.name}: {verdict} in {gate['seconds']} s")
            if gate["passed"]:
                with open(done_path, "w", encoding="utf-8") as f:
                    f.write(key)
            else:
                if os.path.exists(done_path):
                    os.remove(done_path)
                for reason in gate.get("failures", []):
                    print(f"    - {reason}")
                if gate.get("error"):
                    print(f"    - {gate['error']}")
                if ctx.cfg.get("stop_on_failed_gate", True):
                    failed_at = stage.name
                    break

        ran = list(report["stages"].keys())
        all_pass = all(report["stages"][n].get("passed") for n in ran)
        if failed_at:
            report["status"] = "FAILED"
            report["failed_stage"] = failed_at
        elif all_pass and len(ran) == stop_idx + 1:
            report["status"] = "SUCCESS" if stop_idx == len(self.stages) - 1 else "PARTIAL_SUCCESS"
        else:
            report["status"] = "PARTIAL"
        report["until"] = self.names[stop_idx]
        report["decisions"] = ctx.decisions
        report["total_seconds"] = round(time.perf_counter() - t_total, 2)
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_json(ctx.path("report.json"), report)
        self._status(ctx, failed_at or self.names[stop_idx], stop_idx, report["status"].lower(), t_total)
        print(f"\n[pipeline] status {report['status']} ({report['total_seconds']} s) -> {ctx.path('report.json')}")
        return report

    def _status(self, ctx: RunContext, stage: str, idx: int, state: str, t0: float) -> None:
        write_json(ctx.path("status.json"), {
            "stage": stage, "stage_index": idx, "num_stages": len(self.stages), "state": state,
            "percent": round(100.0 * idx / max(1, len(self.stages)), 1),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        })


def gate_result(checks: Dict[str, bool], metrics: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    """Build a gate dict from named boolean checks plus metrics."""
    failures = [name for name, ok in checks.items() if not ok]
    out = {"passed": not failures, "checks": checks, "failures": failures, "metrics": metrics}
    out.update(extra)
    return out
