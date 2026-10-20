#!/usr/bin/env python3
"""
classic_runner.py - the deterministic (no-LLM) pipeline.

Per hour T (same two jobs as the agent session):
  1. get_evaluation_window(T), get_corridor_status() for both directions
  2. PREDICT   every active node: get_flows -> baseline_predict
  3. EVALUATE  every active node (if evaluation_expected):
               get_pending_prediction -> get_actual_flow -> check_deviation
  4. Verdicts  with the same rules as coordinator_recipe.yaml
  5. log_run_summary (own MLflow experiment) and one JSON file per hour

Isolation (nothing in mcp_server.py is changed):
  - predictions are written under their own Ditto thing prefix
    (default historicVCIpredictionsClassic) so they never overwrite the agent's
  - MLflow experiment is "goose-traffic-detector-classic"
  - tool trace goes to <out-dir>/trace_classic.jsonl, not the agent's trace
  - log_final_ditto is NOT called: it writes to one fixed Ditto thing and would
    overwrite the agent's assessment. The summary is written to JSON instead.

Run inside coordinator container (same env vars as the agent):
  python classic_runner.py                       # TEST_START..TEST_END + final hour
  python classic_runner.py --hour 2024-07-21T16:00:00
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

CORRIDORS = ["arrabida-freixo", "freixo-arrabida"]
LAG_KEYS = ("flows_1", "flows_24", "flows_168")
DEFAULT_PREFIX = "historicVCIpredictionsClassic"


# --------------------------------------------------------------------------- #
# Server loading and tool calling (shared with compare.py)
# --------------------------------------------------------------------------- #
def load_server(*, experiment_tag: str, trace_file: str | Path, prediction_prefix: str | None = None):
    """Import mcp_server with its side effects redirected, and return the module.

    GOOSE_MODEL only feeds the MLflow experiment name; TRACE_FILE is read at
    import time. Both must be set before the import. The prediction prefix is a
    module global read at call time, so it can be overridden afterwards.
    """
    os.environ["GOOSE_MODEL"] = experiment_tag
    os.environ["TRACE_FILE"] = str(trace_file)
    Path(trace_file).parent.mkdir(parents=True, exist_ok=True)
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mcp_server  # noqa: WPS433 (import inside function on purpose)

    if prediction_prefix:
        mcp_server.PREDICTION_THING_PREFIX = prediction_prefix
    return mcp_server


def unwrap(res: Any) -> Any:
    """Turn a FastMCP CallToolResult into plain Python data."""
    data = getattr(res, "structured_content", None)
    if isinstance(data, dict):
        if set(data) == {"result"}:
            return data["result"]
        return data
    for part in getattr(res, "content", None) or []:
        text = getattr(part, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return getattr(res, "data", None)


class Tools:
    """Calls MCP tools in-process (same schema validation the agent goes through)."""

    def __init__(self, client):
        self.client = client
        self.calls = 0
        self.errors: list[dict] = []

    async def call(self, name: str, **args) -> tuple[Any, str | None]:
        self.calls += 1
        try:
            res = await self.client.call_tool(name, args)
        except Exception as exc:  # ToolError, validation error, transport error
            err = f"{type(exc).__name__}: {exc}"
            self.errors.append({"tool": name, "args": _short(args), "error": err})
            return None, err
        if getattr(res, "is_error", False):
            err = str(unwrap(res))
            self.errors.append({"tool": name, "args": _short(args), "error": err})
            return None, err
        return unwrap(res), None


def _short(args: dict) -> dict:
    return {k: (v if not isinstance(v, (dict, list)) else "...") for k, v in args.items()}


# --------------------------------------------------------------------------- #
# Verdict rules (identical to coordinator_recipe.yaml step 6)
# --------------------------------------------------------------------------- #
def node_verdicts(nodes: list[dict], evaluations: dict[str, dict]) -> dict[str, str]:
    """nodes: ordered [{"node","status"}]; evaluations: node -> evaluation record.

    A flagged node (is_deviation true) is part of a run if its neighbours in the
    ordered nodes list are flagged too. Any non-flagged node, including pending
    or unevaluated ones, ends a run. Runs of 3+ are macro_drift.
    """
    ids = [n["node"] for n in nodes]
    flagged = [
        bool(evaluations.get(i, {}).get("status") == "evaluated" and evaluations[i].get("is_deviation"))
        for i in ids
    ]
    run_len = [0] * len(ids)
    i = 0
    while i < len(ids):
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < len(ids) and flagged[j]:
            j += 1
        for k in range(i, j):
            run_len[k] = j - i
        i = j

    out = {}
    for idx, node in enumerate(ids):
        ev = evaluations.get(node, {})
        if ev.get("status") != "evaluated":
            out[node] = "pending"
        elif not flagged[idx]:
            out[node] = "within_baseline"
        elif run_len[idx] >= 3:
            out[node] = "macro_drift"
        else:
            out[node] = "localized_event"
    return out


def rollup(verdicts: list[str]) -> str:
    for v in ("macro_drift", "localized_event", "within_baseline"):
        if v in verdicts:
            return v
    return "pending"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
async def predict_node(tools: Tools, sem: asyncio.Semaphore, node: str, hour: str) -> dict:
    sensor_id, direction = node.split("_")
    rec: dict[str, Any] = {"node": node, "flows": None, "lags_missing": None,
                           "status": None, "baseline_prediction": None, "error": None}
    async with sem:
        flows, err = await tools.call("get_flows", sensor_id=sensor_id, direction=direction, timestamp=hour)
        if err or not isinstance(flows, dict):
            rec.update(status="error", error=err or "get_flows returned no data")
            return rec
        rec["flows"] = flows
        rec["lags_missing"] = any(flows.get(k) is None for k in LAG_KEYS)
        res, err = await tools.call("baseline_predict", sensor_id=sensor_id, direction=direction,
                                    flows_payload=flows)
        if err or not isinstance(res, dict):
            rec.update(status="error", error=err or "baseline_predict returned no data")
            return rec
        rec["status"] = res.get("status")
        rec["baseline_prediction"] = res.get("baseline_prediction")
        rec["error"] = res.get("error")
    return rec


async def evaluate_node(tools: Tools, sem: asyncio.Semaphore, node: str, evaluate_ts: str,
                        is_simulation: bool) -> dict:
    sensor_id, direction = node.split("_")
    rec: dict[str, Any] = {"node": node, "status": "pending", "reason": None,
                           "baseline_prediction": None, "actual_flow": None,
                           "smape": None, "is_deviation": None}
    async with sem:
        pred, err = await tools.call("get_pending_prediction", sensor_id=sensor_id, direction=direction,
                                     target_timestamp=evaluate_ts)
        if err or not isinstance(pred, dict) or pred.get("status") != "pending_evaluation":
            rec["reason"] = err or (pred or {}).get("status", "no_prediction")
            return rec
        rec["baseline_prediction"] = pred["baseline_prediction"]

        actual, err = await tools.call("get_actual_flow", predict_timestamp=evaluate_ts, sensor_id=sensor_id,
                                       direction=direction, is_simulation=is_simulation)
        if err or not isinstance(actual, dict) or actual.get("actual_flow") is None:
            rec["reason"] = err or (actual or {}).get("status", "no_actual_flow")
            return rec
        rec["actual_flow"] = actual["actual_flow"]

        dev, err = await tools.call("check_deviation", det_id=node,
                                    baseline_prediction=rec["baseline_prediction"],
                                    actual_flow=rec["actual_flow"])
        if err or not isinstance(dev, dict):
            rec["reason"] = err or "check_deviation returned no data"
            return rec
        rec.update(status="evaluated", smape=dev.get("smape"), is_deviation=dev.get("is_deviation"))
    return rec


async def run_hour(tools: Tools, hour: str, *, concurrency: int, is_simulation: bool) -> dict:
    t_start = time.perf_counter()
    calls0, errors0 = tools.calls, len(tools.errors)
    out: dict[str, Any] = {"hour": hour, "run_at": datetime.now().isoformat(timespec="seconds"),
                           "is_simulation": is_simulation}

    window, err = await tools.call("get_evaluation_window", current_timestamp=hour)
    if err or not isinstance(window, dict):
        out["fatal_error"] = f"get_evaluation_window failed: {err}"
        return out
    evaluate_ts = window["evaluate_timestamp"]
    out.update(evaluate_timestamp=evaluate_ts, predict_timestamp=window["predict_timestamp"],
               evaluation_expected=bool(window["evaluation_expected"]))

    nodes: dict[str, list[dict]] = {}
    for label in CORRIDORS:
        status, err = await tools.call("get_corridor_status", direction_label=label)
        nodes[label] = status["nodes"] if isinstance(status, dict) and "nodes" in status else []
        if err:
            out.setdefault("corridor_errors", {})[label] = err
    out["nodes"] = nodes
    active = [n["node"] for label in CORRIDORS for n in nodes[label] if n["status"] == "active"]

    sem = asyncio.Semaphore(concurrency)

    t0 = time.perf_counter()
    predictions = await asyncio.gather(*(predict_node(tools, sem, n, window["predict_timestamp"]) for n in active))
    out["predictions"] = {p["node"]: p for p in predictions}
    predict_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    if out["evaluation_expected"]:
        evals = await asyncio.gather(*(evaluate_node(tools, sem, n, evaluate_ts, is_simulation) for n in active))
        out["evaluations"] = {e["node"]: e for e in evals}
    else:
        out["evaluations"] = {n: {"node": n, "status": "pending", "reason": "evaluation_not_expected"}
                              for n in active}
    evaluate_s = time.perf_counter() - t0

    # verdicts (same rules as the coordinator recipe)
    verdicts: dict[str, str] = {}
    corridor_verdicts: dict[str, str] = {}
    for label in CORRIDORS:
        v = node_verdicts(nodes[label], out["evaluations"])
        verdicts.update(v)
        corridor_verdicts[label] = rollup(list(v.values()))
    overall = rollup(list(verdicts.values()))
    out.update(verdicts=verdicts, corridor_verdicts=corridor_verdicts, overall_verdict=overall)

    evaluated = [e for e in out["evaluations"].values() if e["status"] == "evaluated"]
    smapes = [(e["smape"], e["node"]) for e in evaluated if e["smape"] is not None]
    flagged = [n for n, e in out["evaluations"].items() if e["status"] == "evaluated" and e["is_deviation"]]
    summary = {
        "timestamp": evaluate_ts,
        "sensors_attempted": len(active) if out["evaluation_expected"] else 0,
        "sensors_completed": len(evaluated),
        "sensors_missing_lags": (len(active) - len(evaluated)) if out["evaluation_expected"] else 0,
        "sensors_errored": 0,
        "deviations_flagged": len(flagged),
        "deviations_overridden_as_drift": sum(1 for n in flagged if verdicts.get(n) == "macro_drift"),
        "corridor_verdict": overall,
        "mean_smape": round(sum(s for s, _ in smapes) / len(smapes), 4) if smapes else None,
        "max_smape": max(smapes)[0] if smapes else None,
        "max_smape_detector": max(smapes)[1] if smapes else None,
    }
    out["summary"] = summary
    res, err = await tools.call("log_run_summary", **{k: v for k, v in summary.items() if v is not None})
    out["log_run_summary"] = {"ok": err is None, "error": err}

    out["counts"] = {
        "active_nodes": len(active),
        "predictions_logged": sum(1 for p in predictions if p["status"] == "logged"),
        "predictions_failed": sum(1 for p in predictions if p["status"] != "logged"),
        "predictions_lags_missing": sum(1 for p in predictions if p["lags_missing"]),
        "evaluated": len(evaluated),
    }
    out["timing"] = {"predict_s": round(predict_s, 2), "evaluate_s": round(evaluate_s, 2),
                     "total_s": round(time.perf_counter() - t_start, 2), "tool_calls": tools.calls - calls0}
    out["tool_errors"] = tools.errors[errors0:]
    return out


def hours_to_run(start: str, end: str, include_final: bool) -> list[str]:
    cur, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
    hours = []
    while cur <= last:
        hours.append(cur.isoformat())
        cur += timedelta(hours=1)
    if include_final:  # same extra session the entrypoint runs: evaluates TEST_END
        hours.append(cur.isoformat())
    return hours


async def run_all(srv, hours: list[str], out_dir: Path, concurrency: int, is_simulation: bool) -> None:
    from fastmcp import Client

    out_dir.mkdir(parents=True, exist_ok=True)
    async with Client(srv.mcp) as client:
        tools = Tools(client)
        for hour in hours:
            rec = await run_hour(tools, hour, concurrency=concurrency, is_simulation=is_simulation)
            fname = out_dir / f"hour_{hour.replace(':', '-')}.json"
            fname.write_text(json.dumps(rec, indent=2, default=str))
            with (out_dir / "classic_results.jsonl").open("a") as f:
                f.write(json.dumps({k: rec.get(k) for k in ("hour", "overall_verdict", "counts", "timing")},
                                   default=str) + "\n")
            c, t = rec.get("counts", {}), rec.get("timing", {})
            print(f"[classic] {hour}  verdict={rec.get('overall_verdict')}  "
                  f"logged={c.get('predictions_logged')}/{c.get('active_nodes')}  "
                  f"evaluated={c.get('evaluated')}  calls={t.get('tool_calls')}  {t.get('total_s')}s",
                  flush=True)
            if rec.get("fatal_error"):
                print(f"[classic] FATAL {rec['fatal_error']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic (no-LLM) twin of the agentic pipeline.")
    ap.add_argument("--start", default=os.getenv("TEST_START"), help="first prediction hour (default $TEST_START)")
    ap.add_argument("--end", default=os.getenv("TEST_END"), help="last prediction hour (default $TEST_END)")
    ap.add_argument("--hour", help="run a single hour instead of start..end")
    ap.add_argument("--no-final", action="store_true", help="skip the extra final evaluation session")
    ap.add_argument("--out-dir", default=os.getenv("CLASSIC_OUT_DIR", "/app/output/classic"))
    ap.add_argument("--prefix", default=os.getenv("CLASSIC_PREDICTION_PREFIX", DEFAULT_PREFIX),
                    help="Ditto thing prefix for the classic predictions (must differ from the agent's)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--live", action="store_true",
                    help="read actual flows from live Ditto instead of historic (production mode)")
    args = ap.parse_args()

    if args.hour:
        hours = [args.hour]
    else:
        if not (args.start and args.end):
            ap.error("give --start and --end (or set TEST_START and TEST_END), or use --hour")
        hours = hours_to_run(args.start, args.end, include_final=not args.no_final)
    if args.start:
        os.environ["TEST_START"] = args.start  # read by get_evaluation_window at call time

    out_dir = Path(args.out_dir)
    srv = load_server(experiment_tag="classic", trace_file=out_dir / "trace_classic.jsonl",
                      prediction_prefix=args.prefix)
    if srv.PREDICTION_THING_PREFIX == "historicVCIpredictions":
        sys.exit("refusing to run: classic prefix equals the agent's prefix")
    print(f"[classic] hours={len(hours)} prefix={srv.PREDICTION_THING_PREFIX} out={out_dir}", flush=True)
    asyncio.run(run_all(srv, hours, out_dir, args.concurrency, is_simulation=not args.live))


if __name__ == "__main__":
    main()
