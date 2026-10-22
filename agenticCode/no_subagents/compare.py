#!/usr/bin/env python3
"""
compare.py - agentic pipeline vs classic (deterministic) pipeline.

Inputs
  --classic-dir   one hour_*.json per hour written by classic_runner.py
  --trace         the agent's tool trace (/app/output/trace.jsonl, written by mcp_server.py)
  --logs-dir      (optional) the agent's goose logs (/app/logs/<hour>.log)
  agent predictions are read back from Ditto through the same get_pending_prediction
  tool the agent uses (cached in <out-dir>/agent_predictions.json)

What is compared (the classic run is the reference for what the pipeline SHOULD do,
because both use the same models and the same data; actual flows are the ground truth
for forecast accuracy):

  1. Orchestration fidelity
       - coverage: nodes the agent logged a prediction for vs nodes classic logged
       - input fidelity: the flows_payload the agent passed to baseline_predict vs the
         real get_flows output (wrong value / wrong hour / cross-wired node / missing field)
       - prediction fidelity: agent's stored prediction vs classic's, per node and hour
       - evaluation fidelity: baseline/actual the agent passed to check_deviation vs the
         stored prediction and the observed flow
  2. Forecast accuracy: SMAPE of both predictions against the observed flow
  3. Judgment layer: per-node and overall verdict agreement (the rules are a reference,
     not ground truth), also restricted to nodes whose inputs were correct
  4. Failure modes and cost: tool errors, repeated calls, calls to nodes that should not
     be touched, rate-limit and parameter errors, coordinator calling subagent tools
     directly (from the goose logs), tool calls and wall time per hour

Everything is written to --out-dir: per_hour.csv, per_node.csv, summary.json, summary.md
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from mlflow import store

FLOW_FIELDS = ("flows_1", "flows_24", "flows_168", "temperature", "precipitation")
PIPELINE_TOOLS = ("get_flows", "baseline_predict", "get_pending_prediction", "get_actual_flow", "check_deviation")
CORRIDORS = ("arrabida-freixo", "freixo-arrabida")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def norm_ts(v: Any) -> str | None:
    if v is None:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "")).replace(tzinfo=None).isoformat()
    except ValueError:
        return str(v)


def node_key(sensor_id: Any, direction: Any) -> str:
    try:
        return f"{sensor_id}_{float(direction)}"
    except (TypeError, ValueError):
        return f"{sensor_id}_{direction}"


def norm_node_id(raw: Any) -> str:
    s = str(raw).strip().strip("'\"")
    if "_" not in s:
        return s
    sid, d = s.split("_", 1)
    return node_key(sid, d)


def close(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def smape(pred: float | None, actual: float | None) -> float | None:
    if pred is None or actual is None:
        return None
    return abs(pred - actual) / ((abs(actual) + abs(pred)) / 2 + 1e-16)


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_classic(directory: Path) -> dict[str, dict]:
    recs = {}
    for p in sorted(directory.glob("hour_*.json")):
        r = json.loads(p.read_text())
        if r.get("hour") and not r.get("fatal_error"):
            recs[norm_ts(r["hour"])] = r
    return recs


def load_sessions(trace_path: Path) -> list[dict]:
    """Split the agent trace into sessions.

    The server process restarts for every `goose run`, so its call counter restarts
    too: a get_current_timestamp CALL with repeat==1 marks a new session. The hour T
    of a session is the argument of its first get_evaluation_window call.
    """
    sessions: list[dict] = []
    cur: dict | None = None
    for line in trace_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        tool, phase = e.get("tool"), e.get("phase")
        if tool == "get_current_timestamp" and phase == "CALL" and e.get("repeat", 1) == 1:
            cur = {"T": None, "events": []}
            sessions.append(cur)
        if cur is None:
            if tool == "get_evaluation_window" and phase == "CALL":
                cur = {"T": None, "events": []}
                sessions.append(cur)
            else:
                continue
        cur["events"].append(e)
        if tool == "get_evaluation_window" and phase == "CALL" and cur["T"] is None:
            cur["T"] = norm_ts((e.get("args") or {}).get("current_timestamp"))
    return sessions


def pair_calls(events: list[dict]) -> list[dict]:
    """Match each CALL with its RESULT/ERROR (same tool and arguments, in order)."""
    waiting: dict[tuple, list[dict]] = defaultdict(list)
    calls: list[dict] = []
    for e in events:
        args = e.get("args") or {}
        key = (e.get("tool"), json.dumps(args, sort_keys=True, default=str))
        if e.get("phase") == "CALL":
            rec = {"tool": e["tool"], "args": args, "repeat": e.get("repeat", 1), "ts": e.get("ts"),
                   "result": None, "extra": ""}
            waiting[key].append(rec)
            calls.append(rec)
        elif waiting[key]:
            rec = waiting[key].pop(0)
            rec["result"] = "error" if e.get("phase") == "ERROR" else "ok"
            rec["extra"] = e.get("extra", "") or ""
    return calls


def result_status(call: dict) -> str | None:
    m = re.search(r"status=(\S+)", call.get("extra") or "")
    return m.group(1) if m else None


def scan_logs(directory: Path | None) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    if not directory or not directory.is_dir():
        return stats
    tools = "|".join(PIPELINE_TOOLS)
    for p in directory.glob("*.log"):
        if "T" not in p.stem:
            continue
        d, t = p.stem.split("T", 1)
        hour = norm_ts(f"{d}T{t.replace('-', ':')}")
        text = p.read_text(errors="ignore")
        stats[hour] = {
            "log_rate_limit_errors": text.count("Rate limit exceeded"),
            "log_param_errors": text.count("Missing required parameters"),
            "log_delegate_calls": len(re.findall(r"▸\s+delegate\b", text)),
            # tool lines without the "[subagent:N]" prefix are the coordinator calling the tool itself
            "log_coordinator_direct_tool_calls": len(re.findall(rf"▸\s+(?:{tools})\b", text)),
            "log_subagent_tool_calls": len(re.findall(rf"▸\s+\[subagent:\d+\]\s+(?:{tools})\b", text)),
            "log_subagent_load_misuse": len(re.findall(r"▸\s+\[subagent:\d+\]\s+load\b", text)),
        }
    return stats


# --------------------------------------------------------------------------- #
# agent predictions from Ditto (read back through the real tool)
# --------------------------------------------------------------------------- #
async def fetch_agent_store(pairs: list[tuple[str, str]], prefix: str | None, concurrency: int) -> dict[str, float | None]:
    from fastmcp import Client

    from classic_runner import Tools, load_server

    srv = load_server(experiment_tag="compare", trace_file="/tmp/compare_trace.jsonl", prediction_prefix=prefix)
    sem = asyncio.Semaphore(concurrency)
    store: dict[str, float | None] = {}
    async with Client(srv.mcp) as client:
        tools = Tools(client)

        async def one(node: str, ts: str) -> None:
            sid, d = node.split("_")
            async with sem:
                res, err = await tools.call("get_pending_prediction", sensor_id=sid, direction=d, target_timestamp=ts)
            ok = isinstance(res, dict) and res.get("status") == "pending_evaluation"
            store[f"{node}|{ts}"] = res["baseline_prediction"] if ok else None
            if err:
                print(f"[compare] fetch error {node} {ts}: {err}", file=sys.stderr)

        await asyncio.gather(*(one(n, t) for n, t in pairs))
    return store


# --------------------------------------------------------------------------- #
# analysis of one agent session
# --------------------------------------------------------------------------- #
def payload_issues(args: dict, flows_true: dict | None, hour: str) -> list[str]:
    if flows_true is None:
        return ["no_reference"]
    payload = args.get("flows_payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return ["invalid_payload"]
    if not isinstance(payload, dict):
        return ["invalid_payload"]
    issues = []
    if "sensor_id" in payload and str(payload["sensor_id"]) != str(args.get("sensor_id")):
        issues.append("cross_node_sensor")
    if "direction" in payload and not close(payload["direction"], args.get("direction")):
        issues.append("cross_node_direction")
    if norm_ts(payload.get("timestamp")) != hour:
        issues.append("wrong_timestamp")
    for k in FLOW_FIELDS:
        if k not in flows_true:
            continue
        if k not in payload:
            issues.append(f"missing_field:{k}")
        elif not close(payload[k], flows_true.get(k)):
            issues.append(f"wrong_value:{k}")
    if payload.get("game_time") != flows_true.get("game_time"):
        issues.append("wrong_value:game_time")
    return issues


def analyze_session(sess: dict, is_last: bool, ctx: dict) -> tuple[dict, list[dict], list[tuple[str, str]], list[tuple[str, str]]]:
    T = sess["T"]
    classic, store, actual, cpred, logs, tol = (ctx[k] for k in ("classic", "store", "actual", "cpred", "logs", "tol"))
    crec = classic[T]
    calls = pair_calls(sess["events"])
    nodes_by_label = crec["nodes"]
    label_of = {n["node"]: lab for lab, ns in nodes_by_label.items() for n in ns}
    active = [n["node"] for ns in nodes_by_label.values() for n in ns if n["status"] == "active"]
    active_set = set(active)
    preds = crec.get("predictions", {})
    classic_logged = {n for n in active if preds.get(n, {}).get("status") == "logged"}

    def node_of(c: dict) -> str:
        return node_key(c["args"].get("sensor_id"), c["args"].get("direction"))

    bp = [c for c in calls if c["tool"] == "baseline_predict"]
    gf = [c for c in calls if c["tool"] == "get_flows"]
    agent_logged = {node_of(c) for c in bp if c["result"] == "ok" and result_status(c) == "logged"}

    # input fidelity per node
    issues_by_node: dict[str, list[str]] = defaultdict(list)
    bad_payload_calls = 0
    for c in bp:
        n = node_of(c)
        iss = payload_issues(c["args"], (preds.get(n) or {}).get("flows"), T) if n in active_set else ["unexpected_node"]
        if iss:
            bad_payload_calls += 1
            issues_by_node[n].extend(iss)

    pipeline_calls = [c for c in calls if c["tool"] in PIPELINE_TOOLS]
    unexpected_node_calls = sum(
        1 for c in pipeline_calls if "sensor_id" in c["args"] and node_of(c) not in active_set)
    wrong_hour_get_flows = sum(1 for c in gf if norm_ts(c["args"].get("timestamp")) != T)

    row: dict[str, Any] = {
        "hour": T, "session_is_last_for_hour": is_last,
        "active_nodes": len(active), "classic_logged": len(classic_logged), "agent_logged": len(agent_logged),
        "coverage_vs_classic": rate(len(agent_logged & classic_logged), len(classic_logged)),
        "missing_nodes": " ".join(sorted(classic_logged - agent_logged)),
        "extra_nodes": " ".join(sorted(agent_logged - classic_logged)),
        "baseline_predict_calls": len(bp), "payload_bad_calls": bad_payload_calls,
        "get_flows_wrong_hour": wrong_hour_get_flows, "unexpected_node_calls": unexpected_node_calls,
    }

    # evaluation fidelity
    prev_ts = norm_ts(crec.get("evaluate_timestamp"))
    classic_eval = {n for n, e in crec.get("evaluations", {}).items() if e.get("status") == "evaluated"}
    cd = [c for c in calls if c["tool"] == "check_deviation"]
    agent_eval = {norm_node_id(c["args"].get("det_id")) for c in cd}
    eval_bad = 0
    for c in cd:
        n = norm_node_id(c["args"].get("det_id"))
        exp_p, exp_a = store.get((n, prev_ts)), actual.get((n, prev_ts))
        bad_p = exp_p is not None and not close(c["args"].get("baseline_prediction"), exp_p, tol)
        bad_a = exp_a is not None and not close(c["args"].get("actual_flow"), exp_a, tol)
        eval_bad += int(bad_p or bad_a)
    expected = bool(crec.get("evaluation_expected"))
    row.update(
        evaluation_expected=expected,
        eval_coverage_vs_classic=rate(len(agent_eval & classic_eval), len(classic_eval)) if expected else None,
        check_deviation_calls=len(cd), check_deviation_bad_args=eval_bad,
    )

    # verdicts
    fv_by_label: dict[str, dict[str, str]] = {}
    for c in calls:
        if c["tool"] == "build_summary_table" and c["result"] == "ok":
            fv = c["args"].get("final_verdicts")
            if isinstance(fv, dict):
                fv_by_label[c["args"].get("direction_label")] = {norm_node_id(k): v for k, v in fv.items()}
    lrs = [c for c in calls if c["tool"] == "log_run_summary" and c["result"] == "ok"]
    summ = crec.get("summary", {})
    a_over = lrs[-1]["args"].get("corridor_verdict") if lrs else None
    row.update(
        build_summary_table_ok=len(fv_by_label), log_run_summary_ok=bool(lrs),
        log_final_ditto_ok=any(c["tool"] == "log_final_ditto" and c["result"] == "ok" for c in calls),
        agent_overall_verdict=a_over, classic_overall_verdict=crec.get("overall_verdict"),
        overall_verdict_match=(a_over == crec.get("overall_verdict")) if lrs else False,
        summary_counts_match=bool(lrs) and all(
            lrs[-1]["args"].get(k) == summ.get(k) for k in ("sensors_completed", "deviations_flagged")),
    )

    # process / cost
    ts = [datetime.strptime(c["ts"], "%Y-%m-%dT%H:%M:%S") for c in calls if c.get("ts")]
    tool_counts = Counter(c["tool"] for c in calls)
    row.update(
        tool_calls=len(calls), tool_errors=sum(1 for c in calls if c["result"] == "error"),
        unanswered_calls=sum(1 for c in calls if c["result"] is None),
        repeated_calls=sum(1 for c in calls if c["repeat"] > 1),
        trace_duration_s=int((max(ts) - min(ts)).total_seconds()) if len(ts) > 1 else None,
        classic_tool_calls=(crec.get("timing") or {}).get("tool_calls"),
        classic_total_s=(crec.get("timing") or {}).get("total_s"),
        tool_counts=json.dumps(dict(tool_counts)),
    )
    row.update(logs.get(T, {}))

    node_rows: list[dict] = []
    verdict_pairs: list[tuple[str, str]] = []
    verdict_pairs_ok: list[tuple[str, str]] = []
    if is_last:
        for n in active:
            cp = cpred.get((n, T))
            ap = store.get((n, T)) if n in agent_logged else None
            
            act = actual.get((n, T))
            a_prev, c_prev = store.get((n, prev_ts)), cpred.get((n, prev_ts))
            prev_ok = (a_prev is None and c_prev is None) or (
                a_prev is not None and c_prev is not None and abs(a_prev - c_prev) <= tol)
            cv = crec.get("verdicts", {}).get(n)
            av = fv_by_label.get(label_of.get(n), {}).get(n)
            if cv is not None:
                verdict_pairs.append((cv, av or "not_reported"))
                if prev_ok:
                    verdict_pairs_ok.append((cv, av or "not_reported"))
            node_rows.append({
                "hour": T, "node": n, "classic_prediction": cp, "agent_prediction": ap,
                "abs_diff": round(abs(ap - cp), 4) if ap is not None and cp is not None else None,
                "pred_match": (abs(ap - cp) <= tol) if ap is not None and cp is not None else None,
                "actual_flow": act, "smape_classic": smape(cp, act), "smape_agent": smape(ap, act),
                "payload_issues": " ".join(sorted(set(issues_by_node.get(n, [])))),
                "prev_hour_prediction_matched": prev_ok,
                "verdict_classic": cv, "verdict_agent": av,
                "verdict_match": (cv == av) if cv is not None else None,
            })
    return row, node_rows, verdict_pairs, verdict_pairs_ok


# --------------------------------------------------------------------------- #
# aggregation and output
# --------------------------------------------------------------------------- #
def aggregate(hours: list[dict], nodes: list[dict], vp: list, vp_ok: list) -> dict:
    last = [r for r in hours if r["session_is_last_for_hour"]]
    both = [r for r in nodes if r["abs_diff"] is not None]
    with_actual = [r for r in nodes if r["smape_agent"] is not None and r["smape_classic"] is not None]
    diffs = [r["smape_agent"] - r["smape_classic"] for r in with_actual]
    eval_rows = [r for r in hours if r["evaluation_expected"]]
    vm = [r for r in nodes if r["verdict_match"] is not None]
    vm_ok = [r for r in vm if r["prev_hour_prediction_matched"]]
    payload_calls = sum(r["baseline_predict_calls"] for r in hours)
    cd_calls = sum(r["check_deviation_calls"] for r in hours)
    has_logs = any("log_rate_limit_errors" in r for r in hours)
    S: dict[str, Any] = {
        "sessions": {"agent_sessions": len(hours), "hours_compared": len({r["hour"] for r in hours}),
                     "repeated_sessions": len(hours) - len(last)},
        "coverage": {
            "mean_coverage_vs_classic": mean([r["coverage_vs_classic"] for r in hours]),
            "sessions_with_full_coverage": sum(1 for r in hours if r["coverage_vs_classic"] == 1.0),
            "sessions_with_missing_nodes": sum(1 for r in hours if r["missing_nodes"]),
            "sessions_with_extra_nodes": sum(1 for r in hours if r["extra_nodes"]),
        },
        "input_fidelity": {
            "baseline_predict_calls": payload_calls,
            "calls_with_wrong_payload": sum(r["payload_bad_calls"] for r in hours),
            "wrong_payload_rate": rate(sum(r["payload_bad_calls"] for r in hours), payload_calls),
            "get_flows_wrong_hour": sum(r["get_flows_wrong_hour"] for r in hours),
            "calls_to_nodes_that_should_not_be_touched": sum(r["unexpected_node_calls"] for r in hours),
        },
        "prediction_fidelity": {
            "node_hours_compared": len(both),
            "exact_match_rate": rate(sum(1 for r in both if r["pred_match"]), len(both)),
            "mean_abs_diff": mean([r["abs_diff"] for r in both]),
            "max_abs_diff": max((r["abs_diff"] for r in both), default=None),
        },
        "forecast_accuracy_vs_actual": {
            "node_hours_with_actual": len(with_actual),
            "mean_smape_classic": mean([r["smape_classic"] for r in with_actual]),
            "mean_smape_agent": mean([r["smape_agent"] for r in with_actual]),
            "mean_paired_diff_agent_minus_classic": mean(diffs),
            "share_agent_worse": rate(sum(1 for d in diffs if d > 1e-9), len(diffs)),
        },
        "evaluation_fidelity": {
            "sessions_with_evaluation": len(eval_rows),
            "mean_eval_coverage_vs_classic": mean([r["eval_coverage_vs_classic"] for r in eval_rows]),
            "check_deviation_calls": cd_calls,
            "check_deviation_calls_with_wrong_args": sum(r["check_deviation_bad_args"] for r in hours),
        },
        "verdicts": {
            "node_level_agreement": rate(sum(1 for r in vm if r["verdict_match"]), len(vm)),
            "node_level_agreement_when_inputs_correct": rate(sum(1 for r in vm_ok if r["verdict_match"]), len(vm_ok)),
            "node_level_cohen_kappa": cohen_kappa(vp),
            "overall_verdict_match_rate": rate(sum(1 for r in hours if r["overall_verdict_match"]), len(hours)),
            "summary_counts_match_rate": rate(sum(1 for r in hours if r["summary_counts_match"]), len(hours)),
            "sessions_without_log_run_summary": sum(1 for r in hours if not r["log_run_summary_ok"]),
            "sessions_without_log_final_ditto": sum(1 for r in hours if not r["log_final_ditto_ok"]),
        },
        "failure_modes_and_cost": {
            "tool_errors": sum(r["tool_errors"] for r in hours),
            "repeated_calls": sum(r["repeated_calls"] for r in hours),
            "unanswered_calls": sum(r["unanswered_calls"] for r in hours),
            "mean_tool_calls_agent": mean([r["tool_calls"] for r in hours]),
            "mean_tool_calls_classic": mean([r["classic_tool_calls"] for r in hours]),
            "mean_duration_s_agent_trace": mean([r["trace_duration_s"] for r in hours]),
            "mean_duration_s_classic": mean([r["classic_total_s"] for r in hours]),
        },
    }
    if has_logs:
        S["failure_modes_and_cost"].update({
            "rate_limit_errors": sum(r.get("log_rate_limit_errors", 0) for r in hours),
            "delegate_parameter_errors": sum(r.get("log_param_errors", 0) for r in hours),
            "sessions_where_coordinator_called_pipeline_tools_directly":
                sum(1 for r in hours if r.get("log_coordinator_direct_tool_calls", 0) > 0),
            "subagents_using_load_instead_of_tools": sum(r.get("log_subagent_load_misuse", 0) for r in hours),
            "delegate_calls": sum(r.get("log_delegate_calls", 0) for r in hours),
        })
    return S


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, S: dict) -> None:
    lines = ["# Agentic vs classic pipeline", "",
             "Reference for fidelity = classic run (same models, same data). "
             "Ground truth for accuracy = observed flows. Verdict rules are a reference, not ground truth.", ""]
    for section, values in S.items():
        lines += [f"## {section.replace('_', ' ')}", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in values.items()]
        lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Compare the agentic pipeline with the classic pipeline.")
    ap.add_argument("--classic-dir", default="/app/output/classic")
    ap.add_argument("--trace", default="/app/output/trace.jsonl")
    ap.add_argument("--logs-dir", default="/app/logs")
    ap.add_argument("--out-dir", default="/app/output/comparison")
    ap.add_argument("--agent-prefix", default=None, help="Ditto prediction prefix of the agent (default: server default)")
    ap.add_argument("--agent-predictions-file", help="cache of agent predictions (default <out-dir>/agent_predictions.json)")
    ap.add_argument("--refresh", action="store_true", help="re-read agent predictions from Ditto")
    ap.add_argument("--tol", type=float, default=0.01, help="max abs difference (veh/h) to call two predictions equal")
    ap.add_argument("--start", help="only agent sessions for hours >= this")
    ap.add_argument("--end", help="only agent sessions for hours <= this")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classic = load_classic(Path(args.classic_dir))
    if not classic:
        sys.exit(f"no classic results in {args.classic_dir}; run classic_runner.py first")
    trace = Path(args.trace)
    if not trace.exists():
        sys.exit(f"agent trace not found: {trace}")

    lo, hi = norm_ts(args.start), norm_ts(args.end)
    sessions = [s for s in load_sessions(trace) if s["T"]
                and (lo is None or s["T"] >= lo) and (hi is None or s["T"] <= hi)]
    unmatched = sorted({s["T"] for s in sessions if s["T"] not in classic})
    sessions = [s for s in sessions if s["T"] in classic]
    if not sessions:
        sys.exit("no agent sessions match the classic hours (check --trace, --start/--end)")

    # classic reference values and observed flows
    cpred: dict[tuple[str, str], float] = {}
    actual: dict[tuple[str, str], float] = {}
    for T, rec in classic.items():
        for n, p in rec.get("predictions", {}).items():
            if p.get("status") == "logged" and p.get("baseline_prediction") is not None:
                cpred[(n, T)] = p["baseline_prediction"]
        ev_ts = norm_ts(rec.get("evaluate_timestamp"))
        for n, e in rec.get("evaluations", {}).items():
            if e.get("status") == "evaluated" and e.get("actual_flow") is not None:
                actual[(n, ev_ts)] = e["actual_flow"]

    # agent predictions, read back from Ditto (cached)
    cache = Path(args.agent_predictions_file) if args.agent_predictions_file else out / "agent_predictions.json"
    raw: dict[str, float | None] = json.loads(cache.read_text()) if cache.exists() and not args.refresh else {}
    wanted = sorted({(n["node"], T) for T, rec in classic.items() for ns in rec["nodes"].values()
                     for n in ns if n["status"] == "active"})
    todo = [(n, T) for n, T in wanted if f"{n}|{T}" not in raw]
    if todo:
        print(f"[compare] reading {len(todo)} agent predictions from Ditto ...", flush=True)
        raw.update(asyncio.run(fetch_agent_store(todo, args.agent_prefix, args.concurrency)))
        cache.write_text(json.dumps(raw, indent=1, sort_keys=True))
    store = {tuple(k.split("|")): v for k, v in raw.items()}

    ctx = {"classic": classic, "store": store, "actual": actual, "cpred": cpred,
           "logs": scan_logs(Path(args.logs_dir) if args.logs_dir else None), "tol": args.tol}

    last_idx = {s["T"]: i for i, s in enumerate(sessions)}
    hour_rows, node_rows, vp, vp_ok = [], [], [], []
    seen: Counter = Counter()
    for i, s in enumerate(sessions):
        seen[s["T"]] += 1
        row, nrows, a, b = analyze_session(s, i == last_idx[s["T"]], ctx)
        row["session_number_for_hour"] = seen[s["T"]]
        hour_rows.append(row)
        node_rows += nrows
        vp += a
        vp_ok += b

    S = aggregate(hour_rows, node_rows, vp, vp_ok)
    if unmatched:
        S["sessions"]["agent_hours_without_classic_reference"] = unmatched
    write_csv(out / "per_hour.csv", hour_rows)
    write_csv(out / "per_node.csv", node_rows)
    (out / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    write_markdown(out / "summary.md", S)
    print(json.dumps(S, indent=2, default=str))
    print(f"\n[compare] wrote {out}/per_hour.csv, per_node.csv, summary.json, summary.md")


if __name__ == "__main__":
    main()
