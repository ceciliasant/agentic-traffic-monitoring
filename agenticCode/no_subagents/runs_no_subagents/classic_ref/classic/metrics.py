#!/usr/bin/env python3
"""
metrics.py - run-level metrics for one model's agent replay.

Inputs (all in the output volume, written during the agent run):
  timing.csv       model,hour,start_epoch,end_epoch,seconds,goose_exit   (entrypoint.sh)
  llm_calls.jsonl  one line per LLM request                              (llm_proxy.py)
Optional:
  comparison/per_hour.csv   (compare.py) for tool errors / coverage per session

Every LLM request is one agent turn, so requests per session are the turn count
to compare with GOOSE_MAX_TURNS.

  python metrics.py [--model gemma-4-26b-a4b] [--out-dir /app/output/comparison]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PCTS = [0.5, 0.9, 0.95, 0.99]


def dist(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    out = {"n": int(s.count()), "mean": s.mean(), "std": s.std(), "min": s.min()}
    out.update({f"p{int(q * 100)}": s.quantile(q) for q in PCTS})
    out["max"] = s.max()
    return {k: round(float(v), 2) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/app/output")
    ap.add_argument("--out-dir", default="/app/output/comparison")
    ap.add_argument("--model", default=None, help="filter timing.csv to this model")
    ap.add_argument("--context-limit", type=int, default=24000)
    ap.add_argument("--max-turns", type=int, default=150)
    ap.add_argument("--start"), ap.add_argument("--end")
    a = ap.parse_args()
    root, out = Path(a.output), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    timing = pd.read_csv(root / "timing.csv")
    if a.model:
        timing = timing[timing.model == a.model]
    timing = timing.drop_duplicates("hour", keep="last").set_index("hour")
    calls = pd.DataFrame(json.loads(l) for l in (root / "llm_calls.jsonl").read_text().splitlines() if l.strip())
    calls = calls[calls.hour.isin(timing.index)]
    if a.start:
        calls = calls[calls.hour >= a.start]
    if a.end:
        calls = calls[calls.hour <= a.end]

    for c in ("prompt_tokens", "completion_tokens", "summarize_hint", "finish_reason", "status", "error",
              "retries", "queue_wait_s"):
        if c not in calls:
            calls[c] = None
    calls["ok"] = calls.status.between(200, 299)
    calls["ctx_pct"] = calls.prompt_tokens / a.context_limit * 100
    calls["summ"] = calls.summarize_hint.fillna(False).astype(bool)
    calls["truncated"] = calls.finish_reason.eq("length")
    calls["failed"] = ~calls.ok | calls.error.notna()

    g = calls.groupby("hour")
    s = pd.DataFrame({
        "llm_calls": g.size(),
        "input_tokens": g.prompt_tokens.sum(min_count=1),
        "output_tokens": g.completion_tokens.sum(min_count=1),
        "max_prompt_tokens": g.prompt_tokens.max(),
        "max_ctx_pct": g.ctx_pct.max(),
        "summarize_calls": g.summ.sum(),
        "truncated_outputs": g.truncated.sum(),
        "llm_failures": g.failed.sum(),
        "mean_llm_latency_s": g.latency_s.mean(),
    })
    s["wall_s"] = timing.seconds
    s["goose_exit"] = timing.goose_exit
    pf = root / "comparison" / "per_hour.csv"
    if pf.exists():
        ph = pd.read_csv(pf).drop_duplicates("hour", keep="last").set_index("hour")
        s = s.join(ph[["tool_errors", "coverage_vs_classic", "log_run_summary_ok"]])
    s = s.reindex(timing.index)
    s.to_csv(out / "metrics_per_hour.csv")

    nonzero_exit = pd.to_numeric(s.goose_exit, errors="coerce").fillna(-1).ne(0)
    hit_turns = s.llm_calls >= a.max_turns
    M = {
        "sessions": int(len(s)),
        "wall_time_s": dist(s.wall_s),
        "wall_time_total_h": round(float(s.wall_s.sum()) / 3600, 2),
        "llm_calls_per_session (turns)": dist(s.llm_calls),
        f"turns_vs_max_{a.max_turns}": {
            "mean_share_of_max": round(float(s.llm_calls.mean() / a.max_turns), 3),
            "sessions_at_or_over_max": int(hit_turns.sum()),
        },
        "input_tokens_per_session": dist(s.input_tokens),
        "output_tokens_per_session": dist(s.output_tokens),
        "total_input_tokens": int(s.input_tokens.sum()),
        "total_output_tokens": int(s.output_tokens.sum()),
        f"max_prompt_vs_context_{a.context_limit}": {
            "pct_of_limit_per_session": dist(s.max_ctx_pct),
            "sessions_over_80pct": int((s.max_ctx_pct > 80).sum()),
            "sessions_over_100pct": int((s.max_ctx_pct > 100).sum()),
        },
        "summarize": {
            "calls_flagged_as_summarize (heuristic)": int(calls.summ.sum()),
            "sessions_with_summarize": int((s.summarize_calls > 0).sum()),
        },
        "failures": {
            "sessions_goose_nonzero_exit": int(nonzero_exit.sum()),
            "sessions_at_turn_limit": int(hit_turns.sum()),
            "llm_requests_failed_or_timed_out": int(calls.failed.sum()),
            "llm_requests_retried_after_429_503": int(calls.retries.fillna(0).gt(0).sum()),
            "llm_retries_total": int(calls.retries.fillna(0).sum()),
            "max_queue_wait_s": round(float(calls.queue_wait_s.max()), 1) if calls.queue_wait_s.notna().any() else None,
            "total_queue_wait_s": round(float(calls.queue_wait_s.sum()), 1) if calls.queue_wait_s.notna().any() else None,
            "outputs_truncated_finish_reason_length": int(calls.truncated.sum()),
            "sessions_with_truncated_output": int((s.truncated_outputs > 0).sum()),
            "output_tokens_in_truncated_requests": int(calls.loc[calls.truncated, "completion_tokens"].sum()),
            "share_of_all_output_tokens_in_truncated": round(float(
                calls.loc[calls.truncated, "completion_tokens"].sum() / max(calls.completion_tokens.sum(), 1)), 3),
            "longest_single_request_s": round(float(calls.latency_s.max()), 1),
            "requests_over_300s": int((calls.latency_s > 300).sum()),
            "prompt_tokens_over_context_limit_requests": int((calls.prompt_tokens > a.context_limit).sum()),
            "sessions_with_tool_errors": int((s.get("tool_errors", pd.Series(dtype=float)) > 0).sum()),
        },
        "usage_reported_by_server_share": round(float((calls.usage_source == "server").mean()), 3),
    }
    (out / "metrics.json").write_text(json.dumps(M, indent=2, default=str))
    lines = ["# Run metrics", ""]
    for k, v in M.items():
        if isinstance(v, dict):
            lines += [f"## {k}", ""] + [f"- {kk}: {vv}" for kk, vv in v.items()] + [""]
        else:
            lines.append(f"- **{k}**: {v}")
    (out / "metrics.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
