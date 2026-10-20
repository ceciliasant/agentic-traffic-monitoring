"""
MCP tool server for Goose.

WORKFLOW (two-phase predict/evaluate):
baseline_predict: given lags computed up to timestamp T, it predicts the flow
at T. T is "now": the actual flow at T is not yet available when the
prediction is made (traffic for the current hour hasn't finished happening).

Each hourly session does two things, both anchored off the session's own
current_timestamp (t):
  1. EVALUATE the prediction persisted by the PREVIOUS session for hour t-1
     (get_pending_prediction + get_actual_flow + check_deviation).
  2. PREDICT for hour t (get_flows + baseline_predict), persisted to historic database, 
  to be evaluated by the NEXT session once hour t has elapsed.

Tools:
  - get_current_timestamp    : current (or simulated) session hour
  - get_evaluation_window    : t-1 (evaluate) / t (predict) split, deterministic
  - get_highway_topology     : corridor graph + detector metadata
  - get_corridor_status      : topology sequence merged with active/pending status
  - get_flows                : lag flows, football context, weather for a target hour
  - baseline_predict         : XGBoost forecast for a target hour, persists it to historic Ditto
  - get_pending_prediction   : reads a prediction persisted by an earlier session
  - get_actual_flow          : real observed flow for a given timestamp
  - check_deviation          : SMAPE-based deviation flag
  - log_run_summary / log_final_ditto / build_summary_table : reporting
"""

import os
import sys
import time
import json
import asyncio
import functools
from collections import Counter
import datetime as dt
from pathlib import Path
from typing import Annotated, Any, Literal

import boto3
import httpx
import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from APItoken import AuthenticationService

os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")

auth_service = AuthenticationService()

# MCP server
mcp = FastMCP("traffic-detector")

# MLflow
MLFLOW_EXPERIMENT_NAME = f"goose-traffic-detector-{os.getenv("GOOSE_MODEL")}"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

HISTORIC_URL      = os.getenv("VCI_HISTORIC_DITTO_URL")
DITTO_URL         = os.getenv("VCI_DITTO_URL")
TRAIN_START       = pd.Timestamp("2022-01-01 00:00:00")
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.30"))

# AWS S3
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
PREDICTION_THING_PREFIX = os.getenv("PREDICTION_THING_PREFIX", "historicVCIpredictions")
S3_PREFIX = "agent_models"

# ── Testing clock ──────────────────────────────────────────────────────────
_SIM_ANCHOR: pd.Timestamp | None = (
    pd.Timestamp(os.getenv("SIMULATION_START"))
    if os.getenv("SIMULATION_START")
    else None
)
_WALL_ANCHOR: pd.Timestamp = pd.Timestamp.now("UTC")

def _now() -> pd.Timestamp:
    if _SIM_ANCHOR is None:
        return pd.Timestamp.now("UTC").floor("h")
    elapsed = pd.Timestamp.now("UTC") - _WALL_ANCHOR
    return (_SIM_ANCHOR + elapsed).floor("h")

# ── Holidays ──────────────────────────────────────────────────────────────────
HOLIDAYS = [
    dt.date(2024, 1, 1), dt.date(2024, 2, 13), dt.date(2024, 3, 29), dt.date(2024, 3, 31),
    dt.date(2024, 4, 25), dt.date(2024, 5, 1), dt.date(2024, 6, 10), dt.date(2024, 5, 30),
    dt.date(2024, 6, 24), dt.date(2024, 8, 15), dt.date(2024, 10, 5), dt.date(2024, 11, 1),
    dt.date(2024, 12, 1), dt.date(2024, 12, 8), dt.date(2024, 12, 25),
    dt.date(2026, 1, 1), dt.date(2026, 2, 17), dt.date(2026, 4, 3), dt.date(2026, 4, 5),
    dt.date(2026, 4, 25), dt.date(2026, 5, 1), dt.date(2026, 6, 10), dt.date(2026, 6, 4),
    dt.date(2026, 6, 24), dt.date(2026, 8, 15), dt.date(2026, 10, 5), dt.date(2026, 11, 1),
    dt.date(2026, 12, 1), dt.date(2026, 12, 8), dt.date(2026, 12, 25),
]

# ── Scaled features ───────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "t", "day", "year", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "is_weekend", "is_holiday",
    "lag_1", "lag_24", "lag_168", "Avg_Temperature", "Avg_Precipitation",
]
SCALED_COLS = ["t", "lag_1", "lag_24", "lag_168", "Avg_Temperature", "Avg_Precipitation"]

# ── Corridor topology ────────────────────────────────────────────────────────────
DETECTORS = {
    "741":  {"road": "A28", "km": 1.050},
    "742":  {"road": "A28", "km": 1.850},
    "723":  {"road": "A20", "km": 15.850},
    "722":  {"road": "A20", "km": 13.890},
    "3":    {"road": "A20", "km": 12.980},
    "716":  {"road": "A20", "km": 12.325},
    "719":  {"road": "A20", "km": 11.600},
    "718":  {"road": "A20", "km": 10.150},
    "1236": {"road": "A20", "km": 8.860},
}

ROAD_TOPOLOGY = {
    "network": "VCI",
    "detectors": DETECTORS,
    "corridors": {
        "arrabida-freixo": {
            "sequence": ["741_1.0", "742_1.0", "723_-1.0", "722_-1.0", "3_-1.0", "716_-1.0", "719_-1.0", "718_-1.0", "1236_-1.0"],
            "graph": {
                "741_1.0": ["742_1.0"], "742_1.0": ["723_-1.0"], "723_-1.0": ["722_-1.0"],
                "722_-1.0": ["3_-1.0"], "3_-1.0": ["716_-1.0"], "716_-1.0": ["719_-1.0"],
                "719_-1.0": ["718_-1.0"], "718_-1.0": ["1236_-1.0"], "1236_-1.0": [],
            }
        },
        "freixo-arrabida": {
            "sequence": ["1236_1.0", "718_1.0", "719_1.0", "716_1.0", "3_1.0", "722_1.0", "723_1.0", "742_-1.0", "741_-1.0"],
            "graph": {
                "1236_1.0": ["718_1.0"], "718_1.0": ["719_1.0"], "719_1.0": ["716_1.0"],
                "716_1.0": ["3_1.0"], "3_1.0": ["722_1.0"], "722_1.0": ["723_1.0"],
                "723_1.0": ["742_-1.0"], "742_-1.0": ["741_-1.0"], "741_-1.0": [],
            }
        },
    },
}

# ── Parameter types ───────────────────────────────────────────────────────────
SensorId = Annotated[str, "Detector ID as a string, e.g. '741' or '1236'."]
Direction = Annotated[str, "Direction value = the suffix of the node ID: '1.0' or '-1.0'."]
DirectionLabel = Literal["arrabida-freixo", "freixo-arrabida"]
IsoTimestamp = Annotated[str, "Hour-aligned ISO-8601 timestamp, e.g. '2026-09-28T10:00:00'."]

# ── Tool annotations ──────────────────────────────────────────────────────────
LOCAL_READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
REMOTE_READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)
REMOTE_WRITE_IDEMPOTENT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True)
REMOTE_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True)

# ── Model cache ───────────────────────────────────────────────────────────────
_cache: dict[str, dict] = {}
_s3_client = None

def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3", aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION, endpoint_url=AWS_ENDPOINT_URL,
        )
    return _s3_client

def _download_artefacts(sensor_id: str, direction: str) -> Path:
    key = f"{sensor_id}_{direction}"
    s3_prefix = f"{S3_PREFIX}/{key}"
    local_dir = Path("/tmp/model_cache") / key
    local_dir.mkdir(parents=True, exist_ok=True)
    bucket = AWS_BUCKET_NAME
    needed = ["model.ubj", "feat_scaler.pkl", "tgt_scaler.pkl"]
    s3 = _get_s3()
    for filename in needed:
        local_path = local_dir / filename
        if not local_path.exists():
            s3_key = f"{s3_prefix}/{filename}"
            try:
                s3.download_file(bucket, s3_key, str(local_path))
            except Exception as e:
                raise FileNotFoundError(f"Could not download s3://{bucket}/{s3_key}: {e}")
    return local_dir

def _load_artefacts(sensor_id: str, direction: str) -> dict:
    key = f"{sensor_id}_{direction}"
    if key not in _cache:
        local_dir = _download_artefacts(sensor_id, direction)
        booster = xgb.Booster({"nthread": 4})
        booster.load_model(str(local_dir / "model.ubj"))
        _cache[key] = {
            "booster": booster,
            "feat_scaler": joblib.load(local_dir / "feat_scaler.pkl"),
            "tgt_scaler": joblib.load(local_dir / "tgt_scaler.pkl"),
        }
    return _cache[key]


# ── Ditto/Database helpers ─────────────────────────────────────────────────────────────
def _base_url(is_simulation: bool) -> str:
    url = HISTORIC_URL if is_simulation else DITTO_URL
    if not url:
        env = "VCI_HISTORIC_DITTO_URL" if is_simulation else "VCI_DITTO_URL"
        raise ToolError(f"{env} is not configured on the MCP server.")
    return url

def _prediction_thing_id(sensor_id: str, direction: str) -> str:
    return f"{PREDICTION_THING_PREFIX}:{sensor_id}_{float(direction)}"

def _flow_thing_id(sensor_id: str, direction: str) -> str:
    return f"historicFlowData:{sensor_id}_{float(direction)}"

async def _get_detector_ids() -> set[str]:
    url = f"{HISTORIC_URL}things"
    headers = {"Authorization": f"Bearer {auth_service.get_token()}"}
    params = {"thing_id_prefix": "historicFlowData"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
    return {thing.split(":")[-1].split("_")[0] for thing in data}

async def _get_flow_time(client, url, headers, time, direction) -> float | None:
    params = {"since": time.isoformat(), "until": time.isoformat()}
    try:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        for item in resp.json():
            props = item.get("value", {}).get("properties", {})
            if float(props.get("sentido")) == float(direction):
                return float(props.get("flows", 0))
    except Exception as e:
        print(f"Error getting flow for {time}: {e}", file=sys.stderr, flush=True)
    return None

async def _get_football_time(client, url, headers, target_time) -> str | None:
    params = {"since": str(target_time), "until": str(target_time + pd.Timedelta(days=1))}
    try:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        for item in resp.json():
            ts = item.get("value", {}).get("properties", {}).get("timestamp")
            if ts:
                return str(ts)
    except Exception as e:
        print(f"Error getting football time: {e}", file=sys.stderr, flush=True)
    return None

async def _get_weather(client, url, headers, target_time) -> dict:
    day_start = target_time.normalize()
    params = {"since": str(day_start), "until": str(day_start)}
    weather = {"temperature": None, "precipitation": None}
    try:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        for item in resp.json():
            props = item.get("value", {}).get("properties", {})
            if "temperature" in props or "precipitation" in props:
                weather["temperature"] = float(props["temperature"]) if props.get("temperature") is not None else None
                weather["precipitation"] = float(props["precipitation"]) if props.get("precipitation") is not None else None
                break
    except Exception as e:
        print(f"Error getting weather: {e}", file=sys.stderr, flush=True)
    return weather


# ── MLflow helpers ────────────────────────────────────────────────────────────
def _run_tags(sensor_id: str, direction: str | None = None) -> dict:
    meta = DETECTORS.get(sensor_id, {})
    tags = {"sensor_id": sensor_id, "road": meta.get("road", "unknown"), "km": str(meta.get("km", ""))}
    if direction is not None:
        tags["direction"] = str(direction)
    return tags

def _log_tool_call(tool_name, sensor_id, direction, inputs, outputs, metrics=None) -> None:
    """MLflow logging."""
    run_name = f"{tool_name}/{sensor_id}" + (f"_{direction}" if direction else "")
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({**_run_tags(sensor_id, direction), "tool": tool_name})
            mlflow.log_params({k: str(v)[:250] for k, v in inputs.items()})
            mlflow.log_dict(outputs, "output.json")
            if metrics:
                mlflow.log_metrics(metrics)
    except Exception as e:
        print(f"MLflow logging failed for {run_name}: {e}", file=sys.stderr, flush=True)


# ── Prediction helpers ────────────────────────────────────────────────────────
def _preprocess_scale(data: dict, feat_scaler) -> pd.DataFrame:
    df1 = pd.DataFrame([data])
    df1["timestamp"] = pd.to_datetime(df1["timestamp"])
    df1 = df1.set_index("timestamp").sort_index()
    df1["t"] = ((df1.index - TRAIN_START).total_seconds() / 3600).astype(int)
    df1["hour"], df1["day"], df1["month"], df1["year"] = df1.index.hour, df1.index.day, df1.index.month, df1.index.year
    df1["weekday"] = df1.index.weekday
    df1["hour_sin"] = np.sin(2 * np.pi * df1["hour"] / 24)
    df1["hour_cos"] = np.cos(2 * np.pi * df1["hour"] / 24)
    df1["weekday_sin"] = np.sin(2 * np.pi * df1["weekday"] / 7)
    df1["weekday_cos"] = np.cos(2 * np.pi * df1["weekday"] / 7)
    df1["month_sin"] = np.sin(2 * np.pi * df1["month"] / 12)
    df1["month_cos"] = np.cos(2 * np.pi * df1["month"] / 12)
    df1["is_weekend"] = (df1.index.weekday >= 5).astype(int)
    df1["is_holiday"] = df1.index.normalize().isin(pd.to_datetime(HOLIDAYS)).astype(int)
    df1["lag_1"], df1["lag_24"], df1["lag_168"] = df1["flows_1"], df1["flows_24"], df1["flows_168"]
    df1 = df1.rename(columns={"temperature": "Avg_Temperature", "precipitation": "Avg_Precipitation"})
    X = df1[FEATURE_COLS].copy()
    X_scaled = X.copy()
    X_scaled[SCALED_COLS] = feat_scaler.transform(X[SCALED_COLS])
    return X_scaled

def _compute_prediction(sensor_id: str, direction: str, data: dict) -> tuple[str, float]:
    """S3 download, model load, prediction"""
    art = _load_artefacts(sensor_id, direction)
    X_scaled = _preprocess_scale(data, art["feat_scaler"])

    if X_scaled.isna().any().any():
        raise ValueError(f"NaN features: {X_scaled.columns[X_scaled.isna().any()].tolist()}")
   
    ts = X_scaled.index[0]
    pred_scaled = float(art["booster"].predict(xgb.DMatrix(X_scaled))[0])
    pred = float(art["tgt_scaler"].inverse_transform([[pred_scaled]])[0][0])
    return ts.isoformat(), round(pred, 2)


async def _persist_prediction(sensor_id: str, direction: str, predict_timestamp: str, prediction: float) -> None:
    """Post prediction to historic database."""
    headers = {"Authorization": f"Bearer {auth_service.get_token()}", "Content-Type": "application/json"}
    value = {
        "type": "baseline_prediction",
        "sensor_id": sensor_id,
        "sentido": float(direction),
        "predict_timestamp": predict_timestamp,
        "baseline_prediction": prediction,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
    }
    payload = {"events": [{
        "time": predict_timestamp,
        "thing_id": _prediction_thing_id(sensor_id, direction),
        "action": "modified",
        "revision": 1,
        "path": "properties",
        "value": value,
    }]}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{_base_url(True)}events", headers=headers, json=payload)
        resp.raise_for_status()


# ══ MCP Tools ═════════════════════════════════════════════════════════════════

# ── Context / discovery ───────────────────────────────────────────────────────
# ── Call tracing (who called what, and repeats/retries) ───────────────────────
TRACE_FILE = Path(os.getenv("TRACE_FILE", "/app/output/trace.jsonl"))
_call_counts: Counter = Counter()


def _trace(role: str, tool: str, phase: str, args: dict, extra: str = "") -> None:
    key = (tool, json.dumps(args, sort_keys=True, default=str))
    if phase == "CALL":
        _call_counts[key] += 1
    n = _call_counts[key]
    repeat = f" REPEAT#{n}" if n > 1 and phase == "CALL" else ""
    node = f"{args.get('sensor_id') or args.get('det_id') or ''}_{args.get('direction') or ''}".strip("_")
    line = f"[{role}] {tool} {phase}{repeat} {node} {extra}".rstrip()
    print(line, file=sys.stderr, flush=True)
    try:
        with TRACE_FILE.open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "tool": tool,
                                "phase": phase, "repeat": n, "args": args, "extra": extra}, default=str) + "\n")
    except Exception:
        pass


def _summ(result) -> str:
    if isinstance(result, dict):
        return f"status={result.get('status', 'ok')}"
    return "ok"


def traced(role: str):
    """Logs every tool call/result/error, tagged with the calling role."""
    def deco(fn):
        name = fn.__name__
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*a, **k):
                _trace(role, name, "CALL", k)
                try:
                    r = await fn(*a, **k)
                except Exception as e:
                    _trace(role, name, "ERROR", k, str(e)[:200])
                    raise
                _trace(role, name, "RESULT", k, _summ(r))
                return r
        else:
            @functools.wraps(fn)
            def wrapper(*a, **k):
                _trace(role, name, "CALL", k)
                try:
                    r = fn(*a, **k)
                except Exception as e:
                    _trace(role, name, "ERROR", k, str(e)[:200])
                    raise
                _trace(role, name, "RESULT", k, _summ(r))
                return r
        return wrapper
    return deco


@mcp.tool(tags={"context"}, annotations=LOCAL_READ)
@traced("COORD")
def get_current_timestamp() -> dict[str, Any]:
    """Returns the session's current hour (real clock, or the simulated clock in simulation mode)."""
    return {"current_timestamp": _now().isoformat(), "is_simulation": _SIM_ANCHOR is not None}


@mcp.tool(tags={"context"}, annotations=LOCAL_READ)
@traced("COORD")
def get_evaluation_window(
    current_timestamp: Annotated[str, "The `current_timestamp` returned by get_current_timestamp."],
) -> dict[str, Any]:
    """Splits the session's current_timestamp into the hour to EVALUATE
    (the previous hour, its prediction was persisted by an earlier session
    and its actual flow should now be available) and the hour to PREDICT for
    (the current hour, to be evaluated by the NEXT session once it has
    elapsed).

    `evaluation_expected` is false when the evaluate hour predates the replay
    start (TEST_START): no prediction can exist, so skip evaluator dispatch.
    """
    t = pd.Timestamp(current_timestamp)
    evaluate_ts = t - pd.Timedelta(hours=1)
    # Replay mode: nothing was predicted before TEST_START, so the first
    # session has nothing to evaluate and the coordinator must skip evaluators.
    test_start = os.getenv("TEST_START")
    evaluation_expected = not (test_start and evaluate_ts < pd.Timestamp(test_start))
    return {
        "evaluate_timestamp": evaluate_ts.isoformat(),
        "predict_timestamp": t.isoformat(),
        "evaluation_expected": evaluation_expected,
    }


@mcp.tool(tags={"context"}, annotations=LOCAL_READ)
@traced("COORD")
def get_highway_topology() -> dict[str, Any]:
    """Returns the VCI network: detector metadata (road, km) and, per corridor,
    the ordered node sequence plus the upstream→downstream adjacency graph."""
    return ROAD_TOPOLOGY


@mcp.tool(tags={"context"}, annotations=REMOTE_READ, timeout=30)
@traced("COORD")
async def get_corridor_status(
    direction_label: Annotated[DirectionLabel, "Corridor to inspect."],
) -> dict[str, Any]:
    """Returns the node list for a corridor, each node marked
    `active` (detector exists in historic Ditto) or `pending` (no sensor yet)."""
    try:
        active_ids = await _get_detector_ids()
    except Exception as e:
        raise ToolError(f"Could not list detectors from historic Ditto: {e}") from e
    sequence = ROAD_TOPOLOGY["corridors"][direction_label]["sequence"]
    return {
        "direction_label": direction_label,
        "nodes": [{"node": n, "status": "active" if n.split("_")[0] in active_ids else "pending"} for n in sequence],
    }


# ── Predict phase ─────────────────────────────────────────────────────────────
@mcp.tool(tags={"predict"}, annotations=REMOTE_READ, timeout=30)
@traced("PREDICTOR")
async def get_flows(
    sensor_id: SensorId,
    direction: Direction,
    timestamp: Annotated[str, "The hour to predict (the `predict_timestamp`). Lags are computed relative to it."],
) -> dict[str, Any]:
    """Fetches the lag flows (t-1, t-24, t-168 hours), football-game context and
    daily weather for one sensor/direction at the target hour. A lag that could
    not be fetched is null. Pass the result unchanged to baseline_predict."""
    
    url_flows = f"{_base_url(True)}events/{_flow_thing_id(sensor_id, direction)}"
    url_football = f"{_base_url(True)}events/footballGames:EstadioDragao"
    url_weather = f"{_base_url(True)}events/historicWeatherData:dailyAverage"
    headers = {"Authorization": f"Bearer {auth_service.get_token()}"}
    time_0 = pd.Timestamp(timestamp)
    time_dict = {
        "flows_1": time_0 - pd.Timedelta(hours=1),
        "flows_24": time_0 - pd.Timedelta(hours=24),
        "flows_168": time_0 - pd.Timedelta(hours=168),
    }
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = {key: _get_flow_time(client, url_flows, headers, t, direction) for key, t in time_dict.items()}
        tasks["game_time"] = _get_football_time(client, url_football, headers, time_0)
        tasks["weather"] = _get_weather(client, url_weather, headers, time_0)
        results = await asyncio.gather(*tasks.values())
        flows_data = dict(zip(tasks.keys(), results))

    result = {
        "sensor_id": sensor_id, "direction": float(direction), "timestamp": timestamp,
        "flows_168": flows_data["flows_168"], "flows_24": flows_data["flows_24"], "flows_1": flows_data["flows_1"],
        "game_time": flows_data["game_time"],
        "temperature": flows_data["weather"]["temperature"], 
        "precipitation": flows_data["weather"]["precipitation"],
    }
    await asyncio.to_thread(
        _log_tool_call, "get_flows", sensor_id, direction,
        {"sensor_id": sensor_id, "direction": direction, "timestamp": timestamp}, result,
    )
    return result


@mcp.tool(tags={"predict"}, annotations=REMOTE_WRITE_IDEMPOTENT, timeout=90)
@traced("PREDICTOR")
async def baseline_predict(
    sensor_id: SensorId,
    direction: Direction,
    flows_payload: Annotated[
        dict[str, Any] | str,
        (
            "Pass the COMPLETE JSON object returned by get_flows, "
            "The payload MUST include: timestamp, flows_1, flows_24, "
            "flows_168, temperature, precipitation, and game_time. "
        )]  
    ) -> dict[str, Any]:
    """Predicts the flow at the hour in flows_payload's timestamp (a same-hour
    nowcast, using lags computed up to that hour) and PERSISTS the prediction
    to historic Ditto. A later session evaluates it via get_pending_prediction
    once that hour has elapsed. Re-running the same hour overwrites (latest wins).

    Returns status "logged" on success. If any lag is null, returns status
    "missing_lags" with an `error` message and nothing is persisted (expected
    in the first 168 hours of a simulation). Raises if the prediction cannot be
    computed or persisted, in that case it is NOT logged.
    """
    if isinstance(flows_payload, str):
        try:
            data = json.loads(flows_payload)
        except json.JSONDecodeError as e:
            raise ToolError(f"flows_payload is not valid JSON: {e}") from e
    else:
        data = flows_payload

    required_fields = (
        "timestamp",
        "flows_1",
        "flows_24",
        "flows_168",
        "temperature",
        "precipitation",
    )

    missing_fields = [
        key for key in required_fields
        if key not in data
    ]

    if missing_fields:
        raise ToolError(
            f"Incomplete flows_payload for {sensor_id}_{direction}: "
            f"missing fields {missing_fields}. "
            "Pass the complete result of get_flows."
        )
    
    missing = [k for k in ("flows_1", "flows_24", "flows_168") if data.get(k) is None]
    if missing:
        return {
            "status": "missing_lags",
            "error": f"Missing lag values: {missing}. Cannot predict without full history.",
            "det_id": sensor_id,
        }

    try:
        predict_timestamp, prediction = await asyncio.to_thread(_compute_prediction, sensor_id, direction, data)
    except Exception as e:
        raise ToolError(f"Prediction failed for {sensor_id}_{direction}: {e}") from e

    try:
        await _persist_prediction(sensor_id, direction, predict_timestamp, prediction)
    except Exception as e:
        raise ToolError(f"Prediction computed but could not be persisted to historic Ditto: {e}") from e

    return {
        "prediction_key": f"{sensor_id}|{direction}|{predict_timestamp}",
        "det_id": sensor_id,
        "direction": direction,
        "predict_timestamp": predict_timestamp,
        "baseline_prediction": prediction,
        "status": "logged",
    }


# ── Evaluate phase ────────────────────────────────────────────────────────────
@mcp.tool(tags={"evaluate"}, annotations=REMOTE_READ, timeout=30)
@traced("EVALUATOR")
async def get_pending_prediction(
    sensor_id: SensorId,
    direction: Direction,
    target_timestamp: Annotated[str, "The hour to evaluate (the `evaluate_timestamp` from get_evaluation_window)."],
) -> dict[str, Any]:
    """Looks up the prediction persisted in historic Ditto by an EARLIER session
    for this sensor/direction/target hour, so it can be evaluated now that the
    hour has elapsed.

    Returns status "pending_evaluation" with `baseline_prediction` if found.
    If nothing was persisted, the expected first-cycle case, or after a session
    that failed before baseline_predict, returns status "not_found" with a
    `message`; treat that as "pending", not a failure. Raises only if
    Ditto itself cannot be queried.
    """
    ts = pd.Timestamp(target_timestamp).isoformat()
    not_found = {
        "status": "not_found",
        "message": "No prior prediction persisted for this hour (expected on the first cycle). "
                   "Return status 'pending'. Do NOT retry and do NOT call any other tool.",
        "det_id": sensor_id,
        "target_timestamp": target_timestamp,
    }
    url = f"{_base_url(True)}events/{_prediction_thing_id(sensor_id, direction)}"
    headers = {"Authorization": f"Bearer {auth_service.get_token()}"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers, params={"since": ts, "until": ts})
            if resp.status_code == 404:
                return not_found
            resp.raise_for_status()
            items = resp.json()
    except Exception as e:
        raise ToolError(f"Prediction lookup failed: {e}") from e

    match = None
    for item in items:
        value = item.get("value", {}) or {}
        props = value.get("properties", value)  # tolerate both event shapes
        try:
            if float(props.get("sentido")) != float(direction):
                continue
        except (TypeError, ValueError):
            continue
        if props.get("baseline_prediction") is not None:
            match = props  # last one wins (e.g. a re-run of the same hour)

    if match is None:
        return not_found

    return {
        "prediction_key": f"{sensor_id}|{direction}|{target_timestamp}",
        "det_id": sensor_id,
        "direction": direction,
        "predict_timestamp": target_timestamp,
        "baseline_prediction": float(match["baseline_prediction"]),
        "status": "pending_evaluation",
    }


@mcp.tool(tags={"evaluate"}, annotations=REMOTE_READ, timeout=30)
@traced("EVALUATOR")
async def get_actual_flow(
    predict_timestamp: Annotated[str, "The hour whose observed flow to fetch (the `evaluate_timestamp`)."],
    sensor_id: SensorId,
    direction: Direction,
    is_simulation: Annotated[bool, "Read from historic Ditto (simulation) instead of live Ditto. Leave false unless told otherwise."] = False,
) -> dict[str, Any]:
    """Fetches the real observed flow for one sensor/direction at a given hour.

    Returns `actual_flow` if a reading exists. If the hour has no reading yet
    (sensor outage, Ditto lag) returns status "not_found" with an `error`
    message, treat that as "pending". Raises if Ditto cannot be queried.
    """
    time = pd.Timestamp(predict_timestamp).isoformat()
    url = f"{_base_url(is_simulation or _SIM_ANCHOR is not None)}events/{_flow_thing_id(sensor_id, direction)}"
    headers = {"Authorization": f"Bearer {auth_service.get_token()}"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers, params={"since": time, "until": time})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise ToolError(f"Could not fetch actual flow for {sensor_id}_{direction} at {time}: {e}") from e

    for item in data:
        props = item.get("value", {}).get("properties", {})
        try:
            same_direction = float(props.get("sentido")) == float(direction)
        except (TypeError, ValueError):
            continue
        if same_direction:
            actual = float(props.get("flows", 0))
            result = {"det_id": sensor_id, "direction": direction, "target_timestamp": time, "actual_flow": actual}
            await asyncio.to_thread(
                _log_tool_call, "get_actual_flow", sensor_id, direction,
                {"sensor_id": sensor_id, "direction": direction, "target_timestamp": time},
                result, {"actual_flow": actual},
            )
            return result

    return {
        "status": "not_found",
        "error": "no reading found for that hour",
        "det_id": sensor_id,
        "target_timestamp": time,
    }


@mcp.tool(tags={"evaluate"}, annotations=LOCAL_READ, timeout=30)
@traced("EVALUATOR")
def check_deviation(
    det_id: Annotated[str, "'<sensor_id>_<direction>', e.g. '741_1.0'."],
    baseline_prediction: Annotated[float, "The persisted prediction, from get_pending_prediction."],
    actual_flow: Annotated[float, "The observed flow, from get_actual_flow."],
    threshold: Annotated[float | None, "SMAPE threshold as a fraction (0.30 = 30%). Defaults to the server's ANOMALY_THRESHOLD."] = None,
) -> dict[str, Any]:
    """Scores a prediction against the actual flow using SMAPE (as a fraction, 0–2)
    and flags a deviation when it exceeds the threshold."""
    thr = threshold if threshold is not None else ANOMALY_THRESHOLD
    denominator = (abs(actual_flow) + abs(baseline_prediction)) / 2 + 1e-16
    #2.0 * np.abs(y_hat - y) / (np.abs(y) + np.abs(y_hat) + eps)
    smape = float(abs(baseline_prediction - actual_flow) / denominator)
    is_deviation = smape > thr
    verdict = (
        f"DEVIATION - predicted={baseline_prediction:.1f}, actual={actual_flow:.1f}, smape={smape*100:.1f}% > threshold={thr*100:.0f}%"
        if is_deviation else
        f"OK - predicted={baseline_prediction:.1f}, actual={actual_flow:.1f}, smape={smape*100:.1f}%"
    )
    sensor_id = det_id.split("_")[0] if "_" in det_id else det_id
    direction = det_id.split("_")[1] if "_" in det_id else None
    _log_tool_call("check_deviation", sensor_id, direction,
                   {"baseline_prediction": baseline_prediction, "actual_flow": actual_flow, "threshold": thr},
                   {"det_id": det_id, "smape": smape, "is_deviation": is_deviation, "verdict": verdict},
                   metrics={"baseline_prediction": baseline_prediction, "actual_flow": actual_flow,
                            "smape": smape, "is_deviation": float(is_deviation)})
    return {
        "det_id": det_id, "baseline_prediction": baseline_prediction, "actual_flow": actual_flow,
        "smape": round(smape, 4), "is_deviation": is_deviation, "threshold": thr, "verdict": verdict,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────
@mcp.tool(tags={"report"}, annotations=LOCAL_READ, timeout=30)
@traced("COORD")
def build_summary_table(
    direction_label: Annotated[DirectionLabel, "Corridor the table is for."],
    corridor_nodes: Annotated[list[dict[str, Any]], "The `nodes` list from get_corridor_status: [{'node': '741_1.0', 'status': 'active'}, ...]."],
    results_by_node: Annotated[dict[str, dict[str, Any]], "Evaluator payloads keyed by node ID (evaluations only, never predictions)."],
    final_verdicts: Annotated[dict[str, str], "Final per-node verdicts keyed by node ID, after corridor validation."],
) -> str:
    """Renders the per-corridor markdown-style summary table deterministically."""
    rows = ["Node/Detector | Road  | km | Direction | Direction Label | Prediction (veh/h) | Actual Flow (veh/h) | SMAPE | Final Verdict | Notes"]

    with open("/tmp/summary_debug.log", "a") as f:
        f.write(
            f"direction={direction_label}, "
            f"results_keys={list(results_by_node.keys())}\n"
        )

    for entry in corridor_nodes:
        node_id = entry["node"]
        sensor_id, direction = node_id.split("_")
        km = DETECTORS.get(sensor_id, {}).get("km", "unknown")
        road = DETECTORS.get(sensor_id, {}).get("road", "unknown")
        r = results_by_node.get(node_id)

        with open("/tmp/summary_debug.log", "a") as f:
            f.write(
                f"direction={direction_label}, "
                f"results_keys={list(results_by_node.keys())}\n"
            )

        if r is None:
            rows.append(
                f"{node_id} | {road} | {km} | {direction} | "
                f"{direction_label} | - | - | - | pending | No evaluation result supplied"
            )
            continue

        verdict = final_verdicts.get(node_id, "pending")

        smape = r.get("smape", r.get("SMAPE", "N/A"))

        notes = r.get("notes", "")
        if r.get("status") == "pending":
            notes = notes or "Evaluation pending"

        rows.append(
            f"{node_id} | {road} | {km} | {direction} | "
            f"{direction_label} | "
            f"{r.get('baseline_prediction', 'N/A')} | "
            f"{r.get('actual_flow', 'N/A')} | "
            f"{smape} | {verdict} | {notes}"
        )

    table = "\n".join(rows)
    try:
        with mlflow.start_run(run_name=f"summary_table/{direction_label}"):
            mlflow.log_text(table, f"summary_table_{direction_label}.txt")
    except Exception as e:
        print(f"MLflow logging failed for summary_table_{direction_label}: {e}", file=sys.stderr)
    return table


@mcp.tool(tags={"report"}, annotations=REMOTE_WRITE, timeout=30)
@traced("COORD")
def log_run_summary(
    timestamp: Annotated[str, "The evaluated hour (T-1), ISO-8601."],
    sensors_attempted: Annotated[int, "Active nodes dispatched for evaluation."],
    sensors_completed: Annotated[int, "Nodes with status 'evaluated'."],
    sensors_missing_lags: Annotated[int, "Nodes that could not be scored for missing data."],
    sensors_errored: Annotated[int, "Nodes whose subagent errored."],
    deviations_flagged: Annotated[int, "Nodes flagged as deviating before corridor validation."],
    deviations_overridden_as_drift: Annotated[int, "Flags overridden to macro_drift by corridor validation."],
    corridor_verdict: Annotated[str, "Overall verdict: within_baseline, macro_drift, localized_event, expected_conditions_football, expected_conditions_rain, mixed, or pending."],
    mean_smape: Annotated[float | None, "Mean SMAPE across evaluated nodes; null if none were evaluated."] = None,
    max_smape: Annotated[float | None, "Max SMAPE across evaluated nodes; null if none were evaluated."] = None,
    max_smape_detector: Annotated[str | None, "Node ID with the max SMAPE; null if none."] = None,
    notes: Annotated[str | None, "Optional free-text notes."] = None,
) -> dict[str, Any]:
    """Logs the T-1 evaluation summary for this session to MLflow."""
    coverage = sensors_completed / sensors_attempted if sensors_attempted > 0 else 0.0
    net_deviations = deviations_flagged - deviations_overridden_as_drift
    metrics = {
        "sensors_attempted": float(sensors_attempted), "sensors_completed": float(sensors_completed),
        "sensors_missing_lags": float(sensors_missing_lags), "sensors_errored": float(sensors_errored),
        "coverage": round(coverage, 4), "deviations_flagged": float(deviations_flagged),
        "deviations_overridden_as_drift": float(deviations_overridden_as_drift),
        "net_deviations": float(net_deviations),
    }
    if mean_smape is not None:
        metrics["mean_smape"] = round(mean_smape, 4)
    if max_smape is not None:
        metrics["max_smape"] = round(max_smape, 4)
    try:
        with mlflow.start_run(run_name=f"summary/{timestamp}"):
            mlflow.set_tags({"type": "run_summary", "timestamp": timestamp, "corridor_verdict": corridor_verdict,
                             "is_simulation": str(_SIM_ANCHOR is not None)})
            mlflow.log_params({"timestamp": timestamp, "corridor_verdict": corridor_verdict,
                               "max_smape_detector": max_smape_detector or "", "notes": notes or ""})
            mlflow.log_metrics(metrics)
            run_id = mlflow.active_run().info.run_id
    except Exception as e:
        raise ToolError(f"Could not log run summary to MLflow: {e}") from e
    return {"logged": True, "run_id": run_id, "timestamp": timestamp, "coverage": round(coverage, 4),
            "net_deviations": net_deviations, "corridor_verdict": corridor_verdict}


@mcp.tool(tags={"report"}, annotations=REMOTE_WRITE, timeout=30)
@traced("COORD")
async def log_final_ditto(
    timestamp: Annotated[str, "The evaluated hour (T-1), ISO-8601."],
    corridor_verdict: Annotated[str, "Overall verdict: within_baseline, macro_drift, localized_event, expected_conditions_football, expected_conditions_rain, mixed, or pending."],
    sensors_completed: Annotated[int, "Nodes with status 'evaluated'."],
    sensors_attempted: Annotated[int, "Active nodes dispatched for evaluation."],
    deviations_flagged: Annotated[int, "Nodes flagged as deviating before corridor validation."],
    deviations_overridden_as_drift: Annotated[int, "Flags overridden to macro_drift by corridor validation."],
    mean_smape: Annotated[float | None, "Mean SMAPE across evaluated nodes; null if none were evaluated."] = None,
    max_smape: Annotated[float | None, "Max SMAPE across evaluated nodes; null if none were evaluated."] = None,
    max_smape_detector: Annotated[str | None, "Node ID with the max SMAPE; null if none."] = None,
    notes: Annotated[str | None, "Optional free-text notes."] = None,
    is_simulation: Annotated[bool, "Write to historic Ditto (simulation) instead of live Ditto. Leave false unless told otherwise."] = False,
) -> dict[str, Any]:
    """Publishes the T-1 assessment to Ditto (historic in simulation, live otherwise)."""
    is_simulation = is_simulation or _SIM_ANCHOR is not None
    target_url = _base_url(is_simulation)
    thing_id = "historicVCIcoordinator:agentAssessment" if is_simulation else "hourlyVCIcoordinator:agentAssessment"
    headers = {"Authorization": f"Bearer {auth_service.get_token()}", "Content-Type": "application/json"}

    value = {
        "timestamp": timestamp, "corridor_verdict": corridor_verdict,
        "mean_smape": round(mean_smape, 4) if mean_smape is not None else None,
        "max_smape": round(max_smape, 4) if max_smape is not None else None,
        "max_smape_detector": max_smape_detector, "sensors_completed": sensors_completed,
        "sensors_attempted": sensors_attempted,
        "coverage": round(sensors_completed / sensors_attempted, 4) if sensors_attempted > 0 else 0.0,
        "deviations_flagged": deviations_flagged, "deviations_overridden_as_drift": deviations_overridden_as_drift,
        "net_deviations": deviations_flagged - deviations_overridden_as_drift,
        "notes": notes or "", "is_simulation": is_simulation,
    }
    if is_simulation:
        payload = {"events": [{"time": timestamp, "thing_id": thing_id, "action": "modified",
                                "revision": 1, "path": "properties", "value": value}]}
        url = f"{target_url}events"
    else:
        payload = {"properties": value}
        url = f"{target_url}things/{thing_id}/features/hourly_assessment/properties"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
    except Exception as e:
        raise ToolError(f"Could not publish assessment to {thing_id}: {e}") from e
    return {"success": True, "thing_id": thing_id, "target_url": target_url, "is_simulation": is_simulation}


if __name__ == "__main__":
    mcp.run()