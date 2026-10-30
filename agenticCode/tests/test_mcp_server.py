import json

import numpy as np
import pandas as pd
import pytest

from conftest import FakeResp


# ---------- check_deviation: the SMAPE rule ----------
def test_default_threshold_is_documented_value(srv):
    assert srv.ANOMALY_THRESHOLD == 0.20


def test_smape_value_and_flag(call):
    r = call(
        "check_deviation",
        det_id="741_1.0",
        baseline_prediction=100.0,
        actual_flow=130.0
    )

    assert r["smape"] == pytest.approx(30 / 115, abs=1e-4)
    assert r["is_deviation"] is True
    assert r["threshold"] == 0.20


def test_smape_edge_cases(call):
    assert call("check_deviation", det_id="3_-1.0", baseline_prediction=0.0, actual_flow=0.0)["smape"] == 0.0
    assert call("check_deviation", det_id="3_-1.0", baseline_prediction=0.0, actual_flow=50.0)["smape"] == pytest.approx(2.0)
    sym = [call("check_deviation", det_id="3_-1.0", baseline_prediction=a, actual_flow=b)["smape"] for a, b in ((80, 120), (120, 80))]
    assert sym[0] == sym[1]                                     # symmetric


# ---------- get_evaluation_window ----------
def test_evaluation_window_splits_t_minus_1(call, monkeypatch):
    monkeypatch.setenv("TEST_START", "2024-09-23T00:00:00")
    r = call("get_evaluation_window", current_timestamp="2024-09-23T10:00:00")
    assert r["evaluate_timestamp"] == "2024-09-23T09:00:00"
    assert r["predict_timestamp"] == "2024-09-23T10:00:00"
    assert r["evaluation_expected"] is True


def test_first_replay_session_has_nothing_to_evaluate(call, monkeypatch):
    monkeypatch.setenv("TEST_START", "2024-09-23T00:00:00")
    r = call("get_evaluation_window", current_timestamp="2024-09-23T00:00:00")
    assert r["evaluation_expected"] is False


# ---------- baseline_predict: input validation and persistence ----------
GOOD = {"sensor_id": "741", "direction": 1.0, "timestamp": "2024-09-23T10:00:00",
        "flows_1": 100.0, "flows_24": 110.0, "flows_168": 120.0,
        "game_time": None, "temperature": 18.0, "precipitation": 0.0}


@pytest.fixture
def predictor(srv, monkeypatch):
    persisted = []
    monkeypatch.setattr(srv, "_compute_prediction", lambda s, d, data: (pd.Timestamp(data["timestamp"]).isoformat(), 123.45))
    async def fake_persist(sensor_id, direction, ts, pred):
        persisted.append((sensor_id, direction, ts, pred))
    monkeypatch.setattr(srv, "_persist_prediction", fake_persist)
    return persisted


def test_baseline_predict_logs_and_persists(call, predictor):
    r = call("baseline_predict", sensor_id="741", direction="1.0", flows_payload=GOOD)
    assert r["status"] == "logged" and r["baseline_prediction"] == 123.45
    assert r["prediction_key"] == "741|1.0|2024-09-23T10:00:00"
    assert predictor == [("741", "1.0", "2024-09-23T10:00:00", 123.45)]


def test_baseline_predict_accepts_json_string(call, predictor):
    r = call("baseline_predict", sensor_id="741", direction="1.0", flows_payload=json.dumps(GOOD))
    assert r["status"] == "logged"


def test_baseline_predict_null_lag_persists_nothing(call, predictor):
    bad = {**GOOD, "flows_24": None}
    r = call("baseline_predict", sensor_id="741", direction="1.0", flows_payload=bad)
    assert r["status"] == "missing_lags" and predictor == []


@pytest.mark.parametrize("drop", ["timestamp", "flows_1", "flows_24", "flows_168", "temperature", "precipitation"])
def test_baseline_predict_rejects_incomplete_payload(call, predictor, drop):
    bad = {k: v for k, v in GOOD.items() if k != drop}
    with pytest.raises(Exception, match="Incomplete flows_payload"):
        call("baseline_predict", sensor_id="741", direction="1.0", flows_payload=bad)
    assert predictor == []


def test_baseline_predict_rejects_invalid_json(call, predictor):
    with pytest.raises(Exception, match="not valid JSON"):
        call("baseline_predict", sensor_id="741", direction="1.0", flows_payload="{oops")
    assert predictor == []


# ---------- get_pending_prediction ----------
def _event(sentido, pred):
    return {"value": {"properties": {"sentido": sentido, "baseline_prediction": pred}}}


def test_pending_prediction_found(call, fake_http):
    fake_http.get = FakeResp([_event(1.0, 321.0)])
    r = call("get_pending_prediction", sensor_id="741", direction="1.0", target_timestamp="2024-09-23T10:00:00")
    assert r["status"] == "pending_evaluation" and r["baseline_prediction"] == 321.0


def test_pending_prediction_not_found_cases(call, fake_http):
    args = dict(sensor_id="741", direction="1.0", target_timestamp="2024-09-23T10:00:00")
    fake_http.get = FakeResp([], status=404)
    assert call("get_pending_prediction", **args)["status"] == "not_found"
    fake_http.get = FakeResp([])
    assert call("get_pending_prediction", **args)["status"] == "not_found"
    fake_http.get = FakeResp([_event(-1.0, 99.0)])             # other direction must not match
    assert call("get_pending_prediction", **args)["status"] == "not_found"


def test_pending_prediction_latest_wins(call, fake_http):
    fake_http.get = FakeResp([_event(1.0, 10.0), _event(1.0, 20.0)])
    r = call("get_pending_prediction", sensor_id="741", direction="1.0", target_timestamp="2024-09-23T10:00:00")
    assert r["baseline_prediction"] == 20.0


def test_pending_prediction_uses_configured_prefix(srv, call, fake_http, monkeypatch):
    monkeypatch.setattr(srv, "PREDICTION_THING_PREFIX", "historicVCIpredictionsTestRun")
    fake_http.get = FakeResp([])
    call("get_pending_prediction", sensor_id="741", direction="1.0", target_timestamp="2024-09-23T10:00:00")
    assert "historicVCIpredictionsTestRun:741_1.0" in fake_http.gets[-1][0]


def test_prediction_thing_id_normalises_direction(srv):
    assert srv._prediction_thing_id("741", "1") == f"{srv.PREDICTION_THING_PREFIX}:741_1.0"
    assert srv._prediction_thing_id("3", "-1.0").endswith(":3_-1.0")


# ---------- get_actual_flow ----------
def test_actual_flow_found_and_missing(call, fake_http):
    args = dict(predict_timestamp="2024-09-23T09:00:00", sensor_id="741", direction="1.0")
    fake_http.get = FakeResp([{"value": {"properties": {"sentido": 1.0, "flows": 456}}}])
    assert call("get_actual_flow", **args)["actual_flow"] == 456.0
    fake_http.get = FakeResp([])
    assert call("get_actual_flow", **args)["status"] == "not_found"


# ---------- feature engineering (the numbers every prediction depends on) ----------
class IdentityScaler:
    def transform(self, x):
        return np.asarray(x)


def _features(srv, ts, **over):
    data = {"timestamp": ts, "flows_1": 1.0, "flows_24": 2.0, "flows_168": 3.0, "temperature": 15.0, "precipitation": 0.5, **over}
    return srv._preprocess_scale(data, IdentityScaler())


def test_feature_columns_and_order(srv):
    X = _features(srv, "2024-09-23T10:00:00")
    assert list(X.columns) == srv.FEATURE_COLS
    assert X.shape == (1, 16)


def test_feature_values(srv):
    X = _features(srv, "2024-09-23T10:00:00").iloc[0]            # a Monday
    assert X["t"] == int((pd.Timestamp("2024-09-23 10:00") - srv.TRAIN_START).total_seconds() / 3600)
    assert X["hour_sin"] == pytest.approx(np.sin(2 * np.pi * 10 / 24))
    assert X["weekday_sin"] == pytest.approx(0.0) and X["is_weekend"] == 0
    assert (X["lag_1"], X["lag_24"], X["lag_168"]) == (1.0, 2.0, 3.0)
    assert (X["Avg_Temperature"], X["Avg_Precipitation"]) == (15.0, 0.5)


def test_weekend_and_holiday_flags(srv):
    assert _features(srv, "2024-09-28T10:00:00").iloc[0]["is_weekend"] == 1      # Saturday
    assert _features(srv, "2024-10-05T10:00:00").iloc[0]["is_holiday"] == 1      # listed holiday
    assert _features(srv, "2024-09-23T10:00:00").iloc[0]["is_holiday"] == 0


def test_football_context_is_not_a_model_feature(srv):
    # get_flows fetches game_time, but the model never receives it. Keep this explicit.
    assert not any("game" in c.lower() or "football" in c.lower() for c in srv.FEATURE_COLS)


# ---------- topology ----------
def test_topology_is_consistent(srv):
    for name, c in srv.ROAD_TOPOLOGY["corridors"].items():
        seq, graph = c["sequence"], c["graph"]
        assert set(seq) == set(graph), name
        assert len(seq) == len(set(seq)), name
        for a, b in zip(seq, seq[1:]):
            assert graph[a] == [b], (name, a)
        assert graph[seq[-1]] == []
        for node in seq:
            assert node.split("_")[0] in srv.DETECTORS


# ---------- reporting ----------
def test_summary_table_marks_missing_nodes_pending(call):
    nodes = [{"node": "741_1.0", "status": "active"}, {"node": "742_1.0", "status": "active"}]
    results = {"741_1.0": {"baseline_prediction": 100.0, "actual_flow": 110.0, "smape": 0.0953}}
    table = call("build_summary_table", direction_label="arrabida-freixo", corridor_nodes=nodes,
                 results_by_node=results, final_verdicts={"741_1.0": "within_baseline"})
    rows = table.splitlines()
    assert len(rows) == 3
    assert "within_baseline" in rows[1] and "100.0" in rows[1]
    assert "pending" in rows[2] and "No evaluation result supplied" in rows[2]


def test_final_ditto_payload_shapes(call, fake_http):
    args = dict(timestamp="2024-09-23T09:00:00", corridor_verdict="within_baseline", sensors_completed=12,
                sensors_attempted=12, deviations_flagged=0, deviations_overridden_as_drift=0)
    r = call("log_final_ditto", is_simulation=True, **args)
    url, body = fake_http.posted[-1]
    assert r["is_simulation"] is True and url.endswith("events")
    assert body["events"][0]["thing_id"] == "historicVCIcoordinator:agentAssessment"
    assert body["events"][0]["value"]["coverage"] == 1.0
    call("log_final_ditto", is_simulation=False, **args)
    url, body = fake_http.posted[-1]
    assert "hourlyVCIcoordinator:agentAssessment" in url and "properties" in body
