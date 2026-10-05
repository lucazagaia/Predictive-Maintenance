"""
End-to-end checks against the committed weights, so the numbers in the README stay true.
"""

import json

import numpy as np
import pytest

from conftest import ROOT
from cloud import CloudNode
from edge import EdgeNode
from messages import Header, StateMsg

MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"


@pytest.fixture(scope="module")
def edge():
    return EdgeNode(
        model_path=str(MODELS / "mvt_flow_voraus_ad.pt"),
        scaler_path=str(MODELS / "scaler_voraus_ad.pkl"),
    )


@pytest.fixture(scope="module")
def cloud():
    node = CloudNode()
    assert node.predictor.model_loaded, "committed RUL weights failed to load"
    return node


@pytest.fixture(scope="module")
def adr():
    return json.loads((SAMPLES / "sample_adr_readings.json").read_text())


def _state(window_file, reading):
    return StateMsg(header=Header(), window=np.load(SAMPLES / window_file).astype("float32"),
                    **reading)


def test_healthy_window_plans_maintenance(edge, cloud, adr):
    state = _state("voraus_normal_window.npy", adr["healthy"])
    detection = edge.on_state(state)
    assert detection.model == "MVT-Flow"
    assert detection.status == "healthy"
    decision = cloud.decide(detection, cloud.on_state(state))
    assert decision.action == "schedule_maintenance_soon"


def test_anomaly_window_stops_the_robot(edge, cloud, adr):
    state = _state("voraus_anomaly_window.npy", adr["degraded"])
    detection = edge.on_state(state)
    assert detection.model == "MVT-Flow"
    assert detection.status == "anomaly"
    decision = cloud.decide(detection, cloud.on_state(state))
    assert (decision.action, decision.priority) == ("stop_and_inspect", "critical")


def test_rul_on_real_cmapss_windows(cloud):
    X = np.load(SAMPLES / "cmapss_sample_X.npy")
    y_true = np.load(SAMPLES / "cmapss_sample_y.npy")
    preds = [cloud.predictor.predict_rul_from_window(x) for x in X]
    # The values quoted in the README.
    assert preds == pytest.approx([9.9, 72.1, 120.1], abs=1.0)
    assert list(y_true.astype(int)) == [7, 87, 145]


def test_reported_detection_metrics_match_the_artifact():
    ev = json.loads((MODELS / "voraus_thresholds.json").read_text())["evaluation"]
    assert round(ev["auroc"], 3) == 0.949
    assert round(ev["normal_false_alarm_warning_or_above"], 3) == 0.030
    assert round(ev["anomaly_flagged_anomaly"], 2) == 0.54
    assert round(ev["anomaly_flagged_warning_or_above"], 2) == 0.73
