"""
The contracts between the architecture's layers (thesis §4.2.3, §4.4.3): what each stage
hands the next, independent of which reference model sits behind it.
"""

import numpy as np
import pytest

from conftest import SAMPLES
from messages import Header, StateMsg


def test_detection_has_three_states_with_a_warning_band(edge):
    """Score → healthy / warning / anomaly, with a buffer zone between the two fences."""
    detector = edge.detector
    warn, crit = detector.warning_threshold, detector.anomaly_threshold
    assert warn < crit

    def status(score):
        return detector._convert_to_robot_status(score)["status"]

    assert status(warn - 1) == "healthy"
    assert status((warn + crit) / 2) == "warning"
    assert status(crit + 1) == "anomaly"


@pytest.mark.parametrize("rul,urgency", [
    (10, "immediate"), (40, "urgent"), (70, "soon"), (120, "planned"),
])
def test_rul_maps_to_four_urgency_classes(cloud, rul, urgency):
    """RUL → one of four classes, so every column of the decision matrix is reachable."""
    assert cloud.predictor._convert_to_maintenance_plan(rul)["urgency"] == urgency


def test_robot_id_travels_with_every_message(edge, cloud):
    """Each message keeps the source robot's id, as a fleet needs (thesis state.msg)."""
    window = np.load(SAMPLES / "voraus_normal_window.npy").astype("float32")
    state = StateMsg(header=Header(robot_id="adr-042"), window=window,
                     temperature=72.0, vibration=0.2, torque=12.0, current=9.5)

    detection = edge.on_state(state)
    prediction = cloud.on_state(state)
    decision = cloud.decide(detection, prediction)

    assert {m.header.robot_id for m in (detection, prediction, decision)} == {"adr-042"}
