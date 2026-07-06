"""
Edge node — on-robot, real-time anomaly detection.

Thesis reference architecture (Abbildung 2): the latency-critical anomaly-detection
inference runs in a dedicated Edge layer, locally at the robot, so a fault can trigger a
stop without a cloud round-trip ("Echtzeit-Detektionsentscheidung lokal beim Roboter").

    subscribes: /robot/state      (StateMsg)
    publishes:  /edge/detection   (DetectionMsg)

Thin wrapper over the MVT-Flow detector, exchanging typed messages instead of raw dicts.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from detection import MVTFlowDetectionInterface   # noqa: E402
from messages import StateMsg, DetectionMsg, Header, TOPICS   # noqa: E402


class EdgeNode:
    """On-robot detection node. `LAYER` marks where it is deployed in the architecture."""

    LAYER = "edge"
    PUBLISHES = TOPICS["detection"]

    def __init__(self, model_path: str, scaler_path: str,
                 window_size: int = 1100, n_signals: int = 130):
        self.detector = MVTFlowDetectionInterface(
            model_path=model_path, scaler_path=scaler_path,
            window_size=window_size, n_signals=n_signals,
        )

    def on_state(self, state: StateMsg) -> DetectionMsg:
        """Handle a /robot/state message and publish a /edge/detection message."""
        window = state.window
        if window is None:
            raise ValueError("StateMsg carries no detection window (n_signals x n_timesteps)")
        if window.ndim == 2:
            window = window[np.newaxis, ...]
        r = self.detector.get_robot_status(window)
        return DetectionMsg(
            header=Header(robot_id=state.header.robot_id),
            status=r["status"], confidence=r["confidence"],
            anomaly_score=r["anomaly_score"], model=r["model"],
            details=r.get("details", ""),
        )
