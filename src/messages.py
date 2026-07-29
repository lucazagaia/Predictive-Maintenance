"""
Message contracts between pipeline stages — the software mirror of the thesis's
ROS2 publish/subscribe design (Figure 3, `state.msg`; §4.2.3, on the data path).

In the reference architecture the robot and the edge/cloud components talk over ROS2
*topics*, each carrying a typed message. Here those messages are plain dataclasses so the
pipeline has the same explicit, typed contract at each producer→consumer boundary —
without a ROS2 dependency. It runs in one process, but the boundaries are the same: the
edge node *produces* a DetectionMsg, the cloud node *consumes* it to fuse a decision.

Topic map (who publishes what):

    robot        --/robot/state------>  StateMsg       (proprioceptive sensor state)
    edge node    --/edge/detection--->  DetectionMsg   (real-time health status)
    cloud node   --/cloud/prediction->  PredictionMsg  (RUL / prognostics)
    cloud node   --/decision--------->  DecisionMsg    (maintenance recommendation)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

# ROS2-style topic names — the single point of truth for what flows where.
TOPICS = {
    "state": "/robot/state",
    "detection": "/edge/detection",
    "prediction": "/cloud/prediction",
    "decision": "/decision",
}


def _now() -> str:
    return datetime.now().isoformat()


@dataclass
class Header:
    """ROS2-style header: a per-message timestamp and the source robot id."""
    stamp: str = field(default_factory=_now)   # factory → a fresh timestamp per instance
    robot_id: str = "adr-001"


@dataclass
class StateMsg:
    """
    `/robot/state` — the robot's runtime state (thesis `state.msg`).

    Carries the proprioceptive scalar reading (motor current, vibration, etc., abstracted
    here to the thesis's 4 ADR channels) plus, optionally, the multivariate detection
    window (signals x timesteps) the anomaly detector consumes. In a real ROS2 graph these
    could be two topics; they are bundled here for the single-process demo.
    """
    header: Header
    temperature: float          # [°C]
    vibration: float            # [g]
    torque: float               # [Nm]
    current: float              # [A]
    window: Optional[np.ndarray] = None   # (n_signals, n_timesteps) for detection

    @property
    def adr(self) -> dict:
        """The 4-channel ADR reading the RUL proxy expects."""
        return {"temperature": self.temperature, "vibration": self.vibration,
                "torque": self.torque, "current": self.current}


@dataclass
class DetectionMsg:
    """`/edge/detection` — real-time health status from the edge anomaly detector."""
    header: Header
    status: str                 # healthy | warning | anomaly
    confidence: float
    anomaly_score: float
    model: str
    details: str = ""


@dataclass
class PredictionMsg:
    """`/cloud/prediction` — RUL estimate + planning from the cloud prognostics node."""
    header: Header
    rul_cycles: int
    urgency: str                # immediate | urgent | soon | planned
    maintenance_window: str
    confidence: float


@dataclass
class DecisionMsg:
    """`/decision` — the maintenance recommendation (thesis §4.4, Table 1)."""
    header: Header
    action: str
    priority: str
    reasoning: str
    operator_message: str
