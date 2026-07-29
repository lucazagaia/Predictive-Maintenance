"""
Cloud node — backend prognostics + maintenance decision.

Thesis reference architecture (Figure 2): compute-intensive, latency-tolerant backend work
runs in the Cloud layer — here RUL prognostics and the decision matrix, which the thesis
places in its maintenance-support and application layers (§4.4). The safety-critical
detection already happened at the edge; RUL and planning can absorb cloud latency.

    subscribes: /robot/state      (StateMsg)  → publishes /cloud/prediction (PredictionMsg)
    subscribes: /edge/detection + /cloud/prediction → publishes /decision   (DecisionMsg)

Thin wrapper over the RUL model and the fusion matrix, exchanging typed messages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prediction import PredictionInterface   # noqa: E402
from fusion import MaintenanceFusion          # noqa: E402
from messages import StateMsg, DetectionMsg, PredictionMsg, DecisionMsg, Header   # noqa: E402

# Fusion action code → plain-language operator instruction (application-layer output).
ACTION_MESSAGE = {
    "stop_and_inspect": "⛔ STOP: halt the robot and inspect immediately",
    "schedule_urgent_maintenance": "🚨 URGENT: schedule maintenance now",
    "schedule_maintenance_soon": "📅 PLAN: schedule maintenance within the window",
    "monitor_closely": "👀 MONITOR: increase monitoring frequency",
    "continue_operation": "✅ CONTINUE: normal operation",
}


class CloudNode:
    """Cloud backend: RUL prognostics + the maintenance decision matrix."""

    def __init__(self, model_path: str = None):
        self.predictor = PredictionInterface(model_path) if model_path else PredictionInterface()
        self.fusion = MaintenanceFusion()

    def on_state(self, state: StateMsg) -> PredictionMsg:
        """Handle /robot/state → publish /cloud/prediction (RUL)."""
        # One call suffices: with a single reading in history, the predictor left-pads
        # the 30-step window by replicating it — identical input (and output) to
        # calling it 30 times, without 29 wasted forward passes.
        self.predictor.sensor_history = []
        plan = self.predictor.get_maintenance_planning(state.adr)
        return PredictionMsg(
            header=Header(robot_id=state.header.robot_id),
            rul_cycles=plan["rul_cycles"], urgency=plan["urgency"],
            maintenance_window=plan["maintenance_window"], confidence=plan["confidence"],
        )

    def decide(self, detection: DetectionMsg, prediction: PredictionMsg) -> DecisionMsg:
        """Fuse /edge/detection + /cloud/prediction → publish /decision."""
        d = self.fusion.make_decision(
            {"status": detection.status, "confidence": detection.confidence,
             "details": detection.details},
            {"rul_cycles": prediction.rul_cycles, "urgency": prediction.urgency,
             "maintenance_window": prediction.maintenance_window},
        )
        return DecisionMsg(
            header=Header(robot_id=detection.header.robot_id),
            action=d["action"], priority=d["priority"], reasoning=d["reasoning"],
            operator_message=ACTION_MESSAGE.get(d["action"], d["action"]),
        )
