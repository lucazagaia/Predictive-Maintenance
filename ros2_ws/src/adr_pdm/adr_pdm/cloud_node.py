"""
Cloud node — backend RUL prognostics + maintenance decision (rclpy wrapper).

    subscribes: robot/state      (RobotState)   -> cached per robot for the RUL stage
                edge/detection    (Detection)    -> triggers RUL + fusion
    publishes:  cloud/prediction  (Prediction)
                decision          (Decision)

Latest-value sync: the cloud keeps the most recent RobotState per robot id and, when a
Detection arrives, runs RUL on that state and fuses the two into a decision. Wraps the
repo's PredictionInterface + MaintenanceFusion unchanged.
"""

import rclpy
from rclpy.node import Node

from adr_pdm_interfaces.msg import RobotState, Detection, Prediction, Decision
from adr_pdm.pipeline import MODELS   # side effect: repo src/ on sys.path
from prediction import PredictionInterface   # noqa: E402
from fusion import MaintenanceFusion          # noqa: E402

ACTION_MESSAGE = {
    "stop_and_inspect": "STOP: halt the robot and inspect immediately",
    "schedule_urgent_maintenance": "URGENT: schedule maintenance now",
    "schedule_maintenance_soon": "PLAN: schedule maintenance within the window",
    "monitor_closely": "MONITOR: increase monitoring frequency",
    "continue_operation": "CONTINUE: normal operation",
}


class CloudNode(Node):
    def __init__(self):
        super().__init__("cloud_node")
        self.predictor = PredictionInterface(str(MODELS / "rul_cnn.pt"))
        self.fusion = MaintenanceFusion()
        self.latest_state = {}   # robot_id -> RobotState

        self.pred_pub = self.create_publisher(Prediction, "cloud/prediction", 10)
        self.dec_pub = self.create_publisher(Decision, "decision", 10)
        self.create_subscription(RobotState, "robot/state", self.on_state, 10)
        self.create_subscription(Detection, "edge/detection", self.on_detection, 10)
        self.get_logger().info("cloud_node up")

    def on_state(self, msg: RobotState):
        self.latest_state[msg.header.frame_id] = msg

    def on_detection(self, det: Detection):
        robot_id = det.header.frame_id
        state = self.latest_state.get(robot_id)
        if state is None:
            return   # no matching state yet; wait for the next detection

        adr = {"temperature": state.temperature, "vibration": state.vibration,
               "torque": state.torque, "current": state.current}
        self.predictor.sensor_history = []
        plan = None
        for _ in range(self.predictor.window_size):
            plan = self.predictor.get_maintenance_planning(adr)

        pred = Prediction()
        pred.header = det.header
        pred.rul_cycles = int(plan["rul_cycles"])
        pred.urgency = plan["urgency"]
        pred.maintenance_window = plan["maintenance_window"]
        pred.confidence = float(plan["confidence"])
        self.pred_pub.publish(pred)

        d = self.fusion.make_decision(
            {"status": det.status, "confidence": det.confidence, "details": det.details},
            {"rul_cycles": pred.rul_cycles, "urgency": pred.urgency,
             "maintenance_window": pred.maintenance_window},
        )
        dec = Decision()
        dec.header = det.header
        dec.action = d["action"]
        dec.priority = d["priority"]
        dec.reasoning = d["reasoning"]
        dec.operator_message = ACTION_MESSAGE.get(d["action"], d["action"])
        self.dec_pub.publish(dec)
        self.get_logger().info(
            f"[{robot_id}] RUL={pred.rul_cycles} -> {dec.action} ({dec.priority})")


def main(args=None):
    rclpy.init(args=args)
    node = CloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
