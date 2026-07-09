"""
Edge node — on-robot real-time anomaly detection (rclpy wrapper).

    subscribes: robot/window     (SensorWindow)
    publishes:  edge/detection   (Detection)

Wraps the repo's MVTFlowDetectionInterface unchanged; only the transport is ROS2.
"""

import numpy as np
import rclpy
from rclpy.node import Node

from adr_pdm_interfaces.msg import SensorWindow, Detection
from adr_pdm.pipeline import MODELS   # side effect: repo src/ on sys.path
from detection import MVTFlowDetectionInterface   # noqa: E402


class EdgeNode(Node):
    def __init__(self):
        super().__init__("edge_node")
        self.detector = MVTFlowDetectionInterface(
            model_path=str(MODELS / "mvt_flow_voraus_ad.pt"),
            scaler_path=str(MODELS / "scaler_voraus_ad.pkl"),
            window_size=1100, n_signals=130,
        )
        self.pub = self.create_publisher(Detection, "edge/detection", 10)
        self.create_subscription(SensorWindow, "robot/window", self.on_window, 10)
        self.get_logger().info("edge_node up")

    def on_window(self, msg: SensorWindow):
        window = np.asarray(msg.data, dtype="float32").reshape(msg.n_signals, msg.n_timesteps)
        r = self.detector.get_robot_status(window[np.newaxis, ...])
        d = Detection()
        d.header = msg.header
        d.status = r["status"]
        d.confidence = float(r["confidence"])
        d.anomaly_score = float(r["anomaly_score"])
        d.model = r["model"]
        d.details = r.get("details", "")
        self.pub.publish(d)
        self.get_logger().info(
            f"[{msg.header.frame_id}] detection={d.status} score={d.anomaly_score:.0f}")


def main(args=None):
    rclpy.init(args=args)
    node = EdgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
