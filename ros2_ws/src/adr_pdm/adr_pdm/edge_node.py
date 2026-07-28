"""
Edge node — on-robot real-time anomaly detection (rclpy wrapper).

    subscribes: robot/window     (SensorWindow)
    publishes:  edge/detection   (Detection)

Wraps the repo's MVTFlowDetectionInterface unchanged; only the transport is ROS2.
"""

import sys

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

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
        # Must mirror the driver's window QoS (reliable, keep-last depth 1) — an
        # incompatible subscription profile simply never matches the publisher.
        self.create_subscription(
            SensorWindow, "robot/window", self.on_window,
            QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST, depth=1),
        )
        self.get_logger().info("edge_node up")

    def on_window(self, msg: SensorWindow):
        # msg.data arrives as array('f'); frombuffer reads it zero-copy.
        window = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.n_signals, msg.n_timesteps)
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
    # Shutdown handling per the official demo nodes (ros2 launch sends an
    # external shutdown, not only SIGINT-as-KeyboardInterrupt).
    rclpy.init(args=args)
    node = EdgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
