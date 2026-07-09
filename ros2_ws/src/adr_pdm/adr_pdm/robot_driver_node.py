"""
Robot driver node — stands in for an ADR publishing its runtime state.

Publishes (relative topics, so a fleet namespaces them per robot):
    robot/state    (RobotState)    proprioceptive scalars for the RUL stage
    robot/window   (SensorWindow)  multivariate window for the detection stage

Parameters:
    robot_id  (str)  header.frame_id on every message         [default adr-001]
    scenario  (str)  'healthy' | 'degraded' — which sample to emit [default healthy]
    period    (float) publish period in seconds                [default 2.0]
"""

import json

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from adr_pdm_interfaces.msg import RobotState, SensorWindow
from adr_pdm.pipeline import SAMPLES


class RobotDriver(Node):
    def __init__(self):
        super().__init__("robot_driver")
        self.robot_id = self.declare_parameter("robot_id", "adr-001").value
        scenario = self.declare_parameter("scenario", "healthy").value
        period = self.declare_parameter("period", 2.0).value

        adr = json.loads((SAMPLES / "sample_adr_readings.json").read_text())
        self.reading = adr.get(scenario, adr["healthy"])
        win_name = "voraus_anomaly_window.npy" if scenario == "degraded" else "voraus_normal_window.npy"
        win_path = SAMPLES / win_name
        if win_path.exists():
            w = np.load(win_path).astype("float32")
            self.window = w[0] if w.ndim == 3 else w
        else:
            self.window = np.random.default_rng(0).standard_normal((130, 1100)).astype("float32")

        self.state_pub = self.create_publisher(RobotState, "robot/state", 10)
        self.window_pub = self.create_publisher(SensorWindow, "robot/window", 10)
        self.create_timer(float(period), self.tick)
        self.get_logger().info(f"robot_driver up: {self.robot_id} (scenario={scenario})")

    def _header(self) -> Header:
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.robot_id
        return h

    def tick(self):
        s = RobotState()
        s.header = self._header()
        s.temperature = float(self.reading["temperature"])
        s.vibration = float(self.reading["vibration"])
        s.torque = float(self.reading["torque"])
        s.current = float(self.reading["current"])
        self.state_pub.publish(s)

        w = SensorWindow()
        w.header = self._header()
        w.n_signals = int(self.window.shape[0])
        w.n_timesteps = int(self.window.shape[1])
        w.data = self.window.reshape(-1).tolist()
        self.window_pub.publish(w)


def main(args=None):
    rclpy.init(args=args)
    node = RobotDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
