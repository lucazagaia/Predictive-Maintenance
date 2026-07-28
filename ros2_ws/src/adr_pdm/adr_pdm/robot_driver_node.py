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

import array
import json
import sys

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Header

from adr_pdm_interfaces.msg import RobotState, SensorWindow
from adr_pdm.pipeline import SAMPLES

# QoS for the sensor window. Not the sensor-data profile: that profile is meant for
# high-rate streams where a dropped sample is replaced milliseconds later. One window
# here is ~572 KB carrying an 11-second operation, and it drives a full safety decision,
# so losing one costs a whole monitoring cycle — and at that size a best-effort sample is
# fragmented over UDP, where a single lost fragment discards it. Reliable delivery
# retransmits fragments; depth 1 keeps only the newest window if the consumer lags.
WINDOW_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


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

        self.period = float(period)
        self.state_pub = self.create_publisher(RobotState, "robot/state", 10)
        self.window_pub = self.create_publisher(SensorWindow, "robot/window", WINDOW_QOS)

        # Volatile durability means nothing is retained for late-joining subscribers, so
        # anything published before DDS discovery completes is lost. Poll until the edge
        # and cloud subscriptions are visible, then start streaming — otherwise the first
        # few ticks vanish and the graph looks idle for ~10 s.
        self.timer = self.create_timer(0.2, self._await_subscribers)
        self.get_logger().info(f"robot_driver up: {self.robot_id} (scenario={scenario})")

    def _await_subscribers(self):
        """Hold off publishing until the downstream nodes have been discovered."""
        if self.window_pub.get_subscription_count() == 0 or \
                self.state_pub.get_subscription_count() == 0:
            self.get_logger().info("waiting for edge/cloud subscribers...", once=True)
            return

        self.timer.cancel()
        self.get_logger().info("subscribers discovered, streaming")
        self.tick()                                          # publish at once
        self.timer = self.create_timer(self.period, self.tick)

    def _header(self) -> Header:
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.robot_id
        return h

    def tick(self):
        # One header per tick: state and window share a stamp, so a downstream
        # consumer can associate them by time.
        header = self._header()

        s = RobotState()
        s.header = header
        s.temperature = float(self.reading["temperature"])
        s.vibration = float(self.reading["vibration"])
        s.torque = float(self.reading["torque"])
        s.current = float(self.reading["current"])
        self.state_pub.publish(s)

        w = SensorWindow()
        w.header = header
        w.n_signals = int(self.window.shape[0])
        w.n_timesteps = int(self.window.shape[1])
        # array('f').frombytes is the message's native storage type and avoids
        # boxing 143k Python floats per publish.
        data = array.array("f")
        data.frombytes(np.ascontiguousarray(self.window, dtype=np.float32).tobytes())
        w.data = data
        self.window_pub.publish(w)


def main(args=None):
    # Shutdown handling per the official demo nodes: catch the external shutdown
    # that `ros2 launch` triggers, and use try_shutdown (the context may already
    # be shut down by the signal handler).
    rclpy.init(args=args)
    node = RobotDriver()
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
