#!/bin/bash
# Source ROS2 and the built workspace, then exec the given command.
set -e
source /opt/ros/humble/setup.bash
source /app/ros2_ws/install/setup.bash
exec "$@"
