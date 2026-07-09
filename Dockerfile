# Real ROS2 runtime for the ADR predictive-maintenance pipeline.
# ROS2 Humble (Ubuntu 22.04, Python 3.10) + the CPU PyTorch stack the nodes wrap.
FROM ros:humble

# ros:humble is minimal (ros_core): add pip + ROS build tools (colcon, rosidl generators).
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip ros-dev-tools ros-humble-rosidl-default-generators \
 && rm -rf /var/lib/apt/lists/*

# Copy the repo (src/, models/, data/, ros2_ws/) and point the nodes at it.
WORKDIR /app
COPY . /app
ENV ADR_PDM_ROOT=/app

# Build the ROS2 workspace BEFORE installing the ML stack. pip would upgrade setuptools to
# a version incompatible with ROS Humble's ament_cmake_python (canonicalize_version /
# packaging mismatch), so colcon must run first against the stock setuptools.
WORKDIR /app/ros2_ws
RUN . /opt/ros/humble/setup.sh && colcon build

# ML dependencies for the wrapped pipeline (CPU-only torch), installed after the build.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip3 install --no-cache-dir "numpy>=1.24" "scikit-learn==1.6.1"

ENTRYPOINT ["/app/ros2_ws/entrypoint.sh"]
CMD ["ros2", "launch", "adr_pdm", "pdm.launch.py"]
