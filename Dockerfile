# Real ROS2 runtime for the ADR predictive-maintenance pipeline.
# ROS2 Humble (Ubuntu 22.04, Python 3.10) + the CPU PyTorch stack the nodes wrap.
FROM ros:humble

# ML dependencies for the wrapped pipeline (CPU-only torch).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "numpy>=1.24" "scikit-learn>=1.6"

# Copy the repo (src/, models/, data/, ros2_ws/) and point the nodes at it.
WORKDIR /app
COPY . /app
ENV ADR_PDM_ROOT=/app

# Build the ROS2 workspace (interfaces + nodes).
WORKDIR /app/ros2_ws
RUN . /opt/ros/humble/setup.sh && colcon build

ENTRYPOINT ["/app/ros2_ws/entrypoint.sh"]
CMD ["ros2", "launch", "adr_pdm", "pdm.launch.py"]
