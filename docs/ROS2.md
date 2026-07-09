# Running the pipeline as real ROS2 nodes

The pipeline also ships as a **real ROS2 (Humble) package** — genuine `rclpy` nodes talking
over DDS topics, not a simulation of it. The same detection / RUL / fusion code from `src/`
is wrapped by three nodes; only the transport is ROS2.

```
robot_driver ──robot/state──▶            ┌─ cloud_node ─────────────────┐
             ──robot/window─▶ edge_node ─┤  RUL (rul_cnn.pt) + fusion    │
                              (MVT-Flow)  │  robot/state cached per robot │
                              ─edge/detection─▶                          │
                                          └─ cloud/prediction, decision ─┘
```

| Node | subscribes | publishes |
|------|------------|-----------|
| `robot_driver` | — | `robot/state` (RobotState), `robot/window` (SensorWindow) |
| `edge_node` | `robot/window` | `edge/detection` (Detection) |
| `cloud_node` | `robot/state`, `edge/detection` | `cloud/prediction` (Prediction), `decision` (Decision) |

Custom messages live in `adr_pdm_interfaces`; the nodes in `adr_pdm`.

## Run it (Docker — no local ROS2 install needed)

```bash
# 1. build the image (ROS2 Humble + CPU PyTorch + colcon build)
docker build -t adr-pdm .

# 2. run the graph — you'll see edge detection + cloud decision logs live
docker run --rm -it adr-pdm
# ... [edge_node] detection=anomaly score=...
# ... [cloud_node] RUL=10 -> stop_and_inspect (critical)

# choose the scenario
docker run --rm -it adr-pdm ros2 launch adr_pdm pdm.launch.py scenario:=healthy
```

Inspect the live topics from a second shell:

```bash
docker ps                                   # find the running container id
docker exec -it <id> bash -lc \
  'source /app/ros2_ws/install/setup.bash && ros2 topic echo /decision'
# or: ros2 topic list   |   ros2 node list   |   ros2 topic hz /edge/detection
```

## Local ROS2 (if you have Humble installed)

```bash
cd ros2_ws
colcon build
source install/setup.bash
ros2 launch adr_pdm pdm.launch.py
```

## Notes

- `ADR_PDM_ROOT` tells the nodes where the repo (`src/`, `models/`) is; the Docker image
  sets it to `/app`. Set it yourself for a non-Docker run from a different working directory.
- One robot for now. The nodes use **relative** topic names and `header.frame_id` as the
  robot id, so a fleet is just the same three nodes launched under one namespace per robot
  (the dashboard step builds on this).
- Models are the committed PyTorch weights; no training needed to run the graph.
