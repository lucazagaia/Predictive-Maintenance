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

## Design notes & known deviations

Choices that follow the ROS2 documentation:

- **QoS.** The sensor window is **reliable, keep-last depth 1** — deliberately *not* the
  `qos_profile_sensor_data` profile. That profile targets high-rate streams (lidar,
  camera) where a dropped sample is replaced milliseconds later; here a single ~572 KB
  message carries an 11-second operation and drives a full safety decision, and at that
  size a best-effort sample is fragmented over UDP, so one lost fragment discards the
  whole window. Measured: best-effort dropped roughly half the windows, reliable drops
  none. Depth 1 still avoids queueing stale windows if a consumer lags. The low-rate
  result/command topics use the default reliable profile.
- **Startup.** Volatile durability retains nothing for late-joining subscribers, so the
  driver waits until the edge and cloud subscriptions are discovered
  (`get_subscription_count()`) before its first publish. Without this the opening ticks
  are published into the void and the graph looks idle for ~12 s; with it the first
  detection lands under a second.
- **Shutdown.** Node `main()`s follow the official demo pattern: catch
  `KeyboardInterrupt` *and* `ExternalShutdownException` (what `ros2 launch` triggers) and
  call `rclpy.try_shutdown()`.
- **Interfaces.** Custom messages live in an `ament_cmake` package (interfaces cannot be
  generated from `ament_python`), with the documented `rosidl` declarations.

Deliberate deviations, and what the canonical alternative would be:

- **`header.frame_id` carries the robot id.** `frame_id` is formally a TF frame name;
  using it as a fleet identifier is common practice but a repurpose. A dedicated
  `string robot_id` field would be the strict alternative.
- **The cloud node caches the latest `RobotState` per robot** and fuses when a
  `Detection` arrives. The canonical multi-topic composition is `message_filters`
  (e.g. `ApproximateTimeSynchronizer` or `Cache`); a latest-value cache was chosen
  because the two topics have different rates and only the freshest state matters.
- **Nodes import the repo's `src/` via `ADR_PDM_ROOT`/`sys.path`** instead of installing
  the pipeline as a proper Python package. Packaging-clean alternative: `pip install`
  the repo in the image and import it normally.
- **Plain nodes, not lifecycle nodes; no rosdep.** Managed lifecycle nodes and
  rosdep-resolved dependencies are the production-grade route; a demo pipeline with a
  hand-pinned Dockerfile keeps the surface smaller.
