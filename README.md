# Predictive Maintenance for Autonomous Delivery Robots — ML & Decision Layer

A compact, legible implementation of the machine-learning and decision-logic layer of
a predictive-maintenance pipeline: sensor data in → **anomaly detection** + **remaining
useful life (RUL) estimation** → **fusion** → a maintenance recommendation
(`CONTINUE` / `MONITOR` / `PLAN` / `URGENT` / `STOP`).

> ⚠️ **Portfolio / research prototype.** This is a personal project that validates an
> ML approach. It is intentionally *not* production software and will never run on a
> real robot — it demonstrates the method, not a deployment.

---

## Context

This repository implements the **model and decision-logic layer** from a bachelor's
thesis on predictive maintenance for autonomous last-mile delivery robots (TU Berlin,
graded **1.3**). The thesis is a conceptual *Vorgehensmodell*: it designed the process,
the edge/cloud reference architecture, and the choice of ML/decision methods, but — as its
own limitations note — built **no prototype** and validated the approach conceptually, not
experimentally. This repo is that missing implementation: an **independent, personal
project** that runs the thesis's chosen methods (MVT-Flow, Li et al. CNN) on the benchmark
datasets it identifies. It is not thesis deliverable code.

---

## What this does

Given a sensor sample, the pipeline runs three stages and prints a maintenance action:

1. **Detection — "is the robot healthy right now?"**
   An **MVT-Flow** normalizing-flow model (PyTorch) scores a multivariate time-series
   window and returns `healthy` / `warning` / `anomaly`. Normalizing flows learn the
   density of *normal* operation, so they need only normal data at training time — a
   good fit for machines where real faults are rare and diverse.

2. **Prediction — "how much life is left?"**
   A **Li et al. (2018) 1D-CNN** (PyTorch) estimates RUL in cycles from a
   30-step sensor window, then maps it to an urgency level and a maintenance window.

3. **Fusion — "so what do we do?"**
   A small, auditable **decision matrix** combines the two, always letting a live
   anomaly override an optimistic RUL (safety before planning), and emits an action
   plus a plain-language operator message.

`run_demo.py` runs all three end to end on sample inputs and prints the result.

---

## Architecture

```
                    ┌─ EDGE (on-robot) ──┐   ┌─ CLOUD (backend) ────────┐
/robot/state ──────▶│ MVT-Flow detection │──▶│ decision matrix (fusion) │──▶ /decision
        │           └─ /edge/detection ──┘   │                          │    CONTINUE / MONITOR /
        └──────────────────────────────────▶ │ Li et al. RUL            │    PLAN / URGENT / STOP
                                              └─ /cloud/prediction ──────┘
```

This mirrors the thesis reference architecture (Abbildung 2): the latency-critical anomaly
detector runs at the **edge** (on-robot), the heavier RUL model + decision matrix run in the
**cloud**, and the stages exchange **typed messages over named topics** (`src/messages.py`,
the software analogue of the thesis's ROS2 `state.msg`). `run_demo.py` runs this in one
process (no ROS2 needed), and the **same** detection/RUL/fusion code also ships as a real
**ROS2 (Humble)** package — genuine `rclpy` nodes over DDS topics, runnable via Docker; see
[`docs/ROS2.md`](docs/ROS2.md). Two input modalities on purpose: detection consumes a raw
multivariate window; prediction consumes a 4-channel ADR reading. See
[`docs/architecture.md`](docs/architecture.md) for the component rationale and decision matrix.

```
Predictive-Maintenance/
├── run_demo.py            # ← single entry point ("run this to see it work")
├── src/
│   ├── messages.py        # typed message contracts (ROS2 state.msg-style topics)
│   ├── edge.py            # EdgeNode — on-robot real-time detection
│   ├── cloud.py           # CloudNode — backend RUL prognostics + decision matrix
│   ├── detection.py       # MVT-Flow inference interface (+ synthetic fallback)
│   ├── mvt_flow_model.py  # MVT-Flow normalizing-flow network (PyTorch)
│   ├── prediction.py      # RUL inference interface (ADR→C-MAPSS proxy + CNN)
│   ├── rul_model.py       # Li et al. RUL CNN network (PyTorch)
│   └── fusion.py          # decision matrix
├── scripts/
│   ├── train_rul.py           # RUL: raw C-MAPSS FD001 → preprocess → train → save
│   ├── train_mvtflow.py       # detection: voraus parquet → windows → train → save
│   └── calibrate_detection.py # reproduce detection thresholds from normal data
├── models/                # all inference artifacts (PyTorch): mvt_flow_voraus_ad.pt,
│                          #   scaler_voraus_ad.pkl, rul_cnn.pt, voraus_thresholds.json,
│                          #   normalization_stats.json
├── data/samples/          # demo inputs: ADR readings, real C-MAPSS + voraus windows
├── ros2_ws/               # real ROS2 (Humble) package: rclpy nodes + custom .msg (docs/ROS2.md)
├── Dockerfile             # ROS2 + PyTorch image to build/run the node graph
├── notebooks/             # exploratory + the MVT-Flow Colab trainer (notebooks/README.md)
└── docs/                  # architecture + ROS2 notes
```

---

## Data

Data provenance matters for interpreting the results, so it is stated plainly here:

- **Detection** is trained on **voraus-AD** (Brockmann et al., 2023) — a *real* robot
  anomaly-detection dataset (130 signals, pick-and-place manipulator), **not**
  autonomous-delivery-robot data.
- **Prediction** is trained on **NASA C-MAPSS FD001** — aircraft **turbofan**
  run-to-failure data, a standard RUL benchmark, again not robot data.
- The 4-channel ADR reading (temperature, vibration, torque, current — proprioceptive
  sensors from the thesis's ADR sensor set) fed to the RUL stage is mapped to C-MAPSS
  features by a **hand-built proxy ("virtual sensor abstraction")** in `src/prediction.py`.
  This is a deliberate stand-in, **not** a learned or physically-calibrated mapping.

A small slice of the **real** C-MAPSS test set (3 windows + true RUL labels) is
committed at `data/samples/cmapss_sample_*.npy` so the RUL model can be validated on
genuine benchmark data, independently of the ADR proxy above.

**Why proxy data?** As the thesis discusses, a public run-to-failure dataset from real
autonomous-delivery-robot sensor streams **does not yet exist** — a well-known, field-wide
limitation of ADR predictive maintenance, not a shortcut specific to this project. The
pipeline is therefore validated on the closest available public proxies. Retraining on real
ADR degradation data, once it exists, is the natural next step.

---

## How to run it

```bash
# 1. clone, then create an environment (Python 3.10–3.13)
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the end-to-end demo
python run_demo.py
```

`run_demo.py` runs fully real out of the box: the trained MVT-Flow detector and Li et al.
RUL model are both committed under `models/`. If a weight file is ever missing, detection
falls back to a clearly-labelled synthetic scorer and RUL to a placeholder, so the demo
always runs top to bottom. Retrain either model with the scripts below.

### Training the models

Both trainers go from raw public data to saved weights in one command — preprocessing,
windowing, training and evaluation included.

```bash
# RUL — Li et al. (2018) CNN on NASA C-MAPSS FD001 (~5 min, CPU is fine).
# Download the free C-MAPSS set; point at the folder with train_FD001.txt / RUL_FD001.txt.
python scripts/train_rul.py --cmapss-dir /path/to/CMAPSSData
#   → writes models/rul_cnn.pt + models/normalization_stats.json

# Detection — MVT-Flow on the voraus-AD parquet (GPU recommended).
# Locally:
python scripts/train_mvtflow.py --parquet /path/to/voraus-ad-dataset-100hz.parquet
# or on a GPU in Colab: open notebooks/detection_01_mvt_flow_training.ipynb → Run all
#   → writes models/mvt_flow_voraus_ad.pt + scaler_voraus_ad.pkl + voraus_thresholds.json
```

After training, re-run `python run_demo.py` for fully real, end-to-end results.

---

## Results

### Detection — real, out of the box

`run_demo.py` scores real voraus-AD windows and the fusion safety-override fires correctly:

```
SCENARIO: Healthy operation   (detection window: real sample)
  Detection : healthy   score=-429147.5   [MVT-Flow]
  Prediction: RUL=47 cycles  urgency=urgent
  -> Action : schedule_maintenance_soon   (healthy status, but short RUL → PLAN)
  -> Operator: 📅 PLAN: schedule maintenance within the window

SCENARIO: Degraded / fault    (detection window: real sample)
  Detection : anomaly   score=2726343.25  [MVT-Flow]
  -> Action : stop_and_inspect  (priority: critical)
  -> Operator: ⛔ STOP: halt the robot and inspect immediately
```

The two rows exercise different parts of the thesis decision matrix: a healthy status with
a short RUL yields `PLAN`, while a live anomaly overrides everything to `STOP`.

**AUROC = 0.946** on a held-out normal/anomaly split (`scripts/train_mvtflow.py`, seed 42) —
slightly above the paper's 0.936, which is a mean over 9 runs, so read this as a strong
single-split result rather than a matched benchmark. MVT-Flow scores are unbounded
log-likelihoods, so the healthy/anomaly thresholds are calibrated on real normal windows
(p95/p99 → `models/voraus_thresholds.json`); calibration and inference share preprocessing,
so the decision is self-consistent.

### RUL

Trained by `scripts/train_rul.py` on C-MAPSS FD001 — **test RMSE = 17.5** (an untuned
single run; Li et al. report ≈ 12.6). The demo validates it on three real, pre-normalized
C-MAPSS test windows fed straight to the model (no ADR proxy):

```
window 0: predicted RUL =   9.9 cycles  |  true RUL =   7
window 1: predicted RUL =  72.1 cycles  |  true RUL =  87
window 2: predicted RUL = 120.1 cycles  |  true RUL = 145
```

Predictions track true RUL across the degradation range — a genuine check on the model's
own benchmark domain.

**Component references:**
- Detection: MVT-Flow from the voraus-AD paper (Brockmann et al., 2023, arXiv:2311.04765).
- Prediction: RUL CNN follows Li et al. (2018), *"Remaining useful life estimation in
  prognostics using deep convolution neural networks."*

---

## Limitations & Next Steps

Framed as a roadmap, not an apology — these are the honest edges of a portfolio prototype:

- **Real ROS2 exists, but not on hardware.** The pipeline ships as real `rclpy` nodes over
  DDS topics (`ros2_ws/`, run via Docker — see `docs/ROS2.md`), so the edge/cloud boundary is
  genuine ROS2, not a simulation of it. What's still missing is deployment on an actual robot
  (real sensor drivers, a Gazebo/hardware bring-up, network-distributed edge and cloud hosts).
- **Proxy data, not real ADR streams.** Detection uses voraus-AD, prediction uses
  C-MAPSS, and the ADR→C-MAPSS sensor mapping is a hand-built proxy. Real ADR
  run-to-failure data does not yet exist publicly (see *Data*).
- **Metrics are single-run, not tuned.** Detection AUROC 0.946 and RUL RMSE 17.5 come from
  one training run each with default hyperparameters; the papers report better figures with
  ensembling/tuning. They are honest illustrations, not a benchmark-chasing effort.
- **No real-time performance testing.** Latency/throughput claims are not benchmarked;
  inference is validated for correctness, not speed.
- **No production hardening.** No input validation at API boundaries, no monitoring, no
  retraining loop, no model versioning/serving.
- **RUL confidence is a placeholder.** The CNN is a point-estimate regressor; the
  reported confidence is a constant, not calibrated uncertainty.

**Next steps:** obtain real ADR degradation data → replace the proxy mapping → add
calibrated uncertainty (e.g. an aleatoric two-head RUL variant) → benchmark inference
latency → run the ROS2 nodes on real robot/edge/cloud hosts with live sensor drivers.

---

## License

Released under the [MIT License](LICENSE).
