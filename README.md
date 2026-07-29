# Predictive Maintenance for Autonomous Delivery Robots — ML & Decision Layer

Sensor data in → **anomaly detection** + **remaining-useful-life (RUL) estimation** →
**decision logic** → a maintenance recommendation (`CONTINUE` / `MONITOR` / `PLAN` /
`URGENT` / `STOP`).

> **V1 — the raw Python pipeline.** Two trained models, a decision layer, and one command
> that runs the whole thing. Pure Python: no container, no robotics middleware, no training
> needed to see it work. Released as [`v1.0.0`](../../releases/tag/v1.0.0).

---

## Why this exists

My bachelor's thesis (TU Berlin, graded **1.3**) designed a process model for conceiving a
predictive-maintenance system for autonomous last-mile delivery robots. It is a conceptual
work: it specifies which components matter, which data to collect, which model classes fit,
and how a maintenance decision should be reached. As its own limitations section states,
**no prototype was built** — the approach was validated conceptually, not experimentally.

*(The thesis is written in German. Everything below is in English, with thesis sections
referenced by number — §4.4.3 and the like — since those are the same in either language.)*

This repository is that missing implementation. It takes the methods the thesis selected,
builds them, trains them on the benchmark datasets the thesis identifies, and runs them end
to end. The intention is not a product and not thesis deliverable code: it is an independent
personal project that answers *"does the thing I specified actually work when you build it?"*

That framing also sets the standard the repo holds itself to — every number here comes from
a run you can reproduce, and every gap between the concept and the implementation is stated
rather than glossed over.

---

## Alignment with the thesis

The thesis structures the design into four fields (Chapter 4). This repository implements the
second half of that model — the data-to-decision path:

| Thesis | Design field | In this repo |
|---|---|---|
| **§4.1** | System analysis and maintenance needs — critical components, failure modes | Conceptual — sets which sensors matter (see *Data*) |
| **§4.2** | Data acquisition and preprocessing, and the data path between components | `scripts/train_*.py` (preprocessing), `src/messages.py` (message contracts) |
| **§4.3** | Model selection and inference | `src/mvt_flow_model.py`, `src/rul_model.py`, `src/detection.py`, `src/prediction.py` |
| **§4.4** | Maintenance support — decision inputs, decision logic, operator output | `src/fusion.py` |

Two specifics worth naming, because they are the parts a reader can check directly:

- **The decision matrix** in `src/fusion.py` implements the thesis's decision matrix
  (§4.4.3, Table 1) cell for cell — including its safety-first rule that a live anomaly
  forces `STOP` regardless of the RUL estimate.
- **The sensor set** (temperature, vibration, torque, current) is the thesis's proprioceptive
  ADR sensor set (§4.2.1), not a generic industrial one.

The thesis also sketches an edge/cloud reference architecture (§4, Figure 2) and a
publish/subscribe design for the data path (§4.2.3). V1 models those boundaries in software:
`src/edge.py` (detection) and `src/cloud.py` (RUL + decision) exchange typed messages over
named topics, so the split is explicit in the code while the pipeline still runs anywhere
Python does. Deploying it on real robotics middleware is the next version — see
[Roadmap](#roadmap).

---

## What it does

Given a sensor sample, three stages run and a maintenance action is printed:

1. **Detection — "is the robot healthy right now?"**
   An **MVT-Flow** normalizing flow scores a multivariate time-series window and returns
   `healthy` / `warning` / `anomaly`. A normalizing flow learns the density of *normal*
   operation, so it needs only normal data at training time — the right fit for machines
   where real faults are rare, diverse, and expensive to label.

2. **Prediction — "how much life is left?"**
   A **Li et al. (2018) 1-D CNN** estimates RUL in cycles from a 30-step window, which is
   then mapped to an urgency level and a maintenance window.

3. **Decision — "so what do we do?"**
   A small, auditable **decision matrix** combines the two. Detection answers a safety
   question, prediction answers a planning question, and the matrix keeps safety ahead of
   planning.

```
                          RUL urgency
                immediate  urgent    soon      planned
   Anomaly      STOP       STOP      STOP      STOP
   Warning      URGENT     URGENT    MONITOR   MONITOR
   Healthy      PLAN       PLAN      CONTINUE  CONTINUE
```

```
voraus-AD window ──▶ MVT-Flow detection ─┐
 (130 × 1100)                            ├─▶ decision matrix ──▶ CONTINUE / MONITOR /
ADR sensor reading ─▶ Li et al. RUL ─────┘                       PLAN / URGENT / STOP
 (4 channels)
```

Two input modalities on purpose: detection consumes a raw multivariate window, prediction
consumes a 4-channel ADR reading. See [`docs/architecture.md`](docs/architecture.md) for the
component rationale.

---

## Repository structure

```
Predictive-Maintenance/
├── run_demo.py            # ← the entry point: one command, full pipeline
├── src/
│   ├── detection.py       # MVT-Flow inference: window → healthy/warning/anomaly
│   ├── mvt_flow_model.py  # MVT-Flow network (PyTorch)
│   ├── prediction.py      # RUL inference: ADR reading → RUL → urgency
│   ├── rul_model.py       # Li et al. CNN (PyTorch)
│   ├── fusion.py          # the decision matrix
│   ├── messages.py        # typed message contracts between stages
│   ├── edge.py            # edge node: detection
│   └── cloud.py           # cloud node: RUL + decision
├── scripts/
│   ├── train_rul.py            # raw C-MAPSS FD001 → trained RUL model
│   ├── train_mvtflow.py        # voraus-AD parquet → trained detector
│   └── calibrate_detection.py  # detection thresholds from normal data
├── models/                # trained weights + scalers + thresholds (committed)
├── data/samples/          # demo inputs: ADR readings, real C-MAPSS + voraus windows
├── notebooks/             # the same training runs, stage by stage (see notebooks/README.md)
└── docs/architecture.md   # component rationale and decision matrix
```

Both models are **PyTorch**. `src/` is the runtime library and depends on nothing in
`scripts/`; `scripts/` is build tooling that produces the artifacts `src/` loads.

---

## The papers

Both models are reimplementations of published methods, not novel architectures. That is
deliberate: the thesis selected these model classes, so the contribution here is a faithful,
verifiable build of them.

| Component | Paper |
|---|---|
| **Detection** | Brockmann, Rudolph, Rosenhahn & Wandt (2023), *The voraus-AD Dataset for Anomaly Detection in Robot Applications*, arXiv:2311.04765 |
| **Prediction** | Li, Ding & Sun (2018), *Remaining useful life estimation in prognostics using deep convolution neural networks*, Reliability Engineering & System Safety 172, 1–11 |

The notebooks cite these **by section, table and page** for every hyperparameter and design
choice, so any part of the implementation can be checked against its source. Deviations are
labelled as such — for example this repo feeds the RUL model 17 features (the paper's 14
sensors plus the 3 operational settings), and the MVT-Flow soft-clamping constant is set here
rather than taken from the paper, which defines the parameter but never gives it a value.

---

## Data

Stated plainly, because it matters for reading the results:

- **Detection** is trained on **voraus-AD** — a *real* robot anomaly-detection dataset
  (130 signals, 6-axis pick-and-place manipulator), but **not** delivery-robot data.
- **Prediction** is trained on **NASA C-MAPSS FD001** — aircraft turbofan run-to-failure
  data, the standard RUL benchmark, also not robot data.
- The 4-channel ADR reading feeding the RUL stage is mapped onto C-MAPSS features by a
  **hand-built proxy** in `src/prediction.py`. It is a deliberate stand-in, not a learned or
  physically calibrated mapping.

**Why proxy data?** As the thesis discusses, a public run-to-failure dataset from real
autonomous-delivery-robot sensor streams **does not yet exist** — a field-wide limitation,
not a shortcut specific to this project. The pipeline is therefore validated on the closest
available public proxies, and retraining on real ADR degradation data is the natural next
step once such data exists.

A slice of the **real** C-MAPSS test set (3 windows + true RUL labels) is committed at
`data/samples/cmapss_sample_*.npy`, so the RUL model can be checked on genuine benchmark
data independently of the proxy.

---

## How to run it

```bash
# Python 3.10–3.13
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

Runs fully real out of the box — both trained models are committed. If a weight file is ever
missing, detection falls back to a clearly-labelled synthetic scorer and RUL to a placeholder,
so the demo always completes.

### Retraining

Each trainer goes from raw public data to saved weights in one command:

```bash
# RUL — Li et al. CNN on C-MAPSS FD001 (~5-10 min, CPU is fine)
python scripts/train_rul.py --cmapss-dir /path/to/CMAPSSData
#   → models/rul_cnn.pt + models/normalization_stats.json

# Detection — MVT-Flow on the voraus-AD parquet (GPU recommended)
python scripts/train_mvtflow.py --parquet /path/to/voraus-ad-dataset-100hz.parquet
#   → models/mvt_flow_voraus_ad.pt + scaler_voraus_ad.pkl + voraus_thresholds.json
```

`notebooks/` contains the same two runs broken into readable stages, with the paper citations
attached. `notebooks/detection_01_mvt_flow_training.ipynb` is Colab-ready for a free GPU.

---

## Results

Single runs with default settings, reported as measured.

### Detection

`run_demo.py` on real voraus-AD windows:

```
SCENARIO: Healthy operation   (detection window: real sample)
  Detection : healthy   score=-429869.78  [MVT-Flow]
  Prediction: RUL=47 cycles  urgency=urgent
  -> Action : schedule_maintenance_soon   (healthy status, short RUL → PLAN)

SCENARIO: Degraded / fault    (detection window: real sample)
  Detection : anomaly   score=68150368.0   [MVT-Flow]
  -> Action : stop_and_inspect  (priority: critical)
```

**AUROC = 0.949** on a held-out split of 400 normal / 400 anomaly windows — above the
paper's 0.936, which is a mean over 9 runs, so read this as a strong single split rather
than a matched benchmark.

MVT-Flow scores are unbounded log-likelihoods, so the thresholds are calibrated on **normal
windows only** — Tukey fences (warning = Q3 + 1.5·IQR, anomaly = Q3 + 3·IQR) over 400 normal
windows. Calibration stays normal-only on purpose: the anomaly set evaluates the thresholds
but never sets them, matching how a fleet without labelled faults would actually be
commissioned.

Measured on 400 held-out anomaly windows: **54%** exceed the anomaly threshold and **73%**
reach at least `warning`, against a **3.0%** false-alarm rate on normal windows. Every figure
in this section is written to `models/voraus_thresholds.json` by the calibration run, so it
can be checked against the committed artifact. The ~19%
that land between the two fences are the borderline cases the `warning` band exists for —
the three-state output is doing real work rather than collapsing to healthy/anomaly.

### RUL

**Test RMSE = 17.5** against the official C-MAPSS labels (Li et al. report ≈ 12.6 tuned).
The demo also scores three real, pre-normalized C-MAPSS test windows with no proxy involved:

```
window 0: predicted RUL =   9.9 cycles  |  true RUL =   7
window 1: predicted RUL =  72.1 cycles  |  true RUL =  87
window 2: predicted RUL = 120.1 cycles  |  true RUL = 145
```

Predictions track true RUL across the degradation range — a genuine check on the model's own
benchmark domain.

---

## Limitations

Honest edges, not apologies:

- **Proxy data, not real ADR streams.** Detection uses voraus-AD, prediction uses C-MAPSS,
  and the ADR→C-MAPSS mapping is hand-built (see *Data*).
- **Metrics are single-run and untuned.** AUROC 0.949 and RMSE 17.5 come from one run each
  with the papers' default settings; both papers report better figures with tuning and
  ensembling.
- **RUL resolution is capped.** The model is trained against a piecewise-linear target capped
  at 125 cycles, so it cannot distinguish beyond that horizon; the urgency bands are scaled
  to that range accordingly.
- **No real-time performance testing.** Inference is validated for correctness, not latency.
- **No production hardening.** No input validation for malformed sensor data, no monitoring,
  no retraining loop, no model versioning or serving.
- **RUL confidence is a placeholder.** The CNN is a point-estimate regressor; the reported
  confidence is a constant, not calibrated uncertainty.

---

## Roadmap

`main` is V1 — the complete, self-contained pipeline. Each later version adds one layer
around it, developed on its own branch:

- **V2 — deployment.** The same models as real ROS2 nodes (rclpy, custom `.msg` interfaces,
  DDS transport) in a container, realising the thesis's edge/cloud reference architecture.
  In progress on [`feat/ros2`](../../tree/feat/ros2).
- **V3 — fleet & interface.** Multiple robots, decision history, and an operator-facing
  fleet health view.
- **Beyond.** Real ADR degradation data to replace the proxy mapping, calibrated uncertainty
  on the RUL estimate, and latency benchmarking.

---

## License

Released under the [MIT License](LICENSE). The referenced papers and the thesis are not part
of this repository and remain with their respective authors and publishers.
