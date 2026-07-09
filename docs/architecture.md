# Architecture

This repo implements the **model + decision-logic layer** of the thesis reference
architecture (Abbildung 2). The layered edge/cloud structure and the message flow are
modelled in software — an **edge node** (detection), a **cloud node** (RUL + decision), and
**typed messages over named topics** (`src/messages.py`, mirroring the thesis ROS2
`state.msg`). It runs in one process; there is no real ROS2 transport or on-robot
deployment. The diagram is a conceptual restatement and does not reproduce thesis figures.

```
   robot                        EDGE (on-robot)              CLOUD (backend)
 ┌────────────┐   /robot/state  ┌──────────────────┐        ┌────────────────────────┐
 │ sensors +  │────────────────▶│  DETECTION        │        │  PREDICTION            │
 │ 130×1100   │        │        │  MVT-Flow (PyTorch)│       │  Li et al. CNN (PyTorch)│
 │ window     │        │        │  → healthy/warning/│       │  + ADR→C-MAPSS proxy   │
 └────────────┘        │        │    anomaly         │       │  → RUL, urgency        │
                       │        └────────┬───────────┘       └───────────┬────────────┘
                       │      /edge/detection                 /cloud/prediction
                       │                 │                                │
                       └─────────────────┼────────────────────────────────┘
                                          ▼
                              ┌─────────────────────────┐   (CloudNode.decide)
                              │  FUSION — decision matrix│
                              │  (safety over planning)  │
                              └────────────┬─────────────┘
                                     /decision
                                          ▼
              CONTINUE · MONITOR · PLAN · URGENT · STOP  (+ operator message)
```

## Topics & messages

| Topic | Message (`src/messages.py`) | Publisher | Payload |
|-------|-----------------------------|-----------|---------|
| `/robot/state`     | `StateMsg`      | robot     | 4 ADR channels + detection window |
| `/edge/detection`  | `DetectionMsg`  | EdgeNode  | status, confidence, anomaly score |
| `/cloud/prediction`| `PredictionMsg` | CloudNode | RUL cycles, urgency, window |
| `/decision`        | `DecisionMsg`   | CloudNode | action, priority, operator message |

Each message carries a `Header` (timestamp + robot id) and is JSON-serialisable, the way a
ROS2 message serialises onto a topic.

## Why these components

- **MVT-Flow for detection.** Anomaly detection on robots realistically has *only
  normal data* at training time (faults are rare and diverse). A normalizing flow
  learns the density of normal operation and flags low-likelihood windows — no
  labelled faults required. Follows Brockmann et al. (2023), the voraus-AD baseline.

- **Li et al. CNN for RUL.** A compact 1D-CNN over a 30-step sensor window is a
  well-established, cheap-to-run RUL baseline on C-MAPSS. It fits the "edge-friendly,
  legible" goal better than a heavier sequence model.

- **Rule-based fusion, not a learned meta-model.** The detection and prediction
  outputs answer different questions ("safe now?" vs "how long left?"). A small,
  auditable decision matrix that always lets a live anomaly override a rosy RUL is
  more trustworthy — and more explainable to an operator — than a black-box combiner.

## Decision matrix (fusion)

```
                        RUL urgency
              immediate  urgent    soon      planned
Anomaly       STOP       STOP      STOP      STOP
Warning       URGENT     URGENT    MONITOR   MONITOR
Healthy       PLAN       PLAN      CONTINUE  CONTINUE
```

Safety first: an `anomaly` status triggers STOP regardless of RUL.
