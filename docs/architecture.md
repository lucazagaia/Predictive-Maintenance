# Architecture

This repo implements the **model + decision-logic layer** of the thesis reference
architecture — not the full edge/cloud system. The diagram below is a conceptual
restatement of the data path; it deliberately does not reproduce thesis figures.

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                     run_demo.py                            │
                 └──────────────────────────────────────────────────────────┘
                                          │
        ┌─────────────────────────────────┴─────────────────────────────────┐
        │                                                                     │
 voraus-AD window                                              ADR sensor reading
 (130 signals × 1100 steps)                                    (temp, vibration,
        │                                                       pressure, current)
        ▼                                                                     ▼
┌─────────────────────┐                                      ┌───────────────────────┐
│  DETECTION          │                                      │  PREDICTION           │
│  MVT-Flow           │                                      │  Li et al. CNN        │
│  (normalizing flow, │                                      │  (RUL regressor,      │
│   PyTorch)          │                                      │   TensorFlow/Keras)   │
│                     │                                      │  + ADR→C-MAPSS proxy  │
│  → anomaly score    │                                      │  → RUL (cycles)       │
│  → healthy/warning/ │                                      │  → urgency +          │
│    anomaly          │                                      │    maintenance window │
└─────────┬───────────┘                                      └───────────┬───────────┘
          │                                                              │
          └───────────────────────────┬──────────────────────────────────┘
                                       ▼
                          ┌─────────────────────────┐
                          │  FUSION                 │
                          │  priority decision matrix│
                          │  (safety over planning) │
                          └────────────┬────────────┘
                                       ▼
              CONTINUE · MONITOR · PLAN · URGENT · STOP  (+ operator message)
```

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
