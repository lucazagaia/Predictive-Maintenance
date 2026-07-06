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
graded **1.3**). The thesis itself covered the process methodology and the edge/cloud
reference architecture; this repo is an **independent implementation** that validates
the ML approach behind it. The academic work and this code are separate efforts — this
codebase is a personal project, not thesis deliverable code.

---

## What this does

Given a sensor sample, the pipeline runs three stages and prints a maintenance action:

1. **Detection — "is the robot healthy right now?"**
   An **MVT-Flow** normalizing-flow model (PyTorch) scores a multivariate time-series
   window and returns `healthy` / `warning` / `anomaly`. Normalizing flows learn the
   density of *normal* operation, so they need only normal data at training time — a
   good fit for machines where real faults are rare and diverse.

2. **Prediction — "how much life is left?"**
   A **Li et al. (2018) 1D-CNN** (TensorFlow/Keras) estimates RUL in cycles from a
   30-step sensor window, then maps it to an urgency level and a maintenance window.

3. **Fusion — "so what do we do?"**
   A small, auditable **decision matrix** combines the two, always letting a live
   anomaly override an optimistic RUL (safety before planning), and emits an action
   plus a plain-language operator message.

`run_demo.py` runs all three end to end on sample inputs and prints the result.

---

## Architecture

```
voraus-AD window ─▶ MVT-Flow detection ─┐
                                        ├─▶ fusion matrix ─▶ CONTINUE / MONITOR /
ADR sensor reading ─▶ Li et al. RUL ────┘                   PLAN / URGENT / STOP
```

Two input modalities on purpose: detection consumes a raw multivariate window;
prediction consumes a 4-channel ADR reading. See [`docs/architecture.md`](docs/architecture.md)
for the component rationale and the full decision matrix.

```
Predictive-Maintenance/
├── run_demo.py            # ← single entry point ("run this to see it work")
├── src/
│   ├── detection.py       # MVT-Flow inference interface (+ synthetic fallback)
│   ├── mvt_flow_model.py  # MVT-Flow normalizing-flow model (PyTorch)
│   ├── prediction.py      # RUL inference interface (ADR→C-MAPSS proxy + Keras CNN)
│   └── fusion.py          # decision matrix
├── scripts/
│   ├── train_rul.py           # RUL: raw C-MAPSS FD001 → preprocess → train → save
│   ├── train_mvtflow.py       # detection: voraus parquet → windows → train → save
│   └── calibrate_detection.py # reproduce detection thresholds from normal data
├── models/                # MVT-Flow *.pt/*.pkl + voraus_thresholds.json
│                          #   (RUL .keras is produced by scripts/train_rul.py)
├── results/               # normalization_stats.json (written by train_rul.py)
├── data/samples/          # demo inputs: ADR readings, real C-MAPSS + voraus windows
├── notebooks/             # exploratory + the MVT-Flow Colab trainer (notebooks/README.md)
└── docs/                  # architecture notes
```

---

## Data

Be explicit about this, because it matters for interpreting the results:

- **Detection** is trained on **voraus-AD** (Brockmann et al., 2023) — a *real* robot
  anomaly-detection dataset (130 signals, pick-and-place manipulator), **not**
  autonomous-delivery-robot data.
- **Prediction** is trained on **NASA C-MAPSS FD001** — aircraft **turbofan**
  run-to-failure data, a standard RUL benchmark, again not robot data.
- The 4-channel ADR reading fed to the RUL stage is mapped to C-MAPSS features by a
  **hand-built proxy ("virtual sensor abstraction")** in `src/prediction.py`. This is a
  deliberate stand-in, **not** a learned or physically-calibrated mapping.

A small slice of the **real** C-MAPSS test set (3 windows + true RUL labels) is
committed at `data/samples/cmapss_sample_*.npy` so the RUL model can be validated on
genuine benchmark data, independently of the ADR proxy above.

**Why proxy data?** As discussed in the thesis, a public run-to-failure dataset from
real autonomous-delivery-robot sensor streams **does not yet exist**. This is a
well-known, field-wide limitation of ADR predictive maintenance — not a shortcut
specific to this project. The pipeline is therefore validated on the closest available
public proxies; swapping in real ADR data is future work, not a code change.

---

## How to run it

```bash
# 1. clone, then create an environment (Python 3.10–3.11 recommended)
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the end-to-end demo
python run_demo.py
```

`run_demo.py` **always runs top to bottom.** Detection uses the real MVT-Flow model when
its weights are present (`models/mvt_flow_voraus_ad.pt` + `scaler_voraus_ad.pkl`),
otherwise a clearly-labelled synthetic scorer stands in. The **RUL model is a training
output and is not committed** — until you train it (below) the demo prints placeholder
RUL values and says so; detection and fusion still run for real.

### Training the models

Both trainers go from raw public data to saved weights in one command — preprocessing,
windowing, training and evaluation included.

```bash
# RUL — Li et al. (2018) CNN on NASA C-MAPSS FD001 (~5 min, CPU is fine).
# Download the free C-MAPSS set; point at the folder with train_FD001.txt / RUL_FD001.txt.
python scripts/train_rul.py --cmapss-dir /path/to/CMAPSSData
#   → writes models/li_et_al_cnn_corrected_best.keras + results/normalization_stats.json

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

With the committed MVT-Flow weights, `run_demo.py` scores real voraus-AD windows and the
fusion safety-override fires correctly:

```
SCENARIO: Healthy operation   (detection window: real sample)
  Detection : healthy   score=365083840.0  [MVT-Flow]
  -> Action : continue_operation
  -> Operator: ✅ CONTINUE: normal operation

SCENARIO: Degraded / fault    (detection window: real sample)
  Detection : anomaly   score=448853312.0  [MVT-Flow]
  -> Action : stop_and_inspect  (priority: critical)
  -> Operator: ⛔ STOP: halt the robot and inspect immediately
```

**Honest caveat.** MVT-Flow scores are unbounded log-likelihoods, so the thresholds are
calibrated on real normal windows (`scripts/calibrate_detection.py` →
`models/voraus_thresholds.json`, p95/p99). The **committed** weights were trained with
`n_timesteps=1`, so separation is modest — only ~32% of held-out anomaly windows clear
the anomaly threshold. `scripts/train_mvtflow.py` retrains correctly on full 1100-step
windows and reports AUROC (paper baseline ≈ 0.936); use it to replace the committed
weights. Calibration and inference share preprocessing, so the demo numbers are
self-consistent.

### RUL — after training

The RUL model is produced by `scripts/train_rul.py` (see *How to run it*), which reports
test RMSE against the official C-MAPSS labels (Li et al. FD001 ≈ 12.6). Once trained, the
demo also validates it on three real, pre-normalized C-MAPSS test windows
(`data/samples/cmapss_sample_*.npy`) fed straight to the model — printing predicted-vs-true
RUL as a genuine check on the model's own benchmark domain, independent of the ADR proxy.

**Component references / targets:**
- Detection: MVT-Flow is the anomaly-detection baseline from the voraus-AD paper
  (Brockmann et al., 2023, arXiv:2311.04765); this repo re-implements it.
- Prediction: RUL CNN follows Li et al. (2018), *"Remaining useful life estimation in
  prognostics using deep convolution neural networks."*

Quantitative detection metrics from a specific training run are **not** reported here
yet — they belong with the trained weights and will be added once finalized. Presenting
them as illustrative rather than claiming benchmark numbers is deliberate.

---

## Limitations & Next Steps

Framed as a roadmap, not an apology — these are the honest edges of a portfolio prototype:

- **No ROS2 / edge integration.** Out of scope for this pass. No on-robot deployment,
  no message-bus wiring.
- **Proxy data, not real ADR streams.** Detection uses voraus-AD, prediction uses
  C-MAPSS, and the ADR→C-MAPSS sensor mapping is a hand-built proxy. Real ADR
  run-to-failure data does not yet exist publicly (see *Data*).
- **The committed detection weights are under-trained.** They were trained with
  `n_timesteps=1`, so full-window separation is modest (~32% of anomalies flagged).
  `scripts/train_mvtflow.py` retrains correctly on full 1100-step windows — the fix ships
  with the repo, it just needs a GPU run.
- **No real-time performance testing.** Latency/throughput claims are not benchmarked;
  inference is validated for correctness, not speed.
- **No production hardening.** No input validation at API boundaries, no monitoring, no
  retraining loop, no model versioning/serving.
- **RUL confidence is a placeholder.** The CNN is a point-estimate regressor; the
  reported confidence is a constant, not calibrated uncertainty.

**Next steps:** collect/obtain real ADR degradation data → replace the proxy mapping →
add calibrated uncertainty (e.g. an aleatoric two-head RUL variant) → benchmark inference
latency → then, and only then, consider edge/ROS2 integration.

---

## License

Released under the [MIT License](LICENSE).
