# Notebooks

Both models have a **notebook and a script**, on purpose. The scripts are the canonical
one-command trainers — run these to reproduce a model from raw data anywhere. The notebooks
are the same pipeline broken into readable stages, so preprocessing → architecture →
training → evaluation can be followed and re-run step by step, and so the GPU-heavy detector
can be trained on a free Colab runtime.

Provenance of the shipped weights, to be exact about it: `models/rul_cnn.pt` came from
`scripts/train_rul.py`, and `models/mvt_flow_voraus_ad.pt` came from the detection notebook
on a Colab GPU. `scripts/train_mvtflow.py` is the equivalent local trainer and shares its
calibration code, but the committed detector was trained by the notebook.

| Model | Canonical trainer (local / CLI) | Notebook |
|---|---|---|
| Detection — MVT-Flow (GPU-friendly) | `scripts/train_mvtflow.py` | `detection_01_mvt_flow_training.ipynb` |
| RUL — Li et al. CNN (CPU is fine) | `scripts/train_rul.py` | `prediction_01_li_cnn_modeling.ipynb` |

The notebooks are **self-contained**: each carries its own copy of the model and training
code rather than importing the repo, so it runs in Colab without cloning anything. They are
not a second source of truth — the code in them is lifted from the scripts, and the scripts
win if the two ever disagree.

You do **not** need any notebook to try the pipeline — use `run_demo.py`.

## Which file is responsible for what

Each notebook opens with this mapping, repeated here for orientation:

| Stage | Detection (voraus-AD) | Prediction (C-MAPSS) |
|---|---|---|
| Preprocessing + windowing | `scripts/train_mvtflow.py` | `scripts/train_rul.py` |
| Network definition | `src/mvt_flow_model.py` | `src/rul_model.py` |
| Training + evaluation | `scripts/train_mvtflow.py` | `scripts/train_rul.py` |
| Threshold calibration | `scripts/calibrate_detection.py` | — (RUL bands are in `src/prediction.py`) |
| Inference on live data | `src/detection.py` | `src/prediction.py` |
| Decision from model output | `src/fusion.py` | `src/fusion.py` |
| Edge / cloud wiring | `src/edge.py`, `src/cloud.py` | |

## Datasets

- **voraus-AD** (Brockmann et al. 2023) — 100 Hz parquet, ~1 GB. Not in the repo; the
  detection notebook downloads it.
- **NASA C-MAPSS FD001** — free download; the RUL notebook needs `train_FD001.txt`,
  `RUL_FD001.txt` and `test_FD001.txt`.

## Removed notebooks

`prediction_00_preprocess_cmapss.ipynb` operated on an unrelated dataset with helper imports
that no longer exist. `detection_00_preprocess_voraus.ipynb` contained no preprocessing —
only an environment-setup cell and a markdown dataset summary already covered by the README
and `docs/`. The earlier Keras version of `prediction_01` was replaced by the PyTorch
notebook above, which mirrors the trainer that produced the shipped model.
