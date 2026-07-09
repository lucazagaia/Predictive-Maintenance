# Notebooks

**How the models were originally trained → how that was made reproducible.** The models were
trained interactively on **Google Colab** (these notebooks — GPU runtime, Drive mounts,
`gdown`). That setup is convenient for the compute but **Colab-locked**: `google.colab`,
Drive mounts and `gdown` don't run on a normal machine, so a reviewer can't reproduce it as-is.
The training was therefore **adapted into portable `scripts/`** that run anywhere with one
command. Both are kept on purpose: the notebooks show *how it was done*, the scripts make it
*reproducible by anyone*.

Training a model has **two entry points on purpose**, matching where each model is best run:

| | Canonical (local / CLI) | Colab (free GPU) |
|---|---|---|
| RUL (light, CPU-fine) | `scripts/train_rul.py` | — |
| MVT-Flow (heavy, wants a GPU) | `scripts/train_mvtflow.py` | `detection_01_mvt_flow_training.ipynb` |

- **`scripts/`** are the reproducible, one-command trainers — no GPU or Colab assumptions.
  Use these to reproduce a model anywhere.
- **`detection_01_mvt_flow_training.ipynb`** is the same MVT-Flow training run, packaged for
  a **Colab GPU** (the normalizing-flow training is the one compute-heavy step). It is
  self-contained — it carries its own copy of the model + training code rather than importing
  the repo — so it runs in Colab without cloning a private repository. Treat it as the
  GPU counterpart of `scripts/train_mvtflow.py`, not a second source of truth.

You do **not** need any notebook to try the pipeline — use `run_demo.py`.

## Exploratory notebooks (design documentation, not trainers)

| Notebook | Purpose | Dataset |
|----------|---------|---------|
| `detection_00_preprocess_voraus.ipynb` | Inspecting/standardizing voraus-AD signals | voraus-AD |
| `prediction_01_li_cnn_modeling.ipynb` | Li et al. (2018) architecture walkthrough (original Keras exploration) | C-MAPSS FD001 |

These predate the scripts and assume a Colab environment; they are kept as a record of the
design process. Note `prediction_01` is the **original Keras** exploration — the shipped RUL
model is now PyTorch (`scripts/train_rul.py`, `src/rul_model.py`); the notebook is history, not
the current pipeline. The earlier `prediction_00_preprocess_cmapss.ipynb` was removed — it
operated on an unrelated dataset with helper imports that no longer exist.
