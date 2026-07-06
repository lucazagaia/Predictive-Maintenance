# Notebooks

The **canonical, click-run trainers are the scripts** in [`../scripts/`](../scripts) —
they go from raw public data to saved weights in one command:

- `python scripts/train_rul.py --cmapss-dir <FD001>` — RUL CNN (C-MAPSS).
- `python scripts/train_mvtflow.py --parquet <voraus.parquet>` — MVT-Flow detector.

You do **not** need any notebook to try the pipeline — use `run_demo.py`.

## What's here

| Notebook | Purpose | Dataset |
|----------|---------|---------|
| `detection_01_mvt_flow_training.ipynb` | **Colab GPU** wrapper around `scripts/train_mvtflow.py` (mount Drive → fetch parquet → train → save). Use this for MVT-Flow, which wants a GPU. | voraus-AD (Brockmann et al. 2023) |
| `detection_00_preprocess_voraus.ipynb` | Exploratory: inspecting/standardizing voraus-AD signals. | voraus-AD |
| `prediction_01_li_cnn_modeling.ipynb` | Exploratory: the Li et al. (2018) architecture walkthrough. Superseded for training by `scripts/train_rul.py`. | NASA C-MAPSS FD001 |

The exploratory notebooks (`*_00`, `prediction_01`) predate the scripts and assume a
Colab environment; they are kept as design documentation. The earlier
`prediction_00_preprocess_cmapss.ipynb` was removed — it operated on an unrelated dataset
and its helper imports no longer exist.
