#!/usr/bin/env python3
"""
calibrate_detection.py — derive MVT-Flow decision thresholds from real normal data.

MVT-Flow anomaly scores are unbounded log-likelihoods, so the healthy/warning/anomaly
thresholds are meaningless until calibrated on *normal* windows. This script:

  1. reads the voraus-AD parquet,
  2. builds fixed 1100-step, 130-signal windows per recording,
  3. scores a sample of NORMAL windows with the trained model,
  4. writes models/voraus_thresholds.json (warning = p95, anomaly = p99 of normal),
  5. also exports one real normal + one real anomaly window as demo inputs.

It reproduces exactly how the committed models/voraus_thresholds.json was made.

Requires the raw dataset (NOT in this repo) plus pandas + pyarrow:
    pip install pandas pyarrow
    python scripts/calibrate_detection.py --parquet /path/to/voraus-ad-dataset-100hz.parquet

NOTE ON HONESTY: the trained MVT-Flow weights were produced with n_timesteps=1 and a
preprocessing pipeline that could not be reproduced exactly from the training notebook.
Calibration and inference here use the SAME preprocessing, so the thresholds are
self-consistent, but the resulting normal/anomaly separation is modest (see README).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

META = ["time", "sample", "anomaly", "category", "setting", "action", "active"]
N_SIGNALS, WINDOW = 130, 1100


def tukey_thresholds(normal_scores):
    """
    Warning/anomaly thresholds as Tukey fences over the normal-score distribution.

    Percentiles of the normal scores cannot express both bands: taking warning=p95 and
    anomaly=p99 makes the warning band four percentiles of the normal tail by
    construction, so it is a hair's width next to the normal-anomaly gap and effectively
    never fires. Tukey fences instead measure distance beyond normal in units of the
    normal spread itself — Q3 + 1.5*IQR is the classic "mild outlier" cut, Q3 + 3*IQR the
    "extreme outlier" cut — which keeps calibration normal-only (no fault labels, as the
    semi-supervised method intends) while giving the warning band real width.

    Returns (warning, anomaly, q3, iqr).
    """
    q1, q3 = (float(v) for v in np.percentile(normal_scores, [25, 75]))
    iqr = q3 - q1
    return q3 + 1.5 * iqr, q3 + 3.0 * iqr, q3, iqr


def build_windows(parquet_path):
    """Return (window_fn, normal_ids, anomaly_ids) reading signals once into memory."""
    import pandas as pd
    import pyarrow.parquet as pq

    cols = pq.ParquetFile(parquet_path).schema_arrow.names
    sig_cols = [c for c in cols if c not in META]
    assert len(sig_cols) == N_SIGNALS, f"expected {N_SIGNALS} signals, got {len(sig_cols)}"

    print("Reading signals into memory (~1 GB)...")
    tbl = pq.read_table(parquet_path, columns=sig_cols + ["sample", "anomaly"])
    sig = np.column_stack(
        [tbl.column(c).to_numpy(zero_copy_only=False) for c in sig_cols]
    ).astype("float32")
    sample = tbl.column("sample").to_numpy()
    anom = tbl.column("anomaly").to_numpy()

    row_idx = pd.Series(np.arange(len(sample))).groupby(sample).apply(lambda s: s.values).to_dict()
    label = pd.Series(anom).groupby(sample).first()

    def window(sid):
        w = sig[row_idx[sid]]                                   # (T, 130)
        w = w[:WINDOW] if len(w) >= WINDOW else np.pad(
            w, ((0, WINDOW - len(w)), (0, 0)), mode="edge")
        return w.T[np.newaxis].astype("float32")                # (1, 130, 1100)

    return window, label[~label].index.tolist(), label[label].index.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="path to voraus-ad-dataset-100hz.parquet")
    ap.add_argument("--n-normal", type=int, default=40, help="normal windows for calibration")
    ap.add_argument("--n-anomaly", type=int, default=60, help="anomaly windows to check separation")
    args = ap.parse_args()

    from mvt_flow_model import MVTFlowDetector

    window, normal_ids, anomaly_ids = build_windows(args.parquet)
    print(f"normal samples: {len(normal_ids)} | anomaly samples: {len(anomaly_ids)}")

    det = MVTFlowDetector(
        model_path=str(ROOT / "models" / "mvt_flow_voraus_ad.pt"),
        scaler_path=str(ROOT / "models" / "scaler_voraus_ad.pkl"),
        n_signals=N_SIGNALS, n_timesteps=WINDOW,
    )

    rng = np.random.default_rng(0)
    cal_ids = list(rng.choice(normal_ids, min(args.n_normal, len(normal_ids)), replace=False))
    cal = np.concatenate([window(s) for s in cal_ids])
    cs = det.predict_anomaly_score(cal)
    warn, anom_t, q3, iqr = tukey_thresholds(cs)

    # Anomaly windows are scored to EVALUATE the thresholds, never to set them: MVT-Flow
    # is semi-supervised, and a fleet in the field has no labelled faults to calibrate on.
    test_ids = list(rng.choice(anomaly_ids, min(args.n_anomaly, len(anomaly_ids)), replace=False))
    ascore = det.predict_anomaly_score(np.concatenate([window(s) for s in test_ids]))
    print(f"normal  scores: mean={cs.mean():.3e} range[{cs.min():.3e}, {cs.max():.3e}]")
    print(f"anomaly scores: mean={ascore.mean():.3e} range[{ascore.min():.3e}, {ascore.max():.3e}]")
    print(f"normal Q3={q3:.3e}  IQR={iqr:.3e}")
    print(f"thresholds: warning(Q3+1.5*IQR)={warn:.3e}  anomaly(Q3+3*IQR)={anom_t:.3e}")
    print("evaluation against the held-out anomaly set:")
    print(f"  anomaly windows flagged anomaly:  {(ascore > anom_t).mean() * 100:.0f}%")
    print(f"  anomaly windows flagged >=warning: {(ascore > warn).mean() * 100:.0f}%")
    print(f"  normal windows false-flagged >=warning: {(cs > warn).mean() * 100:.1f}%")

    # export a representative normal + the most-detectable anomaly window for the demo
    norm_pick = cal_ids[int(np.argsort(cs)[len(cs) // 2])]
    anom_pick = test_ids[int(np.argmax(ascore))]
    samples_dir = ROOT / "data" / "samples"
    np.save(samples_dir / "voraus_normal_window.npy", window(norm_pick)[0])
    np.save(samples_dir / "voraus_anomaly_window.npy", window(anom_pick)[0])

    (ROOT / "models" / "voraus_thresholds.json").write_text(json.dumps({
        "warning_threshold": warn,
        "anomaly_threshold": anom_t,
        "calibrated_on": f"{len(cal_ids)} real voraus-AD normal windows (Tukey fences)",
        "normal_q3": q3,
        "normal_iqr": iqr,
        "method": "warning = Q3 + 1.5*IQR, anomaly = Q3 + 3*IQR of the normal scores",
    }, indent=2))
    print("Wrote models/voraus_thresholds.json and data/samples/voraus_*_window.npy")


if __name__ == "__main__":
    main()
