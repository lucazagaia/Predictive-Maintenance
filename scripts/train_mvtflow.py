#!/usr/bin/env python3
"""
train_mvtflow.py — end-to-end MVT-Flow training, from the voraus-AD parquet to weights.

Preprocessing → windowing → training → AUROC → save weights + scaler + thresholds, in
one command. Faithful to:
    Brockmann, Rudolph, Rosenhahn, Wandt (2023), "The voraus-AD Dataset for Anomaly
    Detection in Robot Applications", arXiv:2311.04765.

Data (free): the voraus-AD 100 Hz parquet (https://www.tnt.uni-hannover.de/vorausAD).

    pip install -r requirements.txt          # needs torch, pandas, pyarrow, scikit-learn
    python scripts/train_mvtflow.py --parquet /path/to/voraus-ad-dataset-100hz.parquet

Outputs (overwrite): models/mvt_flow_voraus_ad.pt, models/scaler_voraus_ad.pkl,
models/voraus_thresholds.json.

WHY THIS EXISTS: the original Colab notebook collapsed the time axis (n_timesteps=1) and
flattened windows in timestep-major order while the model reshaped signal-major, scrambling
the axes. This script builds correct signal-major (130, 1100) windows — the exact layout
the inference interface (src/detection.py) feeds — so training and inference agree.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

META = ["time", "sample", "anomaly", "category", "setting", "action", "active"]
N_SIGNALS, WINDOW = 130, 1100


def build_windows(parquet_path):
    """Return (normal, anomaly) arrays of shape (n, 130, 1100), signal-major."""
    import pandas as pd
    import pyarrow.parquet as pq

    cols = pq.ParquetFile(parquet_path).schema_arrow.names
    sig_cols = [c for c in cols if c not in META]
    assert len(sig_cols) == N_SIGNALS, f"expected {N_SIGNALS} signals, got {len(sig_cols)}"

    print("Reading parquet into memory (~1 GB)...")
    tbl = pq.read_table(parquet_path, columns=sig_cols + ["sample", "anomaly"])
    sig = np.column_stack([tbl.column(c).to_numpy(zero_copy_only=False) for c in sig_cols]).astype("float32")
    sample = tbl.column("sample").to_numpy()
    anom = tbl.column("anomaly").to_numpy()

    idx = pd.Series(np.arange(len(sample))).groupby(sample).apply(lambda s: s.values).to_dict()
    label = pd.Series(anom).groupby(sample).first()

    def window(sid):
        w = sig[idx[sid]]                                    # (T, 130)  time-major rows
        w = w[:WINDOW] if len(w) >= WINDOW else np.pad(w, ((0, WINDOW - len(w)), (0, 0)), mode="edge")
        return w.T                                           # (130, 1100)  signal-major

    normal = np.stack([window(s) for s in label[~label].index])
    anomaly = np.stack([window(s) for s in label[label].index])
    print(f"windows: normal {normal.shape}, anomaly {anomaly.shape}")
    return normal, anomaly


def scale_apply(scaler, X):
    """Per-signal standardization, identical to src/mvt_flow_model.MVTFlowDetector.preprocess."""
    n, s, t = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, s)
    return scaler.transform(flat).reshape(n, t, s).transpose(0, 2, 1).astype("float32")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True, help="voraus-ad-dataset-100hz.parquet")
    ap.add_argument("--epochs", type=int, default=70)     # paper baseline
    ap.add_argument("--batch-size", type=int, default=32)  # paper baseline
    ap.add_argument("--lr", type=float, default=8e-4)      # paper baseline
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from mvt_flow_model import MVTFlow

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    normal, anomaly = build_windows(Path(args.parquet).expanduser())

    # Semi-supervised: train on NORMAL only; hold out normal + all anomaly for AUROC.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(normal))
    n_test = int(0.3 * len(normal))
    test_n, train_n = normal[perm[:n_test]], normal[perm[n_test:]]

    # Fit the scaler on training normals only (no leakage), then save it for inference.
    scaler = StandardScaler().fit(train_n.transpose(0, 2, 1).reshape(-1, N_SIGNALS))
    train_s, testn_s, testa_s = scale_apply(scaler, train_n), scale_apply(scaler, test_n), scale_apply(scaler, anomaly)

    model = MVTFlow(n_signals=N_SIGNALS, n_timesteps=WINDOW, n_blocks=4,
                    hidden_channels=2, kernel_sizes=[13, 1, 1], dilations=[2, 1, 1], alpha=1.9).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[11, 61], gamma=0.1)  # paper decay
    loader = DataLoader(TensorDataset(torch.from_numpy(train_s)), batch_size=args.batch_size, shuffle=True)

    print(f"training MVT-Flow: {len(train_n)} normal windows, {args.epochs} epochs")
    for epoch in range(args.epochs):
        model.train(); running = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = model.compute_loss(xb)                 # negative log-likelihood
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(xb)
        sched.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:2d}  nll={running / len(train_n):.2f}")

    # AUROC on held-out normal vs anomaly.
    model.eval()
    def scores(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                xb = torch.from_numpy(X[i:i + args.batch_size]).to(device)
                out.append(model.anomaly_score(xb).cpu().numpy())
        return np.concatenate(out)

    sn, sa = scores(testn_s), scores(testa_s)
    auroc = roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sa))], np.r_[sn, sa])
    warn, anom_t = float(np.percentile(sn, 95)), float(np.percentile(sn, 99))
    print(f"\nAUROC = {auroc:.3f}   (paper baseline ≈ 0.936)")
    print(f"thresholds: warning(p95)={warn:.2f}  anomaly(p99)={anom_t:.2f}")

    torch.save(model.state_dict(), ROOT / "models" / "mvt_flow_voraus_ad.pt")
    with open(ROOT / "models" / "scaler_voraus_ad.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (ROOT / "models" / "voraus_thresholds.json").write_text(json.dumps({
        "warning_threshold": warn, "anomaly_threshold": anom_t,
        "calibrated_on": f"{len(test_n)} held-out normal windows (p95/p99)",
        "auroc": round(float(auroc), 4),
    }, indent=2))
    print("saved models/mvt_flow_voraus_ad.pt, scaler_voraus_ad.pkl, voraus_thresholds.json")


if __name__ == "__main__":
    main()
