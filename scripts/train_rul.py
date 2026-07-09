#!/usr/bin/env python3
"""
train_rul.py — end-to-end RUL training, from raw C-MAPSS FD001 to a saved model.

One command does everything: preprocessing → windowing → training → evaluation → save.
Faithful reimplementation of:
    Li, Ding, Sun (2018), "Remaining useful life estimation in prognostics using deep
    convolution neural networks", Reliability Eng. & System Safety 172, 1–11.

Get the data (free): NASA C-MAPSS "Turbofan Engine Degradation Simulation" set.
You only need FD001 — the folder must contain train_FD001.txt and RUL_FD001.txt
(test_FD001.txt optional, used for reporting RMSE).

    pip install -r requirements.txt
    python scripts/train_rul.py --cmapss-dir /path/to/CMAPSSData

Outputs (overwrites): models/rul_cnn.pt + models/normalization_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# --- Feature selection (why): FD001 runs at a single operating condition, so 7 of the
# 21 sensors are flat and carry no degradation signal. Li et al. keep 14 informative
# sensors. We ALSO keep the 3 operational settings (17 features total) to match the
# model this repo ships; op_setting_3 is constant and normalizes to 0. ---
OP_SETTINGS = [1, 2, 3]
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]  # Li et al. FD001 set
FEATURE_COLUMNS = [f"op_setting_{i}" for i in OP_SETTINGS] + [f"sensor_{i}" for i in SENSORS]

WINDOW = 30       # Li et al. Ntw for FD001
R_EARLY = 125     # piecewise-linear RUL cap
COL_NAMES = ["unit", "cycle"] + [f"op_setting_{i}" for i in (1, 2, 3)] + [f"sensor_{i}" for i in range(1, 22)]


def load_fd001(cmapss_dir: Path):
    """Read the whitespace-delimited C-MAPSS files into DataFrames."""
    import pandas as pd
    train = pd.read_csv(cmapss_dir / "train_FD001.txt", sep=r"\s+", header=None, names=COL_NAMES)
    rul_true = np.loadtxt(cmapss_dir / "RUL_FD001.txt")
    test_path = cmapss_dir / "test_FD001.txt"
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COL_NAMES) if test_path.exists() else None
    return train, test, rul_true


def add_rul(df):
    """Piecewise-linear RUL target: (max_cycle_of_unit - cycle), capped at R_EARLY."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = np.minimum(max_cycle - df["cycle"], R_EARLY)
    return df


def fit_minmax(train_df):
    """Min-max stats on the training features only (no leakage from test)."""
    fmin = {c: float(train_df[c].min()) for c in FEATURE_COLUMNS}
    fmax = {c: float(train_df[c].max()) for c in FEATURE_COLUMNS}
    return {"feature_min": fmin, "feature_max": fmax}


def normalize(df, stats):
    """Map each feature to [-1, 1]; constant features (min==max) → 0."""
    out = df.copy()
    for c in FEATURE_COLUMNS:
        lo, hi = stats["feature_min"][c], stats["feature_max"][c]
        out[c] = 0.0 if hi == lo else 2 * (df[c] - lo) / (hi - lo) - 1
    return out


def make_train_windows(df):
    """Sliding windows of length WINDOW per engine unit; label = RUL at window end."""
    X, y = [], []
    for _, g in df.groupby("unit"):
        feats = g[FEATURE_COLUMNS].to_numpy(dtype="float32")
        ruls = g["RUL"].to_numpy(dtype="float32")
        if len(feats) < WINDOW:                       # short unit: left-pad with first row
            pad = np.repeat(feats[:1], WINDOW - len(feats), axis=0)
            feats = np.vstack([pad, feats]); ruls = np.concatenate([[ruls[0]], ruls])
        for s in range(len(feats) - WINDOW + 1):
            X.append(feats[s:s + WINDOW]); y.append(ruls[s + WINDOW - 1])
    return np.asarray(X, "float32"), np.asarray(y, "float32")


def make_test_windows(df):
    """Last WINDOW cycles per unit — the input Li et al. score against RUL_FD001.txt."""
    X = []
    for _, g in df.groupby("unit"):
        feats = g[FEATURE_COLUMNS].to_numpy(dtype="float32")
        if len(feats) < WINDOW:
            feats = np.vstack([np.repeat(feats[:1], WINDOW - len(feats), axis=0), feats])
        X.append(feats[-WINDOW:])
    return np.asarray(X, "float32")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmapss-dir", required=True, help="folder with train_FD001.txt / RUL_FD001.txt")
    ap.add_argument("--epochs", type=int, default=250, help="Li et al. use 250")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from rul_model import LiCNN
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cmapss = Path(args.cmapss_dir).expanduser()
    train_df, test_df, rul_true = load_fd001(cmapss)
    train_df = add_rul(train_df)

    stats = fit_minmax(train_df)
    Xtr, ytr = make_train_windows(normalize(train_df, stats))
    print(f"train windows: {Xtr.shape}  (features = {len(FEATURE_COLUMNS)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiCNN(n_features=Xtr.shape[2], window=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        # Paper LR schedule: 1e-3 for the first 200 epochs, then 1e-4.
        for g in opt.param_groups:
            g["lr"] = 1e-3 if epoch < 200 else 1e-4
        model.train(); running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(xb)
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  mse={running / len(Xtr):.2f}")

    # Evaluate against the official test RUL labels, if the test file is present.
    if test_df is not None:
        Xte = make_test_windows(normalize(test_df, stats))
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xte).to(device)).cpu().numpy()
        pred = np.clip(pred, 0, None)
        rmse = float(np.sqrt(np.mean((pred - rul_true) ** 2)))
        print(f"\nTEST RMSE = {rmse:.2f}   (Li et al. FD001 ≈ 12.6)")

    model_path = ROOT / "models" / "rul_cnn.pt"
    stats_path = ROOT / "models" / "normalization_stats.json"
    torch.save(model.state_dict(), model_path)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"saved {model_path.relative_to(ROOT)} and {stats_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
