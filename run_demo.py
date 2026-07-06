#!/usr/bin/env python3
"""
run_demo.py — the single "run this to see it work" entry point.

Flow (one sample bundle per scenario):

    voraus-AD window ─▶ MVT-Flow detection ─┐
                                            ├─▶ fusion ─▶ CONTINUE / MONITOR /
    ADR sensor reading ─▶ Li et al. RUL ────┘             PLAN / URGENT / STOP

Two input modalities on purpose (this mirrors the architecture):
  * Detection consumes a raw 130x1100 voraus-AD window (multivariate time series).
  * Prediction consumes a 4-channel ADR reading, mapped synthetically to C-MAPSS
    features (see README — the ADR->C-MAPSS map is a proxy, not real robot data).

Graceful degradation: if the MVT-Flow weights (models/*.pt, *.pkl) or the sample
windows (data/samples/*.npy) are absent, the demo substitutes clearly-labelled
synthetic stand-ins so it always runs top to bottom. Drop the real artifacts in
place for genuine results.
"""

import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from detection import MVTFlowDetectionInterface   # noqa: E402
from prediction import PredictionInterface        # noqa: E402
from fusion import MaintenanceFusion              # noqa: E402

MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"
N_SIGNALS, WINDOW = 130, 1100
SEED = 0

# Maps the fusion layer's action code to a plain-language operator instruction.
ACTION_MESSAGE = {
    "stop_and_inspect": "⛔ STOP: halt the robot and inspect immediately",
    "schedule_urgent_maintenance": "🚨 URGENT: schedule maintenance now",
    "schedule_maintenance_soon": "📅 PLAN: schedule maintenance within the window",
    "monitor_closely": "👀 MONITOR: increase monitoring frequency",
    "continue_operation": "✅ CONTINUE: normal operation",
}


def load_or_make_window(name: str, anomalous: bool):
    """
    Return (window, provenance). Load a committed voraus-AD sample window if present,
    otherwise synthesize a labelled stand-in so the demo still runs. The synthetic
    "anomalous" window simply carries more signal energy (injected bursts) so the
    detector's CONTINUE-vs-STOP contrast is visible even without real data.
    """
    path = SAMPLES / name
    if path.exists():
        window = np.load(path)
        if window.ndim == 2:                       # (signals, timesteps) -> add batch dim
            window = window[np.newaxis, ...]
        return window.astype("float32"), "real sample"

    rng = np.random.default_rng(SEED + int(anomalous))
    window = rng.standard_normal((1, N_SIGNALS, WINDOW)).astype("float32")
    if anomalous:
        window[:, :20, :] += rng.standard_normal((1, 20, WINDOW)).astype("float32") * 4.0
    return window, "SYNTHETIC (no sample committed)"


def main():
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE — END-TO-END DEMO")
    print("=" * 70)

    # Initialise the three stages once. Detection auto-falls back to a synthetic
    # scorer if the weights below are missing; prediction loads the real Keras model.
    detector = MVTFlowDetectionInterface(
        model_path=str(MODELS / "mvt_flow_voraus_ad.pt"),
        scaler_path=str(MODELS / "scaler_voraus_ad.pkl"),
        window_size=WINDOW,
        n_signals=N_SIGNALS,
    )
    predictor = PredictionInterface()
    fusion = MaintenanceFusion()

    if not predictor.model_loaded:
        print("\n⚠️  RUL model not found — showing PLACEHOLDER predictions below.")
        print("    Train the real model (~5 min on CPU):")
        print("    python scripts/train_rul.py --cmapss-dir <path-to-CMAPSS-FD001>\n")

    adr = json.loads((SAMPLES / "sample_adr_readings.json").read_text())

    scenarios = [
        ("Healthy operation", "voraus_normal_window.npy", False, adr["healthy"]),
        ("Degraded / fault", "voraus_anomaly_window.npy", True, adr["degraded"]),
    ]

    for title, win_name, anomalous, adr_reading in scenarios:
        window, provenance = load_or_make_window(win_name, anomalous)

        # STAGE 1 — detection on the raw window.
        status = detector.get_robot_status(window)

        # STAGE 2 — RUL. Warm up the predictor's 30-step history with this reading
        # (a fresh, constant history per scenario keeps the demo deterministic).
        predictor.sensor_history = []
        plan = None
        for _ in range(predictor.window_size):
            plan = predictor.get_maintenance_planning(adr_reading)

        # STAGE 3 — fusion decision.
        decision = fusion.make_decision(status, plan)

        print("\n" + "=" * 70)
        print(f"SCENARIO: {title}   (detection window: {provenance})")
        print("-" * 70)
        print(f"  Detection : {status['status']:<8}  score={status['anomaly_score']}  [{status['model']}]")
        print(f"  Prediction: RUL={plan['rul_cycles']} cycles  urgency={plan['urgency']}  window={plan['maintenance_window']}")
        print(f"  -> Action : {decision['action']}  (priority: {decision['priority']})")
        print(f"  -> Reason : {decision['reasoning']}")
        print(f"  -> Operator: {ACTION_MESSAGE.get(decision['action'], decision['action'])}")

    # --- Real C-MAPSS validation: the RUL model on its actual benchmark domain, ---
    # --- fed real pre-normalized test windows directly (no ADR proxy involved). ---
    xpath, ypath = SAMPLES / "cmapss_sample_X.npy", SAMPLES / "cmapss_sample_y.npy"
    if xpath.exists() and ypath.exists():
        X, y_true = np.load(xpath), np.load(ypath)
        print("\n" + "=" * 70)
        print("REAL C-MAPSS TEST WINDOWS — RUL model validation (no ADR proxy)")
        print("-" * 70)
        for i in range(len(X)):
            rul = predictor.predict_rul_from_window(X[i])
            if rul != rul:  # NaN → real model unavailable
                print("  (RUL model not trained yet — run scripts/train_rul.py, then re-run this demo)")
                break
            print(f"  window {i}: predicted RUL = {rul:6.1f} cycles   |   true RUL = {int(y_true[i]):4d}")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
