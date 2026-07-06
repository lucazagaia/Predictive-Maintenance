#!/usr/bin/env python3
"""
run_demo.py — the single "run this to see it work" entry point.

Drives the pipeline the way the thesis reference architecture (Abbildung 2) wires it:
typed messages flowing over named topics between an edge node and a cloud node.

    robot   --/robot/state------>  EdgeNode  (detection)  --/edge/detection----> \
            --/robot/state------>  CloudNode (RUL)         --/cloud/prediction--> CloudNode.decide
                                                                                  --/decision--> operator

  * EdgeNode  runs the latency-critical MVT-Flow anomaly detection (on-robot).
  * CloudNode runs the heavier Li et al. RUL model + the decision matrix (backend).

Two input modalities on purpose: detection consumes a raw 130x1100 voraus-AD window;
prediction consumes a 4-channel ADR reading mapped synthetically to C-MAPSS features
(see README — the ADR->C-MAPSS map is a proxy, not real robot data).

Graceful degradation: if MVT-Flow weights or the sample windows are absent, the demo
substitutes clearly-labelled synthetic stand-ins so it always runs top to bottom.
"""

import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from messages import StateMsg, Header, TOPICS   # noqa: E402
from edge import EdgeNode                        # noqa: E402
from cloud import CloudNode                      # noqa: E402

MODELS = ROOT / "models"
SAMPLES = ROOT / "data" / "samples"
N_SIGNALS, WINDOW = 130, 1100
SEED = 0


def load_or_make_window(name: str, anomalous: bool):
    """
    Return (window_2d, provenance). Load a committed voraus-AD sample window if present,
    else synthesize a labelled stand-in so the demo still runs. The synthetic "anomalous"
    window carries more signal energy so the CONTINUE-vs-STOP contrast is visible.
    """
    path = SAMPLES / name
    if path.exists():
        window = np.load(path)
        if window.ndim == 3:
            window = window[0]
        return window.astype("float32"), "real sample"

    rng = np.random.default_rng(SEED + int(anomalous))
    window = rng.standard_normal((N_SIGNALS, WINDOW)).astype("float32")
    if anomalous:
        window[:20, :] += rng.standard_normal((20, WINDOW)).astype("float32") * 4.0
    return window, "SYNTHETIC (no sample committed)"


def main():
    print("=" * 70)
    print("PREDICTIVE MAINTENANCE — END-TO-END DEMO (edge/cloud message pipeline)")
    print("=" * 70)

    # Two nodes, wired by messages. Detection auto-falls back to a synthetic scorer if
    # its weights are missing; the RUL model loads inside the cloud node.
    edge = EdgeNode(
        model_path=str(MODELS / "mvt_flow_voraus_ad.pt"),
        scaler_path=str(MODELS / "scaler_voraus_ad.pkl"),
        window_size=WINDOW, n_signals=N_SIGNALS,
    )
    cloud = CloudNode()

    if not cloud.predictor.model_loaded:
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

        # The robot publishes /robot/state (sensor reading + detection window).
        state = StateMsg(header=Header(), window=window, **adr_reading)

        # EdgeNode: /robot/state -> /edge/detection   (real-time, on-robot)
        detection = edge.on_state(state)
        # CloudNode: /robot/state -> /cloud/prediction (backend prognostics)
        prediction = cloud.on_state(state)
        # CloudNode: fuse the two -> /decision
        decision = cloud.decide(detection, prediction)

        print("\n" + "=" * 70)
        print(f"SCENARIO: {title}   (detection window: {provenance})")
        print("-" * 70)
        print(f"  {TOPICS['state']:<19} temp={state.temperature} vib={state.vibration} "
              f"pres={state.pressure} curr={state.current}")
        print(f"  [edge]  {TOPICS['detection']:<18} {detection.status:<8} "
              f"score={detection.anomaly_score}  [{detection.model}]")
        print(f"  [cloud] {TOPICS['prediction']:<18} RUL={prediction.rul_cycles} cycles  "
              f"urgency={prediction.urgency}  window={prediction.maintenance_window}")
        print(f"  [cloud] {TOPICS['decision']:<18} {decision.action}  (priority: {decision.priority})")
        print(f"          -> {decision.operator_message}")

    # --- Real C-MAPSS validation: the RUL model on its actual benchmark domain, ---
    # --- fed real pre-normalized test windows directly (no ADR proxy involved). ---
    xpath, ypath = SAMPLES / "cmapss_sample_X.npy", SAMPLES / "cmapss_sample_y.npy"
    if xpath.exists() and ypath.exists():
        X, y_true = np.load(xpath), np.load(ypath)
        print("\n" + "=" * 70)
        print("REAL C-MAPSS TEST WINDOWS — RUL model validation (no ADR proxy)")
        print("-" * 70)
        for i in range(len(X)):
            rul = cloud.predictor.predict_rul_from_window(X[i])
            if rul != rul:  # NaN → real model unavailable
                print("  (RUL model not trained yet — run scripts/train_rul.py, then re-run this demo)")
                break
            print(f"  window {i}: predicted RUL = {rul:6.1f} cycles   |   true RUL = {int(y_true[i]):4d}")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
