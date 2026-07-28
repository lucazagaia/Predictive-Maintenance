"""
MVT-Flow Detection Interface — real-time anomaly detection for voraus-AD.

Wraps the MVT-Flow (Multivariate Time-series Flow) normalizing-flow model for anomaly
detection on robot machine data. It consumes multivariate time-series windows
(130 signals x 1100 timesteps) and turns the model's likelihood into a health status:

    sensor window ─▶ StandardScaler ─▶ MVT-Flow ─▶ anomaly score ─▶ healthy / warning / anomaly
    (130 x 1100)                       (PyTorch)   (neg. log-lik.)

The anomaly score is a negative log-likelihood (unbounded), so the status thresholds are
calibrated on normal data rather than fixed: scripts/calibrate_detection.py writes
models/voraus_thresholds.json, which this interface requires at load time.

Based on: Brockmann et al. (2023), The voraus-AD Dataset for Anomaly Detection in Robot
Applications (arXiv:2311.04765).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

# mvt_flow_model.py lives alongside this file in src/
sys.path.append(str(Path(__file__).parent))

from mvt_flow_model import MVTFlowDetector


class MVTFlowDetectionInterface:
    """
    Real-time anomaly detection interface using MVT-Flow.
    
    This interface handles:
    - Time series windowing for real-time streaming data
    - Model inference using trained MVT-Flow
    - Conversion of anomaly scores to operational status
    """
    
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        window_size: int = 1100,
        sampling_rate: int = 100,  # Hz
        n_signals: int = 130
    ):
        """
        Initialize MVT-Flow detection interface.
        
        Args:
            model_path: Path to trained MVT-Flow .pt file
            scaler_path: Path to StandardScaler .pkl file
            window_size: Time series window length (default: 1100 = 11 seconds at 100 Hz)
            sampling_rate: Sensor sampling rate in Hz (default: 100)
            n_signals: Number of sensor channels (default: 130)
        """
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.n_signals = n_signals
        
        # Initialize the MVT-Flow detector. The trained weights live outside the repo
        # (they are large and dataset-bound — see README). If they are absent we fall
        # back to a clearly-labelled synthetic scorer so the end-to-end demo still runs;
        # with real weights present this path does real density-based inference.
        try:
            self.detector = MVTFlowDetector(
                model_path=model_path,
                scaler_path=scaler_path,
                n_signals=n_signals,
                n_timesteps=window_size
            )
            self.model_loaded = True
            print("MVT-Flow detection interface initialized (real model)")
        except Exception as e:
            print(f"MVT-Flow weights not loaded ({e})")
            print("Falling back to synthetic detector (illustrative, not real inference)")
            self.model_loaded = False
            self.detector = None

        # Decision thresholds live with the scoring mode, because the two modes emit
        # scores on very different scales. MVT-Flow scores are unbounded log-likelihoods
        # whose range depends entirely on the trained weights, so there is no meaningful
        # default: they must come from calibration on real normal windows
        # (scripts/calibrate_detection.py writes voraus_thresholds.json next to the
        # weights). Without that file every window would score below an invented
        # threshold and be reported "healthy", so we refuse to guess.
        if self.model_loaded:
            self.mode = "MVT-Flow"
            thr_path = Path(model_path).with_name("voraus_thresholds.json")
            if not thr_path.exists():
                raise FileNotFoundError(
                    f"Calibrated thresholds not found: {thr_path}. MVT-Flow scores are "
                    "unbounded log-likelihoods and cannot be thresholded without "
                    "calibration — run scripts/calibrate_detection.py."
                )
            thr = json.loads(thr_path.read_text())
            self.warning_threshold = float(thr["warning_threshold"])
            self.anomaly_threshold = float(thr["anomaly_threshold"])
            print(f"Loaded calibrated thresholds: "
                  f"warn={self.warning_threshold:.0f}  anomaly={self.anomaly_threshold:.0f}")
        else:
            # The synthetic fallback emits a bounded [0, 1] pseudo-score, so fixed cuts
            # are meaningful here.
            self.mode = "synthetic-fallback"
            self.warning_threshold = 0.5
            self.anomaly_threshold = 0.8

    def get_robot_status(self, sensor_window: np.ndarray) -> Dict:
        """
        Primary interface: Convert sensor window → Robot health status.
        
        Args:
            sensor_window: Time series window of shape (n_samples, n_signals, n_timesteps)
                          or (n_signals, n_timesteps) for single sample
        
        Returns:
            Robot status decision:
            {
                "status": "healthy|warning|anomaly",
                "confidence": 0.85,
                "details": "...",
                "anomaly_score": 74500.2,
                "timestamp": "2026-01-15T10:30:00"
            }
        """
        # Handle single sample input
        if sensor_window.ndim == 2:
            sensor_window = sensor_window[np.newaxis, :, :]
        
        # Validate input shape. Raised rather than asserted: `python -O` strips asserts,
        # which would let a malformed window reach the model silently.
        n_samples, n_signals, n_timesteps = sensor_window.shape
        if n_signals != self.n_signals:
            raise ValueError(f"Expected {self.n_signals} signals, got {n_signals}")
        if n_timesteps != self.window_size:
            raise ValueError(f"Expected {self.window_size} timesteps, got {n_timesteps}")
        
        # Score the window: the real MVT-Flow model when its weights are loaded,
        # otherwise the synthetic fallback. Both branches assign anomaly_score.
        if self.model_loaded:
            anomaly_score = self._detect_anomaly_mvtflow(sensor_window)
        else:
            anomaly_score = self._synthetic_score(sensor_window)

        return self._convert_to_robot_status(anomaly_score)
    
    def _detect_anomaly_mvtflow(self, sensor_window: np.ndarray) -> float:
        """
        Run MVT-Flow anomaly detection.
        
        Args:
            sensor_window: Shape (n_samples, n_signals, n_timesteps)
        
        Returns:
            Anomaly score (higher = more anomalous)
        """
        scores = self.detector.predict_anomaly_score(sensor_window)
        return float(scores[0])

    def _synthetic_score(self, sensor_window: np.ndarray) -> float:
        """
        Illustrative fallback used ONLY when the real MVT-Flow weights are absent.

        Maps the window's mean absolute signal energy to a [0, 1] pseudo-score. This
        is NOT anomaly detection — it exists so the end-to-end demo runs without the
        (out-of-repo) trained weights. A higher-energy window (e.g. an injected fault
        burst) scores higher, which is enough to show the CONTINUE-vs-STOP decision
        contrast. Supply real weights in models/ for genuine results.
        """
        energy = float(np.mean(np.abs(sensor_window)))
        return float(np.clip(energy / 2.0, 0.0, 0.99))

    def _convert_to_robot_status(self, anomaly_score: float) -> Dict:
        """
        Convert a raw anomaly score to an actionable robot status.

        MVT-Flow scores are negative log-likelihoods: unbounded, and on a scale set by
        the trained weights (the shipped model produces roughly -4.3e5 for normal windows
        and +2.7e6 for anomalous ones — sign and magnitude are both weight-dependent, so
        never assume a range). Thresholds therefore always come from calibration on
        normal data, never from a constant in this file.

        Args:
            anomaly_score: Raw MVT-Flow output (negative log-likelihood)

        Returns:
            Structured robot status
        """
        # Thresholds are set per scoring mode in __init__ (and can be overridden by
        # calibrate_thresholds on real validation data). `span` is the gap between the
        # warning and anomaly thresholds; scaling confidence by it keeps this logic
        # readable on both the MVT-Flow (~500-wide) and synthetic (~0.3-wide) scales.
        warning_threshold = self.warning_threshold
        anomaly_threshold = self.anomaly_threshold
        span = max(anomaly_threshold - warning_threshold, 1e-6)

        # Determine status
        if anomaly_score > anomaly_threshold:
            status = "anomaly"
            details = f"Anomaly detected (score: {anomaly_score:.2f}, threshold: {anomaly_threshold:.2f})"
            confidence = min(0.95, 0.6 + (anomaly_score - anomaly_threshold) / span * 0.3)

        elif anomaly_score > warning_threshold:
            status = "warning"
            details = f"Elevated risk (score: {anomaly_score:.2f}, threshold: {warning_threshold:.2f})"
            confidence = 0.7

        else:
            status = "healthy"
            details = f"Normal operation (score: {anomaly_score:.2f})"
            confidence = min(0.95, 0.6 + (warning_threshold - anomaly_score) / span * 0.3)
        
        return {
            "status": status,
            "confidence": round(confidence, 3),
            "details": details,
            "anomaly_score": round(anomaly_score, 2),
            "timestamp": datetime.now().isoformat(),
            "model": self.mode
        }
