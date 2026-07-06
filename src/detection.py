"""
MVT-Flow Detection Interface: Real-time Anomaly Detection for voraus-AD

This is an updated detection interface that uses MVT-Flow (Multivariate Time-series Flow)
for anomaly detection on robot sensor data.

KEY DIFFERENCES FROM ORIGINAL:
------------------------------
1. Uses MVT-Flow normalizing flow instead of CNN-RNN
2. Requires multivariate time series windows (130 signals × 1100 timesteps)
3. Real-time sliding window approach for continuous monitoring
4. Direct sensor-to-model mapping (no virtual sensor abstraction needed)

ARCHITECTURE:
------------
ADR Sensors → Time Window Buffer → MVT-Flow Model → Anomaly Score → Robot Status
  [T,V,P,C]      [130×1100 window]    [PyTorch]      [0-1 prob]    [Healthy/Anomaly]

Based on: Brockmann et al. (2023) - voraus-AD Dataset
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from collections import deque

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
            print("✅ MVT-Flow detection interface initialized (real model)")
        except Exception as e:
            print(f"⚠️  MVT-Flow weights not loaded ({e})")
            print("🔄 Falling back to synthetic detector (illustrative, not real inference)")
            self.model_loaded = False
            self.detector = None

        # Decision thresholds live with the scoring mode, because the two modes emit
        # scores on very different scales. MVT-Flow scores are unbounded log-likelihoods,
        # so they are meaningless without calibration on normal data — the hardcoded
        # numbers below are only placeholders. If a calibration file produced from real
        # normal windows sits next to the weights, it overrides them (see README →
        # "Calibrating detection").
        if self.model_loaded:
            self.mode = "MVT-Flow"
            self.warning_threshold = 74500.0
            self.anomaly_threshold = 75000.0
            thr_path = Path(model_path).with_name("voraus_thresholds.json")
            if thr_path.exists():
                import json
                thr = json.loads(thr_path.read_text())
                self.warning_threshold = float(thr["warning_threshold"])
                self.anomaly_threshold = float(thr["anomaly_threshold"])
                print(f"📊 Loaded calibrated thresholds: "
                      f"warn={self.warning_threshold:.0f}  anomaly={self.anomaly_threshold:.0f}")
        else:
            self.mode = "synthetic-fallback"
            self.warning_threshold = 0.5
            self.anomaly_threshold = 0.8
        
        # Initialize sliding window buffer for real-time streaming
        # Each sensor gets its own deque for efficient append/pop
        self.sensor_buffers = {
            f"sensor_{i}": deque(maxlen=window_size) 
            for i in range(n_signals)
        }
        self.buffer_filled = False
        
        # Anomaly score statistics for threshold calibration
        self.score_history = deque(maxlen=1000)
        self.normal_threshold = None
    
    def update_sensor_reading(self, sensor_data: Dict[str, float]) -> Dict:
        """
        Process single timestep of sensor data (real-time streaming mode).
        
        This method handles real-time data streams where sensors send
        readings one timestep at a time (100 Hz sampling).
        
        Args:
            sensor_data: Dictionary mapping sensor names to values
                        {"sensor_0": 0.5, "sensor_1": 0.3, ...}
        
        Returns:
            Robot status decision (or None if window not yet filled)
        """
        # Update sliding window buffer
        for sensor_name, value in sensor_data.items():
            if sensor_name in self.sensor_buffers:
                self.sensor_buffers[sensor_name].append(value)
        
        # Check if we have enough data for a complete window
        if not self.buffer_filled:
            buffer_sizes = [len(buf) for buf in self.sensor_buffers.values()]
            if min(buffer_sizes) >= self.window_size:
                self.buffer_filled = True
                print(f"✅ Buffer filled: {self.window_size} timesteps ready")
        
        # Only make predictions once buffer is filled
        if self.buffer_filled:
            # Extract current window from buffers
            window_data = self._extract_window_from_buffers()
            
            # Run anomaly detection on window
            return self.get_robot_status(window_data)
        else:
            return {
                "status": "initializing",
                "confidence": 0.0,
                "details": f"Buffering data: {min([len(b) for b in self.sensor_buffers.values()])}/{self.window_size}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _extract_window_from_buffers(self) -> np.ndarray:
        """
        Extract current time window from sliding buffers.
        
        Returns:
            Window data of shape (1, n_signals, window_size)
        """
        window = np.zeros((1, self.n_signals, self.window_size))
        
        for i in range(self.n_signals):
            sensor_name = f"sensor_{i}"
            if sensor_name in self.sensor_buffers:
                window[0, i, :] = list(self.sensor_buffers[sensor_name])
        
        return window
    
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
        
        # Validate input shape
        n_samples, n_signals, n_timesteps = sensor_window.shape
        assert n_signals == self.n_signals, \
            f"Expected {self.n_signals} signals, got {n_signals}"
        assert n_timesteps == self.window_size, \
            f"Expected {self.window_size} timesteps, got {n_timesteps}"
        
        # MODEL INFERENCE — real MVT-Flow, or synthetic fallback on the same path.
        # Either branch assigns anomaly_score (the original code left it undefined
        # when the model was not loaded, which crashed the fallback).
        if self.model_loaded:
            anomaly_score = self._detect_anomaly_mvtflow(sensor_window)
        else:
            anomaly_score = self._synthetic_score(sensor_window)

        # Update score history
        self.score_history.append(anomaly_score)
        
        # Convert score to operational status
        status_result = self._convert_to_robot_status(anomaly_score)
        
        return status_result
    
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
        Convert raw anomaly score to actionable robot status.
        
        THRESHOLD CALIBRATION:
        ---------------------
        For MVT-Flow, anomaly scores are negative log probabilities.
        Typical ranges on voraus-AD:
        - Normal: 73,000 - 74,500
        - Anomaly: 74,500 - 76,000+
        
        These thresholds should be calibrated on validation data.
        
        Args:
            anomaly_score: Raw MVT-Flow output (negative log probability)
        
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
    
    def calibrate_thresholds(
        self,
        normal_data: np.ndarray,
        percentile: float = 95.0
    ):
        """
        Calibrate anomaly thresholds using normal validation data.
        
        This should be called once with a representative sample of normal
        operation data to establish baseline thresholds.
        
        Args:
            normal_data: Normal sensor windows (n_samples, n_signals, n_timesteps)
            percentile: Percentile for threshold (default: 95 = allow 5% false positives)
        """
        if not self.model_loaded:
            print("⚠️  Model not loaded, cannot calibrate")
            return
        
        print(f"🔧 Calibrating thresholds on {len(normal_data)} normal samples...")
        
        # Compute anomaly scores for all normal samples
        scores = self.detector.predict_anomaly_score(normal_data)
        
        # Set thresholds at the specified percentile of normal scores.
        self.normal_threshold = float(np.percentile(scores, percentile))
        self.warning_threshold = self.normal_threshold
        self.anomaly_threshold = self.normal_threshold + 500.0

        print(f"✅ Threshold calibrated: {self.normal_threshold:.2f}")
        print(f"   Mean normal score: {scores.mean():.2f} ± {scores.std():.2f}")
        print(f"   {percentile}th percentile: {self.normal_threshold:.2f}")
    
    def reset_buffer(self):
        """Clear sliding window buffer (for starting fresh monitoring)."""
        for buffer in self.sensor_buffers.values():
            buffer.clear()
        self.buffer_filled = False
        print("🔄 Buffer reset")
