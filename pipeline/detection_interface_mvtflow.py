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

# Add detection module to path
sys.path.append(str(Path(__file__).parent.parent / "detection" / "src"))

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
        
        # Initialize MVT-Flow detector
        try:
            self.detector = MVTFlowDetector(
                model_path=model_path,
                scaler_path=scaler_path,
                n_signals=n_signals,
                n_timesteps=window_size
            )
            self.model_loaded = True
            print("✅ MVT-Flow detection interface initialized")
        except Exception as e:
            print(f"⚠️  Failed to load MVT-Flow model: {e}")
            print("🔄 Using mock detection")
            self.model_loaded = False
            self.detector = None
        
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
        
        # MODEL INFERENCE
        if self.model_loaded:
            anomaly_score = self._detect_anomaly_mvtflow(sensor_window)
        else:
            anomaly_score = self._mock_detect_anomaly(sensor_window)
        
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
    
    def _mock_detect_anomaly(self, sensor_window: np.ndarray) -> float:
        """
        Mock anomaly detection for testing when model not loaded.
        
        Uses simple statistical heuristics on sensor data.
        """
        # Calculate statistical features
        mean_vals = np.mean(sensor_window, axis=2)
        std_vals = np.std(sensor_window, axis=2)
        
        # Simple anomaly score: deviation from expected ranges
        # This mimics the structure of real anomaly scores
        base_score = 70000 + np.sum(std_vals) * 1000
        noise = np.random.normal(0, 500)
        
        return base_score + noise
    
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
        # Auto-calibrate thresholds from score history
        if len(self.score_history) >= 100 and self.normal_threshold is None:
            # Use first 100 scores to establish baseline
            self.normal_threshold = np.percentile(list(self.score_history), 95)
            print(f"📊 Auto-calibrated normal threshold: {self.normal_threshold:.2f}")
        
        # Use default thresholds if not calibrated
        if self.normal_threshold is None:
            warning_threshold = 74500
            anomaly_threshold = 75000
        else:
            # Use calibrated thresholds
            warning_threshold = self.normal_threshold
            anomaly_threshold = self.normal_threshold + 500
        
        # Determine status
        if anomaly_score > anomaly_threshold:
            status = "anomaly"
            details = f"Anomaly detected (score: {anomaly_score:.1f}, threshold: {anomaly_threshold:.1f})"
            confidence = min(0.95, 0.6 + (anomaly_score - anomaly_threshold) / 1000)
            
        elif anomaly_score > warning_threshold:
            status = "warning"
            details = f"Elevated risk (score: {anomaly_score:.1f}, threshold: {warning_threshold:.1f})"
            confidence = 0.7
            
        else:
            status = "healthy"
            details = f"Normal operation (score: {anomaly_score:.1f})"
            confidence = min(0.95, 0.6 + (warning_threshold - anomaly_score) / 500)
        
        return {
            "status": status,
            "confidence": round(confidence, 3),
            "details": details,
            "anomaly_score": round(anomaly_score, 2),
            "timestamp": datetime.now().isoformat(),
            "model": "MVT-Flow"
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
        
        # Set threshold at specified percentile
        self.normal_threshold = np.percentile(scores, percentile)
        
        print(f"✅ Threshold calibrated: {self.normal_threshold:.2f}")
        print(f"   Mean normal score: {scores.mean():.2f} ± {scores.std():.2f}")
        print(f"   {percentile}th percentile: {self.normal_threshold:.2f}")
    
    def reset_buffer(self):
        """Clear sliding window buffer (for starting fresh monitoring)."""
        for buffer in self.sensor_buffers.values():
            buffer.clear()
        self.buffer_filled = False
        print("🔄 Buffer reset")









# Quick test
if __name__ == "__main__":
    # Example: Initialize interface with mock paths
    model_path = "../detection/models/mvt_flow_voraus_ad.pt"
    scaler_path = "../detection/models/scaler_voraus_ad.pkl"
    
    try:
        interface = MVTFlowDetectionInterface(
            model_path=model_path,
            scaler_path=scaler_path,
            window_size=1100,
            n_signals=130
        )
        
        # Test with random window data
        test_window = np.random.randn(1, 130, 1100)
        result = interface.get_robot_status(test_window)
        
        print("\n🧪 Test Result:")
        print(f"   Status: {result['status']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Details: {result['details']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   (Expected if model files don't exist yet)")
