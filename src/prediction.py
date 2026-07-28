"""
Prediction Interface — Remaining Useful Life (RUL) estimation.

The prediction stage of the pipeline. It maps a 4-channel ADR sensor reading onto the
17-feature C-MAPSS format (a proxy mapping — see README), runs the trained Li et al. (2018)
CNN over a 30-timestep window, and converts the estimated RUL into a maintenance-planning
recommendation (urgency + maintenance window).

Based on: Li, Ding, Sun (2018), "Remaining useful life estimation in prognostics using deep
convolution neural networks", Reliability Engineering & System Safety 172, 1–11.
"""

import numpy as np
import torch
import sys
from datetime import datetime
from typing import Dict
from pathlib import Path
import os
import json

# rul_model.py lives alongside this file in src/
sys.path.append(str(Path(__file__).parent))
from rul_model import LiCNN

# The model is trained against a piecewise-linear RUL target capped at R_EARLY
# (Li et al.'s FD001 convention, mirrored in scripts/train_rul.py), so its output is
# effectively bounded by it: degradation is only observable once a unit is within
# R_EARLY cycles of failure.
R_EARLY = 125

# Predictions are clamped to a plausible band rather than an arbitrary one. The floor is
# 0 — flooring above it would hide exactly the imminent-failure predictions this system
# exists to surface — and the ceiling is R_EARLY, past which the model cannot discriminate.
RUL_MIN, RUL_MAX = 0.0, float(R_EARLY)

# Urgency bands as fractions of R_EARLY. They must track the trained range: bands built
# for a 0-500 output (50/150/300) leave "soon" and "planned" unreachable on an
# R_EARLY=125 model, which silently kills two columns of the decision matrix.
# (cutoff, urgency, maintenance window) — first matching cutoff wins.
URGENCY_BANDS = [
    (0.20 * R_EARLY, "immediate", "< 1 week"),     # < 25 cycles
    (0.40 * R_EARLY, "urgent", "1-2 weeks"),       # < 50 cycles
    (0.72 * R_EARLY, "soon", "2-4 weeks"),         # < 90 cycles
    (float("inf"), "planned", "> 1 month"),
]


class PredictionInterface:
    """
    RUL prediction using the trained Li et al. (2018) CNN.

    1. Preprocess: ADR sensor reading → 17 C-MAPSS features (proxy mapping).
    2. Inference: CNN over the last 30 timesteps.
    3. Post-process: RUL cycles → urgency + maintenance window.
    """
    
    def __init__(self, model_path: str = None, window_size: int = 30):
        """
        Initialize RUL prediction interface with trained model.
        
        Args:
            model_path: Path to the trained Li et al. CNN weights (PyTorch .pt state_dict).
                       If None, uses ../models/rul_cnn.pt
            window_size: Number of historical readings to maintain (default: 30)
        """
        self.model = None
        self.model_loaded = False
        self.window_size = window_size
        
        # Sensor history buffer for time series (window_size × 17 features)
        # The model expects (batch, 30, 17) - 30 timesteps with 17 features each
        self.sensor_history = []
        
        # Use default model if no path provided. The RUL model is a Li et al. (2018) FD001
        # CNN (17 features = 3 operational settings + the paper's 14 sensors), trained by
        # scripts/train_rul.py and stored as a PyTorch state_dict.
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "../models/rul_cnn.pt")
        
        self.model_path = model_path
        
        # Load normalization statistics
        self.normalization_stats = self._load_normalization_stats()
        
        self._load_rul_model()
    
    def _load_normalization_stats(self) -> Dict:
        """Load the C-MAPSS normalization statistics used during training."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stats_path = os.path.join(current_dir, "../models/normalization_stats.json")
        
        try:
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Normalization stats not found: {stats_path}")
                return None
        except Exception as e:
            print(f"Failed to load normalization stats: {e}")
            return None
    
    def _load_rul_model(self):
        """Load the trained RUL model (PyTorch state_dict into a LiCNN)."""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading trained RUL model: {os.path.basename(self.model_path)}")
                self.model = LiCNN(n_features=17, window=self.window_size)
                self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                self.model.eval()
                self.model_loaded = True
                print("RUL model loaded successfully (17 features x 30 timesteps)")
            else:
                print(f"Model file not found: {self.model_path}")
                print("Using fallback placeholder prediction")
                self.model_loaded = False
        except Exception as e:
            print(f"Failed to load RUL model: {e}")
            print("Using fallback placeholder prediction")
            self.model_loaded = False
    
    def get_maintenance_planning(self, adr_sensors: Dict[str, float]) -> Dict:
        """
        MAIN RUL PREDICTION PIPELINE: ADR sensors → Maintenance recommendations
        
        This method implements the complete RUL estimation workflow:
        1. **PREPROCESSING**: Virtual sensor abstraction (ADR → C-MAPSS format)
        2. Build time series buffer (maintains last 30 readings)
        3. RUL model inference on time series data
        4. Post-processing: Convert RUL to maintenance planning
        
        PREPROCESSING PIPELINE:
        ----------------------
        The preprocessing step is ESSENTIAL for RUL estimation because:
        - Converts ADR robot sensors → C-MAPSS turbofan format  
        - Creates 17 features (3 settings + 14 sensors) from 4 ADR sensors
        - Maintains sliding window of 30 timesteps for time series analysis
        - Normalizes features to the [-1, 1] range the model was trained on
        - Enables RUL models trained on aircraft data to work on robots
        
        Args:
            adr_sensors: Raw ADR sensor readings
                        {"temperature": 75.2, "vibration": 0.3, "torque": 18.0, "current": 10.1}
            
        Returns:
            Maintenance planning information:
            {
                "rul_cycles": 245,                    # Estimated remaining useful cycles
                "confidence": 0.82,                   # Model confidence [0,1]
                "maintenance_window": "2-3 weeks",    # Human-readable timeline
                "urgency": "soon",                    # immediate/urgent/soon/planned
                "timestamp": "2026-01-11T10:30:00"    # When prediction was made
            }
        
        MAINTENANCE PLANNING LOGIC:
        --------------------------
        Bands are fractions of R_EARLY (125), the piecewise-linear cap the model is
        trained against — see URGENCY_BANDS.
        """
        # STEP 1: PREPROCESSING - Virtual sensor abstraction and normalization
        cmapss_features = self._preprocess_adr_to_cmapss(adr_sensors)
        
        # STEP 2: BUILD TIME SERIES - Maintain sliding window of readings
        self.sensor_history.append(cmapss_features)
        
        # Keep only the last window_size readings
        if len(self.sensor_history) > self.window_size:
            self.sensor_history = self.sensor_history[-self.window_size:]
        
        # STEP 3: RUL MODEL INFERENCE - Run on time series data
        rul_cycles = self._predict_rul()
        
        # STEP 4: POST-PROCESSING - Convert RUL to maintenance planning
        maintenance_plan = self._convert_to_maintenance_plan(rul_cycles)

        return maintenance_plan

    def predict_rul_from_window(self, window: np.ndarray) -> float:
        """
        Predict RUL directly from a preprocessed C-MAPSS window (30 timesteps x 17
        features, already min-max normalized to [-1, 1]).

        This bypasses the ADR->C-MAPSS proxy so the model can be validated on REAL
        C-MAPSS test windows (see data/samples/cmapss_sample_X.npy). Returns RUL in
        cycles, or NaN if the model is not loaded.
        """
        if not (self.model_loaded and self.model is not None):
            return float("nan")
        w = np.asarray(window, dtype="float32")
        if w.ndim == 2:                          # (30, 17) -> (1, 30, 17)
            w = w[np.newaxis, ...]
        with torch.no_grad():
            rul = float(self.model(torch.from_numpy(w)).item())
        return float(np.clip(rul, RUL_MIN, RUL_MAX))

    def _convert_to_maintenance_plan(self, rul_cycles: float) -> Dict:
        """
        Convert RUL prediction to maintenance planning recommendations.
        
        Args:
            rul_cycles: Remaining useful life in operational cycles
            
        Returns:
            Maintenance planning dictionary with urgency levels and timeframes
        """
        # Handle NaN or invalid RUL values
        if np.isnan(rul_cycles) or rul_cycles < 0:
            print(f"Invalid RUL value: {rul_cycles}, using default")
            rul_cycles = float(R_EARLY)

        urgency, window = URGENCY_BANDS[-1][1:]
        for cutoff, band_urgency, band_window in URGENCY_BANDS:
            if rul_cycles < cutoff:
                urgency, window = band_urgency, band_window
                break


        return {
            "rul_cycles": int(rul_cycles),
            # Illustrative fixed confidence: the Li et al. CNN is a point-estimate
            # regressor with no calibrated uncertainty head, so we do NOT fabricate a
            # per-prediction confidence. Kept constant and honest rather than random.
            "confidence": 0.80,
            "maintenance_window": window,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat()
        }
    
    def _preprocess_adr_to_cmapss(self, adr_sensors: Dict[str, float]) -> np.ndarray:
        """
        PREPROCESSING PIPELINE: ADR sensors → C-MAPSS degradation indicators
        
        This is the ESSENTIAL preprocessing step that enables RUL estimation on ADR robots.
        
        VIRTUAL SENSOR ABSTRACTION:
        ---------------------------
        Transforms 4 ADR sensors into 17 C-MAPSS-compatible features:
        
        INPUT (ADR Robot Sensors):
        - temperature: Thermal condition [°C]
        - vibration: Mechanical stress [g]
        - torque: Mechanical load [Nm]
        - current: Electrical load [A]
        
        OUTPUT (C-MAPSS Format - 17 features matching training data):
        Maps ADR sensors to realistic C-MAPSS sensor ranges, then normalizes
        using the same min-max scaling that was used during model training.
        
        Args:
            adr_sensors: Raw ADR sensor readings dictionary
            
        Returns:
            C-MAPSS compatible normalized features array [17 features]
        """
        # Extract raw ADR sensor values
        temp = adr_sensors["temperature"]
        vibr = adr_sensors["vibration"]
        torq = adr_sensors["torque"]
        curr = adr_sensors["current"]

        # Normalize ADR sensors to [0,1] range first
        temp_norm = np.clip(temp / 100.0, 0, 1)      # 0-100°C
        vibr_norm = np.clip(vibr * 2.0, 0, 1)        # 0-0.5g scaled
        torq_norm = np.clip(torq / 40.0, 0, 1)       # 0-40 Nm
        curr_norm = np.clip(curr / 12.0, 0, 1)       # 0-12A
        
        # Map normalized ADR sensors to C-MAPSS sensor value ranges
        # C-MAPSS data has specific ranges for each sensor
        
        # Operational settings (3). op_setting_3 is constant (100.0) in FD001, so it
        # normalizes to 0 — but the trained model expects it in the feature vector.
        setting_1 = -0.0087 + temp_norm * (0.0087 - (-0.0087))      # -0.0087 to 0.0087
        setting_2 = -0.0006 + torq_norm * (0.0007 - (-0.0006))      # -0.0006 to 0.0007
        setting_3 = 100.0                                            # constant in FD001

        # Temperature sensors (2, 3, 4, 7, 8)
        sensor_2 = 641.13 + temp_norm * (644.53 - 641.13)                          # Total temp at fan inlet
        sensor_3 = 1569.04 + temp_norm * (1616.91 - 1569.04)                       # Total temp at LPC outlet
        sensor_4 = 1382.25 + temp_norm * (1441.49 - 1382.25)                       # Total temp at LPC outlet (T24)
        sensor_7 = 549.85 + temp_norm * (556.06 - 549.85)                          # Total temp at HPT outlet
        sensor_8 = 2387.89 + temp_norm * 0.1 * (2388.56 - 2387.89)                 # Pressure at HPC outlet (minimal variation)

        # Load-driven sensors (9, 11, 12) — torque as a proxy for mechanical load
        sensor_9 = 9021.73 + torq_norm * (9244.59 - 9021.73)             # Physical fan speed
        sensor_11 = 46.8 + torq_norm * (48.53 - 46.8)                    # Physical core speed
        sensor_12 = 518.69 + torq_norm * (523.76 - 518.69)               # Static pressure at HPC outlet

        # Speed/vibration sensors (13, 14, 15)
        sensor_13 = 2387.88 + vibr_norm * 0.1 * (2388.56 - 2387.88)      # Corrected fan speed
        sensor_14 = 8099.94 + vibr_norm * (8293.72 - 8099.94)            # Corrected core speed
        sensor_15 = 8.3249 + (vibr_norm + temp_norm * 0.3) * 0.5 * (8.5848 - 8.3249)  # Bypass ratio

        # Flow/current sensors (17, 20, 21)
        sensor_17 = 388.0 + curr_norm * (400.0 - 388.0)                  # Physical fan speed
        sensor_20 = 38.14 + curr_norm * (39.43 - 38.14)                  # Ratio of fuel flow to Ps30
        sensor_21 = 22.8942 + curr_norm * (23.6419 - 22.8942)            # HPT coolant bleed

        # Raw C-MAPSS feature vector, in the exact order the model was trained on:
        # 3 operational settings + 14 sensors = 17 features (Li et al. FD001 sensor set).
        cmapss_raw = np.array([
            setting_1, setting_2, setting_3,   # Settings (0-2)
            sensor_2, sensor_3, sensor_4,      # Temps (2-4)
            sensor_7, sensor_8,                # Temps (7-8)
            sensor_9, sensor_11, sensor_12,    # Pressures/speeds (9-12)
            sensor_13, sensor_14, sensor_15,   # Speeds (13-15)
            sensor_17, sensor_20, sensor_21    # Flows (17, 20-21)
        ])
        
        # Apply min-max normalization using training statistics
        if self.normalization_stats:
            cmapss_normalized = self._normalize_features(cmapss_raw)
            return cmapss_normalized
        else:
            # Fallback: return as-is if normalization stats not available
            print("Using raw features without normalization")
            return cmapss_raw
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply min-max normalization using training statistics.
        
        The normalization formula from training: 2(x - x_min)/(x_max - x_min) - 1
        This normalizes to [-1, 1] range (not [0, 1])
        
        Args:
            features: Raw C-MAPSS features [17 values]

        Returns:
            Normalized features in [-1,1] range
        """
        # 17 features = 3 settings + the 14 Li et al. FD001 sensors, in training order.
        feature_columns = [
            "op_setting_1", "op_setting_2", "op_setting_3",
            "sensor_2", "sensor_3", "sensor_4",
            "sensor_7", "sensor_8", "sensor_9",
            "sensor_11", "sensor_12", "sensor_13",
            "sensor_14", "sensor_15", "sensor_17",
            "sensor_20", "sensor_21"
        ]
        
        normalized = np.zeros_like(features)
        feature_min = self.normalization_stats["feature_min"]
        feature_max = self.normalization_stats["feature_max"]
        
        for i, col in enumerate(feature_columns):
            min_val = feature_min[col]
            max_val = feature_max[col]
            
            if max_val - min_val > 0:
                # Apply the training normalization formula: 2(x - min)/(max - min) - 1
                normalized[i] = 2 * (features[i] - min_val) / (max_val - min_val) - 1
            else:
                normalized[i] = 0.0  # Constant feature → 0 in [-1, 1] range
        
        return np.clip(normalized, -1, 1)
    
    def _predict_rul(self) -> float:
        """
        Run the Li et al. CNN over the buffered time series to estimate RUL.

        Builds a (1, 30, 17) input from the last 30 preprocessed readings and returns the
        model's RUL estimate. Falls back to a placeholder if the model is not loaded.
        
        MODEL INPUT REQUIREMENTS:
        ------------------------
        - Shape: (1, 30, 17) - batch_size=1, timesteps=30, features=17
        - Time series: Last 30 sensor readings with 17 features each
        - If fewer than 30 readings available, pad with first reading
        
        Returns:
            Estimated remaining useful life in operational cycles
        """
        if self.model_loaded and self.model is not None:
            # REAL MODEL PREDICTION using trained Li et al. CNN
            try:
                # Build time series input (30 timesteps × 17 features)
                current_history_len = len(self.sensor_history)
                
                if current_history_len == 0:
                    # No data yet - return default value
                    print("No sensor history available yet")
                    return float(R_EARLY)
                
                if current_history_len < self.window_size:
                    # Pad with first reading if we don't have enough history yet
                    padding_needed = self.window_size - current_history_len
                    first_reading = self.sensor_history[0]
                    padded_history = [first_reading] * padding_needed + self.sensor_history
                    time_series = np.array(padded_history)
                else:
                    # Use the last window_size readings
                    time_series = np.array(self.sensor_history[-self.window_size:])
                
                # Reshape for model input: (1, window_size, n_features)
                model_input = time_series.reshape(1, self.window_size, time_series.shape[-1]).astype("float32")

                # Get prediction from the trained model
                with torch.no_grad():
                    rul_prediction = float(self.model(torch.from_numpy(model_input)).item())

                # Clamp to the range the model was trained to express
                rul_prediction = float(np.clip(rul_prediction, RUL_MIN, RUL_MAX))

                return rul_prediction
                
            except Exception as e:
                print(f"Model prediction failed: {e}")
                print(f"   Sensor history length: {len(self.sensor_history)}")
                print(f"   Expected input shape: (1, {self.window_size}, 17)")
                print("Falling back to placeholder prediction")
                
        # FALLBACK: deterministic estimate, used only if the model failed to load or is
        # absent. Worse mean health → less remaining life. No randomness, so the demo
        # stays reproducible.
        if len(self.sensor_history) > 0:
            health_score = float(np.mean(self.sensor_history[-1]))
        else:
            health_score = 0.5

        # Expressed on the same R_EARLY scale as the real model: a healthy window
        # (mean ~ -1 after [-1, 1] normalization) sits near the cap, a degraded one
        # well below it.
        rul = R_EARLY * (0.6 - 0.4 * health_score)
        return float(np.clip(rul, RUL_MIN, RUL_MAX))


# Quick test  
if __name__ == "__main__":
    predictor = PredictionInterface()
    
    # Test with healthy ADR readings
    healthy_sensors = {
        "temperature": 72.0,
        "vibration": 0.2,
        "torque": 12.0,
        "current": 9.5
    }

    result = predictor.get_maintenance_planning(healthy_sensors)
    print("Healthy sensors:", result)

    # Test with degraded readings
    degraded_sensors = {
        "temperature": 82.0,  # Higher temperature
        "vibration": 0.8,     # Higher vibration
        "torque": 28.0,       # Higher torque (friction / wear)
        "current": 11.5       # Higher current
    }
    
    result = predictor.get_maintenance_planning(degraded_sensors)
    print("Degraded sensors:", result)