"""
Detection Interface: Real-time Robot Health Status System

PURPOSE:
--------
This interface implements the first half of the separated architecture approach:
- Connects ADR sensor data to anomaly detection models (trained on UCI/SMARTPDM datasets)  
- Provides real-time robot health status for immediate operational decisions
- Implements virtual sensor abstraction to bridge ADR sensors → model-specific features

ARCHITECTURE PRINCIPLE:
----------------------
Following the key insight: "Models do NOT define architecture - the pipeline does"

ADR Physical Sensors → Virtual Health Indicators → Detection Model → Robot Status
    [T,V,P,C]              [Stress Indices]         [UCI Format]      [Healthy/Anomaly]

VIRTUAL SENSOR ABSTRACTION:
--------------------------
Instead of trying to match ADR sensors exactly to training data, we create 
"health indicators" that represent the same physical phenomena:

ADR Sensor          Virtual Health Indicator    Physical Meaning
----------          ------------------------    ----------------
Temperature  →      Thermal Stress Index       Heat-related degradation
Vibration    →      Mechanical Stress Index    Mechanical wear/loosening  
Pressure     →      Hydraulic Health Index     System pressure integrity
Current      →      Load Indicator Index       Electrical/motor stress

This abstraction allows any detection model to work with ADR data without
requiring identical sensor configurations.
"""

import numpy as np
import tensorflow as tf
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple


class DetectionInterface:
    """
    Real-time anomaly detection interface for ADR robots.
    
    DESIGN PHILOSOPHY:
    -----------------
    - Separation of Concerns: Only handles real-time status, not maintenance planning
    - Virtual Sensors: Maps ADR sensors to health indicators, not raw sensor matching
    - Model Agnostic: Can work with any anomaly detection model via feature mapping
    - Real-time Focus: Designed for immediate operational decisions ("Is robot OK now?")
    
    INTENDED USE:
    ------------
    - Continuous monitoring during robot operations
    - Immediate alerts for safety-critical situations  
    - Real-time dashboard status updates
    - Emergency stop decision support
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the detection interface.
        
        Args:
            model_path: Path to trained detection model (UCI/SMARTPDM format)
                       If None, uses mock implementation for testing
        
        Design Note:
        -----------
        The interface gracefully degrades to mock detection when real models
        aren't available, ensuring the pipeline always functions for testing.
        """
        self.model_path = model_path
        self.model_loaded = False
        self.model = None
        
        # Model initialization with graceful degradation
        try:
            if model_path:
                if os.path.exists(model_path):
                    print(f"🔄 Loading detection model: {os.path.basename(model_path)}")
                    self.model = tf.keras.models.load_model(model_path)
                    print("✅ Detection model loaded successfully")
                else:
                    print(f"⚠️  Model file not found: {model_path}")
                    print("🔄 Using mock detection for testing")
            else:
                print("ℹ️  No model path provided; using mock detection for testing")
            self.model_loaded = self.model is not None
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("🔄 Using mock detection")
            self.model_loaded = False
    
    def get_robot_status(self, adr_sensors: Dict[str, float]) -> Dict:
        """
        Primary interface method: Convert ADR sensors → Robot health status.
        
        This is the main entry point that external systems call for real-time
        robot health monitoring.
        
        Args:
            adr_sensors: Raw ADR sensor readings
                        {"temperature": 75.2, "vibration": 0.3, "pressure": 14.8, "current": 10.1}
        
        Returns:
            Robot status decision:
            {
                "status": "healthy|warning|anomaly",     # Operational status
                "confidence": 0.85,                      # Model confidence (0-1)  
                "details": "Normal operation: 0.123",    # Human-readable explanation
                "timestamp": "2026-01-11T10:30:00"       # When assessment was made
            }
        
        PROCESSING PIPELINE:
        -------------------
        1. PREPROCESSING: Virtual sensor abstraction and normalization
        2. MODEL INFERENCE: Anomaly detection on preprocessed features
        3. POST-PROCESSING: Convert model output to operational status
        
        DECISION LOGIC:
        --------------
        - healthy:  anomaly_prob ≤ 0.3  → Continue normal operation
        - warning:  0.3 < anomaly_prob ≤ 0.6  → Increased monitoring  
        - anomaly:  anomaly_prob > 0.6  → Immediate attention required
        
        These thresholds can be adjusted based on operational requirements and
        tolerance for false positives vs missed detections.
        """
        # STEP 1: PREPROCESSING - Virtual sensor abstraction and normalization
        preprocessed_features = self._preprocess_adr_sensors(adr_sensors)
        
        # STEP 2: MODEL INFERENCE - Run detection model on preprocessed features
        anomaly_probability = self._detect_anomaly(preprocessed_features)
        
        # STEP 3: POST-PROCESSING - Convert model output to operational status
        status_result = self._convert_to_robot_status(anomaly_probability)
        
        return status_result
    
    def _preprocess_adr_sensors(self, adr_sensors: Dict[str, float]) -> np.ndarray:
        """
        PREPROCESSING: Critical architectural component for virtual sensor abstraction.
        
        This preprocessing step is ESSENTIAL to the architecture because it enables:
        1. Model-agnostic operation (any detection model can work with ADR data)
        2. Virtual sensor abstraction (physical phenomena mapping)
        3. Scale normalization (consistent [0,1] feature ranges)
        4. Robust feature engineering (outlier handling, domain knowledge)
        
        PREPROCESSING PHILOSOPHY:
        ------------------------
        Instead of trying to replicate training sensors exactly, preprocessing creates
        "virtual sensors" that capture the same degradation phenomena that models
        were trained to recognize. This is the key insight that makes the separated
        architecture work.
        
        PREPROCESSING PIPELINE:
        ----------------------
        Raw ADR Sensors → Physical Phenomena Mapping → Normalization → Health Indicators
            [T,V,P,C]         [Stress Classifications]      [0,1 Scale]     [ML Features]
        
        ADR Sensor          Virtual Health Indicator    Physical Meaning
        ----------          ------------------------    ----------------
        Temperature  →      Thermal Stress Index       Heat-related degradation
        Vibration    →      Mechanical Stress Index    Mechanical wear/loosening  
        Pressure     →      Hydraulic Health Index     System pressure integrity
        Current      →      Load Indicator Index       Electrical/motor stress
        
        Args:
            adr_sensors: Raw sensor readings {"temperature": float, "vibration": float, 
                                            "pressure": float, "current": float}
        
        Returns:
            preprocessed_features: Normalized health indicators [0,1] ready for model input
                                 [mechanical_stress, thermal_stress, electrical_stress, hydraulic_stress]
        
        PREPROCESSING STAGES:
        --------------------
        Stage 1: Extract raw sensor values
        Stage 2: Map to physical degradation phenomena (virtual sensors)  
        Stage 3: Normalize to [0,1] health indicator scale
        Stage 4: Assemble feature vector for model input
        """
        
        # PREPROCESSING STAGE 1: Extract raw ADR sensor values
        temperature = adr_sensors["temperature"]  # °C
        vibration = adr_sensors["vibration"]      # mm/s  
        pressure = adr_sensors["pressure"]        # PSI
        current = adr_sensors["current"]          # Amperes
        
        # PREPROCESSING STAGE 2: Virtual Sensor Mapping (Physical Phenomena Extraction)
        # Map each ADR sensor to its corresponding health degradation indicator
        
        # VIRTUAL SENSOR 1: Mechanical Stress Index (from Vibration)
        # Normal vibration: 0.2-0.4 mm/s, Concerning: >0.8 mm/s, Critical: >1.2 mm/s
        mechanical_stress = self._normalize_to_health_indicator(
            value=vibration,
            healthy_min=0.1,    # Below this = perfect mechanical condition
            healthy_max=0.4,    # Above this = mechanical degradation starts  
            critical_max=1.2,   # Above this = critical mechanical condition
            sensor_name="vibration"
        )
        
        # VIRTUAL SENSOR 2: Thermal Stress Index (from Temperature)
        # Normal temp: 70-80°C, Concerning: >85°C, Critical: >95°C
        thermal_stress = self._normalize_to_health_indicator(
            value=temperature,
            healthy_min=70,     # Minimum optimal temperature
            healthy_max=80,     # Maximum optimal temperature 
            critical_max=95,    # Critical thermal threshold
            sensor_name="temperature"
        )
        
        # VIRTUAL SENSOR 3: Electrical Stress Index (from Current)
        # Normal current: 8-12A, Concerning: >13A, Critical: >16A  
        electrical_stress = self._normalize_to_health_indicator(
            value=current,
            healthy_min=8,      # Minimum normal current draw
            healthy_max=12,     # Maximum normal current draw
            critical_max=16,    # Critical electrical load threshold
            sensor_name="current"
        )
        
        # VIRTUAL SENSOR 4: Hydraulic Health Index (from Pressure)
        # Normal pressure: 14-16 PSI, Concerning: outside 12-18 PSI, Critical: outside 10-20 PSI
        pressure_deviation = abs(pressure - 15)  # Distance from optimal (15 PSI)
        hydraulic_stress = self._normalize_to_health_indicator(
            value=pressure_deviation,
            healthy_min=0,      # No deviation = perfect hydraulic health
            healthy_max=1,      # 1 PSI deviation = hydraulic degradation starts
            critical_max=5,     # 5 PSI deviation = critical hydraulic condition
            sensor_name="pressure_deviation"
        )
        
        # PREPROCESSING STAGE 3: Feature Vector Assembly
        # Combine all normalized health indicators into model-ready feature vector
        preprocessed_features = np.array([
            mechanical_stress,
            thermal_stress, 
            electrical_stress,
            hydraulic_stress
        ])
        
        # PREPROCESSING STAGE 4: Quality Validation
        # Ensure all features are within expected [0,1] range
        assert np.all((preprocessed_features >= 0) & (preprocessed_features <= 1)), \
            f"Preprocessing error: features outside [0,1] range: {preprocessed_features}"
        
        return preprocessed_features
    
    def _normalize_to_health_indicator(self, value: float, healthy_min: float, 
                                      healthy_max: float, critical_max: float, 
                                      sensor_name: str) -> float:
        """
        PREPROCESSING NORMALIZATION: Convert raw sensor value to health indicator [0,1].
        
        This normalization is a critical preprocessing step that enables virtual sensor
        abstraction by converting different sensor units and ranges to a unified
        health indicator scale.
        
        PREPROCESSING NORMALIZATION STRATEGY:
        ------------------------------------
        0.0 ←→ Healthy range (healthy_min ≤ sensor ≤ healthy_max)
        0.0 → 1.0 Linear mapping from healthy_max to critical_max  
        1.0 ←→ Critical condition (sensor ≥ critical_max)
        
        This normalization ensures:
        - All virtual sensors use the same [0,1] scale
        - Model receives consistent feature ranges regardless of physical sensor units
        - Health interpretation is intuitive (0=healthy, 1=critical)
        
        Args:
            value: Raw sensor reading (any physical unit)
            healthy_min: Lower bound of healthy operational range
            healthy_max: Upper bound of healthy operational range  
            critical_max: Value representing critical condition (maps to 1.0)
            sensor_name: Sensor identifier for debugging/logging
            
        Returns:
            health_indicator: Preprocessed normalized value [0,1] where 0=healthy, 1=critical
        """
        # PREPROCESSING ROBUSTNESS: Clamp to reasonable bounds (prevent extreme outliers)
        value = max(0, min(value, critical_max * 1.5))
        
        # PREPROCESSING LOGIC: Map sensor value to health indicator scale
        
        # Perfect health range → 0.0 (no degradation detected)
        if healthy_min <= value <= healthy_max:
            return 0.0
        
        # Degradation range → Linear mapping [0.0, 1.0]
        if value > healthy_max:
            # Map from healthy_max to critical_max → 0.0 to 1.0
            degradation_range = critical_max - healthy_max
            excess = value - healthy_max
            health_indicator = min(1.0, excess / degradation_range)
        else:
            # Value below healthy range (unusual but possible)
            # Map as moderate concern but not critical
            health_indicator = 0.2
        
        return health_indicator
    
    def _detect_anomaly(self, preprocessed_features: np.ndarray) -> float:
        """
        Run anomaly detection on PREPROCESSED health indicators.
        
        INPUT REQUIREMENTS:
        ------------------
        This method expects preprocessed features from _preprocess_adr_sensors():
        - Feature vector: [mechanical_stress, thermal_stress, electrical_stress, hydraulic_stress]
        - Value range: [0,1] normalized health indicators
        - Physical meaning: 0=healthy, 1=critical degradation
        
        MODEL INTERFACE:
        ---------------
        In production, this method interfaces with the trained detection model:
        - UCI/SMARTPDM trained model expects normalized health indicator features
        - Model outputs anomaly probability [0,1]
        - Higher values = more likely to be anomalous
        
        MOCK IMPLEMENTATION:
        -------------------
        For testing/development, we use a rule-based approach that
        mimics realistic model behavior on preprocessed features:
        
        Args:
            preprocessed_features: Normalized health indicators from preprocessing stage
                                 [mechanical, thermal, electrical, hydraulic] stress indices [0,1]
        
        Returns:
            anomaly_probability: Float [0,1] where 0=definitely healthy, 1=definitely anomalous
        """
        # MOCK DETECTION LOGIC (replace with real model inference)
        
        # Aggregate preprocessed health indicators into overall system stress
        overall_system_stress = np.mean(preprocessed_features)
        
        # Add realistic model uncertainty/noise (real models aren't perfectly deterministic)
        model_uncertainty = np.random.normal(0, 0.05)  # Small random variation
        
        # Convert aggregated health stress to anomaly probability
        base_anomaly_probability = overall_system_stress
        final_anomaly_probability = base_anomaly_probability + model_uncertainty
        
        # Clamp to valid probability range
        anomaly_probability = max(0.01, min(0.99, final_anomaly_probability))
        
        return anomaly_probability
        
        # TODO: Replace mock with actual model inference on preprocessed features:
        # return self.model.predict(preprocessed_features.reshape(1, -1))[0]
    
    def _convert_to_robot_status(self, anomaly_probability: float) -> Dict:
        """
        Convert raw anomaly probability to actionable robot status.
        
        BUSINESS LOGIC TRANSLATION:
        --------------------------
        Transforms ML model output into operational decisions that maintenance
        teams and robot operators can act upon immediately.
        
        STATUS CATEGORIES:
        -----------------
        - healthy:  Robot operating normally, continue operations
        - warning:  Elevated risk detected, increase monitoring frequency  
        - anomaly:  Significant risk detected, immediate attention required
        
        THRESHOLD RATIONALE:
        -------------------
        - 0.6 threshold for anomaly: Conservative approach prioritizing safety
        - 0.3 threshold for warning: Early warning system for proactive maintenance
        - These can be adjusted based on operational experience and false positive tolerance
        
        Args:
            anomaly_probability: Raw model output [0,1]
            
        Returns:
            status_dict: Structured robot status with reasoning
        """
        # Determine operational status based on probability thresholds
        if anomaly_probability > 0.6:
            status = "anomaly"
            details = f"Anomaly detected: {anomaly_probability:.3f}"
            priority_level = "immediate_attention"
            
        elif anomaly_probability > 0.3:
            status = "warning"  
            details = f"Warning - elevated risk: {anomaly_probability:.3f}"
            priority_level = "increased_monitoring"
            
        else:
            status = "healthy"
            details = f"Normal operation: {anomaly_probability:.3f}"
            priority_level = "routine_monitoring"
        
        # Calculate confidence metric
        # Higher confidence when probability is clearly high or low (not borderline)
        confidence = self._calculate_confidence(anomaly_probability)
        
        # Return structured status response
        return {
            "status": status,
            "confidence": confidence,
            "details": details,
            "anomaly_probability": round(anomaly_probability, 4),  # For debugging
            "priority_level": priority_level,                     # For operational planning  
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_confidence(self, anomaly_probability: float) -> float:
        """
        Calculate model confidence based on how decisive the probability is.
        
        CONFIDENCE LOGIC:
        ----------------
        - High confidence: probability near 0 (clearly healthy) or near 1 (clearly anomalous)
        - Low confidence: probability near 0.5 (uncertain/borderline case)
        
        This helps operators understand when the model is very sure vs uncertain.
        
        Args:
            anomaly_probability: Model output probability
            
        Returns:
            confidence: How confident the model is in its assessment [0.5, 0.95]
        """
        # Distance from uncertain middle point (0.5)
        decisiveness = abs(anomaly_probability - 0.5)
        
        # Map to confidence range [0.5, 0.95]
        # More decisive = higher confidence
        confidence = 0.5 + (decisiveness * 0.9)  # 0.9 = max confidence boost
        
        return round(confidence, 3)















# Quick test
if __name__ == "__main__":
    detector = DetectionInterface()
    
    # Test with normal ADR readings
    normal_sensors = {
        "temperature": 75.0,
        "vibration": 0.3,
        "pressure": 14.8, 
        "current": 10.0
    }
    
    result = detector.get_robot_status(normal_sensors)
    print("Normal sensors:", result)
    
    # Test with anomalous readings
    anomaly_sensors = {
        "temperature": 85.0,  # High temperature
        "vibration": 1.2,     # High vibration  
        "pressure": 12.0,     # Low pressure
        "current": 15.0       # High current
    }
    
    result = detector.get_robot_status(anomaly_sensors)
    print("Anomaly sensors:", result)
