# 🤖 Predictive Maintenance for Robot Systems

Complete predictive maintenance pipeline combining real-time **anomaly detection** and **RUL prediction** for industrial robot systems.

## Project Overview

**Components:**
- **Detection**: CNN-RNN hybrid ensemble for real-time anomaly detection (Recall: 94.7%, Precision: 85.3%)
- **Prediction**: Edge-optimized RUL models with TensorFlow Lite (<10MB, <50ms inference)
- **Fusion**: Integrated decision-making pipeline combining detection and prediction outputs

**Status:** Detection complete ✅ | Prediction models trained ✅ | Integration pipeline ready ✅

## Architecture

```
Sensor Data → Detection (anomaly) + Prediction (RUL) → Fusion → Maintenance Decision
```

## Data Flow & Communication Process

### End-to-End Pipeline: From Sensors to Maintenance Decisions

The system processes raw sensor data through multiple stages, transforming it into actionable maintenance decisions. Here's the complete communication flow:

#### **Stage 1: Sensor Data Collection**
**Format:** JSON dictionary with physical measurements
```json
{
  "temperature": 75.2,
  "vibration": 0.3,
  "pressure": 14.8,
  "current": 10.1
}
```
**Information Type:** Raw sensor readings from robot hardware  
**Communication:** Real-time data stream from IoT sensors

---

#### **Stage 2A: Detection Processing**
**Input:** Raw ADR sensor data → **Process:** Virtual sensor abstraction & normalization → CNN-RNN ensemble inference

**Output Format:** Health status assessment
```json
{
  "status": "warning",
  "confidence": 0.85,
  "details": "Elevated anomaly score: 0.42",
  "timestamp": "2026-01-13T14:23:15"
}
```
**Information Type:** Real-time operational health (healthy/warning/anomaly)  
**Inference Time:** <20ms  
**Communication:** Synchronous API response

---

#### **Stage 2B: Prediction Processing**
**Input:** Raw ADR sensor data → **Process:** Time-series windowing (30 samples) → Li et al. CNN model inference

**Output Format:** RUL estimation with maintenance planning
```json
{
  "rul_cycles": 245,
  "confidence": 0.82,
  "maintenance_window": "2-4 weeks",
  "urgency": "soon",
  "timestamp": "2026-01-13T14:23:15"
}
```
**Information Type:** Remaining useful life prediction  
**Inference Time:** <50ms  
**Communication:** Synchronous API response

---

#### **Stage 3: Fusion Layer Decision**
**Input:** Combined detection + prediction results → **Process:** Priority-based decision matrix

**Decision Matrix Logic:**
```
                           RUL Status
                 |  Immediate | Urgent  | Soon   |  Planned
    ─────────────────────────────────────────
Status  Anomaly  │  STOP       STOP       STOP      STOP
        Warning  │  URGENT     URGENT     MONITOR   MONITOR
        Healthy  │  SCHEDULE   CONTINUE   CONTINUE  CONTINUE
```

**Output Format:** Actionable maintenance decision
```json
{
  "action": "monitor_closely",
  "priority": "low",
  "reasoning": "Warning status but adequate RUL (245 cycles)",
  "robot_status": "warning",
  "rul_cycles": 245,
  "maintenance_window": "2-4 weeks",
  "timestamp": "2026-01-13T14:23:15"
}
```
**Information Type:** Maintenance command with business logic  
**Communication:** Final decision message to operators/systems

---

### Example Communication Scenarios

**Scenario 1: Critical Anomaly**
```
Sensors: {temp: 95.2, vibration: 1.8, pressure: 8.2, current: 18.5}
   ↓
Detection: {"status": "anomaly", "confidence": 0.92}
Prediction: {"rul_cycles": 150, "urgency": "soon"}
   ↓
FUSION DECISION:
{
  "action": "stop_and_inspect",
  "priority": "critical",
  "reasoning": "Anomaly detected: High vibration and temperature",
  "robot_status": "anomaly"
}
→ MESSAGE TO OPERATOR: "⛔ CRITICAL: Stop robot immediately for inspection"
```

**Scenario 2: Healthy with Low RUL**
```
Sensors: {temp: 72.1, vibration: 0.2, pressure: 14.5, current: 9.8}
   ↓
Detection: {"status": "healthy", "confidence": 0.88}
Prediction: {"rul_cycles": 45, "urgency": "immediate"}
   ↓
FUSION DECISION:
{
  "action": "schedule_maintenance_soon",
  "priority": "medium",
  "reasoning": "Low RUL (45 cycles) but robot status OK",
  "maintenance_window": "< 1 week"
}
→ MESSAGE TO OPERATOR: "📅 SCHEDULE: Plan maintenance within 1 week (45 cycles remaining)"
```

**Scenario 3: Normal Operation**
```
Sensors: {temp: 68.5, vibration: 0.15, pressure: 15.2, current: 10.0}
   ↓
Detection: {"status": "healthy", "confidence": 0.91}
Prediction: {"rul_cycles": 320, "urgency": "planned"}
   ↓
FUSION DECISION:
{
  "action": "continue_operation",
  "priority": "normal",
  "reasoning": "Healthy robot, RUL: 320 cycles (> 1 month)",
  "robot_status": "healthy"
}
→ MESSAGE TO OPERATOR: "✅ NORMAL: Continue operation - next maintenance in 1+ month"
```

## Project Structure

```
detection/
├── models/              # CNN-RNN ensemble, recall-optimized models
├── notebooks/           # 00-03: inference, exploration, optimization, ensemble
└── src/                 # model.py, data utilities

prediction/
├── models/              # Li et al. CNN models, TFLite optimized versions
├── notebooks/           # 00-01: preprocessing, DNN modeling
└── src/                 # model.py, preprocessing utilities

pipeline/
├── detection_interface.py    # Detection wrapper
├── prediction_interface.py   # Prediction wrapper
├── fusion.py                 # Decision fusion logic
└── simple_test.py           # Integration test
```

## Key Models

**Detection:**
- `cnn_rnn_hybridensemble.keras` - Production ensemble model
- `cnn_rnn_recall_optimized.keras` - High-recall variant

**Prediction:**
- `li_et_al_cnn_corrected_best.keras` - Best performing RUL model
- `robot_rul_edge_optimized.tflite` - Edge deployment version

## Quick Start

**Detection:**
```python
from pipeline.detection_interface import DetectionInterface
detector = DetectionInterface()
result = detector.predict(sensor_window)  # Returns anomaly probability
```

**Prediction:**
```python
from pipeline.prediction_interface import PredictionInterface
predictor = PredictionInterface()
result = predictor.predict(sensor_data)  # Returns RUL estimate
```

**Integrated Pipeline:**
```python
from pipeline.fusion import MaintenanceFusion
fusion = MaintenanceFusion()
decision = fusion.make_decision(detection_result, prediction_result)
# Returns: action, priority, reasoning
```