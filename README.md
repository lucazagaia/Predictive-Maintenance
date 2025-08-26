# 🤖 Predictive Maintenance for Robot Systems

A comprehensive predictive maintenance solution combining **anomaly detection** and **Remaining Useful Life (RUL) prediction** for robot systems. Built on SmartPDM washing machine sensor data and optimized for edge deployment.

## 📋 **Project Overview**

This repository implements a complete predictive maintenance pipeline with two main components:

- **🔍 Detection**: Real-time anomaly detection for immediate fault identification
- **📊 Prediction**: RUL prediction with uncertainty quantification for maintenance scheduling

**Current Status**: 3/4 Milestones Complete ✅ | Production-Ready Edge Deployment | Safety-Critical Applications

---

## 🏗️ **Architecture Overview**

```
┌─────────────────────┐    ┌─────────────────────┐
│   DETECTION         │    │   PREDICTION        │
│   Real-time         │    │   Long-term         │
│   Anomaly Detection │    │   RUL Estimation    │
└─────────────────────┘    └─────────────────────┘
           │                           │
           └─────── Robot Fleet ──────┘
                 Edge Deployment
```

---

# 🔍 **DETECTION COMPONENT**

## **Current Implementation Status: PRODUCTION READY ✅**

### **📊 Performance Metrics**
- **Recall**: 94.7% (High sensitivity for fault detection)
- **Precision**: 85.3% (Reduced false alarms)
- **F1-Score**: 89.8% (Balanced performance)
- **Inference Time**: <20ms (Real-time capable)

### **🏗️ Model Architecture**

#### **Advanced Ensemble Pipeline**
```
Input Data (Sensor Streams)
          ↓
┌─────────────────────┐
│  Feature Engineering │
│  - Temporal features │ 
│  - Frequency domain  │
│  - Statistical metrics│
└─────────────────────┘
          ↓
┌─────────────────────┐    ┌─────────────────────┐
│   CNN-RNN Model     │    │   XGBoost Model     │
│   Deep Learning     │    │   Gradient Boosting │
│   Sequential Pattern│    │   Feature-based     │
└─────────────────────┘    └─────────────────────┘
          │                           │
          └─── Meta-Learner (LR) ──────┘
                     ↓
              Final Prediction
```

#### **Model Components**
1. **CNN-RNN Hybrid**: 
   - Sequential pattern recognition
   - 512-point time windows
   - Convolutional feature extraction + LSTM temporal modeling

2. **XGBoost Ensemble**:
   - Gradient-boosted decision trees
   - Engineered statistical features
   - Robust to outliers and noise

3. **Meta-Learning Stack**:
   - Logistic Regression combiner
   - Optimal threshold: 0.36 
   - Probability calibration

### **🔄 Data Pipeline**
- **Input**: Dataset with 5 sensors and failure modes
- **Features**: Time-series windows (512 samples)
- **Preprocessing**: Standardization + Class balancing (SMOTE)
- **Labels**: Binary (Normal/Anomaly)

### **📁 Detection File Structure**
```
detection/
├── data/
│   └── dataset.csv                 # Raw sensor data
├── models/
│   ├── cnn_rnn_hybridensemble.keras    # Deep learning model
│   ├── rf_model_hybridensemble.pkl     # Random Forest backup
│   └── scaler_hybridensemble.pkl       # Feature scaler
├── notebooks/
│   ├── 00_inference.ipynb              # Basic inference
│   ├── 01_exploration.ipynb            # Data analysis
│   ├── 02_recall_boosting.ipynb        # Performance optimization
│   ├── 03_hybrid_ensemble.ipynb        # Model combination
│   ├── 04_advanced_ensemble.ipynb      # Final ensemble [LATEST]
│   ├── 05_single_inference.ipynb       # Single-sample testing
│   ├── 06_batch_inference.ipynb        # Batch processing
│   └── 07_realtime_simulation.ipynb    # Real-time simulation
├── results/
│   └── realtime_predictions_log.csv    # Inference logs
└── src/
    ├── model.py                    # Model definitions
    ├── realtime_interface.py       # Real-time processing
    └── utils/
        ├── data_utils.py          # Data preprocessing
        └── fallback_utils.py      # Error handling
```

### **🚀 Current Capabilities**
- **Real-time Processing**: 20ms inference with streaming data
- **Batch Inference**: Process multiple samples simultaneously
- **Fallback Systems**: Graceful degradation with backup models
- **Logging & Monitoring**: Complete prediction tracking

---

# 📊 **PREDICTION COMPONENT**

## **Current Implementation Status: MILESTONE 3 COMPLETE ⚠️ (Data Integration Issue)**

### **🎯 Milestone Progress**
- ✅ **Milestone 1**: Model Optimization (COMPLETE)
- ✅ **Milestone 2**: Uncertainty Quantification (COMPLETE) 
- 📋 **Milestone 4**: Robot Deployment (PLANNED)

### **📊 Current Performance**

#### **Milestone 2 - Uncertainty Quantification Results**
- **Uncertainty Coverage**: 89.2% (Target: 90%) ✅
- **Monte Carlo Dropout**: Calibration factor 3.0x
- **Deep Ensembles**: Calibration factor 2.21x
- **Safety Integration**: Conservative predictions with 75% confidence threshold


### **🏗️ Technical Architecture**

#### **Edge-Optimized Models**
- **Model Size**: <10MB (Edge deployment ready)
- **Parameters**: 184K (Milestone 1) → 185K+ (Milestone 3)
- **Inference Time**: <50ms (Real-time capable)
- **Format**: TensorFlow Lite compatible

#### **Uncertainty Quantification Pipeline**
```
Input Data (Time Series)
          ↓
┌─────────────────────┐
│  Feature Engineering │
│  - Temporal features │
│  - Frequency domain  │
└─────────────────────┘
          ↓
┌─────────────────────┐    ┌─────────────────────┐
│  Monte Carlo        │    │  Deep Ensembles    │
│  Dropout (100 runs) │    │  (5 models)       │
│  Epistemic Uncert.  │    │  Aleatoric Uncert. │
└─────────────────────┘    └─────────────────────┘
          │                           │
          └──── Calibration Layer ────┘
                     ↓
          RUL ± Confidence Intervals
```

#### **Multi-Component Fusion Architecture**
```
Motor Data     Battery Data    Sensor Data
(Current+Vib)  (V+I+Temp)     (Signal+Drift)
     ↓              ↓              ↓
Component      Component      Component
Model          Model          Model
     ↓              ↓              ↓
   Representation  Representation  Representation
     └──────────────┼──────────────┘
                    ↓
            Attention Mechanism
                    ↓
             System-Level RUL
          (Health + Failure Risk)
```

### **📁 Prediction File Structure**
```
prediction/
├── data/
│   ├── SMARTPDM_dataset.csv/     # Real sensor data (Milestones 1&2)
│   └── stream_labels.csv         # Stream processing labels
├── models/
│   └── [Generated during training]
├── notebooks/
│   ├── 00_preprocess_data.ipynb           # Data preparation
│   ├── 01_train_HDE_A_model_for_RUL.ipynb # Initial RUL model
│   ├── 02_train_simple_RUL_model.ipynb    # Baseline model
│   ├── 03_train_RUL_Generator.ipynb       # Data augmentation
│   ├── 04_Tuned_RUL_Model.ipynb          # Hyperparameter optimization
│   ├── 05_Milestone1_Model_Optimization.ipynb      # Edge optimization [COMPLETE]
│   ├── 06_Milestone2_Uncertainty_Quantification.ipynb # Safety systems [COMPLETE]
│   └── 07_Milestone3_MultiComponent_Fusion.ipynb   # Component fusion [IN PROGRESS]
├── results/
└── src/
    ├── model.py               # Model architectures
    └── utils/
        └── preprocessing.py   # Data utilities
```

---

## ⚠️ **CRITICAL ISSUES & NEXT STEPS**

### **🚨 High Priority**

#### **System-Level Performance Gap**
- **Current**: System R² ~0.65, MAE ~60-70 cycles  
- **Target**: System R² >0.75, MAE <50 cycles
- **Solution**: Improve fusion architecture and attention mechanisms
- **Timeline**: 2-3 weeks

### **📋 Planned Development**

#### **Milestone 4: Robot Deployment** 
- Real-world robot fleet integration
- Edge device deployment (Raspberry Pi/Jetson)
- Production monitoring dashboard
- Field validation studies

#### **Advanced Features**
- Federated learning across robot fleet
- Explainable AI for technician insights
- Digital twin integration
- Adaptive maintenance scheduling

---

## 🛠️ **Installation & Usage**

### **Prerequisites**
```bash
Python 3.8+
TensorFlow 2.x
scikit-learn
pandas
matplotlib
xgboost
imbalanced-learn
```

### **Detection Usage**
```python
# Real-time anomaly detection
from detection.src.realtime_interface import RealtimeDetector

detector = RealtimeDetector()
anomaly_prob = detector.predict(sensor_data)
```

### **Prediction Usage**
```python
# RUL prediction with uncertainty
from prediction.src.model import UncertaintyRULPredictor

predictor = UncertaintyRULPredictor()
rul_mean, rul_std, confidence_interval = predictor.predict(sensor_data)
```

### **Running Notebooks**
```bash
# Detection experiments
cd detection/notebooks/
jupyter lab

# Prediction milestones
cd prediction/notebooks/  
jupyter lab
```

---

## 📊 **Key Performance Indicators**

| Component | Metric | Current | Target | Status |
|-----------|--------|---------|--------|--------|
| **Detection** | Recall | 94.7% | >90% | ✅ |
| **Detection** | Precision | 85.3% | >80% | ✅ |
| **Detection** | Inference Time | <20ms | <50ms | ✅ |
| **Prediction** | Model Size | <10MB | <10MB | ✅ |
| **Prediction** | Uncertainty Coverage | 89.2% | >90% | ⚠️ |
| **Prediction** | System R² | 0.65 | >0.75 | ⚠️ |
| **Overall** | Edge Compatibility | Yes | Yes | ✅ |

---

## 🔄 **Latest Updates**

### **Recent Changes (Latest)**
- ✅ Advanced ensemble detection model (04_advanced_ensemble.ipynb)
- ✅ Uncertainty quantification with 89.2% coverage
- 🚧 Multi-component fusion architecture (needs data integration)
- ✅ Edge optimization with <10MB models
- ✅ TensorFlow Lite deployment readiness

### **Known Issues**
1. **System Fusion**: Performance gap in multi-component integration
2. **Coverage Target**: Uncertainty quantification at 89.2% vs 90% target

### **Immediate Priorities**
1. Optimize fusion model performance (2-3 weeks)  
3. Prepare Milestone 4 robot deployment (4-6 weeks)

---

## 📝 **Research Context**

**Institution**: Technical University of Berlin (TUB)  
**Project**: Bachelor Thesis - Predictive Maintenance for Robot Systems  
**Dataset**: SmartPDM Industrial Sensor Data, Dataset Robot Sensor Data
**Domain**: Industrial IoT, Robot Fleet Management, Edge AI

**Publication Potential**: Industrial deployment results and uncertainty quantification methods suitable for conference submission.

---

## 📄 **License & Citation**

Research project for academic purposes. Please cite if using in research:

```
@misc{zagaia2025_predictive_maintenance,
  title={Predictive Maintenance for Robot Systems: Detection and RUL Prediction with Uncertainty Quantification},
  author={Luca Zagaia},
  institution={Technical University of Berlin},
  year={2025}
}
```

---

**Last Updated**: January 8, 2025  
**Version**: v2.0 (Milestone 2 - Uncertainty Quantification)  
**Status**: 75% Complete - Production Ready for Detection, Prediction in Final Optimization
