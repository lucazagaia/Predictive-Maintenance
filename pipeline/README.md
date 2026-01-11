Th# Predictive Maintenance Pipeline Architecture

## Simple Architecture Overview

The pipeline consists of **two independent systems** that feed into a maintenance decision layer:

```
ADR Sensor Data
       ├── Detection System (UCI Dataset) → Real-time Robot Status
       └── Prediction System (C-MAPSS Dataset) → Maintenance Planning
                                    ↓
                          Maintenance Decision Layer
                                    ↓
                            Maintenance Actions
```

## Core Components

### 1. **Detection System** (`detection/`)
- **Purpose**: Real-time anomaly detection for immediate robot status
- **Dataset**: UCI format (existing detection model)  
- **Output**: Current robot health status (Normal/Anomaly)
- **Use Case**: "Is the robot healthy right now?"

### 2. **Prediction System** (`prediction/`)
- **Purpose**: RUL estimation for maintenance planning
- **Dataset**: C-MAPSS format (existing CNN model)
- **Output**: Remaining useful life in cycles
- **Use Case**: "When should we schedule maintenance?"

### 3. **Maintenance Decision Layer** (`pipeline/core/fusion.py`)
- **Purpose**: Combine both outputs into actionable decisions
- **Logic**: 
  - Detection → Immediate status and alerts
  - Prediction → Maintenance scheduling and planning
- **Output**: Risk level + recommended actions

## Data Flow

```
Raw ADR Sensors [T, V, P, C]
        ↓
┌───────────────┬───────────────┐
│ Detection     │ Prediction    │
│ System        │ System        │
├───────────────┼───────────────┤
│ UCI Format    │ C-MAPSS       │
│ Preprocessing │ Preprocessing │
├───────────────┼───────────────┤
│ Anomaly       │ RUL           │
│ Detection     │ Estimation    │
│ Model         │ Model         │
├───────────────┼───────────────┤
│ Status:       │ Planning:     │
│ Healthy/      │ RUL = 150     │
│ Anomaly       │ cycles        │
└───────────────┴───────────────┘
        ↓
Maintenance Decision Layer
        ↓
Action: "Robot healthy, schedule 
maintenance in 2 weeks"
```

## Key Benefits

1. **Separation of Concerns**: Each system optimized for its specific dataset and purpose
2. **Real-time Status**: Detection provides immediate robot health monitoring  
3. **Planning Intelligence**: Prediction enables proactive maintenance scheduling
4. **Independent Operation**: Systems can work separately if one fails
5. **Clear Responsibilities**: Detection = status, Prediction = planning

## Implementation Structure

```
pipeline/
├── core/
│   ├── types.py           # Data structures
│   ├── fusion.py          # Maintenance decision layer
│   └── integration.py     # API interfaces
├── detection_interface.py  # Connects to detection system
├── prediction_interface.py # Connects to prediction system
└── tests/
    └── standalone_test.py  # Core logic validation
```

## Quick Start

```python
# Example usage
from pipeline.core.integration import MaintenanceSystem

# Initialize system
system = MaintenanceSystem()

# Process ADR sensor data
decision = system.process_sensor_data(
    asset_id="ROBOT_001",
    temperature=78.5,
    vibration=0.42,
    pressure=14.8,
    current=10.2
)

# Get result
print(f"Robot Status: {decision.robot_status}")      # From detection
print(f"RUL: {decision.remaining_cycles}")           # From prediction  
print(f"Action: {decision.recommended_action}")      # From fusion
```