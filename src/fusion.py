"""
FUSION LAYER: Decision-Making Engine for Predictive Maintenance

PURPOSE:
--------
This module acts as the "brain" of the predictive maintenance system. It combines
two sources of information to make intelligent maintenance decisions:

1. DETECTION (Real-time Status): Is the robot healthy RIGHT NOW?
   - Answers: "Is it safe to operate?"
   - Output: healthy/warning/anomaly

2. PREDICTION (Future Planning): How much life is left?
   - Answers: "When should we schedule maintenance?"
   - Output: RUL cycles, urgency level, maintenance window

The fusion layer translates these ML outputs into actionable business decisions
like "stop and inspect" or "schedule maintenance in 2 weeks".

DECISION MATRIX:
---------------
The fusion layer uses a priority-based decision matrix:

                    RUL Status
              Immediate  Urgent   Soon    Planned
Status    ┌─────────────────────────────────────┐
Anomaly   │ STOP     STOP     STOP     STOP     │ Critical
Warning   │ URGENT   URGENT   MONITOR  MONITOR  │ High/Low
Healthy   │ PLAN     PLAN     CONTINUE CONTINUE │ Medium/Normal
          └─────────────────────────────────────┘

Safety first: Anomalies always trigger immediate action, regardless of RUL.
"""

from datetime import datetime
from typing import Dict


class MaintenanceFusion:
    """
    Decision-making engine that combines detection and prediction results.
    
    This class implements the fusion logic that determines:
    - What action to take (stop, schedule, monitor, continue)
    - Priority level (critical, high, medium, low, normal)
    - Business reasoning for the decision
    """
    
    def __init__(self):
        """Initialize the fusion decision engine."""
        # No configuration needed - decision logic is self-contained
        pass
    
    def make_decision(self, detection_result: Dict, prediction_result: Dict) -> Dict:
        """
        MAIN DECISION ENGINE: Combine real-time status + RUL planning → action
        
        This is the core fusion logic that translates ML outputs into maintenance
        decisions. It prioritizes safety (detection) over planning (prediction).
        
        DECISION PRIORITY:
        -----------------
        1. Safety First: Anomalies always trigger immediate action
        2. Degradation: Combine warnings with low RUL for urgent action
        3. Planning: Schedule maintenance based on RUL when robot is healthy
        4. Monitoring: Track warnings if RUL is adequate
        5. Normal Operation: Continue when all indicators are good
        
        Args:
            detection_result: Real-time health status from detection model
                             {"status": "healthy/warning/anomaly", 
                              "confidence": 0.85, 
                              "details": "..."}
            
            prediction_result: RUL estimation from prediction model
                              {"rul_cycles": 245, 
                               "urgency": "immediate/urgent/soon/planned",
                               "maintenance_window": "2-4 weeks"}
            
        Returns:
            Maintenance decision with action, priority, and reasoning:
            {"action": "schedule_maintenance",
             "priority": "medium",
             "reasoning": "Low RUL but robot status OK",
             "robot_status": "healthy",
             "rul_cycles": 245,
             "maintenance_window": "2-4 weeks",
             "timestamp": "2026-01-11T10:30:00"}
        """
        # Extract key indicators from ML models
        robot_status = detection_result["status"]      # Current health: healthy/warning/anomaly
        rul_cycles = prediction_result["rul_cycles"]   # Remaining useful life
        urgency = prediction_result["urgency"]         # RUL urgency: immediate/urgent/soon/planned
        
        # DECISION LOGIC: Apply fusion rules based on safety + planning
        
        # RULE 1: ANOMALY DETECTED → Immediate action (highest priority)
        # Safety override: Stop operations regardless of RUL prediction
        if robot_status == "anomaly":
            action = "stop_and_inspect"
            priority = "critical"
            reasoning = f"Anomaly detected: {detection_result['details']}"
        
        # RULE 2: WARNING + LOW RUL → Urgent maintenance needed
        # Degradation signs + short remaining life = high priority action.
        # Matrix: Warning × {Immediate, Urgent} → URGENT. Previously only "immediate"
        # was caught, so a warning + "urgent" RUL silently fell through to "monitor".
        elif robot_status == "warning" and urgency in ("immediate", "urgent"):
            action = "schedule_urgent_maintenance"
            priority = "high"
            reasoning = f"Warning status with {urgency} RUL need ({rul_cycles} cycles)"
        
        # RULE 3: LOW RUL + HEALTHY → Schedule maintenance soon (PLAN)
        # Robot seems fine but running out of life - plan maintenance.
        # Matrix: Healthy × {Immediate, Urgent} → PLAN (both short-horizon RUL levels
        # escalate, mirroring the Warning row; "urgent" must be included, not just
        # "immediate", or a healthy + urgent case wrongly falls through to CONTINUE.
        elif urgency in ("immediate", "urgent"):
            action = "schedule_maintenance_soon"
            priority = "medium"
            reasoning = f"Low RUL ({rul_cycles} cycles) but robot status OK"
        
        # RULE 4: WARNING + ADEQUATE RUL → Monitor closely
        # Some degradation detected but time available - watch carefully
        elif robot_status == "warning":
            action = "monitor_closely"
            priority = "low"
            reasoning = f"Warning status but adequate RUL ({rul_cycles} cycles)"
        
        # RULE 5: HEALTHY + GOOD RUL → Continue normal operations
        # All indicators good - routine operation
        else:
            action = "continue_operation"
            priority = "normal"
            reasoning = f"Healthy robot, RUL: {rul_cycles} cycles ({prediction_result['maintenance_window']})"
        
        # Return comprehensive decision package
        return {
            "action": action,                                    # What to do
            "priority": priority,                                # How urgent
            "reasoning": reasoning,                              # Why this decision
            "robot_status": robot_status,                        # Current health
            "rul_cycles": rul_cycles,                           # Remaining life
            "maintenance_window": prediction_result["maintenance_window"],  # Time window
            "timestamp": datetime.now().isoformat()              # When decided
        }


# ============================================================================
# TESTING & DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    """
    Demonstrate the fusion decision logic with various scenarios.
    
    This shows how different combinations of detection status and RUL predictions
    lead to different maintenance actions and priorities.
    """
    print("=" * 70)
    print("🧠 FUSION LAYER DECISION TESTING")
    print("=" * 70)
    
    fusion = MaintenanceFusion()
    
    # Test Scenario 1: Healthy robot with good RUL
    # Expected: Continue normal operations
    print("\n📋 Scenario 1: Healthy Robot + Good RUL")
    print("-" * 70)
    detection = {"status": "healthy", "confidence": 0.85, "details": "Normal operation"}
    prediction = {"rul_cycles": 300, "urgency": "planned", "maintenance_window": "1-2 months"}
    decision = fusion.make_decision(detection, prediction)
    print(f"Detection: {detection['status']} (confidence: {detection['confidence']})")
    print(f"Prediction: {prediction['rul_cycles']} cycles ({prediction['urgency']})")
    print(f"→ Action: {decision['action']}")
    print(f"→ Priority: {decision['priority']}")
    print(f"→ Reasoning: {decision['reasoning']}")
    
    # Test Scenario 2: Anomaly detected
    # Expected: Immediate stop and inspection (highest priority)
    print("\n📋 Scenario 2: Anomaly Detected")
    print("-" * 70)
    detection = {"status": "anomaly", "confidence": 0.92, "details": "High vibration detected"}
    prediction = {"rul_cycles": 150, "urgency": "soon", "maintenance_window": "2-3 weeks"}
    decision = fusion.make_decision(detection, prediction)
    print(f"Detection: {detection['status']} (confidence: {detection['confidence']})")
    print(f"Prediction: {prediction['rul_cycles']} cycles ({prediction['urgency']})")
    print(f"→ Action: {decision['action']}")
    print(f"→ Priority: {decision['priority']}")
    print(f"→ Reasoning: {decision['reasoning']}")
    
    # Test Scenario 3: Healthy but low RUL
    # Expected: Schedule maintenance soon (medium priority)
    print("\n📋 Scenario 3: Healthy Robot + Low RUL")
    print("-" * 70)
    detection = {"status": "healthy", "confidence": 0.78, "details": "Normal operation"}
    prediction = {"rul_cycles": 45, "urgency": "immediate", "maintenance_window": "< 1 week"}
    decision = fusion.make_decision(detection, prediction)
    print(f"Detection: {detection['status']} (confidence: {detection['confidence']})")
    print(f"Prediction: {prediction['rul_cycles']} cycles ({prediction['urgency']})")
    print(f"→ Action: {decision['action']}")
    print(f"→ Priority: {decision['priority']}")
    print(f"→ Reasoning: {decision['reasoning']}")
    
    # Test Scenario 4: Warning with low RUL
    # Expected: Urgent maintenance (high priority)
    print("\n📋 Scenario 4: Warning Status + Low RUL")
    print("-" * 70)
    detection = {"status": "warning", "confidence": 0.65, "details": "Elevated temperature and vibration"}
    prediction = {"rul_cycles": 40, "urgency": "immediate", "maintenance_window": "< 1 week"}
    decision = fusion.make_decision(detection, prediction)
    print(f"Detection: {detection['status']} (confidence: {detection['confidence']})")
    print(f"Prediction: {prediction['rul_cycles']} cycles ({prediction['urgency']})")
    print(f"→ Action: {decision['action']}")
    print(f"→ Priority: {decision['priority']}")
    print(f"→ Reasoning: {decision['reasoning']}")
    
    print("\n" + "=" * 70)
    print("✅ Fusion layer demonstration complete!")
    print("=" * 70)
