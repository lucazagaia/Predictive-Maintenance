"""
Fusion Layer: Combines detection status and prediction planning into maintenance decisions.

This is where the magic happens - turning ML outputs into business actions.
"""

from datetime import datetime
from typing import Dict


class MaintenanceFusion:
    """Combines robot status (detection) + maintenance planning (prediction) → actions."""
    
    def __init__(self):
        """Initialize fusion engine."""
        pass
    
    def make_decision(self, detection_result: Dict, prediction_result: Dict) -> Dict:
        """
        Combine detection status + prediction planning into maintenance decision.
        
        Args:
            detection_result: {"status": "healthy", "confidence": 0.85, "details": "..."}
            prediction_result: {"rul_cycles": 245, "urgency": "soon", "maintenance_window": "2-4 weeks"}
            
        Returns:
            {"action": "schedule_maintenance", "priority": "medium", "reasoning": "..."}
        """
        robot_status = detection_result["status"]
        rul_cycles = prediction_result["rul_cycles"] 
        urgency = prediction_result["urgency"]
        
        # Decision logic: combine real-time status + planning
        if robot_status == "anomaly":
            # Immediate action required regardless of RUL
            action = "stop_and_inspect"
            priority = "critical"
            reasoning = f"Anomaly detected: {detection_result['details']}"
            
        elif robot_status == "warning" and urgency == "immediate":
            # Warning + low RUL = urgent action
            action = "schedule_urgent_maintenance" 
            priority = "high"
            reasoning = f"Warning status with immediate RUL need ({rul_cycles} cycles)"
            
        elif urgency == "immediate":
            # Low RUL but robot seems healthy
            action = "schedule_maintenance_soon"
            priority = "medium" 
            reasoning = f"Low RUL ({rul_cycles} cycles) but robot status OK"
            
        elif robot_status == "warning":
            # Warning but RUL is OK
            action = "monitor_closely"
            priority = "low"
            reasoning = f"Warning status but adequate RUL ({rul_cycles} cycles)"
            
        else:
            # Normal operation
            action = "continue_operation"
            priority = "normal"
            reasoning = f"Healthy robot, RUL: {rul_cycles} cycles ({prediction_result['maintenance_window']})"
        
        return {
            "action": action,
            "priority": priority,
            "reasoning": reasoning,
            "robot_status": robot_status,
            "rul_cycles": rul_cycles,
            "maintenance_window": prediction_result["maintenance_window"],
            "timestamp": datetime.now().isoformat()
        }


# Quick test
if __name__ == "__main__":
    fusion = MaintenanceFusion()
    
    # Test scenario 1: Healthy robot, good RUL
    detection = {"status": "healthy", "confidence": 0.85, "details": "Normal operation"}
    prediction = {"rul_cycles": 300, "urgency": "planned", "maintenance_window": "1-2 months"}
    
    decision = fusion.make_decision(detection, prediction)
    print("Scenario 1 - Healthy + Good RUL:", decision)
    
    # Test scenario 2: Anomaly detected
    detection = {"status": "anomaly", "confidence": 0.92, "details": "High vibration detected"}
    prediction = {"rul_cycles": 150, "urgency": "soon", "maintenance_window": "2-3 weeks"}
    
    decision = fusion.make_decision(detection, prediction)
    print("Scenario 2 - Anomaly:", decision)
    
    # Test scenario 3: Healthy but low RUL
    detection = {"status": "healthy", "confidence": 0.78, "details": "Normal operation"}
    prediction = {"rul_cycles": 45, "urgency": "immediate", "maintenance_window": "< 1 week"}
    
    decision = fusion.make_decision(detection, prediction)
    print("Scenario 3 - Healthy but Low RUL:", decision)