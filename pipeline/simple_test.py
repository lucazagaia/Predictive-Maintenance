"""
Simple test of the complete pipeline architecture.

Tests: ADR Sensors → Detection Interface → Prediction Interface → Fusion → Decision
"""

from detection_interface import DetectionInterface
from prediction_interface import PredictionInterface  
from fusion import MaintenanceFusion
import time


def test_pipeline():
    """Test the complete simplified pipeline."""
    print("🚀 Testing Simplified Pipeline Architecture")
    print("=" * 50)
    
    # Initialize all components
    detector = DetectionInterface()
    predictor = PredictionInterface()
    fusion = MaintenanceFusion()
    
    print("✅ All components initialized")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Normal Operation",
            "sensors": {"temperature": 75.0, "vibration": 0.3, "pressure": 15.0, "current": 10.0}
        },
        {
            "name": "Slight Degradation", 
            "sensors": {"temperature": 80.0, "vibration": 0.6, "pressure": 14.0, "current": 11.5}
        },
        {
            "name": "Anomalous Condition",
            "sensors": {"temperature": 90.0, "vibration": 1.5, "pressure": 12.0, "current": 15.0}
        }
    ]
    
    print("\n🧪 Testing Pipeline with Different Scenarios")
    print("-" * 50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        sensors = scenario['sensors']
        
        # Step 1: Detection Interface (Real-time Status)
        detection_result = detector.get_robot_status(sensors)
        print(f"   📊 Detection: {detection_result['status']} ({detection_result['confidence']:.2f})")
        
        # Step 2: Prediction Interface (Maintenance Planning)
        prediction_result = predictor.get_maintenance_planning(sensors)
        print(f"   🔮 Prediction: {prediction_result['rul_cycles']} cycles ({prediction_result['urgency']})")
        
        # Step 3: Fusion Layer (Maintenance Decision)
        decision = fusion.make_decision(detection_result, prediction_result)
        print(f"   🎯 Decision: {decision['action']} (Priority: {decision['priority']})")
        print(f"   💡 Reasoning: {decision['reasoning']}")
        
        time.sleep(0.5)  # Brief pause for readability
    
    print("\n" + "=" * 50)
    print("✅ Pipeline Test Complete!")
    print("\n📋 Architecture Summary:")
    print("   • Detection Interface: ADR sensors → Robot status")
    print("   • Prediction Interface: ADR sensors → Maintenance planning")
    print("   • Fusion Layer: Status + Planning → Actions")
    print("\n🎯 This validates the separated architecture approach!")


if __name__ == "__main__":
    test_pipeline()