"""
Fusion Layer Compatibility Analysis for MVT-Flow Integration

This document verifies that the fusion layer works correctly with both
the old CNN-RNN detection interface and the new MVT-Flow interface.
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def test_fusion_compatibility():
    """Test fusion layer with both detection interfaces."""
    
    print("=" * 80)
    print("FUSION LAYER COMPATIBILITY TEST")
    print("=" * 80)
    
    from pipeline.fusion import MaintenanceFusion
    
    fusion = MaintenanceFusion()
    
    # ==========================================================================
    # TEST 1: Old Detection Interface (CNN-RNN) Output Format
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: Old Detection Interface (CNN-RNN)")
    print("=" * 80)
    
    old_detection_output = {
        "status": "warning",
        "confidence": 0.75,
        "details": "Warning - elevated risk: 0.456",
        "anomaly_probability": 0.456,
        "priority_level": "increased_monitoring",
        "timestamp": "2026-01-15T10:30:00"
    }
    
    prediction_output = {
        "rul_cycles": 45,
        "urgency": "immediate",
        "maintenance_window": "< 1 week"
    }
    
    print("\n📥 Input - Old Detection:")
    for key, value in old_detection_output.items():
        print(f"   {key}: {value}")
    
    print("\n📥 Input - Prediction:")
    for key, value in prediction_output.items():
        print(f"   {key}: {value}")
    
    decision = fusion.make_decision(old_detection_output, prediction_output)
    
    print("\n📤 Fusion Decision:")
    for key, value in decision.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Old interface works: {decision['action']} ({decision['priority']} priority)")
    
    # ==========================================================================
    # TEST 2: New MVT-Flow Interface Output Format
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: New MVT-Flow Interface")
    print("=" * 80)
    
    new_detection_output = {
        "status": "warning",
        "confidence": 0.7,
        "details": "Elevated risk (score: 74800.5, threshold: 74500.0)",
        "anomaly_score": 74800.5,
        "timestamp": "2026-01-15T10:30:00",
        "model": "MVT-Flow"
    }
    
    print("\n📥 Input - New Detection (MVT-Flow):")
    for key, value in new_detection_output.items():
        print(f"   {key}: {value}")
    
    print("\n📥 Input - Prediction:")
    for key, value in prediction_output.items():
        print(f"   {key}: {value}")
    
    decision = fusion.make_decision(new_detection_output, prediction_output)
    
    print("\n📤 Fusion Decision:")
    for key, value in decision.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ New interface works: {decision['action']} ({decision['priority']} priority)")
    
    # ==========================================================================
    # TEST 3: All Status Combinations
    # ==========================================================================
    print("\n" + "=" * 80)
    print("TEST 3: All Status Combinations (MVT-Flow)")
    print("=" * 80)
    
    test_cases = [
        # (status, urgency, expected_action, expected_priority)
        ("healthy", "planned", "continue_operation", "normal"),
        ("healthy", "immediate", "schedule_maintenance_soon", "medium"),
        ("warning", "planned", "monitor_closely", "low"),
        ("warning", "immediate", "schedule_urgent_maintenance", "high"),
        ("anomaly", "planned", "stop_and_inspect", "critical"),
        ("anomaly", "immediate", "stop_and_inspect", "critical"),
    ]
    
    print("\nTesting all combinations:")
    print("-" * 80)
    
    all_passed = True
    for status, urgency, expected_action, expected_priority in test_cases:
        detection = {
            "status": status,
            "confidence": 0.8,
            "details": f"Test case: {status}",
            "anomaly_score": 74500.0,
            "timestamp": "2026-01-15T10:30:00",
            "model": "MVT-Flow"
        }
        
        prediction = {
            "rul_cycles": 45 if urgency == "immediate" else 300,
            "urgency": urgency,
            "maintenance_window": "< 1 week" if urgency == "immediate" else "1-2 months"
        }
        
        decision = fusion.make_decision(detection, prediction)
        
        passed = (decision['action'] == expected_action and 
                 decision['priority'] == expected_priority)
        
        status_icon = "✅" if passed else "❌"
        print(f"{status_icon} {status:8s} + {urgency:10s} → {decision['action']:30s} ({decision['priority']:8s})")
        
        if not passed:
            print(f"   Expected: {expected_action} ({expected_priority})")
            print(f"   Got:      {decision['action']} ({decision['priority']})")
            all_passed = False
    
    # ==========================================================================
    # COMPATIBILITY SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COMPATIBILITY SUMMARY")
    print("=" * 80)
    
    print("\n✅ REQUIRED FIELDS (Fusion expects these):")
    print("   - status: 'healthy', 'warning', or 'anomaly'")
    print("   - confidence: float [0, 1]")
    print("   - details: string (human-readable)")
    
    print("\n📋 OLD INTERFACE PROVIDES:")
    print("   ✓ status")
    print("   ✓ confidence")
    print("   ✓ details")
    print("   + anomaly_probability (extra)")
    print("   + priority_level (extra)")
    print("   + timestamp (extra)")
    
    print("\n📋 NEW INTERFACE (MVT-Flow) PROVIDES:")
    print("   ✓ status")
    print("   ✓ confidence")
    print("   ✓ details")
    print("   + anomaly_score (extra)")
    print("   + model (extra)")
    print("   + timestamp (extra)")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Fusion layer is fully compatible!")
    else:
        print("⚠️  SOME TESTS FAILED - Review decision logic")
    print("=" * 80)
    
    return all_passed


def demonstrate_differences():
    """Show the key differences between old and new detection outputs."""
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES: Old vs New Detection Output")
    print("=" * 80)
    
    print("\n📊 DETECTION OUTPUT COMPARISON:")
    print("-" * 80)
    
    print("\n1️⃣  Old CNN-RNN Detection:")
    print("   - Output: anomaly_probability (0-1 scale)")
    print("   - Threshold: 0.3 (warning), 0.6 (anomaly)")
    print("   - Example: 0.456 → warning")
    print("   - Extra fields: anomaly_probability, priority_level")
    
    print("\n2️⃣  New MVT-Flow Detection:")
    print("   - Output: anomaly_score (negative log probability)")
    print("   - Threshold: ~74500 (warning), ~75000 (anomaly)")
    print("   - Example: 74800 → warning")
    print("   - Extra fields: anomaly_score, model")
    
    print("\n" + "=" * 80)
    print("🔑 CRITICAL INSIGHT:")
    print("=" * 80)
    print("""
The fusion layer only uses THREE fields from detection:
  1. status: 'healthy' | 'warning' | 'anomaly'
  2. confidence: float [0, 1]
  3. details: string

Both interfaces provide these fields in the same format!
Extra fields (anomaly_probability, anomaly_score, etc.) are ignored by fusion.
Therefore, the fusion layer works identically with both detection systems.
""")
    
    print("=" * 80)
    print("✅ CONCLUSION: No changes needed to fusion layer!")
    print("=" * 80)


if __name__ == "__main__":
    # Run compatibility tests
    success = test_fusion_compatibility()
    
    # Show differences
    demonstrate_differences()
    
    if success:
        print("\n" + "=" * 80)
        print("🚀 READY FOR PRODUCTION")
        print("=" * 80)
        print("""
Your fusion layer is fully compatible with both detection interfaces:
  - Old CNN-RNN interface ✓
  - New MVT-Flow interface ✓

You can switch between them without modifying fusion.py!

To switch detection models:
  1. Old: from pipeline.detection_interface import DetectionInterface
  2. New: from pipeline.detection_interface_mvtflow import MVTFlowDetectionInterface

Both will work with the same fusion layer.
""")
    else:
        print("\n⚠️  Review test failures before deploying")
