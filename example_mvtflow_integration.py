"""
Example: Complete MVT-Flow Integration Pipeline

This script demonstrates how to:
1. Save the trained model from notebook
2. Load and use it in the detection pipeline
3. Process sensor data for anomaly detection
4. Handle results and trigger alerts

Run this after training MVT-Flow in the notebook.
"""

import numpy as np
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipeline.detection_interface_mvtflow import MVTFlowDetectionInterface


def save_trained_model_from_notebook():
    """
    Step 1: Save model from training notebook.
    
    Run this code in the notebook (Cell 24) to save the model:
    """
    code_to_run_in_notebook = """
    import joblib
    from pathlib import Path
    
    # Create models directory
    MODEL_SAVE_PATH = Path("../models/mvt_flow_voraus_ad.pt")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    trainer.save_model(str(MODEL_SAVE_PATH))
    
    # Save scaler
    scaler_path = MODEL_SAVE_PATH.parent / "scaler_voraus_ad.pkl"
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
    print(f"✅ Scaler saved to: {scaler_path}")
    """
    
    print("=" * 80)
    print("STEP 1: Save Model from Notebook")
    print("=" * 80)
    print("\nCopy and run this code in your training notebook (Cell 24):\n")
    print(code_to_run_in_notebook)
    print("\n" + "=" * 80 + "\n")


def example_batch_detection():
    """
    Step 2: Use saved model for batch anomaly detection.
    """
    print("=" * 80)
    print("STEP 2: Batch Anomaly Detection")
    print("=" * 80)
    
    # Paths to saved model and scaler
    model_path = "detection/models/mvt_flow_voraus_ad.pt"
    scaler_path = "detection/models/scaler_voraus_ad.pkl"
    
    try:
        # Initialize detector
        print("\n🔧 Initializing MVT-Flow detector...")
        detector = MVTFlowDetectionInterface(
            model_path=model_path,
            scaler_path=scaler_path,
            window_size=1100,
            n_signals=130
        )
        
        # Generate sample data (replace with your actual data loading)
        print("\n📊 Loading sensor data...")
        n_samples = 10
        sensor_windows = np.random.randn(n_samples, 130, 1100)
        print(f"   Data shape: {sensor_windows.shape}")
        
        # Optional: Calibrate thresholds with normal data
        print("\n🎯 Calibrating thresholds...")
        normal_data = sensor_windows[:5]  # Assume first 5 are normal
        detector.calibrate_thresholds(normal_data, percentile=95)
        
        # Run detection on all samples
        print("\n🔍 Running anomaly detection...")
        results = []
        for i, window in enumerate(sensor_windows):
            status = detector.get_robot_status(window)
            results.append(status)
            
            print(f"\n   Sample {i+1}:")
            print(f"      Status: {status['status']}")
            print(f"      Confidence: {status['confidence']:.2%}")
            print(f"      Anomaly Score: {status['anomaly_score']:.1f}")
            print(f"      Details: {status['details']}")
        
        # Summary
        status_counts = {}
        for result in results:
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for status, count in status_counts.items():
            print(f"   {status.upper()}: {count}/{n_samples} ({count/n_samples:.0%})")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model files not found")
        print(f"   {e}")
        print(f"\n💡 Did you save the model? Run Step 1 first!")
        return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Batch detection completed!\n")
    return True


def example_streaming_detection():
    """
    Step 3: Real-time streaming detection with sliding window.
    """
    print("=" * 80)
    print("STEP 3: Real-time Streaming Detection")
    print("=" * 80)
    
    model_path = "detection/models/mvt_flow_voraus_ad.pt"
    scaler_path = "detection/models/scaler_voraus_ad.pkl"
    
    try:
        # Initialize detector
        print("\n🔧 Initializing streaming detector...")
        detector = MVTFlowDetectionInterface(
            model_path=model_path,
            scaler_path=scaler_path,
            window_size=1100,
            n_signals=130
        )
        
        # Simulate real-time sensor readings (100 Hz)
        print("\n📡 Simulating real-time sensor stream (100 Hz)...")
        print("   Window size: 1100 timesteps = 11 seconds")
        
        n_timesteps = 1500  # Simulate 15 seconds of data
        anomaly_detected = False
        
        for t in range(n_timesteps):
            # Simulate current sensor reading (all 130 channels)
            current_reading = {
                f"sensor_{i}": np.random.randn() 
                for i in range(130)
            }
            
            # Add anomaly after 12 seconds (1200 timesteps)
            if t > 1200:
                # Inject spike in first few sensors
                for i in range(5):
                    current_reading[f"sensor_{i}"] += 3.0
            
            # Update detector with new reading
            status = detector.update_sensor_reading(current_reading)
            
            # Print progress every second (100 timesteps)
            if (t + 1) % 100 == 0:
                print(f"\n   t={t+1:4d} ({(t+1)/100:.1f}s): ", end="")
                print(f"Status={status['status']}, ", end="")
                
                if status['status'] != 'initializing':
                    print(f"Score={status['anomaly_score']:.1f}")
                    
                    # Trigger alert on anomaly
                    if status['status'] == 'anomaly' and not anomaly_detected:
                        print("\n   🚨 ALERT: Anomaly detected!")
                        print(f"      Confidence: {status['confidence']:.2%}")
                        print(f"      Details: {status['details']}")
                        anomaly_detected = True
                else:
                    print(f"Buffering ({t+1}/1100)")
        
        print("\n✅ Streaming simulation completed!\n")
        return True
        
    except FileNotFoundError:
        print("\n❌ Model files not found. Run Step 1 first!\n")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return False


def example_adr_integration():
    """
    Step 4: Integration with ADR 4-sensor format.
    """
    print("=" * 80)
    print("STEP 4: ADR Sensor Integration")
    print("=" * 80)
    
    print("\n📝 Example: Converting ADR sensors to MVT-Flow format\n")
    
    # Simulate ADR sensor history (4 sensors × 1100 timesteps)
    adr_history = []
    for t in range(1100):
        reading = {
            "temperature": 75.0 + np.random.randn() * 2,
            "vibration": 0.3 + np.random.randn() * 0.05,
            "pressure": 14.8 + np.random.randn() * 0.3,
            "current": 10.0 + np.random.randn() * 0.5
        }
        adr_history.append(reading)
    
    print(f"   ADR data: {len(adr_history)} timesteps")
    print(f"   Sensors: temperature, vibration, pressure, current")
    
    # Convert to MVT-Flow format
    mvtflow_window = np.zeros((1, 130, 1100))
    
    for t, reading in enumerate(adr_history):
        mvtflow_window[0, 0, t] = reading["temperature"]
        mvtflow_window[0, 1, t] = reading["vibration"]
        mvtflow_window[0, 2, t] = reading["pressure"]
        mvtflow_window[0, 3, t] = reading["current"]
        # Channels 4-129 remain zeros (or add more sensors)
    
    print(f"\n   Converted to MVT-Flow format: {mvtflow_window.shape}")
    print(f"   ✅ Ready for detection!")
    
    # Example detection
    model_path = "detection/models/mvt_flow_voraus_ad.pt"
    scaler_path = "detection/models/scaler_voraus_ad.pkl"
    
    try:
        detector = MVTFlowDetectionInterface(model_path, scaler_path)
        status = detector.get_robot_status(mvtflow_window)
        
        print(f"\n   Detection result:")
        print(f"      Status: {status['status']}")
        print(f"      Confidence: {status['confidence']:.2%}")
        print(f"      Score: {status['anomaly_score']:.1f}")
        
    except FileNotFoundError:
        print(f"\n   (Skipping detection - model not found)")
    
    print("\n")


def main():
    """Run complete integration example."""
    print("\n" + "=" * 80)
    print("MVT-FLOW INTEGRATION EXAMPLE")
    print("=" * 80 + "\n")
    
    # Step 1: Instructions for saving model
    save_trained_model_from_notebook()
    
    input("Press Enter to continue with examples (or Ctrl+C to exit)...")
    
    # Step 2: Batch detection
    if example_batch_detection():
        input("\nPress Enter to continue...")
    
    # Step 3: Streaming detection
    if example_streaming_detection():
        input("\nPress Enter to continue...")
    
    # Step 4: ADR integration
    example_adr_integration()
    
    print("=" * 80)
    print("INTEGRATION COMPLETE!")
    print("=" * 80)
    print("\n📚 For more details, see: MVT_FLOW_INTEGRATION_GUIDE.md\n")


if __name__ == "__main__":
    main()
