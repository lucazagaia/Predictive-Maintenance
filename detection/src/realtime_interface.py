# File: src/realtime_interface.py

import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from src.utils.data_utils import preprocess_input, EXPECTED_FEATURES

# Load pre-trained scaler and model
scaler = joblib.load("models/scaler.pkl")
model = load_model("models/cnn_rnn_model.h5")

# Function: Run inference on new sensor row
def predict_failure(sensor_row: pd.DataFrame) -> int:
    """
    Predict machine failure from a single-row DataFrame.
    Returns 1 if failure is predicted, else 0.
    """
    X_processed, _ = preprocess_input(sensor_row, scaler)
    prob = model.predict(X_processed).flatten()[0]
    return int(prob > 0.36)  # or another calibrated threshold

# Example usage (for test/demo only)
if __name__ == "__main__":
    df_new = pd.read_csv("data/sensor_data.csv")
    row = df_new.iloc[[0]]  # Get first row as DataFrame
    prediction = predict_failure(row)
    print(f"Prediction for row 0: {'Failure' if prediction == 1 else 'No Failure'}")
