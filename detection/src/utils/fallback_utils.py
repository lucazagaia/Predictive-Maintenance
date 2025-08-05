import pandas as pd
import numpy as np

# ✅ Define expected features (same list for consistency)
EXPECTED_FEATURES = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# ✅ Step 1: Validate presence of required features
def validate_input(df, expected_features=EXPECTED_FEATURES):
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required features: {missing}")

# ✅ Step 2: Align features and fill missing columns
def align_features(df, expected_features=EXPECTED_FEATURES):
    df = df.copy()
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
    return df[expected_features]