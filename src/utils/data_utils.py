import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ Define expected features (for training)
EXPECTED_FEATURES = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# ✅ General preprocessing (for inference or training)
def preprocess_input(df, scaler=None, expected_features=EXPECTED_FEATURES):
    df_aligned = align_features(df, expected_features)
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_aligned)
    else:
        X_scaled = scaler.transform(df_aligned)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    return X_scaled, scaler

# ✅ Training-specific function
def load_and_preprocess_data(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    X_scaled, scaler = preprocess_input(df)
    y = df['Machine failure']
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state), scaler

# ✅ Align features helper (used internally)
def align_features(df, expected_features=EXPECTED_FEATURES, fill_value=0):
    df = df.copy()
    for col in expected_features:
        if col not in df.columns:
            df[col] = fill_value
    df = df[expected_features]
    unexpected = set(df.columns) - set(expected_features)
    if unexpected:
        print(f"⚠️ Warning: Unexpected features {unexpected} will be ignored.")
        df = df.drop(columns=unexpected)
    return df