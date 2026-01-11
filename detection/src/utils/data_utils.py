"""
Data Processing Utilities for Predictive Maintenance

Main Functions:
    - load_and_preprocess_data(): Complete pipeline for training data
    - preprocess_input(): Process new data for inference
    - align_features(): Ensure data has expected feature columns
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Optional, List

# Expected UCI dataset features (in correct order)
EXPECTED_FEATURES = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]


def load_and_preprocess_data(
    path: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], StandardScaler]:
    """
    Load and preprocess UCI dataset for training.
    
    Args:
        path: Path to CSV dataset file
        test_size: Proportion for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: ((X_train, X_test, y_train, y_test), scaler)
    """
    print(f"📊 Loading dataset from: {path}")
    
    # Load data
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Prepare features and target
    df_features = align_features(df, EXPECTED_FEATURES)
    X = df_features.values
    y = df['Machine failure'].values
    
    print(f"Features: {df_features.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Scale and reshape for CNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"✅ Training: {X_train.shape}, Test: {X_test.shape}")
    
    return (X_train, X_test, y_train, y_test), scaler


def align_features(
    df: pd.DataFrame, 
    expected_features: List[str] = EXPECTED_FEATURES, 
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Ensure DataFrame has expected features in correct order.
    
    Args:
        df: Input DataFrame
        expected_features: Required feature names in correct order
        fill_value: Value for missing features (default: 0.0)
        
    Returns:
        DataFrame with expected features only
    """
    df = df.copy()
    
    # Add missing features
    for col in expected_features:
        if col not in df.columns:
            df[col] = fill_value
    
    # Return only expected features
    return df[expected_features]




# def preprocess_input(
#     df: pd.DataFrame, 
#     scaler: Optional[StandardScaler] = None, 
#     expected_features: List[str] = EXPECTED_FEATURES
# ) -> Tuple[np.ndarray, StandardScaler]:
#     """
#     Preprocess data for inference or training.
    
#     Args:
#         df: DataFrame with sensor data
#         scaler: Pre-fitted scaler (None for training, fitted for inference)
#         expected_features: List of required feature names
        
#     Returns:
#         tuple: (X_scaled, scaler)
#     """
#     # Align features
#     df_aligned = align_features(df, expected_features)
    
#     # Scale data
#     if scaler is None:
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(df_aligned)
#     else:
#         X_scaled = scaler.transform(df_aligned)
    
#     # Reshape for CNN
#     X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
#     return X_scaled, scaler
