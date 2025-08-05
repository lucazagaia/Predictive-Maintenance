import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def parse_filename(filename):
    """
    Extract (device_id, timestamp_begin, timestamp_end) from filename like '3_1651823820_1651825740_fast.csv'
    """
    parts = filename.replace("_fast.csv", "").split("_")
    return parts[0], int(parts[1]), int(parts[2])

def load_fast_csv(file_path):
    """
    Load Current and Vibration columns from one fast CSV file
    """
    df = pd.read_csv(file_path)
    return df[['Current', 'Vibration']].values

def window_data(array, window_size=2048, stride=2048):
    """
    Split full signal into fixed-length windows (no overlap by default)
    """
    windows = []
    for start in range(0, len(array) - window_size + 1, stride):
        windows.append(array[start:start + window_size])
    return np.array(windows)

def prepare_dataset(fast_folder_path, window_size=2048):
    """
    Loop through fast files, slice into windows, assign RUL labels
    Returns X, y arrays
    """
    X, y = [], []

    for fname in os.listdir(fast_folder_path):
        if not fname.endswith("_fast.csv"):
            continue

        device_id, ts_begin, ts_end = parse_filename(fname)

        full_signal = load_fast_csv(os.path.join(fast_folder_path, fname))
        windows = window_data(full_signal, window_size)
        num_windows = len(windows)

        rul_labels = np.arange(num_windows - 1, -1, -1)  # [n-1, ..., 1, 0]

        X.extend(windows)
        y.extend(rul_labels)

    return np.array(X), np.array(y)


def normalize_features(X):
    """
    Normalize Current and Vibration features separately across all samples.
    X shape: (n_samples, window_size, 2)
    Returns: normalized X, fitted scalers
    """
    X_reshaped = X.reshape(-1, 2)  # flatten time dimension

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    X_norm = X_scaled.reshape(X.shape)
    return X_norm, scaler