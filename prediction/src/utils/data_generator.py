import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from src.utils.preprocessing import load_fast_csv, window_data, parse_filename

class RULDataGenerator(Sequence):
    def __init__(self, fast_folder_path, label_dict, batch_size=32, window_size=2048, shuffle=True):
        self.files = [f for f in os.listdir(fast_folder_path) if f.endswith("_fast.csv")]
        self.label_dict = label_dict
        self.fast_folder_path = fast_folder_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        files_batch = [self.files[k] for k in self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        X_batch, y_batch = [], []

        for fname in files_batch:
            device_id, ts_begin, ts_end = parse_filename(fname)
            key = (device_id, ts_begin, ts_end)
            label = self.label_dict.get(key)

            if label is None:
                continue

            signal = load_fast_csv(os.path.join(self.fast_folder_path, fname))
            windows = window_data(signal, window_size=self.window_size)

            X_batch.extend(windows)
            y_batch.extend([label] * len(windows))  # Same RUL for all windows

        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)