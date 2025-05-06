import os
import re
import pickle

import numpy as np

def get_subject_files(dataset, files, sid):
    if "sleepedfx" in dataset.lower():
        subject_files = [f for f in files if f"data_{str(sid)}.pkl" in f]  
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return subject_files

def load_data(subject_files):
    signals = []
    labels = []
    durations = []
    onsets = []
    sampling_rate = None
    
    if not subject_files:
        raise ValueError("No files found for the specified subject ID")
        
    for sf in subject_files:
        try:
            with open(sf, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {sf}: {e}")
            continue

        if 'x' not in data or 'y' not in data or 'durations' not in data or 'onsets' not in data:
            print(f"Missing required keys in {sf}. Available keys: {list(data.keys())}")
            continue

        x = data['x']
        if x.ndim == 2:
            x = x[:, :, np.newaxis]
        signals.append(x.astype(np.float32))
        labels.append(data['y'].astype(np.int32))  # y là mảng 2D (n_epochs, n_classes)
        durations.append(data['durations'].astype(np.float32))
        onsets.append(data['onsets'].astype(np.float32))

        if sampling_rate is None:
            sampling_rate = data['fs']
        elif sampling_rate != data['fs']:
            print(f"Warning: Inconsistent sampling rates. Expected {sampling_rate}, got {data['fs']} in {sf}")

    if not signals:
        raise ValueError("No data was loaded from the provided files")
        
    return signals, labels, durations, onsets, sampling_rate