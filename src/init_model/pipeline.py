import mne
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import importlib
import os
from model import NeuroSight 

select_ch = "EEG C3-M2"  
epoch_duration = 30  
sfreq = 256.0 

def load_edf_file(edf_path):
    try:
        edf_f = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        if select_ch not in edf_f.ch_names:
            raise ValueError(f"Kênh {select_ch} không có trong {edf_path}")
        data, _ = edf_f[select_ch]
        return data, edf_f.info['sfreq']
    except Exception as e:
        print(f"Lỗi khi đọc file EDF: {e}")
        return None, None

def extract_epochs(data, sfreq, epoch_duration):
    n_samples = data.shape[1]
    n_samples_per_epoch = int(epoch_duration * sfreq)
    n_epochs = n_samples // n_samples_per_epoch
    epochs = []
    for i in range(n_epochs):
        start = i * n_samples_per_epoch
        end = start + n_samples_per_epoch
        epoch_data = data[0, start:end]  
        epochs.append(epoch_data)
    return np.array(epochs)

def compute_statistics(predictions, durations, onsets):
    stats = {}
    total_duration = np.sum(durations)
    n_samples = predictions.shape[0]
    for label in range(predictions.shape[1]):
        indices = np.where(predictions[:, label] == 1)[0]
        stats[label] = {
            "count": len(indices),
            "percentage": (len(indices) / n_samples * 100) if n_samples > 0 else 0.0,
            "mean_duration": np.mean(durations[indices]) if len(indices) > 0 else 0.0,
            "max_duration": np.max(durations[indices]) if len(indices) > 0 else 0.0,
            "frequency": (len(indices) / (total_duration / 3600)) if total_duration > 0 else 0.0
        }
    return stats

def load_config(config_file):
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.train

def predict_disease_from_edf(edf_path, model_path, config_file, rf_model_path, thresholds_path):
    data, file_sfreq = load_edf_file(edf_path)
    if data is None:
        return None
    sampling_freq = sfreq
    if file_sfreq != sfreq:
        print(f"Tần số lấy mẫu của file ({file_sfreq} Hz) khác với tần số mặc định ({sfreq} Hz).")
        sampling_freq = file_sfreq
    epochs = extract_epochs(data, sampling_freq, epoch_duration)
    print(f"Đã trích xuất {epochs.shape[0]} epoch từ file EDF.")

    config = load_config(config_file)

    model = NeuroSight(config=config)

    input_size = int(epoch_duration * sampling_freq)
    model.build(input_shape=(None, input_size, 1))

    if not os.path.exists(model_path):
        print(f"Không tìm thấy file trọng số tại {model_path}")
        return None

    if os.path.exists(thresholds_path):
        thresholds = np.load(thresholds_path)
    else:
        print(f"Không tìm thấy file ngưỡng tại {thresholds_path}")
        thresholds = np.ones(config["n_classes"]) * 0.5  

    predictions = []
    for epoch in epochs:
        if len(epoch) != input_size:
            if len(epoch) < input_size:
                epoch = np.pad(epoch, (0, input_size - len(epoch)), 'constant')
            else:
                epoch = epoch[:input_size]
                
        epoch = epoch[np.newaxis, :, np.newaxis] 
        
        try:
            logits = model(epoch, training=False)
            probs = tf.sigmoid(logits).numpy()
            preds = (probs >= thresholds).astype(int)
            predictions.append(preds)
        except Exception as e:
            print(f"Lỗi khi dự đoán epoch: {e}")
            predictions.append(np.zeros((1, config["n_classes"])))
            
    predictions = np.vstack(predictions)

    durations = np.full((predictions.shape[0],), epoch_duration)
    onsets = np.arange(0, predictions.shape[0] * epoch_duration, epoch_duration)
    stats = compute_statistics(predictions, durations, onsets)

    labels = list(range(config["n_classes"]))  
    data = {}
    for label in labels:
        col = f"label_{label}"
        data[f'{col}_count'] = [stats[label]['count']]
        data[f'{col}_mean_duration'] = [stats[label]['mean_duration']]
        data[f'{col}_max_duration'] = [stats[label]['max_duration']]
        data[f'{col}_frequency'] = [stats[label]['frequency']]
        data[f'{col}_percentage'] = [stats[label]['percentage']]

    temp_df = pd.DataFrame(data)

    expected_columns = 60
    current_columns = temp_df.shape[1]
    if current_columns < expected_columns:
        for i in range(current_columns, expected_columns):
            temp_df[f'feature_{i}'] = 0
    elif current_columns > expected_columns:
        temp_df = temp_df.iloc[:, :expected_columns]
    print(f"Đã tạo DataFrame với {temp_df.shape[1]} cột.")

    # Lưu DataFrame thành file CSV (tùy chọn)
    temp_df.to_csv("statistical_features.csv", index=False)
    print("Đã lưu đặc trưng thống kê vào statistical_features.csv")

    try:
        with open(rf_model_path, 'rb') as file:
            rf_model = pickle.load(file)

        input_data = temp_df.values
        prediction = rf_model.predict(input_data)

        prediction_index = int(prediction[0])
        disease_labels = ["Không bệnh", "Động kinh", "Rối loạn giấc ngủ", "Nhồi máu não"]

        if 0 <= prediction_index < len(disease_labels):
            predicted_disease = disease_labels[prediction_index]
        else:
            print(f"Chỉ số không hợp lệ: {prediction_index}. Giá trị gốc: {prediction[0]}")
            predicted_disease = "Không xác định"
            
        return predicted_disease
    except Exception as e:
        import traceback
        print(f"Lỗi khi dự đoán bệnh: {e}")
        print(traceback.format_exc())  
        return None

if __name__ == "__main__":
    edf_path = "data/raw/12820_8146.edf" 
    model_path = "results/0/best_model.weights.h5"  
    config_file = "config/sleepedfx.py"  
    rf_model_path = "data/random_forest_model.pkl"  
    thresholds_path = "results/0/optimal_thresholds.npy"  

    predicted_disease = predict_disease_from_edf(
        edf_path=edf_path,
        model_path=model_path,
        config_file=config_file,
        rf_model_path=rf_model_path,
        thresholds_path=thresholds_path
    )

    if predicted_disease:
        print(f"Dự đoán bệnh: {predicted_disease}")
    else:
        print("Không thể dự đoán do lỗi xử lý.")