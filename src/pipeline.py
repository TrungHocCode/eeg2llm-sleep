import mne
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import importlib
import os
from model_eeg.model import NeuroSight 
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration

select_ch = "EEG C3-M2"  
epoch_duration = 30  
sfreq = 256.0 

def load_edf_file(edf_path):
    try:
        edf_f = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        if select_ch not in edf_f.ch_names:
            raise ValueError(f"Channel {select_ch} not in {edf_path}")
        data, _ = edf_f[select_ch]
        return data, edf_f.info['sfreq']
    except Exception as e:
        print(f"Error when read file edf: {e}")
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
        print(f"The sampling frequency of the file ({file_sfreq} Hz) is different from the default frequency ({sfreq} Hz).")
        sampling_freq = file_sfreq
    epochs = extract_epochs(data, sampling_freq, epoch_duration)
    print(f"Extracted {epochs.shape[0]} epoch from EDF.")

    config = load_config(config_file)

    model = NeuroSight(config=config)

    input_size = int(epoch_duration * sampling_freq)
    model.build(input_shape=(None, input_size, 1))

    if not os.path.exists(model_path):
        print(f"Can't find model at {model_path}")
        return None

    if os.path.exists(thresholds_path):
        thresholds = np.load(thresholds_path)
    else:
        print(f"Can't find thresholds at {thresholds_path}")
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
            print(f"Errors: {e}")
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
    print(f"Created data frame with {temp_df.shape[1]} columns.")

    # Lưu DataFrame thành file CSV (tùy chọn)
    temp_df.to_csv("statistical_features.csv", index=False)
    print("Saved features into statistical_features.csv")

    try:
        with open(rf_model_path, 'rb') as file:
            rf_model = pickle.load(file)

        input_data = temp_df.values
        prediction = rf_model.predict(input_data)

        prediction_index = int(prediction[0])
        disease_labels = ["Normal", "Seizures", "Sleep disorders", "Stroke"]

        if 0 <= prediction_index < len(disease_labels):
            predicted_disease = disease_labels[prediction_index]
        else:
            print(f"Invalid index: {prediction_index}. Index: {prediction[0]}")
            predicted_disease = "Can't determined"
            
        return predicted_disease
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())  
        return None

if __name__ == "__main__":
    edf_path = "data/raw/study_sleep/11338_3454.edf"    #Duong dan file edf
    model_path = "model/best_model.weights.h5"        #Duong dan toi model egg
    config_file = "src/model_eeg/config/sleepedfx.py"   #Duong dan cau hinh 
    rf_model_path = "model/random_forest_model.pkl"   #Duong dan toi model diagnose
    thresholds_path = "model/optimal_thresholds.npy"  #Duong dan toi file nguong quyet dinh

    predicted_disease = predict_disease_from_edf(
        edf_path=edf_path,
        model_path=model_path,
        config_file=config_file,
        rf_model_path=rf_model_path,
        thresholds_path=thresholds_path
    )
    #Duong dan toi mo hinh da finetuned
    model_path="results/t5_base"

    tokenizer=T5Tokenizer.from_pretrained(model_path)
    model=T5ForConditionalGeneration.from_pretrained(model_path)


    print("=== Patient information ===")
    # Vòng lặp kiểm tra giới tính hợp lệ
    valid_genders = {"male", "female"}

    while True:
        gender = input("Gender (male/female): ").strip().lower()
        if gender in valid_genders:
            gender = gender.capitalize()
            break
        else:
            print("Invalid gender! Pleas enter \"male\" or \"female\".")
    gender=gender.capitalize()
    past_diag = input("Past disease: ").strip()
    
    while True:
        bmi_value = input("BMI value: ").strip()
        age_day = input("Age day (vd: 273): ").strip()
        try: 
            age_day=float(age_day)
            float(bmi_value)
            break
        except ValueError:
            print("Error value! Please enter numeric values for age and BMI")
    #Tao prompt
    prompt=f"Patient information: - Diagnose: {predicted_disease}, - Past_diseases: {past_diag}, - Day-age: {age_day}, - Gender: {gender}, -Meas-type-value: {bmi_value}"
    #Tokenizer
    inputs=tokenizer(prompt,return_tensors="pt",truncation=True,padding=True)
    #Sinh van ban
    output=model.generate(**inputs,max_new_tokens=200)
    output_text=tokenizer.decode(output[0],skip_special_tokens=True)
    #In ket qua
    print(output_text)