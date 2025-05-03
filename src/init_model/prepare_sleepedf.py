import argparse
import glob
import ntpath
import shutil
import os
import numpy as np 
import pandas as pd 
import mne
import pickle

from sleepstage import stage_dictionary
from logger import get_logger

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
    "Obstructive Apnea" : 5,
    "Obstructive Hypopnea" : 6,
    "Mixed Apnea": 7,
    "Central Apnea": 8,
    "Oxygen Desaturation" : 9,
    "EEG arousal" : 10,
    "Hypopnea" : 11
}

# Nếu dữ liệu ngắn hơn : 
# xác định số mẫu cần bổ sung -> lấy đoạn cuối của signal để lặp lại cho đến cuối
# tính trung bình của toàn epoch + tạo 1 mảng với kích thước cần padding
# kết hợp 2 mảng = tb cộng để làm mềm padding
def fix_epoch_length(epoch_data, desired_length):
    current_length = epoch_data.shape[1]
    if current_length > desired_length:
        return epoch_data[:, :desired_length]
    elif current_length < desired_length:
        pad_width = desired_length - current_length
        last_seg_length = min(current_length, pad_width)
        last_segment = epoch_data[:, -last_seg_length:]
        repeated = np.tile(last_segment, int(np.ceil(pad_width / last_seg_length)))
        repeated = repeated[:, :pad_width]
        epoch_mean = np.mean(epoch_data)
        mean_array = np.full((epoch_data.shape[0], pad_width), epoch_mean, dtype=epoch_data.dtype)
        pad_values = 0.5 * (repeated + mean_array)
        return np.concatenate((epoch_data, pad_values), axis=1)
    return epoch_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/raw", help="Đường dẫn tới dữ liệu EDF.")
    parser.add_argument("--output_dir", type=str, default="./data/processed", help="Đường dẫn lưu file edf đã xử lý")
    parser.add_argument("--select_ch", type=str, default="EEG C3-M2", help="Tên kênh lựa chọn")
    parser.add_argument("--log_file", type=str, default="info_channel_extract.log", help="Log file")
    parser.add_argument("--epoch_duration", type=int, default=30, help="Thời lượng mỗi epoch (giây)")

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    log_file = os.path.join(args.output_dir, args.log_file)

    logger = get_logger(log_file, level="info")

    select_ch = args.select_ch
    epoch_duration = args.epoch_duration

    edf_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*.tsv"))

    if not edf_fnames or not ann_fnames:
        logger.error("No EDF or TSV files found in the data directory.")
        return

    edf_fnames.sort()
    ann_fnames.sort()

    for edf_fname, ann_fname in zip(edf_fnames, ann_fnames):
        logger.info(f"Processing Signal File: {edf_fname}")
        logger.info(f"Annotation File: {ann_fname}")

        try:
            edf_f = mne.io.read_raw_edf(edf_fname, preload=True, verbose=False)
        except Exception as e:
            logger.error(f"Failed to load EDF file {edf_fname}: {e}")
            continue

        try:
            ann_f = pd.read_csv(ann_fname, sep="\t", engine="python")
        except Exception as e:
            logger.error(f"Failed to load TSV file {ann_fname}: {e}")
            continue

        valid_ann_f = ann_f[ann_f['description'].isin(ann2label.keys())]
        invalid_ann_f = ann_f[~ann_f['description'].isin(ann2label.keys())]
        for idx, row in invalid_ann_f.iterrows():
            logger.warning(f"Skipping invalid annotation at index {idx}: {row['description']}")

        if valid_ann_f.empty:
            logger.error(f"No valid annotations found in {ann_fname}. Skipping file.")
            continue 

        start_datetime = edf_f.info['meas_date']
        logger.info(f"Start datetime: {start_datetime}")

        sfreq = edf_f.info['sfreq']
        file_duration = edf_f.n_times / sfreq
        logger.info(f"File duration: {file_duration} seconds")

        n_samples_per_epoch = int(epoch_duration * sfreq)
        logger.info(f"Samples per epoch: {n_samples_per_epoch}")

        n_epochs = int(np.ceil(file_duration / epoch_duration))
        epochs = []
        epochs_labels = []
        epoch_onsets = []  
        epoch_durations = []
        n_labels = len(ann2label)

        if select_ch not in edf_f.ch_names:
            logger.error(f"Channel {select_ch} not found in {edf_fname}. Available channels : {edf_fnames}")
            continue

        for i in range(n_epochs):
            epoch_start = i * epoch_duration
            epoch_end = (i + 1) * epoch_duration

            overlapping_ann = valid_ann_f[(valid_ann_f['onset'] < epoch_end) & 
                                         (valid_ann_f['onset'] + valid_ann_f['duration'] > epoch_start)]

            if overlapping_ann.empty:
                logger.warning(f"Epoch {i}: Start={epoch_start}s, End={epoch_end}s has no valid annotations. Skipping.")
                continue

            label_vector = np.zeros(n_labels, dtype=int)
            for _, row in overlapping_ann.iterrows():
                label = row['description']
                if label in ann2label:
                    label_idx = ann2label[label]
                    label_vector[label_idx] = 1

            start_sample = int(epoch_start * sfreq)
            end_sample = int(min(epoch_end * sfreq, edf_f.n_times))
            try:
                ch_data = edf_f.copy().pick([select_ch]).get_data(start=start_sample, stop=end_sample)
                ch_data_fixed = fix_epoch_length(ch_data, n_samples_per_epoch)
            except Exception as e:
                logger.warning(f"Error processing epoch {i} in {edf_fname}: {e}")
                continue

            epochs.append(ch_data_fixed)
            epochs_labels.append(label_vector)
            epoch_onsets.append(epoch_start)
            epoch_durations.append(epoch_end - epoch_start)
            logger.info(f"Epoch {i}: Start={epoch_start}s, End={epoch_end}s, Labels={label_vector}")

        if not epochs:
            logger.warning(f"No valid epochs extracted from {edf_fname}")
            continue

        study_pat_id = os.path.basename(edf_fname).split("_")[0]

        data_dict = {
            'x': np.array(epochs, dtype=np.float32).squeeze()[:, :, np.newaxis],
            'y': np.array(epochs_labels, dtype=np.int32),  
            'fs': sfreq,
            'ch_label': np.array([select_ch], dtype='<U10'),
            'start_datetime': start_datetime,
            'file_duration': file_duration,
            'epoch_duration': epoch_duration,
            'n_epochs': len(epochs),
            'onsets': np.array(epoch_onsets, dtype=np.float32),  
            'durations': np.array(epoch_durations, dtype=np.float32),
        }

        output_path = os.path.join(args.output_dir, f"data_{study_pat_id}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(data_dict, f)

        logger.info(f"Saved {len(epochs)} epochs from {edf_fname} with STUDY_PAT_ID {study_pat_id}")

if __name__ == "__main__":
    main()