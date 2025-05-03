import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from utils import manual_oversample
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from data import load_data, get_subject_files
from model import NeuroSight
from prepare_sleepedf import ann2label
from sleepstage import class_dictionary
from class_balancing import calculate_class_weights, ThresholdOptimizationCallback, apply_smote_multilabel, augment_signal
from utils import (print_n_samples_each_class,
                   compute_portion_each_class,
                   load_seq_ids)
from logger import get_logger

import logging
tf.get_logger().setLevel(logging.ERROR)

# class MultiClassPrecision(tf.keras.metrics.Metric):
#     def __init__(self, num_classes, name='precision', **kwargs):
#         super(MultiClassPrecision, self).__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.true_positives = self.add_weight(
#             name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
#         )
#         self.false_positives = self.add_weight(
#             name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
#         )
#         self.support = self.add_weight(
#             name='support', shape=(num_classes,), initializer='zeros', dtype=tf.float32
#         )

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)  # Chuyển logits thành nhãn
#         y_true = tf.cast(y_true, tf.int32)

#         # Tính TP, FP, và support cho từng lớp
#         for i in range(self.num_classes):
#             true_mask = tf.equal(y_true, i)
#             pred_mask = tf.equal(y_pred, i)

#             tp = tf.reduce_sum(tf.cast(true_mask & pred_mask, tf.float32))
#             fp = tf.reduce_sum(tf.cast((~true_mask) & pred_mask, tf.float32))
#             support = tf.reduce_sum(tf.cast(true_mask, tf.float32))

#             # Cập nhật trạng thái
#             self.true_positives[i].assign(
#                 self.true_positives[i] + tp
#             )
#             self.false_positives[i].assign(
#                 self.false_positives[i] + fp
#             )
#             self.support[i].assign(
#                 self.support[i] + support
#             )

#     def result(self):
#         precisions = tf.where(
#             self.true_positives + self.false_positives > 0,
#             self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon()),
#             tf.zeros_like(self.true_positives)
#         )
#         total_support = tf.reduce_sum(self.support)
#         weighted_precision = tf.reduce_sum(precisions * self.support) / (total_support + tf.keras.backend.epsilon())
#         return weighted_precision

#     def reset_state(self):
#         self.true_positives.assign(tf.zeros(self.num_classes, dtype=tf.float32))
#         self.false_positives.assign(tf.zeros(self.num_classes, dtype=tf.float32))
#         self.support.assign(tf.zeros(self.num_classes, dtype=tf.float32))

# class MultiClassRecall(tf.keras.metrics.Metric):
#     def __init__(self, num_classes, name='recall', **kwargs):
#         super(MultiClassRecall, self).__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.true_positives = self.add_weight(
#             name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32
#         )
#         self.false_negatives = self.add_weight(
#             name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32
#         )
#         self.support = self.add_weight(
#             name='support', shape=(num_classes,), initializer='zeros', dtype=tf.float32
#         )

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)  # Chuyển logits thành nhãn
#         y_true = tf.cast(y_true, tf.int32)

#         # Tính TP, FN, và support cho từng lớp
#         for i in range(self.num_classes):
#             true_mask = tf.equal(y_true, i)
#             pred_mask = tf.equal(y_pred, i)

#             tp = tf.reduce_sum(tf.cast(true_mask & pred_mask, tf.float32))
#             fn = tf.reduce_sum(tf.cast(true_mask & (~pred_mask), tf.float32))
#             support = tf.reduce_sum(tf.cast(true_mask, tf.float32))

#             # Cập nhật trạng thái
#             self.true_positives[i].assign(
#                 self.true_positives[i] + tp
#             )
#             self.false_negatives[i].assign(
#                 self.false_negatives[i] + fn
#             )
#             self.support[i].assign(
#                 self.support[i] + support
#             )

#     def result(self):
#         recalls = tf.where(
#             self.true_positives + self.false_negatives > 0,
#             self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon()),
#             tf.zeros_like(self.true_positives)
#         )
#         total_support = tf.reduce_sum(self.support)
#         weighted_recall = tf.reduce_sum(recalls * self.support) / (total_support + tf.keras.backend.epsilon())
#         return weighted_recall

#     def reset_state(self):
#         self.true_positives.assign(tf.zeros(self.num_classes, dtype=tf.float32))
#         self.false_negatives.assign(tf.zeros(self.num_classes, dtype=tf.float32))
#         self.support.assign(tf.zeros(self.num_classes, dtype=tf.float32))

def create_tf_dataset(x_list, y_list, batch_size, seq_length=None, augment=False, shuffle=False):
    x = np.vstack(x_list)
    y = np.vstack(y_list)

    # Apply data augmentation for minority classes
    if augment:
        augmented_x = []
        augmented_y = []
        class_counts = np.sum(y, axis=0)
        max_count = np.max(class_counts)
        
        for class_idx in range(y.shape[1]):
            # Find samples with this class
            indices = np.where(y[:, class_idx] == 1)[0]
            class_count = len(indices)
            
            # Skip well-represented classes
            if class_count >= 0.5 * max_count:
                continue
                
            # For rare classes (less than 100 samples), augment more
            augment_factor = min(5, max(1, int(max_count / (class_count + 1) * 0.5)))
            if class_count > 0:
                for _ in range(augment_factor):
                    for idx in indices:
                        noise_level = np.random.uniform(0.02, 0.08)
                        time_shift = np.random.randint(10, 30)
                        aug_x = augment_signal(
                            x[idx], 
                            noise_level=noise_level, 
                            time_shift_max=time_shift
                        )
                        augmented_x.append(aug_x)
                        augmented_y.append(y[idx])
        
        # Add augmented samples if any
        if augmented_x:
            augmented_x = np.array(augmented_x)
            augmented_y = np.array(augmented_y)
            x = np.vstack([x, augmented_x])
            y = np.vstack([y, augmented_y])

    if augment and x.shape[0] < 10000 and np.all(class_counts > 5):
        try:
            # Reshape data if needed for SMOTE
            orig_shape = x.shape
            reshaped_x = x.reshape(x.shape[0], -1)
            
            # Apply SMOTE for further balancing
            x_resampled, y_resampled = apply_smote_multilabel(reshaped_x, y)
            
            # Reshape back to original shape
            x = x_resampled.reshape(-1, orig_shape[1], orig_shape[2])
            y = y_resampled
            print(f"Applied SMOTE, new dataset shape: {x.shape}")
        except Exception as e:
            print(f"SMOTE application failed: {e}, continuing with augmented data")
    
    # Create sample weights (higher weights for minority classes)
    sample_weights = np.ones(y.shape[0])
    class_counts = np.sum(y, axis=0)
    for i in range(y.shape[1]):
        if class_counts[i] > 0:
            # Log scale to prevent extreme weights
            weight = np.log(np.sum(class_counts) / class_counts[i] + 1)
            weight = min(weight, 10.0)  # Cap the weight to avoid extreme values
            indices = np.where(y[:, i] == 1)[0]
            sample_weights[indices] = np.maximum(sample_weights[indices], weight)
    
    dataset = tf.data.Dataset.from_tensor_slices((x, y, sample_weights))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(10000, x.shape[0]))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

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
            "frequency": (len(indices) / (total_duration / 3600)) if total_duration > 0 else 0.0,
            "time_pattern": np.diff(onsets[indices]).tolist() if len(indices) > 1 else []
        }
    return stats

def find_optimal_thresholds(model, val_dataset, n_classes):
    """Find optimal decision thresholds for each class"""
    all_preds = []
    all_labels = []
    
    # Gather predictions and true labels
    for x_batch, y_batch, _ in val_dataset:
        logits = model(x_batch, training=False)
        probs = tf.sigmoid(logits).numpy()
        all_preds.append(probs)
        all_labels.append(y_batch.numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Find best threshold for each class
    optimal_thresholds = np.zeros(n_classes)
    
    for c in range(n_classes):
        best_f1 = 0
        best_threshold = 0.5
        
        # Skip if no positive examples
        if np.sum(all_labels[:, c]) == 0:
            optimal_thresholds[c] = 0.5
            continue
            
        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds_c = (all_preds[:, c] >= threshold).astype(int)
            f1 = f1_score(all_labels[:, c], preds_c, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        optimal_thresholds[c] = best_threshold
        print(f"Class {c}: optimal threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")
        
    return optimal_thresholds

def train(
    config_file, # đường dẫn tới file cấu hình sleepedf.py
    fold_idx, # chỉ số của fold trong k-fold cross-validation
    output_dir, # thư mục lưu kết quả đầu ra
    log_file, # lưu lại log
    restart=False, 
    random_seed=42, 
):
    spec = importlib.util.spec_from_file_location("*", config_file) # tạo ra đặc tả cho module
    config = importlib.util.module_from_spec(spec) # tạo module mới từ spec vừa tạo
    spec.loader.exec_module(config) # thực thi module
    config = config.train 

    # Create output directory for the specified fold_idx
    output_dir = os.path.join(output_dir, str(fold_idx)) # lưu kết quả riêng cho mỗi fold
    if restart and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = get_logger(log_file, level="info")

    # Load danh sách subject
    subject_files = glob.glob(os.path.join(config["data_dir"], "*.pkl"))
    seq_sids = load_seq_ids(f"{config['dataset']}.txt")
    logger.info(f"Loaded {len(seq_sids)} subjects from {config['dataset']}")

    # Chia train/valid/test
    fold_pids = np.array_split(seq_sids, config["n_folds"])
    test_sids = fold_pids[fold_idx]
    train_sids = np.setdiff1d(seq_sids, test_sids)

    # Tách validation từ train
    np.random.seed(random_seed)
    valid_size = max(1, int(len(train_sids) * 0.1)) 
    valid_sids = np.random.choice(train_sids, size=valid_size, replace=False)
    train_sids = np.setdiff1d(train_sids, valid_sids)

    def load_subset(sids):
        files = []
        for sid in sids:
            sid_files = get_subject_files(config["dataset"], subject_files, sid)
            logger.info(f"Subject {sid} -> Files: {sid_files}")
            files.extend(sid_files)
        if not files:
            raise ValueError(f"No files found for subject IDs: {sids}")
        return load_data(files)

    try:
        train_x, train_y, train_durations, train_onsets, _ = load_subset(train_sids)
        valid_x, valid_y, valid_durations, valid_onsets, _ = load_subset(valid_sids)
        test_x, test_y, test_durations, test_onsets, _ = load_subset(test_sids)
    except ValueError as e:
        logger.error(f"Data loading failed: {e}")
        return

    # Log thông tin dataset
    train_labels = np.vstack(train_y)
    logger.info(f"Train set: {train_labels.shape[0]} samples")
    logger.info(f"Valid set: {len(np.vstack(valid_y))} samples")
    logger.info(f"Test set: {len(np.vstack(test_y))} samples")

    logger.info("--------------------------------------------")
    logger.info("Train set class distribution:")
    print_n_samples_each_class(train_labels)
    logger.info("--------------------------------------------")

    test_labels = np.vstack(test_y)
    logger.info("Test set class distribution:")
    print_n_samples_each_class(test_labels)

    class_weights = calculate_class_weights(train_labels)
    logger.info(f"Class weights: {class_weights}")

    train_dataset = create_tf_dataset(train_x, train_y, config["batch_size"], 
                                      augment=True, shuffle=True)
    valid_dataset = create_tf_dataset(valid_x, valid_y, config["batch_size"])
    test_dataset = create_tf_dataset(test_x, test_y, config["batch_size"])

    model = NeuroSight(
        config=config, 
        output_dir=output_dir, 
        use_rnn=config.get("use_rnn", False),
        use_attention=config.get("use_attention", False),
        use_multi_dropout=config.get("use_multi_dropout", True),  # Enable multi-sample dropout by default
        mixup_alpha=config.get("mixup_alpha", 0.2)  # Enable mixup augmentation
    )

    model.set_class_weights(class_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.get("learning_rate", 0.001),
            beta_1=config.get("adam_beta_1", 0.9),
            beta_2=config.get("adam_beta_2", 0.999),
            epsilon=config.get("adam_epsilon", 1e-7),
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    checkpoint_path = os.path.join(output_dir, "best_model.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_binary_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_binary_accuracy",
            patience=config.get("patience", 10),
            restore_best_weights=True,
            mode="max"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ThresholdOptimizationCallback(valid_dataset, patience=8)
    ]

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=config["n_epochs"],
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Finding optimal classification thresholds...")
    optimal_thresholds = find_optimal_thresholds(model, valid_dataset, config["n_classes"])
    model.set_thresholds(optimal_thresholds)
    logger.info(f"Optimal thresholds: {optimal_thresholds}")

    test_results = model.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")

    test_predictions = []
    test_true_labels = []
    for x_batch, y_batch, _ in test_dataset:
        logits = model(x_batch, training=False)
        probs = tf.sigmoid(logits)
        # Use optimized thresholds
        preds = tf.cast(tf.greater_equal(probs, model.thresholds), tf.float32).numpy()
        test_predictions.append(preds)
        test_true_labels.append(y_batch.numpy())

    test_predictions = np.vstack(test_predictions)
    test_true_labels = np.vstack(test_true_labels)
    test_durations_flat = np.concatenate(test_durations)
    test_onsets_flat = np.concatenate(test_onsets)

    # Generate detailed classification report
    target_names = [class_dictionary.get(i, f"Unknown Label {i}") for i in range(test_predictions.shape[1])]
    report = classification_report(test_true_labels, test_predictions, target_names=target_names, zero_division=0)

    # Log classification report
    logger.info("Classification Report:")
    logger.info("\n" + report)

    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_true_labels, test_predictions, average=None, zero_division=0
    )

    # Log per-class metrics
    for i in range(len(target_names)):
        logger.info(f"Class {target_names[i]}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")
    
    stats = compute_statistics(test_predictions, test_durations_flat, test_onsets_flat)
    label2ann = {v: k for k, v in ann2label.items()}
    
    for label, stat in stats.items():
        event_name = label2ann.get(label, f"Unknown Label {label}")
        logger.info(f"Event: {event_name}")
        logger.info(f"  Count: {stat['count']}")
        logger.info(f"  Percentage: {stat['percentage']:.2f}%")
        logger.info(f"  Mean Duration: {stat['mean_duration']:.2f}s")
        logger.info(f"  Max Duration: {stat['max_duration']:.2f}s")
        logger.info(f"  Frequency: {stat['frequency']:.2f}/hour")
    
    # Save optimal thresholds
    np.save(os.path.join(output_dir, "optimal_thresholds.npy"), optimal_thresholds)
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--log_file", type=str, default="training.log", help="Log file path")
    parser.add_argument("--restart", action="store_true", help="Clear existing output directory")
    args = parser.parse_args()

    train(
        config_file=args.config,
        fold_idx=args.fold,
        output_dir=args.output_dir,
        log_file=args.log_file,
        restart=args.restart
    )


# train 1 fold : python train.py --config config/sleepedfx.py --fold 0 --output_dir ./results --log_file training.log