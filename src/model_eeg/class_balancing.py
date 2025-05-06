import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras import layers, backend as K

# 1. Class Weighting for Loss Function
def calculate_class_weights(y):
    """
    Calculate class weights inversely proportional to class frequencies
    """
    # Get class frequencies
    class_counts = np.sum(y, axis=0)
    total_samples = len(y)
    
    # Calculate weights (inverse of frequency with smoothing)
    class_weights = {}
    for i in range(y.shape[1]):
        # Add smoothing to avoid division by zero
        freq = max(class_counts[i], 1) / total_samples
        # Inverse frequency with scaling to avoid extremely large weights
        class_weights[i] = min(np.log(1.0/freq + 1) * 2.0, 10.0)
    
    return class_weights

def weighted_binary_crossentropy(y_true, y_pred, class_weights):
    """
    Apply class weights to binary crossentropy loss
    """
    # Convert class weights to tensor
    weight_vector = tf.constant(
        [class_weights[i] for i in range(len(class_weights))], 
        dtype=tf.float32
    )
    
    # Standard binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    
    # Apply class weights
    class_weights_applied = tf.reduce_sum(tf.cast(y_true, tf.float32) * weight_vector, axis=1) 
    
    # Add weight scaling to ensure loss doesn't become too small
    weight_scaling = tf.maximum(
        tf.reduce_sum(y_true, axis=1), 
        tf.ones_like(tf.reduce_sum(y_true, axis=1))
    )
    
    final_weights = class_weights_applied / tf.cast(weight_scaling, tf.float32)
    weighted_bce = bce * final_weights
    
    return tf.reduce_mean(weighted_bce)

# 2. Focal Loss - Focuses more on hard-to-classify examples
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for addressing class imbalance
    Focuses on hard examples by down-weighting easy ones
    """
    y_pred = tf.nn.sigmoid(y_pred)  # Apply sigmoid to logits
    
    # Calculate binary cross entropy
    bce = K.binary_crossentropy(y_true, y_pred)
    
    # Calculate focal weight
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weight = alpha_factor * K.pow(1 - p_t, gamma)
    
    # Apply the weights
    focal_loss = focal_weight * bce
    
    return K.mean(focal_loss)

# 3. Data Augmentation Functions
def augment_minority_samples(x, y, augment_factor=2):
    """
    Augment samples from minority classes
    """
    augmented_x = []
    augmented_y = []
    
    # Get indices for each class
    for class_idx in range(y.shape[1]):
        # Find samples that have this class activated
        indices = np.where(y[:, class_idx] == 1)[0]
        
        # Skip if no samples or many samples
        if len(indices) == 0 or len(indices) > 50:  # Skip well-represented classes
            continue
            
        # For classes with few examples, augment them
        for _ in range(augment_factor):
            for idx in indices:
                aug_x = augment_signal(x[idx])
                augmented_x.append(aug_x)
                augmented_y.append(y[idx])
    
    if augmented_x:
        return np.array(augmented_x), np.array(augmented_y)
    else:
        return np.array([]), np.array([])

def augment_signal(signal, noise_level=0.05, time_shift_max=20):
    """
    Augment a signal with noise and time shifts
    """
    augmented = signal.copy()
    
    # Add noise
    noise = np.random.normal(0, noise_level * np.std(signal), size=signal.shape)
    augmented += noise

    shift = np.random.randint(-time_shift_max, time_shift_max)
    
    # Random time shift
    if augmented.ndim == 3:  # Shape: (batch, time, channels)
        if shift > 0:
            pad_width = ((0, 0), (shift, 0), (0, 0))
            augmented = np.pad(augmented, pad_width, mode='edge')[:, :augmented.shape[1], :]
        elif shift < 0:
            pad_width = ((0, 0), (0, -shift), (0, 0))
            augmented = np.pad(augmented, pad_width, mode='edge')[:, -shift:, :]
    elif augmented.ndim == 2:  # Shape: (time, channels) or (batch, time)
        if shift > 0:
            pad_width = ((shift, 0), (0, 0))
            augmented = np.pad(augmented, pad_width, mode='edge')[:augmented.shape[0], :]
        elif shift < 0:
            pad_width = ((0, -shift), (0, 0))
            augmented = np.pad(augmented, pad_width, mode='edge')[-shift:, :]
    else:
        raise ValueError(f"Unsupported signal dimension: {augmented.ndim}")
    
    return augmented

# 4. Multi-Sample Dropout for better generalization
class MultiSampleDropout(layers.Layer):
    """
    Apply dropout multiple times and average results
    Good for improving performance on rare classes
    """
    def __init__(self, rate=0.5, samples=5, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.samples = samples
        self.dropout = layers.Dropout(rate)
    
    def call(self, inputs, training=None):
        if training:
            return tf.reduce_mean(
                tf.stack([self.dropout(inputs, training=training) 
                         for _ in range(self.samples)], axis=0),
                axis=0
            )
        return inputs

# 5. SMOTE Implementation for Multi-label 
def apply_smote_multilabel(X, y, sampling_strategy='auto'):
    """
    Apply SMOTE to multilabel data by:
    1. Converting to multiclass problem
    2. Applying SMOTE
    3. Converting back to multilabel
    """
    # Convert multilabel to multiclass for SMOTE
    # Each unique combination of labels becomes a class
    y_multiclass = np.argmax(y, axis=1)
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_multiclass_resampled = smote.fit_resample(X, y_multiclass)
    
    # Convert back to multilabel
    y_resampled = np.zeros((len(y_multiclass_resampled), y.shape[1]))
    unique_combinations = np.unique(y, axis=0)
    for i, idx in enumerate(y_multiclass_resampled):
        y_resampled[i] = unique_combinations[idx]
    
    return X_resampled, y_resampled

# 6. Modified ThresholdCallback - Dynamically adjust decision thresholds
class ThresholdOptimizationCallback(tf.keras.callbacks.Callback):
    """
    Dynamically adjust classification thresholds per class
    based on validation performance
    """
    def __init__(self, validation_data, patience=5):
        super().__init__()
        self.validation_data = validation_data
        self.best_thresholds = None
        self.best_f1 = 0
        self.patience = patience
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on validation data
        # x_val, y_val, _ = self.validation_data
        x_val = []
        y_val = []
        for x_batch, y_batch, _ in self.validation_data:
            x_val.append(x_batch.numpy())
            y_val.append(y_batch.numpy())
        
        # Concatenate batches
        x_val = np.concatenate(x_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        y_pred = self.model.predict(x_val)
        y_pred = tf.sigmoid(y_pred).numpy()
        
        # Try different thresholds
        best_f1_sum = 0
        best_thresholds = np.ones(y_val.shape[1]) * 0.5
        
        for c in range(y_val.shape[1]):
            # Skip classes with no positive examples
            if np.sum(y_val[:, c]) == 0:
                continue
                
            best_f1 = 0
            best_threshold = 0.5
            
            # Try thresholds from 0.1 to 0.9
            for threshold in np.arange(0.1, 0.95, 0.05):
                y_pred_class = (y_pred[:, c] >= threshold).astype(int)
                
                # Calculate F1 score for this class and threshold
                tp = np.sum((y_val[:, c] == 1) & (y_pred_class == 1))
                fp = np.sum((y_val[:, c] == 0) & (y_pred_class == 1))
                fn = np.sum((y_val[:, c] == 1) & (y_pred_class == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            best_thresholds[c] = best_threshold
            best_f1_sum += best_f1
        
        # Check if we should update the thresholds
        if best_f1_sum > self.best_f1:
            self.best_f1 = best_f1_sum
            self.best_thresholds = best_thresholds
            self.wait = 0
            print(f"\nEpoch {epoch}: Updated thresholds. Best F1 sum: {best_f1_sum:.4f}")
            print(f"New thresholds: {best_thresholds}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEpoch {epoch}: Early stopping threshold optimization.")
                self.model.stop_training = True