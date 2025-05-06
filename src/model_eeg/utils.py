import numpy as np
from sleepstage import class_dictionary
import logging
logger = logging.getLogger("default_log")

def manual_oversample(x, y, target_samples=50, fs=100, random_seed=None):
    """Lặp lại mẫu thủ công với augmentation cho lớp hiếm."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    oversampled_x = []
    oversampled_y = []
    
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        n_samples = len(indices)
        
        if n_samples <= 10:  # Lớp cực kỳ hiếm
            current_x = x[indices]
            current_y = y[indices]
            
            oversampled_x.append(current_x)
            oversampled_y.append(current_y)
            
            # Lặp lại mẫu với augmentation
            for _ in range(target_samples // n_samples):
                for i in indices:
                    aug_x = augment_signal(x[i:i+1], fs=fs, random_seed=random_seed)
                    oversampled_x.append(aug_x)
                    oversampled_y.append(y[i])
        else:
            oversampled_x.append(x[indices])
            oversampled_y.append(y[indices])
    
    return np.vstack(oversampled_x), np.hstack(oversampled_y)

def augment_signal(signal, fs, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    augmented = signal.copy()
    n_samples = signal.shape[1]
    
    noise_amplitude = 0.05 * np.std(signal)
    noise = np.random.normal(0, noise_amplitude, signal.shape)
    augmented += noise
    
    max_shift = int(0.05 * n_samples)
    shift = np.random.randint(-max_shift, max_shift)
    augmented = np.roll(augmented, shift, axis=1)
    
    scale = np.random.uniform(0.95, 1.05)
    augmented *= scale
    
    return augmented

def save_seq_ids(fname, ids):
    """Save sequence of IDs into txt file."""
    with open(fname, "w") as f:
        for _id in ids:
            f.write(str(_id) + "\n")


def load_seq_ids(fname):
    """Load sequence of IDs from txt file."""
    ids = []
    try:
        with open(fname, "r") as f:
            for line in f:
                ids.append(line.strip())  # Keep as strings to match file naming
    except FileNotFoundError:
        logger.error(f"Sequence ID file {fname} not found")
        raise
    return np.asarray(ids)


def print_n_samples_each_class(labels):
    """Print the number of samples in each class."""
    # unique_labels = np.unique(labels)
    # for c in unique_labels:
    #     n_samples = len(np.where(labels == c)[0])
    #     label_name = class_dictionary.get(int(c), f"Unknown Label {c}")
    #     logger.info(f"{label_name}: {n_samples}")
    if labels.ndim == 1:
        unique_labels = np.unique(labels)
        for c in unique_labels:
            n_samples = len(np.where(labels == c)[0])
            label_name = class_dictionary.get(int(c), f"Unknown Label {c}")
            logger.info(f"{label_name}: {n_samples}")
    elif labels.ndim == 2:
        n_classes = labels.shape[1]
        for c in range(n_classes):
            n_samples = np.sum(labels[:, c])
            label_name = class_dictionary.get(c, f"Unknown Label {c}")
            logger.info(f"{label_name}: {n_samples} samples")

def compute_portion_each_class(labels):
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    class_weights = {}
    max_samples = max(len(np.where(labels == c)[0]) for c in unique_labels)
    
    for c in unique_labels:
        n_class_samples = len(np.where(labels == c)[0])
        if n_class_samples <= 10:  # Lớp cực kỳ hiếm
            class_weights[int(c)] = 100.0  # Trọng số rất cao
        elif n_class_samples <= 100:  # Lớp thiểu số
            class_weights[int(c)] = 20.0  # Trọng số trung bình
        else:  # Lớp đa số
            class_weights[int(c)] = max(1.0, max_samples / n_class_samples)  # Trọng số tỷ lệ nghịch
    return class_weights


def get_balance_class_oversample(x, y):
    """Balance the number of samples of all classes by (oversampling).

    The process is as follows:
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """

    class_labels = np.unique(y)
    n_max_classes = max(len(np.where(y == c)[0]) for c in class_labels)

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        if n_samples == 0:
            continue
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    
    if not balance_x:
        raise ValueError("No valid samples to balance")
    
    return np.vstack(balance_x), np.hstack(balance_y)


def get_balance_class_sample(x, y):
    """Balance the number of samples of all classes by sampling.

    The process is as follows:
        1. Find the class that has the smallest number of samples
        2. Randomly select samples in each class equal to that smallest number
    """

    class_labels = np.unique(y)
    n_min_classes = min(len(np.where(y == c)[0]) for c in class_labels if len(np.where(y == c)[0]) > 0)

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        sample_idx = np.random.choice(idx, size=n_min_classes, replace=False)
        balance_x.append(x[sample_idx])
        balance_y.append(y[sample_idx])
    
    if not balance_x:
        raise ValueError("No valid samples to balance")
    
    return np.vstack(balance_x), np.hstack(balance_y)