#!/usr/bin/env python
"""
Kiểm tra môi trường để đảm bảo tất cả các thư mục và file cần thiết tồn tại
trước khi chạy run_all_folds.py
"""

import os
import sys
import glob
import importlib.util
import tensorflow as tf
import argparse

def check_file_exists(file_path, message=None):
    """Kiểm tra file tồn tại"""
    if not os.path.exists(file_path):
        if message:
            print(f"ERROR: {message}")
        else:
            print(f"ERROR: File not found: {file_path}")
        return False
    return True

def check_directory_exists(dir_path, create=True):
    """Kiểm tra thư mục tồn tại, tạo nếu cần"""
    if not os.path.exists(dir_path):
        if create:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
                return True
            except Exception as e:
                print(f"ERROR: Failed to create directory {dir_path}: {e}")
                return False
        else:
            print(f"ERROR: Directory not found: {dir_path}")
            return False
    return True

def check_config(config_file):
    """Kiểm tra và tải file cấu hình"""
    if not check_file_exists(config_file, f"Config file not found: {config_file}"):
        return None
        
    try:
        spec = importlib.util.spec_from_file_location("*", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.train
        print(f"Config loaded successfully with {config.get('n_classes', 'unknown')} classes")
        
        # Kiểm tra các tham số bắt buộc
        required_params = ["dataset", "data_dir", "input_size", "n_classes", "n_folds"]
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            print(f"ERROR: Missing required parameters in config: {missing_params}")
            return None
            
        return config
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}")
        return None

def check_dataset_files(config):
    """Kiểm tra các file dataset"""
    if not config:
        return False
        
    # Kiểm tra danh sách subject
    dataset_file = f"{config['dataset']}.txt"
    if not check_file_exists(dataset_file, f"Dataset file not found: {dataset_file}"):
        return False
        
    # Kiểm tra thư mục data
    if not check_directory_exists(config["data_dir"], create=False):
        return False
        
    # Kiểm tra các file .pkl
    pkl_files = glob.glob(os.path.join(config["data_dir"], "*.pkl"))
    if not pkl_files:
        print(f"ERROR: No .pkl files found in {config['data_dir']}")
        return False
        
    print(f"Found {len(pkl_files)} .pkl files in {config['data_dir']}")
    return True

def check_gpu_availability():
    """Kiểm tra GPU sẵn có"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU available: {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            return True
        else:
            print("WARNING: No GPU detected, training will be slow")
            return False
    except Exception as e:
        print(f"ERROR checking GPU: {e}")
        return False

def check_tf_version():
    """Kiểm tra phiên bản TensorFlow"""
    try:
        print(f"TensorFlow version: {tf.__version__}")
        return True
    except Exception as e:
        print(f"ERROR checking TensorFlow: {e}")
        return False

def check_modules():
    """Kiểm tra các module Python cần thiết"""
    required_modules = [
        "numpy", "pandas", "sklearn", "matplotlib", "tensorflow", 
        "imblearn", "sleepstage", "logger"
    ]
    missing_modules = []
    
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"Module {module_name} is available")
        except ImportError:
            missing_modules.append(module_name)
            print(f"ERROR: Module {module_name} is not installed")
    
    if missing_modules:
        print(f"Missing required modules: {missing_modules}")
        print("Please install missing modules with: pip install " + " ".join(missing_modules))
        return False
    return True

def check_output_dir(output_dir):
    """Kiểm tra và tạo thư mục đầu ra"""
    if check_directory_exists(output_dir):
        print(f"Output directory is ready: {output_dir}")
        return True
    return False

def main(args):
    """Chạy kiểm tra"""
    print("===== Environment Check =====")
    
    # Kiểm tra GPU và TensorFlow
    check_gpu_availability()
    check_tf_version()
    
    # Kiểm tra file cấu hình
    config = check_config(args.config)
    
    # Kiểm tra các module cần thiết
    modules_ok = check_modules()
    
    # Kiểm tra dataset
    dataset_ok = check_dataset_files(config)
    
    # Kiểm tra thư mục đầu ra
    output_ok = check_output_dir(args.output_dir)
    
    # Kiểm tra các script cần thiết
    scripts = ["train.py", "model.py", "data.py", "utils.py", "class_balancing.py"]
    scripts_ok = True
    
    for script in scripts:
        if not check_file_exists(script, f"Required script {script} is missing"):
            scripts_ok = False
    
    # Kiểm tra các thư viện bổ sung nếu được chỉ định
    if args.check_libraries:
        try:
            import tensorflow as tf
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.metrics import classification_report
            from imblearn.over_sampling import SMOTE
            
            print("All required libraries are available.")
        except ImportError as e:
            print(f"ERROR: Library not available: {e}")
            return False
    
    # Tổng kết kết quả
    all_ok = config is not None and modules_ok and dataset_ok and output_ok and scripts_ok
    
    print("\n===== Environment Check Summary =====")
    print(f"Config: {'OK' if config else 'FAILED'}")
    print(f"Modules: {'OK' if modules_ok else 'FAILED'}")
    print(f"Dataset: {'OK' if dataset_ok else 'FAILED'}")
    print(f"Output directory: {'OK' if output_ok else 'FAILED'}")
    print(f"Required scripts: {'OK' if scripts_ok else 'FAILED'}")
    print(f"Overall status: {'OK - Ready to run' if all_ok else 'FAILED - Please fix issues'}")
    
    return all_ok

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check environment for NeuroSight training")
    parser.add_argument("--config", type=str, default="config/sleepedfx.py", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--check_libraries", action="store_true", help="Check additional Python libraries")
    args = parser.parse_args()
    
    success = main(args)
    if not success:
        print("Environment check failed. Please fix the issues before running run_all_folds.py")
        sys.exit(1)
    else:
        print("Environment check passed successfully. You can now run run_all_folds.py")


# python check_environment.py --config config/sleepedfx.py --output_dir ./results --check_libraries