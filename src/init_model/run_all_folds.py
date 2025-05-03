#!/usr/bin/env python
"""
Run training and evaluation across all folds and compute average results.
This script helps overcome class imbalance by applying the techniques 
across all folds and calculating robust metrics.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import tensorflow as tf
from train import train
from model import NeuroSight
from data import load_data, get_subject_files
from utils import load_seq_ids
from logger import get_logger
import glob
import json
import importlib
from sleepstage import class_dictionary

def evaluate_fold(config, fold_idx, output_dir, log_file):
    """Evaluate a single fold"""
    logger = get_logger(log_file, level="info")
    logger.info(f"Evaluating fold {fold_idx}")

    # Load configuration
    spec = importlib.util.spec_from_file_location("*", config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.train
    
    # Load optimal thresholds
    fold_dir = os.path.join(output_dir, str(fold_idx))
    try:
        thresholds = np.load(os.path.join(fold_dir, "optimal_thresholds.npy"))
    except FileNotFoundError:
        thresholds = np.ones(config["n_classes"]) * 0.5
        logger.warning(f"No thresholds found for fold {fold_idx}, using default 0.5")
    
    # Load test data
    subject_files = glob.glob(os.path.join(config["data_dir"], "*.pkl"))
    seq_sids = load_seq_ids(f"{config['dataset']}.txt")
    
    # Get test subjects for this fold
    fold_pids = np.array_split(seq_sids, config["n_folds"])
    test_sids = fold_pids[fold_idx]
    
    # Load test data
    test_files = []
    for sid in test_sids:
        test_files.extend(get_subject_files(config["dataset"], subject_files, sid))
    
    if not test_files:
        logger.error(f"No test files found for fold {fold_idx}")
        return None
    
    test_x, test_y, test_durations, test_onsets, _ = load_data(test_files)
    
    # Load model weights
    model = NeuroSight(
        config=config, 
        output_dir=fold_dir, 
        use_rnn=config.get("use_rnn", False),
        use_attention=config.get("use_attention", False),
        use_multi_dropout=config.get("use_multi_dropout", True)
    )
    
    # Build model with the right input shape
    model.build(input_shape=(None, config["input_size"], 1))
    
    # Load weights
    checkpoint_path = os.path.join(fold_dir, "best_model.weights.h5")
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
    else:
        logger.error(f"No model weights found at {checkpoint_path}")
        return None
    
    # Set thresholds
    model.set_thresholds(thresholds)
    
    # Get predictions
    all_preds = []
    all_probs = []  # Save probabilities for ROC-AUC
    all_true = []
    
    for x in test_x:
        # Get batch prediction
        logits = model(x, training=False)
        probs = tf.sigmoid(logits).numpy()
        preds = (probs >= thresholds).astype(int)
        
        all_probs.append(probs)
        all_preds.append(preds)
        
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_true = np.vstack(test_y)
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_preds, average=None, zero_division=0
    )
    
    # Calculate ROC-AUC and PR-AUC for each class
    roc_auc = []
    pr_auc = []
    
    for c in range(all_true.shape[1]):
        if np.sum(all_true[:, c]) > 0 and np.sum(all_true[:, c]) < len(all_true[:, c]):
            try:
                roc_auc.append(roc_auc_score(all_true[:, c], all_probs[:, c]))
                pr_auc.append(average_precision_score(all_true[:, c], all_probs[:, c]))
            except Exception as e:
                logger.warning(f"Error calculating AUC for class {c}: {e}")
                roc_auc.append(0.5)
                pr_auc.append(0.0)
        else:
            roc_auc.append(0.5)  # Default for no variation
            pr_auc.append(0.0)
    
    # Combine metrics
    class_metrics = {}
    for i in range(len(support)):
        class_name = class_dictionary.get(i, f"Class_{i}")
        class_metrics[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "roc_auc": float(roc_auc[i]),
            "pr_auc": float(pr_auc[i]),
            "threshold": float(thresholds[i])
        }
    
    # Calculate weighted metrics
    weight = support / np.sum(support)
    weighted_metrics = {
        "precision": float(np.sum(precision * weight)),
        "recall": float(np.sum(recall * weight)),
        "f1": float(np.sum(f1 * weight)),
        "roc_auc": float(np.sum(np.array(roc_auc) * weight)),
        "pr_auc": float(np.sum(np.array(pr_auc) * weight))
    }
    
    # Save metrics
    results = {
        "fold": fold_idx,
        "class_metrics": class_metrics,
        "weighted_metrics": weighted_metrics
    }
    
    # Save to file
    with open(os.path.join(fold_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def run_all_folds(config_file, output_dir, n_folds, log_file, retrain=False):
    """Run training and evaluation on all folds"""
    logger = get_logger(log_file, level="info")
    logger.info(f"Running all {n_folds} folds with config {config_file}")
    
    all_results = []
    
    for fold_idx in range(n_folds):
        fold_log = f"{os.path.splitext(log_file)[0]}_fold{fold_idx}.log"
        fold_dir = os.path.join(output_dir, str(fold_idx))
        
        # Check if results already exist
        eval_file = os.path.join(fold_dir, "evaluation_results.json")
        
        if os.path.exists(eval_file) and not retrain:
            logger.info(f"Fold {fold_idx} already evaluated, loading results")
            with open(eval_file, "r") as f:
                results = json.load(f)
        else:
            # Train model if needed
            if retrain or not os.path.exists(os.path.join(fold_dir, "best_model.weights.h5")):
                logger.info(f"Training fold {fold_idx}")
                train(
                    config_file=config_file,
                    fold_idx=fold_idx,
                    output_dir=output_dir,
                    log_file=fold_log,
                    restart=retrain
                )
            
            # Evaluate model
            results = evaluate_fold(config_file, fold_idx, output_dir, fold_log)
            
        if results:
            all_results.append(results)
    
    # Aggregate results across folds
    if all_results:
        aggregate_results(all_results, output_dir, log_file)
    else:
        logger.error("No results to aggregate")

def aggregate_results(all_results, output_dir, log_file):
    """Aggregate results from all folds and generate visualizations"""
    logger = get_logger(log_file, level="info")
    logger.info("Aggregating results from all folds")
    
    # Extract class names from first fold results
    class_names = list(all_results[0]["class_metrics"].keys())
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    
    # Initialize structures for aggregation
    aggregated = {
        "per_class": {name: {metric: [] for metric in metrics} for name in class_names},
        "weighted": {metric: [] for metric in metrics}
    }
    
    # Collect metrics from all folds
    for result in all_results:
        # Per-class metrics
        for class_name in class_names:
            for metric in metrics:
                aggregated["per_class"][class_name][metric].append(
                    result["class_metrics"][class_name][metric]
                )
        
        # Weighted metrics
        for metric in metrics:
            aggregated["weighted"][metric].append(result["weighted_metrics"][metric])
    
    # Calculate mean and std for each metric
    summary = {
        "per_class": {},
        "weighted": {}
    }
    
    for class_name in class_names:
        summary["per_class"][class_name] = {}
        for metric in metrics:
            values = aggregated["per_class"][class_name][metric]
            summary["per_class"][class_name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
    
    for metric in metrics:
        values = aggregated["weighted"][metric]
        summary["weighted"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }
    
    # Save summary
    with open(os.path.join(output_dir, "summary_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create visualization for F1 scores
    create_f1_visualization(summary, output_dir)
    create_pr_curve_visualization(all_results, output_dir)
    create_class_balance_visualization(all_results, output_dir)
    
    # Log summary
    logger.info("Cross-validation summary:")
    
    for metric in metrics:
        mean = summary["weighted"][metric]["mean"]
        std = summary["weighted"][metric]["std"]
        logger.info(f"Weighted {metric}: {mean:.4f} ± {std:.4f}")
    
    logger.info("Per-class F1 scores:")
    for class_name in class_names:
        mean = summary["per_class"][class_name]["f1"]["mean"]
        std = summary["per_class"][class_name]["f1"]["std"]
        logger.info(f"  {class_name}: {mean:.4f} ± {std:.4f}")

def create_f1_visualization(summary, output_dir):
    """Create bar chart with F1 scores for each class"""
    class_names = list(summary["per_class"].keys())
    f1_means = [summary["per_class"][name]["f1"]["mean"] for name in class_names]
    f1_stds = [summary["per_class"][name]["f1"]["std"] for name in class_names]
    
    plt.figure(figsize=(12, 8))
    
    # Sort classes by F1 score
    sorted_indices = np.argsort(f1_means)
    sorted_names = [class_names[i] for i in sorted_indices]
    sorted_means = [f1_means[i] for i in sorted_indices]
    sorted_stds = [f1_stds[i] for i in sorted_indices]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(class_names)))
    
    plt.barh(range(len(sorted_names)), sorted_means, xerr=sorted_stds, 
             color=colors, alpha=0.7, ecolor='black', capsize=5)
    
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('F1 Score')
    plt.title('F1 Scores by Class (Mean ± Std)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add weighted average line
    weighted_f1_mean = summary["weighted"]["f1"]["mean"]
    plt.axvline(x=weighted_f1_mean, color='r', linestyle='--', 
                label=f'Weighted Avg: {weighted_f1_mean:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_scores_by_class.png'), dpi=300)
    plt.close()

def create_pr_curve_visualization(all_results, output_dir):
    """Create precision-recall scatter plot to visualize class imbalance impact"""
    class_names = list(all_results[0]["class_metrics"].keys())
    
    # Collect precision and recall for each class across folds
    precisions = {name: [] for name in class_names}
    recalls = {name: [] for name in class_names}
    supports = {name: [] for name in class_names}
    
    for result in all_results:
        for class_name in class_names:
            metrics = result["class_metrics"][class_name]
            precisions[class_name].append(metrics["precision"])
            recalls[class_name].append(metrics["recall"])
            supports[class_name].append(metrics["support"])
    
    # Calculate average support for each class
    avg_supports = {name: np.mean(supports[name]) for name in class_names}
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    for class_name in class_names:
        size = np.log(avg_supports[class_name] + 1) * 20  # Log scale for point size
        plt.scatter(
            np.mean(recalls[class_name]), 
            np.mean(precisions[class_name]),
            s=size,
            alpha=0.7,
            label=f"{class_name} (n={int(avg_supports[class_name])})"
        )
    
    # Add labels and reference lines
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall for Each Class')
    plt.grid(linestyle='--', alpha=0.7)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    
    # Add diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Add legend with adjusted font size based on number of classes
    if len(class_names) > 10:
        plt.legend(fontsize='small', loc='lower left', bbox_to_anchor=(1, 0))
    else:
        plt.legend(loc='lower left', bbox_to_anchor=(1, 0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_scatter.png'), dpi=300)
    plt.close()

def create_class_balance_visualization(all_results, output_dir):
    """Create visualization showing class distribution and performance"""
    class_names = list(all_results[0]["class_metrics"].keys())
    
    # Collect data
    supports = []
    f1_scores = []
    thresholds = []
    
    for class_name in class_names:
        # Average across folds
        class_supports = []
        class_f1s = []
        class_thresholds = []
        
        for result in all_results:
            metrics = result["class_metrics"][class_name]
            class_supports.append(metrics["support"])
            class_f1s.append(metrics["f1"])
            class_thresholds.append(metrics["threshold"])
        
        supports.append(np.mean(class_supports))
        f1_scores.append(np.mean(class_f1s))
        thresholds.append(np.mean(class_thresholds))
    
    # Convert to numpy arrays
    supports = np.array(supports)
    f1_scores = np.array(f1_scores)
    thresholds = np.array(thresholds)
    
    # Calculate percentages
    total_samples = np.sum(supports)
    percentages = supports / total_samples * 100
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Create bar chart in first subplot
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(class_names)))
    bars = ax1.bar(class_names, percentages, color=colors, alpha=0.7)
    
    # Add F1 scores as line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(class_names, f1_scores, 'ro-', linewidth=2, markersize=8, alpha=0.7)
    
    # Add labels and titles
    ax1.set_ylabel('Class Distribution (%)')
    ax1_twin.set_ylabel('F1 Score', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Class Distribution and F1 Scores')
    
    # Rotate x-axis labels if needed
    if len(class_names) > 8:
        plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    # Add threshold visualization in second subplot
    ax2.plot(class_names, thresholds, 'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Default threshold')
    ax2.set_ylabel('Optimal Threshold')
    ax2.set_title('Optimized Classification Thresholds')
    
    # Rotate x-axis labels if needed
    if len(class_names) > 8:
        plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_balance_analysis.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--log_file", type=str, default="crossval.log", help="Log file path")
    parser.add_argument("--retrain", action="store_true", help="Force retraining of all folds")
    args = parser.parse_args()

    run_all_folds(
        config_file=args.config,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        log_file=args.log_file,
        retrain=args.retrain
    )