import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from datasets.daphnet import get_dataloader
from networks.diffusion import CSDI
from utils.misc import load_config, set_seed


def test(net, test_dl, config, n_samples=2, threshold=0.5, save_dir=None, strategy_name=""):
    """
    Perform model evaluation on test data and save visualizations.
    """
    with torch.no_grad():
        net.eval()
        
        all_preds = []
        all_labels = []
        all_recon_errors = []
        
        with tqdm(test_dl) as it:
            for batch_idx, data in enumerate(it, start=1):
                # Get model output
                output = net.evaluate(data, n_samples)
                samples, c_target, eval_points, observed_mask, _ = output
                samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
                c_target = c_target.permute(0, 2, 1)  # (B, L, K)
                
                # Handle NaN values
                samples = torch.nan_to_num(samples, nan=0.0)
                
                # Calculate median of samples
                samples_median = samples.median(dim=1).values
                
                # Calculate reconstruction error
                recon_error = torch.mean(torch.abs(samples_median - c_target), dim=-1)
                
                # Get test labels
                labels = data["label"].flatten().cpu().numpy()
                
                # Calculate anomaly scores (reconstruction error)
                anomaly_scores = recon_error.cpu().numpy()
                
                # Store all reconstruction errors for visualization
                all_recon_errors.extend(list(zip(anomaly_scores.flatten(), labels)))
                
                # Threshold-based prediction
                predictions = (anomaly_scores > threshold).astype(int)
                
                all_preds.extend(predictions.flatten())
                all_labels.extend(labels)
                
                it.set_postfix(
                    ordered_dict={
                        "processed_batches": batch_idx,
                    },
                    refresh=True,
                )
        
        # Calculate evaluation metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        sensitivity = recall  # Recall = Sensitivity
        if cm.shape[0] > 1 and cm.shape[1] > 1:
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        else:
            specificity = 0
        
        # Create visualizations if save_dir is provided
        if save_dir is not None:
            # Convert list of tuples to separate arrays for plotting
            recon_errors = np.array([x[0] for x in all_recon_errors])
            error_labels = np.array([x[1] for x in all_recon_errors])
            
            normal_errors = recon_errors[error_labels == 0]
            abnormal_errors = recon_errors[error_labels == 1]
            
            # 1. Scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(normal_errors)), normal_errors, c='blue', label='Normal', alpha=0.5)
            plt.scatter(range(len(normal_errors), len(recon_errors)), abnormal_errors, c='red', label='Abnormal', alpha=0.5)
            plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold = {threshold}')
            plt.xlabel('Sample Index')
            plt.ylabel('Reconstruction Error')
            plt.title(f'Reconstruction Error Distribution - {strategy_name}')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'scatter_recon_error_{strategy_name}.png'), dpi=300)
            plt.close()
            
            # 2. Boxplot
            plt.figure(figsize=(8, 6))
            box_data = [normal_errors, abnormal_errors]
            plt.boxplot(box_data, labels=['Normal', 'Abnormal'])
            plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold = {threshold}')
            plt.ylabel('Reconstruction Error')
            plt.title(f'Reconstruction Error Boxplot - {strategy_name}')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'boxplot_recon_error_{strategy_name}.png'), dpi=300)
            plt.close()
            
            # 3. Confusion Matrix Heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Normal', 'Abnormal'], 
                        yticklabels=['Normal', 'Abnormal'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - {strategy_name}')
            plt.savefig(os.path.join(save_dir, f'confusion_matrix_{strategy_name}.png'), dpi=300)
            plt.close()
            
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "confusion_matrix": cm,
            "normal_errors": normal_errors if save_dir is not None else None,
            "abnormal_errors": abnormal_errors if save_dir is not None else None
        }


def find_optimal_threshold(normal_errors, abnormal_errors):
    """
    Find the optimal threshold for separating normal and abnormal samples.
    """
    all_errors = np.concatenate([normal_errors, abnormal_errors])
    all_labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(abnormal_errors))])
    
    best_f1 = 0
    best_threshold = 0
    
    # Try different thresholds
    thresholds = np.linspace(np.min(all_errors), np.max(all_errors), 100)
    
    for threshold in thresholds:
        preds = (all_errors > threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def main(config, model_path, threshold=None, save_dir=None, sample_ratio=0.3, abnormal_ratio=0.1):
    """
    Load model and run tests with normal samples from train and abnormal samples from test.
    """
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    set_seed(config["etc"]["seed"])
    device = torch.device(
        config["etc"]["device"] if torch.cuda.is_available() else "cpu"
    )
    
    # Get data loaders (train and test)
    train_dl, _, test_0_dl, test_1_dl = get_dataloader(config)
    
    # Sample normal data from train dataset
    normal_samples = []
    normal_count = 0
    max_normal_samples = int(len(train_dl) * sample_ratio)
    
    print("Collecting normal samples from training data...")
    for batch in tqdm(train_dl):
        if normal_count >= max_normal_samples:
            break
        normal_samples.append(batch)
        normal_count += 1
    
    # Collect abnormal samples from test datasets
    all_abnormal_samples = []
    
    print("Finding all abnormal samples from test data...")
    # 전략 0에서 abnormal 샘플 수집
    for batch in tqdm(test_0_dl):
        labels = batch["label"]
        # 비정상 샘플(label=1)이 있는 배치만 추가
        if torch.any(labels > 0):
            all_abnormal_samples.append(batch)
    
    # 전략 1에서도 abnormal 샘플 수집
    for batch in tqdm(test_1_dl):
        labels = batch["label"]
        if torch.any(labels > 0):
            all_abnormal_samples.append(batch)
    
    print(f"Total abnormal batches found: {len(all_abnormal_samples)}")
    
    # 모든 abnormal 샘플 중 10%만 선택
    max_abnormal_samples = int(len(all_abnormal_samples) * abnormal_ratio)
    selected_indices = np.random.choice(len(all_abnormal_samples), max_abnormal_samples, replace=False)
    abnormal_samples = [all_abnormal_samples[i] for i in selected_indices]
    
    print(f"Collected {len(normal_samples)} normal batches from training data")
    print(f"Collected {len(abnormal_samples)} abnormal batches from test data (10% of all abnormal)")
    
    # Count class distribution
    normal_class_count = 0
    abnormal_class_count = 0
    
    for batch in normal_samples:
        labels = batch["label"].flatten().cpu().numpy()
        normal_class_count += np.sum(labels == 0)
        abnormal_class_count += np.sum(labels == 1)  # training data should only have normal but just in case
    
    abnormal_in_test_count = 0
    for batch in abnormal_samples:
        labels = batch["label"].flatten().cpu().numpy()
        abnormal_in_test_count += np.sum(labels == 1)
    
    print("\n===== Custom Test Set Distribution =====")
    print(f"Normal samples from train: {normal_class_count}")
    print(f"Abnormal samples from test: {abnormal_in_test_count}")
    print(f"Total samples: {normal_class_count + abnormal_in_test_count}")
    print(f"Ratio (Normal:Abnormal): {normal_class_count / (abnormal_in_test_count if abnormal_in_test_count > 0 else 1):.2f}:1")
    
    # Combine normal and abnormal samples
    combined_samples = normal_samples + abnormal_samples
    
    # Create a custom iterator
    class CustomDataIterator:
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    # Initialize model
    net = CSDI(config, device).to(device)
    
    # Load model weights
    net.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate on custom test set
    print("Evaluating on custom test set:")
    custom_results = test(net, CustomDataIterator(combined_samples), config, n_samples=2, 
                         threshold=0.5, save_dir=save_dir, strategy_name="custom")
    
    # Find optimal threshold
    if threshold is None:
        normal_errors = custom_results["normal_errors"]
        abnormal_errors = custom_results["abnormal_errors"]
        
        opt_threshold, opt_f1 = find_optimal_threshold(normal_errors, abnormal_errors)
        print(f"\nOptimal threshold found: {opt_threshold:.4f} with F1 score: {opt_f1:.4f}")
        threshold = opt_threshold
    else:
        print(f"\nUsing provided threshold: {threshold}")
    
    # Re-evaluate with optimal/provided threshold
    print("\nRe-evaluating with the optimal/provided threshold:")
    final_results = test(net, CustomDataIterator(combined_samples), config, n_samples=5, 
                        threshold=threshold, save_dir=save_dir, strategy_name="custom_optimal")
    
    # Plot distribution of reconstruction errors
    if save_dir is not None:
        normal_errors = final_results["normal_errors"]
        abnormal_errors = final_results["abnormal_errors"]
        
        # Histogram for reconstruction errors
        plt.figure(figsize=(10, 6))
        plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal', color='blue')
        plt.hist(abnormal_errors, bins=50, alpha=0.5, label='Abnormal', color='red')
        plt.axvline(x=threshold, color='green', linestyle='--', 
                   label=f'Threshold = {threshold:.4f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'reconstruction_error_histogram.png'), dpi=300)
        plt.close()
    
    # Output results
    print("\n===== Final Test Results =====")
    print(f"Accuracy: {final_results['accuracy']:.4f}")
    print(f"Sensitivity: {final_results['sensitivity']:.4f}")
    print(f"Specificity: {final_results['specificity']:.4f}")
    print(f"F1 Score: {final_results['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{final_results['confusion_matrix']}")
    
    # Save results to a text file
    if save_dir is not None:
        with open(os.path.join(save_dir, 'custom_test_results.txt'), 'w') as f:
            f.write("===== Custom Test Results =====\n")
            f.write(f"Normal samples from train (30%): {normal_class_count}\n")
            f.write(f"Abnormal samples from test (10%): {abnormal_in_test_count}\n")
            f.write(f"Threshold: {threshold:.4f}\n\n")
            f.write(f"Accuracy: {final_results['accuracy']:.4f}\n")
            f.write(f"Sensitivity: {final_results['sensitivity']:.4f}\n")
            f.write(f"Specificity: {final_results['specificity']:.4f}\n")
            f.write(f"F1 Score: {final_results['f1_score']:.4f}\n")
            f.write(f"Confusion Matrix:\n{final_results['confusion_matrix']}\n")
    
    return final_results, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/config_daphnet.json", help="Path to config file")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to model weights file")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Anomaly detection threshold (if None, optimal threshold will be found)")
    parser.add_argument("--output_dir", "-o", type=str, default="custom_results", help="Directory to save results and visualizations")
    parser.add_argument("--sample", "-s", type=float, default=0.01, help="Ratio of train data to sample (0-1)")
    parser.add_argument("--abnormal", "-a", type=float, default=0.01, help="Ratio of abnormal data to sample (0-1)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    main(config, args.model, threshold=args.threshold, save_dir=args.output_dir,
         sample_ratio=args.sample, abnormal_ratio=args.abnormal)