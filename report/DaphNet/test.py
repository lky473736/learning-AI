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


def main(config, model_path, threshold=None, save_dir=None, sample_ratio=0.3):
    """
    Load model and run tests with sampling.
    """
    if save_dir is not None and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    set_seed(config["etc"]["seed"])
    device = torch.device(
        config["etc"]["device"] if torch.cuda.is_available() else "cpu"
    )
    
    # Get data loaders
    _, _, test_0_dl, test_1_dl = get_dataloader(config)
    
    # Sample the test data to reduce processing time
    def sample_dataloader(dataloader, ratio=0.3):
        all_data = []
        for batch in dataloader:
            all_data.append(batch)
        
        # Calculate sample size
        total_size = len(all_data)
        sample_size = max(1, int(total_size * ratio))
        
        # Randomly sample batches
        indices = np.random.choice(total_size, sample_size, replace=False)
        return [all_data[i] for i in indices]
    
    # Sample the test dataloaders
    sampled_test_0 = sample_dataloader(test_0_dl, ratio=sample_ratio)
    sampled_test_1 = sample_dataloader(test_1_dl, ratio=sample_ratio)
    
    print(f"Sampled {len(sampled_test_0)} batches from test strategy 0 (original: {len(test_0_dl)})")
    print(f"Sampled {len(sampled_test_1)} batches from test strategy 1 (original: {len(test_1_dl)})")
    
    # Count class distribution in sampled test_0_dl
    class_0_count_sampled_0 = 0
    class_1_count_sampled_0 = 0
    
    print("Counting class distribution in sampled test strategy 0...")
    for batch in sampled_test_0:
        labels = batch["label"].flatten().cpu().numpy()
        class_0_count_sampled_0 += np.sum(labels == 0)
        class_1_count_sampled_0 += np.sum(labels == 1)
    
    # Count class distribution in sampled test_1_dl
    class_0_count_sampled_1 = 0
    class_1_count_sampled_1 = 0
    
    print("Counting class distribution in sampled test strategy 1...")
    for batch in sampled_test_1:
        labels = batch["label"].flatten().cpu().numpy()
        class_0_count_sampled_1 += np.sum(labels == 0)
        class_1_count_sampled_1 += np.sum(labels == 1)
    
    print("\n===== Sampled Class Distribution =====")
    print("Test Strategy 0:")
    print(f"  Normal (class 0): {class_0_count_sampled_0}")
    print(f"  Abnormal (class 1): {class_1_count_sampled_0}")
    print(f"  Total: {class_0_count_sampled_0 + class_1_count_sampled_0}")
    print(f"  Ratio (Normal:Abnormal): {class_0_count_sampled_0 / (class_1_count_sampled_0 if class_1_count_sampled_0 > 0 else 1):.2f}:1")
    
    print("\nTest Strategy 1:")
    print(f"  Normal (class 0): {class_0_count_sampled_1}")
    print(f"  Abnormal (class 1): {class_1_count_sampled_1}")
    print(f"  Total: {class_0_count_sampled_1 + class_1_count_sampled_1}")
    print(f"  Ratio (Normal:Abnormal): {class_0_count_sampled_1 / (class_1_count_sampled_1 if class_1_count_sampled_1 > 0 else 1):.2f}:1")
    
    # Initialize model
    net = CSDI(config, device).to(device)
    
    # Load model weights
    net.load_state_dict(torch.load(model_path, map_location=device))
        
    # Create a simple iterator for the sampled batches
    class BatchIterator:
        def __init__(self, batches):
            self.batches = batches
        
        def __iter__(self):
            return iter(self.batches)
        
        def __len__(self):
            return len(self.batches)
    
    print("Evaluating on test strategy 0:")
    results_0 = test(net, BatchIterator(sampled_test_0), config, n_samples=2, threshold=0.5, 
                     save_dir=save_dir, strategy_name="strategy_0")
    
    print("Evaluating on test strategy 1:")
    results_1 = test(net, BatchIterator(sampled_test_1), config, n_samples=2, threshold=0.5, 
                     save_dir=save_dir, strategy_name="strategy_1")
    
    # If threshold is not provided, find optimal threshold
    if threshold is None:
        # Combine errors from both strategies
        normal_errors = np.concatenate([results_0["normal_errors"], results_1["normal_errors"]])
        abnormal_errors = np.concatenate([results_0["abnormal_errors"], results_1["abnormal_errors"]])
        
        opt_threshold, opt_f1 = find_optimal_threshold(normal_errors, abnormal_errors)
        print(f"\nOptimal threshold found: {opt_threshold:.4f} with F1 score: {opt_f1:.4f}")
        threshold = opt_threshold
    else:
        print(f"\nUsing provided threshold: {threshold}")
    
    # Re-run test with optimal/provided threshold
    print("\nRe-evaluating with the optimal/provided threshold:")
    results_0 = test(net, test_0_dl, config, n_samples=10, threshold=threshold, 
                     save_dir=save_dir, strategy_name="strategy_0_optimal")
    
    results_1 = test(net, test_1_dl, config, n_samples=10, threshold=threshold, 
                     save_dir=save_dir, strategy_name="strategy_1_optimal")
    
    # Calculate average results
    avg_results = {
        "accuracy": (results_0["accuracy"] + results_1["accuracy"]) / 2,
        "sensitivity": (results_0["sensitivity"] + results_1["sensitivity"]) / 2,
        "specificity": (results_0["specificity"] + results_1["specificity"]) / 2,
        "f1_score": (results_0["f1_score"] + results_1["f1_score"]) / 2,
    }
    
    # Plot distribution of reconstruction errors for combined strategies
    if save_dir is not None:
        normal_errors = np.concatenate([results_0["normal_errors"], results_1["normal_errors"]])
        abnormal_errors = np.concatenate([results_0["abnormal_errors"], results_1["abnormal_errors"]])
        
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
    print("\n===== Test Results =====")
    print("Strategy 0 Results:")
    print(f"Accuracy: {results_0['accuracy']:.4f}")
    print(f"Sensitivity: {results_0['sensitivity']:.4f}")
    print(f"Specificity: {results_0['specificity']:.4f}")
    print(f"F1 Score: {results_0['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{results_0['confusion_matrix']}")
    
    print("\nStrategy 1 Results:")
    print(f"Accuracy: {results_1['accuracy']:.4f}")
    print(f"Sensitivity: {results_1['sensitivity']:.4f}")
    print(f"Specificity: {results_1['specificity']:.4f}")
    print(f"F1 Score: {results_1['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{results_1['confusion_matrix']}")
    
    print("\nAverage Results:")
    print(f"Accuracy: {avg_results['accuracy']:.4f}")
    print(f"Sensitivity: {avg_results['sensitivity']:.4f}")
    print(f"Specificity: {avg_results['specificity']:.4f}")
    print(f"F1 Score: {avg_results['f1_score']:.4f}")
    
    # Save results to a text file
    if save_dir is not None:
        with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
            f.write("===== Test Results =====\n")
            f.write(f"Optimal/Provided Threshold: {threshold:.4f}\n\n")
            
            f.write("Strategy 0 Results:\n")
            f.write(f"Accuracy: {results_0['accuracy']:.4f}\n")
            f.write(f"Sensitivity: {results_0['sensitivity']:.4f}\n")
            f.write(f"Specificity: {results_0['specificity']:.4f}\n")
            f.write(f"F1 Score: {results_0['f1_score']:.4f}\n")
            f.write(f"Confusion Matrix:\n{results_0['confusion_matrix']}\n\n")
            
            f.write("Strategy 1 Results:\n")
            f.write(f"Accuracy: {results_1['accuracy']:.4f}\n")
            f.write(f"Sensitivity: {results_1['sensitivity']:.4f}\n")
            f.write(f"Specificity: {results_1['specificity']:.4f}\n")
            f.write(f"F1 Score: {results_1['f1_score']:.4f}\n")
            f.write(f"Confusion Matrix:\n{results_1['confusion_matrix']}\n\n")
            
            f.write("Average Results:\n")
            f.write(f"Accuracy: {avg_results['accuracy']:.4f}\n")
            f.write(f"Sensitivity: {avg_results['sensitivity']:.4f}\n")
            f.write(f"Specificity: {avg_results['specificity']:.4f}\n")
            f.write(f"F1 Score: {avg_results['f1_score']:.4f}\n")
    
    return avg_results, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="configs/config_daphnet.json", help="Path to config file")
    parser.add_argument("--model", "-m", type=str, required=True, help="Path to model weights file")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Anomaly detection threshold (if None, optimal threshold will be found)")
    parser.add_argument("--output_dir", "-o", type=str, default="results", help="Directory to save results and visualizations")
    parser.add_argument("--sample", "-s", type=float, default=0.3, help="Ratio of test data to sample (0-1)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    main(config, args.model, threshold=args.threshold, save_dir=args.output_dir, sample_ratio=args.sample)