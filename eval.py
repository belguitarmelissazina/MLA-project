"""
DANN Evaluation Script
Authors: Ganin & Lempitsky, 2015 Reproduction

This script implements:
- Source and target accuracy evaluation
- Domain classifier accuracy
- t-SNE feature visualization
"""

import os
import argparse
import yaml
import warnings
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

# Suppress ALL warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DANN, create_dann_model, create_source_only_model
from data import get_dataloaders, get_experiment_info, _get_experiment_config, _create_dataset
from utils import (
    set_seed, get_device,
    compute_accuracy, compute_domain_accuracy, get_confusion_matrix,
    extract_features, load_checkpoint,
    plot_tsne, plot_training_curves,
    save_metrics_json, save_metrics_csv
)


def evaluate_model(
    model: nn.Module,
    source_test_loader: DataLoader,
    target_test_loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict:
    """
    Comprehensive evaluation of a trained model.
    
    Args:
        model: Trained model (DANN or SourceOnly)
        source_test_loader: Source domain test data
        target_test_loader: Target domain test data
        device: Device to use
        num_classes: Number of classes
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Compute accuracies
    source_acc = compute_accuracy(model, source_test_loader, device)
    target_acc = compute_accuracy(model, target_test_loader, device)
    
    # Domain accuracy (only for DANN)
    domain_acc = None
    if hasattr(model, 'domain_classifier'):
        domain_acc = compute_domain_accuracy(
            model, source_test_loader, target_test_loader, device
        )
    
    return {
        'source_accuracy': source_acc,
        'target_accuracy': target_acc,
        'domain_accuracy': domain_acc
    }


def generate_tsne_visualizations(
    model: nn.Module,
    source_test_loader: DataLoader,
    target_test_loader: DataLoader,
    device: torch.device,
    save_dir: str,
    experiment_name: str,
    max_samples: int = 2000
) -> None:
    """
    Generate t-SNE visualizations of feature distributions.
    
    Args:
        model: Feature extractor model
        source_test_loader: Source test data
        target_test_loader: Target test data
        device: Device to use
        save_dir: Directory to save plots
        experiment_name: Name for saved files
        max_samples: Maximum samples per domain
    """
    print(f"  Generating t-SNE visualization...")
    
    # Extract features (silently)
    source_features, source_labels = extract_features(
        model, source_test_loader, device, max_samples=max_samples
    )
    target_features, target_labels = extract_features(
        model, target_test_loader, device, max_samples=max_samples
    )

    # Generate and save t-SNE plot
    save_path = os.path.join(save_dir, f"{experiment_name}_tsne.png")
    plot_tsne(
        source_features, source_labels,
        target_features, target_labels,
        save_path=save_path,
        title=f"t-SNE Visualization - {experiment_name}"
    )
    
    print(f"  t-SNE saved to: {save_path}")


def run_evaluation(config: Dict) -> Dict:
    """
    Run comprehensive evaluation for a trained model.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Dictionary of all evaluation results
    """
    set_seed(config.get('seed', 42))
    device = get_device()

    # Quiet mode suppresses verbose output
    quiet = config.get('quiet', False)
    
    # Create directories
    os.makedirs(config['results_dir'], exist_ok=True)

    # Load checkpoint first to get architecture info
    checkpoint_path = config.get('dann_checkpoint') or config.get('source_only_checkpoint')
    checkpoint_source = None
    checkpoint_target = None
    
    if checkpoint_path:
        temp_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Get architecture from checkpoint config if available
        checkpoint_config = temp_checkpoint.get('config', {})
        checkpoint_source = checkpoint_config.get('source')
        checkpoint_target = checkpoint_config.get('target')
        
        if checkpoint_source and checkpoint_target:
            # Use original source/target from training to get correct architecture
            exp_info = get_experiment_info(checkpoint_source, checkpoint_target)
            if not quiet:
                print(f"Using architecture from checkpoint config: {exp_info['architecture']}")
        else:
            # Fallback to inferring from command line source/target
            exp_info = get_experiment_info(config['source'], config['target'])
            if not quiet:
                print(f"Inferred architecture from source/target: {exp_info['architecture']}")
    else:
        exp_info = get_experiment_info(config['source'], config['target'])

    # Create dataloaders with correct transforms for the architecture
    # Use checkpoint's source/target to get correct transforms, but load command-line datasets
    if checkpoint_source and checkpoint_target:
        # Get transforms based on checkpoint architecture
        arch_config = _get_experiment_config(checkpoint_source, checkpoint_target)
        transform_test = arch_config['transform_test']
        
        # Load actual datasets with architecture-appropriate transforms
        source_test_dataset = _create_dataset(
            config['source'], config['data_root'], train=False,
            transform=transform_test, download=False
        )
        target_test_dataset = _create_dataset(
            config['target'], config['data_root'], train=False,
            transform=transform_test, download=False
        )
        
        source_test = DataLoader(
            source_test_dataset,
            batch_size=config.get('batch_size', 128),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        target_test = DataLoader(
            target_test_dataset,
            batch_size=config.get('batch_size', 128),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
    else:
        # Fallback to standard data loading
        source_train, source_test, target_train, target_test = get_dataloaders(
            source=config['source'],
            target=config['target'],
            data_root=config['data_root'],
            batch_size=config.get('batch_size', 128),
            num_workers=config.get('num_workers', 4),
            download=False
        )

    results = {}

    # Load and evaluate DANN model
    if config.get('dann_checkpoint'):
        # Detect model type from checkpoint filename
        checkpoint_name = os.path.basename(config['dann_checkpoint'])
        if 'source_only' in checkpoint_name:
            model_type = 'source_only'
            if not quiet:
                print("Loading Source-Only model...")
        elif 'target_only' in checkpoint_name:
            model_type = 'target_only'
            if not quiet:
                print("Loading Target-Only model...")
        else:
            model_type = 'dann'
            if not quiet:
                print("Loading DANN model...")

        dann_model = create_dann_model(
            architecture=exp_info['architecture'],
            num_classes=exp_info['num_classes'],
            pretrained=False
        )
        load_checkpoint(dann_model, config['dann_checkpoint'], device=device)
        dann_model = dann_model.to(device)

        if not quiet:
            print(f"Evaluating {model_type}...")
        dann_metrics = evaluate_model(
            dann_model, source_test, target_test, device, exp_info['num_classes']
        )
        results[model_type] = dann_metrics

        print(f"{model_type} - Source: {dann_metrics['source_accuracy']:.4f}, "
              f"Target: {dann_metrics['target_accuracy']:.4f}")

        # t-SNE visualization
        if config.get('tsne', True):
            generate_tsne_visualizations(
                dann_model, source_test, target_test, device,
                config['results_dir'], f"{config['experiment_name']}_{model_type}"
            )
    
    # Load and evaluate Source-Only model
    if config.get('source_only_checkpoint'):
        if not quiet:
            print("Loading Source-Only model...")
        source_model = create_source_only_model(
            architecture=exp_info['architecture'],
            num_classes=exp_info['num_classes'],
            pretrained=False
        )
        load_checkpoint(source_model, config['source_only_checkpoint'], device=device)
        source_model = source_model.to(device)

        if not quiet:
            print("Evaluating Source-Only...")
        source_metrics = evaluate_model(
            source_model, source_test, target_test, device, exp_info['num_classes']
        )
        results['source_only'] = source_metrics

        print(f"Source-Only - Source: {source_metrics['source_accuracy']:.4f}, "
              f"Target: {source_metrics['target_accuracy']:.4f}")
        
        # t-SNE visualization for non-adapted features
        if config.get('tsne', True):
            generate_tsne_visualizations(
                source_model, source_test, target_test, device,
                config['results_dir'], f"{config['experiment_name']}_source_only"
            )
    
    # Save all results
    results_path = os.path.join(
        config['results_dir'],
        f"{config['experiment_name']}_evaluation.json"
    )
    save_metrics_json(results, results_path)
    if not quiet:
        print(f"\nResults saved to {results_path}")
    
    return results


def evaluate_multiple_seeds(
    base_config: Dict,
    seeds: List[int]
) -> Dict:
    """
    Evaluate models trained with multiple seeds and compute statistics.
    
    Args:
        base_config: Base configuration
        seeds: List of random seeds
        
    Returns:
        Aggregated results with mean and std
    """
    all_results = []
    
    for seed in seeds:
        config = base_config.copy()
        config['seed'] = seed
        config['experiment_name'] = f"{base_config['experiment_name']}_seed{seed}"
        
        # Update checkpoint paths
        if 'dann_checkpoint' in config:
            config['dann_checkpoint'] = config['dann_checkpoint'].replace(
                '.pt', f'_seed{seed}.pt'
            )
        if 'source_only_checkpoint' in config:
            config['source_only_checkpoint'] = config['source_only_checkpoint'].replace(
                '.pt', f'_seed{seed}.pt'
            )
        
        try:
            results = run_evaluation(config)
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating seed {seed}: {e}")
    
    if not all_results:
        return {}
    
    # Aggregate results
    aggregated = {
        'seeds': seeds,
        'num_seeds': len(all_results)
    }
    
    # DANN metrics
    if 'dann' in all_results[0]:
        dann_target_accs = [r['dann']['target_accuracy'] for r in all_results]
        dann_source_accs = [r['dann']['source_accuracy'] for r in all_results]
        
        aggregated['dann'] = {
            'target_accuracy_mean': np.mean(dann_target_accs),
            'target_accuracy_std': np.std(dann_target_accs),
            'source_accuracy_mean': np.mean(dann_source_accs),
            'source_accuracy_std': np.std(dann_source_accs),
            'all_target_accuracies': dann_target_accs,
            'all_source_accuracies': dann_source_accs
        }
    
    # Source-only metrics
    if 'source_only' in all_results[0]:
        so_target_accs = [r['source_only']['target_accuracy'] for r in all_results]
        so_source_accs = [r['source_only']['source_accuracy'] for r in all_results]
        
        aggregated['source_only'] = {
            'target_accuracy_mean': np.mean(so_target_accs),
            'target_accuracy_std': np.std(so_target_accs),
            'source_accuracy_mean': np.mean(so_source_accs),
            'source_accuracy_std': np.std(so_source_accs),
            'all_target_accuracies': so_target_accs,
            'all_source_accuracies': so_source_accs
        }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate DANN models")
    
    # Data arguments
    parser.add_argument('--source', type=str, required=True,
                       help='Source domain')
    parser.add_argument('--target', type=str, required=True,
                       help='Target domain')
    parser.add_argument('--data_root', type=str, default='./datasets',
                       help='Root directory for datasets')
    
    # Model arguments
    parser.add_argument('--dann_checkpoint', type=str, default=None,
                       help='Path to DANN checkpoint')
    parser.add_argument('--source_only_checkpoint', type=str, default=None,
                       help='Path to source-only checkpoint')
    
    # Evaluation arguments
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for experiment')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory for results')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Evaluation options
    parser.add_argument('--tsne', action='store_true', default=True,
                       help='Generate t-SNE visualizations')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Build config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = vars(args)
    
    # Set experiment name if not provided
    if config.get('experiment_name') is None:
        config['experiment_name'] = f"{config['source']}_to_{config['target']}_eval"
    
    # Run evaluation
    results = run_evaluation(config)
    
    # Print summary
    print(f"\nEvaluation Summary")
    
    if 'dann' in results:
        print(f"DANN Target Accuracy: {results['dann']['target_accuracy']:.4f}")
    if 'source_only' in results:
        print(f"Source-Only Target Accuracy: {results['source_only']['target_accuracy']:.4f}")
    if 'target_only' in results:
        print(f"Target-Only Target Accuracy: {results['target_only']['target_accuracy']:.4f}")
    
    if 'dann' in results and 'source_only' in results:
        improvement = results['dann']['target_accuracy'] - results['source_only']['target_accuracy']
        print(f"Improvement: {improvement:+.4f}")


if __name__ == '__main__':
    main()