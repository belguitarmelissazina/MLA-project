#!/usr/bin/env python3
# Plotting Script

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


EXPERIMENTS = [
    'mnist_to_mnistm',
    'svhn_to_mnist',
    'syn_numbers_to_svhn',
    'syn_signs_to_gtsrb',
    'amazon_to_webcam',
    'dslr_to_webcam',
    'webcam_to_dslr',
]

# Office-31 experiments that need seed averaging
OFFICE31_EXPERIMENTS = ['amazon_to_webcam', 'dslr_to_webcam', 'webcam_to_dslr']
SEEDS = [0, 1, 2, 3, 4]


def load_single_history(experiment_name, model_type, results_dir):
    """Load a single history file (non-seeded experiments)."""
    candidates = [
        Path(results_dir) / f"{experiment_name}_{model_type}_history.json",
        Path(results_dir) / f"{experiment_name}_history.json" if model_type == 'dann' else None,
    ]
    
    for path in candidates:
        if path and path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                if 'target_acc' in data and data['target_acc']:
                    return data, len(data['target_acc'])
    return None, 0


def load_seeded_histories(experiment_name, results_dir):
    """
    Load and average seed histories for Office-31 experiments.
    Returns dict with 'dann' containing {'mean': [...], 'std': [...]}.
    """
    curves = []
    
    for seed in SEEDS:
        path = Path(results_dir) / f"{experiment_name}_seed{seed}_history.json"
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                if 'target_acc' in data and data['target_acc']:
                    curves.append(np.array(data['target_acc']))
    
    if not curves:
        return None
    
    # Truncate to shortest length
    min_len = min(len(c) for c in curves)
    curves = [c[:min_len] for c in curves]
    stacked = np.stack(curves, axis=0)
    
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    
    # Get final accuracy stats
    final_accs = [c[-1] for c in curves]
    final_mean = np.mean(final_accs)
    final_std = np.std(final_accs)
    
    print(f"Loaded dann: {len(curves)} seeds, {min_len} epochs (averaged)")
    
    return {
        'dann': {
            'target_acc': mean.tolist(),
            'std': std.tolist(),
            'n_seeds': len(curves),
            'final_acc': final_mean,
            'final_std': final_std
        }
    }


def plot_comparison(experiment_name, results_dir='./results', save_dir='./plots'):
    """Plot validation accuracy comparison."""
    
    is_office31 = experiment_name in OFFICE31_EXPERIMENTS
    histories = {}
    std_data = {}
    
    if is_office31:
        # Load and average seeded histories
        seeded = load_seeded_histories(experiment_name, results_dir)
        if seeded:
            histories['dann'] = seeded['dann']
            std_data['dann'] = seeded['dann'].get('std', None)
    else:
        # Load regular histories
        for model_type in ['dann', 'source_only', 'target_only']:
            data, n_epochs = load_single_history(experiment_name, model_type, results_dir)
            if data:
                histories[model_type] = data
                print(f"Loaded {model_type}: {n_epochs} epochs")

    if not histories:
        print(f"No history files found for {experiment_name}!")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {
        'source_only': '#1f77b4',  # Blue
        'dann': '#2ca02c',          # Green
        'target_only': '#9467bd',   # Purple
    }
    
    base_labels = {
        'source_only': 'Source-Only',
        'dann': 'DANN',
        'target_only': 'Target-Only',
    }
    
    for model_type, history in histories.items():
        target_acc = np.array(history['target_acc'])
        epochs = range(1, len(target_acc) + 1)
        
        # Build label with final accuracy
        final_acc = target_acc[-1]
        
        if is_office31 and model_type == 'dann':
            # Show mean ± std for seeded experiments
            final_std = history.get('final_std', 0)
            n_seeds = history.get('n_seeds', 0)
            label = f"{base_labels[model_type]}: {final_acc:.2f} ± {final_std:.2f} (n={n_seeds})"
        else:
            label = f"{base_labels[model_type]}: {final_acc:.2f}"
        
        ax.plot(epochs, target_acc, 
                label=label, 
                color=colors[model_type], 
                linewidth=2.5)
        
        # Add std shading for seeded experiments
        if model_type in std_data and std_data[model_type] is not None:
            std = np.array(std_data[model_type])
            ax.fill_between(epochs, target_acc - std, target_acc + std, 
                           color=colors[model_type], alpha=0.2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title(f"Classification Accuracy: {experiment_name.replace('_', ' → ').upper()}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.0, 1.0)
    
    plt.tight_layout()
    
    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"{experiment_name}_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot DANN experiment results')
    parser.add_argument('experiment', type=str, nargs='?', default=None,
                        choices=EXPERIMENTS,
                        help=f'Experiment name. Choices: {", ".join(EXPERIMENTS)}')
    parser.add_argument('--all', action='store_true',
                        help='Plot all experiments')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing history JSON files (default: ./results)')
    parser.add_argument('--save-dir', type=str, default='./plots',
                        help='Directory to save plots (default: ./plots)')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("Available experiments:")
        for exp in EXPERIMENTS:
            note = " (multi-seed averaged)" if exp in OFFICE31_EXPERIMENTS else ""
            print(f"  - {exp}{note}")
        return
    
    if args.all:
        for exp in EXPERIMENTS:
            print(f"\n{'='*50}")
            print(f"Plotting {exp}")
            print('='*50)
            plot_comparison(exp, args.results_dir, args.save_dir)
    elif args.experiment:
        plot_comparison(args.experiment, args.results_dir, args.save_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()