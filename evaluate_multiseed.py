"""
Multi-Seed Evaluation Script

Evaluates multiple checkpoint seeds and reports mean ± std.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import subprocess
import sys


def run_single_eval(checkpoint, source, target, results_dir='./results', tsne=True):
    """Run evaluation for a single checkpoint."""
    cmd = [
        sys.executable, 'eval.py',
        '--dann_checkpoint', checkpoint,
        '--source', source,
        '--target', target,
        '--results_dir', results_dir
    ]
    
    if tsne:
        cmd.append('--tsne')

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return None

    # Parse results from JSON file
    checkpoint_name = Path(checkpoint).stem
    results_file = Path(results_dir) / f"{source}_to_{target}_eval_evaluation.json"

    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)

    return None


def evaluate_multiseed(source, target, seeds, checkpoint_pattern='checkpoints/{source}_to_{target}_seed{seed}_dann_final.pt', tsne=True):
    """
    Evaluate multiple seeds and compute mean ± std.

    Args:
        source: Source domain name
        target: Target domain name
        seeds: List of seed numbers
        checkpoint_pattern: Pattern for checkpoint filenames
        tsne: Whether to generate t-SNE visualizations
    """
    print(f"{'='*80}")
    print(f"Multi-Seed Evaluation: {source} → {target}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}")

    results = []

    for seed in seeds:
        checkpoint = checkpoint_pattern.format(source=source, target=target, seed=seed)
        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint}")
            continue

        result = run_single_eval(checkpoint, source, target, tsne=tsne)

        if result and 'dann' in result:
            target_acc = result['dann']['target_accuracy']
            source_acc = result['dann']['source_accuracy']
            results.append({
                'seed': seed,
                'target_accuracy': target_acc,
                'source_accuracy': source_acc
            })
            print(f"Seed {seed}: Target Acc = {target_acc:.4f}, Source Acc = {source_acc:.4f}")
        else:
            print(f"Warning: Could not get results for seed {seed}")

    if not results:
        print(f"\nNo results found for {source} → {target}")
        return

    # Compute statistics
    target_accs = [r['target_accuracy'] for r in results]
    source_accs = [r['source_accuracy'] for r in results]

    target_mean = np.mean(target_accs)
    target_std = np.std(target_accs)
    source_mean = np.mean(source_accs)
    source_std = np.std(source_accs)

    print(f"\n{'='*80}")
    print(f"RESULTS: {source} → {target}")
    print(f"{'='*80}")
    print(f"Number of seeds evaluated: {len(results)}")
    print(f"Target Accuracy: {target_mean:.2f}% ± {target_std:.2f}%")
    print(f"Source Accuracy: {source_mean:.2f}% ± {source_std:.2f}%")
    print(f"{'='*80}\n")

    # Save aggregated results
    output_file = Path('results') / f'{source}_to_{target}_multiseed_results.json'
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'source': source,
            'target': target,
            'seeds': seeds,
            'num_evaluated': len(results),
            'target_accuracy_mean': float(target_mean),
            'target_accuracy_std': float(target_std),
            'source_accuracy_mean': float(source_mean),
            'source_accuracy_std': float(source_std),
            'individual_results': results
        }, f, indent=2)

    print(f"Saved results to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-seed evaluation')
    parser.add_argument('--source', type=str, required=True, help='Source domain')
    parser.add_argument('--target', type=str, required=True, help='Target domain')
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help='List of seeds')
    parser.add_argument('--checkpoint_pattern', type=str,
                        default='checkpoints/{source}_to_{target}_seed{seed}_dann_final.pt',
                        help='Checkpoint filename pattern')
    parser.add_argument('--no_tsne', action='store_true', help='Disable t-SNE visualization')

    args = parser.parse_args()

    evaluate_multiseed(args.source, args.target, args.seeds, args.checkpoint_pattern, tsne=not args.no_tsne)


if __name__ == '__main__':
    main()