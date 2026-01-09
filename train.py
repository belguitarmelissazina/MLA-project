"""
DANN Training Script
"""

import os
import argparse
import yaml
import time
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DANN, create_dann_model, create_source_only_model
from data import get_dataloaders, get_experiment_info
from utils import (
    set_seed, get_device, 
    LambdaScheduler, LRScheduler, compute_lambda, compute_lr,
    compute_accuracy, compute_domain_accuracy,
    save_checkpoint, load_checkpoint,
    ExperimentLogger, save_metrics_json, save_metrics_csv,
    print_model_summary, format_time
)


def train_dann(
    model: DANN,
    source_train_loader: DataLoader,
    target_train_loader: DataLoader,
    source_test_loader: DataLoader,
    target_test_loader: DataLoader,
    config: Dict,
    device: torch.device,
    logger: ExperimentLogger,
    architecture: str
) -> Dict:
    model = model.to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # Create optimizer with parameter groups for Office architecture
    base_lr = config['lr']

    if architecture == 'office':
        # Office: 10x LR for new layers (bottleneck + heads)
        pretrained_params = []
        new_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # New layers: bottleneck + label head + domain head
            if ("bottleneck" in name) or ("label_predictor" in name) or ("domain_classifier" in name):
                new_params.append(p)
            else:
                pretrained_params.append(p)

        optimizer = optim.SGD(
            [
                {"params": pretrained_params, "lr": base_lr},
                {"params": new_params, "lr": 10.0 * base_lr},
            ],
            momentum=config["momentum"]
        )
    else:
        # Other architectures: standard single LR
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum']
        )
    
    # Calculate total steps
    steps_per_epoch = min(len(source_train_loader), len(target_train_loader))
    total_steps = config['epochs'] * steps_per_epoch

    lr_scheduler = LRScheduler(
        optimizer,
        total_steps,
        mu_0=config['lr'],
        alpha=config.get('lr_alpha', 10.0),
        beta=config.get('lr_beta', 0.75)
    )

    lambda_scheduler = LambdaScheduler(
        total_steps,
        gamma=config.get('gamma', 10.0)
    )
    
    logger.log(f"Starting DANN training for {config['epochs']} epochs")
    logger.log(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")

    global_step = 0

    for epoch in range(config['epochs']):
        model.train()
        epoch_class_loss = 0.0
        epoch_domain_loss = 0.0
        num_batches = 0

        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{config['epochs']}")

        for step in pbar:
            try:
                source_images, source_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(source_train_loader)
                source_images, source_labels = next(source_iter)

            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_train_loader)
                target_images, _ = next(target_iter)

            source_images = source_images.to(device)
            source_labels = source_labels.to(device)
            target_images = target_images.to(device)

            current_lambda = lambda_scheduler.step()
            current_lr = lr_scheduler.step()

            # Maintain 10x ratio for Office architecture
            if architecture == 'office':
                optimizer.param_groups[0]["lr"] = current_lr
                optimizer.param_groups[1]["lr"] = 10.0 * current_lr

            combined_images = torch.cat([source_images, target_images], dim=0)

            domain_labels = torch.cat([
                torch.zeros(source_images.size(0)),
                torch.ones(target_images.size(0))
            ]).unsqueeze(1).to(device)

            optimizer.zero_grad()

            class_logits, domain_logits, _ = model(combined_images, alpha=current_lambda)

            source_class_logits = class_logits[:source_images.size(0)]
            class_loss = class_criterion(source_class_logits, source_labels)

            domain_loss = domain_criterion(domain_logits, domain_labels)

            total_loss = class_loss + domain_loss

            total_loss.backward()

            if architecture == 'svhn':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_class_loss += class_loss.item()
            epoch_domain_loss += domain_loss.item()
            num_batches += 1
            global_step += 1

            logger.log_step(
                global_step,
                class_loss.item(),
                domain_loss.item(),
                current_lambda,
                current_lr
            )

            pbar.set_postfix({
                'c_loss': f'{class_loss.item():.4f}',
                'd_loss': f'{domain_loss.item():.4f}',
                'λ': f'{current_lambda:.3f}',
                'lr': f'{current_lr:.6f}'
            })

        avg_class_loss = epoch_class_loss / num_batches
        avg_domain_loss = epoch_domain_loss / num_batches

        source_acc = compute_accuracy(model, source_test_loader, device)
        target_acc = compute_accuracy(model, target_test_loader, device)
        domain_acc = compute_domain_accuracy(
            model, source_test_loader, target_test_loader, device, current_lambda
        )

        logger.log_epoch(epoch + 1, source_acc, target_acc, domain_acc)
        logger.log(
            f"Epoch {epoch+1}: Class Loss={avg_class_loss:.4f}, "
            f"Domain Loss={avg_domain_loss:.4f}"
        )

    final_checkpoint_path = os.path.join(
        config['checkpoint_dir'],
        f"{config['experiment_name']}_dann_final.pt"
    )
    save_checkpoint(
        model, optimizer, config['epochs'],
        {'source_acc': source_acc, 'target_acc': target_acc},
        final_checkpoint_path, config
    )

    logger.save_history()

    return {
        'source_acc': source_acc,
        'target_acc': target_acc,
        'domain_acc': domain_acc
    }


def train_source_only(
    model: nn.Module,
    source_train_loader: DataLoader,
    source_test_loader: DataLoader,
    target_test_loader: DataLoader,
    config: Dict,
    device: torch.device,
    logger: ExperimentLogger,
    architecture: str,
    model_name: str = "source_only"
) -> Dict:
    model = model.to(device)

    class_criterion = nn.CrossEntropyLoss()

    # Create optimizer with parameter groups for Office architecture
    base_lr = config['lr']

    if architecture == 'office':
        # Office: 10x LR for new layers (bottleneck + classifier)
        pretrained_params = []
        new_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # For source-only Office model: bottleneck + label predictor are new
            if ("bottleneck" in name) or ("label_predictor" in name):
                new_params.append(p)
            else:
                pretrained_params.append(p)

        optimizer = optim.SGD(
            [
                {"params": pretrained_params, "lr": base_lr},
                {"params": new_params, "lr": 10.0 * base_lr},
            ],
            momentum=config["momentum"]
        )
    else:
        # Other architectures: standard single LR
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum']
        )

    steps_per_epoch = len(source_train_loader)
    total_steps = config['epochs'] * steps_per_epoch

    lr_scheduler = LRScheduler(
        optimizer,
        total_steps,
        mu_0=config['lr'],
        alpha=config.get('lr_alpha', 10.0),
        beta=config.get('lr_beta', 0.75)
    )

    logger.log(f"Starting {model_name} training for {config['epochs']} epochs")

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(source_train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            current_lr = lr_scheduler.step()

            # Maintain 10x ratio for Office architecture
            if architecture == 'office':
                optimizer.param_groups[0]["lr"] = current_lr
                optimizer.param_groups[1]["lr"] = 10.0 * current_lr

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = class_criterion(logits, labels)
            loss.backward()

            if architecture == 'svhn':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})

        source_acc = compute_accuracy(model, source_test_loader, device)
        target_acc = compute_accuracy(model, target_test_loader, device)

        logger.log_epoch(epoch + 1, source_acc, target_acc)

    final_checkpoint_path = os.path.join(
        config['checkpoint_dir'],
        f"{config['experiment_name']}_{model_name}_final.pt"
    )
    save_checkpoint(
        model, optimizer, config['epochs'],
        {'source_acc': source_acc, 'target_acc': target_acc},
        final_checkpoint_path, config
    )

    logger.save_history()

    return {
        'source_acc': source_acc,
        'target_acc': target_acc
    }


def train_target_only(
    model: nn.Module,
    target_train_loader: DataLoader,
    target_test_loader: DataLoader,
    config: Dict,
    device: torch.device,
    logger: ExperimentLogger,
    architecture: str
) -> Dict:
    model = model.to(device)

    class_criterion = nn.CrossEntropyLoss()

    # Create optimizer with parameter groups for Office architecture
    base_lr = config['lr']

    if architecture == 'office':
        # Office: 10x LR for new layers (bottleneck + classifier)
        pretrained_params = []
        new_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # For target-only Office model: bottleneck + label predictor are new
            if ("bottleneck" in name) or ("label_predictor" in name):
                new_params.append(p)
            else:
                pretrained_params.append(p)

        optimizer = optim.SGD(
            [
                {"params": pretrained_params, "lr": base_lr},
                {"params": new_params, "lr": 10.0 * base_lr},
            ],
            momentum=config["momentum"]
        )
    else:
        # Other architectures: standard single LR
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum']
        )

    steps_per_epoch = len(target_train_loader)
    total_steps = config['epochs'] * steps_per_epoch

    lr_scheduler = LRScheduler(
        optimizer,
        total_steps,
        mu_0=config['lr'],
        alpha=config.get('lr_alpha', 10.0),
        beta=config.get('lr_beta', 0.75)
    )

    logger.log(f"Starting Train-on-Target (upper bound) for {config['epochs']} epochs")

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(target_train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            current_lr = lr_scheduler.step()

            # Maintain 10x ratio for Office architecture
            if architecture == 'office':
                optimizer.param_groups[0]["lr"] = current_lr
                optimizer.param_groups[1]["lr"] = 10.0 * current_lr

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = class_criterion(logits, labels)
            loss.backward()

            if architecture == 'svhn':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})

        target_acc = compute_accuracy(model, target_test_loader, device)

        logger.log_epoch(epoch + 1, target_acc, target_acc)

    final_checkpoint_path = os.path.join(
        config['checkpoint_dir'],
        f"{config['experiment_name']}_target_only_final.pt"
    )
    save_checkpoint(
        model, optimizer, config['epochs'],
        {'target_acc': target_acc},
        final_checkpoint_path, config
    )

    logger.save_history()

    return {
        'target_acc': target_acc
    }


def run_experiment(config: Dict) -> Dict:
    set_seed(config['seed'])
    device = get_device()

    print(f"\nRunning experiment: {config['experiment_name']}")
    print(f"Source: {config['source']} -> Target: {config['target']}")
    print(f"Device: {device}")
    print(f"Seed: {config['seed']}\n")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    exp_info = get_experiment_info(config['source'], config['target'])

    source_train, source_test, target_train, target_test = get_dataloaders(
        source=config['source'],
        target=config['target'],
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        download=config.get('download', True),
        seed=config['seed']
    )

    logger = ExperimentLogger(
        config['results_dir'],
        config['experiment_name']
    )

    results = {}

    if config.get('train_dann', True):
        logger.log("Training DANN model...")

        dann_model = create_dann_model(
            architecture=exp_info['architecture'],
            num_classes=exp_info['num_classes'],
            pretrained=config.get('pretrained', False)
        )
        print_model_summary(dann_model, "DANN")

        dann_results = train_dann(
            dann_model,
            source_train, target_train,
            source_test, target_test,
            config, device, logger,
            architecture=exp_info['architecture']
        )
        results['dann'] = dann_results

        logger.log(f"DANN Results: Source={dann_results['source_acc']:.4f}, "
                  f"Target={dann_results['target_acc']:.4f}")

    if config.get('train_source_only', True):
        logger.log("Training Source-Only baseline...")

        source_logger = ExperimentLogger(
            config['results_dir'],
            f"{config['experiment_name']}_source_only"
        )

        source_model = create_source_only_model(
            architecture=exp_info['architecture'],
            num_classes=exp_info['num_classes'],
            pretrained=config.get('pretrained', False)
        )
        print_model_summary(source_model, "Source-Only")

        source_results = train_source_only(
            source_model,
            source_train, source_test, target_test,
            config, device, source_logger,
            architecture=exp_info['architecture'],
            model_name="source_only"
        )
        results['source_only'] = source_results

        logger.log(f"Source-Only Results: Source={source_results['source_acc']:.4f}, "
                  f"Target={source_results['target_acc']:.4f}")

    if config.get('train_target_only', False):
        logger.log("Training Target-Only (upper bound)...")

        target_logger = ExperimentLogger(
            config['results_dir'],
            f"{config['experiment_name']}_target_only"
        )

        target_model = create_source_only_model(
            architecture=exp_info['architecture'],
            num_classes=exp_info['num_classes'],
            pretrained=config.get('pretrained', False)
        )
        print_model_summary(target_model, "Target-Only (Upper Bound)")

        target_results = train_target_only(
            target_model,
            target_train, target_test,
            config, device, target_logger,
            architecture=exp_info['architecture']
        )
        results['target_only'] = target_results

        logger.log(f"Target-Only Results: Target={target_results['target_acc']:.4f}")

    results['config'] = config
    results_path = os.path.join(
        config['results_dir'],
        f"{config['experiment_name']}_results.json"
    )
    save_metrics_json(results, results_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train DANN for domain adaptation")
    
    # Data arguments
    parser.add_argument('--source', type=str, required=False,
                       help='Source domain (mnist, svhn, syn_numbers, etc.)')
    parser.add_argument('--target', type=str, required=False,
                       help='Target domain')
    parser.add_argument('--data_root', type=str, default='./datasets',
                       help='Root directory for datasets')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')

    parser.add_argument('--gamma', type=float, default=10.0,
                       help='Gamma for lambda schedule')

    parser.add_argument('--lr_alpha', type=float, default=10.0,
                       help='Alpha for LR schedule')
    parser.add_argument('--lr_beta', type=float, default=0.75,
                       help='Beta for LR schedule')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights (for Office)')
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for experiment')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory for results')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')

    parser.add_argument('--train_dann', action='store_true', dest='train_dann',
                       help='Train DANN model')
    parser.add_argument('--no_train_dann', action='store_false', dest='train_dann',
                       help='Skip DANN training')
    parser.set_defaults(train_dann=True)

    parser.add_argument('--train_source_only', action='store_true', dest='train_source_only',
                       help='Train source-only baseline')
    parser.add_argument('--no_train_source_only', action='store_false', dest='train_source_only',
                       help='Skip source-only training')
    parser.set_defaults(train_source_only=True)

    parser.add_argument('--train_target_only', action='store_true', dest='train_target_only',
                       help='Train target-only (upper bound)')
    parser.add_argument('--no_train_target_only', action='store_false', dest='train_target_only',
                       help='Skip target-only training')
    parser.set_defaults(train_target_only=False)
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip dataset download')
    
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        defaults = {action.dest: action.default for action in parser._actions}
        args_dict = vars(args)

        for key, value in args_dict.items():
            if key in ['config', 'skip_download']:
                continue
            if value is not None and value != defaults.get(key):
                config[key] = value
    else:
        config = vars(args)

    if config.get('experiment_name') is None:
        config['experiment_name'] = f"{config['source']}_to_{config['target']}"

    config['download'] = not args.skip_download

    print("\nTRAINING CONFIGURATION")
    print(f"Source: {config['source']} -> Target: {config['target']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Experiment name: {config['experiment_name']}\n")

    start_time = time.time()
    results = run_experiment(config)
    elapsed_time = time.time() - start_time

    print(f"\nExperiment completed in {format_time(elapsed_time)}")

    print("\nEXPERIMENT RESULTS SUMMARY")

    if 'source_only' in results:
        print(f"Source-Only (Baseline - Lower Bound):")
        print(f"  Source Accuracy: {results['source_only']['source_acc']:.4f}")
        print(f"  Target Accuracy: {results['source_only']['target_acc']:.4f}")
        print()

    if 'dann' in results:
        print(f"DANN (Proposed Method):")
        print(f"  Source Accuracy: {results['dann']['source_acc']:.4f}")
        print(f"  Target Accuracy: {results['dann']['target_acc']:.4f}")
        if 'source_only' in results:
            improvement = results['dann']['target_acc'] - results['source_only']['target_acc']
            print(f"  Improvement over Source-Only: {improvement:+.4f}")
        print()

    if 'target_only' in results:
        print(f"Target-Only (Upper Bound):")
        print(f"  Target Accuracy: {results['target_only']['target_acc']:.4f}")
        print()

    print(f"\nCheckpoints saved to: {config['checkpoint_dir']}/")
    print(f"Results saved to: {config['results_dir']}/")


if __name__ == '__main__':
    main()
