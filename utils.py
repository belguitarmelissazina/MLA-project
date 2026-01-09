"""
Utilities for DANN Experiments

Random seed setting, metrics computation, plotting functions,
checkpoint saving/loading, and lambda scheduling.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score
import csv
from datetime import datetime
from pathlib import Path


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms where possible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set CUBLAS workspace config for CUDA determinism
        if torch.cuda.is_available():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                # Use warn_only=True to allow non-deterministic operations
                # (like adaptive_avg_pool2d in AlexNet) to run with a warning
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (RuntimeError, TypeError):
                # TypeError for older PyTorch that doesn't support warn_only
                # RuntimeError for operations without deterministic implementation
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError:
                    pass


def get_device() -> torch.device:
    """Get the best available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_lambda(p: float, gamma: float = 10.0) -> float:
    """
    Compute lambda (adaptation factor) using the schedule formula.

    λ_p = 2 / (1 + exp(-γ * p)) - 1

    Args:
        p: Training progress (0 to 1)
        gamma: Schedule parameter
        
    Returns:
        Lambda value (ranges from 0 to 1 as p goes from 0 to 1)
    """
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


def compute_lr(
    p: float,
    mu_0: float = 0.01,
    alpha: float = 10.0,
    beta: float = 0.75
) -> float:
    """
    Compute learning rate using the schedule formula.

    μ_p = μ_0 / (1 + α * p)^β

    Args:
        p: Training progress (0 to 1)
        mu_0: Initial learning rate
        alpha: Schedule parameter
        beta: Schedule parameter
        
    Returns:
        Learning rate at progress p
    """
    return mu_0 / ((1.0 + alpha * p) ** beta)


class LambdaScheduler:
    """Scheduler for lambda (adaptation factor) during training."""
    
    def __init__(
        self,
        total_steps: int,
        gamma: float = 10.0,
        start_lambda: float = 0.0
    ):
        """
        Args:
            total_steps: Total number of training steps
            gamma: Gamma parameter for lambda schedule
            start_lambda: Starting lambda value (0 by default)
        """
        self.total_steps = total_steps
        self.gamma = gamma
        self.current_step = 0
        self.start_lambda = start_lambda
    
    def step(self) -> float:
        """Update and return current lambda value."""
        self.current_step += 1
        p = min(1.0, self.current_step / self.total_steps)
        return compute_lambda(p, self.gamma)
    
    def get_lambda(self) -> float:
        """Get current lambda without incrementing step."""
        p = min(1.0, self.current_step / self.total_steps)
        return compute_lambda(p, self.gamma)
    
    def reset(self) -> None:
        """Reset scheduler."""
        self.current_step = 0


class LRScheduler:
    """Learning rate scheduler."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        mu_0: float = 0.01,
        alpha: float = 10.0,
        beta: float = 0.75
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            mu_0: Initial learning rate
            alpha: Schedule parameter
            beta: Schedule parameter
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.mu_0 = mu_0
        self.alpha = alpha
        self.beta = beta
        self.current_step = 0
    
    def step(self) -> float:
        """Update learning rate and return new value."""
        self.current_step += 1
        p = min(1.0, self.current_step / self.total_steps)
        new_lr = compute_lr(p, self.mu_0, self.alpha, self.beta)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        p = min(1.0, self.current_step / self.total_steps)
        return compute_lr(p, self.mu_0, self.alpha, self.beta)


def compute_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """
    Compute classification accuracy on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to use
        
    Returns:
        Accuracy as float (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model.predict(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def compute_domain_accuracy(
    model: nn.Module,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 1.0
) -> float:
    """
    Compute domain classification accuracy.
    
    Args:
        model: DANN model
        source_loader: Source domain dataloader
        target_loader: Target domain dataloader
        device: Device to use
        alpha: Lambda value for GRL
        
    Returns:
        Domain accuracy as float (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Source domain (label = 0)
        for images, _ in source_loader:
            images = images.to(device)
            _, domain_logits, _ = model(images, alpha=alpha)
            predictions = (torch.sigmoid(domain_logits) < 0.5).float()
            correct += predictions.sum().item()
            total += images.size(0)
        
        # Target domain (label = 1)
        for images, _ in target_loader:
            images = images.to(device)
            _, domain_logits, _ = model(images, alpha=alpha)
            predictions = (torch.sigmoid(domain_logits) >= 0.5).float()
            correct += predictions.sum().item()
            total += images.size(0)
    
    return correct / total if total > 0 else 0.5


def get_confusion_matrix(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix for a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to use
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model.predict(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))


def extract_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a model for visualization.
    
    Args:
        model: Model with get_features method
        dataloader: DataLoader for feature extraction
        device: Device to use
        max_samples: Maximum number of samples to extract
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(all_labels) >= max_samples:
                break
            
            images = images.to(device)
            features = model.get_features(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels = np.array(all_labels)[:max_samples]
    
    return features, labels


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
) -> plt.Figure:
    """
    Plot training curves (losses, accuracies, lambda).
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Classification Loss
    if 'class_loss' in history:
        axes[0, 0].plot(history['class_loss'], label='Classification Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Classification Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Domain Loss
    if 'domain_loss' in history:
        axes[0, 1].plot(history['domain_loss'], label='Domain Loss', color='orange')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Domain Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracies
    if 'source_acc' in history:
        axes[1, 0].plot(history['source_acc'], label='Source Accuracy')
    if 'target_acc' in history:
        axes[1, 0].plot(history['target_acc'], label='Target Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Classification Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Lambda Schedule
    if 'lambda' in history:
        axes[1, 1].plot(history['lambda'], label='Lambda (α)', color='green')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Lambda')
    axes[1, 1].set_title('Adaptation Factor Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_tsne(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "t-SNE Visualization",
    perplexity: int = 30,
    n_iter: int = 1000
) -> plt.Figure:
    """
    Create t-SNE visualization of source and target features.

    Args:
        source_features: Features from source domain
        source_labels: Labels from source domain
        target_features: Features from target domain
        target_labels: Labels from target domain
        save_path: Path to save the figure
        title: Plot title
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        
    Returns:
        Matplotlib figure
    """
    # Combine features
    all_features = np.concatenate([source_features, target_features], axis=0)
    
    # Create domain labels (0=source, 1=target)
    domain_labels = np.concatenate([
        np.zeros(len(source_features)),
        np.ones(len(target_features))
    ])
    
    # Run t-SNE (silently)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        verbose=0  # Suppress t-SNE output
    )
    embeddings = tsne.fit_transform(all_features)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Color by domain
    source_mask = domain_labels == 0
    target_mask = domain_labels == 1
    
    axes[0].scatter(
        embeddings[source_mask, 0], embeddings[source_mask, 1],
        c='blue', alpha=0.5, s=10, label='Source'
    )
    axes[0].scatter(
        embeddings[target_mask, 0], embeddings[target_mask, 1],
        c='red', alpha=0.5, s=10, label='Target'
    )
    axes[0].set_title('Domain Visualization')
    axes[0].legend()
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Plot 2: Color by class
    all_labels = np.concatenate([source_labels, target_labels])
    scatter = axes[1].scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=all_labels, cmap='tab10', alpha=0.5, s=10
    )
    axes[1].set_title('Class Visualization')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=axes[1], label='Class')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names for each class
        save_path: Path to save the figure
        title: Plot title
        normalize: Whether to normalize the matrix
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names:
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_lambda_schedule(
    num_steps: int = 10000,
    gamma: float = 10.0,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize the lambda schedule.

    Args:
        num_steps: Total number of training steps
        gamma: Gamma parameter
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    steps = np.arange(num_steps)
    p = steps / num_steps
    lambdas = [compute_lambda(pi, gamma) for pi in p]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, lambdas, 'b-', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('λ (Lambda)')
    ax.set_title(f'Lambda Schedule (γ = {gamma})')
    ax.grid(True, alpha=0.3)
    
    # Add formula annotation
    ax.text(
        0.95, 0.05,
        r'$\lambda_p = \frac{2}{1 + e^{-\gamma \cdot p}} - 1$',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    config: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Current metrics
        save_path: Path to save checkpoint
        config: Optional configuration dict
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    torch.save(checkpoint, save_path)
    # Silent save (no print)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to restore
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Try to load state dict, handling source-only models loaded into DANN models
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        if 'domain_classifier' in str(e):
            # This is a source-only model being loaded into a DANN model
            # Load only the feature extractor and classifier parts
            state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()

            # Filter out domain_classifier keys
            filtered_dict = {k: v for k, v in state_dict.items()
                           if not k.startswith('domain_classifier')}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            # Silent load (no print)
        else:
            raise e

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Silent load (no print)
    return checkpoint


def save_metrics_csv(
    metrics: Dict[str, Any],
    csv_path: str,
    append: bool = True
) -> None:
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        csv_path: Path to CSV file
        append: Whether to append or overwrite
    """
    file_exists = os.path.exists(csv_path)
    mode = 'a' if append else 'w'
    
    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        
        if not file_exists or not append:
            writer.writeheader()
        
        writer.writerow(metrics)


def save_metrics_json(
    metrics: Dict[str, Any],
    json_path: str
) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        json_path: Path to JSON file
    """
    # Handle numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
    
    converted = {k: convert(v) for k, v in metrics.items()}
    
    with open(json_path, 'w') as f:
        json.dump(converted, f, indent=2)


def create_results_table(
    results: List[Dict],
    save_path: Optional[str] = None
) -> str:
    """
    Create results table.

    Args:
        results: List of result dictionaries
        save_path: Optional path to save table
        
    Returns:
        Formatted table string
    """
    # Header
    header = "| Method | Source | Target | Accuracy |\n"
    header += "|--------|--------|--------|----------|\n"
    
    rows = []
    for r in results:
        row = f"| {r.get('method', 'N/A')} | {r.get('source', 'N/A')} | "
        row += f"{r.get('target', 'N/A')} | {r.get('accuracy', 0):.4f} ± {r.get('std', 0):.4f} |"
        rows.append(row)
    
    table = header + "\n".join(rows)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)
    
    return table


class ExperimentLogger:
    """Logger for tracking experiment progress."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str
    ):
        """
        Args:
            log_dir: Directory for logs
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'class_loss': [],
            'domain_loss': [],
            'source_acc': [],
            'target_acc': [],
            'domain_acc': [],
            'lambda': [],
            'lr': []
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
    
    def log(self, message: str, print_msg: bool = True) -> None:
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        if print_msg:
            print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + '\n')
    
    def log_step(
        self,
        step: int,
        class_loss: float,
        domain_loss: float,
        lambda_val: float,
        lr: float
    ) -> None:
        """Log training step metrics."""
        self.history['class_loss'].append(class_loss)
        self.history['domain_loss'].append(domain_loss)
        self.history['lambda'].append(lambda_val)
        self.history['lr'].append(lr)
    
    def log_epoch(
        self,
        epoch: int,
        source_acc: float,
        target_acc: float,
        domain_acc: Optional[float] = None
    ) -> None:
        """Log epoch metrics."""
        self.history['source_acc'].append(source_acc)
        self.history['target_acc'].append(target_acc)
        if domain_acc is not None:
            self.history['domain_acc'].append(domain_acc)
        
        msg = f"Epoch {epoch}: Source={source_acc:.4f}, Target={target_acc:.4f}"
        if domain_acc is not None:
            msg += f", Domain={domain_acc:.4f}"
        self.log(msg)
    
    def save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.log_dir / f"{self.experiment_name}_history.json"
        save_metrics_json(self.history, str(history_path))
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_summary(model: nn.Module, name: str = "Model") -> None:
    """Print model summary."""
    print(f"\n{name} Summary")
    print(f"Total parameters: {count_parameters(model):,}\n")
