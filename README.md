# Domain-Adversarial Neural Networks (DANN) - Reproduction

This repository contains a reproduction of **"Unsupervised Domain Adaptation by Backpropagation"** by Ganin & Lempitsky (2015). The implementation includes the Domain-Adversarial Neural Network (DANN) for unsupervised domain adaptation across multiple benchmark datasets.

## Overview

DANN enables models to adapt from a labeled source domain to an unlabeled target domain by learning domain-invariant features through adversarial training. The gradient reversal layer allows simultaneous optimization of task accuracy and domain confusion.

## Supported Datasets

### Digit Recognition
- **MNIST → MNIST-M**: Grayscale digits to colored digits
- **SVHN → MNIST**: Street View House Numbers to handwritten digits
- **Synthetic Numbers → SVHN**: Generated synthetic digits to real street numbers

### Traffic Sign Recognition
- **Synthetic Signs → GTSRB**: Synthetic traffic signs to German Traffic Sign Recognition Benchmark

### Office-31 (Object Recognition)
- **Amazon → Webcam**: Product images across different domains
- **DSLR → Webcam**: Professional camera to webcam images
- **Webcam → DSLR**: Webcam to professional camera images

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd projet

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn pillow pyyaml tqdm
```

## Quick Start
in order for training to work you need to download the datasets they are not included here due to size limitations. 

## Dataset Preparation

### Automatic download:
MNIST, SVHN, GTSRB are automatically downloaded when you run training 
### Manual download required:
- **office 31**
- **BSDS500** (for MNIST-M generation)
- **Synthetic signs**

### Synthetic dataset generation:
```bash
# Generate synthetic numbers dataset
python syngen.py
# Output: datasets/syn_numbers/{train,test}.pkl
```

### 1. Downloading trained models 
all trained models are available via this link : 
### 2. Train DANN Model

#### Using configuration files (recommended):

```bash
# MNIST → MNIST-M
python train.py --config configs/mnist_mnistm.yaml

# SVHN → MNIST
python train.py --config configs/svhn_mnist.yaml

# Synthetic Numbers → SVHN
python train.py --config configs/syn_numbers_svhn.yaml

# Synthetic Signs → GTSRB
python train.py --config configs/syn_signs_gtsrb.yaml

# Office-31 experiments
python train.py --config configs/amazon_webcam.yaml
python train.py --config configs/dslr_webcam.yaml
python train.py --config configs/webcam_dslr.yaml
```

#### Using command-line arguments:

```bash
python train.py \
  --source mnist \
  --target mnistm \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.01 \
  --seed 42
```

#### Common training arguments:

```bash
python train.py \
  --source <source_domain> \
  --target <target_domain> \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.01 \
  --momentum 0.9 \
  --gamma 10.0 \
  --seed 42 \
  --pretrained  # Use pretrained weights (Office-31 only)
```

#### Training options:

```bash
# Train only DANN (skip baselines)
python train.py --config configs/svhn_mnist.yaml --no_train_source_only

# Train only source-only baseline
python train.py --config configs/svhn_mnist.yaml --no_train_dann

# Include target-only upper bound
python train.py --config configs/svhn_mnist.yaml --train_target_only
```

### 3. Evaluate Trained Models

#### Evaluate a single checkpoint:

```bash
python eval.py \
  --dann_checkpoint checkpoints/mnist_to_mnistm_dann_final.pt \
  --source mnist \
  --target mnistm \
  --tsne  # Generate t-SNE visualization
```

#### Evaluate source-only baseline:

```bash
python eval.py \
  --source_only_checkpoint checkpoints/mnist_to_mnistm_source_only_final.pt \
  --source mnist \
  --target mnistm
```

#### Evaluate without t-SNE (faster):

```bash
python eval.py \
  --dann_checkpoint checkpoints/svhn_to_mnist_dann_final.pt \
  --source svhn \
  --target mnist \
  --no_tsne
```

### 4. Multi-Seed Evaluation

For statistical significance (especially important for Office-31 experiments):

```bash
# First, train with multiple seeds
for seed in 0 1 2 3 4; do
  python train.py \
    --config configs/amazon_webcam.yaml \
    --seed $seed \
    --experiment_name amazon_to_webcam_seed${seed}
done

# Then evaluate all seeds and compute statistics
python evaluate_multiseed.py \
  --source amazon \
  --target webcam \
  --seeds 0 1 2 3 4
```

This will output mean ± std accuracy across all seeds.

### 5. Visualize Results

#### Plot training curves:

```bash
# Plot a specific experiment
python plot_results.py mnist_to_mnistm

# Plot all available experiments
python plot_results.py --all

# List available experiments
python plot_results.py --list

# Custom directories
python plot_results.py mnist_to_mnistm \
  --results-dir ./results \
  --save-dir ./plots
```

#### Visualize datasets:

```bash
# Show dataset info
python visualize_data.py --mode info

# Compare source vs target domains
python visualize_data.py --mode compare \
  --source mnist \
  --target mnistm

# Visualize from pickle file
python visualize_data.py --mode pickle \
  --pickle_path datasets/syn_numbers/train.pkl
```

## Directory Structure

```
projet/
├── train.py                    # Main training script
├── eval.py                     # Evaluation script
├── evaluate_multiseed.py       # Multi-seed evaluation
├── syngen.py                   # Synthetic dataset generation
├── model.py                    # DANN model architecture
├── data.py                     # Dataset handling
├── utils.py                    # Utility functions
├── plot_results.py             # Plotting script
├── visualize_data.py           # Dataset visualization
├── configs/                    # Configuration files
│   ├── mnist_mnistm.yaml
│   ├── svhn_mnist.yaml
│   ├── syn_numbers_svhn.yaml
│   ├── syn_signs_gtsrb.yaml
│   ├── amazon_webcam.yaml
│   ├── dslr_webcam.yaml
│   └── webcam_dslr.yaml
├── datasets/                   # Datasets (some are auto downloaded some are )
├── checkpoints/                # Model checkpoints
├── results/                    # Training logs and metrics
└── plots/                      # Generated plots
```

## Output Files

### Training outputs:
- **Checkpoints**: `checkpoints/<experiment>_dann_final.pt`
- **Training logs**: `results/<experiment>_log.txt`
- **Training history**: `results/<experiment>_history.json`
- **Results summary**: `results/<experiment>_results.json`

### Evaluation outputs:
- **Evaluation metrics**: `results/<experiment>_evaluation.json`
- **t-SNE visualizations**: `results/<experiment>_tsne.png`

### Plotting outputs:
- **Comparison plots**: `plots/<experiment>_comparison.png`

## Configuration Files

Configuration files (YAML) contain all hyperparameters:

```yaml
source: svhn
target: mnist
data_root: ./datasets
epochs: 100
batch_size: 128
lr: 0.005
momentum: 0.9
gamma: 5.0
pretrained: false
experiment_name: svhn_to_mnist
checkpoint_dir: ./checkpoints
results_dir: ./results
num_workers: 4
train_dann: true
train_source_only: true
seed: 42
```

## Key Hyperparameters

- `--lr`: Learning rate (0.01 for most, 0.005 for SVHN)
- `--gamma`: Controls adaptation speed (λ schedule), default 10.0
- `--lr_alpha`: Learning rate decay parameter, default 10.0
- `--lr_beta`: Learning rate decay parameter, default 0.75
- `--momentum`: SGD momentum, default 0.9
- `--batch_size`: Batch size, default 128
- `--epochs`: Number of training epochs, default 100
- `--seed`: Random seed for reproducibility

