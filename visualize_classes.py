"""
Visualize original images from source and target domains with matching classes.

Shows N classes side-by-side with examples from both domains.

Usage:
    python visualize_classes.py --source mnist --target mnistm
    python visualize_classes.py --source svhn --target mnist
    python visualize_classes.py --source syn_signs --target gtsrb
    python visualize_classes.py --source amazon --target webcam
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from data import _create_dataset, _get_experiment_config
from utils import set_seed


def get_raw_images_by_class(dataset, num_classes, samples_per_class=2):
    """
    Extract raw images organized by class without transforms.
    
    Handles different dataset structures:
    - MNIST wrapper (has self.mnist)
    - SVHN wrapper (has self.svhn)  
    - GTSRB (has self.data as file paths)
    - MNISTM, SyntheticNumbers, etc (has self.data as numpy arrays)

    Args:
        dataset: Dataset object
        num_classes: Total number of classes
        samples_per_class: How many examples per class

    Returns:
        dict: {class_id: [PIL Images]}
    """
    class_images = {i: [] for i in range(num_classes)}
    
    # Determine dataset type and extract data/targets appropriately
    data = None
    targets = None
    
    # Case 1: Custom MNIST wrapper (has self.mnist attribute)
    if hasattr(dataset, 'mnist'):
        data = dataset.mnist.data.numpy() if hasattr(dataset.mnist.data, 'numpy') else dataset.mnist.data
        targets = dataset.mnist.targets.numpy() if hasattr(dataset.mnist.targets, 'numpy') else dataset.mnist.targets
        targets = list(targets)
        
    # Case 2: Custom SVHN wrapper (has self.svhn attribute)
    elif hasattr(dataset, 'svhn'):
        # SVHN data is (N, 3, 32, 32), need to transpose to (N, 32, 32, 3)
        data = dataset.svhn.data
        if len(data.shape) == 4 and data.shape[1] == 3:
            data = np.transpose(data, (0, 2, 3, 1))
        targets = list(dataset.svhn.labels)
        
    # Case 3: GTSRB with torchvision (has self.dataset attribute with _samples)
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, '_samples'):
        # GTSRB stores file paths, we need to load images
        samples = dataset.dataset._samples
        for idx in range(len(samples)):
            img_path, target = samples[idx]
            
            if len(class_images[target]) < samples_per_class:
                img = Image.open(img_path).convert('RGB')
                class_images[target].append(img)
            
            # Check if we have enough
            if all(len(imgs) >= samples_per_class for imgs in class_images.values()):
                break
        return class_images
    
    # Case 4: Direct data/targets attributes (MNISTM, SyntheticNumbers, SyntheticSigns, Office)
    elif hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
        data = dataset.data
        targets = dataset.targets
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()
        targets = list(targets)
    
    else:
        print(f"Warning: Unknown dataset structure for {type(dataset).__name__}")
        return class_images
    
    # Now extract images
    for idx in range(len(data)):
        img_data = data[idx]
        target = targets[idx]
        
        # Convert target to int if needed
        if hasattr(target, 'item'):
            target = target.item()
        
        # Skip if we have enough for this class
        if len(class_images[target]) >= samples_per_class:
            continue
        
        # Convert to PIL Image
        if isinstance(img_data, str):
            # It's a file path (Office dataset)
            img = Image.open(img_data).convert('RGB')
        elif isinstance(img_data, np.ndarray):
            # Handle grayscale (28, 28) -> convert to RGB for display
            if len(img_data.shape) == 2:
                img = Image.fromarray(img_data, mode='L')
            else:
                img = Image.fromarray(img_data)
        elif hasattr(img_data, 'convert'):
            # Already PIL Image
            img = img_data
        else:
            continue
        
        class_images[target].append(img)
        
        # Check if we have enough samples
        if all(len(imgs) >= samples_per_class for imgs in class_images.values()):
            break

    return class_images


def visualize_classes(source, target, num_classes=3, samples_per_class=2, save_dir='results'):
    """
    Create visualization showing selected classes from both domains.

    Args:
        source: Source domain name
        target: Target domain name
        num_classes: Number of classes to display (default: 3)
        samples_per_class: Number of examples per class (default: 2)
        save_dir: Directory to save visualization
    """
    set_seed(42)

    print(f"\nCreating class visualization: {source.upper()} → {target.upper()}")

    # Get experiment config
    config = _get_experiment_config(source, target)
    total_num_classes = config['num_classes']

    # Load datasets WITHOUT transforms to get raw images
    print(f"Loading {source} dataset...")
    source_dataset = _create_dataset(
        source,
        root='./datasets',
        train=True,
        transform=None,  # No transform - raw images
        download=True
    )

    print(f"Loading {target} dataset...")
    target_dataset = _create_dataset(
        target,
        root='./datasets',
        train=True,
        transform=None,  # No transform - raw images
        download=True
    )

    # Extract images by class
    print(f"Extracting {samples_per_class} samples per class from {source}...")
    source_images_by_class = get_raw_images_by_class(
        source_dataset, total_num_classes, samples_per_class
    )

    print(f"Extracting {samples_per_class} samples per class from {target}...")
    target_images_by_class = get_raw_images_by_class(
        target_dataset, total_num_classes, samples_per_class
    )

    src_count = sum(len(imgs) for imgs in source_images_by_class.values())
    tgt_count = sum(len(imgs) for imgs in target_images_by_class.values())
    print(f"Extracted {src_count} source images, {tgt_count} target images")

    # Select which classes to display (evenly spaced)
    if num_classes >= total_num_classes:
        classes_to_display = list(range(total_num_classes))
    else:
        step = total_num_classes / num_classes
        classes_to_display = [int(i * step) for i in range(num_classes)]

    print(f"Displaying {len(classes_to_display)} classes: {classes_to_display}")

    # Create visualization
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(4 * samples_per_class, len(classes_to_display) * 1.5))

    gs = gridspec.GridSpec(
        len(classes_to_display), 2,
        figure=fig,
        wspace=0.3,
        hspace=0.4,
        left=0.15,
        right=0.95,
        top=0.92,
        bottom=0.05
    )

    # Domain names for headers
    source_display = source.replace('_', ' ').title()
    target_display = target.replace('_', ' ').title()

    for row_idx, class_id in enumerate(classes_to_display):
        # Create sub-grids for source and target
        gs_source = gridspec.GridSpecFromSubplotSpec(
            1, samples_per_class, subplot_spec=gs[row_idx, 0],
            wspace=0.05
        )
        gs_target = gridspec.GridSpecFromSubplotSpec(
            1, samples_per_class, subplot_spec=gs[row_idx, 1],
            wspace=0.05
        )

        # Plot source images
        for i in range(min(samples_per_class, len(source_images_by_class[class_id]))):
            ax = fig.add_subplot(gs_source[i])
            img = source_images_by_class[class_id][i]
            
            # Convert to RGB if grayscale for consistent display
            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.axis('off')

            # Add class label on first image
            if i == 0:
                ax.text(
                    -0.15, 0.5, f'Class {class_id}',
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight='bold',
                    va='center',
                    ha='right'
                )

        # Plot target images
        for i in range(min(samples_per_class, len(target_images_by_class[class_id]))):
            ax = fig.add_subplot(gs_target[i])
            img = target_images_by_class[class_id][i]
            
            # Convert to RGB if grayscale for consistent display
            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.axis('off')

    # Add column headers
    fig.text(0.33, 0.96, f'{source_display} (Source)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.73, 0.96, f'{target_display} (Target)', ha='center', fontsize=12, fontweight='bold')

    # Save figure
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    output_file = save_path / f'{source}_to_{target}_classes.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")

    plt.close()

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize original images from source and target domains'
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Source domain (mnist, svhn, syn_signs, amazon, dslr, webcam)')
    parser.add_argument('--target', type=str, required=True,
                       help='Target domain (mnistm, mnist, gtsrb, webcam, dslr)')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of classes to display (default: 3)')
    parser.add_argument('--samples_per_class', type=int, default=2,
                       help='Number of samples per class (default: 2)')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save visualizations')

    args = parser.parse_args()

    visualize_classes(
        args.source,
        args.target,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()