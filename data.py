"""
Dataset Handling for unsupervised domain adaptation by backpropagation 
Authors: Ganin & Lempitsky, 2015 Reproduction

This module provides dataset classes and data loaders for various domain adaptation experiments.:
- MNIST, MNIST-M (generated), SVHN datasets , Synthetic Numbers (generated), Synthetic Signs (generated)
- GTSRB (German Traffic Sign Recognition Benchmark)
- Office dataset (Amazon, DSLR, Webcam)
- data preprocessing
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
import pickle
import struct
import gzip
from typing import Tuple, Optional, List, Callable
import urllib.request
import tarfile
import zipfile
from pathlib import Path


class DADataset(Dataset):
    """
    Base class for domain adaptation datasets.
    Provides consistent interface for source/target domains.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, target = self.data[idx], self.targets[idx]
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class MNIST(Dataset):
    """
    Standard MNIST dataset wrapper.
    Images are converted to 3-channel RGB for consistency.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = root
        self.transform = transform
        self.train = train

        # Use torchvision MNIST 
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download
        )
    
    def __len__(self) -> int:
        return len(self.mnist)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, target = self.mnist[idx]
        
        # Convert to RGB (3 channels)
        img = img.convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class MNISTM(Dataset):
    """
    MNIST-M Dataset: MNIST digits blended over color patches.

    Digits from the original MNIST are blended over patches randomly extracted
    from color photos from BSDS500 using absolute difference blending.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
        
        # Check if preprocessed MNIST-M exists
        mnistm_dir = self.root / 'mnist_m'
        pkl_path = mnistm_dir / ('train.pkl' if train else 'test.pkl')
        
        if pkl_path.exists():
            self._load_preprocessed(pkl_path)
        elif download:
            self._generate_mnistm()
        else:
            raise RuntimeError(
                f"MNIST-M not found at {pkl_path}. "
                "Set download=True to generate it."
            )
    
    def _load_preprocessed(self, path: Path) -> None:
        """Load preprocessed MNIST-M from pickle file."""
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict['data']
        self.targets = data_dict['targets']
    
    def _generate_mnistm(self) -> None:
        """Generate MNIST-M by blending MNIST with BSDS500 patches."""
        print("Generating MNIST-M dataset...")

        mnistm_dir = self.root / 'mnist_m'
        mnistm_dir.mkdir(parents=True, exist_ok=True)

        # Download and load MNIST 
        mnist = datasets.MNIST(
            root=str(self.root),
            train=self.train,
            download=True
        )
        
        # Download BSDS500 
        bsds_patches = self._get_background_patches()
        
        # Generate blended images
        data = []
        targets = []
        
        for idx in range(len(mnist)):
            img, target = mnist[idx]
            img = np.array(img)
            
            # Select random background patch
            bg_idx = np.random.randint(len(bsds_patches))
            background = bsds_patches[bg_idx]
            
            # Resize background to 28x28 if needed
            if background.shape[:2] != (28, 28):
                background = np.array(
                    Image.fromarray(background).resize((28, 28))
                )
            
            # Create 3-channel digit image
            digit_rgb = np.stack([img, img, img], axis=-1)

            blended = np.abs(digit_rgb.astype(np.int32) - background.astype(np.int32))
            blended = blended.astype(np.uint8)
            
            data.append(blended)
            targets.append(target)
        
        self.data = np.array(data)
        self.targets = targets
        
        # Save preprocessed data
        pkl_path = mnistm_dir / ('train.pkl' if self.train else 'test.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({'data': self.data, 'targets': self.targets}, f)
        
        print(f"MNIST-M saved to {pkl_path}")
    
    def _get_background_patches(self, num_patches: int = 10000) -> np.ndarray:
        """Get background patches from BSDS500 or generate colored patches."""
        bsds_dir = self.root / 'BSR' / 'BSDS500' / 'data' / 'images'
        
        if bsds_dir.exists():
            return self._extract_bsds_patches(bsds_dir, num_patches)
        else:
            print("BSDS500 not found, generating random colored backgrounds...")
            return self._generate_colored_backgrounds(num_patches)
    
    def _extract_bsds_patches(
        self,
        bsds_dir: Path,
        num_patches: int
    ) -> np.ndarray:
        """Extract random 28x28 patches from BSDS500 images."""
        patches = []
        
        # Get all image files
        image_files = []
        for split in ['train', 'val', 'test']:
            split_dir = bsds_dir / split
            if split_dir.exists():
                image_files.extend(list(split_dir.glob('*.jpg')))
        
        if not image_files:
            return self._generate_colored_backgrounds(num_patches)
        
        patches_per_image = num_patches // len(image_files) + 1
        
        for img_path in image_files:
            img = np.array(Image.open(img_path))
            h, w = img.shape[:2]
            
            for _ in range(patches_per_image):
                if h > 28 and w > 28:
                    y = np.random.randint(0, h - 28)
                    x = np.random.randint(0, w - 28)
                    patch = img[y:y+28, x:x+28]
                    patches.append(patch)
                    
                    if len(patches) >= num_patches:
                        break
            
            if len(patches) >= num_patches:
                break
        
        return np.array(patches[:num_patches])
    
    def _generate_colored_backgrounds(self, num_patches: int) -> np.ndarray:
        """Generate random colored background patches."""
        patches = []
        
        for _ in range(num_patches):
            # Generate random gradient or noise patterns
            if np.random.random() > 0.5:
                # Gradient background
                color1 = np.random.randint(0, 256, 3)
                color2 = np.random.randint(0, 256, 3)
                
                gradient = np.linspace(0, 1, 28).reshape(-1, 1, 1)
                patch = (color1 * (1 - gradient) + color2 * gradient).astype(np.uint8)
                patch = np.broadcast_to(patch, (28, 28, 3)).copy()
            else:
                # Random color noise
                patch = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
            
            patches.append(patch)
        
        return np.array(patches)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx]
        target = self.targets[idx]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class SVHN(Dataset):
    """
    Street View House Numbers (SVHN) Dataset.
    Uses torchvision's SVHN with consistent interface.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = root
        self.transform = transform

        # Save in SVHN subfolder for organization
        svhn_path = os.path.join(root, 'SVHN')
        split = 'train' if train else 'test'
        self.svhn = datasets.SVHN(
            root=svhn_path,
            split=split,
            download=download
        )
    
    def __len__(self) -> int:
        return len(self.svhn)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, target = self.svhn[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class SyntheticNumbers(Dataset):
    """
    Synthetic Numbers Dataset.

    500,000 images generated from Windows fonts by varying the text
    (one-, two-, and three-digit numbers), positioning, orientation,
    background and stroke colors, and the amount of blur.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        num_samples: int = 500000
    ):
        self.root = Path(root)
        self.transform = transform
        self.train = train
        self.num_samples = num_samples if train else 50000
        
        syn_dir = self.root / 'syn_numbers'
        pkl_path = syn_dir / ('train.pkl' if train else 'test.pkl')
        
        if pkl_path.exists():
            self._load_preprocessed(pkl_path)
        elif download:
            self._generate_synthetic_numbers()
        else:
            raise RuntimeError(
                f"Synthetic Numbers not found at {pkl_path}. "
                "Set download=True to generate it."
            )
    
    def _load_preprocessed(self, path: Path) -> None:
        """Load preprocessed data from pickle file."""
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict['data']
        self.targets = data_dict['targets']
    
    def _generate_synthetic_numbers(self) -> None:
        """Generate synthetic number images."""
        print(f"Generating Synthetic Numbers dataset ({self.num_samples} samples)...")
        
        syn_dir = self.root / 'syn_numbers'
        syn_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from PIL import ImageDraw, ImageFont, ImageFilter
        except ImportError:
            raise RuntimeError("PIL is required for generating synthetic numbers")
        
        data = []
        targets = []
        
        # Use default font if system fonts not available
        try:
            # Try to load a monospace font
            fonts = [
                ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20),
                ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 20),
            ]
        except:
            fonts = [ImageFont.load_default()]
        
        for i in range(self.num_samples):
            # Generate random digit (0-9 for 10 classes, matching SVHN)
            digit = np.random.randint(0, 10)
            
            # Random colors
            bg_color = tuple(np.random.randint(0, 256, 3))
            text_color = tuple(np.random.randint(0, 256, 3))
            
            # Create image
            img = Image.new('RGB', (32, 32), bg_color)
            draw = ImageDraw.Draw(img)
            
            # Random font
            font = np.random.choice(fonts)
            
            # Draw digit with some positioning variation
            x = np.random.randint(5, 15)
            y = np.random.randint(2, 10)
            draw.text((x, y), str(digit), font=font, fill=text_color)
            
            # Random blur
            if np.random.random() > 0.5:
                blur_radius = np.random.uniform(0, 1.5)
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            
            # Random rotation
            if np.random.random() > 0.7:
                angle = np.random.uniform(-15, 15)
                img = img.rotate(angle, fillcolor=bg_color)
            
            data.append(np.array(img))
            targets.append(digit)
            
            if (i + 1) % 50000 == 0:
                print(f"  Generated {i + 1}/{self.num_samples} samples")
        
        self.data = np.array(data)
        self.targets = targets
        
        # Save preprocessed data
        pkl_path = syn_dir / ('train.pkl' if self.train else 'test.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({'data': self.data, 'targets': self.targets}, f)
        
        print(f"Synthetic Numbers saved to {pkl_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx]
        target = self.targets[idx]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class SyntheticSigns(Dataset):
    """
    Synthetic Traffic Signs Dataset (SynSigns from SynsetSignsetGermany).

    Loads realistic rendered traffic signs from the Cycles folder:
    datasets/SynsetSignsetGermany/Cycles/

    The Cycles folder contains high-quality 3D rendered images (208x208 RGB)
    with realistic lighting and textures, NOT semantic segmentation masks.
    Images are named with suffix "_cycles.png".

    Args:
        root: Root directory containing datasets/
        train: If True, returns training split (80%), else test split (20%)
        transform: Image transformations to apply
        download: Not used (dataset must be pre-downloaded)
        use_augmented: Not currently used (kept for backwards compatibility)
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        use_augmented: bool = False
    ):
        self.root = Path(root)
        self.transform = transform
        self.train = train
        self.use_augmented = use_augmented
        self.num_classes = 43

        self.data = []
        self.targets = []

        # Use Cycles folder from SynsetSignsetGermany (realistic rendered images)
        synsigns_base = self.root / 'SynsetSignsetGermany'
        synsigns_dir = synsigns_base / 'Cycles'

        if not synsigns_dir.exists():
            error_msg = f"SynSigns Cycles folder not found at {synsigns_dir}.\n"
            error_msg += "Please ensure the SynsetSignsetGermany dataset is extracted to datasets/\n"
            error_msg += "Expected structure: datasets/SynsetSignsetGermany/Cycles/\n"
            error_msg += "The Cycles folder should contain realistic rendered traffic sign images."
            raise RuntimeError(error_msg)

        self._load_data(synsigns_dir)

        # Split into train/test (80/20 split)
        total_samples = len(self.data)
        split_idx = int(0.8 * total_samples)

        if train:
            self.data = self.data[:split_idx]
            self.targets = self.targets[:split_idx]
        else:
            self.data = self.data[split_idx:]
            self.targets = self.targets[split_idx:]

        print(f"SynSigns Cycles {'train' if train else 'test'}: {len(self.data)} images")

    def _load_data(self, data_dir: Path) -> None:
        """Load SynSigns images from class folders."""
        import random

        # Get all class folders (0_Geschwindigkeit20, 1_Geschwindigkeit30, etc.)
        class_folders = sorted([f for f in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, f))])

        all_samples = []

        for class_folder in class_folders:
            # Extract class number from folder name (e.g., "0_Geschwindigkeit20" -> 0)
            class_num = int(class_folder.split('_')[0])

            # Only process classes 0-42 (43 classes total)
            if class_num >= self.num_classes:
                continue

            class_path = os.path.join(data_dir, class_folder)

            # Get all Cycles images (*_cycles.png) in this class
            for img_file in os.listdir(class_path):
                if img_file.endswith('_cycles.png'):
                    img_path = os.path.join(class_path, img_file)
                    all_samples.append((img_path, class_num))

        # Shuffle samples to mix classes (deterministic with seed)
        random.Random(42).shuffle(all_samples)

        # Separate into data and targets
        self.data = [sample[0] for sample in all_samples]
        self.targets = [sample[1] for sample in all_samples]

        print(f"Loaded {len(self.data)} SynSigns images from {data_dir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data[idx]
        target = self.targets[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class GTSRB(Dataset):
    """
    German Traffic Sign Recognition Benchmark (GTSRB).
    43 classes of traffic signs.

    Uses torchvision's built-in GTSRB dataset for automatic downloading.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        from torchvision.datasets import GTSRB as TorchvisionGTSRB

        # Use torchvision's GTSRB dataset
        # Note: torchvision only provides the training split (test set has no labels)
        # We'll split the training set in get_dataloaders()
        self.dataset = TorchvisionGTSRB(
            root=root,
            split='train',  # Always use train split because it has labels
            transform=transform,
            download=download
        )

        # Extract data and targets for compatibility
        self.data = self.dataset._samples  # Access internal samples list
        self.targets = [s[1] for s in self.dataset._samples]  # Extract labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class Office(Dataset):
    """
    Office Dataset (Amazon, DSLR, Webcam).
    31 object categories.

    This version supports explicit subsetting via subset_paths,
    which we use to create proper train/test splits (no leakage).
    """

    def __init__(
        self,
        root: str,
        domain: str = 'amazon',
        transform: Optional[Callable] = None,
        download: bool = False,
        samples_per_class: Optional[int] = None,
        seed: int = 42,
        subset_paths: Optional[List[str]] = None,   # NEW
    ):
        self.root = Path(root)
        self.domain = domain.lower()
        self.transform = transform
        self.samples_per_class = samples_per_class
        self.seed = seed

        self.data: List[str] = []
        self.targets: List[int] = []
        self.classes: List[str] = []

        office_dir = self.root / 'office31' / self.domain / 'images'

        if not office_dir.exists():
            if download:
                print("Office dataset requires manual download.")
                print("Please download from: https://faculty.cc.gatech.edu/~judy/domainadapt/")
                print(f"Extract to: {self.root}/office31/")
            raise RuntimeError(
                f"Office dataset not found at {office_dir}. "
                "Please download from https://faculty.cc.gatech.edu/~judy/domainadapt/ "
                f"and extract to {self.root}/office31/"
            )

        # If subset_paths is provided, we build ONLY from those paths
        if subset_paths is not None:
            self._load_from_subset(office_dir, subset_paths)
        else:
            self._load_data(office_dir)

    def _scan_all(self, data_dir: Path) -> Tuple[List[str], List[int], List[str]]:
        """
        Scan all images and return (paths, targets, classes).
        Deterministic class ordering.
        """
        classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        all_paths: List[str] = []
        all_targets: List[int] = []

        for class_name in classes:
            class_dir = data_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    all_paths.append(str(img_path))
                    all_targets.append(class_to_idx[class_name])

        return all_paths, all_targets, classes

    def _load_from_subset(self, data_dir: Path, subset_paths: List[str]) -> None:
        """
        Load a dataset consisting ONLY of subset_paths.
        Targets are inferred from folder name.
        """
        # Build class mapping from directory structure
        self.classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        loaded_count = 0
        for p in subset_paths:
            if not os.path.exists(p):
                continue
            # folder name is the class
            class_name = Path(p).parent.name
            if class_name not in class_to_idx:
                continue
            self.data.append(p)
            self.targets.append(class_to_idx[class_name])
            loaded_count += 1

    def _load_data(self, data_dir: Path) -> None:
        """
        Original behavior: optionally sample samples_per_class per class.
        NOTE: we won't use this for source splits anymore — we'll build splits outside.
        """
        import random

        self.classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        rng = random.Random(self.seed)

        for class_name in self.classes:
            class_dir = data_dir / class_name
            class_images = [
                str(p) for p in class_dir.glob('*')
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]

            if self.samples_per_class is not None and len(class_images) > self.samples_per_class:
                class_images = rng.sample(class_images, self.samples_per_class)

            for img_path in class_images:
                self.data.append(img_path)
                self.targets.append(class_to_idx[class_name])

        print(f"[DEBUG] _load_data for {self.domain}: loaded {len(self.data)} images")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data[idx]
        target = self.targets[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_mnist_transforms(train: bool = True) -> transforms.Compose:
    """
    Transforms for MNIST/MNIST-M experiments.
    Images are 28x28 with mean subtraction normalization.
    """
    transform_list = [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
    
    transform_list.append(
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    )
    
    return transforms.Compose(transform_list)


def get_svhn_transforms(train: bool = True, data_root: str = './datasets') -> transforms.Compose:
    """
    Transforms for SVHN/SynNumbers experiments.
    Standardization to [-1, 1] range for stability.
    """
    transform_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # Fixed normalization to [-1, 1] range
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]

    return transforms.Compose(transform_list)


def get_gtsrb_transforms(train: bool = True) -> transforms.Compose:
    """
    Transforms for GTSRB/SynSigns experiments.
    Images are resized to 40x40.
    No data augmentation.
    """
    transform_list = [
        transforms.Resize((40, 40)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]

    return transforms.Compose(transform_list)


def get_office_transforms(train: bool = True) -> transforms.Compose:
    """
    Transforms for Office dataset experiments.
    Uses AlexNet-style preprocessing (224x224).
    No data augmentation.
    """
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

    return transforms.Compose(transform_list)


def get_dataloaders(
    source: str,
    target: str,
    data_root: str = './datasets',
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    seed: int = 42,
    use_augmented_synsigns: bool = False  # Not currently used
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    experiment_config = _get_experiment_config(source, target)

    source_transform_train = experiment_config['transform_train']
    source_transform_test = experiment_config['transform_test']
    target_transform_train = experiment_config['transform_train']
    target_transform_test = experiment_config['transform_test']

    # --- Office special handling: proper 5-split protocol without leakage ---
    if source.lower() in ['amazon', 'dslr', 'webcam'] and target.lower() in ['amazon', 'dslr', 'webcam']:
        # N labeled source examples per class (standard protocol)
        source_samples = 20 if source.lower() == 'amazon' else 8

        # Build a full list of source images first (no sampling)
        source_full = Office(
            root=data_root,
            domain=source,
            transform=source_transform_train,
            download=download,
            samples_per_class=None,
            seed=seed
        )

        # Group indices by class, sample N/class for train, rest for test
        import random
        rng = random.Random(seed)

        class_to_paths = {c: [] for c in range(31)}
        for p, y in zip(source_full.data, source_full.targets):
            class_to_paths[y].append(p)

        source_train_paths = []
        source_test_paths = []

        for c in range(31):
            paths = class_to_paths[c]
            rng.shuffle(paths)
            n = min(source_samples, len(paths))
            source_train_paths.extend(paths[:n])
            source_test_paths.extend(paths[n:])

        print(f"[DEBUG] {source.upper()} split: {len(source_train_paths)} train, {len(source_test_paths)} test")

        # Source train/test datasets from explicit subsets
        source_train = Office(
            root=data_root,
            domain=source,
            transform=source_transform_train,
            download=download,
            subset_paths=source_train_paths,
            seed=seed
        )
        source_test = Office(
            root=data_root,
            domain=source,
            transform=source_transform_test,
            download=download,
            subset_paths=source_test_paths,
            seed=seed
        )

        # Target: use ALL images unlabeled for training (transductive)
        target_train = Office(
            root=data_root,
            domain=target,
            transform=target_transform_train,
            download=download,
            samples_per_class=None,
            seed=seed
        )
        # Target eval: typically evaluate on ALL target images (labels not used in training)
        target_test = Office(
            root=data_root,
            domain=target,
            transform=target_transform_test,
            download=download,
            samples_per_class=None,
            seed=seed
        )

        print(f"[DEBUG] {target.upper()} dataset: {len(target_train)} train, {len(target_test)} test")

    elif target.lower() == 'gtsrb' or source.lower() == 'gtsrb':
        # GTSRB special handling: test set has no labels, so split training set
        from torch.utils.data import Subset
        import random

        # Load full GTSRB dataset (always from training data which has labels)
        if source.lower() == 'gtsrb':
            source_full = GTSRB(data_root, train=True, transform=source_transform_train, download=download)

            # Split into train/test (80/20)
            rng = random.Random(seed)
            indices = list(range(len(source_full)))
            rng.shuffle(indices)

            split_idx = int(0.8 * len(indices))
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            source_train = Subset(source_full, train_indices)

            # Create test set with test transform
            source_test_full = GTSRB(data_root, train=True, transform=source_transform_test, download=download)
            source_test = Subset(source_test_full, test_indices)

            print(f"GTSRB source split: {len(source_train)} train, {len(source_test)} test")
        else:
            source_train = _create_dataset(
                source, data_root, train=True,
                transform=source_transform_train, download=download,
                use_augmented=use_augmented_synsigns
            )
            source_test = _create_dataset(
                source, data_root, train=False,
                transform=source_transform_test, download=download,
                use_augmented=use_augmented_synsigns
            )

        if target.lower() == 'gtsrb':
            target_full = GTSRB(data_root, train=True, transform=target_transform_train, download=download)

            # Split into train/test (80/20)
            rng = random.Random(seed)
            indices = list(range(len(target_full)))
            rng.shuffle(indices)

            split_idx = int(0.8 * len(indices))
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]

            target_train = Subset(target_full, train_indices)

            # Create test set with test transform
            target_test_full = GTSRB(data_root, train=True, transform=target_transform_test, download=download)
            target_test = Subset(target_test_full, test_indices)

            print(f"GTSRB target split: {len(target_train)} train, {len(target_test)} test")
        else:
            target_train = _create_dataset(
                target, data_root, train=True,
                transform=target_transform_train, download=download,
                use_augmented=use_augmented_synsigns
            )
            target_test = _create_dataset(
                target, data_root, train=False,
                transform=target_transform_test, download=download,
                use_augmented=use_augmented_synsigns
            )

    else:
        # --- Non-Office, non-GTSRB datasets: keep original logic ---
        source_samples = None
        if source.lower() in ['amazon', 'dslr', 'webcam']:
            source_samples = 20 if source.lower() == 'amazon' else 8

        source_train = _create_dataset(
            source, data_root, train=True,
            transform=source_transform_train, download=download,
            samples_per_class=source_samples, seed=seed,
            use_augmented=use_augmented_synsigns
        )
        source_test = _create_dataset(
            source, data_root, train=False,
            transform=source_transform_test, download=download,
            use_augmented=use_augmented_synsigns
        )

        target_train = _create_dataset(
            target, data_root, train=True,
            transform=target_transform_train, download=download,
            use_augmented=use_augmented_synsigns
        )
        target_test = _create_dataset(
            target, data_root, train=False,
            transform=target_transform_test, download=download,
            use_augmented=use_augmented_synsigns
        )
    
    # Create dataloaders
    # Paper: "128 sized batches. A half of each batch is populated by the samples
    # from the source domain (with known labels), the rest is comprised of the
    # target domain (with unknown labels)."
    # So: batch_size/2 source + batch_size/2 target = batch_size total
    source_train_loader = DataLoader(
        source_train,
        batch_size=batch_size // 2,  # Half batch for source
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    source_test_loader = DataLoader(
        source_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    target_train_loader = DataLoader(
        target_train,
        batch_size=batch_size // 2,  # Half batch for target
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    target_test_loader = DataLoader(
        target_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return source_train_loader, source_test_loader, target_train_loader, target_test_loader


def _get_experiment_config(source: str, target: str) -> dict:
    """Get transforms and config for experiment type."""
    source = source.lower()
    target = target.lower()
    
    # MNIST <-> MNIST-M (handle all naming variations: mnist_m, mnistm)
    if source in ['mnist', 'mnistm', 'mnist_m'] and target in ['mnist', 'mnistm', 'mnist_m']:
        return {
            'transform_train': get_mnist_transforms(train=True),
            'transform_test': get_mnist_transforms(train=False),
            'architecture': 'mnist',
            'num_classes': 10
        }
    
    # SVHN, SynNumbers, MNIST experiments
    if source in ['svhn', 'syn_numbers', 'mnist'] and target in ['svhn', 'syn_numbers', 'mnist']:
        return {
            'transform_train': get_svhn_transforms(train=True),
            'transform_test': get_svhn_transforms(train=False),
            'architecture': 'svhn',
            'num_classes': 10
        }
    
    # GTSRB, SynSigns experiments
    if source in ['gtsrb', 'syn_signs'] and target in ['gtsrb', 'syn_signs']:
        return {
            'transform_train': get_gtsrb_transforms(train=True),
            'transform_test': get_gtsrb_transforms(train=False),
            'architecture': 'gtsrb',
            'num_classes': 43
        }
    
    # Office experiments
    if source in ['amazon', 'dslr', 'webcam'] and target in ['amazon', 'dslr', 'webcam']:
        return {
            'transform_train': get_office_transforms(train=True),
            'transform_test': get_office_transforms(train=False),
            'architecture': 'office',
            'num_classes': 31
        }
    
    # Default to SVHN-style transforms
    return {
        'transform_train': get_svhn_transforms(train=True),
        'transform_test': get_svhn_transforms(train=False),
        'architecture': 'svhn',
        'num_classes': 10
    }


def _create_dataset(
    name: str,
    root: str,
    train: bool,
    transform: Callable,
    download: bool,
    samples_per_class: Optional[int] = None,
    seed: int = 42,
    use_augmented: bool = False  # Not currently used
)-> Dataset:
    name = name.lower()

    # Handle naming variations
    name_mappings = {
        'mnist_m': 'mnistm',  # Support both mnist_m and mnistm
        'synsigns': 'syn_signs',
    }
    name = name_mappings.get(name, name)

    if name == 'mnist':
        return MNIST(root, train=train, transform=transform, download=download)
    elif name == 'mnistm':
        return MNISTM(root, train=train, transform=transform, download=download)
    elif name == 'svhn':
        return SVHN(root, train=train, transform=transform, download=download)
    elif name == 'syn_numbers':
        return SyntheticNumbers(root, train=train, transform=transform, download=download)
    elif name == 'syn_signs':
        # Use SynSigns Cycles dataset (realistic 3D rendered traffic signs)
        return SyntheticSigns(root, train=train, transform=transform,
                            use_augmented=use_augmented)
    elif name == 'gtsrb':
        return GTSRB(root, train=train, transform=transform, download=download)
    elif name in ['amazon', 'dslr', 'webcam']:
        return Office(root, domain=name, transform=transform, download=download,
                     samples_per_class=samples_per_class, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_experiment_info(source: str, target: str) -> dict:
    """Get experiment configuration info."""
    config = _get_experiment_config(source, target)
    return {
        'architecture': config['architecture'],
        'num_classes': config['num_classes'],
        'source': source,
        'target': target
    }