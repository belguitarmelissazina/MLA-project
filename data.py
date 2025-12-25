"""
Dataset Handling for DANN Experiments
Authors: Ganin & Lempitsky, 2015 Reproduction

This module implements:
- MNIST, MNIST-M (generated), SVHN datasets
- Synthetic Numbers and Synthetic Signs generation
- GTSRB (German Traffic Sign Recognition Benchmark)
- Office dataset (Amazon, DSLR, Webcam)
- Data augmentation and preprocessing as per paper
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

        # Use torchvision MNIST (already creates MNIST/ subfolder automatically)
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
    
    As described in the paper:
    "We blend digits from the original set over patches randomly extracted 
    from color photos from BSDS500. This operation is formally defined as:
    I_out_ijk = |I_1_ijk - I_2_ijk|"
    
    Where I_1 is the digit and I_2 is the background patch.
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

        # Download and load MNIST (torchvision creates MNIST/ subfolder automatically)
        mnist = datasets.MNIST(
            root=str(self.root),
            train=self.train,
            download=True
        )
        
        # Download BSDS500 or use random colored patches as fallback
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
            
            # Blend using absolute difference (as per paper)
            # I_out = |I_1 - I_2| where I_1 is digit, I_2 is background
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
    
    As described in the paper:
    "500,000 images generated from Windows fonts by varying the text
    (that includes different one-, two-, and three-digit numbers),
    positioning, orientation, background and stroke colors, and the
    amount of blur."
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
    Synthetic Traffic Signs Dataset.
    
    As described in the paper:
    "100,000 synthetic images simulating various photo-shooting conditions"
    for 43 traffic sign classes.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        num_samples: int = 100000
    ):
        self.root = Path(root)
        self.transform = transform
        self.train = train
        self.num_samples = num_samples if train else 10000
        self.num_classes = 43
        
        syn_dir = self.root / 'syn_signs'
        pkl_path = syn_dir / ('train.pkl' if train else 'test.pkl')
        
        if pkl_path.exists():
            self._load_preprocessed(pkl_path)
        elif download:
            self._generate_synthetic_signs()
        else:
            raise RuntimeError(
                f"Synthetic Signs not found at {pkl_path}. "
                "Set download=True to generate it."
            )
    
    def _load_preprocessed(self, path: Path) -> None:
        """Load preprocessed data from pickle file."""
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict['data']
        self.targets = data_dict['targets']
    
    def _generate_synthetic_signs(self) -> None:
        """Generate synthetic traffic sign images."""
        print(f"Generating Synthetic Signs dataset ({self.num_samples} samples)...")
        
        syn_dir = self.root / 'syn_signs'
        syn_dir.mkdir(parents=True, exist_ok=True)
        
        from PIL import ImageDraw, ImageFilter
        
        # Traffic sign colors by category
        sign_colors = {
            'prohibitory': ((255, 255, 255), (255, 0, 0)),    # White bg, red border
            'mandatory': ((0, 0, 255), (255, 255, 255)),       # Blue bg, white symbol
            'danger': ((255, 255, 255), (255, 0, 0)),          # White bg, red border
            'other': ((255, 255, 0), (0, 0, 0)),               # Yellow bg, black symbol
        }
        
        data = []
        targets = []
        
        for i in range(self.num_samples):
            # Random class (0-42)
            class_id = np.random.randint(0, self.num_classes)
            
            # Determine sign type based on GTSRB class groupings
            if class_id < 8:
                sign_type = 'danger'
            elif class_id < 18:
                sign_type = 'prohibitory'
            elif class_id < 33:
                sign_type = 'mandatory'
            else:
                sign_type = 'other'
            
            bg_color, symbol_color = sign_colors[sign_type]
            
            # Random background
            scene_bg = tuple(np.random.randint(100, 200, 3))
            img = Image.new('RGB', (40, 40), scene_bg)
            draw = ImageDraw.Draw(img)
            
            # Draw sign (simplified)
            margin = np.random.randint(2, 6)
            
            if sign_type in ['danger']:
                # Triangle
                points = [
                    (20, margin),
                    (40 - margin, 40 - margin),
                    (margin, 40 - margin)
                ]
                draw.polygon(points, fill=bg_color, outline=symbol_color)
            elif sign_type in ['prohibitory', 'mandatory']:
                # Circle
                draw.ellipse(
                    [margin, margin, 40 - margin, 40 - margin],
                    fill=bg_color,
                    outline=symbol_color
                )
            else:
                # Rectangle
                draw.rectangle(
                    [margin, margin, 40 - margin, 40 - margin],
                    fill=bg_color,
                    outline=symbol_color
                )
            
            # Add class number as simple symbol
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
                text = str(class_id % 10)
                draw.text((15, 15), text, fill=symbol_color, font=font)
            except:
                pass
            
            # Random blur
            if np.random.random() > 0.5:
                blur_radius = np.random.uniform(0, 1.0)
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
            
            # Random brightness adjustment
            if np.random.random() > 0.5:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(img)
                factor = np.random.uniform(0.7, 1.3)
                img = enhancer.enhance(factor)
            
            data.append(np.array(img))
            targets.append(class_id)
            
            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1}/{self.num_samples} samples")
        
        self.data = np.array(data)
        self.targets = targets
        
        # Save preprocessed data
        pkl_path = syn_dir / ('train.pkl' if self.train else 'test.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({'data': self.data, 'targets': self.targets}, f)
        
        print(f"Synthetic Signs saved to {pkl_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx]
        target = self.targets[idx]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class GTSRB(Dataset):
    """
    German Traffic Sign Recognition Benchmark (GTSRB).
    43 classes of traffic signs.
    """
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = Path(root)
        self.transform = transform
        self.train = train
        
        self.data = []
        self.targets = []
        
        gtsrb_dir = self.root / 'GTSRB'
        
        if train:
            data_dir = gtsrb_dir / 'Final_Training' / 'Images'
        else:
            data_dir = gtsrb_dir / 'Final_Test' / 'Images'
        
        if data_dir.exists():
            self._load_data(data_dir)
        elif download:
            self._download_and_extract()
            self._load_data(data_dir)
        else:
            raise RuntimeError(
                f"GTSRB not found at {gtsrb_dir}. "
                "Please download from: https://benchmark.ini.rub.de/gtsrb_dataset.html "
                "and extract to datasets/GTSRB/"
            )
    
    def _download_and_extract(self) -> None:
        """Download GTSRB dataset."""
        print("GTSRB requires manual download.")
        print("Please download from: https://benchmark.ini.rub.de/gtsrb_dataset.html")
        print("Extract to: datasets/GTSRB/")
        raise RuntimeError("GTSRB download not implemented. Please download manually.")
    
    def _load_data(self, data_dir: Path) -> None:
        """Load GTSRB images and labels."""
        import csv
        
        if self.train:
            # Load training data from class directories
            for class_id in range(43):
                class_dir = data_dir / f'{class_id:05d}'
                if not class_dir.exists():
                    continue
                
                # Read annotation file
                annotation_file = class_dir / f'GT-{class_id:05d}.csv'
                if annotation_file.exists():
                    with open(annotation_file, 'r') as f:
                        reader = csv.DictReader(f, delimiter=';')
                        for row in reader:
                            img_path = class_dir / row['Filename']
                            if img_path.exists():
                                self.data.append(str(img_path))
                                self.targets.append(int(row['ClassId']))
        else:
            # Load test data
            annotation_file = data_dir / 'GT-final_test.csv'
            if annotation_file.exists():
                with open(annotation_file, 'r') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    for row in reader:
                        img_path = data_dir / row['Filename']
                        if img_path.exists():
                            self.data.append(str(img_path))
                            self.targets.append(int(row['ClassId']))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.data[idx]
        target = self.targets[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


class Office(Dataset):
    """
    Office Dataset (Amazon, DSLR, Webcam).
    31 object categories.
    
    Requires manual download from:
    https://faculty.cc.gatech.edu/~judy/domainadapt/
    """
    
    def __init__(
        self,
        root: str,
        domain: str = 'amazon',  # 'amazon', 'dslr', 'webcam'
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = Path(root)
        self.domain = domain.lower()
        self.transform = transform
        
        self.data = []
        self.targets = []
        self.classes = []
        
        office_dir = self.root / 'office31' / self.domain / 'images'
        
        if office_dir.exists():
            self._load_data(office_dir)
        elif download:
            print(f"Office dataset requires manual download.")
            print("Please download from: https://faculty.cc.gatech.edu/~judy/domainadapt/")
            print(f"Extract to: {self.root}/office31/")
            raise RuntimeError("Office dataset not found. Please download manually.")
        else:
            raise RuntimeError(
                f"Office dataset not found at {office_dir}. "
                "Please download from https://faculty.cc.gatech.edu/~judy/domainadapt/ "
                f"and extract to {self.root}/office31/"
            )
    
    def _load_data(self, data_dir: Path) -> None:
        """Load Office dataset images and labels."""
        self.classes = sorted([
            d.name for d in data_dir.iterdir() if d.is_dir()
        ])
        
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = data_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.data.append(str(img_path))
                    self.targets.append(class_to_idx[class_name])
    
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
    Paper uses mean subtraction. Images are 28x28.
    """
    transform_list = [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
    
    # Mean subtraction (paper mentions this)
    # Using ImageNet-style normalization as approximation
    transform_list.append(
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    )
    
    return transforms.Compose(transform_list)


def get_svhn_transforms(train: bool = True) -> transforms.Compose:
    """
    Transforms for SVHN/SynNumbers experiments.
    Paper uses mean subtraction. Images are 32x32.
    """
    transform_list = [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
    
    return transforms.Compose(transform_list)


def get_gtsrb_transforms(train: bool = True) -> transforms.Compose:
    """
    Transforms for GTSRB/SynSigns experiments.
    Images are resized to 40x40.
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
    """
    if train:
        transform_list = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    else:
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
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for source and target domains.
    
    Args:
        source: Source domain name
        target: Target domain name
        data_root: Root directory for datasets
        batch_size: Batch size (paper uses 128)
        num_workers: Number of data loading workers
        download: Whether to download/generate datasets
        
    Returns:
        Tuple of (source_train_loader, source_test_loader,
                  target_train_loader, target_test_loader)
    """
    # Determine experiment type and get appropriate transforms/datasets
    experiment_config = _get_experiment_config(source, target)
    
    source_transform_train = experiment_config['transform_train']
    source_transform_test = experiment_config['transform_test']
    target_transform_train = experiment_config['transform_train']
    target_transform_test = experiment_config['transform_test']
    
    # Create source datasets
    source_train = _create_dataset(
        source, data_root, train=True,
        transform=source_transform_train, download=download
    )
    source_test = _create_dataset(
        source, data_root, train=False,
        transform=source_transform_test, download=download
    )
    
    # Create target datasets
    target_train = _create_dataset(
        target, data_root, train=True,
        transform=target_transform_train, download=download
    )
    target_test = _create_dataset(
        target, data_root, train=False,
        transform=target_transform_test, download=download
    )
    
    # Create dataloaders
    source_train_loader = DataLoader(
        source_train,
        batch_size=batch_size,
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
        batch_size=batch_size,
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
    
    # MNIST <-> MNIST-M
    if source in ['mnist', 'mnistm'] and target in ['mnist', 'mnistm']:
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
    download: bool
) -> Dataset:
    """Create dataset by name."""
    name = name.lower()
    
    if name == 'mnist':
        return MNIST(root, train=train, transform=transform, download=download)
    elif name == 'mnistm':
        return MNISTM(root, train=train, transform=transform, download=download)
    elif name == 'svhn':
        return SVHN(root, train=train, transform=transform, download=download)
    elif name == 'syn_numbers':
        return SyntheticNumbers(root, train=train, transform=transform, download=download)
    elif name == 'syn_signs':
        return SyntheticSigns(root, train=train, transform=transform, download=download)
    elif name == 'gtsrb':
        return GTSRB(root, train=train, transform=transform, download=download)
    elif name in ['amazon', 'dslr', 'webcam']:
        return Office(root, domain=name, transform=transform, download=download)
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
