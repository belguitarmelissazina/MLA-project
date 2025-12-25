"""
DANN Model Components - Unsupervised Domain Adaptation by Backpropagation
Authors: Ganin & Lempitsky, 2015

This module implements:
- Gradient Reversal Layer (GRL)
- Feature Extractors for different architectures
- Label Predictors
- Domain Classifiers
- Complete DANN model
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) implementation.
    
    During forward propagation: acts as identity transform
    During backpropagation: multiplies gradient by -lambda
    
    Mathematically defined as pseudo-function R_lambda(x):
        R_lambda(x) = x (forward)
        dR_lambda/dx = -lambda * I (backward)
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass: identity transform.
        
        Args:
            ctx: Context for backward pass
            x: Input tensor
            lambda_: Adaptation factor (stored for backward)
            
        Returns:
            Input tensor unchanged
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: reverse gradient by multiplying with -lambda.
        
        Args:
            ctx: Context from forward pass
            grad_output: Gradient from subsequent layer
            
        Returns:
            Gradient multiplied by -lambda, None for lambda gradient
        """
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module wrapper.
    
    Can be used in nn.Sequential or as standalone layer.
    Lambda can be set dynamically during training.
    """
    
    def __init__(self, lambda_: float = 1.0):
        """
        Args:
            lambda_: Initial adaptation factor (default: 1.0)
        """
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Update the lambda value."""
        self.lambda_ = lambda_


class MNISTFeatureExtractor(nn.Module):
    """
    Feature extractor for MNIST experiments.
    Architecture from Appendix B, inspired by LeNet-5.
    
    Structure:
    - Conv 5x5, 32 maps, ReLU
    - Max-pool 2x2, stride 2
    - Conv 5x5, 48 maps, ReLU  
    - Max-pool 2x2, stride 2
    
    Output: 48 * 4 * 4 = 768 features (for 28x28 input)
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        # Output size for 28x28 input: 48 * 4 * 4 = 768
        self.output_dim = 48 * 4 * 4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # 28 -> 24
        x = self.pool1(x)              # 24 -> 12
        x = self.relu(self.conv2(x))  # 12 -> 8
        x = self.pool2(x)              # 8 -> 4
        x = x.view(x.size(0), -1)
        return x


class MNISTLabelPredictor(nn.Module):
    """
    Label predictor for MNIST experiments.
    
    Structure:
    - FC 100 units, ReLU
    - FC 100 units, ReLU
    - FC 10 units (num_classes), Softmax
    """
    
    def __init__(self, input_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MNISTDomainClassifier(nn.Module):
    """
    Domain classifier for MNIST experiments.
    Simpler architecture: x -> 100 -> 2
    
    Structure:
    - FC 100 units, ReLU
    - FC 1 unit, Logistic (binary)
    """
    
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SVHNFeatureExtractor(nn.Module):
    """
    Feature extractor for SVHN experiments.
    Architecture from Appendix B, adopted from Srivastava et al., 2014.
    
    Structure:
    - Conv 5x5, 64 maps, ReLU
    - Max-pool 3x3, stride 2
    - Conv 5x5, 64 maps, ReLU
    - Max-pool 3x3, stride 2
    - Conv 5x5, 128 maps, ReLU
    
    For 32x32 input: output is 128 * 3 * 3 = 1152 features
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU(inplace=True)
        
        # Output for 32x32 input
        self.output_dim = 128 * 8 * 8  
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # 32 -> 32
        x = self.pool1(x)              # 32 -> 16
        x = self.relu(self.conv2(x))  # 16 -> 16
        x = self.pool2(x)              # 16 -> 8
        x = self.relu(self.conv3(x))  # 8 -> 8
        x = x.view(x.size(0), -1)
        return x


class SVHNLabelPredictor(nn.Module):
    """
    Label predictor for SVHN experiments.
    
    Structure:
    - FC 3072 units, ReLU
    - FC 2048 units, ReLU
    - FC 10 units, Softmax
    """
    
    def __init__(self, input_dim: int = 8192, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class SVHNDomainClassifier(nn.Module):
    """
    Domain classifier for SVHN experiments.
    Standard 3-layer architecture: x -> 1024 -> 1024 -> 1
    
    Structure:
    - FC 1024 units, ReLU
    - FC 1024 units, ReLU
    - FC 1 unit, Logistic
    """
    
    def __init__(self, input_dim: int = 8192):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class GTSRBFeatureExtractor(nn.Module):
    """
    Feature extractor for GTSRB (traffic signs) experiments.
    Architecture from Appendix B, based on Ciresan et al., 2012.
    
    Structure:
    - Conv 5x5, 96 maps, ReLU
    - Max-pool 2x2, stride 2
    - Conv 3x3, 144 maps, ReLU
    - Max-pool 2x2, stride 2
    - Conv 5x5, 256 maps, ReLU
    - Max-pool 2x2, stride 2
    
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(96, 144, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(144, 256, kernel_size=5, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        # For 40x40 input
        self.output_dim = 256 * 2 * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # 40 -> 36
        x = self.pool1(x)              # 36 -> 18
        x = self.relu(self.conv2(x))  # 18 -> 16
        x = self.pool2(x)              # 16 -> 8
        x = self.relu(self.conv3(x))  # 8 -> 4
        x = self.pool3(x)              # 4 -> 2
        x = x.view(x.size(0), -1)
        return x


class GTSRBLabelPredictor(nn.Module):
    """
    Label predictor for GTSRB experiments.
    
    Structure:
    - FC 512 units, ReLU
    - FC num_classes units (43 for GTSRB), Softmax
    """
    
    def __init__(self, input_dim: int = 1024, num_classes: int = 43):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GTSRBDomainClassifier(nn.Module):
    """
    Domain classifier for GTSRB experiments.
    Standard 3-layer architecture: x -> 1024 -> 1024 -> 1
    """
    
    def __init__(self, input_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OfficeFeatureExtractor(nn.Module):
    """
    Feature extractor for Office dataset experiments.
    Uses pretrained AlexNet with modified fc7 bottleneck (256-dim).
    
    Following Tzeng et al., 2014 architecture used in the paper.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import alexnet, AlexNet_Weights
            if pretrained:
                weights = AlexNet_Weights.IMAGENET1K_V1
            else:
                weights = None
            alexnet_model = alexnet(weights=weights)
        except ImportError:
            # Fallback for older torchvision
            from torchvision.models import alexnet
            alexnet_model = alexnet(pretrained=pretrained)
        
        # Extract features up to fc6
        self.features = alexnet_model.features
        self.avgpool = alexnet_model.avgpool
        
        # Modified classifier with 256-dim bottleneck
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 256)  # Bottleneck
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        # Copy pretrained weights for fc6
        self.fc6.weight.data = alexnet_model.classifier[1].weight.data
        self.fc6.bias.data = alexnet_model.classifier[1].bias.data
        
        self.output_dim = 256
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.relu(self.fc7(x))
        return x


class OfficeLabelPredictor(nn.Module):
    """
    Label predictor for Office dataset.
    Single FC layer from 256-dim bottleneck to num_classes.
    """
    
    def __init__(self, input_dim: int = 256, num_classes: int = 31):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class OfficeDomainClassifier(nn.Module):
    """
    Domain classifier for Office dataset.
    2-layer architecture: x -> 1024 -> 1024 -> 1
    
    As specified: "2-layer domain classifier (x → 1024 → 1024 → 2)"
    """
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DANN(nn.Module):
    """
    Domain-Adversarial Neural Network (DANN).
    
    Complete model combining:
    - Feature extractor G_f
    - Label predictor G_y
    - Domain classifier G_d with GRL
    
    The model jointly optimizes:
    - Classification loss on source domain
    - Domain discrimination loss (adversarially via GRL)
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        label_predictor: nn.Module,
        domain_classifier: nn.Module,
        lambda_: float = 1.0
    ):
        """
        Args:
            feature_extractor: Feature extraction network G_f
            label_predictor: Classification network G_y
            domain_classifier: Domain discrimination network G_d
            lambda_: Initial GRL adaptation factor
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
        self.domain_classifier = domain_classifier
        self.grl = GradientReversalLayer(lambda_)
    
    def forward(
        self,
        x: torch.Tensor,
        alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete DANN.
        
        Args:
            x: Input images
            alpha: Optional lambda override for GRL
            
        Returns:
            Tuple of (class_logits, domain_logits, features)
        """
        if alpha is not None:
            self.grl.set_lambda(alpha)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Class prediction (no GRL)
        class_logits = self.label_predictor(features)
        
        # Domain prediction (with GRL for gradient reversal)
        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)
        
        return class_logits, domain_logits, features
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions only (for inference)."""
        features = self.feature_extractor(x)
        return self.label_predictor(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features only (for visualization)."""
        return self.feature_extractor(x)


def create_dann_model(
    architecture: str,
    num_classes: int = 10,
    pretrained: bool = True,
    lambda_: float = 1.0
) -> DANN:
    """
    Factory function to create DANN model for different experiments.
    
    Args:
        architecture: One of 'mnist', 'svhn', 'gtsrb', 'office'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for Office)
        lambda_: Initial GRL lambda value
        
    Returns:
        Configured DANN model
    """
    if architecture == 'mnist':
        feature_extractor = MNISTFeatureExtractor()
        label_predictor = MNISTLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
        domain_classifier = MNISTDomainClassifier(
            input_dim=feature_extractor.output_dim
        )
    elif architecture == 'svhn':
        feature_extractor = SVHNFeatureExtractor()
        label_predictor = SVHNLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
        domain_classifier = SVHNDomainClassifier(
            input_dim=feature_extractor.output_dim
        )
    elif architecture == 'gtsrb':
        feature_extractor = GTSRBFeatureExtractor()
        label_predictor = GTSRBLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
        domain_classifier = GTSRBDomainClassifier(
            input_dim=feature_extractor.output_dim
        )
    elif architecture == 'office':
        feature_extractor = OfficeFeatureExtractor(pretrained=pretrained)
        label_predictor = OfficeLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
        domain_classifier = OfficeDomainClassifier(
            input_dim=feature_extractor.output_dim
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return DANN(
        feature_extractor=feature_extractor,
        label_predictor=label_predictor,
        domain_classifier=domain_classifier,
        lambda_=lambda_
    )


class SourceOnlyModel(nn.Module):
    """
    Source-only baseline model (no domain adaptation).
    Used for comparison with DANN.
    """
    
    def __init__(
        self,
        feature_extractor: nn.Module,
        label_predictor: nn.Module
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (class_logits, features)
        """
        features = self.feature_extractor(x)
        class_logits = self.label_predictor(features)
        return class_logits, features
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        features = self.feature_extractor(x)
        return self.label_predictor(features)


def create_source_only_model(
    architecture: str,
    num_classes: int = 10,
    pretrained: bool = True
) -> SourceOnlyModel:
    """
    Factory function to create source-only baseline model.
    
    Args:
        architecture: One of 'mnist', 'svhn', 'gtsrb', 'office'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for Office)
        
    Returns:
        Configured SourceOnlyModel
    """
    if architecture == 'mnist':
        feature_extractor = MNISTFeatureExtractor()
        label_predictor = MNISTLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
    elif architecture == 'svhn':
        feature_extractor = SVHNFeatureExtractor()
        label_predictor = SVHNLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
    elif architecture == 'gtsrb':
        feature_extractor = GTSRBFeatureExtractor()
        label_predictor = GTSRBLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
    elif architecture == 'office':
        feature_extractor = OfficeFeatureExtractor(pretrained=pretrained)
        label_predictor = OfficeLabelPredictor(
            input_dim=feature_extractor.output_dim,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return SourceOnlyModel(
        feature_extractor=feature_extractor,
        label_predictor=label_predictor
    )
