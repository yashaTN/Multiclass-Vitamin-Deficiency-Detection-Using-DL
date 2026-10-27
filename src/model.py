"""
ResNet Transfer Learning Architecture for Multiclass Vitamin Deficiency Detection.
Follows pipeline:
Input Image -> Image Preprocessing -> ResNet Feature Extractor (Transfer Learning) -> Classification Head -> Deficiency Class Detection
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from .config import NUM_CLASSES, DEVICE, CLASS_NAMES

class VitaminResNet(nn.Module):
    """
    Transfer learning network based on ResNet backbone.
    Utilizes pretrained ImageNet weights for robust feature representations
    of dermatological, mucosal, oral, ocular, and ungual clinical manifestations.
    """
    def __init__(self, architecture: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super(VitaminResNet, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        # Load ResNet backbone
        if architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
        else:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features

        # Backbone layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.avgpool = backbone.avgpool

        # Custom high-capacity classification head with regularizing Dropout and Batch Normalization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

        # Gradient placeholders for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward_features(self, x):
        """Extract spatial convolutional features up to layer4."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        
        # Register hook on last feature map for Grad-CAM explainability
        if x.requires_grad:
            h = features.register_hook(self.activations_hook)
        self.activations = features

        pooled = self.avgpool(features)
        logits = self.classifier(pooled)
        return logits

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations


def build_model(architecture: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True) -> VitaminResNet:
    """Factory helper to instantiate VitaminResNet."""
    model = VitaminResNet(architecture=architecture, num_classes=num_classes, pretrained=pretrained)
    return model.to(DEVICE)
