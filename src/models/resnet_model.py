"""ResNet-based deepfake audio detector model.

This module provides ResNetDeepfakeDetector which adapts torchvision's
ResNet18 to accept 1-channel inputs (grayscale/CQCC) and a binary classifier head.
"""
from typing import Optional

import torch
import torch.nn as nn


class ResNetDeepfakeDetector(nn.Module):
    """ResNet18 backbone adapted for deepfake audio detection.

    The class lazily imports `torchvision.models.resnet18` inside __init__ so that
    importing this module does not immediately require torchvision to be installed.
    """

    def __init__(self, pretrained: bool = False, dropout_p: float = 0.5):
        super().__init__()

        try:
            # Import lazily to avoid hard dependency at module-import time
            from torchvision.models import resnet18
        except Exception as e:
            raise RuntimeError(
                "torchvision is required to build ResNetDeepfakeDetector. "
                "Install with: pip install torchvision"
            ) from e

        # Create base model
        base = resnet18(pretrained=pretrained)

        # Replace first conv to accept 1-channel input
        in_channels = 1
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        # He initialization for the new conv
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        base.conv1 = new_conv

        # Build a backbone that outputs pooled features
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )

        # Embedding size is base.fc.in_features (should be 512 for ResNet18)
        embedding_size = base.fc.in_features

        # Classification head
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(embedding_size, 2)

        # He initialization for linear layer
        nn.init.kaiming_normal_(self.classifier.weight, a=0, mode='fan_out', nonlinearity='relu')
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2)
        """
        emb = self.get_embedding(x)
        out = self.dropout(emb)
        logits = self.classifier(out)
        return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding (features before classification).

        Runs input through the backbone and returns a flattened embedding tensor
        of shape (batch, embedding_size).
        """
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x
