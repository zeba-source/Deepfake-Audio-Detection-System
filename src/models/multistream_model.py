
from typing import Optional

import torch
import torch.nn as nn

from src.models.resnet_model import ResNetDeepfakeDetector


class MultiStreamDetector(nn.Module):
    """Multi-stream detector that fuses mel-spectrogram and CQCC branches.

    Each branch is a ResNet18-based encoder that outputs a 512-dim embedding.
    The concatenated 1024-dim embedding is passed through fusion layers to produce
    logits for binary classification.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Two independent ResNet branches
        self.mel_branch = ResNetDeepfakeDetector(pretrained=pretrained)
        self.cqcc_branch = ResNetDeepfakeDetector(pretrained=pretrained)

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2),
        )

        self._initialize_weights()

    def forward(self, mel_input: torch.Tensor, cqcc_input: torch.Tensor) -> torch.Tensor:
        mel_emb = self.mel_branch.get_embedding(mel_input)
        cqcc_emb = self.cqcc_branch.get_embedding(cqcc_input)

        # Concatenate embeddings
        emb = torch.cat([mel_emb, cqcc_emb], dim=1)
        logits = self.fusion(emb)
        return logits

    def _initialize_weights(self):
        # Initialize fusion layers with Kaiming/He initialization
        for m in self.fusion:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
