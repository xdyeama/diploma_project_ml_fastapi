"""
SMG-Net model architecture for MRI image classification.
SMG-Net (Spatial Multi-Granularity Network) for neurological disorder classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MRFEBlock(nn.Module):
    """Multi-Receptive Field Extraction block with three parallel branches."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.branch2 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.branch3 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2)

        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1)
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        x_cat = torch.cat([b1, b2, b3], dim=1)
        fused = self.fusion(x_cat)
        res = self.residual(x)
        return self.relu(self.bn(fused + res))


class SMGNet(nn.Module):
    """
    SMG-Net architecture for multi-class neurological disorder classification.

    Input: Grayscale MRI images (1 channel, 512x512)
    Output: Classification logits (num_classes)
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.initial = nn.Conv2d(in_channels, 32, 3, padding=1)

        self.block1 = MRFEBlock(32, 64)
        self.block2 = MRFEBlock(64, 128)
        self.block3 = MRFEBlock(128, 256)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)

        x = self.block1(x)
        x = F.max_pool2d(x, 2)

        x = self.block2(x)
        x = F.max_pool2d(x, 2)

        x = self.block3(x)
        x = F.max_pool2d(x, 2)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
