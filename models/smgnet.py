"""
SMG-Net model architecture for MRI image classification.
SMG-Net (Spatial Multi-Granularity Network) for neurological disorder classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMGNet(nn.Module):
    """
    SMG-Net architecture for multi-class neurological disorder classification.
    
    Input: Grayscale MRI images (1 channel, 512x512)
    Output: Classification logits (num_classes)
    """
    
    def __init__(self, num_classes: int = 4, in_channels: int = 1):
        """
        Initialize SMG-Net model.
        
        Args:
            num_classes: Number of classification classes (default: 4)
            in_channels: Number of input channels (default: 1 for grayscale)
        """
        super(SMGNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Feature extraction backbone (CNN layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
