"""
Legacy U-Net architecture matching the training notebook (unet_model_training.ipynb).
Used to load checkpoints saved from that notebook; do not use for new training.
"""

import torch
import torch.nn as nn


def center_crop(tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """Crop tensor to match target_tensor spatial size (center crop)."""
    _, _, h, w = target_tensor.shape
    _, _, H, W = tensor.shape
    delta_h = (H - h) // 2
    delta_w = (W - w) // 2
    return tensor[:, :, delta_h : delta_h + h, delta_w : delta_w + w]


class DoubleConv(nn.Module):
    """Double conv block from the training notebook (conv only, no BN)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_init: str = "he"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        if kernel_init == "he":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Legacy U-Net from the training notebook.
    Encoder: conv1..conv5, pool1..pool4, drop5.
    Decoder: up6..up9, conv6..conv9, out.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512)
        self.drop5 = nn.Dropout(dropout)

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.drop5(self.conv5(p4))

        u6 = self.up6(c5)
        c6 = self.conv6(torch.cat([center_crop(c4, u6), u6], dim=1))
        u7 = self.up7(c6)
        c7 = self.conv7(torch.cat([center_crop(c3, u7), u7], dim=1))
        u8 = self.up8(c7)
        c8 = self.conv8(torch.cat([center_crop(c2, u8), u8], dim=1))
        u9 = self.up9(c8)
        c9 = self.conv9(torch.cat([center_crop(c1, u9), u9], dim=1))
        return self.out(c9)
