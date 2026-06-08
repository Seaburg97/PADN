import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock3D(in_channels, out_channels),
            ConvBlock3D(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels + in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            (
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ),
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_channels=64,
        bilinear=True,
        deep_supervision=True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        c = base_channels

        self.inc = DoubleConv3D(in_channels, c)
        self.down1 = Down3D(c, c * 2)
        self.down2 = Down3D(c * 2, c * 4)
        self.down3 = Down3D(c * 4, c * 8)
        self.down4 = Down3D(c * 8, c * 16)

        self.up1 = Up3D(c * 16, c * 8, bilinear)
        self.up2 = Up3D(c * 8, c * 4, bilinear)
        self.up3 = Up3D(c * 4, c * 2, bilinear)
        self.up4 = Up3D(c * 2, c, bilinear)
        self.outc = OutConv3D(c, num_classes)

        if self.deep_supervision:
            self.ds1 = OutConv3D(c * 8, num_classes)
            self.ds2 = OutConv3D(c * 4, num_classes)
            self.ds3 = OutConv3D(c * 2, num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        logits = self.outc(d1)

        if self.training and self.deep_supervision:
            ds1 = F.interpolate(self.ds1(d4), size=input_shape, mode="trilinear", align_corners=True)
            ds2 = F.interpolate(self.ds2(d3), size=input_shape, mode="trilinear", align_corners=True)
            ds3 = F.interpolate(self.ds3(d2), size=input_shape, mode="trilinear", align_corners=True)
            return [logits, ds1, ds2, ds3]

        return logits


def create_model(in_channels=1, num_classes=3, base_channels=32, device="cuda", deep_supervision=True):
    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        bilinear=True,
        deep_supervision=deep_supervision,
    )
    return model.to(device)


def remove_small_components(mask, min_size):
    labels, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return mask.astype(np.uint8)

    component_sizes = np.bincount(labels.ravel())
    keep = component_sizes >= min_size
    keep[0] = False
    return keep[labels].astype(np.uint8)


def convert_probabilities_to_label_map(probs, threshold=0.5, min_size=10):
    seg = (probs > threshold).astype(np.uint8)

    if min_size > 0:
        for class_index in range(seg.shape[0]):
            if seg[class_index].sum() > 0:
                seg[class_index] = remove_small_components(seg[class_index], min_size)

    masked_probs = probs * seg
    label_map = masked_probs.argmax(axis=0).astype(np.uint8) + 1
    label_map[masked_probs.max(axis=0) <= 0] = 0
    return label_map
