import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Basic Conv Block
# --------------------------------------------------
class ConvBlock(nn.Module):
    """
    Conv2D + BatchNorm + ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# --------------------------------------------------
# Set Pooling (core idea of GaitSet)
# --------------------------------------------------
class SetPooling(nn.Module):
    """
    Pooling over the temporal dimension.
    Input shape: (B, T, C, H, W)
    Output shape: (B, C, H, W)
    """
    def __init__(self, pool_type='max'):
        super().__init__()
        assert pool_type in ['max', 'mean']
        self.pool_type = pool_type

    def forward(self, x):
        if self.pool_type == 'max':
            return torch.max(x, dim=1)[0]
        else:
            return torch.mean(x, dim=1)


# --------------------------------------------------
# Horizontal Pooling Pyramid (HPP)
# --------------------------------------------------
class HorizontalPoolingPyramid(nn.Module):
    """
    Horizontal Pooling Pyramid used in GaitSet.
    Splits feature map horizontally into bins and pools each bin.
    """
    def __init__(self, bin_num=[1, 2, 4]):
        super().__init__()
        self.bin_num = bin_num

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, sum(bin_num), C)
        """
        B, C, H, W = x.size()
        features = []

        for b in self.bin_num:
            # NOTE: use reshape instead of view (safe for non-contiguous tensors)
            z = x.reshape(B, C, b, H // b, W)
            z = z.mean(dim=3)          # pool height
            z = z.mean(dim=3)          # pool width
            features.append(z)

        return torch.cat(features, dim=2).permute(0, 2, 1)


# --------------------------------------------------
# Fully Connected Block (for bin-wise embedding)
# --------------------------------------------------
class FCBlock(nn.Module):
    """
    Linear + BatchNorm
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        """
        x: (B, N, C)
        """
        B, N, C = x.size()

        # CRITICAL FIX: reshape instead of view
        x = x.reshape(B * N, C)
        x = self.bn(self.fc(x))
        return x.reshape(B, N, -1)

