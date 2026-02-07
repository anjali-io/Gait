import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_blocks import ConvBlock, SetPooling, HorizontalPoolingPyramid, FCBlock


class SetNet(nn.Module):
    """
    GaitSet backbone (Set-based CNN).

    Input:
        silho: list of tensors OR tensor
            shape after collate:
            (num_gpu, sum_frames, H, W) internally handled
    Output:
        feature: (B, num_bins, hidden_dim)
        None: placeholder for compatibility
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        # ----------- Frame-level feature extractor -----------
        self.conv1 = ConvBlock(1, 32, kernel_size=5, padding=2)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        # ----------- Set pooling (temporal) -----------
        self.set_pool = SetPooling(pool_type='max')

        # ----------- Horizontal Pooling Pyramid -----------
        self.hpp = HorizontalPoolingPyramid(bin_num=[1, 2, 4])

        # ----------- Bin-wise embedding -----------
        self.fc_bin = FCBlock(128, hidden_dim)

    def forward(self, silho, batch_frame=None):
        """
        silho:
            Tensor of shape (B, T, H, W) OR (sum_frames, H, W)
        batch_frame:
            frame count per GPU (used in multi-GPU mode)

        Returns:
            feature: (B, num_bins, hidden_dim)
            None
        """

        # ---- Input reshape ----
        if silho.dim() == 5:
            # (B, T, 1, H, W) → (B*T, 1, H, W)
            B, T, C, H, W = silho.size()
            x = silho.view(B * T, C, H, W)
        elif silho.dim() == 4:
            # (B, T, H, W)
            B, T, H, W = silho.size()
            x = silho.view(B * T, 1, H, W)
        else:
            raise ValueError("Unsupported silho shape")

        # ---- CNN backbone ----
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)

        # ---- Restore temporal dimension ----
        x = x.view(B, T, x.size(1), x.size(2), x.size(3))

        # ---- Set pooling ----
        x = self.set_pool(x)  # (B, C, H, W)

        # ---- Horizontal pooling pyramid ----
        x = self.hpp(x)       # (B, num_bins, C)

        # ---- Bin-wise FC ----
        feature = self.fc_bin(x)  # (B, num_bins, hidden_dim)

        return feature, None
