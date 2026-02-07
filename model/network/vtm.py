# model/network/vtm.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewTransitionModule(nn.Module):
    """
    View Transition Module (VTM) for Cyclic View Learning.

    Properties:
    - Residual transition (prevents feature drift)
    - L2-normalized outputs (stable metric learning)
    - Supports iterative application for cycle consistency

    Input : (B, D)
    Output: (B, D)
    """

    def __init__(self, feat_dim=256, hidden_dim=512, scale=1.0):
        super().__init__()

        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, feat_dim)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale  # controls transition strength

        # ---- Initialization ----
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, feat):
        """
        Args:
            feat: (B, D) input feature

        Returns:
            next_feat: (B, D) transitioned feature
        """

        # Learn a delta (view transition)
        delta = self.fc2(self.relu(self.fc1(feat)))

        # Residual connection (CRITICAL)
        next_feat = feat + self.scale * delta

        # Normalize for metric stability
        next_feat = F.normalize(next_feat, p=2, dim=1)

        return next_feat

    def iterate(self, feat, steps=1):
        """
        Apply VTM repeatedly.

        Args:
            feat : (B, D) starting feature
            steps: number of transitions

        Returns:
            List of features [f1, f2, ..., f_steps]
        """

        outputs = []
        cur = feat

        for _ in range(steps):
            cur = self.forward(cur)
            outputs.append(cur)

        return outputs


