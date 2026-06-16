# model/network/cvl_model.py

import torch
import torch.nn as nn

from .gaitset import SetNet
from .vtm import ViewTransitionModule


class CVL_GaitSet(nn.Module):
    """
    Cyclic View Learning wrapper over GaitSet (SetNet).

    - Uses SetNet as backbone (unchanged)
    - Pools sequence features to sequence-level embedding
    - Applies View Transition Module (VTM)
    - Exposes utilities for adjacency & cycle-consistency loss
    """

    def __init__(self, hidden_dim=256, vtm_hidden=512):
        super().__init__()

        # Backbone (original GaitSet)
        self.backbone = SetNet(hidden_dim)

        # View Transition Module
        self.vtm = ViewTransitionModule(
            feat_dim=hidden_dim,
            hidden_dim=vtm_hidden
        )

    def forward(self, silho, batch_frame=None, view=None):
        """
        Forward compatible with existing model.py

        Args:
            silho: silhouette input (as expected by SetNet)
            batch_frame: optional frame index tensor
            view: list or tensor of view labels (degrees)

        Returns:
            feature: raw SetNet feature (used for triplet loss)
            cvl_out: dict with CVL-related outputs
        """

        # ---- GaitSet forward ----
        feature, _ = self.backbone(silho, batch_frame)

        # ---- Sequence-level pooling ----
        # Common SetNet output: (B, L, D)
        if feature.dim() == 3:
            seq_feat = feature.mean(dim=1)   # (B, D)
        elif feature.dim() == 2:
            seq_feat = feature               # (B, D)
        else:
            # fallback (rare)
            seq_feat = feature.view(
                feature.size(0), feature.size(1), -1
            ).mean(dim=1)

        # ---- One-step view transition ----
        pred_next = self.vtm(seq_feat)        # (B, D)

        cvl_out = {
            'seq_feat': seq_feat,             # original embedding
            'pred_next': pred_next,           # adjacent-view prediction
            'view': view                      # raw view labels
        }

        return feature, cvl_out

    # ------------------------------------------------------------------
    # Utilities for CVL loss (used by cvl_loss.py)
    # ------------------------------------------------------------------

    def cycle_from(self, seq_feat, steps):
        """
        Iteratively apply VTM to simulate a view cycle.

        Args:
            seq_feat: (B, D) starting embedding
            steps: number of transitions

        Returns:
            List of embeddings after each transition
        """
        return self.vtm.iterate(seq_feat, steps=steps)

    def cycle_loss_inputs(self, seq_feat, steps):
        """
        Returns embeddings needed for cycle-consistency loss:
            original_feat vs cycled_back_feat

        Used for enforcing:
            A -> B -> C -> A
        """
        cycled_feats = self.vtm.iterate(seq_feat, steps=steps)
        cycled_back = cycled_feats[-1]
        return seq_feat, cycled_back



