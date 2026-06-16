# model/network/cvl_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CVLLoss(nn.Module):
    """
    CVL Loss = Adjacency Loss + Cycle Consistency Loss

    This module assumes:
    - TripletLoss is computed separately in model.py
    - seq_feat and pred_next come from CVL_GaitSet
    """

    def __init__(self, lambda_adj=0.3, lambda_cycle=0.1):
        super().__init__()
        self.lambda_adj = lambda_adj
        self.lambda_cycle = lambda_cycle

    # --------------------------------------------------
    # 1) Adjacency loss (self-supervised, batch-based)
    # --------------------------------------------------
    def adjacency_loss(self, seq_feat, pred_next, labels):
        """
        seq_feat : (B, D) original sequence features
        pred_next: (B, D) predicted adjacent-view features
        labels   : (B,) identity labels

        For each sample, we pull pred_next toward
        another sample with SAME identity but DIFFERENT view.
        """

        B, D = seq_feat.size()
        loss = 0.0
        count = 0

        for i in range(B):
            same_id = (labels == labels[i]).nonzero(as_tuple=False).squeeze(1)
            if same_id.numel() <= 1:
                continue

            # pick a different sample of same ID
            j = same_id[torch.randint(0, same_id.numel(), (1,))].item()
            if j == i:
                continue

            loss += F.mse_loss(pred_next[i], seq_feat[j])
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=seq_feat.device)

        return loss / count

    # --------------------------------------------------
    # 2) Cycle consistency loss
    # --------------------------------------------------
    def cycle_loss(self, seq_feat, cycled_feat):
        """
        seq_feat   : (B, D) original features
        cycled_feat: (B, D) features after full cycle

        Enforces: f ≈ cycle(f)
        """
        return F.mse_loss(cycled_feat, seq_feat)

    # --------------------------------------------------
    # Combined forward
    # --------------------------------------------------
    def forward(self, cvl_out, labels, cycled_feat=None):
        """
        cvl_out: dict from CVL_GaitSet forward
            {
              'seq_feat': (B, D),
              'pred_next': (B, D),
              'view': ...
            }
        labels: (B,)
        cycled_feat: (B, D) output of full cycle (optional)

        Returns:
            total_loss, loss_dict
        """

        seq_feat = cvl_out['seq_feat']
        pred_next = cvl_out['pred_next']

        loss_adj = self.adjacency_loss(seq_feat, pred_next, labels)

        if cycled_feat is not None:
            loss_cycle = self.cycle_loss(seq_feat, cycled_feat)
        else:
            loss_cycle = torch.tensor(0.0, device=seq_feat.device)

        total = self.lambda_adj * loss_adj + self.lambda_cycle * loss_cycle

        info = {
            'loss_adj': float(loss_adj.item()),
            'loss_cycle': float(loss_cycle.item()),
            'loss_cvl': float(total.item())
        }

        return total, info

