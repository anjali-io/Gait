# model/network/triplet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss used in GaitSet.

    Supports:
    - hard triplet loss
    - full triplet loss

    Input:
        feature: (num_bin, batch_size, feature_dim)
        label  : (num_bin, batch_size)
    Output:
        full_loss_metric
        hard_loss_metric
        mean_dist
        full_loss_num
    """

    def __init__(self, batch_size, hard_or_full='full', margin=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.hard_or_full = hard_or_full
        self.margin = margin

    def forward(self, feature, label):
        """
        feature: (num_bin, batch_size, dim)
        label  : (num_bin, batch_size)
        """

        num_bin, batch_size, dim = feature.size()
        loss_full = []
        loss_hard = []
        mean_dist = []
        full_loss_num = []

        for i in range(num_bin):
            feat = feature[i]          # (B, D)
            lab = label[i]             # (B,)

            # pairwise distance matrix
            dist = torch.cdist(feat, feat, p=2)  # (B, B)
            mean_dist.append(dist.mean())

            # mask for positives and negatives
            is_pos = lab.unsqueeze(1) == lab.unsqueeze(0)
            is_neg = lab.unsqueeze(1) != lab.unsqueeze(0)

            pos_dist = dist[is_pos].view(batch_size, -1)
            neg_dist = dist[is_neg].view(batch_size, -1)

            # remove self-distance
            pos_dist = pos_dist[:, 1:]

            # -------- FULL TRIPLET --------
            ap = pos_dist.unsqueeze(2)    # (B, P, 1)
            an = neg_dist.unsqueeze(1)    # (B, 1, N)

            triplet_loss = F.relu(ap - an + self.margin)
            loss_full.append(triplet_loss.mean())
            full_loss_num.append((triplet_loss > 0).float().mean())

            # -------- HARD TRIPLET --------
            hardest_pos = pos_dist.max(dim=1)[0]
            hardest_neg = neg_dist.min(dim=1)[0]

            hard_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss_hard.append(hard_loss.mean())

        return (
            torch.stack(loss_full),
            torch.stack(loss_hard),
            torch.stack(mean_dist),
            torch.stack(full_loss_num)
        )
