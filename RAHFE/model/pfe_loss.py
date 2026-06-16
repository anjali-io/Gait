# encoding: utf-8
"""
Mutual Likelihood Score (MLS) Loss for Uncertainty-Propagated Metric Learning (UPML).

Implements a probabilistic triplet loss where embeddings are Gaussian distributions
(mu, sigma) and pairwise distances are computed using the MLS between distributions.

High-uncertainty sequences (large sigma, driven by low reliability scores) are
automatically downweighted: they contribute less to the loss because the large
sigma in the denominator damps their distance contribution. This directly connects
the RAHFE reliability estimator to the metric-learning objective.

This is the core loss function for the UPML Tier-1 contribution described in:
"Uncertainty-Propagated Metric Learning for Reliability-Aware Gait Recognition"

Reference: Adapted from Probabilistic Face Embeddings (Shi et al., ICCV 2019),
extended to multi-part gait descriptors with reliability-conditioned sigma scaling.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class MutualLikelihoodScoreLoss(BaseLoss):
    """
    Probabilistic Triplet Loss via Mutual Likelihood Score (MLS).

    Instead of comparing fixed embedding points with L2 distance,
    this loss treats each embedding as a Gaussian distribution N(mu, sigma^2)
    and computes the MLS between pairs:

        MLS(i,j) = -0.5 * sum[ (mu_i - mu_j)^2 / (sigma_i^2 + sigma_j^2)
                               + log(sigma_i^2 + sigma_j^2) ]

    A triplet loss is then applied using MLS-based distances:
        d_MLS(i,j) = -MLS(i,j)

    Args:
        margin (float): Triplet loss margin (default: 0.2).
        loss_term_weight (float): Weight of this loss term (default: 1.0).
    """

    def __init__(self, margin=0.2, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.margin = margin

    @staticmethod
    def compute_mls_matrix(mu, sigma):
        """
        Compute pairwise Mutual Likelihood Score (MLS) matrix.

        Args:
            mu    : [N, D, P] — mean embeddings (N samples, D dimensions, P parts)
            sigma : [N, D, P] — std embeddings (always positive)

        Returns:
            mls_mat : [N, N] — pairwise MLS, averaged over D and P.
                      Higher values = more similar (less distance).
        """
        eps = 1e-8

        # Expand for broadcasting: [N, 1, D, P] vs [1, N, D, P]
        mu_a    = mu.unsqueeze(1)            # [N, 1, D, P]
        mu_b    = mu.unsqueeze(0)            # [1, N, D, P]
        var_a   = sigma.pow(2).unsqueeze(1)  # [N, 1, D, P]
        var_b   = sigma.pow(2).unsqueeze(0)  # [1, N, D, P]

        var_sum = var_a + var_b + eps        # [N, N, D, P]  — always positive
        diff_sq = (mu_a - mu_b).pow(2)      # [N, N, D, P]

        # MLS per dimension per part: -0.5*(diff^2/var_sum + log(var_sum))
        mls = -0.5 * (diff_sq / var_sum + var_sum.log())  # [N, N, D, P]

        # Average over feature dimension D and parts P -> [N, N]
        return mls.mean(dim=2).mean(dim=-1)

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, sigma=None, **kwargs):
        """
        Compute the MLS-based triplet loss.

        Args:
            embeddings : [N, D, P] — mu (mean) embeddings
            labels     : [N]       — class labels
            sigma      : [N, D, P] — uncertainty (std) embeddings.
                         If None, falls back to standard L2-based triplet
                         with unit variance (reduces to standard triplet loss).
            **kwargs   : ignored extra keys from training_feat dict

        Returns:
            (loss, info): scalar loss and info dict
        """
        mu = embeddings.float()
        N  = mu.size(0)
        labels = labels.view(-1)

        # ── Compute pairwise MLS-based distance matrix ────────────────────────
        if sigma is None:
            # Fallback: unit sigma → MLS reduces to a scaled L2 distance
            sigma = torch.ones_like(mu)

        sigma = sigma.float().clamp(min=1e-6)

        # [N, N] — higher MLS = more similar
        mls_mat  = self.compute_mls_matrix(mu, sigma)

        # Convert to distance: d = -MLS (lower = more similar)
        dist_mat = -mls_mat    # [N, N]

        # ── Build positive/negative masks ─────────────────────────────────────
        eq_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1))  # [N, N]
        pos_mask = eq_mask.clone()
        pos_mask.fill_diagonal_(False)                            # exclude self
        neg_mask = ~eq_mask

        # ── Hardest-pair triplet mining ───────────────────────────────────────
        # per part P: we already averaged P inside compute_mls_matrix,
        # so this is a scalar [N, N] dist matrix → mine globally
        loss_total = torch.tensor(0.0, device=mu.device)
        valid_count = 0

        for i in range(N):
            pos_d = dist_mat[i][pos_mask[i]]   # distances to same-class
            neg_d = dist_mat[i][neg_mask[i]]   # distances to diff-class

            if pos_d.numel() == 0 or neg_d.numel() == 0:
                continue

            hardest_pos = pos_d.max()           # furthest positive
            hardest_neg = neg_d.min()           # closest negative

            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss_total   = loss_total + triplet_loss
            valid_count  += 1

        if valid_count == 0:
            dummy = torch.tensor(0.0, device=mu.device, requires_grad=True)
            self.info.update({'loss': dummy.detach(), 'loss_num': 0})
            return dummy, self.info

        loss_avg = loss_total / valid_count

        # ── Uncertainty regularization: penalise sigma collapsing to 0 ────────
        # Without this, the model could cheat by making sigma huge everywhere.
        # We lightly penalise the average log-sigma to keep it in check.
        sigma_reg = -0.1 * sigma.log().mean()    # positive when sigma < 1
        loss_final = loss_avg + sigma_reg.clamp(min=0)

        self.info.update({
            'loss':      loss_final.detach().clone(),
            'loss_num':  torch.tensor(float(valid_count), device=mu.device),
            'mean_dist': dist_mat.mean().detach().clone(),
        })

        return loss_final, self.info
