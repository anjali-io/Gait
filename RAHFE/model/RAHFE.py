# encoding: utf-8
"""RAHFE: Reliability-Aware Hierarchical Feature Extraction
Paper: RAHFE_Methodology_Paper.docx
Built on top of SeeGait (SeeGait_for_sustech1k.py).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math, numpy as np, warnings
from collections import OrderedDict
from functools import partial
from einops import rearrange
from ..base_model import BaseModel
from ..modules import (SeparateBNNecks, SeparateFCs, SetBlockWrapper,
                       HorizontalPoolingPyramid, PackSequenceWrapper,
                       conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D,
                       BasicBlock3D, BasicConv3d)
from .SeeGait_for_sustech1k import (
    GeMHPP, MLP, Attention, SpatioBlock, TemporalBlock,
    CrossModalAttention, AdaptiveFeatureFusion,
    Sil_STP, Pose_STP, trunc_normal_, UnitConv2D, blocks_map
)

# ── COCO-17 body region assignments (B=5) ───────────────────────────────────
# 0:head  1:upper_body  2:lower_body  3:left_limbs  4:right_limbs
JOINT_TO_REGION = torch.tensor(
    [0, 0, 0, 0, 0,          # 0-4  head/face
     1, 1, 3, 4, 3, 4,       # 5-10 shoulders, elbows, wrists
     2, 2, 3, 4, 3, 4],      # 11-16 hips, knees, ankles
    dtype=torch.long)

# Limb pairs for joint-reliability regularisation
LIMB_PAIRS = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16)]

NUM_BODY_REGIONS = 5

# ── Frame Reliability Estimator ──────────────────────────────────────────────
class FrameReliabilityEstimator(nn.Module):
    """phi^F = [density, contour_irr, temp_diff, hole_ratio, cnn_norm] -> r^F
    MLP^F: 5 -> 16 -> 1, Sigmoid output."""
    def __init__(self, d_in=5, hidden=16, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid())

    @staticmethod
    def compute_features(sils: torch.Tensor) -> tuple:
        """Compute phi^F and delta_t from silhouette tensor.
        Args:
            sils: [N, S, H, W] binary silhouettes in [0,1]
        Returns:
            phi  : [N, S, 5]
            delta: [N, S]  inter-frame L1 diff (delta[:,0]=0)
        """
        N, S, H, W = sils.shape
        HW = float(H * W)
        # density
        density = sils.view(N, S, -1).mean(-1)           # [N,S]
        # horizontal projection profile variance (contour irregularity)
        proj = sils.sum(-1).float() / W                   # [N,S,H]
        contour_irr = proj.var(-1) / (proj.mean(-1).clamp(min=1e-6))
        # temporal diff
        padded = torch.cat([sils[:, :1], sils], dim=1)    # [N,S+1,H,W]
        delta = (sils - padded[:, :-1]).abs().view(N, S, -1).mean(-1)
        # hole ratio approximation: 1 - ratio of filled rows
        row_filled = (sils.sum(-1) > 0).float().mean(-1)  # [N,S]
        hole_ratio = 1.0 - row_filled
        # cnn_norm placeholder (filled in during forward with actual feat norm)
        cnn_norm = torch.zeros_like(density)
        phi = torch.stack([density, contour_irr, delta, hole_ratio, cnn_norm], dim=-1)
        return phi, delta

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """phi: [N,S,5] -> r_F: [N,S]"""
        return self.mlp(phi).squeeze(-1)


# ── Joint Reliability Estimator ──────────────────────────────────────────────
class JointReliabilityEstimator(nn.Module):
    """phi^J = [conf, entropy_proxy, anat_dist, velocity] -> r^J
    Shared-weight MLP^J applied per joint: 4 -> 16 -> 1, Sigmoid."""
    def __init__(self, d_in=4, hidden=16, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1), nn.Sigmoid())

    @staticmethod
    def compute_features(pose: torch.Tensor) -> torch.Tensor:
        """Extract phi^J from raw pose tensor.
        Args:
            pose: [N,S,J,C]  C>=4, channels: 0-2 xyz, 3 confidence, 4-6 vel
        Returns:
            phi_J: [N,S,J,4]
        """
        conf  = pose[..., 3].clamp(0, 1)                  # [N,S,J]
        # entropy proxy from confidence: H ~ -c*log(c+eps)
        eps = 1e-6
        entropy = -(conf * (conf + eps).log()).clamp(min=0) # [N,S,J]
        # velocity: use channels 4-6 if available, else compute from coords
        if pose.shape[-1] >= 7:
            vel = pose[..., 4:7].norm(dim=-1)
        else:
            coords = pose[..., :3]
            diff = coords[:, 1:] - coords[:, :-1]          # [N,S-1,J,3]
            vel_core = diff.norm(dim=-1)                    # [N,S-1,J]
            vel = torch.cat([vel_core[:, :1], vel_core], dim=1)
        # anatomical distance: deviation of each joint from mean skeleton
        coords = pose[..., :3]
        mean_pose = coords.mean(dim=(1, 2), keepdim=True)
        anat_dist = (coords - mean_pose).norm(dim=-1)
        anat_dist = anat_dist / (anat_dist.max() + eps)
        phi = torch.stack([conf, entropy, anat_dist, vel.clamp(max=5)/5.0], dim=-1)
        return phi

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """phi: [N,S,J,4] -> r_J: [N,S,J]"""
        return self.mlp(phi).squeeze(-1)


# ── Body Reliability Estimator ───────────────────────────────────────────────
class BodyReliabilityEstimator(nn.Module):
    """Estimates B=5 body-region reliability from silhouette stripe coverage
    and skeleton joint availability per region.
    MLP^B: 10 -> 32 -> 5, Sigmoid output."""
    def __init__(self, B=5, hidden=32, dropout=0.1):
        super().__init__()
        self.B = B
        # Input: B sil-coverage features + B skeleton-availability features
        self.mlp = nn.Sequential(
            nn.Linear(B * 2, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, B), nn.Sigmoid())
        j2r = JOINT_TO_REGION  # [J]
        self.register_buffer('joint_to_region', j2r)

    def compute_features(self, sils: torch.Tensor, r_J: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sils: [N,S,H,W] binary silhouettes
            r_J : [N,S,J] joint reliability
        Returns:
            phi_B: [N,S, B*2]
        """
        N, S, H, W = sils.shape
        B = self.B
        # Silhouette coverage per region: divide H into B equal stripes
        stripe_h = H // B
        sil_cov = []
        for b in range(B):
            h0, h1 = b * stripe_h, (b + 1) * stripe_h if b < B-1 else H
            stripe = sils[..., h0:h1, :]          # [N,S,stripe_h,W]
            cov = stripe.float().mean(dim=(-2, -1))
            sil_cov.append(cov)
        sil_cov = torch.stack(sil_cov, dim=-1)    # [N,S,B]

        # Skeleton availability per region: mean r_J of joints in each region
        j2r = self.joint_to_region                # [J]
        skel_avail = torch.zeros(N, S, B, device=r_J.device)
        counts = torch.zeros(B, device=r_J.device)
        for b in range(B):
            mask = (j2r == b)
            if mask.any():
                skel_avail[..., b] = r_J[..., mask].mean(-1)
                counts[b] = mask.float().sum()

        phi_B = torch.cat([sil_cov, skel_avail], dim=-1)  # [N,S,B*2]
        return phi_B

    def forward(self, sils: torch.Tensor, r_J: torch.Tensor) -> torch.Tensor:
        """-> r_B: [N,S,B]"""
        phi_B = self.compute_features(sils, r_J)
        # Average over S for a sequence-level body reliability
        phi_avg = phi_B.mean(dim=1)               # [N, B*2]
        r_B_seq = self.mlp(phi_avg)               # [N, B]
        # Expand to [N,S,B]
        r_B = r_B_seq.unsqueeze(1).expand(-1, sils.shape[1], -1)
        return r_B


# ── Modality Reliability Estimator ───────────────────────────────────────────
class ModalityReliabilityEstimator(nn.Module):
    """Produces r^M in [N,2] from silhouette + pose quality summary features.
    MLP^M: (d_sil + d_pose) -> 32 -> 2, Sigmoid."""
    def __init__(self, d_sil=5, d_pose=4, hidden=32, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_sil + d_pose, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2), nn.Sigmoid())

    def forward(self, phi_sil: torch.Tensor, phi_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi_sil : [N, d_sil]  summary of frame-level sil quality
            phi_pose: [N, d_pose] summary of joint-level pose quality
        Returns:
            r_M: [N,2]  [:,0]=sil, [:,1]=pose
        """
        phi = torch.cat([phi_sil, phi_pose], dim=-1)
        return self.mlp(phi)


# ── Weighted Temporal Pooling ────────────────────────────────────────────────
class WeightedTemporalPool(nn.Module):
    """Replaces SeeGait's max-pool over time with soft reliability-weighted mean.
    F^(sil,R)_k = sum_t(r^F_t * F_k(:,:,t,:,:)) / sum_t(r^F_t)
    """
    def forward(self, feat: torch.Tensor, r_F: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [N,C,S,H,W]
            r_F : [N,S]
        Returns:
            pooled: [N,C,H,W]
        """
        w = r_F[:, :, None, None, None]           # [N,S,1,1,1]
        w = w.transpose(1, 2)                      # [N,1,S,1,1]  -- matches feat dim ordering
        # make sure w broadcasts over C,H,W
        num   = (feat * w).sum(dim=2)              # [N,C,H,W]
        denom = w.sum(dim=2).clamp(min=1e-6)       # [N,1,H,W]
        return num / denom


# ── Body-Weighted Part Pooling ───────────────────────────────────────────────
class BodyWeightedPool(nn.Module):
    """Reweights horizontal-stripe part features by body-region reliability.
    f^(sil,RB)_k  =  f^(sil,R)_k * w^B   (broadcast over parts mapped to each region)

    Args:
        P  : number of horizontal parts (e.g. 32 for HPP with bin_num=[32])
        B  : number of body regions (5)
    """
    def __init__(self, P=32, B=5):
        super().__init__()
        self.P = P
        self.B = B
        # Assign each horizontal part to a body region (top→head, bottom→legs)
        # Simple uniform split: parts 0..P//B-1 → head, etc.
        part_to_region = torch.zeros(P, dtype=torch.long)
        seg = P // B
        for b in range(B):
            lo = b * seg
            hi = (b + 1) * seg if b < B - 1 else P
            part_to_region[lo:hi] = b
        self.register_buffer('part_to_region', part_to_region)

    def forward(self, feat: torch.Tensor, r_B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [N, C, P]  part features
            r_B : [N, B]     body-region reliability (sequence-averaged)
        Returns:
            weighted: [N, C, P]
        """
        # gather region weight for each part: [N, P]
        p2r = self.part_to_region                 # [P]
        w = r_B[:, p2r]                           # [N, P]
        w = w.unsqueeze(1)                        # [N,1,P]
        return feat * w


# ── Reliability-Biased Spatial Attention ─────────────────────────────────────
class ReliabilityBiasedSpatioBlock(nn.Module):
    """SpatioBlock variant that adds log(r^J_j) as an additive attention bias.
    Used in HSTE_R to implement joint-reliability-weighted spatial attention.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.norm1 = norm_layer(dim)
        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj  = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        mlp_hidden = int(dim * mlp_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp   = MLP(in_features=dim, hidden_features=mlp_hidden,
                         out_features=dim, drop=drop)

    def forward(self, x: torch.Tensor, r_J: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x  : [BF, J, C]  (BF = batch*frames)
            r_J: [BF, J]     joint reliability scores
        Returns:
            x  : [BF, J, C]
        """
        BF, J, C = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(BF, J, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)         # [3,BF,H,J,d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [BF,H,J,J]

        # Additive reliability bias: log(r^J_j) broadcast over queries
        # Shape: [BF, J] -> [BF,1,1,J] (key dimension)
        eps = 1e-6
        bias = torch.log(r_J.clamp(min=eps)).unsqueeze(1).unsqueeze(2)  # [BF,1,1,J]
        attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(BF, J, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = residual + x
        x = x + self.mlp(self.norm2(x))
        return x


# ── HSTE with Joint Reliability (HSTE_R) ─────────────────────────────────────
class HSTE_R(nn.Module):
    """HSTE variant where SpatioBlock uses joint-reliability attention bias.
    Replaces uniform spatial self-attention with ReliabilityBiasedSpatioBlock.
    Architecture mirrors HSTE (depth=6, dim_feat=256, att_fuse=True).
    """
    def __init__(self, dim_in=10, dim_feat=256, dim_rep=256, depth=6,
                 num_heads=8, mlp_ratio=2, num_joints=17, maxlen=720,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True, t_kernel_size=9):
        super().__init__()
        self.dim_feat  = dim_feat
        self.depth     = depth
        self.att_fuse  = att_fuse
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop  = nn.Dropout(p=drop_rate)
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed,  std=.02)
        # Reliability-biased spatio blocks (one per depth level)
        self.blocks_s = nn.ModuleList([
            ReliabilityBiasedSpatioBlock(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for _ in range(depth)])
        # Standard temporal blocks (unchanged)
        self.blocks_t = nn.ModuleList([
            TemporalBlock(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer, st_mode="stage_ts")
            for _ in range(depth)])
        self.norm = norm_layer(dim_feat)
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)), ('act', nn.Tanh())]))
        self.tcn = UnitConv2D(D_in=dim_feat, D_out=dim_feat,
                              kernel_size=t_kernel_size)
        self.residual_s = nn.Sequential(
            nn.Conv2d(dim_feat, dim_feat, 1), nn.BatchNorm2d(dim_feat))
        if self.att_fuse:
            self.ts_attn = nn.ModuleList(
                [nn.Linear(dim_feat * 2, 2) for _ in range(depth)])
            for m in self.ts_attn:
                m.weight.data.fill_(0); m.bias.data.fill_(0.5)

    def forward(self, x: torch.Tensor, r_J: torch.Tensor,
                return_intermediate=False):
        """
        Args:
            x  : [B, F, J, C_in]
            r_J: [B, F, J]  joint reliability
        Returns:
            x  : [B, F, J, dim_rep]
            intermediates (optional): list of [B,F,J,dim_feat]
        """
        B, F, J, Cin = x.shape
        x = x.reshape(-1, J, Cin)
        BF = x.shape[0]
        x = self.joints_embed(x)
        x = x + self.pos_embed
        _, J2, C = x.shape
        x = x.reshape(B, F, J2, C) + self.temp_embed[:, :F]
        # TCN temporal processing
        x_tcn = x.permute(0, 3, 1, 2).contiguous()       # [B,C,F,J]
        x_tcn = self.tcn(x_tcn) + self.residual_s(x_tcn)
        x = x_tcn.permute(0, 2, 3, 1).reshape(BF, J2, C)
        x = self.pos_drop(x)
        # reliability: [B,F,J] -> [BF,J]
        r_J_flat = r_J.reshape(BF, J2)
        intermediates = []
        for idx, (blk_s, blk_t) in enumerate(zip(self.blocks_s, self.blocks_t)):
            x_st = blk_s(x, r_J_flat)                     # reliability-biased spatial
            x_ts = blk_t(x, F)                            # standard temporal
            if self.att_fuse:
                alpha = self.ts_attn[idx](
                    torch.cat([x_st, x_ts], dim=-1)).softmax(dim=-1)
                x = x_st * alpha[..., 0:1] + x_ts * alpha[..., 1:2]
            else:
                x = (x_st + x_ts) * 0.5
            if return_intermediate:
                intermediates.append(self.norm(x).reshape(B, F, J2, -1))
        x = self.norm(x).reshape(B, F, J2, -1)
        x = self.pre_logits(x)
        if return_intermediate:
            return x, intermediates
        return x


# ── Reliability-Gated Cross-Modal Fusion (RGCMF) ─────────────────────────────
class ReliabilityGatedCrossModalFusion(nn.Module):
    """RGCMF: Bidirectional cross-attention with reliability bias + modality gating.

    At each hierarchical level l:
      A_{sil->pose} = Softmax(Q_sil K_pose^T / sqrt(d) + Bias_R(r^B, r^J))
      A_{pose->sil} = Softmax(Q_pose K_sil^T  / sqrt(d) + Bias_R(r^B, r^J))
      Bias_R_{p,j}  = log(r^B_{b(p)}) + log(r^J_j)
      f_fused = r~^M_sil*(f_sil + A_{pose->sil}*V_pose)
               + r~^M_pose*(f_pose + A_{sil->pose}*V_sil)
    """
    def __init__(self, sil_ch: int, pose_ch: int, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        # Projectors
        self.sil_proj  = nn.Conv1d(sil_ch,  embed_dim, 1)
        self.pose_proj = nn.Conv1d(pose_ch, embed_dim, 1)
        # QKV for silhouette and pose
        self.q_sil  = nn.Linear(embed_dim, embed_dim)
        self.k_sil  = nn.Linear(embed_dim, embed_dim)
        self.v_sil  = nn.Linear(embed_dim, embed_dim)
        self.q_pose = nn.Linear(embed_dim, embed_dim)
        self.k_pose = nn.Linear(embed_dim, embed_dim)
        self.v_pose = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm  = nn.LayerNorm(embed_dim)
        # Modality gate normaliser: normalise r^M to sum=1
        self.register_buffer('_dummy', torch.zeros(1))

    @staticmethod
    def _make_bias(r_B: torch.Tensor, r_J: torch.Tensor,
                   P: int, J: int, B: int) -> torch.Tensor:
        """Bias_R_{p,j} = log(r^B_{b(p)}) + log(r^J_j)
        Args:
            r_B: [N, B]
            r_J: [N, J]
            P  : number of silhouette parts
            J  : number of joints
        Returns:
            bias: [N, P, J]
        """
        eps = 1e-6
        seg = P // B
        part_regions = torch.zeros(P, dtype=torch.long, device=r_B.device)
        for b in range(B):
            lo = b * seg; hi = (b+1)*seg if b < B-1 else P
            part_regions[lo:hi] = b
        log_rB_per_part = torch.log(r_B.clamp(min=eps))[:, part_regions]  # [N,P]
        log_rJ = torch.log(r_J.clamp(min=eps))                            # [N,J]
        bias = log_rB_per_part.unsqueeze(-1) + log_rJ.unsqueeze(-2)       # [N,P,J]
        return bias

    def forward(self, f_sil: torch.Tensor, f_pose: torch.Tensor,
                r_B: torch.Tensor, r_J: torch.Tensor,
                r_M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_sil : [N, sil_ch,  P]   silhouette part features
            f_pose: [N, pose_ch, 1]   pose summary features
            r_B   : [N, B]            body reliability
            r_J   : [N, J]            joint reliability (sequence-avg)
            r_M   : [N, 2]            modality reliability [sil, pose]
        Returns:
            fused : [N, embed_dim, 1]
        """
        N = f_sil.shape[0]
        # Project to common embedding space
        fs = self.sil_proj(f_sil).permute(0, 2, 1)    # [N, P, D]
        fp = self.pose_proj(f_pose).permute(0, 2, 1)   # [N, 1, D]
        P, D = fs.shape[1], fs.shape[2]
        J_r = r_J.shape[-1]
        B_r = r_B.shape[-1]

        # Expand pose to P tokens for cross-attention (tile)
        fp_exp = fp.expand(-1, P, -1)                  # [N,P,D]

        # Reliability bias [N, P, J] clamped, averaged over J for sil<->pose
        bias = self._make_bias(r_B, r_J, P, J_r, B_r)  # [N,P,J]
        bias_scalar = bias.mean(-1, keepdim=True)       # [N,P,1]

        # ── sil attends to pose ──
        Qs = self.q_sil(fs)                             # [N,P,D]
        Kp = self.k_pose(fp_exp)
        Vp = self.v_pose(fp_exp)
        attn_sp = (Qs @ Kp.transpose(-2,-1)) * self.scale + bias_scalar
        attn_sp = attn_sp.softmax(-1)
        out_sp  = attn_sp @ Vp                          # [N,P,D]

        # ── pose attends to sil ──
        Qp = self.q_pose(fp_exp)
        Ks = self.k_sil(fs)
        Vs = self.v_sil(fs)
        attn_ps = (Qp @ Ks.transpose(-2,-1)) * self.scale + bias_scalar.transpose(-2,-1)
        attn_ps = attn_ps.softmax(-1)
        out_ps  = attn_ps @ Vs                          # [N,P,D]

        # ── modality gating ──
        r_sil_n  = r_M[:, 0] / (r_M[:, 0] + r_M[:, 1] + 1e-6)   # [N]
        r_pose_n = r_M[:, 1] / (r_M[:, 0] + r_M[:, 1] + 1e-6)    # [N]
        w_sil  = r_sil_n[:, None, None]                # [N,1,1]
        w_pose = r_pose_n[:, None, None]

        fused = w_sil * (fs + out_ps) + w_pose * (fp_exp + out_sp)  # [N,P,D]
        fused = self.norm(self.out_proj(fused))                      # [N,P,D]
        # Pool to [N,D,1]
        fused = fused.mean(dim=1, keepdim=True).permute(0, 2, 1)    # [N,D,1]
        return fused


# ── Hierarchical SeeGait with Cross-Attention Synergy (for BiHCASM2) ─────────
class HierarchicalCrossAttentionSynergy(nn.Module):
    """Reused from SeeGait for BiHCASM2 (final-level pair). Identical to original."""
    def __init__(self, sil_channels, pos_channels, embed_dim=256,
                 num_heads=8, attn_drop=0.1):
        super().__init__()
        self.sil_proj = nn.ModuleList(
            [nn.Conv1d(c, embed_dim, 1) for c in sil_channels])
        self.pos_proj = nn.ModuleList(
            [nn.Conv1d(c, embed_dim, 1) for c in pos_channels])
        self.cross_attns = nn.ModuleList(
            [CrossModalAttention(embed_dim, num_heads, attn_drop=attn_drop)
             for _ in range(len(sil_channels))])
        self.adapt_fuse = nn.ModuleList(
            [AdaptiveFeatureFusion(embed_dim) for _ in range(len(sil_channels))])
        self.scale_fusion = nn.Sequential(
            nn.Conv1d(embed_dim * len(sil_channels), embed_dim, 1),
            nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True))
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(embed_dim * 2, embed_dim))

    def forward(self, sil_features, pos_features):
        sp = [p(f) for f, p in zip(sil_features, self.sil_proj)]
        pp = [p(f) for f, p in zip(pos_features, self.pos_proj)]
        fused = []
        for i, (sf, pf, ca, af) in enumerate(
                zip(sp, pp, self.cross_attns, self.adapt_fuse)):
            out = ca(sf.permute(0,2,1), pf.permute(0,2,1)).permute(0,2,1)
            fused.append(af(out))
        cat = torch.cat(fused, dim=1)
        out = self.scale_fusion(cat).squeeze(-1)
        out = self.final_proj(out)
        return out.unsqueeze(-1), sp, pp


# ── RAHFE Top-Level Model ────────────────────────────────────────────────────
class RAHFE(BaseModel):
    """Reliability-Aware Hierarchical Feature Extraction.

    Integrates four-level hierarchical reliability estimation into SeeGait:
      Level 1 (Frame) : reliability-weighted temporal pooling
      Level 2 (Joint) : reliability-biased spatial attention in HSTE
      Level 3 (Body)  : body-region reliability reweights part features
      Level 4 (Modality): dynamic modality trust gates cross-modal fusion

    The three-level RGCMF replaces SeeGait's BiHCASM1.
    BiHCASM2 (final level) is kept identical to SeeGait.
    """

    def build_network(self, model_cfg):
        # ── Reliability config ────────────────────────────────────────────────
        self.beta_F   = model_cfg.get('beta_F',   0.1)
        self.beta_J   = model_cfg.get('beta_J',   0.05)
        self.beta_M   = model_cfg.get('beta_M',   0.2)
        self.alpha_F  = model_cfg.get('alpha_F',  1.0)
        self.tau_M    = model_cfg.get('tau_M',    0.5)
        self.num_body = NUM_BODY_REGIONS            # 5

        # ── Reliability Estimators ────────────────────────────────────────────
        self.frame_rel  = FrameReliabilityEstimator(d_in=5,  hidden=16)
        self.joint_rel  = JointReliabilityEstimator(d_in=4,  hidden=16)
        self.body_rel   = BodyReliabilityEstimator(B=self.num_body, hidden=32)
        self.modal_rel  = ModalityReliabilityEstimator(d_sil=5, d_pose=4, hidden=32)

        # ── Temporal / Part Pooling ───────────────────────────────────────────
        self.weighted_tp = WeightedTemporalPool()
        self.body_wp     = BodyWeightedPool(P=32, B=self.num_body)
        self.sil_stp     = Sil_STP()
        self.pose_stp    = Pose_STP()
        self.HPP         = HorizontalPoolingPyramid(bin_num=[32])
        self.GeMHPP_pool = GeMHPP(bin_num=[1])

        # ── Silhouette Backbone (identical to SeeGait) ────────────────────────
        self.t_kernel_size = model_cfg['t_kernel_size']
        mode   = model_cfg['Backbone']['mode']
        assert mode in blocks_map
        block  = blocks_map[mode]
        in_ch  = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        self.inference_use_emb2 = model_cfg.get('use_emb2', False)

        if mode == '3d':
            strides = [[1,1],[1,1,1],[1,2,2],[1,1,1]]
        else:
            strides = [[1,1],[1,1],[2,2],[1,1]]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_ch, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True)))
        self.layer1 = SetBlockWrapper(
            self._make_layer(BasicBlock2D, channels[0], strides[0], layers[0], mode))
        self.layer2 = self._make_layer(block, channels[1], strides[1], layers[1], mode)
        self.layer3 = self._make_layer(block, channels[2], strides[2], layers[2], mode)
        self.layer4 = self._make_layer(block, channels[3], strides[3], layers[3], mode)
        if mode == '2d':
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        # ── Skeleton Backbone (HSTE_R) ────────────────────────────────────────
        self.hste_r = HSTE_R(
            dim_in=10, dim_feat=256, dim_rep=256, depth=6, num_heads=8,
            mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=720, num_joints=17,
            t_kernel_size=self.t_kernel_size, att_fuse=True)

        # ── RGCMF: Three hierarchical levels ─────────────────────────────────
        # Level 1: sil=channels[0], pose=256ch
        self.rgcmf1 = ReliabilityGatedCrossModalFusion(channels[0], 256, embed_dim=256)
        # Level 2: sil=channels[1], pose=256ch
        self.rgcmf2 = ReliabilityGatedCrossModalFusion(channels[1], 256, embed_dim=256)
        # Level 3: sil=channels[2], pose=256ch
        self.rgcmf3 = ReliabilityGatedCrossModalFusion(channels[2], 256, embed_dim=256)

        # ── BiHCASM2 (final level, identical to SeeGait) ──────────────────────
        self.BiHCASM2 = HierarchicalCrossAttentionSynergy(
            sil_channels=[channels[3]], pos_channels=[256], embed_dim=256)

        # ── Classification head (41 parts = 32 sil + 1 pose + 1 final + 3 rgcmf + 4 aux) ──
        # 32 sil + 1 pose + 1 BiHCASM2 + 3 RGCMF + 3 sil_proj + 1 = 41  (same as SeeGait)
        class_num = model_cfg['SeparateBNNecks']['class_num']
        self.FCs_sil  = SeparateFCs(32, channels[3], channels[2])
        self.FCs_fuse = SeparateFCs(parts_num=41, in_channels=256, out_channels=256)
        self.BNNecks  = SeparateBNNecks(41, channels[2], class_num=class_num)

        # ── Uncertainty Head (UPML — Uncertainty-Propagated Metric Learning) ──
        # Produces log-sigma for each of the 41 parts in the 256-dim space.
        # Sigma is then scaled by the inverse of the per-sample reliability so
        # that unreliable sequences automatically receive higher uncertainty.
        self.sigma_head = SeparateFCs(parts_num=41, in_channels=256, out_channels=256)

    # ── Backbone helpers ──────────────────────────────────────────────────────
    def _make_layer(self, block, planes, stride, blocks_num, mode='2d'):
        from ..modules import BasicBlock2D as BB2D
        from .SeeGait_for_sustech1k import blocks_map as bm
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              [1,1,1], stride, [0,0,0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            elif mode == '2d':
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion))
            elif mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              [1,1,1], [1,*stride], [0,0,0], bias=False),
                    nn.BatchNorm3d(planes * block.expansion))
            else:
                raise TypeError(mode)
        else:
            downsample = lambda x: x
        lyr = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        s = [1,1] if mode in ['2d','p3d'] else [1,1,1]
        for _ in range(1, blocks_num):
            lyr.append(block(self.inplanes, planes, stride=s))
        return nn.Sequential(*lyr)


    # ── Forward Pass ─────────────────────────────────────────────────────────
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[-1]                                    # [N,S,H,W]
        if len(ipts) > 1:
            pose = ipts[0]                                 # [N,S,J,C,1] or [N,S,J,C]
        else:
            pose = torch.zeros(sils.size(0), sils.size(1), 10, 17, 1, device=sils.device)
        x = sils.unsqueeze(1).contiguous()                 # [N,1,S,H,W]
        n, _, s, h, w = x.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            x = x.repeat(1, 1, repeat, 1, 1)
            pose_repeat_args = [1, repeat] + [1] * (pose.dim() - 2)
            pose = pose.repeat(*pose_repeat_args)
            s = x.shape[2]

        # ── Step 1: Frame Reliability ──────────────────────────────────────
        sils_f = x.squeeze(1)                              # [N,S,H,W]
        phi_F, delta_t = FrameReliabilityEstimator.compute_features(sils_f)
        phi_F = phi_F.to(x.device)
        delta_t = delta_t.to(x.device)

        # ── Step 2: Joint Reliability ──────────────────────────────────────
        pose_sq = pose.squeeze(-1) if pose.dim() == 5 else pose  # [N,S,J,C]
        pose_in = pose_sq.permute(0, 1, 3, 2).contiguous()       # [N,S,17,10]
        phi_J = JointReliabilityEstimator.compute_features(pose_in)
        r_J = self.joint_rel(phi_J.to(x.device))                 # [N,S,J]

        # ── Step 3: Frame Reliability scores ──────────────────────────────
        # Inject CNN norm placeholder (will update after backbone layer1)
        r_F_pre = self.frame_rel(phi_F)                           # [N,S]

        # ── Step 4: Silhouette Backbone ────────────────────────────────────
        out0 = self.layer0(x)
        out1 = self.layer1(out0)
        # Update phi_F cnn_norm with actual layer1 norm
        feat_norm = out1.norm(dim=1, keepdim=False)               # [N,S,H',W']
        feat_norm_scalar = feat_norm.view(n, s, -1).mean(-1)      # [N,S]
        phi_F[..., 4] = feat_norm_scalar.cpu()
        r_F = self.frame_rel(phi_F.to(x.device))                  # [N,S]

        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # ── Step 5: Body Reliability ───────────────────────────────────────
        r_B_full = self.body_rel(sils_f, r_J)                    # [N,S,B]
        r_B = r_B_full.mean(dim=1)                               # [N,B] seq-avg

        # ── Step 6: Frame-weighted temporal pooling ────────────────────────
        # out4: [N,C,S,H',W'] after layer4
        pooled_4 = self.weighted_tp(out4, r_F)                   # [N,C,H',W']
        embed_sil_raw = self.HPP(pooled_4)                       # [N,C,32]
        embed_sil = self.FCs_sil(embed_sil_raw)                  # [N,256,32]

        # Apply body-weighted part reweighting
        embed_sil = self.body_wp(embed_sil, r_B)                 # [N,256,32]

        # Intermediate sil features (frame-weighted then GeMHPP into 1 part)
        sil_outs0 = self.GeMHPP_pool(self.weighted_tp(out1, r_F))  # [N,64,1]
        sil_outs1 = self.GeMHPP_pool(self.weighted_tp(out2, r_F))  # [N,128,1]
        sil_outs2 = self.GeMHPP_pool(self.weighted_tp(out3, r_F))  # [N,256,1]

        # ── Step 7: Skeleton Hierarchy via HSTE_R ─────────────────────────
        feat_final, inter = self.hste_r(pose_in, r_J, return_intermediate=True)
        pos_stp0 = self.pose_stp(inter[2].permute(0,3,1,2))      # [N,256,1]
        pos_stp1 = self.pose_stp(inter[3].permute(0,3,1,2))
        pos_stp2 = self.pose_stp(inter[4].permute(0,3,1,2))
        pos_stp3 = self.pose_stp(feat_final.permute(0,3,1,2))    # [N,256,1]

        # ── Step 8: Modality Reliability ──────────────────────────────────
        phi_sil_summary  = phi_F.mean(dim=1).to(x.device)        # [N,5]
        phi_pose_summary = phi_J.mean(dim=(1,2)).to(x.device)    # [N,4]
        r_M = self.modal_rel(phi_sil_summary, phi_pose_summary)  # [N,2]

        # r_J sequence-averaged for RGCMF bias: [N,J]
        r_J_avg = r_J.mean(dim=1)                                # [N,J]

        # ── Step 9: RGCMF at three hierarchical levels ─────────────────────
        fused1 = self.rgcmf1(sil_outs0, pos_stp0, r_B, r_J_avg, r_M)  # [N,256,1]
        fused2 = self.rgcmf2(sil_outs1, pos_stp1, r_B, r_J_avg, r_M)
        fused3 = self.rgcmf3(sil_outs2, pos_stp2, r_B, r_J_avg, r_M)

        # ── BiHCASM2 (final level, identical to SeeGait) ──────────────────
        fused_final, sil_proj, pos_proj = self.BiHCASM2([sil_outs2], [pos_stp3])

        # ── Step 10: Assemble final descriptor (41 parts) ─────────────────
        embed_1 = torch.cat([
            embed_sil,                                   # [N,256,32]
            pos_stp3,                                    # [N,256,1]
            fused_final,                                 # [N,256,1]
            fused1, fused2, fused3,                      # [N,256,1] x3
            sil_proj[0], pos_proj[0],                   # [N,256,1] x2
            sil_outs2,                                   # [N,256,1]  <- extra
        ], dim=2)                                        # [N,256,41]

        fused_out = self.FCs_fuse(embed_1)               # [N,256,41]
        emb2, logits = self.BNNecks(fused_out)

        # ── UPML: Uncertainty-Propagated Metric Learning ───────────────────
        # Step 1: Predict raw log-sigma from the same feature representation
        log_sigma  = self.sigma_head(embed_1)                          # [N,256,41]
        sigma_base = F.softplus(log_sigma) + 1e-6                      # strictly positive

        # Step 2: Scale sigma by inverse reliability.
        # r_per_sample ∈ [0,1] — average of all four reliability levels.
        # High reliability → low uncertainty (narrow bubble).
        # Low reliability  → high uncertainty (wide bubble), e.g. for distractors.
        r_per_sample = (
            r_F.mean(dim=1)           +   # [N]  frame-level
            r_J.mean(dim=(1, 2))      +   # [N]  joint-level
            r_B.mean(dim=-1)          +   # [N]  body-region-level
            r_M.mean(dim=-1)              # [N]  modality-level
        ) / 4.0                           # normalise to [0,1]
        uncertainty_scale = (1.0 - r_per_sample).clamp(0.0, 1.0)      # [N]
        # Broadcast [N] → [N,256,41] to scale every dimension and part
        sigma = sigma_base * (1.0 + uncertainty_scale[:, None, None])  # [N,256,41]

        # ── Reliability-consistency regularization losses ──────────────────
        loss_F = self._frame_reg_loss(r_F, delta_t)
        loss_J = self._joint_reg_loss(r_J)
        loss_M = self._modal_reg_loss(r_M)
        reliability_loss = (self.beta_F * loss_F
                            + self.beta_J * loss_J
                            + self.beta_M * loss_M)

        retval = {
            'training_feat': {
                'triplet': {'embeddings': fused_out, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
                # UPML loss: passes mu (fused_out) + sigma so that
                # MutualLikelihoodScoreLoss can compute probabilistic distances.
                'pfe':     {'embeddings': fused_out, 'sigma': sigma, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': x.view(n * s, 1, h, w)
            },
            'inference_feat': {
                # At inference time we use mu only (backward compatible).
                # sigma is also stored so visualisation scripts can read it.
                'embeddings': fused_out,
                'sigma':      sigma.detach()
            },
            # Expose reliability scores for analysis / ablation
            'reliability': {
                'r_F': r_F.detach(),
                'r_J': r_J.detach(),
                'r_B': r_B.detach(),
                'r_M': r_M.detach(),
                'uncertainty_scale': uncertainty_scale.detach(),
                'reg_loss': reliability_loss.detach()
            }
        }
        return retval

    # ── Reliability-consistency losses ────────────────────────────────────────
    def _frame_reg_loss(self, r_F, delta_t):
        """L^F_reg: temporal smoothness."""
        import torch.nn.functional as F_nn
        diff    = (r_F[:, 1:] - r_F[:, :-1]).abs()
        allowed = self.alpha_F * delta_t[:, 1:].clamp(min=0)
        return F_nn.relu(diff - allowed).mean()

    def _joint_reg_loss(self, r_J):
        """L^J_reg: limb-pair consistency."""
        total = torch.zeros(1, device=r_J.device)
        for j1, j2 in LIMB_PAIRS:
            total = total + ((r_J[..., j1] - r_J[..., j2]) ** 2).mean()
        return total.squeeze()

    def _modal_reg_loss(self, r_M):
        """L^M_reg: modality diversity."""
        import torch.nn.functional as F_nn
        return F_nn.relu(self.tau_M - (r_M[:, 0] + r_M[:, 1])).mean()
