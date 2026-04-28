"""
train_autoencoder.py
====================

Train an autoencoder that compresses spherical codes (N unit vectors on
S^{D-1}, with variable N and D) into a fixed-dimensional latent suitable for
use as a state representation in a PPO policy searching for optimal packings.

Architecture
------------

Encoder — 2-IGN (2-Invariant Graph Network), SO(D)-invariant:
  1. Gram matrix  G = X Xᵀ  collapses the rotation symmetry exactly; only
     the pairwise inner products enter the network.
  2. GramEdgeEmbed maps each scalar G_ij to a pair feature vector.
  3. T × PairRefineLayer steps perform 2-WL message passing over the pair
     tensor (analogous to a 2-FWL graph network): p^{t+1}_{ij} ← LayerNorm(
     p^t_{ij} + MLP_u(cat(p^t_{ij}, (1/N) Σ_k MLP_in(p^t_{ik}) ⊙
     MLP_out(p^t_{kj})))).
  4. Pair features are averaged over all (i, j) pairs and concatenated with
     sinusoidal encodings of (log N, D) before a final MLP head produces the
     fixed-dim latent z.

Decoder — AdaLN self-attention with learnable slot queries:
  Each output slot has its own learned embedding (nn.Embedding of size N_max).
  AdaLNSelfAttnLayer blocks let slots coordinate via self-attention while z
  conditions the computation through Adaptive LayerNorm (DiT-style): each
  layer modulates the LayerNorm scale/shift and residual gates from z.
  This avoids the cross-attention homogenization bottleneck where all slots
  converged to cos_sim ≈ 0.9.  AdaLN-Zero initialisation (gates start near 0)
  stabilises early training.  A final linear head maps to D_max coordinates;
  outputs are masked per (N, D).

Loss
----

Primary — Entropic Gromov-Wasserstein (EGW) reconstruction:
  The reconstruction target is the Gram matrix of the input code; EGW matches
  the Gram of the prediction to the Gram of the target via an optimal
  transport plan T*.  Because both sides are represented by their Gram
  matrices, the loss is invariant to rotation AND permutation of the points.

  Solver: mirror-descent outer loop with Sinkhorn inner loop; T* is computed
  inside torch.no_grad() and the loss is re-evaluated at T* (envelope theorem
  gradient — no backprop through the solver).  epsilon is scaled to the
  median pairwise squared distance so the Sinkhorn temperature is
  automatically calibrated to the code geometry.  Non-converged batch
  elements are masked out (zero gradient) rather than poisoning the update.

Optional — contrastive EGW (anti-collapse, w_contrastive > 0):
  Build the full B×B matrix L[i,j] = EGW(pred_i, target_j), apply
  cross-entropy with labels = arange(B) and temperature tau.  A collapsed
  decoder (same output for all inputs) produces near-uniform assignment
  probability and incurs ~log(B) penalty.  A subsampled variant
  (contrastive_subsample = K) keeps K random negatives per positive to reduce
  cost from O(B²) to O(B·K).

Latent regularisation — VICReg (w_var, w_cov):
  Variance hinge: relu(1 − std_j) per dimension (fires only during collapse).
  Covariance: squared off-diagonal correlations encourage full use of the
  latent capacity.  Both terms use EMA-smoothed statistics (momentum 0.99)
  for stability at small batch sizes.

Smoothness — denoising penalty (w_smooth, warmed up over training):
  ||z_clean − φ(x + σε)||² where ε is Gaussian noise projected to the active
  D subspace and the perturbed points are re-normalised to the sphere.

Usage
-----

    python train_autoencoder.py                         # defaults (3000 steps)
    python train_autoencoder.py --train-steps 500 --batch-size 16
    python train_autoencoder.py --help                  # full CLI

Outputs are written to ./runs/<run_name>_<unix_ts>/:
  * config.json      - resolved config
  * train.log        - timestamped progress log
  * metrics.json     - list of per-eval-step metric dicts
  * ckpt.pt          - encoder + decoder state dicts + config
  * ckpt_step<N>.pt  - intermediate checkpoints (interval: ckpt_every)
  * curves.png       - training curves
  * smoothness.png   - ||Δz|| vs sigma empirical smoothness curve
  * recon_by_bin.png - reconstruction error heat map over (N, D) bins
  * nd_grid.png      - per-(N, D) EGW grid
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint
from tqdm.auto import tqdm

from data import sample_spherical_code, pad_batch, make_loader
from evaluation import run_invariance_tests, build_val_codes, evaluate
from diagnostics import (
    setup_logger, plot_curves, plot_latent_stats,
    plot_recon_breakdown, plot_smoothness,
    compute_grad_norms, plot_grad_norms,
    evaluate_nd_grid, plot_nd_grid, nd_grid_summary,
)
from loss import egw_gram_loss, EGWConfig


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # Data
    D_min: int = 3
    D_max: int = 32
    N_min: int = 20
    N_max: int = 600
    train_steps: int = 3000
    batch_size: int = 8
    val_size: int = 256
    val_every: int = 100
    ckpt_every: int = 1000

    # Model
    feat_dim: int = 128
    latent_dim: int = 128
    gt_d_pair: int = 64       # pair feature dim
    gt_d_msg: int = 32        # hidden dim inside MLP_m / MLP_u
    gt_layers: int = 4        # pair refinement steps
    gt_use_checkpoint: bool = False  # gradient checkpointing on each refinement layer
    dec_dim: int = 128
    dec_heads: int = 4
    dec_layers: int = 2
    pe_dim: int = 64

    # Loss weights
    w_egw: float = 1.0
    w_smooth_init: float = 0.0
    w_smooth_final: float = 0.0
    smooth_delay_frac: float = 0.0
    smooth_sigma: float = 0.02
    # VICReg latent regularization
    # w_var: soft hinge that penalises any z dimension with std < 1.
    #   Zero cost when the encoder is healthy; kicks in only during collapse.
    # w_cov: decorrelates z dimensions so the full latent_dim is used.
    #   Keep small — it exists to prevent rank collapse, not to dominate.
    w_var: float = 0.0
    w_cov: float = 0.0

    # --- B^2 contrastive EGW loss (anti-collapse) ---------------------------
    # w_contrastive = 0 disables contrastive mode and the loss falls back to
    # the plain per-sample EGW reconstruction (backward-compatible).
    # When enabled, we compute the full B×B EGW matrix L[i,j] =
    # EGW(pred_i, target_j) and apply cross-entropy with labels = arange(B),
    # temperature `contrastive_tau`. A generic/collapsed decoder produces
    # uniform assignment probability (≈1/B across targets) and incurs a
    # near-log(B) penalty — this is the anti-collapse signal.
    #
    # contrastive_subsample = -1 uses all B^2 pairs (most accurate, most
    # expensive: ~B× the plain-EGW forward cost). Setting e.g. 4 keeps only
    # 4 random off-diagonal negatives per positive (cheap variant).
    w_contrastive: float = 0.0         # 0 = off (plain EGW); try 1.0 to enable
    contrastive_tau: float = 0.1       # softmax temperature
    contrastive_subsample: int = -1    # K negatives per positive; -1 = all

    # --- EGW solver hyperparameters (exposed for the sweep) -----------------
    # Each surfaces an EGWConfig knob so you can vary it without editing code.
    egw_n_restarts: int = 1
    egw_epsilon_rel: float = 0.02
    egw_symmetry_break: float = 1.0
    egw_max_inner: int = 20
    egw_max_outer: int = 60
    egw_eps_anneal_steps: int = 3
    egw_identity_init: bool = True
    egw_sorted_row_init: bool = True
    egw_use_compile: bool = True

    # Optim
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # Runtime
    log_dir: str = "./runs"
    run_name: str = "fsw_vn_sphere"
    seed: int = 0
    device: str = "auto"                     # "auto" | "cuda" | "cpu"
    num_workers: int = 0                     # synthetic data, CPU-cheap
    ckpt_path: str = ""                      # source checkpoint for fine-tuning; "" = train from scratch

    # Data pipeline
    archive_dirs: list = field(default_factory=lambda: ["combined_points_archive", "spherical_code_archive"])
    perturb_prob: float = 0.4
    max_perturb_rounds: int = 3
    optimize_prob: float = 0.15
    perturb_sigma_lo: float = 0.001
    perturb_sigma_hi: float = 0.15
    quick_opt_steps_lo: int = 3
    quick_opt_steps_hi: int = 15
    source_weight_random: float = 0.15
    source_weight_archive: float = 0.50
    source_weight_fps: float = 0.35

    # PCGrad: treat each batch sample as an independent task and project
    # conflicting gradients before the update step.  Requires one backward
    # pass per sample so it is O(batch_size)× slower than a plain backward.
    # Incompatible with AMP (GradScaler), which is disabled automatically.
    use_pcgrad: bool = False


# =============================================================================
# 2-IGN pair-feature encoder (SO(D)-invariant, 2-WL-complete)
# =============================================================================

class GramEdgeEmbed(nn.Module):
    """Map scalar G_{ij} ∈ [-1, 1] to pair feature p⁰_{ij} ∈ ℝ^{d_p}."""

    def __init__(self, d_p: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_p), nn.GELU(), nn.Linear(d_p, d_p),
        )

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        # G: (B, N, N) → (B, N, N, d_p)
        return self.mlp(G.unsqueeze(-1))


class PairRefineLayer(nn.Module):
    """One 2-WL refinement step. 
    Memory-optimized to bypass einsum workspace spikes and torch.cat allocations.
    """
    def __init__(self, d_p: int, d_m: int):
        super().__init__()
        self.mlp_in = nn.Sequential(nn.Linear(d_p, d_m), nn.GELU())
        self.mlp_out = nn.Sequential(nn.Linear(d_p, d_m), nn.GELU())
        self.mlp_u = nn.Sequential(
            nn.Linear(d_p + d_m, d_p), nn.GELU(), nn.Linear(d_p, d_p),
        )
        self.norm = nn.LayerNorm(d_p)

    def forward(self, p: torch.Tensor, pair_mask: torch.Tensor,
                n_b: torch.Tensor) -> torch.Tensor:
        
        # 1. In-place masking avoids allocating new copies of the heavy pair tensors
        m_in = self.mlp_in(p)
        m_in.mul_(pair_mask)
        m_out = self.mlp_out(p)
        m_out.mul_(pair_mask)

        # 2. Map the contraction strictly to Batched GEMM to bypass einsum memory overhead
        # (B, N, N, d_m) -> (B, d_m, N, N)
        m_in = m_in.permute(0, 3, 1, 2)
        m_out = m_out.permute(0, 3, 1, 2)
        
        # Matrix multiplication computes the exact equivalent of the previous einsum
        msg = torch.matmul(m_in, m_out)  # (B, d_m, N, N)
        msg = msg.permute(0, 2, 3, 1)    # (B, N, N, d_m)
        
        # 3. In-place division
        msg.div_(n_b[:, None, None, None])

        # 4. Destructure the Linear layer to avoid the 1.1GB torch.cat allocation
        d_p = p.shape[-1]
        
        # nn.Linear weights are stored as (out_features, in_features)
        w_p = self.mlp_u[0].weight[:, :d_p]  # Weights mapping p
        w_m = self.mlp_u[0].weight[:, d_p:]  # Weights mapping msg
        b_u = self.mlp_u[0].bias
        
        # Compute the projections and combine them in-place
        h = F.linear(p, w_p)
        h.add_(F.linear(msg, w_m, b_u))
        
        # Explicitly free the heavy intermediate tensors before passing through the norm
        del m_in, m_out, msg

        # Apply the remaining layers of mlp_u
        h = self.mlp_u[1](h)  # GELU
        h = self.mlp_u[2](h)  # Linear
        
        # 5. Compute the residual, normalize, and apply the final in-place mask
        p = self.norm(p + h)
        p.mul_(pair_mask)
        
        return p


class GramPairEncoder(nn.Module):
    """GramEdgeEmbed + T × PairRefineLayer.  Returns pair tensor with padded
    entries zeroed.

    With ``use_checkpoint=True`` each refinement layer is wrapped in
    ``torch.utils.checkpoint`` — forward activations for the expensive O(N²·d_p)
    tensors are dropped and recomputed during backward, cutting peak memory by
    roughly gt_layers× at the cost of one extra forward pass through the
    refinement stack.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = GramEdgeEmbed(cfg.gt_d_pair)
        self.layers = nn.ModuleList([
            PairRefineLayer(cfg.gt_d_pair, cfg.gt_d_msg)
            for _ in range(cfg.gt_layers)
        ])
        self.use_checkpoint = getattr(cfg, "gt_use_checkpoint", False)

    def forward(self, G: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # G: (B, N, N); mask: (B, N)
        pair_mask_bool = mask.unsqueeze(-1) & mask.unsqueeze(-2)              # (B, N, N)
        pair_mask = pair_mask_bool.unsqueeze(-1).to(G.dtype)                  # (B, N, N, 1)
        n_b = mask.sum(dim=1).clamp_min(1).to(G.dtype)                        # (B,)
        p = self.embed(G) * pair_mask
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                p = torch_checkpoint.checkpoint(
                    layer, p, pair_mask, n_b, use_reentrant=False,
                )
            else:
                p = layer(p, pair_mask, n_b)
        return p, pair_mask


# =============================================================================
# Full encoder and decoder
# =============================================================================

class SphereCodeEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = GramPairEncoder(cfg)
        self.pair_proj = nn.Linear(cfg.gt_d_pair, cfg.feat_dim)
        self.nd_proj = nn.Sequential(
            nn.Linear(2 * cfg.pe_dim, cfg.pe_dim), nn.GELU(),
            nn.Linear(cfg.pe_dim, cfg.pe_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.feat_dim + cfg.pe_dim, 2 * cfg.latent_dim), nn.GELU(),
            nn.LayerNorm(2 * cfg.latent_dim),
            nn.Linear(2 * cfg.latent_dim, cfg.latent_dim),
            # Balance the variance across latent dims so the (N, D) channel
            # does not dominate z_proj downstream (see diagnosis: effective
            # rank 1.1 / 32 despite probes recovering many independent signals).
            nn.LayerNorm(cfg.latent_dim),
        )

    def diagnostic_submodules(self) -> dict[str, "nn.Module | nn.Parameter"]:
        """Named sub-modules for architecture-generic gradient diagnostics."""
        return {
            "backbone": self.backbone,
            "pair_proj": self.pair_proj,
            "nd_proj": self.nd_proj,
            "head": self.head,
        }

    def forward(self, x, mask, Ds):
        # Crop to the batch's active max-N to avoid paying O(N_max²) memory on
        # batches of small codes.  With BucketBatchSampler grouping by (D, N),
        # this is a 10-200× memory win on small-N batches.  Padded rows are
        # all zero — removing them does not change the output.
        N_active_max = int(mask.sum(dim=1).max().item())
        if N_active_max < x.shape[1]:
            x = x[:, :N_active_max, :]
            mask = mask[:, :N_active_max]

        # Crop D to batch-local max — Gram only needs the active dimensions
        D_active_max = int(Ds.max().item())
        if D_active_max < x.shape[2]:
            x = x[:, :, :D_active_max]

        G = x @ x.transpose(-1, -2)                         # (B, N, N) — rotation-invariant
        p, pair_mask = self.backbone(G, mask)               # (B, N, N, d_p); (B, N, N, 1)
        phi = self.pair_proj(p) * pair_mask                 # (B, N, N, feat_dim)

        n_b = mask.sum(dim=1).clamp_min(1).to(p.dtype)      # (B,)
        n_pairs = n_b * n_b                                  # (B,)
        pooled = phi.sum(dim=(1, 2)) / n_pairs[:, None]     # (B, feat_dim)

        n_enc = sinusoidal_encoding(torch.log(n_b), self.cfg.pe_dim)
        d_enc = sinusoidal_encoding(Ds.float(), self.cfg.pe_dim)
        nd_feat = self.nd_proj(torch.cat([n_enc, d_enc], dim=-1))

        return self.head(torch.cat([pooled, nd_feat], dim=-1))


_SINUSOIDAL_FREQS: dict[int, torch.Tensor] = {}  # half -> CPU freqs tensor


def sinusoidal_encoding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal encoding of a scalar per element."""
    half = dim // 2
    if half not in _SINUSOIDAL_FREQS:
        _SINUSOIDAL_FREQS[half] = torch.exp(
            torch.arange(half).float() * -(math.log(10000.0) / max(half - 1, 1))
        )
    freqs = _SINUSOIDAL_FREQS[half].to(device=values.device)
    args = values.unsqueeze(-1).float() * freqs
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if enc.shape[-1] < dim:
        enc = F.pad(enc, (0, dim - enc.shape[-1]))
    return enc


class AdaLNSelfAttnLayer(nn.Module):
    """Self-attention + FFN conditioned on z via Adaptive LayerNorm (DiT-style).

    z modulates LayerNorm parameters (scale, shift) at every layer, so the
    latent shapes every forward pass — posterior collapse is impossible because
    the slots can't produce anything without z.

    AdaLN-Zero: the residual gate alphas are initialised near zero so the
    network starts as near-identity and gradually learns to use self-attention
    and FFN, stabilising early training.

    Self-attention lets slots coordinate ("you go there, I go elsewhere"),
    which is essential for packing — independent slots have no stable ordering
    under the permutation-invariant EGW loss."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 z_dim: int):
        super().__init__()
        # 6 modulation parameters per layer: gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(z_dim, 6 * d_model),
        )
        # Initialise the final linear so alphas (gates) start near zero
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=0.0)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim) → 6 modulation vectors: (B, d_model) each
        g1, b1, a1, g2, b2, a2 = self.adaLN_modulation(z).chunk(6, dim=-1)
        # Self-attention with AdaLN conditioning
        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + a1.unsqueeze(1) * h
        # FFN with AdaLN conditioning
        h = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        h = self.ffn(h)
        x = x + a2.unsqueeze(1) * h
        return x


class SphereCodeDecoder(nn.Module):
    """AdaLN self-attention decoder with anonymous (unordered) slots.

    All slots start as the same learned prototype plus i.i.d. Gaussian noise.
    This makes the decoder equivariant to slot permutations: no slot has a
    preferred identity, so the permutation-invariant EGW loss cannot create
    conflicting gradients through slot-specific parameters.

    Symmetry breaking comes from the random noise (different each forward
    pass). Self-attention lets slots coordinate ("you go there, I go
    elsewhere"), and z conditions the computation via AdaLN so the latent
    determines the *content* (which Gram matrix) while noise determines the
    *ordering* (which slot produces which point)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Shared prototype for all slots — no per-slot identity.
        self.slot_proto = nn.Parameter(torch.randn(cfg.dec_dim) * 0.02)
        # Learnable noise scale — network can tune how much diversity the
        # slots need before self-attention kicks in.
        self.noise_log_scale = nn.Parameter(torch.tensor(0.0))
        # Direct z → slot path: project z to dec_dim and add to every slot.
        # This gives z an ungated gradient path (no alpha gate in the way),
        # preventing the AdaLN-Zero chicken-and-egg deadlock where alpha=0
        # blocks gradient flow to z.  Because the same z-bias is added to
        # ALL slots, z can only encode global (permutation-invariant) info.
        self.z_to_slots = nn.Linear(cfg.latent_dim, cfg.dec_dim)
        self.layers = nn.ModuleList([
            AdaLNSelfAttnLayer(
                d_model=cfg.dec_dim, nhead=cfg.dec_heads,
                dim_feedforward=4 * cfg.dec_dim,
                z_dim=cfg.latent_dim,
            )
            for _ in range(cfg.dec_layers)
        ])
        self.norm_out = nn.LayerNorm(cfg.dec_dim)
        self.out = nn.Linear(cfg.dec_dim, cfg.D_max)

    def diagnostic_submodules(self) -> dict[str, "nn.Module | nn.Parameter"]:
        """Named sub-modules for architecture-generic gradient diagnostics."""
        return {
            "slot_proto": self.slot_proto,
            "z_to_slots": self.z_to_slots,
            "layers": self.layers,
            "norm_out": self.norm_out,
            "out": self.out,
        }

    @torch.no_grad()
    def z_conditioning_summary(self, z: torch.Tensor) -> list[dict]:
        """Per-layer summary of how z conditions the decoder.

        Returns a list of dicts (one per layer) with modulation statistics.
        Architecture-generic: callers do not need to know the conditioning
        mechanism.
        """
        summaries = []
        for i, layer in enumerate(self.layers):
            entry: dict = {"layer": i}
            if hasattr(layer, "adaLN_modulation"):
                mod = layer.adaLN_modulation(z)  # (B, 6*d_model)
                chunks = mod.chunk(6, dim=-1)
                names = ["gamma1", "beta1", "alpha1", "gamma2", "beta2", "alpha2"]
                entry["type"] = "AdaLN"
                entry["modulations"] = {}
                for name, ch in zip(names, chunks):
                    entry["modulations"][name] = {
                        "norm_mean": ch.norm(dim=-1).mean().item(),
                        "inter_sample_std": ch.std(dim=0).mean().item(),
                    }
                if z.shape[0] >= 2:
                    entry["cos_01"] = torch.nn.functional.cosine_similarity(
                        mod[0].unsqueeze(0), mod[1].unsqueeze(0)
                    ).item()
            else:
                entry["type"] = "unknown"
            summaries.append(entry)
        return summaries

    @torch.no_grad()
    def slot_init_and_forward_stages(
        self, z: torch.Tensor, N: int,
    ) -> dict[str, torch.Tensor]:
        """Run the decoder forward pass and return named intermediate tensors.

        Useful for diagnostics that want to inspect slot diversity, entropy
        rank, etc. at each pipeline stage without knowing the decoder's
        internal structure.
        """
        B = z.shape[0]
        device = z.device
        noise_scale = self.noise_log_scale.exp()
        noise = torch.randn(B, N, self.cfg.dec_dim, device=device) * noise_scale
        z_bias = self.z_to_slots(z).unsqueeze(1)
        slots = self.slot_proto.unsqueeze(0).unsqueeze(0).expand(B, N, -1) + noise + z_bias

        stages: dict[str, torch.Tensor] = {"0_slot_init": slots.clone()}
        h = slots
        for i, layer in enumerate(self.layers):
            h = layer(h, z)
            stages[f"1_layer_{i}"] = h.clone()
        h_normed = self.norm_out(h)
        stages["2_norm_out"] = h_normed
        stages["3_output"] = self.out(h_normed)
        return stages

    def forward(self, z, Ns, Ds):
        B = z.shape[0]
        device = z.device
        N_max, D_max = self.cfg.N_max, self.cfg.D_max

        # Crop to batch-local max N and D to avoid wasting compute on padding
        N_active = int(Ns.max().item())
        D_active = int(Ds.max().item())

        # Anonymous slots: shared prototype + i.i.d. noise for symmetry breaking.
        # All slots start identically (up to noise), so no slot has a preferred
        # identity — permutation equivariant by construction.
        noise_scale = self.noise_log_scale.exp()
        noise = torch.randn(B, N_active, self.cfg.dec_dim, device=device) * noise_scale
        # z-bias: same vector added to every slot → encodes global structure,
        # not per-slot position.  Gradient flows ∂L/∂pred → ∂pred/∂slots → z
        # with no alpha gate in the path.
        z_bias = self.z_to_slots(z).unsqueeze(1)             # (B, 1, dec_dim)
        slots = self.slot_proto.unsqueeze(0).unsqueeze(0).expand(B, N_active, -1) + noise + z_bias
        h = slots

        for layer in self.layers:
            h = layer(h, z)
        h = self.norm_out(h)
        raw = self.out(h)[:, :, :D_active]            # (B, N_active, D_active)

        pos = torch.arange(N_active, device=device).unsqueeze(0).expand(B, -1)
        mask = pos < Ns.unsqueeze(1)
        D_mask = (torch.arange(D_active, device=device).unsqueeze(0)
                  < Ds.unsqueeze(1))
        raw = raw * D_mask.unsqueeze(1)
        pred = raw * mask.unsqueeze(-1)
        # Normalize to unit sphere
        #norms = pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        #pred = (pred / norms) * mask.unsqueeze(-1)

        # Pad back to (N_max, D_max) so shapes match the fixed-size input
        # tensors from pad_batch (avoids CUDA allocator fragmentation).
        if N_active < N_max or D_active < D_max:
            pred = F.pad(pred, (0, D_max - D_active, 0, N_max - N_active))
            mask = F.pad(mask, (0, N_max - N_active))
        return pred, mask


# =============================================================================
# Losses
# =============================================================================

def _make_egw_cfg(cfg: "Config") -> EGWConfig:
    """Build an EGWConfig with all the solver-side knobs exposed via CLI."""
    return EGWConfig(
        epsilon_rel=cfg.egw_epsilon_rel,
        max_inner=cfg.egw_max_inner,
        max_outer=cfg.egw_max_outer,
        eps_anneal_steps=cfg.egw_eps_anneal_steps,
        n_restarts=cfg.egw_n_restarts,
        symmetry_break=cfg.egw_symmetry_break,
        identity_init=cfg.egw_identity_init,
        sorted_row_init=cfg.egw_sorted_row_init,
        use_compile=cfg.egw_use_compile,
    )


def _make_egw_loss_fn(cfg: "Config"):
    """Return a closure that uses the model's EGW solver settings."""
    egw_cfg = _make_egw_cfg(cfg)
    def _fn(pred, target, pred_mask, target_mask, n_slices=None, Ds=None):
        return egw_gram_loss(pred, target, pred_mask, target_mask, cfg=egw_cfg)
    return _fn


# Backward compat: default-config version used when no Config is available.
def _egw_loss_fn(pred, target, pred_mask, target_mask, n_slices=None, Ds=None):
    """EGW reconstruction loss with default solver settings."""
    return egw_gram_loss(pred, target, pred_mask, target_mask, cfg=EGWConfig())


def egw_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    egw_cfg: EGWConfig,
    tau: float = 0.1,
    subsample: int = -1,
):
    """B^2 contrastive EGW loss.

    Builds the full B×B matrix L[i, j] = EGW(pred_i, target_j) in one batched
    solver call (effective batch size B^2), then applies cross-entropy with
    labels = arange(B) and temperature `tau`.

    When subsample >= 0, only keeps that many random negatives per positive
    (plus the positive itself), cutting the effective batch from B^2 to
    B*(K+1). This is the cheap variant recommended in the discrimination
    analysis.

    Returns
    -------
    l_contrastive : scalar
        Cross-entropy loss. Random-guess value is log(B) (or log(K+1) for the
        subsampled variant); dropping significantly below that is the sign
        that the decoder is producing target-specific outputs.
    l_rec : scalar
        Mean of the diagonal L[i,i] — the standard per-sample EGW. Kept so
        callers can optionally add it with its own weight.
    L : (B, B) tensor
        The full EGW matrix (detached) — useful for logging / analysis.
    """
    B, N_p, D = pred.shape
    _, N_t, _ = target.shape
    device = pred.device

    if subsample >= 0 and subsample < B - 1:
        # Build a (B, K+1) index tensor: [positive i, K random negatives]
        K = int(subsample)
        # Vectorized: sample K indices from {0..B-2} per row, then shift
        # values >= i up by 1 to exclude the positive.
        pos = torch.arange(B, device=device).unsqueeze(1)          # (B, 1)
        neg_raw = torch.stack([
            torch.randperm(B - 1, device=device)[:K] for _ in range(B)
        ])                                                          # (B, K)
        neg = neg_raw + (neg_raw >= pos).long()
        j_idx = torch.cat([pos, neg], dim=1)                       # (B, K+1)
        pred_stk = pred.unsqueeze(1).expand(B, K + 1, N_p, D)
        target_stk = target[j_idx]                           # (B, K+1, N_t, D)
        pm_stk = pred_mask.unsqueeze(1).expand(B, K + 1, N_p)
        tm_stk = target_mask[j_idx]                          # (B, K+1, N_t)
        # Flatten to batched solver call
        flat_shape = (B * (K + 1),)
        pred_flat = pred_stk.reshape(*flat_shape, N_p, D)
        target_flat = target_stk.reshape(*flat_shape, N_t, D)
        pm_flat = pm_stk.reshape(*flat_shape, N_p)
        tm_flat = tm_stk.reshape(*flat_shape, N_t)
        L_flat = egw_gram_loss(pred_flat, target_flat, pm_flat, tm_flat,
                                cfg=egw_cfg, reduction="none")
        L_sub = L_flat.view(B, K + 1)                        # [i, 0] = EGW(pred_i, target_i)
        logits = -L_sub / tau
        labels = torch.zeros(B, dtype=torch.long, device=device)
        l_contrastive = F.cross_entropy(logits, labels)
        l_rec = L_sub[:, 0].mean()
        # Construct a sparse L for logging (diag only populated densely)
        L_full = torch.full((B, B), float("nan"), device=device)
        L_full.scatter_(1, j_idx, L_sub.detach())
        return l_contrastive, l_rec, L_full

    # Full B×B matrix
    pred_stk = pred.unsqueeze(1).expand(B, B, N_p, D).reshape(B * B, N_p, D)
    target_stk = target.unsqueeze(0).expand(B, B, N_t, D).reshape(B * B, N_t, D)
    pm_stk = pred_mask.unsqueeze(1).expand(B, B, N_p).reshape(B * B, N_p)
    tm_stk = target_mask.unsqueeze(0).expand(B, B, N_t).reshape(B * B, N_t)

    L_flat = egw_gram_loss(pred_stk, target_stk, pm_stk, tm_stk,
                            cfg=egw_cfg, reduction="none")
    L = L_flat.view(B, B)
    logits = -L / tau
    labels = torch.arange(B, device=device)
    l_contrastive = F.cross_entropy(logits, labels)
    l_rec = L.diag().mean()
    return l_contrastive, l_rec, L.detach()


def vicreg_latent(
    z: torch.Tensor,
    ema_mean: torch.Tensor,
    ema_var: torch.Tensor,
    ema_cov: torch.Tensor,
    momentum: float = 0.99,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance and covariance terms using EMA statistics.

    With small batches (B=8) the per-batch estimates of std and covariance are
    too noisy to be useful.  Instead we maintain exponential moving averages of
    the mean, per-dimension variance, and covariance matrix across steps
    (momentum=0.99 ≈ 100-step window).  The loss is computed against these
    smoothed statistics, making it stable even at B=8.

    The EMA tensors are updated in-place and must be initialised as zeros
    before the training loop.

    Variance term: relu(1 - std_j) per dimension, averaged.
      Zero when all dims have std >= 1; only fires during collapse.
    Covariance term: squared off-diagonal correlations / D.
      Encourages full use of latent capacity.

        The forward pass uses EMA-smoothed statistics, while gradients flow through
        the current batch statistics via a straight-through estimator.  This keeps
        the regularizer numerically stable without disconnecting it from ``z``.
    """
    B, D = z.shape
    with torch.no_grad():
        batch_mean = z.mean(dim=0)
        z_c = z - batch_mean
        batch_var = z_c.var(dim=0, unbiased=False)
        batch_cov = (z_c.T @ z_c) / B
        ema_mean.mul_(momentum).add_(batch_mean * (1 - momentum))
        ema_var.mul_(momentum).add_(batch_var * (1 - momentum))
        ema_cov.mul_(momentum).add_(batch_cov * (1 - momentum))

    # Straight-through smoothing: forward uses EMA stats, backward uses live
    # batch stats so gradients still flow through the current minibatch.
    live_mean = z.mean(dim=0)
    mean_st = live_mean + (ema_mean - live_mean).detach()
    z_c_grad = z - mean_st                               # (B, D)

    live_var = z_c_grad.var(dim=0, unbiased=False)
    var_st = live_var + (ema_var - live_var).detach()
    std_st = (var_st + eps).sqrt()
    l_var = F.relu(1.0 - std_st).mean()

    if not hasattr(vicreg_latent, "_mask_off_cache"):
        vicreg_latent._mask_off_cache = {}
    _key = (D, str(z.device))
    if _key not in vicreg_latent._mask_off_cache:
        vicreg_latent._mask_off_cache[_key] = ~torch.eye(D, dtype=torch.bool, device=z.device)
    mask_off = vicreg_latent._mask_off_cache[_key]

    live_cov = (z_c_grad.T @ z_c_grad) / B
    cov_st = live_cov + (ema_cov - live_cov).detach()
    norm_st = std_st.unsqueeze(1) * std_st.unsqueeze(0)
    corr_st = cov_st / norm_st.clamp_min(eps)
    l_cov = corr_st[mask_off].pow(2).sum() / D
    return l_var, l_cov


def denoising_smoothness(encoder, x, mask, Ds, z_clean, sigma: float = 0.02):
    """||z_clean - phi(x + sigma*eps)||^2, where eps stays within the active D
    subspace and the perturbed points are re-projected to the sphere.
    Accepts pre-computed z_clean to avoid a redundant encoder forward pass."""
    noise = torch.randn_like(x) * sigma
    D_mask = (torch.arange(x.shape[-1], device=x.device).unsqueeze(0)
              < Ds.unsqueeze(1)).unsqueeze(1)
    noise = noise * D_mask
    x_pert = x + noise
    norms = x_pert.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x_pert = (x_pert / norms) * mask.unsqueeze(-1)
    z2 = encoder(x_pert, mask, Ds)
    return ((z_clean - z2) ** 2).sum(-1).mean()


# =============================================================================
# Training
# =============================================================================

def train(cfg: Config) -> Path:
    # Reproducibility
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Device
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    # Run directory
    run_dir = Path(cfg.log_dir) / f"{cfg.run_name}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    logger = setup_logger(run_dir)
    logger.info(f"device: {device}")
    logger.info(f"run_dir: {run_dir}")
    logger.info(f"config: {json.dumps(asdict(cfg))}")

    # Metric store
    metrics_log: list = []

    def log_scalars(step, **kv):
        rec = {"step": step, **kv}
        metrics_log.append(rec)
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in kv.items())
        logger.info(f"step {step:6d} | {msg}")

    # Models
    enc_model = SphereCodeEncoder(cfg).to(device)
    dec_model = SphereCodeDecoder(cfg).to(device)
    n_enc = sum(p.numel() for p in enc_model.parameters())
    n_dec = sum(p.numel() for p in dec_model.parameters())
    logger.info(f"encoder params: {n_enc/1e6:.2f}M  decoder params: {n_dec/1e6:.2f}M")

    if cfg.ckpt_path:
        logger.info(f"loading weights from checkpoint: {cfg.ckpt_path}")
        _ckpt = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        enc_model.load_state_dict(_ckpt["encoder"])
        dec_model.load_state_dict(_ckpt["decoder"])
        logger.info("checkpoint weights loaded successfully")

    opt = torch.optim.AdamW(
        list(enc_model.parameters()) + list(dec_model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min((s + 1) / max(cfg.warmup_steps, 1), 1.0)
    )

    # PCGrad wraps the optimizer; AMP is disabled when PCGrad is active
    # because PCGrad calls .backward() internally without the GradScaler.
    if cfg.use_pcgrad:
        _pcgrad_dir = str(Path(__file__).parent / "Pytorch-PCGrad")
        if _pcgrad_dir not in sys.path:
            sys.path.insert(0, _pcgrad_dir)
        from pcgrad import PCGrad
        pc_opt = PCGrad(opt)
        logger.info("PCGrad enabled: one backward per sample, AMP disabled")
    else:
        pc_opt = None

    logger.info("=== invariance check at init ===")
    run_invariance_tests(enc_model, cfg, device)

    # Performance optimization: cuDNN autotuner
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Use AMP for potential memory & speed improvements.
    # PCGrad calls .backward() internally so it cannot use GradScaler.
    use_amp = (device == "cuda") and not cfg.use_pcgrad
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Validation set
    val_codes = build_val_codes(cfg)

    # Smoothness weight schedule (with optional delay)
    smooth_delay_steps = int(cfg.smooth_delay_frac * cfg.train_steps)
    smooth_ramp_steps = min(300, cfg.train_steps - smooth_delay_steps)

    def smooth_weight(step):
        if step < smooth_delay_steps:
            return 0.0
        ramp_step = step - smooth_delay_steps
        frac = min(ramp_step / smooth_ramp_steps, 1.0)
        return cfg.w_smooth_init + frac * (cfg.w_smooth_final - cfg.w_smooth_init)

    loader = make_loader(cfg.batch_size * cfg.train_steps, cfg, seed=None)
    loader_iter = iter(loader)

    # Rolling window: track mean N and D over the last 100 batches
    recent_Ns: collections.deque = collections.deque(maxlen=100)

    # EMA state for VICReg latent regularization (robust at small batch sizes)
    _D = cfg.latent_dim
    vicreg_ema_mean = torch.zeros(_D, device=device)
    vicreg_ema_var  = torch.ones(_D, device=device)   # init to 1 → l_var starts at 0
    vicreg_ema_cov  = torch.zeros(_D, _D, device=device)
    recent_Ds: collections.deque = collections.deque(maxlen=100)

    egw_cfg = _make_egw_cfg(cfg)

    use_contrastive = cfg.w_contrastive > 0.0
    contrastive_random_nll = (
        math.log(cfg.batch_size) if cfg.contrastive_subsample < 0
        else math.log(cfg.contrastive_subsample + 1)
    ) if use_contrastive else 0.0
    all_params = list(enc_model.parameters()) + list(dec_model.parameters())
    _zero = torch.zeros((), device=device)

    def next_batch(current_iter):
        try:
            batch = next(current_iter)
        except StopIteration:
            current_iter = iter(make_loader(cfg.batch_size * cfg.train_steps,
                                            cfg, seed=None))
            batch = next(current_iter)
        x, mask, Ds, Ns = [t.to(device, non_blocking=True) for t in batch]
        return current_iter, x, mask, Ds, Ns

    def compute_reconstruction_terms(x, mask, Ds, Ns):
        with torch.amp.autocast(enabled=use_amp, device_type=device):
            z = enc_model(x, mask, Ds)
            pred, pred_mask = dec_model(z, Ns, Ds)

            if use_contrastive:
                l_con, l_egw, L_mat = egw_contrastive_loss(
                    pred, x, pred_mask, mask, egw_cfg=egw_cfg,
                    tau=cfg.contrastive_tau,
                    subsample=cfg.contrastive_subsample,
                )
            else:
                l_egw = egw_gram_loss(pred, x, pred_mask, mask, cfg=egw_cfg)
                l_con, L_mat = _zero, None

            l_var, l_cov = vicreg_latent(
                z, vicreg_ema_mean, vicreg_ema_var, vicreg_ema_cov,
            )
            main_loss = (cfg.w_egw * l_egw
                         + cfg.w_contrastive * l_con
                         + cfg.w_var * l_var
                         + cfg.w_cov * l_cov)

        return z, l_egw, l_con, L_mat, l_var, l_cov, main_loss

    def compute_smoothness_loss(step, x, mask, Ds, z):
        w_sm = smooth_weight(step)
        if w_sm <= 0:
            return w_sm, _zero

        with torch.amp.autocast(enabled=use_amp, device_type=device):
            l_smooth = denoising_smoothness(
                enc_model, x, mask, Ds, z.detach(), sigma=cfg.smooth_sigma,
            )
        return w_sm, l_smooth

    def contrastive_log_scalars(l_con, L_mat):
        if not use_contrastive:
            return {}

        scalars = {
            "train_contrastive": l_con.item(),
            "train_contrastive_vs_random": (
                l_con.item() - contrastive_random_nll
            ),
        }
        if L_mat is not None and cfg.contrastive_subsample < 0:
            diag = L_mat.diag()
            off_mean = (
                (L_mat.sum() - diag.sum())
                / (L_mat.numel() - diag.numel())
            ).item()
            scalars["train_egw_diag_offdiag_gap"] = (
                off_mean - diag.mean().item()
            )
        return scalars

    def maybe_log_validation(step, l_egw, l_smooth, l_var, l_cov,
                             gn, w_sm, l_con, L_mat, grad_norms):
        if step % cfg.val_every != 0 and step != cfg.train_steps - 1:
            return

        val = evaluate(enc_model, dec_model, val_codes, cfg, device,
                       loss_fn=_make_egw_loss_fn(cfg))
        scalar_kwargs = dict(
            train_egw=l_egw.item(),
            train_smooth=l_smooth.item(),
            train_var=l_var.item(), train_cov=l_cov.item(),
            grad_norm=float(gn),
            lr=opt.param_groups[0]["lr"], w_smooth=w_sm,
            mean_N=float(np.mean(recent_Ns)),
            mean_D=float(np.mean(recent_Ds)),
        )
        scalar_kwargs.update(contrastive_log_scalars(l_con, L_mat))
        log_scalars(step, **scalar_kwargs, **grad_norms, **val)

    def maybe_save_checkpoint(step):
        if cfg.ckpt_every <= 0 or step <= 0 or step % cfg.ckpt_every != 0:
            return

        ckpt_path = run_dir / f"ckpt_step{step}.pt"
        torch.save({
            "encoder": enc_model.state_dict(),
            "decoder": dec_model.state_dict(),
            "cfg": asdict(cfg),
            "step": step,
        }, ckpt_path)
        logger.info(f"saved checkpoint: {ckpt_path}")

    def update_progress_bar(step, l_egw, gn, l_con):
        if step % 25 != 0:
            return

        post = {
            "egw": f"{l_egw.item():.3f}",
            "gn": f"{float(gn):.1f}",
            "mN": f"{np.mean(recent_Ns):.0f}",
            "mD": f"{np.mean(recent_Ds):.1f}",
        }
        if use_contrastive:
            post["con"] = f"{l_con.item():.3f}"
        pbar.set_postfix(post)

    logger.info("=== starting training ===")
    t_start = time.time()
    pbar = tqdm(range(cfg.train_steps), desc="train")

    for step in pbar:
        loader_iter, x, mask, Ds, Ns = next_batch(loader_iter)

        recent_Ns.append(Ns.float().mean().item())
        recent_Ds.append(Ds.float().mean().item())

        if cfg.use_pcgrad:
            # ------------------------------------------------------------------
            # PCGrad path: each sample in the batch is a separate task.
            # One backward pass per sample; gradients projected to remove
            # inter-sample conflicts before the parameter update.
            # AMP is disabled (see use_amp assignment above).
            # ------------------------------------------------------------------
            z = enc_model(x, mask, Ds)
            pred, pred_mask = dec_model(z, Ns, Ds)

            # Per-sample EGW losses → one task per sample
            l_egw_vec = egw_gram_loss(
                pred, x, pred_mask, mask, cfg=egw_cfg, reduction="none"
            )  # (B,)
            l_egw = l_egw_vec.mean()

            # Batch-level losses (VICReg, optional contrastive)
            if use_contrastive:
                l_con, _, L_mat = egw_contrastive_loss(
                    pred, x, pred_mask, mask, egw_cfg=egw_cfg,
                    tau=cfg.contrastive_tau,
                    subsample=cfg.contrastive_subsample,
                )
            else:
                l_con, L_mat = _zero, None

            l_var, l_cov = vicreg_latent(
                z, vicreg_ema_mean, vicreg_ema_var, vicreg_ema_cov,
            )
            w_sm, l_smooth = compute_smoothness_loss(step, x, mask, Ds, z)

            # One task per sample (EGW) + one combined batch-reg task
            tasks = [cfg.w_egw * l_egw_vec[i] for i in range(x.shape[0])]
            batch_reg = (
                cfg.w_contrastive * l_con
                + cfg.w_var * l_var
                + cfg.w_cov * l_cov
            )
            if w_sm > 0:
                batch_reg = batch_reg + w_sm * l_smooth
            tasks.append(batch_reg)

            pc_opt.pc_backward(tasks)
            grad_norms = compute_grad_norms(enc_model, dec_model)
            gn = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            opt.step()
            sched.step()
        else:
            # ------------------------------------------------------------------
            # Standard path (AMP + single backward)
            # ------------------------------------------------------------------
            z, l_egw, l_con, L_mat, l_var, l_cov, main_loss = (
                compute_reconstruction_terms(x, mask, Ds, Ns)
            )
            w_sm, l_smooth = compute_smoothness_loss(step, x, mask, Ds, z)

            opt.zero_grad(set_to_none=True)
            scaler.scale(main_loss).backward()
            if w_sm > 0:
                scaler.scale(w_sm * l_smooth).backward()
            scaler.unscale_(opt)
            grad_norms = compute_grad_norms(enc_model, dec_model)
            gn = torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

        update_progress_bar(step, l_egw, gn, l_con)
        maybe_save_checkpoint(step)
        maybe_log_validation(
            step, l_egw, l_smooth, l_var, l_cov,
            gn, w_sm, l_con, L_mat, grad_norms,
        )

    logger.info(f"training complete in {(time.time() - t_start):.1f}s")

    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    torch.save({
        "encoder": enc_model.state_dict(),
        "decoder": dec_model.state_dict(),
        "cfg": asdict(cfg),
    }, run_dir / "ckpt.pt")
    logger.info(f"saved checkpoint: {run_dir / 'ckpt.pt'}")

    # Plots
    plot_curves(metrics_log, run_dir)
    plot_latent_stats(metrics_log, run_dir)
    _loss_fn = _make_egw_loss_fn(cfg)
    plot_recon_breakdown(enc_model, dec_model, cfg, device, run_dir,
                         loss_sw=_loss_fn)
    plot_smoothness(enc_model, cfg, device, run_dir)
    plot_grad_norms(metrics_log, run_dir)
    plot_nd_grid(enc_model, dec_model, cfg, device, run_dir,
                 loss_sw=_loss_fn)
    nd_grid, nd_N, nd_D = evaluate_nd_grid(enc_model, dec_model, cfg, device,
                                           loss_sw=_loss_fn)
    logger.info(nd_grid_summary(nd_grid, nd_N, nd_D))
    logger.info(f"saved figures to {run_dir}")

    return run_dir


# =============================================================================
# Checkpoint loading helper (for downstream PPO integration)
# =============================================================================

def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _cfg_from_ckpt(cfg_dict: dict) -> Config:
    """Build a Config from a checkpoint dict, filtering legacy keys."""
    from dataclasses import fields as _fields
    _valid = {f.name for f in _fields(Config)}
    return Config(**{k: v for k, v in cfg_dict.items() if k in _valid})


def load_checkpoint(
    ckpt_path: str | Path,
    device: str = "auto",
) -> tuple[SphereCodeEncoder, SphereCodeDecoder, Config]:
    """Canonical loader — returns (encoder, decoder, cfg).

    All external consumers (diagnose_model, latent_test_bed, etc.) should use
    this instead of importing model classes and reconstructing manually.
    Old checkpoints are handled transparently (legacy config keys filtered).
    """
    device = _resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_ckpt(ckpt["cfg"])
    enc = SphereCodeEncoder(cfg)
    dec = SphereCodeDecoder(cfg)
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.to(device).eval()
    dec.to(device).eval()
    return enc, dec, cfg


def load_encoder(ckpt_path: str | Path, device: str = "auto"):
    """Load just the encoder from a checkpoint, for PPO use.
    Returns (encoder, cfg_dict). Example:

        enc, cfg = load_encoder("runs/.../ckpt.pt")
        enc.eval()
        x, mask, Ds, Ns = pad_batch(codes, cfg["D_max"], cfg["N_max"])
        z = enc(x.to(device), mask.to(device))       # fixed-dim latent
    """
    device = _resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg_obj = _cfg_from_ckpt(cfg_dict)
    enc = SphereCodeEncoder(cfg_obj)
    enc.load_state_dict(ckpt["encoder"])
    return enc.to(device), cfg_dict


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> Config:
    _d = Config()  # single source of truth for all defaults
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--D-min", type=int, default=_d.D_min)
    p.add_argument("--D-max", type=int, default=_d.D_max)
    p.add_argument("--N-min", type=int, default=_d.N_min)
    p.add_argument("--N-max", type=int, default=_d.N_max)
    p.add_argument("--train-steps", type=int, default=_d.train_steps)
    p.add_argument("--batch-size", type=int, default=_d.batch_size)
    p.add_argument("--val-size", type=int, default=_d.val_size)
    p.add_argument("--val-every", type=int, default=_d.val_every)
    p.add_argument("--ckpt-every", type=int, default=_d.ckpt_every,
                   help="save a checkpoint every N steps; 0 = disabled")
    # Model
    p.add_argument("--feat-dim", type=int, default=_d.feat_dim)
    p.add_argument("--latent-dim", type=int, default=_d.latent_dim)
    p.add_argument("--gt-d-pair", type=int, default=_d.gt_d_pair,
                   help="pair feature dim in 2-IGN encoder")
    p.add_argument("--gt-d-msg", type=int, default=_d.gt_d_msg,
                   help="hidden dim inside pair-refinement MLPs")
    p.add_argument("--gt-layers", type=int, default=_d.gt_layers,
                   help="number of pair refinement steps")
    p.add_argument("--gt-use-checkpoint", action=argparse.BooleanOptionalAction,
                   default=_d.gt_use_checkpoint,
                   help="gradient-checkpoint each pair refinement layer; cuts "
                        "encoder activation memory by ~gt_layers× at the cost "
                        "of one extra forward through the refinement stack")
    p.add_argument("--dec-dim", type=int, default=_d.dec_dim)
    p.add_argument("--dec-heads", type=int, default=_d.dec_heads)
    p.add_argument("--dec-layers", type=int, default=_d.dec_layers)
    p.add_argument("--pe-dim", type=int, default=_d.pe_dim)
    # Loss
    p.add_argument("--w-egw", type=float, default=_d.w_egw)
    p.add_argument("--w-smooth-init", type=float, default=_d.w_smooth_init)
    p.add_argument("--w-smooth-final", type=float, default=_d.w_smooth_final)
    p.add_argument("--smooth-delay-frac", type=float, default=_d.smooth_delay_frac)
    p.add_argument("--smooth-sigma", type=float, default=_d.smooth_sigma)
    p.add_argument("--w-var", type=float, default=_d.w_var,
                   help="VICReg variance hinge weight (anti-collapse in latent)")
    p.add_argument("--w-cov", type=float, default=_d.w_cov,
                   help="VICReg covariance weight (latent decorrelation)")

    # --- Contrastive EGW (anti-collapse) ----------------------------------
    p.add_argument("--w-contrastive", type=float, default=_d.w_contrastive,
                   help="weight on B^2 contrastive cross-entropy; 0 = off "
                        "(plain EGW), 1.0 is a reasonable starting point")
    p.add_argument("--contrastive-tau", type=float, default=_d.contrastive_tau,
                   help="softmax temperature for contrastive CE. Smaller = "
                        "sharper; try 0.05-0.3")
    p.add_argument("--contrastive-subsample", type=int, default=_d.contrastive_subsample,
                   help="K random negatives per positive (cheap variant). "
                        "-1 = use all B-1 negatives (full B^2). Try 4-8 for "
                        "B=16 to cut cost ~2-3x.")

    # --- EGW solver knobs (exposed for sweeping) --------------------------
    p.add_argument("--egw-n-restarts", type=int, default=_d.egw_n_restarts,
                   help="solver feature-warmstart restarts; identity + "
                        "sorted-row always run. Higher = more stable "
                        "gradient, roughly linear cost.")
    p.add_argument("--egw-epsilon-rel", type=float, default=_d.egw_epsilon_rel,
                   help="Sinkhorn temperature as fraction of median D². "
                        "Sharper (0.01) = more discriminative but Sinkhorn "
                        "can become unstable.")
    p.add_argument("--egw-symmetry-break", type=float, default=_d.egw_symmetry_break,
                   help="log-space noise scale in warmstart. 0 disables "
                        "(more deterministic but may stall on point-"
                        "transitive codes); 1.0 is default.")
    p.add_argument("--egw-max-inner", type=int, default=_d.egw_max_inner,
                   help="inner Sinkhorn iters (fixed, no early exit)")
    p.add_argument("--egw-max-outer", type=int, default=_d.egw_max_outer,
                   help="mirror-descent outer iters cap")
    p.add_argument("--egw-eps-anneal-steps", type=int, default=_d.egw_eps_anneal_steps,
                   help="number of annealing stages; 1 = no anneal (fastest)")
    p.add_argument("--egw-identity-init", action=argparse.BooleanOptionalAction,
                   default=_d.egw_identity_init,
                   help="include diag(μ) restart (safety net for G_p≈G_t)")
    p.add_argument("--egw-sorted-row-init", action=argparse.BooleanOptionalAction,
                   default=_d.egw_sorted_row_init,
                   help="include sorted-Gram-row-OT restart")
    p.add_argument("--egw-use-compile", action=argparse.BooleanOptionalAction,
                   default=_d.egw_use_compile,
                   help="torch.compile the inner Sinkhorn")

    # Optim
    p.add_argument("--lr", type=float, default=_d.lr)
    p.add_argument("--weight-decay", type=float, default=_d.weight_decay)
    p.add_argument("--grad-clip", type=float, default=_d.grad_clip)
    p.add_argument("--warmup-steps", type=int, default=_d.warmup_steps)
    # Runtime
    p.add_argument("--log-dir", type=str, default=_d.log_dir)
    p.add_argument("--run-name", type=str, default=_d.run_name)
    p.add_argument("--seed", type=int, default=_d.seed)
    p.add_argument("--device", type=str, default=_d.device,
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=_d.num_workers)
    # Data pipeline
    p.add_argument("--archive-dirs", nargs="*", dest="archive_dirs",
                   default=_d.archive_dirs,
                   help="one or more archive directories to load spherical codes from; "
                        "each directory is sampled with equal probability")
    p.add_argument("--perturb-prob", type=float, default=_d.perturb_prob,
                   help="probability of applying perturbation augmentation per sample")
    p.add_argument("--max-perturb-rounds", type=int, default=_d.max_perturb_rounds,
                   help="maximum number of perturbation rounds (geometric draw)")
    p.add_argument("--optimize-prob", type=float, default=_d.optimize_prob,
                   help="probability of applying quick Riemannian optimization per sample")
    p.add_argument("--perturb-sigma-lo", type=float, default=_d.perturb_sigma_lo,
                   help="lower bound of log-uniform perturbation noise range")
    p.add_argument("--perturb-sigma-hi", type=float, default=_d.perturb_sigma_hi,
                   help="upper bound of log-uniform perturbation noise range")
    p.add_argument("--quick-opt-steps-lo", type=int, default=_d.quick_opt_steps_lo,
                   help="minimum Riemannian repulsion steps in quick_optimize")
    p.add_argument("--quick-opt-steps-hi", type=int, default=_d.quick_opt_steps_hi,
                   help="maximum Riemannian repulsion steps in quick_optimize")
    p.add_argument("--source-weight-random", type=float, default=_d.source_weight_random,
                   help="unnormalized weight for pure-random source")
    p.add_argument("--source-weight-archive", type=float, default=_d.source_weight_archive,
                   help="unnormalized weight for raw archive source")
    p.add_argument("--source-weight-fps", type=float, default=_d.source_weight_fps,
                   help="unnormalized weight for furthest-point-sampled archive source")
    # PCGrad
    p.add_argument("--use-pcgrad", action=argparse.BooleanOptionalAction,
                   default=_d.use_pcgrad,
                   help="treat each batch sample as a separate PCGrad task; "
                        "projects conflicting per-sample gradients before the "
                        "update. O(batch_size)x slower; disables AMP.")
    # Checkpoint / resume
    p.add_argument("--ckpt", dest="ckpt_path", type=str, default="",
                   help="path to a checkpoint to resume/fine-tune from; model weights "
                        "and config are loaded from this checkpoint")
    p.add_argument("--use-default-config", action="store_true", default=False,
                   help="when --ckpt is given, use the default config instead of the "
                        "checkpoint's saved config; CLI args still override both")
    args = p.parse_args()
    args_dict = vars(args)

    # Architecture parameters that cannot be changed on a pretrained model
    _ARCH_PARAMS = frozenset({
        "feat_dim", "latent_dim", "gt_d_pair", "gt_d_msg", "gt_layers",
        "dec_dim", "dec_heads", "dec_layers", "pe_dim", "D_max", "N_max",
    })

    # Detect which args were explicitly passed by comparing against argparse defaults.
    # An arg that equals its default is assumed to be the implicit default, not an
    # intentional override.  This is used to merge checkpoint config with CLI flags.
    _skip_dests = {"help", "use_default_config"}
    explicitly_set = {
        action.dest
        for action in p._actions
        if action.dest not in _skip_dests
        and args_dict.get(action.dest) != action.default
    }

    ckpt_path = args_dict.get("ckpt_path", "")
    use_default_config = args_dict.get("use_default_config", False)

    # Determine base config: checkpoint config or fresh defaults
    if ckpt_path and not use_default_config:
        _ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base_cfg = _cfg_from_ckpt(_ckpt_data["cfg"])
    else:
        base_cfg = _d

    # Warn if architecture params were explicitly overridden — they are fixed
    # by the pretrained model's weights and changing them would be silently wrong.
    if ckpt_path:
        arch_overrides = explicitly_set & _ARCH_PARAMS
        if arch_overrides:
            import warnings
            warnings.warn(
                "Architecture parameters passed via CLI cannot be changed on a "
                f"pretrained model and will be ignored: {sorted(arch_overrides)}.",
                UserWarning,
            )
            explicitly_set = explicitly_set - _ARCH_PARAMS

    # Apply explicit CLI overrides on top of the base config
    base_dict = asdict(base_cfg)
    for dest in explicitly_set:
        if dest in base_dict:
            base_dict[dest] = args_dict[dest]
    base_dict["ckpt_path"] = ckpt_path

    valid_keys = set(asdict(_d).keys())
    return Config(**{k: v for k, v in base_dict.items() if k in valid_keys})


if __name__ == "__main__":
    cfg = parse_args()
    run_dir = train(cfg)
    print(f"\nAll artifacts saved to: {run_dir}")