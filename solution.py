"""
solution.py
===========

Solver module for the spherical-code autoencoder.

The autonomous research agent may edit this file (alongside loss.py) to
improve the encoder/decoder architecture, training procedure, or loss wiring.
All other files in the repository are part of the fixed evaluation harness and
must not be modified during experiments.

Architecture
------------
Encoder — 2-IGN (2-Invariant Graph Network), SO(D)-invariant:
  Gram matrix G = X Xᵀ collapses rotation symmetry; GramEdgeEmbed maps each
  scalar G_ij to a pair feature vector; T × PairRefineLayer steps perform 2-WL
  message passing; features are pooled and concatenated with sinusoidal
  (log N, D) encodings before a final MLP head produces the fixed-dim latent z.

Decoder — AdaLN self-attention with anonymous slot queries:
  Slots start as a shared learned prototype plus i.i.d. noise for symmetry
  breaking.  AdaLNSelfAttnLayer blocks condition the computation on z via
  Adaptive LayerNorm (DiT-style).  A final linear head maps to D_max coords.

Loss (via loss.py)
------------------
Primary: Entropic Gromov-Wasserstein (EGW) reconstruction.
Optional: contrastive EGW (anti-collapse), VICReg latent regularisation,
  denoising smoothness penalty.

Public API (asserted by test_solution.py)
-----------------------------------------
  Config                  — dataclass with all hyperparameters
  SphereCodeEncoder       — 2-IGN encoder
  SphereCodeDecoder       — AdaLN self-attention decoder
  build_training_state    — set up encoder, decoder, optimiser, scheduler, EMA
  train_one_step          — one forward/backward/step; returns scalar metrics
  build_val_loss_fn       — return the validation loss callable
  save_checkpoint         — persist encoder + decoder + config + step
  load_checkpoint         — restore (encoder, decoder, Config) from disk
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field, fields as _dc_fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

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
    gt_d_pair: int = 64
    gt_d_msg: int = 32
    gt_layers: int = 4
    gt_use_checkpoint: bool = False
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
    w_var: float = 0.0
    w_cov: float = 0.0

    # Contrastive EGW (anti-collapse)
    w_contrastive: float = 0.0
    contrastive_tau: float = 0.1
    contrastive_subsample: int = -1

    # EGW solver knobs
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
    log_dir: str = "./autoresearch-runs"
    # MODIFY THE "run_name" to be exp<n>_<name>
    run_name: str = "exp0_baseline"
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0
    ckpt_path: str = ""

    # Data pipeline
    archive_dirs: list = field(
        default_factory=lambda: ["combined_points_archive", "spherical_code_archive"]
    )
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


# =============================================================================
# 2-IGN pair-feature encoder (SO(D)-invariant)
# =============================================================================

class GramEdgeEmbed(nn.Module):
    """Map scalar G_{ij} ∈ [-1, 1] to pair feature p⁰_{ij} ∈ ℝ^{d_p}."""

    def __init__(self, d_p: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_p), nn.GELU(), nn.Linear(d_p, d_p),
        )

    def forward(self, G: torch.Tensor) -> torch.Tensor:
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
        m_in = self.mlp_in(p)
        m_in.mul_(pair_mask)
        m_out = self.mlp_out(p)
        m_out.mul_(pair_mask)

        m_in = m_in.permute(0, 3, 1, 2)
        m_out = m_out.permute(0, 3, 1, 2)
        msg = torch.matmul(m_in, m_out)
        msg = msg.permute(0, 2, 3, 1)
        msg.div_(n_b[:, None, None, None])

        d_p = p.shape[-1]
        w_p = self.mlp_u[0].weight[:, :d_p]
        w_m = self.mlp_u[0].weight[:, d_p:]
        b_u = self.mlp_u[0].bias

        h = F.linear(p, w_p)
        h.add_(F.linear(msg, w_m, b_u))
        del m_in, m_out, msg

        h = self.mlp_u[1](h)
        h = self.mlp_u[2](h)
        p = self.norm(p + h)
        p.mul_(pair_mask)
        return p


class GramPairEncoder(nn.Module):
    """GramEdgeEmbed + T × PairRefineLayer.

    With ``use_checkpoint=True`` each refinement layer is wrapped in
    ``torch.utils.checkpoint`` — activations for the O(N²·d_p) tensors are
    dropped and recomputed during backward, cutting peak memory by ~gt_layers×.
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
        pair_mask_bool = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask = pair_mask_bool.unsqueeze(-1).to(G.dtype)
        n_b = mask.sum(dim=1).clamp_min(1).to(G.dtype)
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
# Encoder
# =============================================================================

_SINUSOIDAL_FREQS: dict[int, torch.Tensor] = {}


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
            nn.LayerNorm(cfg.latent_dim),
        )

    def diagnostic_submodules(self) -> dict[str, "nn.Module | nn.Parameter"]:
        return {
            "backbone": self.backbone,
            "pair_proj": self.pair_proj,
            "nd_proj": self.nd_proj,
            "head": self.head,
        }

    def forward(self, x, mask, Ds):
        N_active_max = int(mask.sum(dim=1).max().item())
        if N_active_max < x.shape[1]:
            x = x[:, :N_active_max, :]
            mask = mask[:, :N_active_max]

        D_active_max = int(Ds.max().item())
        if D_active_max < x.shape[2]:
            x = x[:, :, :D_active_max]

        G = x @ x.transpose(-1, -2)
        p, pair_mask = self.backbone(G, mask)
        phi = self.pair_proj(p) * pair_mask

        n_b = mask.sum(dim=1).clamp_min(1).to(p.dtype)
        n_pairs = n_b * n_b
        pooled = phi.sum(dim=(1, 2)) / n_pairs[:, None]

        n_enc = sinusoidal_encoding(torch.log(n_b), self.cfg.pe_dim)
        d_enc = sinusoidal_encoding(Ds.float(), self.cfg.pe_dim)
        nd_feat = self.nd_proj(torch.cat([n_enc, d_enc], dim=-1))

        return self.head(torch.cat([pooled, nd_feat], dim=-1))


# =============================================================================
# Decoder
# =============================================================================

class AdaLNSelfAttnLayer(nn.Module):
    """Self-attention + FFN conditioned on z via Adaptive LayerNorm (DiT-style).

    AdaLN-Zero: residual gate alphas initialised near zero so the network starts
    as near-identity and gradually learns to use self-attention and FFN.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, z_dim: int):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(z_dim, 6 * d_model),
        )
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
        g1, b1, a1, g2, b2, a2 = self.adaLN_modulation(z).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        h = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + a1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        h = self.ffn(h)
        x = x + a2.unsqueeze(1) * h
        return x


class SphereCodeDecoder(nn.Module):
    """AdaLN self-attention decoder with anonymous (unordered) slots.

    Slots start as a shared prototype plus i.i.d. noise — no slot has a
    preferred identity, making the decoder permutation-equivariant.
    Self-attention lets slots coordinate; z conditions via AdaLN.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.slot_proto = nn.Parameter(torch.randn(cfg.dec_dim) * 0.02)
        self.noise_log_scale = nn.Parameter(torch.tensor(0.0))
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
        return {
            "slot_proto": self.slot_proto,
            "z_to_slots": self.z_to_slots,
            "layers": self.layers,
            "norm_out": self.norm_out,
            "out": self.out,
        }

    def forward(self, z, Ns, Ds):
        B = z.shape[0]
        device = z.device
        N_max, D_max = self.cfg.N_max, self.cfg.D_max

        N_active = int(Ns.max().item())
        D_active = int(Ds.max().item())

        noise_scale = self.noise_log_scale.exp()
        noise = torch.randn(B, N_active, self.cfg.dec_dim, device=device) * noise_scale
        z_bias = self.z_to_slots(z).unsqueeze(1)
        slots = self.slot_proto.unsqueeze(0).unsqueeze(0).expand(B, N_active, -1) + noise + z_bias
        h = slots

        for layer in self.layers:
            h = layer(h, z)
        h = self.norm_out(h)
        raw = self.out(h)[:, :, :D_active]

        pos = torch.arange(N_active, device=device).unsqueeze(0).expand(B, -1)
        mask = pos < Ns.unsqueeze(1)
        D_mask = (torch.arange(D_active, device=device).unsqueeze(0) < Ds.unsqueeze(1))
        raw = raw * D_mask.unsqueeze(1)
        pred = raw * mask.unsqueeze(-1)

        norms = pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        pred = (pred / norms) * mask.unsqueeze(-1)

        if N_active < N_max or D_active < D_max:
            pred = F.pad(pred, (0, D_max - D_active, 0, N_max - N_active))
            mask = F.pad(mask, (0, N_max - N_active))
        return pred, mask


# =============================================================================
# Loss functions
# =============================================================================

def egw_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    egw_cfg: EGWConfig,
    tau: float = 0.1,
    subsample: int = -1,
):
    """B² contrastive EGW loss.

    Returns (l_contrastive, l_rec, L) where L is the B×B EGW matrix (detached).
    l_rec = diagonal mean = same quantity as the plain per-sample EGW loss.
    """
    B, N_p, D = pred.shape
    _, N_t, _ = target.shape
    device = pred.device

    if subsample >= 0 and subsample < B - 1:
        K = int(subsample)
        pos = torch.arange(B, device=device).unsqueeze(1)
        neg_raw = torch.stack([
            torch.randperm(B - 1, device=device)[:K] for _ in range(B)
        ])
        neg = neg_raw + (neg_raw >= pos).long()
        j_idx = torch.cat([pos, neg], dim=1)
        pred_stk = pred.unsqueeze(1).expand(B, K + 1, N_p, D)
        target_stk = target[j_idx]
        pm_stk = pred_mask.unsqueeze(1).expand(B, K + 1, N_p)
        tm_stk = target_mask[j_idx]
        flat_shape = (B * (K + 1),)
        pred_flat = pred_stk.reshape(*flat_shape, N_p, D)
        target_flat = target_stk.reshape(*flat_shape, N_t, D)
        pm_flat = pm_stk.reshape(*flat_shape, N_p)
        tm_flat = tm_stk.reshape(*flat_shape, N_t)
        L_flat = egw_gram_loss(pred_flat, target_flat, pm_flat, tm_flat,
                               cfg=egw_cfg, reduction="none")
        L_sub = L_flat.view(B, K + 1)
        logits = -L_sub / tau
        labels = torch.zeros(B, dtype=torch.long, device=device)
        l_contrastive = F.cross_entropy(logits, labels)
        l_rec = L_sub[:, 0].mean()
        L_full = torch.full((B, B), float("nan"), device=device)
        L_full.scatter_(1, j_idx, L_sub.detach())
        return l_contrastive, l_rec, L_full

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

    Variance hinge: relu(1 − std_j) per dimension.
    Covariance: squared off-diagonal correlations.
    EMA-smoothed statistics for stability at small batch sizes.
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

    live_mean = z.mean(dim=0)
    mean_st = live_mean + (ema_mean - live_mean).detach()
    z_c_grad = z - mean_st

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


def denoising_smoothness(
    encoder, x: torch.Tensor, mask: torch.Tensor, Ds: torch.Tensor,
    z_clean: torch.Tensor, sigma: float = 0.02,
) -> torch.Tensor:
    """||z_clean - phi(x + sigma*eps)||² where eps lives in the active D subspace
    and the perturbed points are re-projected to the sphere."""
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
# EGW config helpers
# =============================================================================

def _make_egw_cfg(cfg: Config) -> EGWConfig:
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


def build_val_loss_fn(cfg: Config):
    """Return a loss callable suitable for evaluation.py's ``evaluate()``."""
    egw_cfg = _make_egw_cfg(cfg)
    def _fn(pred, target, pred_mask, target_mask):
        return egw_gram_loss(pred, target, pred_mask, target_mask, cfg=egw_cfg)
    def _per_sample(pred, target, pred_mask, target_mask):
        return egw_gram_loss(
            pred,
            target,
            pred_mask,
            target_mask,
            cfg=egw_cfg,
            reduction="none",
        )
    _fn.per_sample = _per_sample
    return _fn


# =============================================================================
# Gradient diagnostics (inlined to keep solution.py self-contained)
# =============================================================================

def _module_grad_norm(module: nn.Module) -> float:
    grads = [p.grad.data for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.stack([g.norm(2) ** 2 for g in grads]).sum()
    return total.sqrt().item()


def _param_grad_norm(param_or_module) -> float:
    if isinstance(param_or_module, nn.Parameter):
        return param_or_module.grad.norm().item() if param_or_module.grad is not None else 0.0
    return _module_grad_norm(param_or_module)


def _compute_grad_norms(enc: nn.Module, dec: nn.Module) -> dict[str, float]:
    """Per-component gradient norms. Call after backward() and unscale_()."""
    norms: dict[str, float] = {}
    if hasattr(enc, "diagnostic_submodules"):
        for name, sub in enc.diagnostic_submodules().items():
            norms[f"grad/enc_{name}"] = _param_grad_norm(sub)
    norms["grad/enc_total"] = _module_grad_norm(enc)
    if hasattr(dec, "diagnostic_submodules"):
        for name, sub in dec.diagnostic_submodules().items():
            norms[f"grad/dec_{name}"] = _param_grad_norm(sub)
    norms["grad/dec_total"] = _module_grad_norm(dec)
    return norms


# =============================================================================
# Smoothness weight schedule
# =============================================================================

def _smooth_weight(cfg: Config, step: int) -> float:
    smooth_delay_steps = int(cfg.smooth_delay_frac * cfg.train_steps)
    smooth_ramp_steps = max(1, min(300, cfg.train_steps - smooth_delay_steps))
    if step < smooth_delay_steps:
        return 0.0
    ramp_step = step - smooth_delay_steps
    frac = min(ramp_step / smooth_ramp_steps, 1.0)
    return cfg.w_smooth_init + frac * (cfg.w_smooth_final - cfg.w_smooth_init)


# =============================================================================
# Training state
# =============================================================================

@dataclass
class TrainingState:
    """All mutable state for one training run."""
    cfg: Config
    enc: SphereCodeEncoder
    dec: SphereCodeDecoder
    opt: Any           # torch.optim.AdamW
    sched: Any         # torch.optim.lr_scheduler.LambdaLR
    scaler: Any        # torch.amp.GradScaler
    device: str
    use_amp: bool
    vicreg_ema_mean: torch.Tensor
    vicreg_ema_var: torch.Tensor
    vicreg_ema_cov: torch.Tensor
    egw_cfg: EGWConfig
    use_contrastive: bool
    all_params: list
    zero: torch.Tensor


def build_training_state(
    cfg: Config,
    device: str,
    ckpt_path: str = "",
) -> TrainingState:
    """Construct all training objects from *cfg*.

    Parameters
    ----------
    cfg :
        Fully-specified Config.  Architecture and loss knobs come from here.
    device :
        ``"cuda"`` or ``"cpu"``.
    ckpt_path :
        If non-empty, load encoder/decoder weights from this checkpoint
        (fine-tuning / warm-start).  Architecture must match cfg.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    enc = SphereCodeEncoder(cfg).to(device)
    dec = SphereCodeDecoder(cfg).to(device)

    if ckpt_path:
        _ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc.load_state_dict(_ckpt["encoder"])
        dec.load_state_dict(_ckpt["decoder"])

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: min((s + 1) / max(cfg.warmup_steps, 1), 1.0)
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    _D = cfg.latent_dim
    return TrainingState(
        cfg=cfg,
        enc=enc,
        dec=dec,
        opt=opt,
        sched=sched,
        scaler=scaler,
        device=device,
        use_amp=use_amp,
        vicreg_ema_mean=torch.zeros(_D, device=device),
        vicreg_ema_var=torch.ones(_D, device=device),
        vicreg_ema_cov=torch.zeros(_D, _D, device=device),
        egw_cfg=_make_egw_cfg(cfg),
        use_contrastive=cfg.w_contrastive > 0.0,
        all_params=list(enc.parameters()) + list(dec.parameters()),
        zero=torch.zeros((), device=device),
    )


# =============================================================================
# Training step implementations
# =============================================================================

def _train_step_standard(
    state: TrainingState,
    x: torch.Tensor, mask: torch.Tensor,
    Ds: torch.Tensor, Ns: torch.Tensor,
    w_sm: float,
) -> dict:
    cfg = state.cfg
    state.opt.zero_grad(set_to_none=True)

    with torch.amp.autocast(enabled=state.use_amp, device_type=state.device):
        z = state.enc(x, mask, Ds)
        pred, pred_mask = state.dec(z, Ns, Ds)

        if state.use_contrastive:
            l_con, l_egw, _ = egw_contrastive_loss(
                pred, x, pred_mask, mask,
                egw_cfg=state.egw_cfg,
                tau=cfg.contrastive_tau,
                subsample=cfg.contrastive_subsample,
            )
        else:
            l_egw = egw_gram_loss(pred, x, pred_mask, mask, cfg=state.egw_cfg)
            l_con = state.zero

        l_var, l_cov = vicreg_latent(
            z, state.vicreg_ema_mean, state.vicreg_ema_var, state.vicreg_ema_cov,
        )
        main_loss = (
            cfg.w_egw * l_egw
            + cfg.w_contrastive * l_con
            + cfg.w_var * l_var
            + cfg.w_cov * l_cov
        )

    state.scaler.scale(main_loss).backward()

    l_smooth = state.zero
    if w_sm > 0:
        with torch.amp.autocast(enabled=state.use_amp, device_type=state.device):
            l_smooth = denoising_smoothness(
                state.enc, x, mask, Ds, z.detach(), sigma=cfg.smooth_sigma,
            )
        state.scaler.scale(w_sm * l_smooth).backward()

    state.scaler.unscale_(state.opt)
    grad_norms = _compute_grad_norms(state.enc, state.dec)
    gn = torch.nn.utils.clip_grad_norm_(state.all_params, cfg.grad_clip)
    state.scaler.step(state.opt)
    state.scaler.update()
    state.sched.step()

    return {
        "l_egw": l_egw.item(),
        "l_con": l_con.item(),
        "l_smooth": l_smooth.item(),
        "l_var": l_var.item(),
        "l_cov": l_cov.item(),
        "grad_norm": float(gn),
        "lr": state.opt.param_groups[0]["lr"],
        "w_smooth": w_sm,
        "mean_N": Ns.float().mean().item(),
        "mean_D": Ds.float().mean().item(),
        **grad_norms,
    }


def train_one_step(
    state: TrainingState,
    batch: tuple,
    step: int,
) -> dict:
    """Run one full forward/backward/update step.

    Parameters
    ----------
    state :
        Mutable training state from ``build_training_state``.
    batch :
        ``(x, mask, Ds, Ns)`` tensors already on the correct device.
    step :
        Current step index (0-based), used for the smoothness weight schedule.

    Returns
    -------
    dict
        Scalar diagnostics for this step. Any finite scalar values returned here
        are averaged over each reporting window and printed by
        ``test_solution.py`` during training.
    """
    x, mask, Ds, Ns = batch
    w_sm = _smooth_weight(state.cfg, step)
    return _train_step_standard(state, x, mask, Ds, Ns, w_sm)


# =============================================================================
# Checkpoint I/O
# =============================================================================

def save_checkpoint(
    state: TrainingState,
    path: "str | Path",
    step: int,
    extra: "dict | None" = None,
) -> None:
    """Save encoder, decoder, config, and step to *path*."""
    payload = {
        "encoder": state.enc.state_dict(),
        "decoder": state.dec.state_dict(),
        "cfg": asdict(state.cfg),
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _cfg_from_ckpt(cfg_dict: dict) -> Config:
    """Build a Config from a checkpoint dict, filtering unknown keys."""
    valid = {f.name for f in _dc_fields(Config)}
    return Config(**{k: v for k, v in cfg_dict.items() if k in valid})


def load_checkpoint(
    ckpt_path: "str | Path",
    device: str = "auto",
) -> tuple[SphereCodeEncoder, SphereCodeDecoder, Config]:
    """Load a saved checkpoint.

    Returns
    -------
    encoder, decoder, cfg
        Both models are in eval mode on the resolved device.
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
