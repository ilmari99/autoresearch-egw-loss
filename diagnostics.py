"""diagnostics.py
==============

Logging and diagnostic computations for spherical code autoencoder
training runs.  Plotting functions have been removed; this module is
plot-free so that it can be imported without matplotlib.

Public API
----------
  Logging:
    - setup_logger(run_dir)

  Model-probing diagnostics:
    - recon_breakdown(enc, dec, cfg, device, loss_sw, ...)  -> grid, N_bins, D_bins
    - smoothness_curve(enc, cfg, device, ...)  -> sigmas, mean, std

  N×D grid evaluation:
    - evaluate_nd_grid(enc, dec, cfg, device, loss_sw, ...)  -> grid, N_values, D_values
    - nd_grid_summary(grid, N_values, D_values)              -> str

  Gradient diagnostics:
    - compute_grad_norms(enc, dec) -> dict[str, float]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from data import sample_spherical_code, pad_batch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(run_dir: Path, name: str = "sphere_ae") -> logging.Logger:
    """Create a logger that writes to both *run_dir*/train.log and stderr."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh); logger.addHandler(sh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reconstruction breakdown by (N, D) bin
# ---------------------------------------------------------------------------

@torch.no_grad()
def recon_breakdown(
    enc_model,
    dec_model,
    cfg,
    device,
    *,
    loss_sw: Callable,
    n_per_bin: int = 6,
):
    """Compute mean reconstruction error on a grid of (N, D) bins.

    ``cfg`` must expose: N_min, N_max, D_min, D_max.

    ``loss_sw`` is the sliced-Wasserstein callable::

        loss_sw(pred, target, pred_mask, target_mask, n_slices=...) -> scalar

    Returns (grid, N_bins, D_bins).
    """
    def edges(lo, hi, n):
        es = np.linspace(lo, hi, n + 1)
        es = sorted(set(int(round(e)) for e in es))
        return [(es[i], max(es[i + 1], es[i] + 1)) for i in range(len(es) - 1)]

    N_bins = edges(cfg.N_min, cfg.N_max, 4)
    D_bins = edges(cfg.D_min, cfg.D_max, 4)
    enc_model.eval(); dec_model.eval()
    grid = np.zeros((len(N_bins), len(D_bins)))
    for i, (Nlo, Nhi) in enumerate(N_bins):
        for j, (Dlo, Dhi) in enumerate(D_bins):
            errs = []
            for _ in range(n_per_bin):
                N = min(torch.randint(Nlo, Nhi + 1, (1,)).item(), cfg.N_max)
                D = min(torch.randint(Dlo, Dhi + 1, (1,)).item(), cfg.D_max)
                c = sample_spherical_code(N, D)
                x, mask, Ds, Ns = pad_batch([c], cfg.D_max, cfg.N_max)
                x, mask, Ds, Ns = [t.to(device) for t in (x, mask, Ds, Ns)]
                z = enc_model(x, mask, Ds)
                pred, pred_mask = dec_model(z, Ns, Ds)
                errs.append(loss_sw(
                    pred, x, pred_mask, mask, n_slices=128,
                    Ds=Ds,
                ).item())
            grid[i, j] = float(np.mean(errs))
    return grid, N_bins, D_bins


# ---------------------------------------------------------------------------
# Encoder smoothness curve
# ---------------------------------------------------------------------------

@torch.no_grad()
def smoothness_curve(
    enc_model,
    cfg,
    device,
    n_samples: int = 32,
    sigmas: Sequence[float] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
):
    """Measure ||delta z|| as a function of input perturbation scale *sigma*.

    ``cfg`` must expose: D_min, D_max, N_min, N_max.

    Returns (sigmas_arr, mean_arr, std_arr).
    """
    enc_model.eval()
    out = {s: [] for s in sigmas}
    for _ in range(n_samples):
        D = torch.randint(cfg.D_min, cfg.D_max + 1, (1,)).item()
        N = torch.randint(cfg.N_min, cfg.N_max + 1, (1,)).item()
        c = sample_spherical_code(N, D)
        x, mask, Ds, Ns = pad_batch([c], cfg.D_max, cfg.N_max)
        x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
        z0 = enc_model(x, mask, Ds)
        for sig in sigmas:
            delta = torch.randn_like(x) * sig
            delta[:, :, D:] = 0; delta[:, N:, :] = 0
            xp = x + delta
            xp[0, :N, :D] /= xp[0, :N, :D].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dz = (z0 - enc_model(xp, mask, Ds)).norm().item()
            out[sig].append(dz)
    mean = np.array([np.mean(out[s]) for s in sigmas])
    std = np.array([np.std(out[s]) for s in sigmas])
    return np.array(sigmas), mean, std


# ---------------------------------------------------------------------------
# N×D grid evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nd_grid(
    enc_model,
    dec_model,
    cfg,
    device,
    *,
    loss_sw: Callable,
    N_values: Sequence[int] | None = None,
    D_values: Sequence[int] | None = None,
    n_per_cell: int = 8,
):
    """Evaluate mean SW reconstruction loss on an explicit grid of (N, D) values.

    ``cfg`` must expose: N_min, N_max, D_min, D_max.

    Returns (grid, N_values_used, D_values_used) where ``grid[i, j]`` is the
    mean SW loss for ``(N_values_used[i], D_values_used[j])``.  Cells outside
    the cfg range are set to ``NaN``.
    """
    if N_values is None:
        N_values = sorted(set(
            int(round(v))
            for v in np.geomspace(cfg.N_min, cfg.N_max, 5)
        ))
    if D_values is None:
        D_values = sorted(set(
            int(round(v))
            for v in np.linspace(cfg.D_min, cfg.D_max, 6)
        ))

    enc_model.eval(); dec_model.eval()
    grid = np.full((len(N_values), len(D_values)), np.nan)

    for i, N in enumerate(N_values):
        if N < cfg.N_min or N > cfg.N_max:
            continue
        for j, D in enumerate(D_values):
            if D < cfg.D_min or D > cfg.D_max:
                continue
            errs = []
            for _ in range(n_per_cell):
                c = sample_spherical_code(N, D)
                x, mask, Ds, Ns = pad_batch([c], cfg.D_max, cfg.N_max)
                x, mask, Ds, Ns = [t.to(device) for t in (x, mask, Ds, Ns)]
                z = enc_model(x, mask, Ds)
                pred, pred_mask = dec_model(z, Ns, Ds)
                errs.append(loss_sw(
                    pred, x, pred_mask, mask, n_slices=128, Ds=Ds,
                ).item())
            grid[i, j] = float(np.mean(errs))

    return grid, list(N_values), list(D_values)


def nd_grid_summary(
    grid: np.ndarray,
    N_values: list,
    D_values: list,
    top_k: int = 3,
) -> str:
    """Return a compact text summary of (N, D) reconstruction discrepancy.

    Reports the best / worst cells and the worst-to-best ratio so the caller
    can log it directly.
    """
    cells = [
        (grid[i, j], N_values[i], D_values[j])
        for i in range(len(N_values))
        for j in range(len(D_values))
        if not np.isnan(grid[i, j])
    ]
    if not cells:
        return "nd_grid: no valid cells"
    cells.sort(key=lambda t: t[0])
    best = cells[:top_k]
    worst = cells[-top_k:]
    ratio = cells[-1][0] / max(cells[0][0], 1e-12)
    lines = [
        f"nd_grid  ratio_worst/best={ratio:.1f}",
        "  best:  " + "  ".join(
            f"N={n},D={d} -> {v:.4f}" for v, n, d in best
        ),
        "  worst: " + "  ".join(
            f"N={n},D={d} -> {v:.4f}" for v, n, d in reversed(worst)
        ),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------

def _module_grad_norm(module: torch.nn.Module) -> float:
    """L2 norm of all gradients in a module (0.0 if none)."""
    grads = [p.grad.data for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.stack([g.norm(2) ** 2 for g in grads]).sum()
    return total.sqrt().item()


def _param_grad_norm(param_or_module) -> float:
    """Grad norm for an nn.Module or a bare nn.Parameter."""
    if isinstance(param_or_module, torch.nn.Parameter):
        return param_or_module.grad.norm().item() if param_or_module.grad is not None else 0.0
    return _module_grad_norm(param_or_module)


def compute_grad_norms(enc_model, dec_model) -> dict[str, float]:
    """Return per-component gradient norms for encoder and decoder sub-modules.

    Must be called **after** ``backward()`` and ``scaler.unscale_(opt)``
    but **before** ``clip_grad_norm_``.

    If the model classes expose ``diagnostic_submodules() -> dict[str, Module]``
    those are used for per-component breakdown.  Otherwise only totals are
    reported.

    Returns a dict with keys like ``grad/enc_backbone``, ``grad/dec_layers``, etc.
    """
    norms: dict[str, float] = {}

    # ---- encoder components ----
    if hasattr(enc_model, "diagnostic_submodules"):
        for name, sub in enc_model.diagnostic_submodules().items():
            norms[f"grad/enc_{name}"] = _param_grad_norm(sub)
    norms["grad/enc_total"] = _module_grad_norm(enc_model)

    # ---- decoder components ----
    if hasattr(dec_model, "diagnostic_submodules"):
        for name, sub in dec_model.diagnostic_submodules().items():
            norms[f"grad/dec_{name}"] = _param_grad_norm(sub)
    norms["grad/dec_total"] = _module_grad_norm(dec_model)

    return norms
