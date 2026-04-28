"""
diagnostics.py
==============

Logging, plotting, and diagnostic computations for spherical code autoencoder
training runs.

All public functions accept duck-typed ``cfg`` objects and, where needed,
loss callables injected by the caller — keeping this module decoupled from
the training script's Config / loss implementations.

Public API
----------
  Logging:
    - setup_logger(run_dir)

  Metric-series plots (from metrics_log):
    - plot_curves(metrics_log, run_dir)
    - plot_latent_stats(metrics_log, run_dir)

  Model-probing diagnostics:
    - recon_breakdown(enc, dec, cfg, device, loss_sw, ...)  -> grid, N_bins, D_bins
    - plot_recon_breakdown(enc, dec, cfg, device, run_dir, loss_sw)
    - smoothness_curve(enc, cfg, device, ...)  -> sigmas, mean, std
    - plot_smoothness(enc, cfg, device, run_dir)

  N×D grid evaluation:
    - evaluate_nd_grid(enc, dec, cfg, device, loss_sw, ...)  -> grid, N_values, D_values
    - plot_nd_grid(enc, dec, cfg, device, run_dir, loss_sw)  -> Path
    - nd_grid_summary(grid, N_values, D_values)              -> str

  Gradient diagnostics:
    - compute_grad_norms(enc, dec) -> dict[str, float]
    - plot_grad_norms(metrics_log, run_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# Helpers
# ---------------------------------------------------------------------------

def _series(metrics_log: list, key: str):
    return [r.get(key, np.nan) for r in metrics_log]


def _steps(metrics_log: list):
    return [r["step"] for r in metrics_log]


# ---------------------------------------------------------------------------
# Training-curve plots
# ---------------------------------------------------------------------------

def plot_curves(metrics_log: list, run_dir: Path) -> Path:
    """Plot loss curves and invariance residuals from *metrics_log*.

    Returns the path to the saved figure.
    """
    steps = _steps(metrics_log)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, _series(metrics_log, "train_egw"), label="train")
    axes[0, 0].plot(steps, _series(metrics_log, "val_egw"), label="val")
    axes[0, 0].set_title("EGW reconstruction")
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_yscale("log"); axes[0, 0].legend()

    axes[0, 1].plot(steps, _series(metrics_log, "train_smooth"), label="train")
    axes[0, 1].set_title("Denoising smoothness")
    _smooth = _series(metrics_log, "train_smooth")
    axes[0, 1].set_xlabel("step")
    if any(v > 0 for v in _smooth):
        axes[0, 1].set_yscale("log")
    axes[0, 1].legend()

    axes[1, 0].plot(steps, _series(metrics_log, "inv_perm"), label="perm residual")
    axes[1, 0].plot(steps, _series(metrics_log, "inv_rot"), label="rot residual")
    axes[1, 0].set_title("Invariance residuals (max |dz|)")
    axes[1, 0].set_xlabel("step"); axes[1, 0].set_yscale("log"); axes[1, 0].legend()

    axes[1, 1].plot(steps, _series(metrics_log, "lip_pert"))
    axes[1, 1].set_title("Local Lipschitz estimate  ||dz||/||dx||  (eps=1e-3)")
    axes[1, 1].set_xlabel("step")

    fig.tight_layout()
    path = run_dir / "curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_latent_stats(metrics_log: list, run_dir: Path) -> Path:
    """Plot latent-space distribution statistics from *metrics_log*.

    Returns the path to the saved figure.
    """
    steps = _steps(metrics_log)
    recs = metrics_log

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))

    axes[0].plot(steps, _series(recs, "val_latnorm"), label="norm")
    axes[0].plot(steps, _series(recs, "val_latstd"), label="std (per-dim)")
    axes[0].plot(steps, _series(recs, "val_latpdist"), label="pairwise dist")
    axes[0].set_title("Latent Distribution Stats")
    axes[0].set_xlabel("step")
    axes[0].legend()

    axes[1].plot(steps, _series(recs, "val_latdead"), color="red")
    axes[1].set_title("Fraction of Dead Latent Dims (std < 1e-4)")
    axes[1].set_xlabel("step")

    val_lats = _series(recs, "val_latstd")
    if len(val_lats) > 0 and not np.isnan(val_lats[-1]):
        axes[0].text(0.05, 0.95, f"final std: {val_lats[-1]:.3f}",
                     transform=axes[0].transAxes, verticalalignment='top')

    if sum(1 for r in recs if "train_smooth" in r) > 0:
        axes[2].plot(steps, _series(recs, "train_smooth"), label="smooth (train)")
        axes[2].set_title("Denoising Smoothness Penalty")
        axes[2].set_xlabel("step")
        _s2 = _series(recs, "train_smooth")
        if any(v > 0 for v in _s2):
            axes[2].set_yscale("log")

    if sum(1 for r in recs if "val_entropy_rank" in r) > 0:
        axes[3].plot(steps, _series(recs, "val_rank"), label="Numeric Rank")
        axes[3].plot(steps, _series(recs, "val_entropy_rank"), label="Entropy Rank (variance concentration)")
        axes[3].set_title("Latent Rank Metrics")
        axes[3].set_xlabel("step")
        axes[3].legend()

    if sum(1 for r in recs if "val_disc_ratio_min" in r) > 0:
        axes[4].plot(steps, _series(recs, "val_disc_ratio_min"),
                     label="disc ratio min")
        axes[4].plot(steps, _series(recs, "val_disc_ratio_mean"),
                     label="disc ratio mean")
        if sum(1 for r in recs if "val_disc_pass" in r) > 0:
            axes[4].plot(steps, _series(recs, "val_disc_pass"),
                         label="disc pass", linestyle="--", alpha=0.7)
        axes[4].set_title("Validation Discrimination")
        axes[4].set_xlabel("step")
        axes[4].legend()

    fig.tight_layout()
    path = run_dir / "latent_stats.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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


def plot_recon_breakdown(
    enc_model,
    dec_model,
    cfg,
    device,
    run_dir: Path,
    *,
    loss_sw: Callable,
) -> Path:
    """Plot a heat map of reconstruction error over (N, D) bins.

    Returns the path to the saved figure.
    """
    grid, Nb, Db = recon_breakdown(enc_model, dec_model, cfg, device,
                                   loss_sw=loss_sw)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(Db))); ax.set_xticklabels([f"D in {b}" for b in Db])
    ax.set_yticks(range(len(Nb))); ax.set_yticklabels([f"N in {b}" for b in Nb])
    ax.set_title("Sliced-W^2 reconstruction by (N, D) bin")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                    color="white" if grid[i, j] < grid.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    path = run_dir / "recon_by_bin.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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


def plot_smoothness(enc_model, cfg, device, run_dir: Path) -> Path:
    """Plot the encoder smoothness curve and save to *run_dir*.

    Returns the path to the saved figure.
    """
    sig, m, s = smoothness_curve(enc_model, cfg, device)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sig, m, yerr=s, marker="o")
    ax.plot(sig, sig * m[-1] / sig[-1], "--", alpha=0.4, label="linear reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sigma (input perturbation scale)")
    ax.set_ylabel("||dz||")
    ax.set_title("Encoder smoothness")
    ax.legend()
    fig.tight_layout()
    path = run_dir / "smoothness.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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


def plot_nd_grid(
    enc_model,
    dec_model,
    cfg,
    device,
    run_dir: Path,
    *,
    loss_sw: Callable,
    N_values: Sequence[int] | None = None,
    D_values: Sequence[int] | None = None,
    n_per_cell: int = 8,
) -> Path:
    """Evaluate and save a heat map of SW loss over an explicit (N, D) grid.

    Returns the path to the saved figure.
    """
    grid, N_vals, D_vals = evaluate_nd_grid(
        enc_model, dec_model, cfg, device,
        loss_sw=loss_sw, N_values=N_values, D_values=D_values,
        n_per_cell=n_per_cell,
    )
    fig, ax = plt.subplots(
        figsize=(max(7, len(D_vals) * 1.2), max(5, len(N_vals) * 0.9))
    )
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(D_vals)))
    ax.set_xticklabels([str(d) for d in D_vals])
    ax.set_yticks(range(len(N_vals)))
    ax.set_yticklabels([str(n) for n in N_vals])
    ax.set_xlabel("D (dimension)")
    ax.set_ylabel("N (points)")
    ax.set_title("Sliced-W\u00b2 reconstruction loss by (N, D)")
    vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
    vmin = float(np.nanmin(grid)) if not np.all(np.isnan(grid)) else 0.0
    mid = (vmax + vmin) / 2
    for i in range(len(N_vals)):
        for j in range(len(D_vals)):
            if not np.isnan(grid[i, j]):
                color = "white" if grid[i, j] < mid else "black"
                ax.text(j, i, f"{grid[i, j]:.3f}",
                        ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    path = run_dir / "nd_grid.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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


def plot_grad_norms(metrics_log: list, run_dir: Path) -> Path:
    """Plot per-component gradient norms over training.

    Returns the path to the saved figure.
    """
    steps = _steps(metrics_log)

    # Collect all grad/* keys that exist in the log
    grad_keys = sorted({k for r in metrics_log for k in r if k.startswith("grad/")})
    if not grad_keys:
        return run_dir / "grad_norms.png"  # nothing to plot

    enc_keys = [k for k in grad_keys if k.startswith("grad/enc_")]
    dec_keys = [k for k in grad_keys if k.startswith("grad/dec_")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: encoder components
    for k in enc_keys:
        label = k.replace("grad/enc_", "enc.")
        axes[0].plot(steps, _series(metrics_log, k), label=label)
    axes[0].set_title("Encoder gradient norms")
    axes[0].set_xlabel("step"); axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    # Panel 2: decoder components
    for k in dec_keys:
        label = k.replace("grad/dec_", "dec.")
        axes[1].plot(steps, _series(metrics_log, k), label=label)
    axes[1].set_title("Decoder gradient norms")
    axes[1].set_xlabel("step"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)

    # Panel 3: encoder total vs decoder total
    if "grad/enc_total" in grad_keys:
        axes[2].plot(steps, _series(metrics_log, "grad/enc_total"), label="encoder")
    if "grad/dec_total" in grad_keys:
        axes[2].plot(steps, _series(metrics_log, "grad/dec_total"), label="decoder")
    axes[2].set_title("Total gradient norms (enc vs dec)")
    axes[2].set_xlabel("step"); axes[2].set_yscale("log")
    axes[2].legend()

    fig.tight_layout()
    path = run_dir / "grad_norms.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
