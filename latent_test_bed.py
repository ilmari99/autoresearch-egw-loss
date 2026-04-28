"""
latent_test_bed.py
==================

Probe a trained spherical-code encoder's latent space.

Three diagnostics
-----------------
1. Linear ridge probe  — latent → analytic target.
   Fit on train, evaluate on test.
   Per-latent-dimension R² heatmap shows which dims carry which information.

2. MLP probe           — same target set, small 2-layer MLP.
   Early stopping on val; final evaluation on test.

3. Trajectory analysis — encode Coulomb-optimisation paths and check that
   the latent moves smoothly as the code improves.

Target families
---------------
- Core geometry/statistics (N, D, cosine stats, min distance, entropy rank)
- Riesz energies (log, 0.5, 1.0, 1.5, 2.0, 5.0)
- Spherical-harmonic power proxy at l = 1, 2, 3, 10, 20
- KNN distance summary stats for k = 2, 3, 5, 10
- Approximate kissing number near min distance
- Expanded normalized eigenvalue spectrum (top PROBE_EIGEN_COUNT)

Outputs are saved both globally and split by target category (plots + JSON).

Baselines
---------
Engineered (N, D) features: [N, D, N², D², N·D, log N, log D, N/D, √N, √D].
The latent is always evaluated on its own; N and D are never added to it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from data import (
    ArchiveCache,
    pad_batch,
    perturb_tangent,
    quick_optimize,
    sample_spherical_code,
)
from loss import dsq_from_gram

# =============================================================================
# Configuration
# =============================================================================

ROOT        = Path(__file__).resolve().parent
RUNS_DIR    = ROOT / "runs"
ARCHIVE_DIR = ROOT / "combined_points_archive"
OUTPUT_ROOT = ROOT / "latent_test_bed_outputs"

CKPT_PATH: str | None = None   # None → latest checkpoint in RUNS_DIR
DEVICE: str = "auto"
SEED: int = 42

# Probe dataset
PROBE_N_STEPS: int = 4                # N values on geomspace(N_min, N_max)
PROBE_D_STEPS: int = 4                # D values on linspace(D_min, D_max)
PROBE_SAMPLES_PER_SPEC: int = 20      # codes per (N, D) per source
PROBE_SOURCES = ("random", "optimized", "perturbed", "archive")
PROBE_OPTIMIZE_STEPS: int = 12
PROBE_PERTURB_SIGMA: float = 0.03
PROBE_EIGEN_COUNT: int = 20
PROBE_RIESZ_ORDERS = (0.5, 1.0, 1.5, 2.0, 5.0)
PROBE_HARMONIC_ORDERS = (1, 2, 3, 10, 20)
PROBE_KNN_ORDERS = (2, 3, 5, 10)
PROBE_KISSING_EPS: float = 0.01
ENCODE_BATCH_SIZE: int = 8

# Train / val / test fractions (of all samples, randomly shuffled)
TRAIN_FRAC: float = 0.60
VAL_FRAC:   float = 0.20   # test = remaining 0.20

# Linear probe
RIDGE_ALPHA: float = 1e-2

# MLP probe
MLP_HIDDEN  = (128, 64)
MLP_EPOCHS: int = 600
MLP_LR: float = 3e-3
MLP_PATIENCE: int = 40
MLP_BATCH: int = 64

# Trajectory
TRAJECTORY_STEPS: int = 30
TRAJECTORY_LR: float = 0.05

# Class-mix latent PCA diagnostic
CLASS_PCA_SAMPLES_PER_SPEC: int = 20
CLASS_PCA_LIGHT_OPT_STEPS: int = 3
CLASS_PCA_HEAVY_OPT_STEPS: int = 30
CLASS_PCA_PERTURB_SIGMAS = (0.01, 0.03, 0.08)

# =============================================================================
# Data types
# =============================================================================

INTEGER_TARGETS = {"N", "D"}


@dataclass
class Record:
    code: torch.Tensor   # (N, D) unit vectors
    N: int
    D: int
    source: str

# =============================================================================
# Utilities
# =============================================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def find_latest_ckpt(runs_dir: Path) -> Path:
    ckpts = sorted(runs_dir.glob("*/ckpt.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt.pt found under {runs_dir}")
    return ckpts[0]


def load_encoder(ckpt_path: Path, device: str):
    from solution import load_checkpoint as _load_checkpoint
    enc, _dec, cfg_obj = _load_checkpoint(ckpt_path, device=device)
    cfg = SimpleNamespace(**vars(cfg_obj))
    return enc, cfg

# =============================================================================
# Dataset
# =============================================================================

def _unique_ints(values) -> list[int]:
    return sorted({int(round(v)) for v in values})


def _sample_one(N: int, D: int, source: str,
                archive: ArchiveCache | None) -> torch.Tensor | None:
    if source == "random":
        return sample_spherical_code(N, D).cpu()
    if source == "optimized":
        base = sample_spherical_code(N, D)
        return quick_optimize(base, steps=PROBE_OPTIMIZE_STEPS, lr=0.05).cpu()
    if source == "perturbed":
        base = archive.sample_with_nd(N, D) if archive is not None else None
        if base is None:
            base = sample_spherical_code(N, D)
        return perturb_tangent(base.clone(), sigma=PROBE_PERTURB_SIGMA).cpu()
    if source == "archive":
        if archive is None:
            return None
        code = archive.sample_with_nd(N, D)
        return None if code is None else code.clone().cpu()
    raise ValueError(f"Unknown source: {source!r}")


def build_records(cfg, archive: ArchiveCache | None) -> list[Record]:
    n_vals = _unique_ints(np.geomspace(cfg.N_min, cfg.N_max, PROBE_N_STEPS))
    d_vals = _unique_ints(np.linspace(cfg.D_min, cfg.D_max, PROBE_D_STEPS))
    records: list[Record] = []
    for N in n_vals:
        for D in d_vals:
            for source in PROBE_SOURCES:
                for _ in range(PROBE_SAMPLES_PER_SPEC):
                    code = _sample_one(N, D, source, archive)
                    if code is not None:
                        records.append(Record(code=code.float(), N=N, D=D, source=source))
    random.shuffle(records)
    return records


def split_records(records: list[Record]):
    n     = len(records)
    n_tr  = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return records[:n_tr], records[n_tr:n_tr + n_val], records[n_tr + n_val:]


def _class_label_perturbed(sigma: float) -> str:
    return f"perturbed_sigma_{_float_token(sigma)}"


def _sample_class_code(N: int, D: int, category: str,
                       archive: ArchiveCache | None) -> torch.Tensor | None:
    if category == "archive_raw":
        if archive is None:
            return None
        code = archive.sample_with_nd(N, D)
        return None if code is None else code.clone().cpu()

    if category == "optimized":
        base = archive.sample_with_nd(N, D) if archive is not None else None
        if base is None:
            base = sample_spherical_code(N, D)
        return quick_optimize(
            base, steps=CLASS_PCA_HEAVY_OPT_STEPS, lr=0.05, potential="coulomb"
        ).cpu()

    if category == "slightly_optimized":
        base = archive.sample_with_nd(N, D) if archive is not None else None
        if base is None:
            base = sample_spherical_code(N, D)
        return quick_optimize(
            base, steps=CLASS_PCA_LIGHT_OPT_STEPS, lr=0.05, potential="coulomb"
        ).cpu()

    if category.startswith("perturbed_sigma_"):
        sigma_token = category.split("perturbed_sigma_", 1)[1].replace("p", ".").replace("m", "-")
        sigma = float(sigma_token)
        base = archive.sample_with_nd(N, D) if archive is not None else None
        if base is None:
            base = sample_spherical_code(N, D)
            base = quick_optimize(
                base, steps=CLASS_PCA_LIGHT_OPT_STEPS, lr=0.05, potential="coulomb"
            )
        return perturb_tangent(base.clone(), sigma=sigma).cpu()

    if category == "random":
        return sample_spherical_code(N, D).cpu()

    raise ValueError(f"Unknown class category: {category!r}")


def build_class_pca_records(cfg, archive: ArchiveCache | None) -> list[Record]:
    n_vals = _unique_ints(np.geomspace(cfg.N_min, cfg.N_max, PROBE_N_STEPS))
    d_vals = _unique_ints(np.linspace(cfg.D_min, cfg.D_max, PROBE_D_STEPS))
    categories = [
        "archive_raw",
        "optimized",
        "slightly_optimized",
        *[_class_label_perturbed(s) for s in CLASS_PCA_PERTURB_SIGMAS],
        "random",
    ]

    records: list[Record] = []
    for N in n_vals:
        for D in d_vals:
            for category in categories:
                for _ in range(CLASS_PCA_SAMPLES_PER_SPEC):
                    code = _sample_class_code(N, D, category, archive)
                    if code is None:
                        continue
                    records.append(Record(code=code.float(), N=N, D=D, source=category))
    random.shuffle(records)
    return records

# =============================================================================
# Analytic targets
# =============================================================================

def _target_names(eig_count: int) -> list[str]:
    base = [
        "N", "D",
        "min_cos", "max_cos", "mean_cos", "std_cos",
        "min_dist", "coulomb_energy", "log_energy",
        "entropy_rank", "eig_flatness",
    ]
    riesz = [f"riesz_{_float_token(s)}" for s in PROBE_RIESZ_ORDERS]
    harmonics = [f"harmonic_power_l{l}" for l in PROBE_HARMONIC_ORDERS]
    knn = [f"knn_k{k}_{stat}" for k in PROBE_KNN_ORDERS for stat in ("mean", "var")]
    kissing = [f"kissing_count_eps_{_float_token(PROBE_KISSING_EPS)}"]
    eigs = [f"eig_{i+1}" for i in range(eig_count)]
    return base + riesz + harmonics + knn + kissing + eigs


def _float_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _legendre_polynomial(order: int, x: torch.Tensor) -> torch.Tensor:
    """Legendre P_l(x) on tensor x; used as a spherical-harmonic power proxy."""
    if order == 0:
        return torch.ones_like(x)
    if order == 1:
        return x
    p_nm2 = torch.ones_like(x)
    p_nm1 = x
    for n in range(2, order + 1):
        p_n = ((2 * n - 1) * x * p_nm1 - (n - 1) * p_nm2) / n
        p_nm2, p_nm1 = p_nm1, p_n
    return p_nm1


def build_target_categories(names: list[str]) -> dict[str, list[int]]:
    categories = {
        "core_geometry": [],
        "riesz_energies": [],
        "spherical_harmonics": [],
        "knn": [],
        "kissing": [],
        "eigenvalues": [],
    }
    for i, name in enumerate(names):
        if name.startswith("eig_"):
            categories["eigenvalues"].append(i)
        elif name.startswith("riesz_") or name in {"coulomb_energy", "log_energy"}:
            categories["riesz_energies"].append(i)
        elif name.startswith("harmonic_power_l"):
            categories["spherical_harmonics"].append(i)
        elif name.startswith("knn_k"):
            categories["knn"].append(i)
        elif name.startswith("kissing_count_eps_"):
            categories["kissing"].append(i)
        else:
            categories["core_geometry"].append(i)
    return {k: v for k, v in categories.items() if v}


def _compute_analytics(code: torch.Tensor, eig_count: int) -> dict[str, float]:
    N, D = code.shape
    G = code @ code.T
    eye = torch.eye(N, dtype=torch.bool, device=code.device)
    off = G[~eye]
    dsq_mat = dsq_from_gram(G)                          # ||x_i - x_j||²
    dist_mat = dsq_mat.clamp_min(1e-12).sqrt()
    dist = dist_mat[~eye]

    eigvals      = torch.linalg.eigvalsh(G).clamp_min(0).sort(descending=True).values
    eigvals_norm = eigvals / eigvals.sum().clamp_min(1e-12)
    active       = eigvals_norm[eigvals_norm > 1e-12]
    entropy_rank = torch.exp(-(active * active.log()).sum()).item()
    top_d        = eigvals[:max(1, min(D, eigvals.numel()))]
    eig_flatness = (top_d.min() / top_d.max().clamp_min(1e-12)).item()
    out = {
        "N": float(N), "D": float(D),
        "min_cos": off.min().item(), "max_cos": off.max().item(),
        "mean_cos": off.mean().item(), "std_cos": off.std(unbiased=False).item(),
        "min_dist": dist.min().item(),
        "coulomb_energy": (1.0 / dist).mean().item(),
        "log_energy": (-dist.log()).mean().item(),
        "entropy_rank": entropy_rank,
        "eig_flatness": eig_flatness,
    }

    for s in PROBE_RIESZ_ORDERS:
        out[f"riesz_{_float_token(s)}"] = (dist.pow(-s)).mean().item()

    clipped_off = off.clamp(-1.0, 1.0)
    for l in PROBE_HARMONIC_ORDERS:
        p_l = _legendre_polynomial(l, clipped_off)
        out[f"harmonic_power_l{l}"] = (p_l * p_l).mean().item()

    dist_for_knn = dist_mat.clone()
    dist_for_knn[eye] = float("inf")
    sorted_dist, _ = torch.sort(dist_for_knn, dim=1)
    max_idx = max(0, sorted_dist.shape[1] - 1)
    for k in PROBE_KNN_ORDERS:
        idx = min(max(0, k - 1), max_idx)
        kth = sorted_dist[:, idx]
        out[f"knn_k{k}_mean"] = kth.mean().item()
        out[f"knn_k{k}_var"] = kth.var(unbiased=False).item()

    triu = torch.triu_indices(N, N, offset=1, device=code.device)
    pair_dists = dist_mat[triu[0], triu[1]]
    min_dist = pair_dists.min()
    out[f"kissing_count_eps_{_float_token(PROBE_KISSING_EPS)}"] = float(
        (pair_dists <= min_dist + PROBE_KISSING_EPS).sum().item()
    )

    for i in range(eig_count):
        out[f"eig_{i+1}"] = eigvals_norm[i].item() if i < eigvals_norm.numel() else 0.0
    return out

# =============================================================================
# Encoding and feature matrices
# =============================================================================

def encode_and_target(records: list[Record], names: list[str],
                      encoder, cfg, device: str):
    """Returns latents (n, latent_dim) and targets (n, T)."""
    order = sorted(range(len(records)),
                   key=lambda i: (records[i].N, records[i].D))
    latent_blocks: list[tuple[list[int], np.ndarray]] = []
    for start in range(0, len(order), ENCODE_BATCH_SIZE):
        idx   = order[start:start + ENCODE_BATCH_SIZE]
        codes = [records[i].code for i in idx]
        x, mask, Ds, _ = pad_batch(codes, cfg.D_max, cfg.N_max)
        with torch.no_grad():
            z = encoder(x.to(device), mask.to(device), Ds.to(device))
        latent_blocks.append((idx, z.cpu().numpy()))

    latent_dim = latent_blocks[0][1].shape[1]
    latents = np.zeros((len(records), latent_dim), dtype=np.float32)
    for idx, z in latent_blocks:
        for local, global_ in enumerate(idx):
            latents[global_] = z[local]

    targets_rows: list[list[float]] = []
    for r in records:
        stats = _compute_analytics(r.code, PROBE_EIGEN_COUNT)
        targets_rows.append([stats[n] for n in names])
    targets = np.array(targets_rows, dtype=np.float32)
    return latents, targets


def encode_records(records: list[Record], encoder, cfg, device: str) -> np.ndarray:
    """Encode records only. Returns latent matrix (n, latent_dim)."""
    order = sorted(range(len(records)), key=lambda i: (records[i].N, records[i].D))
    latent_blocks: list[tuple[list[int], np.ndarray]] = []
    for start in range(0, len(order), ENCODE_BATCH_SIZE):
        idx = order[start:start + ENCODE_BATCH_SIZE]
        codes = [records[i].code for i in idx]
        x, mask, Ds, _ = pad_batch(codes, cfg.D_max, cfg.N_max)
        with torch.no_grad():
            z = encoder(x.to(device), mask.to(device), Ds.to(device))
        latent_blocks.append((idx, z.cpu().numpy()))

    latent_dim = latent_blocks[0][1].shape[1]
    latents = np.zeros((len(records), latent_dim), dtype=np.float32)
    for idx, z in latent_blocks:
        for local, global_ in enumerate(idx):
            latents[global_] = z[local]
    return latents


def nd_features(records: list[Record]) -> np.ndarray:
    """Engineered (N, D) features: N, D, N², D², N·D, logN, logD, N/D, √N, √D."""
    rows = []
    for r in records:
        N, D = float(r.N), float(r.D)
        rows.append([N, D, N*N, D*D, N*D,
                     math.log(N), math.log(D),
                     N / D, math.sqrt(N), math.sqrt(D)])
    return np.array(rows, dtype=np.float32)

# =============================================================================
# Linear probe (ridge regression)
# =============================================================================

def _standardize(X_train: np.ndarray, *others: np.ndarray):
    """Standardize by train statistics. Returns (X_tr_s, ..., mean, std)."""
    mean = X_train.mean(0, keepdims=True)
    std  = X_train.std(0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    scaled = tuple((a - mean) / std for a in (X_train, *others))
    return *scaled, mean, std


def _fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    Xa = torch.cat([Xt, torch.ones(len(Xt), 1, dtype=torch.float64)], 1)
    reg = alpha * torch.eye(Xa.shape[1], dtype=torch.float64)
    reg[-1, -1] = 0.0
    W = torch.linalg.solve(Xa.T @ Xa + reg, Xa.T @ Yt)
    return W.numpy()


def _ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xt = torch.as_tensor(X, dtype=torch.float64)
    Xa = torch.cat([Xt, torch.ones(len(Xt), 1, dtype=torch.float64)], 1)
    return (Xa @ torch.as_tensor(W, dtype=torch.float64)).numpy().astype(np.float32)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """R² per column. Shape (T,)."""
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0, keepdims=True)) ** 2).sum(0)
    return (1.0 - ss_res / np.maximum(ss_tot, 1e-12)).astype(np.float32)


def run_linear_probe(X_tr, Y_tr, X_te, Y_te, alpha=RIDGE_ALPHA) -> np.ndarray:
    """Ridge fit on train, R² on test. Returns (T,)."""
    X_tr_s, X_te_s, *_ = _standardize(X_tr, X_te)
    Y_tr_s, Y_te_s, ym, ys = _standardize(Y_tr, Y_te)
    W    = _fit_ridge(X_tr_s, Y_tr_s, alpha)
    pred = _ridge_predict(X_te_s, W) * ys + ym
    return _r2(Y_te, pred)


def per_dim_r2(X_tr, Y_tr, X_te, Y_te, alpha=RIDGE_ALPHA) -> np.ndarray:
    """Univariate R²[d, t] = R² from single latent dim d alone → target t."""
    X_tr_s, X_te_s, *_ = _standardize(X_tr, X_te)
    Y_tr_s, Y_te_s, ym, ys = _standardize(Y_tr, Y_te)
    D = X_tr_s.shape[1]
    T = Y_tr_s.shape[1]
    out = np.zeros((D, T), dtype=np.float32)
    for d in range(D):
        W    = _fit_ridge(X_tr_s[:, d:d+1], Y_tr_s, alpha)
        pred = _ridge_predict(X_te_s[:, d:d+1], W) * ys + ym
        out[d] = _r2(Y_te, pred)
    return out

# =============================================================================
# MLP probe
# =============================================================================

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple, out_dim: int):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_mlp_probe(X_tr, Y_tr, X_val, Y_val, X_te, Y_te,
                  hidden=MLP_HIDDEN) -> np.ndarray:
    """MLP with early stopping on val. Returns (T,) test R²."""
    X_tr_s, X_val_s, X_te_s, *_ = _standardize(X_tr, X_val, X_te)
    Y_tr_s, Y_val_s, Y_te_s, ym, ys = _standardize(Y_tr, Y_val, Y_te)

    Xtr = torch.tensor(X_tr_s,  dtype=torch.float32)
    Ytr = torch.tensor(Y_tr_s,  dtype=torch.float32)
    Xva = torch.tensor(X_val_s, dtype=torch.float32)
    Yva = torch.tensor(Y_val_s, dtype=torch.float32)
    Xte = torch.tensor(X_te_s,  dtype=torch.float32)

    model = _MLP(Xtr.shape[1], hidden, Ytr.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=MLP_LR)

    best_val   = float("inf")
    best_state = None
    patience   = MLP_PATIENCE

    for _ in range(MLP_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for start in range(0, len(perm), MLP_BATCH):
            b    = perm[start:start + MLP_BATCH]
            loss = nn.functional.mse_loss(model(Xtr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(Xva), Yva).item()

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = MLP_PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy() * ys + ym
    return _r2(Y_te, pred)

# =============================================================================
# Trajectory analysis
# =============================================================================

def _collect_trajectory(code: torch.Tensor, steps: int, lr: float) -> list[torch.Tensor]:
    """Collect per-step states from Coulomb optimisation via quick_optimize(steps=1)."""
    states = [code.detach().cpu()]
    x = code.clone()
    for _ in range(steps):
        x = quick_optimize(x, steps=1, lr=lr, potential="coulomb")
        states.append(x.detach().cpu())
    return states


def _encode_list(codes: list[torch.Tensor], encoder, cfg, device: str) -> np.ndarray:
    blocks = []
    for start in range(0, len(codes), ENCODE_BATCH_SIZE):
        batch = codes[start:start + ENCODE_BATCH_SIZE]
        x, mask, Ds, _ = pad_batch(batch, cfg.D_max, cfg.N_max)
        with torch.no_grad():
            z = encoder(x.to(device), mask.to(device), Ds.to(device))
        blocks.append(z.cpu().numpy())
    return np.concatenate(blocks, 0)


def analyze_trajectory(encoder, cfg, device: str, N: int, D: int) -> dict:
    base    = sample_spherical_code(N, D)
    states  = _collect_trajectory(base, TRAJECTORY_STEPS, TRAJECTORY_LR)
    latents = _encode_list(states, encoder, cfg, device)
    energies = np.array([
        _compute_analytics(s, 0)["coulomb_energy"] for s in states
    ], dtype=np.float32)
    latent_steps = np.linalg.norm(np.diff(latents, axis=0), axis=1).astype(np.float32)
    mean, comps, explained = _fit_pca2(latents)
    pca = (latents - mean) @ comps
    return {"N": N, "D": D,
            "latents": latents, "energies": energies, "latent_steps": latent_steps,
            "pca": pca, "pca_explained": explained.tolist()}


def _fit_pca2(X: np.ndarray):
    mean = X.mean(0)
    Xc   = X - mean
    _, s, vh = np.linalg.svd(Xc, full_matrices=False)
    explained = s[:2] ** 2 / max(float((s ** 2).sum()), 1e-12)
    return mean, vh[:2].T, explained

# =============================================================================
# (Plotting functions removed — this harness is plot-free)
# =============================================================================


def run_class_pca_diagnostic(cfg, encoder, archive: ArchiveCache | None,
                             device: str, output_dir: Path) -> dict | None:
    records = build_class_pca_records(cfg, archive)
    if not records:
        return None

    latents = encode_records(records, encoder, cfg, device)
    mean, comps, explained = _fit_pca2(latents)
    labels = [r.source for r in records]

    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    payload = {
        "explained_variance_ratio": explained.tolist(),
        "counts_by_category": counts,
    }
    (output_dir / "class_pca_results.json").write_text(json.dumps(payload, indent=2))
    return payload

# =============================================================================
# Summary
# =============================================================================

def write_summary(ckpt_path: Path, output_dir: Path, names: list[str],
                  r2_lin: np.ndarray, r2_mlp: np.ndarray, r2_nd: np.ndarray,
                  dim_r2: np.ndarray, trajectories: list[dict],
                  categories: dict[str, list[int]],
                  class_pca_payload: dict | None = None) -> None:
    order = np.argsort(r2_mlp)[::-1]
    lines = [
        f"checkpoint : {ckpt_path}",
        f"output_dir : {output_dir}",
        "",
        "Probe results (test R²)  —  sorted by latent MLP R²",
        f"{'target':20s}  {'linear':>8}  {'MLP':>8}  {'N,D base':>10}  {'Δ MLP-base':>11}",
        "─" * 66,
    ]
    for i in order:
        d = r2_mlp[i] - r2_nd[i]
        lines.append(f"  {names[i]:18s}  {r2_lin[i]:8.3f}  {r2_mlp[i]:8.3f}"
                     f"  {r2_nd[i]:10.3f}  {d:+11.3f}")

    lines += ["", "Top latent dimension per target  (highest univariate R²):"]
    for i in order:
        best_d  = int(np.argmax(dim_r2[:, i]))
        best_r2 = float(dim_r2[best_d, i])
        lines.append(f"  {names[i]:20s}  z{best_d}  (R² = {best_r2:.3f})")

    lines += ["", "Category leaders (latent MLP R²):"]
    for category, idxs in categories.items():
        cat_order = sorted(idxs, key=lambda i: float(r2_mlp[i]), reverse=True)
        lines.append(f"  [{category}]")
        for i in cat_order[: min(5, len(cat_order))]:
            lines.append(f"    {names[i]:20s}  MLP R²={r2_mlp[i]:.3f}  Δ={r2_mlp[i] - r2_nd[i]:+.3f}")

    lines += ["", "Trajectory summary:"]
    for traj in trajectories:
        pc1    = traj.get("pca", np.zeros((2, 2)))[:, 0]
        energy = traj["energies"]
        r = float(np.corrcoef(pc1, energy)[0, 1]) if pc1.std() > 1e-8 else float("nan")
        lines.append(
            f"  N={traj['N']:4d}  D={traj['D']:3d}  "
            f"path_length={traj['latent_steps'].sum():.3f}  "
            f"energy_drop={energy[0] - energy[-1]:.3f}  "
            f"PC1_r_energy={r:.3f}"
        )

    if class_pca_payload is not None:
        evr = class_pca_payload.get("explained_variance_ratio", [0.0, 0.0])
        lines += [
            "",
            "Class PCA diagnostic:",
            f"  explained_variance_ratio: PC1={float(evr[0]):.4f}, PC2={float(evr[1]):.4f}",
        ]
        counts = class_pca_payload.get("counts_by_category", {})
        for k in sorted(counts):
            lines.append(f"  {k:24s} n={int(counts[k])}")

    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")

# =============================================================================
# Programmatic entry point (used by test_solution.py)
# =============================================================================

def run_probe_suite(
    ckpt_path: "str | Path",
    device: str,
    output_dir: "str | Path | None" = None,
) -> dict:
    """Run the full latent probe suite and return results as a dict.

    This is the programmatic entry point called by test_solution.py.
    Results are also written as JSON files under *output_dir*.

    Returns
    -------
    dict with keys:
        target_names, r2_linear, r2_mlp, r2_nd_baseline,
        per_dim_r2, trajectories, categories, categories_idx,
        class_pca_payload, output_dir, ckpt_path
    """
    seed_all(SEED)
    device = resolve_device(device)
    ckpt_path = Path(ckpt_path)

    if output_dir is None:
        output_dir = OUTPUT_ROOT / ckpt_path.parent.name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder, cfg = load_encoder(ckpt_path, device)
    archive = None
    if ARCHIVE_DIR.exists():
        archive = ArchiveCache(
            str(ARCHIVE_DIR),
            N_min=cfg.N_min, N_max=cfg.N_max,
            D_min=cfg.D_min, D_max=cfg.D_max,
            verbose=False,
        )

    records = build_records(cfg, archive)
    r_tr, r_val, r_te = split_records(records)

    names = _target_names(PROBE_EIGEN_COUNT)
    categories_idx = build_target_categories(names)

    Z_tr,  Y_tr  = encode_and_target(r_tr,  names, encoder, cfg, device)
    Z_val, Y_val = encode_and_target(r_val, names, encoder, cfg, device)
    Z_te,  Y_te  = encode_and_target(r_te,  names, encoder, cfg, device)
    ND_tr  = nd_features(r_tr)
    ND_te  = nd_features(r_te)

    r2_lin  = run_linear_probe(Z_tr,  Y_tr, Z_te, Y_te)
    r2_nd   = run_linear_probe(ND_tr, Y_tr, ND_te, Y_te)
    dim_r2_ = per_dim_r2(Z_tr, Y_tr, Z_te, Y_te)
    r2_mlp  = run_mlp_probe(Z_tr, Y_tr, Z_val, Y_val, Z_te, Y_te)

    n_vals = _unique_ints(np.geomspace(cfg.N_min, cfg.N_max, PROBE_N_STEPS))
    d_vals = _unique_ints(np.linspace(cfg.D_min, cfg.D_max, PROBE_D_STEPS))
    traj_specs = sorted({
        (n_vals[0],  d_vals[0]),  (n_vals[0],  d_vals[-1]),
        (n_vals[-1], d_vals[0]),  (n_vals[-1], d_vals[-1]),
    })[:4]
    trajectories = [
        analyze_trajectory(encoder, cfg, device, N, D) for N, D in traj_specs
    ]

    class_pca_payload = run_class_pca_diagnostic(
        cfg, encoder, archive, device, output_dir
    )

    write_summary(
        ckpt_path, output_dir, names,
        r2_lin, r2_mlp, r2_nd, dim_r2_, trajectories,
        categories_idx, class_pca_payload,
    )

    payload = {
        "checkpoint":      str(ckpt_path),
        "target_names":    names,
        "categories":      {k: [names[i] for i in v] for k, v in categories_idx.items()},
        "categories_idx":  {k: v for k, v in categories_idx.items()},
        "r2_linear":       r2_lin.tolist(),
        "r2_mlp":          r2_mlp.tolist(),
        "r2_nd_baseline":  r2_nd.tolist(),
        "per_dim_r2":      dim_r2_.tolist(),
    }
    (output_dir / "probe_results.json").write_text(json.dumps(payload, indent=2))

    return {
        "target_names":    names,
        "r2_linear":       r2_lin.tolist(),
        "r2_mlp":          r2_mlp.tolist(),
        "r2_nd_baseline":  r2_nd.tolist(),
        "per_dim_r2":      dim_r2_.tolist(),
        "trajectories":    [
            {k: (v.tolist() if hasattr(v, "tolist") else v)
             for k, v in t.items()}
            for t in trajectories
        ],
        "categories":      {k: [names[i] for i in v] for k, v in categories_idx.items()},
        "categories_idx":  {k: v for k, v in categories_idx.items()},
        "class_pca_payload": class_pca_payload,
        "output_dir":      str(output_dir),
        "ckpt_path":       str(ckpt_path),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Latent-space test bed")
    parser.add_argument("--ckpt",       default=CKPT_PATH)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    seed_all(SEED)
    device    = resolve_device(DEVICE)
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_ckpt(RUNS_DIR)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else OUTPUT_ROOT / ckpt_path.parent.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"checkpoint : {ckpt_path}")
    print(f"device     : {device}")

    encoder, cfg = load_encoder(ckpt_path, device)
    archive = None
    if ARCHIVE_DIR.exists():
        archive = ArchiveCache(str(ARCHIVE_DIR),
                               N_min=cfg.N_min, N_max=cfg.N_max,
                               D_min=cfg.D_min, D_max=cfg.D_max,
                               verbose=False)

    # ── Build probe dataset ──────────────────────────────────────────────────
    print("Building probe dataset ...")
    records = build_records(cfg, archive)
    r_tr, r_val, r_te = split_records(records)
    print(f"  train={len(r_tr)}  val={len(r_val)}  test={len(r_te)}")

    names = _target_names(PROBE_EIGEN_COUNT)
    categories = build_target_categories(names)

    print("Encoding ...")
    Z_tr,  Y_tr  = encode_and_target(r_tr,  names, encoder, cfg, device)
    Z_val, Y_val = encode_and_target(r_val, names, encoder, cfg, device)
    Z_te,  Y_te  = encode_and_target(r_te,  names, encoder, cfg, device)

    ND_tr  = nd_features(r_tr)
    ND_te  = nd_features(r_te)

    # ── Probes ───────────────────────────────────────────────────────────────
    print("Linear probes ...")
    r2_lin  = run_linear_probe(Z_tr,  Y_tr, Z_te, Y_te)
    r2_nd   = run_linear_probe(ND_tr, Y_tr, ND_te, Y_te)
    dim_r2_ = per_dim_r2(Z_tr, Y_tr, Z_te, Y_te)

    print("MLP probe ...")
    r2_mlp = run_mlp_probe(Z_tr, Y_tr, Z_val, Y_val, Z_te, Y_te)

    # ── Trajectories ─────────────────────────────────────────────────────────
    n_vals = _unique_ints(np.geomspace(cfg.N_min, cfg.N_max, PROBE_N_STEPS))
    d_vals = _unique_ints(np.linspace(cfg.D_min, cfg.D_max, PROBE_D_STEPS))
    traj_specs = sorted({(n_vals[0],  d_vals[0]),  (n_vals[0],  d_vals[-1]),
                         (n_vals[-1], d_vals[0]),  (n_vals[-1], d_vals[-1])})[:4]
    trajectories = []
    for N, D in traj_specs:
        print(f"Trajectory N={N}  D={D} ...")
        trajectories.append(analyze_trajectory(encoder, cfg, device, N, D))

    print("Class-mix PCA diagnostic ...")
    class_pca_payload = run_class_pca_diagnostic(cfg, encoder, archive, device, output_dir)

    # ── Save outputs ─────────────────────────────────────────────────────────
    write_summary(ckpt_path, output_dir, names,
                  r2_lin, r2_mlp, r2_nd, dim_r2_, trajectories, categories,
                  class_pca_payload)

    payload = {
        "checkpoint":     str(ckpt_path),
        "target_names":   names,
        "categories": {
            category: [names[i] for i in idxs]
            for category, idxs in categories.items()
        },
        "r2_linear":      r2_lin.tolist(),
        "r2_mlp":         r2_mlp.tolist(),
        "r2_nd_baseline": r2_nd.tolist(),
        "per_dim_r2":     dim_r2_.tolist(),
    }
    (output_dir / "probe_results.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
