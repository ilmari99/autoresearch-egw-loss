"""
evaluation.py
=============

Validation, invariance tests, and latent-space diagnostics for the spherical
code autoencoder.

All public functions accept duck-typed ``cfg`` objects (any namespace with the
required attributes) and loss callables, so they stay decoupled from the
training script's Config / loss implementations.

Public API
----------
  - random_orthogonal(D)
  - run_invariance_tests(encoder, cfg, device, ...)
  - build_val_codes(cfg)
  - evaluate(enc, dec, val_codes, cfg, device, loss_fns)
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from data import sample_spherical_code, pad_batch, SphereCodeMixedDataset, load_archives
import os


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def random_orthogonal(D: int, device="cpu") -> torch.Tensor:
    """Sample a uniform random element of O(D)."""
    A = torch.randn(D, D, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


# ---------------------------------------------------------------------------
# Invariance tests
# ---------------------------------------------------------------------------

def run_invariance_tests(encoder, cfg, device, n_trials: int = 5,
                         verbose: bool = True) -> dict:
    """Empirically check permutation- and rotation-invariance of *encoder*.

    ``cfg`` must expose: D_min, D_max, N_min, N_max.

    Returns dict with keys ``perm_max``, ``rot_max``, ``lip_ratio``.
    """
    was_training = encoder.training
    encoder.eval()
    results = {"perm_max": 0.0, "rot_max": 0.0, "lip_ratio": 0.0}
    with torch.no_grad():
        for _ in range(n_trials):
            D = torch.randint(cfg.D_min, cfg.D_max + 1, (1,)).item()
            N = torch.randint(cfg.N_min, min(80, cfg.N_max) + 1, (1,)).item()
            c = sample_spherical_code(N, D)
            x, mask, Ds, Ns = pad_batch([c], cfg.D_max, cfg.N_max)
            x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
            z = encoder(x, mask, Ds)

            # Permutation
            perm = torch.randperm(N, device=device)
            x_perm = x.clone(); x_perm[0, :N] = x[0, perm]
            results["perm_max"] = max(results["perm_max"],
                                      (z - encoder(x_perm, mask, Ds)).abs().max().item())

            # Rotation on the active D subspace
            R = random_orthogonal(D, device=device)
            x_rot = x.clone(); x_rot[0, :N, :D] = x[0, :N, :D] @ R
            results["rot_max"] = max(results["rot_max"],
                                     (z - encoder(x_rot, mask, Ds)).abs().max().item())

            # Local Lipschitz estimate
            eps = 1e-3
            delta = torch.randn_like(x) * eps
            delta[:, :, D:] = 0; delta[:, N:, :] = 0
            x_pert = x + delta
            x_pert[0, :N, :D] /= x_pert[0, :N, :D].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            dz = (z - encoder(x_pert, mask, Ds)).norm().item()
            dx = (x - x_pert).norm().item()
            if dx > 0:
                results["lip_ratio"] = max(results["lip_ratio"], dz / dx)
    if verbose:
        print(f"[invariance] perm_max={results['perm_max']:.2e}  "
              f"rot_max={results['rot_max']:.2e}  "
              f"lip_ratio<{results['lip_ratio']:.2f}")
    if was_training:
        encoder.train()
    return results


# ---------------------------------------------------------------------------
# Validation set construction
# ---------------------------------------------------------------------------

def build_val_codes(cfg) -> list:
    """Generate a fixed set of validation codes from the mixed data pipeline.

    Uses the same source mix as training (archive codes, perturbed, FPS, etc.)
    but with a deterministic seed offset so the val set is disjoint.

    ``cfg`` must expose: seed, val_size, D_min, D_max, N_min, N_max.
    Archive directories are read from cfg.archive_dirs (with fallback).
    """
    archive_dirs = getattr(cfg, "archive_dirs", ["combined_points_archive"])
    archive = load_archives(archive_dirs, cfg) if archive_dirs else None

    ds = SphereCodeMixedDataset(
        length=cfg.val_size,
        D_min=cfg.D_min, D_max=cfg.D_max,
        N_min=cfg.N_min, N_max=cfg.N_max,
        archive=archive,
        seed=cfg.seed + 10_000_000,  # offset to avoid overlap with training
    )
    return [ds[i] for i in range(cfg.val_size)]


# ---------------------------------------------------------------------------
# Validation diagnostics helpers
# ---------------------------------------------------------------------------

def _module_grad_norm(module: torch.nn.Module) -> float:
    grads = [p.grad.data for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.stack([g.norm(2) ** 2 for g in grads]).sum()
    return total.sqrt().item()


def _param_grad_norm(param_or_module) -> float:
    if isinstance(param_or_module, torch.nn.Parameter):
        return param_or_module.grad.norm().item() if param_or_module.grad is not None else 0.0
    return _module_grad_norm(param_or_module)


def _component_grad_norms(model, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if hasattr(model, "diagnostic_submodules"):
        for name, sub in model.diagnostic_submodules().items():
            out[f"val_grad_{prefix}_{name}"] = _param_grad_norm(sub)
    return out


def _cumulative_rank(var_ratio: torch.Tensor, threshold: float) -> int:
    if var_ratio.numel() == 0:
        return 0
    cdf = torch.cumsum(var_ratio, dim=0)
    idx = torch.nonzero(cdf >= threshold, as_tuple=False)
    if idx.numel() == 0:
        return int(var_ratio.numel())
    return int(idx[0].item() + 1)


def _summarise_group_losses(per_sample_losses: list[float], values: list[int], key: str) -> list[dict]:
    buckets: dict[int, list[float]] = {}
    for value, loss in zip(values, per_sample_losses):
        buckets.setdefault(int(value), []).append(float(loss))
    rows = [
        {
            key: int(value),
            "mean": float(np.mean(losses)),
            "std": float(np.std(losses)),
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
            "count": int(len(losses)),
        }
        for value, losses in buckets.items()
    ]
    rows.sort(key=lambda row: (row["mean"], row["max"]), reverse=True)
    return rows


def _latent_stats(Z_all: torch.Tensor) -> dict:
    Z_all = Z_all.float()
    Z_c = Z_all - Z_all.mean(dim=0, keepdim=True)
    per_dim_std = Z_all.std(dim=0, unbiased=False)
    sv = torch.linalg.svdvals(Z_c)
    eig = sv.square()
    eig_sum = eig.sum()
    var_ratio = eig / eig_sum.clamp_min(1e-12)
    nz = var_ratio[var_ratio > 1e-12]
    entropy_rank = torch.exp(-(nz * torch.log(nz)).sum()).item() if nz.numel() else 0.0
    participation = (
        eig_sum.square() / eig.square().sum().clamp_min(1e-12)
    ).item() if eig.numel() else 0.0

    pair_dists = torch.pdist(Z_all)
    topk = min(8, int(var_ratio.numel()))
    pca_spectrum = [float(v) for v in var_ratio[:topk].tolist()]
    return {
        "val_latnorm": Z_all.norm(dim=-1).mean().item(),
        "val_latstd": per_dim_std.mean().item(),
        "val_latstd_min": per_dim_std.min().item(),
        "val_latstd_max": per_dim_std.max().item(),
        "val_latpdist": pair_dists.mean().item() if pair_dists.numel() else 0.0,
        "val_latdead": (per_dim_std < 1e-4).float().mean().item(),
        "val_entropy_rank": entropy_rank,
        "val_rank": int(torch.linalg.matrix_rank(Z_c).item()),
        "val_pca_rank_95": _cumulative_rank(var_ratio, 0.95),
        "val_pca_rank_99": _cumulative_rank(var_ratio, 0.99),
        "val_pca_participation": participation,
        "val_pca_spectrum": pca_spectrum,
    }


def _validation_gradient_stats(
    enc_model,
    dec_model,
    val_codes: list,
    cfg,
    device,
    *,
    loss_fn: Callable,
) -> dict[str, float]:
    enc_model.zero_grad(set_to_none=True)
    dec_model.zero_grad(set_to_none=True)

    total_items = max(len(val_codes), 1)
    per_sample_loss = getattr(loss_fn, "per_sample", None)

    for i in range(0, len(val_codes), cfg.batch_size):
        batch = val_codes[i:i + cfg.batch_size]
        x, mask, Ds, Ns = pad_batch(batch, cfg.D_max, cfg.N_max)
        x, mask, Ds, Ns = [t.to(device) for t in (x, mask, Ds, Ns)]
        z = enc_model(x, mask, Ds)
        pred, pred_mask = dec_model(z, Ns, Ds)
        if per_sample_loss is not None:
            losses = per_sample_loss(pred, x, pred_mask, mask)
            loss = losses.sum() / total_items
        else:
            loss = loss_fn(pred, x, pred_mask, mask) * (x.shape[0] / total_items)
        loss.backward()

    grad_stats = {
        **_component_grad_norms(enc_model, "enc"),
        **_component_grad_norms(dec_model, "dec"),
    }
    grad_stats["val_grad_total"] = math.sqrt(
        sum(float(value) ** 2 for value in grad_stats.values())
    )

    enc_model.zero_grad(set_to_none=True)
    dec_model.zero_grad(set_to_none=True)
    return grad_stats


# ---------------------------------------------------------------------------
# Full evaluation pass
# ---------------------------------------------------------------------------

def evaluate(
    enc_model,
    dec_model,
    val_codes: list,
    cfg,
    device,
    *,
    loss_fn: Callable,
) -> dict:
    """Run a full validation pass and return a dict of metrics.

    ``cfg`` must expose: batch_size, D_max, N_max.

    Loss callables are injected to keep this module decoupled from the loss
    implementations::

        loss_fn(pred, target, pred_mask, target_mask) -> scalar
    """
    enc_model.eval(); dec_model.eval()
    agg = dict(egw=0.0, n=0)
    inv_perm = inv_rot = lip_pert = 0.0
    n_inv = 0
    all_z = []
    all_losses: list[float] = []
    all_Ds: list[int] = []
    all_Ns: list[int] = []
    per_sample_loss = getattr(loss_fn, "per_sample", None)

    with torch.no_grad():
        for i in range(0, len(val_codes), cfg.batch_size):
            batch = val_codes[i:i + cfg.batch_size]
            x, mask, Ds, Ns = pad_batch(batch, cfg.D_max, cfg.N_max)
            x, mask, Ds, Ns = [t.to(device) for t in (x, mask, Ds, Ns)]
            z = enc_model(x, mask, Ds)
            all_z.append(z.detach().cpu())
            pred, pred_mask = dec_model(z, Ns, Ds)
            batch_loss_mean = loss_fn(pred, x, pred_mask, mask).item()

            if per_sample_loss is not None:
                batch_losses = per_sample_loss(pred, x, pred_mask, mask)
                all_losses.extend(float(v) for v in batch_losses.detach().cpu().tolist())
            else:
                all_losses.extend([batch_loss_mean] * x.shape[0])

            all_Ds.extend(int(v) for v in Ds.detach().cpu().tolist())
            all_Ns.extend(int(v) for v in Ns.detach().cpu().tolist())
            agg["egw"] += batch_loss_mean * x.shape[0]
            agg["n"] += x.shape[0]

            # Invariance checks on first batch element
            x0, m0, D0, N0 = x[:1], mask[:1], Ds[:1], Ns[:1]
            N_i, D_i = int(N0.item()), int(D0.item())
            perm = torch.randperm(N_i, device=device)
            x0p = x0.clone(); x0p[0, :N_i] = x0[0, perm]
            R = random_orthogonal(D_i, device=device)
            x0r = x0.clone(); x0r[0, :N_i, :D_i] = x0[0, :N_i, :D_i] @ R
            eps = 1e-3
            delta = torch.randn_like(x0) * eps
            delta[:, :, D_i:] = 0; delta[:, N_i:, :] = 0
            x0e = x0 + delta
            x0e[0, :N_i, :D_i] /= x0e[0, :N_i, :D_i].norm(dim=-1, keepdim=True).clamp_min(1e-6)
            z0 = enc_model(x0, m0, D0)
            inv_perm += (z0 - enc_model(x0p, m0, D0)).abs().max().item()
            inv_rot += (z0 - enc_model(x0r, m0, D0)).abs().max().item()
            dx = (x0 - x0e).norm().item()
            if dx > 0:
                lip_pert += (z0 - enc_model(x0e, m0, D0)).norm().item() / dx
            n_inv += 1

    Z_all = torch.cat(all_z, dim=0)
    latent_stats = _latent_stats(Z_all)
    loss_np = np.array(all_losses, dtype=np.float64)
    loss_by_D = _summarise_group_losses(all_losses, all_Ds, "D")
    loss_by_N = _summarise_group_losses(all_losses, all_Ns, "N")
    sample_rows = [
        {"loss": float(loss), "N": int(N), "D": int(D)}
        for loss, N, D in zip(all_losses, all_Ns, all_Ds)
    ]
    sample_rows.sort(key=lambda row: row["loss"], reverse=True)

    grad_stats = _validation_gradient_stats(
        enc_model,
        dec_model,
        val_codes,
        cfg,
        device,
        loss_fn=loss_fn,
    )

    out = {
        "val_egw": agg["egw"] / agg["n"],
        "val_loss_std": float(loss_np.std()),
        "val_loss_p95": float(np.percentile(loss_np, 95)),
        "inv_perm": inv_perm / n_inv,
        "inv_rot": inv_rot / n_inv,
        "lip_pert": lip_pert / n_inv,
        "val_worst_cases": sample_rows[:5],
        "val_loss_by_D": loss_by_D[:8],
        "val_loss_by_N": loss_by_N[:8],
        **latent_stats,
        **grad_stats,
    }

    # Add a validation-friendly fixed-(N,D) discrimination probe.  This is
    # cheaper than the full hard-test suite but still catches the concrete
    # failure mode where the latent geometry collapses inside a single regime
    # even while global mixed-regime rank looks acceptable.
    disc = discrimination_test(
        enc_model,
        cfg,
        device,
        configs=((32, 20, 3), (32, 40, 5), (16, 100, 10), (8, 200, 20)),
        verbose=False,
    )
    disc_cfgs = disc["disc_per_cfg"]
    if disc_cfgs:
        ratios = [v["ratio"] for v in disc_cfgs.values()]
        out["val_disc_ratio_min"] = float(min(ratios))
        out["val_disc_ratio_mean"] = float(np.mean(ratios))
    else:
        out["val_disc_ratio_min"] = float("nan")
        out["val_disc_ratio_mean"] = float("nan")
    out["val_disc_pass"] = disc["disc_pass"]

    enc_model.train(); dec_model.train()
    return out


# ---------------------------------------------------------------------------
# Hard invariance tests
# ---------------------------------------------------------------------------
#
# The tests below target genuinely difficult regimes: long compositional chains
# of perm+rot (accumulating drift), high-symmetry Platonic codes (many tied
# pairwise inner products stress numerical paths), geometric edge cases
# (rank-deficient Gram, near-degenerate clusters, antipodal pairs), and
# spectral-twin discrimination (pairs whose Gram eigenvalue spectra nearly
# coincide — the regime closest to a 1-WL-vs-2-WL separation).
#
# Thresholds are empirically calibrated for a correctly-built encoder at fp32
# with T=4 pair-refinement layers.  A well-trained encoder that fails any of
# these is almost certainly broken.
#

# ---------------------------------------------------------------------------
# Platonic / symmetric code constructions
# ---------------------------------------------------------------------------

def _tetrahedron() -> torch.Tensor:
    v = torch.tensor([
        [ 1,  1,  1], [ 1, -1, -1], [-1,  1, -1], [-1, -1,  1],
    ], dtype=torch.float32) / math.sqrt(3)
    return v


def _octahedron() -> torch.Tensor:
    return torch.tensor([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0, -1, 0],
        [ 0, 0, 1], [ 0, 0, -1],
    ], dtype=torch.float32)


def _cube() -> torch.Tensor:
    s = 1.0 / math.sqrt(3)
    return torch.tensor([
        [ s,  s,  s], [ s,  s, -s], [ s, -s,  s], [ s, -s, -s],
        [-s,  s,  s], [-s,  s, -s], [-s, -s,  s], [-s, -s, -s],
    ], dtype=torch.float32)


def _icosahedron() -> torch.Tensor:
    phi = (1 + math.sqrt(5)) / 2
    raw = torch.tensor([
        [ 0,  1,  phi], [ 0, -1,  phi], [ 0,  1, -phi], [ 0, -1, -phi],
        [ 1,  phi, 0], [-1,  phi, 0], [ 1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [-phi, 0,  1], [ phi, 0, -1], [-phi, 0, -1],
    ], dtype=torch.float32)
    return raw / raw.norm(dim=-1, keepdim=True)


def _simplex(D: int) -> torch.Tensor:
    """Regular D-simplex in R^D: D+1 unit vectors with pairwise inner product
    -1/D.  Constructed by projecting the standard basis of R^{D+1} onto the
    hyperplane orthogonal to (1,...,1), then orthonormalising."""
    N = D + 1
    e = torch.eye(N)
    e = e - e.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(e, full_matrices=False)
    # Keep the first D right-singular vectors; e has rank D.
    coords = U[:, :D] * S[:D]
    return coords / coords.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _cross_polytope(D: int) -> torch.Tensor:
    """2D-point cross-polytope: {±e_1, ..., ±e_D}.  Highly symmetric (its
    automorphism group is the hyperoctahedral group B_D of order 2^D D!)."""
    E = torch.eye(D)
    return torch.cat([E, -E], dim=0)


# ---------------------------------------------------------------------------
# Compositional / stress invariance
# ---------------------------------------------------------------------------

@torch.no_grad()
def stress_invariance_test(encoder, cfg, device, n_steps: int = 100,
                           N: int = 200, D: int = 10,
                           verbose: bool = True) -> dict:
    """Apply n_steps random (permutation ∘ rotation) compositions **in sequence**
    and track cumulative drift of z.  This catches numerical instabilities a
    single-shot perm/rot test would miss, e.g. unstable renormalisation or
    activation recursion that amplifies epsilon-level errors across layers.

    Threshold: max ‖Δz‖∞ over the chain < 1e-3 (very generous for fp32).
    """
    encoder.eval()
    N = min(N, cfg.N_max); D = min(D, cfg.D_max)
    code = sample_spherical_code(N, D)
    x, mask, Ds, _ = pad_batch([code], cfg.D_max, cfg.N_max)
    x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
    z0 = encoder(x, mask, Ds)
    x_cur = x.clone()
    max_drift = 0.0
    for _ in range(n_steps):
        perm = torch.randperm(N, device=device)
        x_new = x_cur.clone()
        x_new[0, :N] = x_cur[0, perm]
        R = random_orthogonal(D, device=device)
        x_new[0, :N, :D] = x_new[0, :N, :D] @ R
        x_new[0, :N, :D] = x_new[0, :N, :D] / x_new[0, :N, :D].norm(
            dim=-1, keepdim=True).clamp_min(1e-12)
        z_cur = encoder(x_new, mask, Ds)
        drift = (z0 - z_cur).abs().max().item()
        if drift > max_drift:
            max_drift = drift
        x_cur = x_new
    passed = max_drift < 1e-3
    if verbose:
        print(f"[stress_inv] N={N} D={D} chain={n_steps}: "
              f"max |Δz|∞ = {max_drift:.2e}  pass={passed}")
    return {"stress_inv_max_drift": max_drift, "stress_inv_pass": passed}


# ---------------------------------------------------------------------------
# Platonic / high-symmetry invariance
# ---------------------------------------------------------------------------

@torch.no_grad()
def platonic_invariance_test(encoder, cfg, device, n_trials: int = 20,
                             verbose: bool = True) -> dict:
    """Invariance on high-symmetry codes (Platonic solids, simplex, cross-
    polytope).  Many pairwise inner products are exactly equal, so a buggy
    encoder with e.g. order-sensitive pooling is more likely to drift here
    than on generic random codes.  Stricter threshold than the generic test.
    """
    encoder.eval()
    codes = [
        ("tetrahedron",   _tetrahedron()),
        ("octahedron",    _octahedron()),
        ("cube",          _cube()),
        ("icosahedron",   _icosahedron()),
        ("simplex-7",     _simplex(7)),
        ("simplex-15",    _simplex(15)),
        ("cross-poly-10", _cross_polytope(10)),
        ("cross-poly-16", _cross_polytope(min(16, cfg.D_max))),
    ]
    max_drift = 0.0
    per_code = {}
    for name, code in codes:
        N, D = code.shape
        if N > cfg.N_max or D > cfg.D_max:
            continue
        x, mask, Ds, _ = pad_batch([code], cfg.D_max, cfg.N_max)
        x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
        z0 = encoder(x, mask, Ds)
        local_max = 0.0
        for _ in range(n_trials):
            R = random_orthogonal(D, device=device)
            perm = torch.randperm(N, device=device)
            x_t = x.clone()
            x_t[0, :N, :D] = (x[0, :N, :D] @ R)[perm]
            x_t[0, :N, :D] = x_t[0, :N, :D] / x_t[0, :N, :D].norm(
                dim=-1, keepdim=True).clamp_min(1e-12)
            z_t = encoder(x_t, mask, Ds)
            d = (z0 - z_t).abs().max().item()
            if d > local_max:
                local_max = d
        per_code[name] = local_max
        if local_max > max_drift:
            max_drift = local_max
        if verbose:
            print(f"[platonic] {name:<15} (N={N:3d}, D={D:2d}): "
                  f"max |Δz|∞ = {local_max:.2e}")
    passed = max_drift < 1e-3
    return {"platonic_max_drift": max_drift, "platonic_per_code": per_code,
            "platonic_pass": passed}


# ---------------------------------------------------------------------------
# Geometric edge cases
# ---------------------------------------------------------------------------

@torch.no_grad()
def edge_case_invariance_test(encoder, cfg, device, n_trials: int = 10,
                              verbose: bool = True) -> dict:
    """Invariance on pathological code geometries:

    1. Low-N high-D — Gram is rank-N < D (rank-deficient in the D-subspace).
    2. High-N high-D — memory / precision stress.
    3. Near-degenerate cluster — all points within a small spherical cap.
    4. Near-repeated points — two near-duplicate slots (nearly collapsing Gram).
    5. Antipodal code — tests O(D)-invariance: for odd D the map x → -x is a
       reflection not in SO(D), but the Gram-based encoder is O(D)-invariant
       by construction.  z(X) should equal z(-X).

    All must satisfy |Δz|∞ < 1e-3 under random perm+rot (and under x → -x for
    the antipodal check).
    """
    encoder.eval()
    N_min_eff = max(cfg.N_min, 5)

    cases: list = []

    # 1. low-N high-D (rank-deficient Gram)
    N_lo = min(max(N_min_eff, 5), cfg.N_max)
    D_hi = cfg.D_max
    cases.append(("low-N high-D", sample_spherical_code(N_lo, D_hi), False))

    # 2. high-N, mid-high-D
    N_hi = min(cfg.N_max, 400)
    D_mid = min(cfg.D_max, 16)
    cases.append(("high-N mid-D", sample_spherical_code(N_hi, D_mid), False))

    # 3. near-degenerate cap (all points within ~0.05 rad of a center)
    N_deg = min(cfg.N_max, 60)
    D_deg = min(cfg.D_max, 6)
    center = torch.randn(D_deg); center = center / center.norm()
    deg = center.unsqueeze(0) + 0.05 * torch.randn(N_deg, D_deg)
    deg = deg / deg.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cases.append(("near-degenerate", deg, False))

    # 4. near-repeated points (two nearly-coincident slots)
    base = sample_spherical_code(max(N_min_eff, 12), min(cfg.D_max, 5))
    rep = base.clone()
    rep[1] = base[0] + 1e-4 * torch.randn_like(base[0])
    rep[1] = rep[1] / rep[1].norm().clamp_min(1e-12)
    cases.append(("near-repeated pts", rep, False))

    # 5. antipodal ({x_i} ∪ {-x_i}); also test x → -x O(D) invariance.
    N_ant = (min(cfg.N_max, 20) // 2) * 2
    D_ant = min(cfg.D_max, 7)
    half = sample_spherical_code(N_ant // 2, D_ant)
    ant = torch.cat([half, -half], dim=0)
    cases.append(("antipodal", ant, True))

    max_drift = 0.0
    per_case = {}
    for name, code, test_negation in cases:
        N, D = code.shape
        x, mask, Ds, _ = pad_batch([code], cfg.D_max, cfg.N_max)
        x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
        z0 = encoder(x, mask, Ds)
        local_max = 0.0
        for _ in range(n_trials):
            R = random_orthogonal(D, device=device)
            perm = torch.randperm(N, device=device)
            x_t = x.clone()
            x_t[0, :N, :D] = (x[0, :N, :D] @ R)[perm]
            x_t[0, :N, :D] = x_t[0, :N, :D] / x_t[0, :N, :D].norm(
                dim=-1, keepdim=True).clamp_min(1e-12)
            z_t = encoder(x_t, mask, Ds)
            d = (z0 - z_t).abs().max().item()
            if d > local_max:
                local_max = d
        # O(D)-invariance: z(-X) == z(X) (via G = XX^T)
        if test_negation:
            x_n = x.clone()
            x_n[0, :N, :D] = -x[0, :N, :D]
            z_n = encoder(x_n, mask, Ds)
            d = (z0 - z_n).abs().max().item()
            if d > local_max:
                local_max = d
        per_case[name] = local_max
        if local_max > max_drift:
            max_drift = local_max
        if verbose:
            print(f"[edge] {name:<18} (N={N:3d}, D={D:2d}): "
                  f"max |Δz|∞ = {local_max:.2e}")
    passed = max_drift < 1e-3
    return {"edge_max_drift": max_drift, "edge_per_case": per_case,
            "edge_pass": passed}


# ---------------------------------------------------------------------------
# Discrimination (broad + hard)
# ---------------------------------------------------------------------------

@torch.no_grad()
def discrimination_test(encoder, cfg, device,
                        configs=((128, 20, 3), (128, 40, 5), (64, 100, 10),
                                 (32, 200, 20)),
                        threshold_frac: float = 0.05,
                        verbose: bool = True) -> dict:
    """Per (n_codes, N, D) setting, sample n_codes random codes and require:

        min pairwise ‖z_i − z_j‖ ≥ threshold_frac · median pairwise distance.

    Tests multiple (N, D) regimes to catch degradation at the low-N, high-N,
    low-D and high-D extremes.  Overall pass requires every configuration
    individually to pass.
    """
    was_training = encoder.training
    encoder.eval()
    per_cfg = {}
    all_passed = True
    for (n, N, D) in configs:
        N = min(N, cfg.N_max); D = min(D, cfg.D_max)
        if n < 2:
            continue
        codes = [sample_spherical_code(N, D) for _ in range(n)]
        x, mask, Ds, _ = pad_batch(codes, cfg.D_max, cfg.N_max)
        x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
        z = encoder(x, mask, Ds)
        dists = torch.pdist(z)
        d_min = dists.min().item(); d_med = dists.median().item()
        ratio = d_min / max(d_med, 1e-12)
        passed = ratio >= threshold_frac
        all_passed = all_passed and passed
        key = f"n{n}_N{N}_D{D}"
        per_cfg[key] = {"min": d_min, "median": d_med, "ratio": ratio,
                       "pass": passed}
        if verbose:
            print(f"[disc] {key}: min={d_min:.4f} med={d_med:.4f} "
                  f"ratio={ratio:.3f} pass={passed}")
    # Flat fields for the most common setting (back-compat)
    first = next(iter(per_cfg.values())) if per_cfg else {"min": 0, "median": 0}
    if was_training:
        encoder.train()
    return {"disc_min": first["min"], "disc_median": first["median"],
            "disc_per_cfg": per_cfg, "disc_pass": all_passed}


# ---------------------------------------------------------------------------
# Near-duplicate / Lipschitz behaviour
# ---------------------------------------------------------------------------

@torch.no_grad()
def near_duplicate_test(encoder, cfg, device, n_siblings: int = 32,
                        sigma: float = 0.01, N: int = 50, D: int = 6,
                        verbose: bool = True) -> dict:
    """Generate n_siblings tiny-perturbation variants of a base code (σ=0.01
    tangent-ish noise + renormalise) and verify two things at once:

    1. Cluster: max ‖z_k − z_0‖ must be bounded (not blow up to O(1)) —
       continuity of the encoder.
    2. Non-degeneracy: min pairwise distance among the n_siblings z's must be
       > 1e-5 — the encoder does NOT collapse near-duplicates to a single
       point (it's discriminative all the way down to tiny perturbations).

    Catches both (a) encoders with catastrophic Lipschitz constants and
    (b) encoders that implicitly quantize / collapse similar inputs.
    """
    encoder.eval()
    N = min(N, cfg.N_max); D = min(D, cfg.D_max)
    base = sample_spherical_code(N, D)
    codes = [base]
    for _ in range(n_siblings - 1):
        c = base + sigma * torch.randn_like(base)
        c = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        codes.append(c)
    x, mask, Ds, _ = pad_batch(codes, cfg.D_max, cfg.N_max)
    x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
    z = encoder(x, mask, Ds)
    max_drift = (z - z[:1]).norm(dim=-1).max().item()
    pdists = torch.pdist(z)
    min_pdist = pdists.min().item()
    median_pdist = pdists.median().item()
    passed = (max_drift < 10.0) and (min_pdist > 1e-5)
    if verbose:
        print(f"[near_dup] σ={sigma} N={N} D={D} n={n_siblings}: "
              f"max |Δz|={max_drift:.4f} min_pdist={min_pdist:.2e} "
              f"med_pdist={median_pdist:.4f} pass={passed}")
    return {"nd_max_drift": max_drift, "nd_min_pdist": min_pdist,
            "nd_median_pdist": median_pdist, "nd_pass": passed}


# ---------------------------------------------------------------------------
# Spectral-twin discrimination probe (1-WL hardness proxy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def spectral_twin_probe(encoder, cfg, device, n_pool: int = 200,
                        N: int = 10, D: int = 4,
                        verbose: bool = True) -> dict:
    """Among n_pool random codes at (N, D), find the pair with the closest
    Gram eigenvalue spectra (sorted) and verify the encoder still separates
    them.

    The Gram eigenvalue spectrum is the strongest genuinely spectral / 1-WL
    invariant of a spherical code (node-level features like row-sum, ‖G_i‖,
    etc. are all spectrum-determined).  A code pair with near-identical
    spectra is the regime closest to a true 1-WL collision, so a 2-WL
    encoder is expected to separate them while a plain-GNN 1-WL encoder
    would fuse them.

    Passes if the z gap on the hardest pair is at least 1% of the median
    pairwise z-gap in the pool — i.e. the "hardest" pair isn't qualitatively
    harder than an average pair.
    """
    encoder.eval()
    N = min(N, cfg.N_max); D = min(D, cfg.D_max)
    codes = [sample_spherical_code(N, D) for _ in range(n_pool)]
    # Sorted eigenvalue spectra
    spectra = []
    for c in codes:
        G = c @ c.transpose(-1, -2)
        vals, _ = torch.linalg.eigvalsh(G).sort(descending=True)
        spectra.append(vals)
    S = torch.stack(spectra, dim=0)                             # (n_pool, N)
    spec_d = torch.cdist(S, S, p=2)
    spec_d.fill_diagonal_(float("inf"))
    i_flat = int(spec_d.argmin())
    i, j = i_flat // n_pool, i_flat % n_pool
    best_spec = spec_d[i, j].item()
    # Off-diagonal Gram entries (sorted multiset distance) to confirm the
    # codes ARE distinct — a low multiset gap with high spectral match would
    # mean they're near-rot-perm-equivalent.
    G_i = codes[i] @ codes[i].transpose(-1, -2)
    G_j = codes[j] @ codes[j].transpose(-1, -2)
    off_mask = ~torch.eye(N, dtype=torch.bool)
    off_i = G_i[off_mask].sort().values
    off_j = G_j[off_mask].sort().values
    off_gap = (off_i - off_j).norm().item()

    # Encode the whole pool, then measure z-gap for the hardest pair and
    # compare to the pool's median pairwise gap.
    x, mask, Ds, _ = pad_batch(codes, cfg.D_max, cfg.N_max)
    x, mask, Ds = x.to(device), mask.to(device), Ds.to(device)
    z = encoder(x, mask, Ds)
    z_gap_ij = (z[i] - z[j]).norm().item()
    z_pdists = torch.pdist(z)
    z_med = z_pdists.median().item()
    z_min = z_pdists.min().item()
    ratio_hard = z_gap_ij / max(z_med, 1e-12)
    ratio_min = z_min / max(z_med, 1e-12)
    passed = (ratio_hard > 0.01) and (z_gap_ij > 1e-4)
    if verbose:
        print(f"[spec_twin] N={N} D={D} pool={n_pool}: "
              f"best_spec_d={best_spec:.4f} off-diag_gap={off_gap:.4f} "
              f"z_gap(hard)={z_gap_ij:.4f} z_med={z_med:.4f} "
              f"ratio={ratio_hard:.3f} min_ratio={ratio_min:.3f} pass={passed}")
    return {"spec_best_dist": best_spec, "spec_off_gap": off_gap,
            "spec_z_gap_hard": z_gap_ij, "spec_z_median": z_med,
            "spec_ratio_hard": ratio_hard, "spec_pass": passed}


# ---------------------------------------------------------------------------
# N-scale consistency (harder)
# ---------------------------------------------------------------------------

@torch.no_grad()
def n_scale_consistency(encoder, cfg, device,
                        Ns=(20, 50, 120, 300, 594),
                        Ds=(3, 5, 10),
                        verbose: bool = True) -> dict:
    """Across a full range of (N, D) covering the training span, verify that
    ‖z‖ stays within a bounded range.  Each N is tested with *repulsion-
    optimised* codes (not pure random) so the test picks up systematic drift
    with code structure, not just with random scale fluctuations.

    Pass: max/min ratio of ‖z‖ across all (N, D) < 4.0.
    """
    from data import quick_optimize                             # lazy import
    encoder.eval()
    z_norms_by_D = {}
    all_norms = []
    for D in Ds:
        if D > cfg.D_max:
            continue
        norms = []
        for N in Ns:
            N_eff = min(N, cfg.N_max)
            if N_eff < cfg.N_min:
                continue
            code = sample_spherical_code(N_eff, D)
            # quick_optimize uses autograd internally, so re-enable grad
            # locally inside this no_grad-wrapped function.
            with torch.enable_grad():
                code = quick_optimize(code, steps=8, lr=0.05, potential="coulomb")
            x, mask, Ds_t, _ = pad_batch([code], cfg.D_max, cfg.N_max)
            x, mask, Ds_t = x.to(device), mask.to(device), Ds_t.to(device)
            z = encoder(x, mask, Ds_t)
            n = z.norm().item()
            norms.append((N_eff, n))
            all_norms.append(n)
        z_norms_by_D[D] = norms
        if verbose:
            parts = [f"N={n_}:{zn:.2f}" for n_, zn in norms]
            print(f"[N-scale D={D}] " + "  ".join(parts))
    all_norms_arr = np.array(all_norms)
    ratio = all_norms_arr.max() / max(all_norms_arr.min(), 1e-12)
    passed = ratio < 4.0
    if verbose:
        print(f"[N-scale] global ratio max/min = {ratio:.2f}  pass={passed}")
    return {"nscale_z_ratio": ratio, "nscale_by_D": z_norms_by_D,
            "nscale_pass": passed}


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def run_all_hard_tests(encoder, cfg, device, verbose: bool = True) -> dict:
    """Run every hard test and return a consolidated dict + an overall pass
    flag.  Use this as a single call post-training to sanity-check the
    encoder before downstream use.
    """
    results: dict = {}
    results.update(run_invariance_tests(encoder, cfg, device, verbose=verbose))
    results.update(stress_invariance_test(encoder, cfg, device, verbose=verbose))
    results.update(platonic_invariance_test(encoder, cfg, device, verbose=verbose))
    results.update(edge_case_invariance_test(encoder, cfg, device, verbose=verbose))
    results.update(discrimination_test(encoder, cfg, device, verbose=verbose))
    results.update(near_duplicate_test(encoder, cfg, device, verbose=verbose))
    results.update(spectral_twin_probe(encoder, cfg, device, verbose=verbose))
    results.update(n_scale_consistency(encoder, cfg, device, verbose=verbose))
    pass_keys = [k for k in results if k.endswith("_pass")]
    results["all_pass"] = all(bool(results[k]) for k in pass_keys)
    if verbose:
        fails = [k for k in pass_keys if not results[k]]
        print(f"\n[summary] all_pass={results['all_pass']}  "
              f"failed: {fails if fails else 'none'}")
    return results
