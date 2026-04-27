"""Robust property evaluation for the spherical-code loss in loss.py.

The harness keeps the verbose per-test diagnostics, but also turns the suite
into a single bounded objective:

* every property test reports a capped penalty in [0, 1]
* fundamental invariance / gradient / optimisation properties carry most weight
* one catastrophic test cannot blow up the aggregate beyond its own cap
* missing, crashing, or timing-out tests are treated as worst-case penalties

The final machine-readable metric is `loss_suitability` and lower is better.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import torch

from data import (
    ArchiveCache,
    apply_rotation,
    normalize_to_sphere,
    pad_batch,
    perturb_global,
    perturb_local,
    perturb_swap,
    perturb_tangent,
    quick_optimize,
    sample_spherical_code,
)
from loss import EGWConfig, dsq_from_gram, egw_gram_loss


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
SEP = "=" * 78
SEP2 = "-" * 78

TIME_BUDGET_SECONDS = 600.0
RUNTIME_TEST_WEIGHT = 1.5


class EvaluationTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class TestSpec:
    number: int
    key: str
    label: str
    weight: float
    runner: Callable[[], dict]
    scorer: Callable[[dict], tuple[float, str]]


@dataclass
class TestOutcome:
    spec: TestSpec
    status: str
    penalty: float
    summary: str
    elapsed_seconds: float
    raw: dict[str, Any] | None = None


_BUDGET_START: float | None = None
_BUDGET_DEADLINE: float | None = None


def header(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subheader(title: str):
    print(f"\n  {title}\n  {SEP2}")


def fmt(x, width=12, prec=4):
    if x is None:
        return f"{'n/a':>{width}}"
    if isinstance(x, bool):
        return f"{str(x):>{width}}"
    if isinstance(x, int):
        return f"{x:>{width}d}"
    if not math.isfinite(x):
        return f"{'nan/inf':>{width}}"
    if abs(x) >= 1e6 or (0 < abs(x) < 1e-3):
        return f"{x:>{width}.{prec}e}"
    return f"{x:>{width}.{prec}f}"


def begin_time_budget(seconds: float):
    global _BUDGET_START, _BUDGET_DEADLINE
    _BUDGET_START = time.perf_counter()
    _BUDGET_DEADLINE = _BUDGET_START + seconds


def elapsed_seconds() -> float:
    if _BUDGET_START is None:
        return 0.0
    return max(0.0, time.perf_counter() - _BUDGET_START)


def remaining_seconds() -> float:
    if _BUDGET_DEADLINE is None:
        return float("inf")
    return _BUDGET_DEADLINE - time.perf_counter()


def check_time_budget(where: str = "evaluation"):
    if _BUDGET_DEADLINE is not None and time.perf_counter() > _BUDGET_DEADLINE:
        raise EvaluationTimeout(f"time budget exceeded during {where}")


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def blend_penalties(*parts: tuple[float, float]) -> float:
    total_weight = sum(weight for _, weight in parts)
    if total_weight <= 0:
        return 0.0
    return sum(clamp01(penalty) * weight for penalty, weight in parts) / total_weight


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(clamp01(q) * (len(ordered) - 1))
    return ordered[idx]


def lower_is_better(value: float | None, good: float, bad: float,
                    *, log_scale: bool = False) -> float:
    if value is None or not math.isfinite(value):
        return 1.0
    if value <= good:
        return 0.0
    if value >= bad:
        return 1.0
    if log_scale:
        value = max(value, 1e-30)
        good = max(good, 1e-30)
        bad = max(bad, good * (1.0 + 1e-12))
        return clamp01((math.log(value) - math.log(good)) / (math.log(bad) - math.log(good)))
    return clamp01((value - good) / max(bad - good, 1e-30))


def higher_is_better(value: float | None, good: float, bad: float,
                     *, log_scale: bool = False) -> float:
    if value is None or not math.isfinite(value):
        return 1.0
    if value >= good:
        return 0.0
    if value <= bad:
        return 1.0
    if log_scale:
        value = max(value, 1e-30)
        good = max(good, 1e-30)
        bad = max(bad, 1e-30)
        return clamp01((math.log(good) - math.log(value)) / (math.log(good) - math.log(bad)))
    return clamp01((good - value) / max(good - bad, 1e-30))


def abs_is_better(value: float | None, good: float, bad: float,
                  *, log_scale: bool = False) -> float:
    if value is None or not math.isfinite(value):
        return 1.0
    return lower_is_better(abs(value), good, bad, log_scale=log_scale)


def score_non_negativity(result: dict) -> tuple[float, str]:
    min_loss = result.get("min")
    frac_negative = result.get("frac_negative")
    penalty = blend_penalties(
        (lower_is_better(max(0.0, -(min_loss or 0.0)), 1e-8, 1e-4, log_scale=True), 3.0),
        (lower_is_better(frac_negative, 0.0, 0.05), 1.0),
    )
    summary = (
        f"min={fmt(min_loss, 10)} frac_neg={fmt(frac_negative, 8, 4)}"
    )
    return penalty, summary


def score_identity(result: dict) -> tuple[float, str]:
    self_max = result.get("self_max")
    rot_max = result.get("rot_max")
    perm_max = result.get("perm_max")
    diff_median = result.get("diff_median")
    diff_min = result.get("diff_min")
    penalty = blend_penalties(
        (abs_is_better(self_max, 1e-6, 1e-3, log_scale=True), 1.5),
        (abs_is_better(rot_max, 1e-6, 1e-3, log_scale=True), 1.5),
        (abs_is_better(perm_max, 1e-5, 1e-1, log_scale=True), 4.0),
        (higher_is_better(diff_median, 5e-2, 1e-3, log_scale=True), 2.5),
        (higher_is_better(diff_min, 1e-2, 1e-4, log_scale=True), 1.5),
    )
    summary = (
        f"self={fmt(self_max, 10)} rot={fmt(rot_max, 10)} "
        f"perm={fmt(perm_max, 10)} diff_med={fmt(diff_median, 10)}"
    )
    return penalty, summary


def score_symmetry(result: dict) -> tuple[float, str]:
    p95 = result.get("overall_p95")
    max_gap = result.get("overall_max")
    penalty = blend_penalties(
        (lower_is_better(p95, 5e-3, 0.1, log_scale=True), 3.0),
        (lower_is_better(max_gap, 2e-2, 0.2, log_scale=True), 1.0),
    )
    summary = f"p95={fmt(p95, 10, 4)} max={fmt(max_gap, 10, 4)}"
    return penalty, summary


def score_differentiability(result: dict) -> tuple[float, str]:
    finite_frac = result.get("finite_frac")
    gated_frac = result.get("gated_frac")
    fd_med_3 = result.get("fd_med_3")
    fd_p90_3 = result.get("fd_p90_3")
    penalty = blend_penalties(
        (higher_is_better(finite_frac, 1.0, 0.8), 2.0),
        (lower_is_better(gated_frac, 0.0, 0.4), 2.0),
        (lower_is_better(fd_med_3, 5e-2, 1.0, log_scale=True), 3.0),
        (lower_is_better(fd_p90_3, 2e-1, 2.0, log_scale=True), 2.0),
    )
    summary = (
        f"finite={fmt(finite_frac, 8, 3)} gated={fmt(gated_frac, 8, 3)} "
        f"fd_med={fmt(fd_med_3, 10, 3)} fd_p90={fmt(fd_p90_3, 10, 3)}"
    )
    return penalty, summary


def score_lipschitz(result: dict) -> tuple[float, str]:
    g_max = result.get("g_max")
    k_max = result.get("K_max")
    penalty = blend_penalties(
        (lower_is_better(g_max, 2.0, 25.0, log_scale=True), 1.0),
        (lower_is_better(k_max, 5.0, 100.0, log_scale=True), 3.0),
    )
    summary = f"g_max={fmt(g_max, 10)} K_max={fmt(k_max, 10)}"
    return penalty, summary


def score_convexity(result: dict) -> tuple[float, str]:
    ratio = result.get("ratio_median")
    gram_err = result.get("gram_err_median")
    recovery = result.get("recovery_mean")
    penalty = blend_penalties(
        (lower_is_better(ratio, 0.1, 1.0, log_scale=True), 2.0),
        (lower_is_better(gram_err, 0.02, 0.4, log_scale=True), 3.0),
        (higher_is_better(recovery, 0.75, 0.0), 3.0),
    )
    summary = (
        f"ratio_med={fmt(ratio, 10, 4)} gram_err={fmt(gram_err, 10, 4)} "
        f"recovery={fmt(recovery, 8, 2)}"
    )
    return penalty, summary


def score_size_invariance(result: dict) -> tuple[float, str]:
    slope_random = result.get("slope_random")
    slope_perturb = result.get("slope_perturb")
    penalty = blend_penalties(
        (abs_is_better(slope_random, 5e-2, 5e-1, log_scale=True), 1.0),
        (abs_is_better(slope_perturb, 5e-2, 5e-1, log_scale=True), 1.0),
    )
    summary = f"slope_rr={fmt(slope_random, 10, 3)} slope_rp={fmt(slope_perturb, 10, 3)}"
    return penalty, summary


def score_perm_equivariance(result: dict) -> tuple[float, str]:
    median = result.get("median")
    max_err = result.get("max")
    penalty = blend_penalties(
        (lower_is_better(median, 1e-3, 1.0, log_scale=True), 3.0),
        (lower_is_better(max_err, 1e-2, 1.5, log_scale=True), 2.0),
    )
    summary = f"median={fmt(median, 10, 4)} max={fmt(max_err, 10, 4)}"
    return penalty, summary


def score_grad_zero(result: dict) -> tuple[float, str]:
    self_max = result.get("self_max")
    rot_max = result.get("rot_max")
    perm_max = result.get("perm_max")
    penalty = blend_penalties(
        (lower_is_better(self_max, 1e-6, 1e-2, log_scale=True), 1.5),
        (lower_is_better(rot_max, 1e-6, 1e-2, log_scale=True), 1.5),
        (lower_is_better(perm_max, 1e-4, 3e-1, log_scale=True), 4.0),
    )
    summary = (
        f"self={fmt(self_max, 10)} rot={fmt(rot_max, 10)} perm={fmt(perm_max, 10)}"
    )
    return penalty, summary


def score_scaling(result: dict) -> tuple[float, str]:
    per_nd = result.get("per_nd", {})
    scaled = []
    for row in per_nd.values():
        if len(row) >= 4:
            scaled.extend([abs(row[0]), abs(row[2]), abs(row[3])])
    median_offscale = statistics.median(scaled) if scaled else float("nan")
    penalty = lower_is_better(median_offscale, 2.0, 5e2, log_scale=True)
    summary = f"median_offscale={fmt(median_offscale, 10, 2)}"
    return penalty, summary


def score_triangle(result: dict) -> tuple[float, str]:
    p95 = result.get("p95")
    max_v = result.get("max")
    penalty = blend_penalties(
        (lower_is_better(p95, 1e-2, 2e-1, log_scale=True), 2.0),
        (lower_is_better(max_v, 3e-2, 3e-1, log_scale=True), 1.0),
    )
    summary = f"p95={fmt(p95, 10, 3)} max={fmt(max_v, 10, 3)}"
    return penalty, summary


def score_precision(result: dict) -> tuple[float, str]:
    per_dtype = result.get("per_dtype", {})
    nan_loss = max((row.get("nan_L_frac", 1.0) for row in per_dtype.values()), default=1.0)
    nan_grad = max((row.get("nan_g_frac", 1.0) for row in per_dtype.values()), default=1.0)
    penalty = blend_penalties(
        (lower_is_better(nan_loss, 0.0, 1.0), 1.0),
        (lower_is_better(nan_grad, 0.0, 1.0), 1.0),
    )
    summary = f"nan_loss={fmt(nan_loss, 8, 3)} nan_grad={fmt(nan_grad, 8, 3)}"
    return penalty, summary


def score_continuity(result: dict) -> tuple[float, str]:
    frac = result.get("discontinuity_frac")
    penalty = lower_is_better(frac, 0.05, 0.8)
    summary = f"discontinuity_frac={fmt(frac, 8, 3)}"
    return penalty, summary


def score_degenerate(result: dict) -> tuple[float, str]:
    bad = result.get("bad")
    total = result.get("total")
    frac_bad = (bad / total) if total else float("nan")
    penalty = lower_is_better(frac_bad, 0.0, 0.2)
    summary = f"bad={bad}/{total}"
    return penalty, summary


def score_tractability(result: dict) -> tuple[float, str]:
    worst_fwd = result.get("worst_fwd_ms")
    worst_bwd = result.get("worst_bwd_ms")
    penalty = blend_penalties(
        (lower_is_better(worst_fwd, 5e2, 5e3, log_scale=True), 1.0),
        (lower_is_better(worst_bwd, 5e2, 5e3, log_scale=True), 1.0),
    )
    summary = f"worst_fwd_ms={fmt(worst_fwd, 10, 1)} worst_bwd_ms={fmt(worst_bwd, 10, 1)}"
    return penalty, summary


def score_padding(result: dict) -> tuple[float, str]:
    median = result.get("median")
    max_diff = result.get("max")
    penalty = blend_penalties(
        (lower_is_better(median, 1e-5, 1e-2, log_scale=True), 1.0),
        (lower_is_better(max_diff, 1e-4, 1e-1, log_scale=True), 3.0),
    )
    summary = f"median={fmt(median, 10, 4)} max={fmt(max_diff, 10, 4)}"
    return penalty, summary


def score_batch(result: dict) -> tuple[float, str]:
    rel_l = result.get("rel_L_max")
    rel_g = result.get("rel_g_max")
    penalty = blend_penalties(
        (lower_is_better(rel_l, 1e-4, 5e-2, log_scale=True), 1.0),
        (lower_is_better(rel_g, 1e-3, 1.0, log_scale=True), 3.0),
    )
    summary = f"rel_L_max={fmt(rel_l, 10, 4)} rel_g_max={fmt(rel_g, 10, 4)}"
    return penalty, summary


def score_runtime(total_seconds: float, timed_out: bool,
                  budget_seconds: float) -> tuple[float, str]:
    penalty = 1.0 if timed_out else lower_is_better(total_seconds, 420.0, budget_seconds)
    summary = f"elapsed={fmt(total_seconds, 8, 1)} budget={fmt(budget_seconds, 8, 1)}"
    return penalty, summary


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------
@dataclass
class Fixture:
    N: int
    D: int
    code_type: str
    scale: float          # perturbation scale (or 0)
    code: torch.Tensor    # (N, D) on CPU


PERT_SCALES = [1e-3, 1e-2, 5e-2, 2e-1]
TANGENT_SCALES = [1e-2, 5e-2]
SWAP_FRACS = [0.1, 0.5]


def _gauss_perturb(x: torch.Tensor, sigma: float, gen: torch.Generator) -> torch.Tensor:
    return normalize_to_sphere(x + torch.randn(x.shape, generator=gen) * sigma)


def build_fixtures(
    sizes: list[tuple[int, int]],
    archive: ArchiveCache | None,
    seed: int = 0,
    quick: bool = False,
) -> list[Fixture]:
    """Build a structured collection of (N, D, code_type) test inputs."""
    out: list[Fixture] = []
    base_gen = torch.Generator().manual_seed(seed)

    for N, D in sizes:
        check_time_budget(f"fixture build N={N} D={D}")
        # 1. Random uniform on sphere
        torch.manual_seed(seed * 1000 + N * 31 + D)
        rand = sample_spherical_code(N, D)
        out.append(Fixture(N, D, "random", 0.0, rand.clone()))

        # 2. Archive code if available
        if archive is not None:
            arch_code = archive.sample_with_nd(N, D, generator=base_gen)
            if arch_code is not None:
                out.append(Fixture(N, D, "archive", 0.0, arch_code.clone()))

        # 3. Quick-optimized random codes
        torch.manual_seed(seed * 1000 + N * 31 + D + 7)
        opt5 = quick_optimize(rand.clone(), steps=5, lr=0.05)
        out.append(Fixture(N, D, "quick_opt_5", 0.0, opt5))
        if not quick:
            torch.manual_seed(seed * 1000 + N * 31 + D + 11)
            opt25 = quick_optimize(rand.clone(), steps=25, lr=0.05)
            out.append(Fixture(N, D, "quick_opt_25", 0.0, opt25))

        # 4. Gaussian (ambient) perturbations of the random code
        for sigma in PERT_SCALES if not quick else PERT_SCALES[1:3]:
            g = torch.Generator().manual_seed(seed * 1000 + N + D + int(sigma * 1e6))
            out.append(Fixture(N, D, f"gauss", sigma,
                               _gauss_perturb(rand.clone(), sigma, g)))

        # 5. Manifold-aware perturbations (tangent / local / global)
        if not quick:
            for sigma in TANGENT_SCALES:
                g = torch.Generator().manual_seed(seed * 1000 + N + D + int(sigma * 1e6) + 1)
                out.append(Fixture(N, D, "tangent", sigma,
                                   perturb_tangent(rand.clone(), sigma, generator=g)))
                g = torch.Generator().manual_seed(seed * 1000 + N + D + int(sigma * 1e6) + 2)
                out.append(Fixture(N, D, "local", sigma,
                                   perturb_local(rand.clone(), sigma, generator=g)))
                g = torch.Generator().manual_seed(seed * 1000 + N + D + int(sigma * 1e6) + 3)
                out.append(Fixture(N, D, "global", sigma,
                                   perturb_global(rand.clone(), sigma, generator=g)))

            # 6. Combinatorial (swap) perturbations
            for frac in SWAP_FRACS:
                g = torch.Generator().manual_seed(seed * 1000 + N + D + int(frac * 1000) + 5)
                out.append(Fixture(N, D, "swap", frac,
                                   perturb_swap(rand.clone(), frac, generator=g)))

    return out


# ---------------------------------------------------------------------------
# Loss-evaluation helpers
# ---------------------------------------------------------------------------

# Module-level solver defaults.  Set from CLI args in main() so that every
# internal make_cfg() call picks up the user-supplied values automatically.
_DEFAULT_EPSILON_REL: float = EGWConfig.epsilon_rel  # type: ignore[attr-defined]
_DEFAULT_MAX_OUTER: int = EGWConfig.max_outer        # type: ignore[attr-defined]
_DEFAULT_MAX_INNER: int = EGWConfig.max_inner        # type: ignore[attr-defined]
_DEFAULT_N_RESTARTS: int = 1  # test default (EGWConfig default may differ)


def make_cfg(use_compile: bool = False,
             n_restarts: int | None = None,
             epsilon_rel: float | None = None,
             max_outer: int | None = None,
             max_inner: int | None = None) -> EGWConfig:
    """A reproducible config: no torch.compile (avoids shape recompiles in our
    sweep).  Per-call overrides take priority; module-level defaults (set from
    CLI) are used otherwise."""
    return EGWConfig(
        use_compile=use_compile,
        n_restarts=n_restarts if n_restarts is not None else _DEFAULT_N_RESTARTS,
        epsilon_rel=epsilon_rel if epsilon_rel is not None else _DEFAULT_EPSILON_REL,
        max_outer=max_outer if max_outer is not None else _DEFAULT_MAX_OUTER,
        max_inner=max_inner if max_inner is not None else _DEFAULT_MAX_INNER,
    )


def to_batch(code: torch.Tensor, device: torch.device,
             pad_extra_n: int = 0, pad_extra_d: int = 0,
             dtype: torch.dtype = torch.float32):
    """Pack a single (N, D) code into (1, N+extra_n, D+extra_d) with mask."""
    N, D = code.shape
    N_pad = N + pad_extra_n
    D_pad = D + pad_extra_d
    x, mask, _, _ = pad_batch([code], D_pad, N_pad)
    return x.to(device=device, dtype=dtype), mask.to(device=device)


def eval_loss(pred: torch.Tensor, target: torch.Tensor,
              pmask: torch.Tensor, tmask: torch.Tensor,
              cfg: EGWConfig | None = None,
              reduction: str = "mean",
              return_info: bool = False):
    check_time_budget("loss evaluation")
    if cfg is None:
        cfg = make_cfg()
    return egw_gram_loss(pred, target, pmask, tmask,
                         cfg=cfg, return_info=return_info, reduction=reduction)


def loss_and_grad(code_pred: torch.Tensor, code_tgt: torch.Tensor,
                  device: torch.device, cfg: EGWConfig | None = None):
    """Return (loss_scalar, grad_w.r.t._pred_unpadded) on (N_p, D) shape."""
    check_time_budget("loss backward")
    Np, Dp = code_pred.shape
    Nt, Dt = code_tgt.shape
    D = max(Dp, Dt)
    N = max(Np, Nt)
    p, pm = to_batch(code_pred, device, pad_extra_n=N - Np, pad_extra_d=D - Dp)
    t, tm = to_batch(code_tgt, device, pad_extra_n=N - Nt, pad_extra_d=D - Dt)
    p = p.detach().clone().requires_grad_(True)
    loss = eval_loss(p, t, pm, tm, cfg=cfg)
    loss.backward()
    g = p.grad[0, :Np, :Dp].detach().clone()
    return float(loss.item()), g


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

def test_non_negativity(fixtures: list[Fixture], device: torch.device,
                        n_pairs: int = 30) -> dict:
    check_time_budget("test 1")
    header("TEST 1: NON-NEGATIVITY")
    print("  L(A, B) ≥ 0 for all (A, B). Threshold: min ≥ -1e-6.\n")
    print(f"  {'N':>4} {'D':>3}  {'pair_kind':<22} {'L':>14}  status")
    print(f"  {SEP2}")

    rng = random.Random(1)
    all_losses = []
    worst = (None, None, None, math.inf)
    n_neg = 0

    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)

    for (N, D), pool in by_nd.items():
        for _ in range(min(n_pairs, len(pool) ** 2)):
            a = rng.choice(pool)
            b = rng.choice(pool)
            p, pm = to_batch(a.code, device)
            t, tm = to_batch(b.code, device)
            with torch.no_grad():
                L = float(eval_loss(p, t, pm, tm).item())
            all_losses.append(L)
            ok = L >= -1e-6
            if L < worst[3]:
                worst = (N, D, f"{a.code_type}->{b.code_type}", L)
            if not ok:
                n_neg += 1
                print(f"  {N:>4} {D:>3}  {a.code_type[:22]:<22} {fmt(L):>14}  FAIL")
    if all_losses:
        mn, md = min(all_losses), statistics.median(all_losses)
        print(f"\n  worst-case L = {fmt(worst[3], 14)}  at (N={worst[0]}, D={worst[1]}, {worst[2]})")
        print(f"  fraction L<-1e-6 = {n_neg}/{len(all_losses)} = {n_neg/len(all_losses):.4f}")
        print(f"  median L over all sampled pairs = {fmt(md, 14)}")
        compliance = "PASS" if mn >= -1e-6 else "FAIL"
        print(f"  COMPLIANCE: {compliance}  (min={fmt(mn, 14)})")
        return dict(min=mn, median=md, frac_negative=n_neg / max(len(all_losses), 1),
                    pass_=mn >= -1e-6)
    return dict(min=None, median=None, frac_negative=0.0, pass_=False)


def test_identity(fixtures: list[Fixture], device: torch.device,
                  n_random_targets: int = 5) -> dict:
    check_time_budget("test 2")
    header("TEST 2: IDENTITY OF INDISCERNIBLES")
    print("  Forward:  L(X, X) ≈ 0,  L(rot(X), X) ≈ 0,  L(perm(X), X) ≈ 0.")
    print("  Reverse:  L > 0 when geometries differ.")
    print("  Reports: |L| values, plus discriminative gap = median(L_diff)/median(L_self).\n")
    print(f"  {'N':>4} {'D':>3}  {'code_type':<22} "
          f"{'L(X,X)':>11} {'L(rot,X)':>11} {'L(perm,X)':>11} "
          f"{'L_diff_med':>12} {'gap':>10}")
    print(f"  {SEP2}")

    rng = random.Random(2)
    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)

    self_vals = []
    rot_vals = []
    perm_vals = []
    gaps = []
    diff_medians = []
    diff_mins = []

    for f in fixtures:
        N, D, ct = f.N, f.D, f.code_type
        # Skip degenerate pure rand entries to keep table smaller; quick_opt_25
        # and archive carry the most signal.
        if ct in {"swap", "local", "global"}:
            continue
        gen = torch.Generator().manual_seed(N + D + hash(ct) % 10000)
        rot = apply_rotation(f.code.clone(), generator=gen)
        perm_idx = torch.randperm(N, generator=gen)
        permed = f.code[perm_idx].clone()

        p, pm = to_batch(f.code, device)
        with torch.no_grad():
            l_self = float(eval_loss(p, p, pm, pm).item())

        rp, rpm = to_batch(rot, device)
        with torch.no_grad():
            l_rot = float(eval_loss(rp, p, rpm, pm).item())

        pp, ppm = to_batch(permed, device)
        with torch.no_grad():
            l_perm = float(eval_loss(pp, p, ppm, pm).item())

        # discriminative scale: loss against random different codes
        diff_pool = [g for g in by_nd[(N, D)] if g is not f]
        if diff_pool:
            diff_vals = []
            for _ in range(n_random_targets):
                other = rng.choice(diff_pool)
                op, opm = to_batch(other.code, device)
                with torch.no_grad():
                    diff_vals.append(float(eval_loss(p, op, pm, opm).item()))
            diff_med = statistics.median(diff_vals)
            diff_min = min(diff_vals)
        else:
            diff_med = float("nan")
            diff_min = float("nan")
        gap = diff_med / max(abs(l_self), 1e-12) if math.isfinite(diff_med) else float("nan")

        self_vals.append(l_self)
        rot_vals.append(l_rot)
        perm_vals.append(l_perm)
        if math.isfinite(diff_med):
            diff_medians.append(diff_med)
        if math.isfinite(diff_min):
            diff_mins.append(diff_min)
        if math.isfinite(gap):
            gaps.append(gap)

        ct_label = ct + (f"@{f.scale:g}" if f.scale > 0 else "")
        print(f"  {N:>4} {D:>3}  {ct_label[:22]:<22} "
              f"{fmt(l_self, 11)} {fmt(l_rot, 11)} {fmt(l_perm, 11)} "
              f"{fmt(diff_med, 12)} {fmt(gap, 10, 2)}")

    print()
    if self_vals:
        print(f"  worst |L(X,X)|       = {fmt(max(map(abs, self_vals)), 12)}")
        print(f"  worst |L(rot, X)|    = {fmt(max(map(abs, rot_vals)), 12)}")
        print(f"  worst |L(perm, X)|   = {fmt(max(map(abs, perm_vals)), 12)}")
    if gaps:
        gaps = sorted(gaps)
        print(f"  discriminative gap   median = {fmt(statistics.median(gaps), 10, 2)}, "
              f"min = {fmt(min(gaps), 10, 2)}, max = {fmt(max(gaps), 10, 2)}")
        print(f"  COMPLIANCE: gap>>1 means geometries are distinguishable; "
              f"|L(.,.)|<1e-3 expected for invariants.")
    return dict(
        self_max=max(map(abs, self_vals)) if self_vals else None,
        rot_max=max(map(abs, rot_vals)) if rot_vals else None,
        perm_max=max(map(abs, perm_vals)) if perm_vals else None,
        gap_median=statistics.median(gaps) if gaps else None,
        diff_median=statistics.median(diff_medians) if diff_medians else None,
        diff_min=min(diff_mins) if diff_mins else None,
    )


def test_symmetry(fixtures: list[Fixture], device: torch.device,
                  n_pairs: int = 30) -> dict:
    check_time_budget("test 3")
    header("TEST 3: SYMMETRY  L(A,B) == L(B,A)")
    print("  Reports relative gap |L_AB - L_BA| / (|L_AB| + |L_BA| + eps).")
    print("  EGW solver is stochastic via warm-starts; we use n_restarts=1 + fixed")
    print("  symmetry_break_seed for the main test, then rerun with n_restarts=3.\n")

    rng = random.Random(3)
    cfg1 = make_cfg(n_restarts=1)
    cfg3 = make_cfg(n_restarts=3)

    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)

    print(f"  {'N':>4} {'D':>3}  "
          f"{'gap_med (r=1)':>14} {'gap_p95 (r=1)':>14} {'gap_max (r=1)':>14} "
          f"{'gap_med (r=3)':>14}")
    print(f"  {SEP2}")
    overall_max = 0.0
    out = {}
    all_gaps = []
    for (N, D), pool in by_nd.items():
        if len(pool) < 2:
            continue
        gaps_r1 = []
        gaps_r3 = []
        for _ in range(n_pairs):
            a = rng.choice(pool)
            b = rng.choice(pool)
            if a is b:
                continue
            p, pm = to_batch(a.code, device)
            t, tm = to_batch(b.code, device)
            with torch.no_grad():
                lab = float(eval_loss(p, t, pm, tm, cfg=cfg1).item())
                lba = float(eval_loss(t, p, tm, pm, cfg=cfg1).item())
                gap = abs(lab - lba) / (abs(lab) + abs(lba) + 1e-12)
                gaps_r1.append(gap)
                all_gaps.append(gap)

                lab3 = float(eval_loss(p, t, pm, tm, cfg=cfg3).item())
                lba3 = float(eval_loss(t, p, tm, pm, cfg=cfg3).item())
                gap3 = abs(lab3 - lba3) / (abs(lab3) + abs(lba3) + 1e-12)
                gaps_r3.append(gap3)
        if not gaps_r1:
            continue
        gaps_r1.sort()
        med1 = statistics.median(gaps_r1)
        p95_1 = gaps_r1[int(0.95 * (len(gaps_r1) - 1))]
        max1 = max(gaps_r1)
        med3 = statistics.median(gaps_r3) if gaps_r3 else float("nan")
        overall_max = max(overall_max, max1)
        out[(N, D)] = dict(med=med1, p95=p95_1, max=max1, med_r3=med3)
        print(f"  {N:>4} {D:>3}  {fmt(med1, 14, 4)} {fmt(p95_1, 14, 4)} "
              f"{fmt(max1, 14, 4)} {fmt(med3, 14, 4)}")

    print(f"\n  worst relative gap (r=1) = {fmt(overall_max, 12, 4)}")
    print("  COMPLIANCE: gap_p95 < 0.05 means functionally symmetric;")
    print("              larger values reflect entropy + warm-start stochasticity.")
    return dict(
        overall_max=overall_max,
        overall_median=statistics.median(all_gaps) if all_gaps else None,
        overall_p95=percentile(all_gaps, 0.95) if all_gaps else None,
        per_nd=out,
    )


def test_differentiability(fixtures: list[Fixture], device: torch.device,
                           n_per_nd: int = 4) -> dict:
    check_time_budget("test 4")
    header("TEST 4: DIFFERENTIABILITY")
    print("  (a) loss.backward() produces finite gradients.")
    print("  (b) Finite-difference numerical-gradient check (only on samples")
    print("      where the convergence-gate did NOT zero the gradient):")
    print("      relerr = |<g, δ> - (L(X+εδ)-L(X-εδ))/(2ε)| / |fd_estimate|.\n")
    print(f"  {'N':>4} {'D':>3}  {'code_type':<18} "
          f"{'gated':>6} {'finite':>7} {'||g||_F':>11} {'fd@1e-3':>11} {'fd@1e-4':>11}")
    print(f"  {SEP2}")

    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)

    rng = random.Random(4)
    fd_active = []        # only gate-passing samples
    finite_count = 0
    gated_count = 0
    total = 0
    grad_norms = []
    for (N, D), pool in by_nd.items():
        sample_pool = rng.sample(pool, min(n_per_nd, len(pool)))
        for f in sample_pool:
            # Use a different random target so loss is meaningfully > 0
            tgt = rng.choice([x for x in pool if x is not f] or [f])
            p, pm = to_batch(f.code, device)
            t, tm = to_batch(tgt.code, device)
            p.requires_grad_(True)
            loss, info = eval_loss(p, t, pm, tm, return_info=True)
            loss.backward()
            g = p.grad[0, :N, :D].detach().clone()
            finite = torch.isfinite(g).all().item()
            gated = info["n_valid_batches"] == 0   # gate killed this sample
            gnorm = float(g.norm().item())
            grad_norms.append(gnorm)
            total += 1
            if finite:
                finite_count += 1
            if gated:
                gated_count += 1

            # Finite-difference along a unit random direction
            torch.manual_seed(N * 13 + D * 17)
            delta = torch.randn_like(f.code)
            delta = delta / delta.norm().clamp_min(1e-12)
            directional = float((g * delta.to(device)).sum().item())

            fd_results = {}
            for eps in (1e-3, 1e-4):
                code_p = f.code + eps * delta
                code_m = f.code - eps * delta
                p_p, _ = to_batch(code_p, device)
                p_m, _ = to_batch(code_m, device)
                with torch.no_grad():
                    Lp = float(eval_loss(p_p, t, pm, tm).item())
                    Lm = float(eval_loss(p_m, t, pm, tm).item())
                fd = (Lp - Lm) / (2 * eps)
                relerr = abs(directional - fd) / max(abs(fd), 1e-9)
                fd_results[eps] = relerr
            if not gated:
                fd_active.append(fd_results)
            print(f"  {N:>4} {D:>3}  {f.code_type[:18]:<18} "
                  f"{str(gated):>6} {str(finite):>7} {fmt(gnorm, 11)} "
                  f"{fmt(fd_results[1e-3], 11, 3)} {fmt(fd_results[1e-4], 11, 3)}")

    finite_frac = finite_count / max(total, 1)
    gated_frac = gated_count / max(total, 1)
    fd_med_3 = statistics.median([d[1e-3] for d in fd_active]) if fd_active else float("nan")
    fd_med_4 = statistics.median([d[1e-4] for d in fd_active]) if fd_active else float("nan")
    fd_p90_3 = percentile([d[1e-3] for d in fd_active], 0.90) if fd_active else float("nan")
    fd_p90_4 = percentile([d[1e-4] for d in fd_active], 0.90) if fd_active else float("nan")
    print(f"\n  finite gradient fraction          = {finite_frac:.3f}  ({finite_count}/{total})")
    print(f"  gated-out (zero-grad) fraction    = {gated_frac:.3f}  ({gated_count}/{total})")
    print(f"  median fd-relerr @ε=1e-3 (active) = {fmt(fd_med_3, 12, 4)}  ({len(fd_active)} samples)")
    print(f"  median fd-relerr @ε=1e-4 (active) = {fmt(fd_med_4, 12, 4)}")
    print(f"  COMPLIANCE: low gated_frac + low fd-relerr ⇒ envelope gradient")
    print(f"              is finite *and* consistent with finite differences.")
    return dict(finite_frac=finite_frac, gated_frac=gated_frac,
                fd_med_3=fd_med_3, fd_med_4=fd_med_4,
                fd_p90_3=fd_p90_3, fd_p90_4=fd_p90_4,
                grad_norm_med=statistics.median(grad_norms) if grad_norms else None)


def test_lipschitz(fixtures: list[Fixture], device: torch.device,
                   n_per_nd: int = 3) -> dict:
    check_time_budget("test 5")
    header("TEST 5: BOUNDED GRADIENT (Lipschitz)")
    print("  Reports per-(N,D) distribution of ||∇_X L||_F across code types,")
    print("  plus an empirical Lipschitz estimate K = sup |ΔL| / ||δ||_F.\n")
    print(f"  {'N':>4} {'D':>3}  {'||g|| med':>11} {'||g|| p95':>11} {'||g|| max':>11}  "
          f"{'K@1e-3':>10} {'K@1e-2':>10} {'K@1e-1':>10}")
    print(f"  {SEP2}")

    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)

    rng = random.Random(5)
    overall_max_g = 0.0
    overall_max_K = 0.0
    out = {}
    for (N, D), pool in by_nd.items():
        sub = rng.sample(pool, min(n_per_nd, len(pool)))
        gnorms = []
        K_per_scale: dict[float, list[float]] = {1e-3: [], 1e-2: [], 1e-1: []}
        for f in sub:
            tgt = rng.choice([x for x in pool if x is not f] or [f])
            _, g = loss_and_grad(f.code, tgt.code, device)
            gnorms.append(float(g.norm().item()))
            # K estimate: probe several directions, several scales
            for s in K_per_scale:
                Ks = []
                for k in range(3):
                    torch.manual_seed(N + D + k + int(s * 1e6))
                    delta = torch.randn_like(f.code)
                    delta = delta / delta.norm().clamp_min(1e-12) * s
                    pp, ppm = to_batch(f.code + delta, device)
                    pn, pnm = to_batch(f.code, device)
                    t, tm = to_batch(tgt.code, device)
                    with torch.no_grad():
                        Lp = float(eval_loss(pp, t, ppm, tm).item())
                        Ln = float(eval_loss(pn, t, pnm, tm).item())
                    Ks.append(abs(Lp - Ln) / s)
                K_per_scale[s].append(max(Ks))
        gnorms.sort()
        med = statistics.median(gnorms)
        p95 = gnorms[int(0.95 * (len(gnorms) - 1))]
        mx = max(gnorms)
        K1, K2, K3 = (max(K_per_scale[s]) for s in (1e-3, 1e-2, 1e-1))
        overall_max_g = max(overall_max_g, mx)
        overall_max_K = max(overall_max_K, K1, K2, K3)
        out[(N, D)] = dict(g_med=med, g_p95=p95, g_max=mx, K1=K1, K2=K2, K3=K3)
        print(f"  {N:>4} {D:>3}  {fmt(med, 11)} {fmt(p95, 11)} {fmt(mx, 11)}  "
              f"{fmt(K1, 10)} {fmt(K2, 10)} {fmt(K3, 10)}")

    print(f"\n  worst ||∇_X L||_F = {fmt(overall_max_g, 12)}")
    print(f"  worst empirical Lipschitz = {fmt(overall_max_K, 12)}")
    print("  COMPLIANCE: bounded means both worst-case quantities stay finite "
          "and roughly O(1) across (N, D).")
    return dict(g_max=overall_max_g, K_max=overall_max_K, per_nd=out)


def test_convexity_recovery(fixtures: list[Fixture], device: torch.device,
                            quick: bool = False) -> dict:
    check_time_budget("test 6")
    header("TEST 6: CONVEXITY OF SURROGATE  (recovery via gradient descent)")
    print("  Initialise X random; run Adam (lr=0.05) on L(X, X_t) for T steps,")
    print("  reproject to sphere each step. Reports final-loss/initial-loss,")
    print("  Gram error to target, and recovery rate (Gram err < 1e-2).\n")

    print(f"  {'N':>4} {'D':>3}  {'target':<18} "
          f"{'L_final/L_init':>14} {'Gram_err':>11} "
          f"{'recovered':>11}")
    print(f"  {SEP2}")
    n_seeds = 2 if quick else 4
    n_steps = 80 if quick else 200

    # Pick a single representative target per (N, D): prefer archive, then quick_opt_25
    by_nd: dict[tuple[int, int], Fixture] = {}
    for f in fixtures:
        if f.code_type in {"archive", "quick_opt_25"} and (f.N, f.D) not in by_nd:
            by_nd[(f.N, f.D)] = f
    if not by_nd:
        for f in fixtures:
            by_nd.setdefault((f.N, f.D), f)
    out = {}
    for (N, D), tgt in by_nd.items():
        if N > 200:  # too slow for default
            if quick:
                continue
        ratios = []
        gram_errs = []
        recovered = 0
        for s in range(n_seeds):
            torch.manual_seed(s * 101 + N + D)
            x_init = sample_spherical_code(N, D)
            t, tm = to_batch(tgt.code, device)
            p, pm = to_batch(x_init, device)
            X = p[:, :N, :D].detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([X], lr=0.05)
            with torch.no_grad():
                p_full = torch.zeros_like(p)
                p_full[:, :N, :D] = X
                L0 = float(eval_loss(p_full, t, pm, tm).item())

            L_last = L0
            for _ in range(n_steps):
                opt.zero_grad()
                p_full = torch.zeros_like(p)
                p_full[:, :N, :D] = X
                loss = eval_loss(p_full, t, pm, tm)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    norms = X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    X.data = X.data / norms
                L_last = float(loss.item())
            ratio = L_last / max(L0, 1e-12)
            ratios.append(ratio)
            with torch.no_grad():
                G_t_t = tgt.code @ tgt.code.T
                G_x = X[0, :N, :D].detach().cpu() @ X[0, :N, :D].detach().cpu().T
                # Compare Gram (rotation-invariant), but also need permutation
                # invariance: use the sorted eigenvalues as a coarse measure.
                ev_t = torch.linalg.eigvalsh(G_t_t).cpu()
                ev_x = torch.linalg.eigvalsh(G_x).cpu()
                gram_err = float((ev_t - ev_x).norm() / (ev_t.norm() + 1e-12))
            gram_errs.append(gram_err)
            if gram_err < 1e-2:
                recovered += 1

        rec_rate = recovered / max(n_seeds, 1)
        out[(N, D)] = dict(
            ratio_med=statistics.median(ratios),
            gram_err_med=statistics.median(gram_errs),
            recovery=rec_rate,
        )
        print(f"  {N:>4} {D:>3}  {tgt.code_type[:18]:<18} "
              f"{fmt(statistics.median(ratios), 14, 4)} "
              f"{fmt(statistics.median(gram_errs), 11, 4)} "
              f"{fmt(rec_rate, 11, 2)}")

    print("\n  COMPLIANCE: recovery > 0.5 means the entropic surrogate is "
          "convex enough for plain GD to find a near-isometric match.")
    ratios = [row["ratio_med"] for row in out.values() if math.isfinite(row["ratio_med"])]
    gram_errs = [row["gram_err_med"] for row in out.values() if math.isfinite(row["gram_err_med"])]
    recoveries = [row["recovery"] for row in out.values() if math.isfinite(row["recovery"])]
    return dict(
        per_nd=out,
        ratio_median=statistics.median(ratios) if ratios else None,
        gram_err_median=statistics.median(gram_errs) if gram_errs else None,
        recovery_mean=(sum(recoveries) / len(recoveries)) if recoveries else None,
    )


def test_size_invariance(archive: ArchiveCache | None, device: torch.device) -> dict:
    check_time_budget("test 7")
    header("TEST 7: MARGINAL UNIFORMITY / SIZE INVARIANCE")
    print("  Same code-type and σ, varying N (D fixed). Loss should not")
    print("  systematically scale with N. Reports loss vs N and log/log slope.\n")

    Ns = [20, 40, 80, 160, 320]
    D = 8
    sigma = 5e-2
    print(f"  D = {D},  Gaussian σ = {sigma}")
    print(f"  {'N':>4}  {'random vs random':>18}  {'X vs perturb(X)':>18}")
    print(f"  {SEP2}")
    rr = []
    rp = []
    for N in Ns:
        torch.manual_seed(N)
        a = sample_spherical_code(N, D)
        b = sample_spherical_code(N, D)
        gen = torch.Generator().manual_seed(N + 1)
        ap = _gauss_perturb(a.clone(), sigma, gen)
        pa, pm = to_batch(a, device)
        pb, _ = to_batch(b, device)
        pap, _ = to_batch(ap, device)
        with torch.no_grad():
            l_rr = float(eval_loss(pa, pb, pm, pm).item())
            l_rp = float(eval_loss(pa, pap, pm, pm).item())
        rr.append((N, l_rr))
        rp.append((N, l_rp))
        print(f"  {N:>4}  {fmt(l_rr, 18)}  {fmt(l_rp, 18)}")

    def slope(pairs):
        xs = [math.log(n) for n, _ in pairs if pairs and pairs[0][1] > 0]
        ys = [math.log(max(v, 1e-30)) for _, v in pairs]
        if len(xs) < 2:
            return float("nan")
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den if den > 0 else float("nan")

    s_rr = slope(rr)
    s_rp = slope(rp)
    print(f"\n  log(L) vs log(N) slope:  random/random = {fmt(s_rr, 9, 3)}, "
          f"X/perturb(X) = {fmt(s_rp, 9, 3)}")
    print("  COMPLIANCE: slope ≈ 0 means scale-invariant in N. Non-zero slope "
          "tells you the loss should be normalised before mixing batches.")
    return dict(slope_random=s_rr, slope_perturb=s_rp)


def test_perm_equivariance(fixtures: list[Fixture], device: torch.device,
                           n_per_nd: int = 3) -> dict:
    check_time_budget("test 8")
    header("TEST 8: GRADIENT PERMUTATION EQUIVARIANCE")
    print("  Permute X by P ⇒ gradient w.r.t. P·X must equal P·gradient(X).")
    print("  Reports relative error ||P·g - g'||_F / (||g||_F + ε).\n")
    print(f"  {'N':>4} {'D':>3}  {'code_type':<18} {'rel_err':>11} {'finite':>7}")
    print(f"  {SEP2}")

    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)

    rng = random.Random(8)
    rel_errs = []
    for (N, D), pool in by_nd.items():
        sub = rng.sample(pool, min(n_per_nd, len(pool)))
        for f in sub:
            tgt = rng.choice([x for x in pool if x is not f] or [f])
            torch.manual_seed(N * 7 + D * 5 + hash(f.code_type) % 1000)
            P_idx = torch.randperm(N)
            x_perm = f.code[P_idx].clone()

            _, g1 = loss_and_grad(f.code, tgt.code, device)
            _, g2 = loss_and_grad(x_perm, tgt.code, device)
            # Pg1: permute rows of g1 by P
            Pg1 = g1[P_idx]
            err = float((Pg1 - g2).norm() / (g1.norm() + 1e-12))
            finite = torch.isfinite(Pg1).all() and torch.isfinite(g2).all()
            rel_errs.append(err)
            print(f"  {N:>4} {D:>3}  {f.code_type[:18]:<18} {fmt(err, 11, 4)} "
                  f"{str(bool(finite)):>7}")
    print()
    if rel_errs:
        print(f"  median = {fmt(statistics.median(rel_errs), 12, 4)}, "
              f"max = {fmt(max(rel_errs), 12, 4)}")
        print("  COMPLIANCE: rel_err << 1 means permutation-equivariant; "
              "EGW solver stochasticity inflates this.")
    return dict(median=statistics.median(rel_errs) if rel_errs else None,
                max=max(rel_errs) if rel_errs else None)


def test_grad_zero_at_target(fixtures: list[Fixture], device: torch.device) -> dict:
    check_time_budget("test 9")
    header("TEST 9: GRADIENT ZERO AT TARGET")
    print("  When G(X) ≅ G(target), ∇_X L should be zero. We test:")
    print("    pred = X (self), pred = rotated(X), pred = permuted(X).\n")
    print(f"  {'N':>4} {'D':>3}  {'code_type':<18} "
          f"{'||g_self||':>12} {'||g_rot||':>12} {'||g_perm||':>12}")
    print(f"  {SEP2}")
    self_norms = []
    rot_norms = []
    perm_norms = []
    for f in fixtures:
        if f.code_type in {"swap", "local", "global"}:
            continue
        gen = torch.Generator().manual_seed(f.N + f.D + hash(f.code_type) % 1000)
        rot = apply_rotation(f.code.clone(), generator=gen)
        P = torch.randperm(f.N, generator=gen)
        permed = f.code[P].clone()
        _, g_self = loss_and_grad(f.code.clone(), f.code, device)
        _, g_rot = loss_and_grad(rot, f.code, device)
        _, g_perm = loss_and_grad(permed, f.code, device)
        ns = float(g_self.norm().item())
        nr = float(g_rot.norm().item())
        np_ = float(g_perm.norm().item())
        self_norms.append(ns)
        rot_norms.append(nr)
        perm_norms.append(np_)
        ct_label = f.code_type + (f"@{f.scale:g}" if f.scale > 0 else "")
        print(f"  {f.N:>4} {f.D:>3}  {ct_label[:18]:<18} "
              f"{fmt(ns, 12)} {fmt(nr, 12)} {fmt(np_, 12)}")
    print()
    if self_norms:
        print(f"  worst ||g_self|| = {fmt(max(self_norms), 12)}")
        print(f"  worst ||g_rot||  = {fmt(max(rot_norms), 12)}")
        print(f"  worst ||g_perm|| = {fmt(max(perm_norms), 12)}")
    print("  COMPLIANCE: all three should be << 1; non-zero values reflect "
          "non-converged solver state at the optimum.")
    return dict(
        self_max=max(self_norms) if self_norms else None,
        rot_max=max(rot_norms) if rot_norms else None,
        perm_max=max(perm_norms) if perm_norms else None,
    )


def test_scaling(fixtures: list[Fixture], device: torch.device) -> dict:
    check_time_budget("test 10")
    header("TEST 10: GEOMETRIC SCALING INVARIANCE")
    print("  L(α·X, X) should be 0 if scale-invariant. (We expect non-zero; ")
    print("  this measures the deviation.)\n")
    alphas = [0.5, 1.0, 2.0, 5.0]
    print(f"  {'N':>4} {'D':>3}  {'code_type':<14}  " +
          " ".join(f"α={a:>4g}" for a in alphas))
    print(f"  {SEP2}")
    by_nd: dict[tuple[int, int], Fixture] = {}
    for f in fixtures:
        if f.code_type == "random" and (f.N, f.D) not in by_nd:
            by_nd[(f.N, f.D)] = f
    losses = {}
    for (N, D), f in by_nd.items():
        row = []
        for a in alphas:
            x = f.code * a
            p, pm = to_batch(x, device)
            t, tm = to_batch(f.code, device)
            with torch.no_grad():
                L = float(eval_loss(p, t, pm, tm).item())
            row.append(L)
        losses[(N, D)] = row
        print(f"  {N:>4} {D:>3}  {f.code_type[:14]:<14}  " +
              " ".join(fmt(v, 7, 2) for v in row))
    print("\n  COMPLIANCE: all values should be near 0 if scale-invariant.")
    print("  EGW with D²=G_ii+G_jj-2G_ij scales D² by α², so the loss is ")
    print("  expected to grow with |α-1|.")
    return dict(per_nd=losses)


def test_triangle(fixtures: list[Fixture], device: torch.device,
                  n_triples: int = 12) -> dict:
    check_time_budget("test 11")
    header("TEST 11: TRIANGLE INEQUALITY VIOLATIONS")
    print("  For triples (A, B, C), report violation:")
    print("     v = max(0, L(A,C) - L(A,B) - L(B,C)) / L(A,C).\n")
    print(f"  {'N':>4} {'D':>3}  {'med v':>10} {'p95 v':>10} {'max v':>10}")
    print(f"  {SEP2}")
    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)
    rng = random.Random(11)
    overall_max = 0.0
    all_violations = []
    for (N, D), pool in by_nd.items():
        if len(pool) < 3:
            continue
        violations = []
        for _ in range(n_triples):
            A, B, C = rng.sample(pool, 3)
            pa, pm = to_batch(A.code, device)
            pb, _ = to_batch(B.code, device)
            pc, _ = to_batch(C.code, device)
            with torch.no_grad():
                lab = float(eval_loss(pa, pb, pm, pm).item())
                lbc = float(eval_loss(pb, pc, pm, pm).item())
                lac = float(eval_loss(pa, pc, pm, pm).item())
            if lac < 1e-12:
                continue
            # standard triangle for distances uses sqrt; we report on raw L
            v = max(0.0, lac - lab - lbc) / lac
            violations.append(v)
            all_violations.append(v)
        if not violations:
            continue
        violations.sort()
        med = statistics.median(violations)
        p95 = violations[int(0.95 * (len(violations) - 1))]
        mx = max(violations)
        overall_max = max(overall_max, mx)
        print(f"  {N:>4} {D:>3}  {fmt(med, 10, 3)} {fmt(p95, 10, 3)} {fmt(mx, 10, 3)}")
    print(f"\n  worst violation = {fmt(overall_max, 12, 3)}")
    print("  COMPLIANCE: small violations (<<1) bound the metric defect; "
          "EGW is not a strict metric so v>0 is expected.")
    return dict(
        median=statistics.median(all_violations) if all_violations else None,
        p95=percentile(all_violations, 0.95) if all_violations else None,
        max=overall_max,
    )


def test_precision(fixtures: list[Fixture], device: torch.device) -> dict:
    check_time_budget("test 12")
    header("TEST 12: STABILITY AT PRECISION LIMITS  (fp16 / bf16)")
    print("  Re-run a subset of fixtures in float16 and bfloat16. Reports")
    print("  fraction NaN/Inf in loss and grad, and relerr to fp32 value.\n")
    print(f"  {'dtype':<10} {'N':>4} {'D':>3}  {'L_fp32':>11} {'L_low':>11} "
          f"{'rel_L':>10} {'finite_L':>9} {'finite_g':>9}")
    print(f"  {SEP2}")
    rng = random.Random(12)
    pool = rng.sample(fixtures, min(8, len(fixtures)))
    out = {dt: [] for dt in (torch.float16, torch.bfloat16)}
    for dtype in (torch.float16, torch.bfloat16):
        for f in pool:
            tgt_idx = rng.randrange(len(fixtures))
            tgt = fixtures[tgt_idx]
            if (tgt.N, tgt.D) != (f.N, f.D):
                tgt = f
            # fp32 reference
            p32, pm = to_batch(f.code, device, dtype=torch.float32)
            t32, tm = to_batch(tgt.code, device, dtype=torch.float32)
            p32 = p32.clone().requires_grad_(True)
            try:
                L32 = eval_loss(p32, t32, pm, tm)
                L32.backward()
                L32_v = float(L32.item())
                g32_finite = bool(torch.isfinite(p32.grad).all().item())
            except Exception:
                L32_v = float("nan")
                g32_finite = False

            # low precision
            try:
                p_lo, _ = to_batch(f.code, device, dtype=dtype)
                t_lo, _ = to_batch(tgt.code, device, dtype=dtype)
                p_lo = p_lo.clone().requires_grad_(True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    L_lo = eval_loss(p_lo, t_lo, pm, tm)
                    L_lo.backward()
                L_lo_v = float(L_lo.item())
                finite_L = bool(torch.isfinite(L_lo).item())
                finite_g = bool(torch.isfinite(p_lo.grad).all().item())
            except Exception as e:
                L_lo_v = float("nan")
                finite_L = False
                finite_g = False

            relerr = abs(L_lo_v - L32_v) / (abs(L32_v) + 1e-12) if math.isfinite(L_lo_v + L32_v) else float("nan")
            out[dtype].append(dict(L32=L32_v, Llo=L_lo_v, finite_L=finite_L, finite_g=finite_g))
            print(f"  {str(dtype).split('.')[-1]:<10} {f.N:>4} {f.D:>3}  "
                  f"{fmt(L32_v, 11)} {fmt(L_lo_v, 11)} {fmt(relerr, 10, 3)} "
                  f"{str(finite_L):>9} {str(finite_g):>9}")
    print()
    summary = {}
    for dtype in out:
        rows = out[dtype]
        if not rows:
            continue
        nan_L = sum(1 for r in rows if not r["finite_L"])
        nan_g = sum(1 for r in rows if not r["finite_g"])
        summary[str(dtype)] = dict(
            n=len(rows),
            nan_L_frac=nan_L / len(rows),
            nan_g_frac=nan_g / len(rows),
        )
        name = str(dtype).split('.')[-1]
        print(f"  {name}: NaN/Inf in loss = {nan_L}/{len(rows)}, "
              f"in grad = {nan_g}/{len(rows)}")
    print("  COMPLIANCE: 0% NaN/Inf is the bar. Sinkhorn's exp(-C/eps) is the "
          "main risk in low precision.")
    return dict(per_dtype=summary)


def test_continuity(fixtures: list[Fixture], device: torch.device,
                    n_per_nd: int = 2) -> dict:
    check_time_budget("test 13")
    header("TEST 13: ROBUSTNESS / ADDITIVE-NOISE CONTINUITY")
    print("  For each X, compute |L(X+sδ, T) - L(X, T)| at scales s.")
    print("  Tabulates Δ vs s. Local Lipschitz Δ/s should converge as s→0.\n")
    scales = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    print(f"  {'N':>4} {'D':>3}  {'code_type':<14}  " +
          " ".join(f"Δ@{s:>4g}" for s in scales) + "  discontinuity")
    print(f"  {SEP2}")

    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)
    rng = random.Random(13)
    discontinuities = 0
    total = 0
    for (N, D), pool in by_nd.items():
        sub = rng.sample(pool, min(n_per_nd, len(pool)))
        for f in sub:
            tgt = rng.choice([x for x in pool if x is not f] or [f])
            t, tm = to_batch(tgt.code, device)
            p_base, pm = to_batch(f.code, device)
            with torch.no_grad():
                L_base = float(eval_loss(p_base, t, pm, tm).item())
            row = []
            ratios = []
            for s in scales:
                deltas = []
                for k in range(3):
                    torch.manual_seed(N + D + k + int(s * 1e8))
                    delta = torch.randn_like(f.code)
                    delta = delta / delta.norm().clamp_min(1e-12) * s
                    p_pert, _ = to_batch(f.code + delta, device)
                    with torch.no_grad():
                        L_pert = float(eval_loss(p_pert, t, pm, tm).item())
                    deltas.append(abs(L_pert - L_base))
                row.append(statistics.median(deltas))
                ratios.append(row[-1] / s)
            disc = False
            for i in range(len(ratios) - 1):
                if ratios[i] > 0 and ratios[i] > 10 * ratios[i + 1]:
                    disc = True
            if disc:
                discontinuities += 1
            total += 1
            print(f"  {N:>4} {D:>3}  {f.code_type[:14]:<14}  " +
                  " ".join(fmt(v, 7, 2) for v in row) +
                  f"  {'YES' if disc else 'no'}")
    print()
    print(f"  discontinuity flag: {discontinuities}/{total} cases.")
    print("  COMPLIANCE: monotone Δ vs s implies continuous loss; "
          "ratio plateauing as s→0 ⇒ locally Lipschitz.")
    return dict(discontinuity_frac=discontinuities / max(total, 1))


def test_degenerate(device: torch.device) -> dict:
    check_time_budget("test 14")
    header("TEST 14: DEGENERATE-STATE HANDLING")
    print("  All-collapsed point cloud (rows identical) should not produce")
    print("  NaN/Inf in loss or gradient.\n")
    print(f"  {'N':>4} {'D':>3}  {'case':<24} "
          f"{'L':>12} {'finite':>7} {'||g||':>12} {'g_finite':>9}")
    print(f"  {SEP2}")

    out = []
    for (N, D) in ((20, 3), (50, 8), (100, 16), (200, 16)):
        e1 = torch.zeros(N, D)
        e1[:, 0] = 1.0
        x_deg = e1
        # Half degenerate
        torch.manual_seed(N + D)
        x_rand = sample_spherical_code(N, D)
        x_half = torch.cat([e1[: N // 2], x_rand[N // 2:]], dim=0)

        cases = [
            ("X_deg vs X_deg", x_deg, x_deg),
            ("X_deg vs X_rand", x_deg, x_rand),
            ("X_rand vs X_deg", x_rand, x_deg),
            ("X_half vs X_rand", x_half, x_rand),
        ]
        for name, a, b in cases:
            try:
                L, g = loss_and_grad(a.clone(), b, device)
                fin = math.isfinite(L)
                gn = float(g.norm().item())
                gfin = bool(torch.isfinite(g).all().item())
            except Exception as e:
                L, fin, gn, gfin = float("nan"), False, float("nan"), False
            print(f"  {N:>4} {D:>3}  {name:<24} "
                  f"{fmt(L, 12)} {str(fin):>7} {fmt(gn, 12)} {str(gfin):>9}")
            out.append(dict(N=N, D=D, name=name, L=L, finite=fin, gn=gn, gfin=gfin))

    bad = sum(1 for r in out if not (r["finite"] and r["gfin"]))
    print(f"\n  {bad}/{len(out)} cases produced NaN/Inf in loss or grad.")
    print("  COMPLIANCE: should be 0/N. Solver gating may zero loss for "
          "non-converged samples — that is acceptable (finite, =0).")
    return dict(bad=bad, total=len(out))


def test_tractability(device: torch.device, quick: bool = False) -> dict:
    check_time_budget("test 15")
    header("TEST 15: TRACTABILITY  (timing)")
    print("  Wall-clock for forward + backward at increasing B, N.\n")
    print(f"  {'B':>3} {'N':>4} {'D':>3}  {'fwd (ms)':>10} {'bwd (ms)':>10}")
    print(f"  {SEP2}")
    cfg = make_cfg(use_compile=False)
    settings = [
        (1, 50, 8),
        (4, 50, 8),
        (1, 200, 16),
        (4, 200, 16),
    ]
    if not quick:
        settings += [(16, 100, 8), (1, 400, 24)]

    out = {}
    worst_fwd = 0.0
    worst_bwd = 0.0
    for B, N, D in settings:
        torch.manual_seed(N * 100 + D)
        codes = [sample_spherical_code(N, D) for _ in range(B)]
        targets = [sample_spherical_code(N, D) for _ in range(B)]
        x, mask, _, _ = pad_batch(codes, D, N)
        y, _, _, _ = pad_batch(targets, D, N)
        x, mask = x.to(device), mask.to(device)
        y = y.to(device)
        ymask = mask.clone()

        # warm-up
        for _ in range(2):
            xc = x.clone().requires_grad_(True)
            L = eval_loss(xc, y, mask, ymask, cfg=cfg)
            L.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()

        fwd_times = []
        bwd_times = []
        for _ in range(5):
            xc = x.clone().requires_grad_(True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            L = eval_loss(xc, y, mask, ymask, cfg=cfg)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            L.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            fwd_times.append((t1 - t0) * 1000)
            bwd_times.append((t2 - t1) * 1000)
        fwd_med = statistics.median(fwd_times)
        bwd_med = statistics.median(bwd_times)
        worst_fwd = max(worst_fwd, fwd_med)
        worst_bwd = max(worst_bwd, bwd_med)
        out[(B, N, D)] = (fwd_med, bwd_med)
        print(f"  {B:>3} {N:>4} {D:>3}  {fmt(fwd_med, 10, 2)} {fmt(bwd_med, 10, 2)}")
    return dict(per_setting=out, worst_fwd_ms=worst_fwd, worst_bwd_ms=worst_bwd)


def test_padding_invariance(fixtures: list[Fixture], device: torch.device,
                            n_per_nd: int = 3) -> dict:
    check_time_budget("test 16")
    header("TEST 16: PADDING INVARIANCE")
    print("  L(X, T) at native size vs L(pad(X), T): mask should make these equal.")
    print("  Reports relative difference per (K_extra_N, J_extra_D).\n")
    print(f"  {'N':>4} {'D':>3}  {'code_type':<14} {'K':>3} {'J':>3}  "
          f"{'L_native':>11} {'L_padded':>11} {'rel diff':>11}")
    print(f"  {SEP2}")
    rng = random.Random(16)
    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)
    rels = []
    for (N, D), pool in by_nd.items():
        sub = rng.sample(pool, min(n_per_nd, len(pool)))
        for f in sub:
            tgt = rng.choice([x for x in pool if x is not f] or [f])
            for K in (0, 5, 50):
                for J in (0, 5):
                    p_n, pm_n = to_batch(f.code, device)
                    t_n, tm_n = to_batch(tgt.code, device)
                    p_p, pm_p = to_batch(f.code, device,
                                         pad_extra_n=K, pad_extra_d=J)
                    t_p, tm_p = to_batch(tgt.code, device,
                                         pad_extra_n=K, pad_extra_d=J)
                    with torch.no_grad():
                        L_n = float(eval_loss(p_n, t_n, pm_n, tm_n).item())
                        L_p = float(eval_loss(p_p, t_p, pm_p, tm_p).item())
                    rel = abs(L_p - L_n) / (abs(L_n) + 1e-12)
                    rels.append(rel)
                    print(f"  {N:>4} {D:>3}  {f.code_type[:14]:<14} {K:>3} {J:>3}  "
                          f"{fmt(L_n, 11)} {fmt(L_p, 11)} {fmt(rel, 11, 4)}")
    print()
    if rels:
        print(f"  median rel diff = {fmt(statistics.median(rels), 12, 4)}")
        print(f"  max    rel diff = {fmt(max(rels), 12, 4)}")
    print("  COMPLIANCE: rel diff < 1e-4 means masking is exact. Larger values")
    print("  suggest padded entries leak into the cost/transport.")
    return dict(median=statistics.median(rels) if rels else None,
                max=max(rels) if rels else None)


def test_batch_invariance(fixtures: list[Fixture], device: torch.device) -> dict:
    check_time_budget("test 17")
    header("TEST 17: BATCH INVARIANCE")
    print("  Per-sample loss with reduction='none' should equal the same loss")
    print("  computed with B=1. Same for ||grad|| (per-sample subset).\n")
    print(f"  {'sample':>20}  {'L_alone':>11} {'L_batched':>11} {'rel_L':>11} "
          f"{'rel_g':>11}")
    print(f"  {SEP2}")
    rng = random.Random(17)
    # build a batch of B=4 with the same N, D for valid batching
    by_nd: dict[tuple[int, int], list[Fixture]] = {}
    for f in fixtures:
        by_nd.setdefault((f.N, f.D), []).append(f)
    candidates = [(nd, pool) for nd, pool in by_nd.items() if len(pool) >= 4]
    if not candidates:
        print("  (no (N,D) bucket has ≥4 samples; skipping)")
        return dict(rel_L_max=None, rel_g_max=None)

    relL = []
    relg = []
    for (N, D), pool in candidates[:3]:  # at most 3 buckets
        chosen = rng.sample(pool, 4)
        target_pool = [x for x in pool if x not in chosen]
        if not target_pool:
            target_pool = chosen
        targets = [rng.choice(target_pool) for _ in chosen]

        # Alone
        L_alone = []
        g_alone = []
        for f, t in zip(chosen, targets):
            L, g = loss_and_grad(f.code, t.code, device)
            L_alone.append(L)
            g_alone.append(g)

        # Batched
        x_b, m_b, _, _ = pad_batch([f.code for f in chosen], D, N)
        y_b, my_b, _, _ = pad_batch([t.code for t in targets], D, N)
        x_b = x_b.to(device).requires_grad_(True)
        y_b = y_b.to(device)
        m_b = m_b.to(device)
        my_b = my_b.to(device)
        L_per = eval_loss(x_b, y_b, m_b, my_b, reduction="none")
        # Use a fixed weighting so per-sample grads can be recovered
        L_per.sum().backward()
        L_batched = [float(v.item()) for v in L_per]
        g_batched = [x_b.grad[i, :N, :D].detach().clone() for i in range(4)]

        for i, f in enumerate(chosen):
            r_L = abs(L_batched[i] - L_alone[i]) / (abs(L_alone[i]) + 1e-12)
            r_g = float((g_batched[i] - g_alone[i]).norm() /
                        (g_alone[i].norm() + 1e-12))
            relL.append(r_L)
            relg.append(r_g)
            print(f"  {f.code_type[:18]:>20}  {fmt(L_alone[i], 11)} "
                  f"{fmt(L_batched[i], 11)} {fmt(r_L, 11, 4)} {fmt(r_g, 11, 4)}")
    print()
    print(f"  max rel L diff = {fmt(max(relL) if relL else 0, 12, 4)}")
    print(f"  max rel g diff = {fmt(max(relg) if relg else 0, 12, 4)}")
    print("  COMPLIANCE: < 1e-4 ideal; small drift acceptable from solver "
          "warm-start path differing across batch dim.")
    return dict(rel_L_max=max(relL) if relL else None,
                rel_g_max=max(relg) if relg else None)


def run_scored_test(spec: TestSpec) -> TestOutcome:
    started = time.perf_counter()
    try:
        raw = spec.runner()
        penalty, summary = spec.scorer(raw)
        status = "ok"
    except EvaluationTimeout as exc:
        raw = None
        penalty = 1.0
        summary = str(exc)
        status = "timeout"
    except Exception as exc:
        raw = None
        penalty = 1.0
        summary = f"{type(exc).__name__}: {exc}"
        status = "crash"
    elapsed = time.perf_counter() - started
    return TestOutcome(
        spec=spec,
        status=status,
        penalty=clamp01(penalty),
        summary=summary,
        elapsed_seconds=elapsed,
        raw=raw,
    )


def outcome_score(outcome: TestOutcome, total_weight: float) -> float:
    if total_weight <= 0.0 or outcome.spec.weight <= 0.0:
        return 0.0
    return 100.0 * outcome.spec.weight * outcome.penalty / total_weight


def build_scorecard(outcomes: list[TestOutcome]):
    total_weight = sum(item.spec.weight for item in outcomes if item.spec.weight > 0)
    header("SCORECARD")
    print("  score = each test's weighted contribution to the final loss_suitability")
    print(f"  {'#':>2} {'test':<23} {'status':<9} {'score':>8} {'weight':>6} {'penalty':>9} {'elapsed_s':>10}")
    print(f"  {SEP2}")
    for outcome in outcomes:
        score = outcome_score(outcome, total_weight)
        print(
            f"  {outcome.spec.number:>2} {outcome.spec.label[:23]:<23} "
            f"{outcome.status:<9} {fmt(score, 8, 4)} {fmt(outcome.spec.weight, 6, 2)} "
            f"{fmt(outcome.penalty, 9, 4)} {fmt(outcome.elapsed_seconds, 10, 2)}"
        )
        print(f"     {outcome.summary}")


def final_loss_suitability(outcomes: list[TestOutcome], total_seconds: float,
                           budget_seconds: float, timed_out: bool) -> tuple[float, TestOutcome]:
    runtime_penalty, runtime_summary = score_runtime(total_seconds, timed_out, budget_seconds)
    runtime_spec = TestSpec(
        number=0,
        key="runtime_budget",
        label="Runtime Budget",
        weight=RUNTIME_TEST_WEIGHT,
        runner=lambda: {},
        scorer=lambda _: (runtime_penalty, runtime_summary),
    )
    runtime_outcome = TestOutcome(
        spec=runtime_spec,
        status="timeout" if timed_out else "ok",
        penalty=runtime_penalty,
        summary=runtime_summary,
        elapsed_seconds=0.0,
        raw=None,
    )
    weighted = outcomes + [runtime_outcome]
    total_weight = sum(item.spec.weight for item in weighted if item.spec.weight > 0)
    aggregate = sum(item.spec.weight * item.penalty for item in weighted if item.spec.weight > 0) / max(total_weight, 1e-12)
    return 100.0 * aggregate, runtime_outcome


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global _DEFAULT_EPSILON_REL, _DEFAULT_MAX_OUTER, _DEFAULT_MAX_INNER, _DEFAULT_N_RESTARTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="alias for the compact benchmark profile")
    parser.add_argument("--stress", action="store_true",
                        help="broader, slower sweep; the default profile is kept under the 10-minute budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="cpu, cuda, or auto (default)")
    parser.add_argument("--archive-dir", default="spherical_code_archive",
                        help="directory of optimized .pt code archive")
    parser.add_argument("--time-budget-seconds", type=float, default=TIME_BUDGET_SECONDS,
                        metavar="S",
                        help=f"hard wall-clock limit for the full evaluation. Default: {TIME_BUDGET_SECONDS:.0f}")
    # Solver knobs forwarded to every make_cfg() call via module-level defaults.
    parser.add_argument("--epsilon-rel", type=float, default=None,
                        metavar="F",
                        help="EGWConfig.epsilon_rel: eps = F * median(D²). "
                             f"Default: {EGWConfig.epsilon_rel}")
    parser.add_argument("--max-outer", type=int, default=None,
                        metavar="N",
                        help="EGWConfig.max_outer: mirror-descent iteration cap. "
                             f"Default: {EGWConfig.max_outer}")
    parser.add_argument("--max-inner", type=int, default=None,
                        metavar="N",
                        help="EGWConfig.max_inner: Sinkhorn iteration cap per "
                             f"outer step. Default: {EGWConfig.max_inner}")
    parser.add_argument("--n-restarts", type=int, default=None,
                        metavar="N",
                        help="Number of independent solver restarts. "
                             f"Default: {_DEFAULT_N_RESTARTS}")
    parser.add_argument("--tests", default=None, metavar="SPEC",
                        help="Comma-separated list of test numbers/ranges to run "
                             "(e.g. '1,3,5-8'). Default: run all tests.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Apply CLI solver knobs to module-level defaults consumed by make_cfg().
    if args.epsilon_rel is not None:
        _DEFAULT_EPSILON_REL = args.epsilon_rel
    if args.max_outer is not None:
        _DEFAULT_MAX_OUTER = args.max_outer
    if args.max_inner is not None:
        _DEFAULT_MAX_INNER = args.max_inner
    if args.n_restarts is not None:
        _DEFAULT_N_RESTARTS = args.n_restarts

    # Parse --tests into a set of ints; None means "all".
    def _parse_tests(spec: str) -> set[int]:
        result: set[int] = set()
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                result.update(range(int(lo), int(hi) + 1))
            else:
                result.add(int(part))
        return result

    enabled_tests: set[int] | None = (
        _parse_tests(args.tests) if args.tests is not None else None
    )

    def run_test(n: int, fn, *a, **kw):
        if enabled_tests is not None and n not in enabled_tests:
            return None
        return fn(*a, **kw)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compact_profile = not args.stress
    benchmark_name = "compact" if compact_profile else "stress"

    begin_time_budget(args.time_budget_seconds)

    print(f"Device: {device}")
    print(f"Benchmark profile: {benchmark_name}")
    print(f"Seed: {args.seed}")
    print(f"Time budget (s): {args.time_budget_seconds:.1f}")

    archive: ArchiveCache | None = None
    if os.path.isdir(args.archive_dir):
        try:
            check_time_budget("archive load")
            archive = ArchiveCache(args.archive_dir, N_min=10, N_max=500,
                                   D_min=3, D_max=32, verbose=False)
            print(f"Loaded archive: {len(archive)} codes from {args.archive_dir}")
        except Exception as e:
            print(f"Archive load failed ({e}); proceeding with random codes only.")
            archive = None
    else:
        print(f"Archive dir '{args.archive_dir}' not found; using random codes.")

    if compact_profile:
        sizes = [(20, 3), (50, 8), (100, 16)]
    else:
        sizes = [(20, 3), (40, 5), (60, 8), (100, 12), (200, 16), (400, 24)]
    print(f"(N, D) sweep: {sizes}")

    fixtures = build_fixtures(sizes, archive, seed=args.seed, quick=compact_profile)
    print(f"Built {len(fixtures)} fixtures across "
          f"{len({(f.N, f.D) for f in fixtures})} (N, D) cells.")

    specs = [
        TestSpec(1, "non_negativity", "Non-Negativity", 2.0,
                 lambda: test_non_negativity(fixtures, device), score_non_negativity),
        TestSpec(2, "identity", "Identity", 6.0,
                 lambda: test_identity(fixtures, device), score_identity),
        TestSpec(3, "symmetry", "Symmetry", 3.0,
                 lambda: test_symmetry(fixtures, device), score_symmetry),
        TestSpec(4, "differentiability", "Differentiability", 6.0,
                 lambda: test_differentiability(fixtures, device), score_differentiability),
        TestSpec(5, "lipschitz", "Bounded Gradient", 5.0,
                 lambda: test_lipschitz(fixtures, device), score_lipschitz),
        TestSpec(6, "convexity", "Optimisation Recovery", 5.0,
                 lambda: test_convexity_recovery(fixtures, device, quick=compact_profile), score_convexity),
        TestSpec(7, "size_invariance", "Size Invariance", 2.0,
                 lambda: test_size_invariance(archive, device), score_size_invariance),
        TestSpec(8, "perm_equivariance", "Gradient Permutation Eq.", 6.0,
                 lambda: test_perm_equivariance(fixtures, device), score_perm_equivariance),
        TestSpec(9, "grad_zero", "Gradient Zero At Target", 5.0,
                 lambda: test_grad_zero_at_target(fixtures, device), score_grad_zero),
        TestSpec(10, "scaling", "Scaling Diagnostic", 0.0,
                 lambda: test_scaling(fixtures, device), score_scaling),
        TestSpec(11, "triangle", "Triangle Defect", 1.5,
                 lambda: test_triangle(fixtures, device), score_triangle),
        TestSpec(12, "precision", "Low-Precision Stability", 0.5,
                 lambda: test_precision(fixtures, device), score_precision),
        TestSpec(13, "continuity", "Continuity", 4.0,
                 lambda: test_continuity(fixtures, device), score_continuity),
        TestSpec(14, "degenerate", "Degenerate States", 3.0,
                 lambda: test_degenerate(device), score_degenerate),
        TestSpec(15, "tractability", "Tractability", 0.5,
                 lambda: test_tractability(device, quick=compact_profile), score_tractability),
        TestSpec(16, "padding", "Padding Invariance", 4.5,
                 lambda: test_padding_invariance(fixtures, device), score_padding),
        TestSpec(17, "batch", "Batch Invariance", 4.5,
                 lambda: test_batch_invariance(fixtures, device), score_batch),
    ]

    outcomes: list[TestOutcome] = []
    timed_out = False
    for spec in specs:
        if enabled_tests is not None and spec.number not in enabled_tests:
            continue
        outcome = run_scored_test(spec)
        outcomes.append(outcome)
        if outcome.status == "timeout":
            timed_out = True
            break

    if timed_out:
        for spec in specs:
            if enabled_tests is not None and spec.number not in enabled_tests:
                continue
            if any(existing.spec.number == spec.number for existing in outcomes):
                continue
            outcomes.append(TestOutcome(
                spec=spec,
                status="skipped",
                penalty=1.0,
                summary="skipped because the global time budget was exhausted",
                elapsed_seconds=0.0,
                raw=None,
            ))

    total_seconds = elapsed_seconds()
    loss_suitability, runtime_outcome = final_loss_suitability(
        outcomes, total_seconds, args.time_budget_seconds, timed_out,
    )
    build_scorecard(outcomes + [runtime_outcome])

    header("FINAL SUMMARY")
    print("---")
    print(f"loss_suitability: {loss_suitability:.6f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"status:           {'timeout' if timed_out else 'ok'}")
    print(f"tests_run:        {sum(1 for item in outcomes if item.status == 'ok')}/{len(outcomes)}")


if __name__ == "__main__":
    main()
