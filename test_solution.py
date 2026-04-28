"""
test_solution.py
================

Fixed evaluation harness for the spherical-code autoencoder.

This file is part of the evaluation infrastructure and must NOT be modified
during autonomous research experiments.  The agent edits only solution.py
and loss.py.

What this script does
---------------------
1. Asserts that solution.py exports the required public API.
2. Trains for up to TRAIN_BUDGET_SECONDS (10 minutes) using the solver defined
    in solution.py.
3. Averages and prints the scalar training diagnostics returned by
   solution.py every REPORT_EVERY_STEPS steps.
4. Runs the fixed evaluation suite once at the end.
5. Saves the final checkpoint and a compact set of artifacts.
6. Runs the latent probe suite once at the end with its own timeout budget.
7. Prints a human-readable scorecard followed by a machine-readable footer.

Machine-readable footer (grep pattern)
---------------------------------------
    grep "^final_score:\\|^train_seconds:\\|^total_seconds:\\|^status:\\|^steps_run:\\|^ckpt:" run.log

Metrics
-------
final_score: mean of clip(max(R²_linear[i], R²_mlp[i]), 0, 1) across all
  probe targets.  Higher is better.  The per-target breakdown appears in the
  printed scorecard above the footer.

Usage
-----
    uv run test_solution.py > run.log 2>&1
    uv run test_solution.py --device cpu --seed 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
import types
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Assert required solution.py API before doing anything else
# ---------------------------------------------------------------------------
import solution as _sol

_REQUIRED_API = [
    "Config",
    "SphereCodeEncoder",
    "SphereCodeDecoder",
    "build_training_state",
    "train_one_step",
    "build_val_loss_fn",
    "save_checkpoint",
    "load_checkpoint",
]
_missing = [name for name in _REQUIRED_API if not hasattr(_sol, name)]
assert not _missing, (
    f"solution.py is missing required exports: {_missing}\n"
    f"The public API of solution.py must include: {_REQUIRED_API}"
)

from solution import (
    Config,
    build_training_state,
    train_one_step,
    build_val_loss_fn,
    save_checkpoint,
)
from data import make_loader
from evaluation import build_val_codes, evaluate
from latent_test_bed import run_probe_suite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_BUDGET_SECONDS: float = 600.0  # 10 minutes for training
PROBE_BUDGET_SECONDS: float = 300.0  # 5 minutes max for probe suite
REPORT_EVERY_STEPS: int = 250
LOADER_CHUNK: int = 4000             # batches per loader instance before refresh

# ---------------------------------------------------------------------------
# Fixed harness parameters — hardcoded here; the agent CANNOT change these
# ---------------------------------------------------------------------------
_N_MAX: int = 200    # maximum N seen during training
_VAL_D_MIN: int = 3  # evaluation D lower bound
_VAL_N_MIN: int = 20 # evaluation N lower bound
_VAL_SIZE: int = 256 # number of evaluation codes
_VAL_SEED: int = 42  # deterministic seed for evaluation set construction
_VAL_BATCH: int = 4  # batch size during evaluation

SEP = "=" * 78
SEP2 = "-" * 78


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _build_run_dir(cfg: Config) -> Path:
    run_dir = Path(cfg.log_dir) / f"{cfg.run_name}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _infinite_batches(cfg: Config):
    """Yield raw (x, mask, Ds, Ns) batches from the data pipeline indefinitely."""
    chunk = cfg.batch_size * LOADER_CHUNK
    while True:
        yield from make_loader(chunk, cfg, seed=None)


# ---------------------------------------------------------------------------
# Scorecard printing
# ---------------------------------------------------------------------------

def _is_finite_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _summarize_training_metrics(window: list[dict]) -> dict:
    if not window:
        return {}

    summary: dict[str, float] = {}
    keys = {
        key
        for record in window
        for key, value in record.items()
        if _is_finite_scalar(value)
    }
    for key in sorted(keys):
        values = [
            float(record[key])
            for record in window
            if _is_finite_scalar(record.get(key))
        ]
        if not values:
            continue
        summary[key] = sum(values) / len(values)
    return summary


def _annotate_training_progress(
    summary: dict,
    *,
    elapsed: float,
    window_elapsed: float,
    window_steps: int,
    batch_size: int,
) -> dict:
    annotated = dict(summary)
    annotated["window_steps"] = float(window_steps)
    annotated["elapsed_seconds"] = elapsed
    annotated["progress_pct"] = 100.0 * min(max(elapsed / TRAIN_BUDGET_SECONDS, 0.0), 1.0)
    annotated["seconds_remaining"] = max(TRAIN_BUDGET_SECONDS - elapsed, 0.0)
    if window_steps > 0 and window_elapsed > 0.0:
        steps_per_sec = window_steps / window_elapsed
        annotated["steps_per_sec"] = steps_per_sec
        annotated["samples_per_sec"] = steps_per_sec * batch_size
    return annotated


def _print_training_metrics(train: dict, step: int, elapsed: float) -> None:
    window_steps = int(train.get("window_steps", 0))
    suffix = f", last {window_steps} steps" if window_steps > 0 else ""
    _header(f"TRAINING SNAPSHOT  (step {step}, {elapsed:.1f}s of training{suffix})")
    progress_items = [
        ("elapsed_seconds", "elapsed seconds"),
        ("progress_pct", "budget used (%)"),
        ("seconds_remaining", "seconds remaining"),
        ("steps_per_sec", "steps per second"),
        ("samples_per_sec", "samples per second"),
    ]
    print("\n  [Progress]")
    for key, label in progress_items:
        value = train.get(key)
        if value is None:
            continue
        print(f"  {label:<38}  {float(value):.6g}")

    diagnostic_keys = sorted(
        key
        for key, value in train.items()
        if _is_finite_scalar(value)
        and key
        not in {
            "window_steps",
            "elapsed_seconds",
            "progress_pct",
            "seconds_remaining",
            "steps_per_sec",
            "samples_per_sec",
            "step",
            "train_elapsed",
            "final",
        }
    )

    print("\n  [Averaged Diagnostics From solution.py]")
    if not diagnostic_keys:
        print("  (no finite scalar diagnostics returned)")
        return

    for key in diagnostic_keys:
        print(f"  {key:<38}  {float(train[key]):.6g}")


def _print_eval_metrics(val: dict, step: int, elapsed: float) -> None:
    _header(f"FINAL EVALUATION  (step {step}, {elapsed:.1f}s of training)")
    sections = [
        (
            "Reconstruction",
            [
                ("val_egw", "EGW reconstruction"),
                ("val_loss_std", "loss std"),
                ("val_loss_p95", "loss p95"),
                ("val_disc_ratio_min", "discrimination ratio (min)"),
                ("val_disc_ratio_mean", "discrimination ratio (mean)"),
                ("val_disc_pass", "discrimination pass"),
            ],
        ),
        (
            "Latent Geometry",
            [
                ("val_latnorm", "latent norm"),
                ("val_latstd", "latent std (per-dim mean)"),
                ("val_latstd_min", "latent std min"),
                ("val_latstd_max", "latent std max"),
                ("val_latpdist", "latent pairwise dist mean"),
                ("val_latdead", "latent dead dims (frac)"),
            ],
        ),
        (
            "PCA",
            [
                ("val_rank", "PCA numeric rank"),
                ("val_pca_rank_95", "PCA rank @95% var"),
                ("val_pca_rank_99", "PCA rank @99% var"),
                ("val_entropy_rank", "entropy rank"),
                ("val_pca_participation", "participation rank"),
            ],
        ),
        (
            "Invariance",
            [
                ("inv_perm", "permutation invariance (max |Δz|)"),
                ("inv_rot", "rotation invariance (max |Δz|)"),
                ("lip_pert", "local Lipschitz estimate"),
            ],
        ),
        (
            "Evaluation Gradients",
            [
                ("val_grad_total", "total grad norm"),
            ],
        ),
    ]

    for title, cols in sections:
        print(f"\n  [{title}]")
        for key, label in cols:
            v = val.get(key)
            if v is None:
                continue
            if isinstance(v, bool):
                print(f"  {label:<38}  {v}")
            elif isinstance(v, float):
                print(f"  {label:<38}  {v:.6g}")
            else:
                print(f"  {label:<38}  {v}")

    grad_items = sorted(
        (
            (key, value)
            for key, value in val.items()
            if key.startswith("val_grad_")
            and key != "val_grad_total"
        ),
        key=lambda kv: abs(float(kv[1])),
        reverse=True,
    )
    if grad_items:
        print("\n  [Evaluation Gradients by Component]")
        for key, value in grad_items:
            label = key.replace("val_grad_", "").replace("_", " ")
            print(f"  {label:<38}  {float(value):.6g}")

    spectrum = val.get("val_pca_spectrum") or []
    if spectrum:
        spec = "  ".join(f"PC{i + 1}={float(v):.3f}" for i, v in enumerate(spectrum))
        print("\n  [PCA Spectrum]")
        print(f"  {spec}")

    for key, title in (("val_loss_by_D", "Worst D Regimes"), ("val_loss_by_N", "Worst N Regimes")):
        rows = val.get(key) or []
        if not rows:
            continue
        print(f"\n  [{title}]")
        for row in rows[:5]:
            axis = "D" if "D" in row else "N"
            value = row[axis]
            print(
                f"  {axis}={value:<4d}  mean={row['mean']:.4f}  std={row['std']:.4f}  "
                f"min={row['min']:.4f}  max={row['max']:.4f}  n={row['count']}"
            )

    rows = val.get("val_worst_cases") or []
    if rows:
        print("\n  [Worst Evaluation Cases]")
        for row in rows:
            print(f"  loss={row['loss']:.4f}  N={row['N']:4d}  D={row['D']:3d}")


def _print_probe_table(
    names: list[str],
    r2_lin: np.ndarray,
    r2_mlp: np.ndarray,
    r2_nd: np.ndarray,
) -> None:
    _header("PROBE RESULTS  (test R²  —  sorted by best probe score)")
    best = np.maximum(r2_lin, r2_mlp)
    order = np.argsort(best)[::-1]
    print(f"  {'target':<22}  {'best probe':>10}  {'N,D base':>10}  {'Δ best-base':>12}")
    print(f"  {SEP2}")
    for i in order:
        d = float(best[i]) - float(r2_nd[i])
        print(
            f"  {names[i]:<22}  {best[i]:10.3f}  {r2_nd[i]:10.3f}  {d:+12.3f}"
        )


def _compute_final_score(
    names: list[str],
    r2_lin: np.ndarray,
    r2_mlp: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Average of clip(max(r2_lin, r2_mlp), 0, 1) across all probe targets."""
    per_target: dict[str, float] = {}
    total = 0.0
    for i, name in enumerate(names):
        best = float(max(r2_lin[i], r2_mlp[i]))
        clipped = float(min(max(best, 0.0), 1.0))
        per_target[name] = clipped
        total += clipped
    final_score = total / max(len(names), 1)
    return final_score, per_target


def _print_scorecard(
    run_dir: Path,
    step: int,
    final_train: dict,
    final_val: dict,
    probe: dict,
    final_score: float,
    per_target: dict[str, float],
    train_seconds: float,
) -> None:
    names = probe["target_names"]
    r2_lin = np.array(probe["r2_linear"])
    r2_mlp = np.array(probe["r2_mlp"])
    r2_nd = np.array(probe["r2_nd_baseline"])
    categories = probe.get("categories_idx", {})

    _print_training_metrics(final_train, step, train_seconds)
    _print_eval_metrics(final_val, step, train_seconds)
    _print_probe_table(names, r2_lin, r2_mlp, r2_nd)

    if categories:
        _header("CATEGORY LEADERS  (best probe R²)")
        for category, idxs in categories.items():
            if not idxs:
                continue
            cat_order = sorted(idxs, key=lambda i: float(max(r2_lin[i], r2_mlp[i])), reverse=True)
            print(f"  [{category}]")
            for i in cat_order[:5]:
                best_v = float(max(r2_lin[i], r2_mlp[i]))
                d = best_v - float(r2_nd[i])
                print(f"    {names[i]:<22}  best R²={best_v:.3f}  base={r2_nd[i]:.3f}  Δ={d:+.3f}")

    trajectories = probe.get("trajectories", [])
    if trajectories:
        _header("TRAJECTORY SUMMARY")
        for traj in trajectories:
            energy = traj["energies"]
            ls = np.array(traj["latent_steps"])
            lats = np.array(traj["latents"])
            pc1 = lats[:, 0] if lats.shape[1] > 0 else np.zeros(len(lats))
            r_corr = (
                float(np.corrcoef(pc1, energy)[0, 1])
                if pc1.std() > 1e-8 and np.array(energy).std() > 1e-8
                else float("nan")
            )
            print(
                f"  N={traj['N']:4d}  D={traj['D']:3d}  "
                f"path_length={float(ls.sum()):.3f}  "
                f"energy_drop={float(energy[0] - energy[-1]):.3f}  "
                f"PC1_r_energy={r_corr:.3f}"
            )

    _header("FINAL SCORE BREAKDOWN")
    order = sorted(per_target.items(), key=lambda kv: kv[1], reverse=True)
    for name, score in order:
        idx = names.index(name)
        best_v = float(max(r2_lin[idx], r2_mlp[idx]))
        base_v = float(r2_nd[idx])
        print(f"  {name:<22}  best={score:.4f}  (probe={best_v:.3f}  base={base_v:.3f})")
    print(f"\n  final_score = {final_score:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the spherical-code autoencoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="compute device (default: auto)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override random seed (default: use Config default)")
    parser.add_argument("--ckpt", type=str, default="",
                        help="checkpoint path for warm-start / fine-tuning")
    args = parser.parse_args()

    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.ckpt:
        cfg.ckpt_path = args.ckpt
    # Enforce fixed constraints — agent cannot override these
    cfg.N_max = _N_MAX
    device = _resolve_device(args.device if args.device != "auto" else cfg.device)

    run_dir = _build_run_dir(cfg)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    # Fixed evaluation config — decoupled from agent's Config so it cannot be gamed
    _val_cfg = types.SimpleNamespace(
        D_min=_VAL_D_MIN, D_max=cfg.D_max,
        N_min=_VAL_N_MIN, N_max=_N_MAX,
        val_size=_VAL_SIZE, seed=_VAL_SEED,
        batch_size=_VAL_BATCH,
        archive_dirs=cfg.archive_dirs,
    )

    print(f"device     : {device}")
    print(f"run_dir    : {run_dir}")
    print(f"budget     : {TRAIN_BUDGET_SECONDS:.0f}s training + {PROBE_BUDGET_SECONDS:.0f}s probe")
    print(f"report_every: {REPORT_EVERY_STEPS} steps (averaged diagnostics from solution.py)")
    print(f"N_max      : {_N_MAX} (fixed)")
    print(f"eval_size  : {_VAL_SIZE} (final pass only)")

    # Training state
    state = build_training_state(cfg, device, ckpt_path=cfg.ckpt_path)
    n_enc = sum(p.numel() for p in state.enc.parameters())
    n_dec = sum(p.numel() for p in state.dec.parameters())
    print(f"encoder params: {n_enc/1e6:.2f}M  decoder params: {n_dec/1e6:.2f}M")

    # Training loop
    _header("TRAINING")
    metrics_log: list[dict] = []
    report_buffer: list[dict] = []
    batch_it = _infinite_batches(cfg)
    train_start = time.perf_counter()
    last_report_elapsed = 0.0
    step = 0
    final_train: dict = {}
    final_val: dict = {}
    last_step_m: dict = {}

    while True:
        elapsed = time.perf_counter() - train_start
        if elapsed >= TRAIN_BUDGET_SECONDS:
            break

        raw_batch = next(batch_it)
        batch = tuple(t.to(device, non_blocking=True) for t in raw_batch)

        state.enc.train()
        state.dec.train()
        step_m = train_one_step(state, batch, step)
        step += 1
        last_step_m = step_m
        report_buffer.append(step_m)

        if step % REPORT_EVERY_STEPS == 0:
            elapsed_now = time.perf_counter() - train_start
            window_steps = len(report_buffer)
            train_summary = _annotate_training_progress(
                _summarize_training_metrics(report_buffer),
                elapsed=elapsed_now,
                window_elapsed=elapsed_now - last_report_elapsed,
                window_steps=window_steps,
                batch_size=cfg.batch_size,
            )
            rec = {
                "step": step,
                "train_elapsed": elapsed_now,
                **train_summary,
            }
            metrics_log.append(rec)
            final_train = rec
            _print_training_metrics(rec, step, elapsed_now)
            report_buffer.clear()
            last_report_elapsed = elapsed_now

    train_seconds = time.perf_counter() - train_start

    print(f"\nTraining finished: {step} steps in {train_seconds:.1f}s")
    if report_buffer:
        window_steps = len(report_buffer)
        final_train = _annotate_training_progress(
            _summarize_training_metrics(report_buffer),
            elapsed=train_seconds,
            window_elapsed=train_seconds - last_report_elapsed,
            window_steps=window_steps,
            batch_size=cfg.batch_size,
        )
    elif final_train:
        final_train = dict(final_train)
        final_train["elapsed_seconds"] = train_seconds
    else:
        final_train = _annotate_training_progress(
            _summarize_training_metrics([last_step_m] if last_step_m else []),
            elapsed=train_seconds,
            window_elapsed=train_seconds,
            window_steps=1 if last_step_m else 0,
            batch_size=cfg.batch_size,
        )
    final_train.update({
        "step": step,
        "train_elapsed": train_seconds,
        "progress_pct": 100.0,
        "seconds_remaining": 0.0,
        "final": True,
    })
    print("Building evaluation set ...")
    val_codes = build_val_codes(_val_cfg)
    loss_fn = build_val_loss_fn(cfg)
    _header("FINAL EVALUATION")
    final_val = evaluate(state.enc, state.dec, val_codes, _val_cfg, device, loss_fn=loss_fn)
    metrics_log.append({
        **final_train,
        **final_val,
    })

    # Save artifacts
    ckpt_path = run_dir / "ckpt.pt"
    save_checkpoint(state, ckpt_path, step)
    (run_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
    print(f"Checkpoint saved: {ckpt_path}")

    # Run latent probe suite (with timeout)
    _header("LATENT PROBE SUITE")
    probe_dir = run_dir / "probe"
    probe_dir.mkdir(exist_ok=True)
    probe: dict = {}
    probe_ok = True
    print(f"Probe budget: {PROBE_BUDGET_SECONDS:.0f}s")
    try:
        with ThreadPoolExecutor(max_workers=1) as _pool:
            _future = _pool.submit(run_probe_suite, ckpt_path, device,
                                   output_dir=probe_dir)
            try:
                probe = _future.result(timeout=PROBE_BUDGET_SECONDS)
                print(f"Probe complete. Output in {probe_dir}")
            except _FuturesTimeout:
                print(
                    f"[WARNING] Probe suite exceeded {PROBE_BUDGET_SECONDS:.0f}s "
                    "budget and was abandoned."
                )
                probe_ok = False
    except Exception as exc:
        print(f"[WARNING] Probe suite failed: {exc}")
        probe_ok = False

    total_seconds = time.perf_counter() - train_start

    # Compute final_score
    final_score = 0.0
    per_target: dict[str, float] = {}
    if probe_ok and probe:
        try:
            names = probe["target_names"]
            r2_lin = np.array(probe["r2_linear"])
            r2_mlp = np.array(probe["r2_mlp"])
            final_score, per_target = _compute_final_score(names, r2_lin, r2_mlp)
        except Exception as exc:
            print(f"[WARNING] final_score computation failed: {exc}")
            probe_ok = False

    # Print scorecard
    if probe_ok and probe:
        _print_scorecard(
            run_dir=run_dir,
            step=step,
            final_train=final_train,
            final_val=final_val,
            probe=probe,
            final_score=final_score,
            per_target=per_target,
            train_seconds=train_seconds,
        )
    else:
        _print_training_metrics(final_train, step, train_seconds)
        _print_eval_metrics(final_val, step, train_seconds)

    # Save scorecard JSON for programmatic use
    scorecard = {
        "final_score": final_score,
        "train_seconds": train_seconds,
        "total_seconds": total_seconds,
        "steps_run": step,
        "status": "ok" if probe_ok else "probe_failed",
        "ckpt": str(ckpt_path),
        "final_train": final_train,
        "final_val": final_val,
        "probe_r2_linear": probe.get("r2_linear", []),
        "probe_r2_mlp": probe.get("r2_mlp", []),
        "probe_target_names": probe.get("target_names", []),
        "per_target_score": per_target,
    }
    (run_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))

    # Machine-readable footer (grep-extractable)
    _header("FINAL SUMMARY")
    print("---")
    print(f"final_score:    {final_score:.6f}")
    print(f"train_seconds:  {train_seconds:.1f}")
    print(f"total_seconds:  {total_seconds:.1f}")
    print(f"status:         {'ok' if probe_ok else 'probe_failed'}")
    print(f"steps_run:      {step}")
    print(f"ckpt:           {ckpt_path}")


if __name__ == "__main__":
    main()
