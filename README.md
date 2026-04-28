# autoresearch

The setup here is **autoencoder architecture research** for spherical codes.
An agent edits [`solution.py`](solution.py) (and optionally [`loss.py`](loss.py)),
runs [`test_solution.py`](test_solution.py), checks whether `final_score` improved,
and keeps or discards the change. Training runs for exactly 10 minutes; the harness
averages and prints the finite scalar diagnostics returned by `train_one_step(...)`
during the run, then runs one fixed evaluation pass and the latent probe suite at the
end before printing a scorecard. The probe has a 5-minute cap.

## How it works

The repo has a small set of files:

- **`solution.py`** — the primary file the agent edits. Must export the public API:
  `Config`, `SphereCodeEncoder`, `SphereCodeDecoder`, `build_training_state`,
  `train_one_step`, `build_val_loss_fn`, `save_checkpoint`, `load_checkpoint`.
  Everything else — architecture, loss functions, training strategy — is up to the
  agent. For each trial, the agent should also set `Config.run_name` to
  `exp<n>_<name>` with the trial number and a short label. **This file is edited by the agent**.
- **`loss.py`** — loss implementation. The agent may freely edit or replace this.
  **The agent may edit this file**.
- **`test_solution.py`** — the fixed evaluation harness. Trains for 10 minutes, logs
  averaged scalar diagnostics returned by `solution.py` during the run, runs one fixed
  evaluation pass and the latent probe suite at the end, prints a scorecard, and emits
  the machine-readable footer.
  **Do not modify during the experiment loop**.
- **`data.py`** — data pipeline: archive loading, perturbation, sampling. **Read-only**.
- **`evaluation.py`** — fixed end-of-run evaluation metrics. **Read-only**.
- **`latent_test_bed.py`** — linear and MLP probe suite for latent quality assessment.
  **Read-only**.
- **`diagnostics.py`** — logging and diagnostic computations. **Read-only**.
- **`program.md`** — instructions for the autonomous agent.

The metric is **`final_score`** — higher is better (range 0–1).  It is the mean of
`clip(max(R²_linear, R²_mlp), 0, 1)` over all probe targets.  A score of 1.0 means the
latent predicts all targets perfectly.

## Quick start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), NVIDIA GPU with CUDA
12.8+ driver.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the evaluation harness (10-minute training + final evaluation + probe suite)
uv run test_solution.py
```

Extract the key metrics from a log file:

```bash
grep "^final_score:\|^train_seconds:\|^total_seconds:\|^status:\|^steps_run:\|^ckpt:" run.log
```

## Running the agent

Point an agent at `program.md`:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

## Project structure

```
solution.py          — solver module: Config, encoder, decoder, training step (agent edits)
loss.py              — Entropic Gromov-Wasserstein OT loss (agent may edit)
test_solution.py     — fixed evaluator: 10-min training + final evaluation + probe scorecard
data.py              — data pipeline (read-only)
evaluation.py        — end-of-run evaluation metrics (read-only)
latent_test_bed.py   — latent probe suite with run_probe_suite() API (read-only)
diagnostics.py       — logging utilities (read-only)
program.md           — agent instructions
pyproject.toml       — dependencies
autoresearch-runs/   — local per-trial diagnostics for later analysis (keep untracked)
```
- **Focused editable surface.** The agent edits `solution.py` and optionally `loss.py`; the harness and support code stay fixed during the experiment loop.
- **Fixed run profile.** `test_solution.py` enforces 10 minutes of training, one fixed evaluation pass at the end, and a 5-minute cap for the final probe.
- **Per-trial run naming.** The agent should update `Config.run_name` in `solution.py` to `exp<n>_<name>` on each trial so `autoresearch-runs/` stays organized.
- **Solution-defined training diagnostics.** During the run, the harness averages and prints whatever finite scalar diagnostics `train_one_step(...)` returns, so the solver can decide what signals it needs.
- **Probe-based score.** `final_score` is the mean of `clip(max(R²_linear, R²_mlp), 0, 1)` across probe targets, with the full scorecard printed at the end.

## Platform support

The default harness in [test_solution.py](test_solution.py) is meant to complete with a
10-minute training budget, one final evaluation pass, and up to 5 minutes of probing on
a reasonable GPU.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
