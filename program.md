# autoresearch

This is an experiment to have the LLM do its own research on spherical-code
autoencoders.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr28`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Verify git preconditions**: Before starting, make sure `solution.py`, `loss.py`, `test_solution.py`, `data.py`, `evaluation.py`, `latent_test_bed.py`, `diagnostics.py`, `README.md`, `program.md`, and `pyproject.toml` are tracked and the working tree is clean.
4. **Read the in-scope files**: Read these for full context:
   - `README.md` — repository overview.
  - `solution.py` — the primary file you edit. Must export the required public API (see README). Architecture, loss, and training strategy are yours to define. For each trial, also update `Config.run_name` to `exp<n>_<name>`, where `n` is the trial number and `name` is a short experiment label.
   - `loss.py` — the secondary file you may edit. Contains a loss implementation you can freely modify or replace.
   - `test_solution.py` — the fixed evaluation harness. Do not modify it.
   - `data.py`, `evaluation.py`, `latent_test_bed.py`, `diagnostics.py` — read-only support code. Do not modify them.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Keep diagnostics untracked**: `autoresearch-runs/` stores per-trial diagnostics for later reference and analysis. Keep it untracked and never commit it.
7. **Confirm and go**: Confirm the setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains the autoencoder for **10 minutes (600 seconds)** using the
solver and configs defined in `solution.py`. The harness averages and prints whatever finite
scalar diagnostics `train_one_step(...)` returns every reporting interval.
After training finishes, the harness runs one fixed evaluation pass and then the probe
suite from `latent_test_bed.py`. The final probe pass has its own **5-minute cap**.

Launch a run simply as:

```bash
uv run test_solution.py > run.log 2>&1
```

**What you CAN do:**
- Modify `solution.py` — anything: Config, model architecture, loss functions, training
  loop, optimiser, scheduler, regularisation.  You are free to completely redesign the
  encoder, decoder, or training strategy.  The only constraint is that the public API
  (`Config`, `SphereCodeEncoder`, `SphereCodeDecoder`, `build_training_state`,
  `train_one_step`, `build_val_loss_fn`, `save_checkpoint`, `load_checkpoint`) must
  remain importable and work correctly. Each trial should also set `Config.run_name`
  to `exp<n>_<name>`.
- Modify `loss.py` — the loss implementation.  You may tune, extend, or completely
  replace it with any formulation that helps `final_score`.

**What you CANNOT do:**
- Modify `data.py`, `evaluation.py`, `latent_test_bed.py`, `diagnostics.py` — read-only.
- Modify `test_solution.py` — the fixed evaluation harness.
- Modify `program.md`, `README.md`, or `pyproject.toml`.
- Install new packages or add dependencies not already in `pyproject.toml`.
- Hack the metrics, score extraction, probe targets, or any evaluation bookkeeping.

**The goal: maximize `final_score`.** Higher is better. The score is the mean of
`clip(max(R²_linear, R²_mlp), 0, 1)` over all probe targets, where R² measures how
well the latent representation recovers analytic properties of the spherical code
(geometry, energies, spectral features, etc.).  A score of 1.0 means the latent
predicts all targets perfectly; 0.0 means it carries no information above chance. A baseline result is around 0.9.

The final scorecard printed above the footer shows per-target breakdown, category
leaders, and trajectory diagnostics — use these to guide the next edit.
The training-time diagnostics are also under your control via the dict returned by
`train_one_step(...)` in `solution.py`.

**First run**: Always run the evaluation as-is first to establish the baseline.

## Output format

When the script finishes it prints a machine-readable footer:

```
---
final_score:    0.621482
train_seconds:  600.0
total_seconds:  842.7
status:         ok
steps_run:      3000
ckpt:           ./runs/fsw_vn_sphere_1714329600/ckpt.pt
```

Extract it from the log with:

```bash
grep "^final_score:\|^train_seconds:\|^total_seconds:\|^status:\|^steps_run:\|^ckpt:" run.log
```

If the grep output is empty the run crashed before the summary.  Run
`tail -n 50 run.log` to read the Python stack trace and fix it.

## Logging results

Log every finished run to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	final_score	status	description
```

1. git commit hash (short, 7 chars)
2. `final_score` achieved — use 0.000000 for crashes/timeouts
3. status: `keep`, `discard`, or `crash`
4. short description of the experiment

Example:

```
commit	final_score	status	description
a1b2c3d	0.92	keep	baseline
c3d4e5f	0.93	keep	increased decoder size
d4e5f6g	0.412100	discard	added sorted Gram matrix auxiliary loss
e5f6g7h	0.000000	crash	removed pair mask guard
```

Do not commit `results.tsv` or anything under `autoresearch-runs/` — leave them untracked by git.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr28`).

LOOP FOREVER:

1. Check the current git state (branch and commit).
2. Edit `solution.py` and/or `loss.py` with your experimental idea. Update `Config.run_name` in `solution.py` to `exp<n>_<name>` for the current trial number and a short trial label.
3. `git commit -am "experiment: <brief description>"`
4. Run: `uv run test_solution.py > run.log 2>&1`
5. Extract data, for example: `grep "^final_score:\|^train_seconds:\|^total_seconds:\|^status:\|^steps_run:\|^ckpt:" run.log`
6. If empty: the run crashed — run `tail -n 50 run.log` and fix.
7. Log results in `results.tsv`.
8. If `final_score` improved (higher), **advance** — keep the commit.
9. If equal or worse, `git reset --hard HEAD~1` to discard the change.


**Crashes**: If a run crashes due to a trivial bug, fix and re-run.  If the idea is
fundamentally broken, log `crash` in the TSV and move on.

**Timeouts**: `train_seconds` will be ≤ 600.0 (10 min). `total_seconds` also includes
the single end-of-run evaluation pass and up to 300.0 seconds of probing, so it will
run longer than the raw training budget. A `status: probe_failed` means training and the
final evaluation finished but the probe errored or timed out — treat the run as a
partial result; inspect the logs.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if
you should continue.  Do NOT ask "should I keep going?" or "is this a good stopping
point?".  The loop runs until the human manually interrupts you, period.  If you run out
of ideas, re-read the scorecard diagnostics (per-target R², category breakdowns,
trajectory PC1–energy correlation) and let the weakest probe targets guide the next
architectural change.


